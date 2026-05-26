#!/usr/bin/env python3
"""
Gradient Signal-to-Noise Ratio (GSNR) Evaluation Script for GRPO Experiments

This script computes the GSNR metric as described in the paper to measure the quality
of the learning signal produced by the GRPO objective. GSNR is computed with respect
to the model's final loop latent-thought logits.

Algorithm:
1. For a given prompt p, sample R=8 independent responses using the current policy
2. Each response is graded to compute advantages (binary reward: correct/incorrect)
3. For each rollout, compute gradient of loss w.r.t. latent-thought logits
4. Compute mean gradient μ_p and noise variance across rollouts
5. GSNR_p = ||μ_p||²₂ / (noise_p + ε)
6. Overall GSNR = mean of log-transformed prompt-level scores

This version uses vLLM for fast rollout generation and HuggingFace for gradient computation.

Multi-Loop GSNR Mode:
When --include_intermediate_loops is enabled, the GSNR is computed with respect to
the gradients of all loop logits (both final and intermediate loops), not just the
final loop. This allows measuring gradient signal quality across the entire
Universal Transformer computation.

Usage:
    python gsnr_eval.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --benchmark math500 \
        --num_rollouts 8 \
        --output_dir /path/to/gsnr_results

    # With intermediate loop gradients:
    python gsnr_eval.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --benchmark math500 \
        --include_intermediate_loops \
        --output_dir /path/to/gsnr_results
"""
import os
import sys
import json
import argparse
import logging
import shutil
import subprocess
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_utils import (
    rl_extract_boxed_answer as extract_boxed_answer,
    rl_check_math_answer as check_math_answer,
    rl_get_gold_answer as get_gold_answer,
    format_math_prompt,
)

# Benchmark configurations
BENCHMARK_CONFIGS = {
    "math500": {
        "test_file": "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
        "max_new_tokens": 2048,
        "max_prompt_length": 1024,
    },
    "gsm8k": {
        "test_file": "/scratch/gpfs/OLGARUS/jw4199/datasets/gsm8k/gsm8k.test.jsonl",
        "max_new_tokens": 512,
        "max_prompt_length": 512,
    },
    "aime24": {
        "test_file": "/scratch/gpfs/OLGARUS/jw4199/datasets/aime24/aime24.test.jsonl",
        "max_new_tokens": 3072,
        "max_prompt_length": 1024,
    },
    "beyondaime": {
        "test_file": "/scratch/gpfs/OLGARUS/jw4199/datasets/BeyondAIME/beyondaime.test.jsonl",
        "max_new_tokens": 3072,
        "max_prompt_length": 1024,
    },
}


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load test set from JSONL."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} problems from {filepath}")
    return data


def find_merged_model(experiment_dir: str, checkpoint: str) -> Optional[str]:
    """Find existing merged model path for a given checkpoint."""
    for dirname in os.listdir(experiment_dir):
        if dirname.startswith("eval_"):
            eval_dir = os.path.join(experiment_dir, dirname)
            merged_path = os.path.join(eval_dir, checkpoint, "merged_model")
            if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
                logger.info(f"Found existing merged model at: {merged_path}")
                return merged_path
    return None


def prepare_model(
    experiment_dir: str,
    checkpoint: str,
    output_dir: str
) -> str:
    """Prepare model checkpoint for evaluation.

    This function:
    1. Finds or creates the merged HuggingFace model from FSDP checkpoint
    2. Returns the path to the prepared model
    """
    step_num = checkpoint.replace("step_", "")
    fsdp_checkpoint_dir = os.path.join(experiment_dir, f"global_step_{step_num}", "actor")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    model_dir = os.path.join(output_dir, "merged_model")

    # Check if already prepared
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info(f"Using existing merged model at: {model_dir}")
        return model_dir

    # First check if there's an existing merged model we can copy from
    existing_merged = find_merged_model(experiment_dir, checkpoint)

    if existing_merged:
        logger.info(f"Copying merged model from: {existing_merged}")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        shutil.copytree(existing_merged, model_dir)
    else:
        # Need to merge the checkpoint
        logger.info(f"Merging FSDP checkpoint from: {fsdp_checkpoint_dir}")

        # Fix checkpoint files first
        ckpt_hf_dir = os.path.join(fsdp_checkpoint_dir, "huggingface")
        base_model_dir = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"

        # Fix config.json
        config_path = os.path.join(ckpt_hf_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_content = f.read()
            config_content = config_content.replace(
                '"AutoModelForCausalLM": "peft_model.PeftModelForCausalLM"',
                '"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"'
            )
            with open(config_path, "w") as f:
                f.write(config_content)

        # Copy modeling_ouro.py if missing
        modeling_path = os.path.join(ckpt_hf_dir, "modeling_ouro.py")
        if not os.path.exists(modeling_path):
            shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

        # Run merger
        os.makedirs(model_dir, exist_ok=True)
        result = subprocess.run([
            "python", "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--trust-remote-code",
            "--local_dir", fsdp_checkpoint_dir,
            "--target_dir", model_dir,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Merge failed: {result.stderr}")
            raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")

    logger.info(f"Model prepared at: {model_dir}")
    return model_dir


def compute_grpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute GRPO policy gradient loss.

    Loss = -advantage * log P(token | context)

    Args:
        logits: Model logits [batch, seq, vocab]
        labels: Token labels [batch, seq]
        advantages: Per-sample advantages [batch]
        attention_mask: Attention mask [batch, seq]

    Returns:
        Per-sample loss tensor [batch]
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Replace -100 (ignore index) with 0 for gather operation
    # The mask will zero out these positions anyway
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels < 0] = 0

    # Gather log probs for actual tokens
    # Shape: [batch, seq-1]
    token_log_probs = log_probs.gather(
        dim=-1,
        index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out padding/prompt tokens and sum
    masked_log_probs = token_log_probs * shift_mask
    sequence_log_probs = masked_log_probs.sum(dim=-1)  # [batch]

    # GRPO loss: -advantage * log P(response)
    loss = -advantages * sequence_log_probs

    return loss


def compute_gsnr_for_prompt(
    model,
    tokenizer,
    prompt_text: str,
    responses: List[str],
    rewards: List[float],
    device: torch.device,
    max_length: int = 4096,
    include_intermediate_loops: bool = False,
) -> Dict[str, Any]:
    """
    Compute GSNR for a single prompt with R rollouts.

    The GSNR is computed with respect to the latent-thought logits (activations),
    which are the final loop's hidden states projected to vocabulary space.

    Per the GSNR paper, we compute:
        g_{p,r} = ∇_{z_{p,r}} L_{p,r}
    where z_{p,r} is the latent-thought logit (the pre-softmax activation vector).

    We use retain_grad() on the logits tensor to capture ∂L/∂logits (the gradient
    w.r.t. the activation), NOT ∂L/∂W (the gradient w.r.t. the weights).

    Multi-Loop Mode:
    When include_intermediate_loops=True, we compute gradients w.r.t. all loop
    logits (not just the final loop), and concatenate them to form a single
    gradient vector. This measures the gradient signal quality across the
    entire Universal Transformer computation.

    Args:
        model: The model (OuroForCausalLM)
        tokenizer: The tokenizer
        prompt_text: The formatted prompt
        responses: List of R response strings
        rewards: List of R reward values (0 or 1)
        device: CUDA device
        max_length: Maximum sequence length
        include_intermediate_loops: If True, compute gradients w.r.t. all loop
            logits (intermediate + final), not just the final loop logits.

    Returns:
        Dict with GSNR value and debug info
    """
    R = len(responses)
    if R == 0:
        return {"gsnr": 0.0, "valid": False, "reason": "no_responses"}

    # Compute advantages (normalized rewards within group)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std()
    if std_reward > 0:
        advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
    else:
        # All same reward - no learning signal
        advantages = torch.zeros_like(rewards_tensor)

    # Collect gradients for each rollout
    gradients = []

    for r_idx, (response, advantage) in enumerate(zip(responses, advantages)):
        # Tokenize prompt + response
        full_text = prompt_text + response
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get prompt length to mask loss
        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Create labels (mask out prompt tokens with -100)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # Create response mask
        response_mask = attention_mask.clone()
        response_mask[:, :prompt_len] = 0

        # Forward pass with gradient computation
        model.zero_grad()

        if include_intermediate_loops:
            # Multi-loop mode: Get hidden states from all UT loops and compute logits
            # The OuroModel returns (outputs, hidden_states_list, gate_list) where
            # hidden_states_list contains the hidden states after each UT loop

            # Access the base model to get hidden_states_list
            base_model = model.model  # OuroModel
            base_outputs, hidden_states_list, gate_list = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            # Compute logits for each loop's hidden states using the lm_head
            per_loop_logits = []
            for hidden_states in hidden_states_list:
                loop_logits = model.lm_head(hidden_states)
                loop_logits.retain_grad()  # Enable gradient retention
                per_loop_logits.append(loop_logits)

            if len(per_loop_logits) == 0:
                # Fallback: something went wrong
                logger.warning("No hidden states returned, falling back to standard forward")
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                per_loop_logits = [outputs.logits]
                per_loop_logits[0].retain_grad()

            # Compute loss using the final loop logits (this is what GRPO trains on)
            loss = compute_grpo_loss(
                logits=per_loop_logits[-1],  # Use final loop for loss computation
                labels=labels,
                advantages=advantage.unsqueeze(0).to(device),
                attention_mask=response_mask,
            )

            # Backpropagate to get gradients w.r.t. all loop logits
            loss.backward()

            # Collect gradients from all loops and concatenate
            all_loop_grads = []
            for loop_idx, loop_logits in enumerate(per_loop_logits):
                logit_grad = loop_logits.grad
                if logit_grad is not None:
                    # Extract response token gradients only
                    response_logit_grad = logit_grad[:, prompt_len:, :]
                    grad_flat = response_logit_grad.detach().clone().flatten()
                    all_loop_grads.append(grad_flat)

                # Clear retained grad
                if loop_logits.grad is not None:
                    loop_logits.grad = None

            if all_loop_grads:
                # Concatenate gradients from all loops into one vector
                combined_grad = torch.cat(all_loop_grads, dim=0)
                gradients.append(combined_grad)

        else:
            # Original mode: Only final loop logits
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logits = outputs.logits  # [1, seq, vocab]

            # IMPORTANT: Enable gradient retention on the logits tensor (an intermediate activation)
            # This allows us to compute ∂L/∂logits (gradient w.r.t. the activation itself)
            # rather than ∂L/∂W (gradient w.r.t. the weights)
            logits.retain_grad()

            # Compute loss for this rollout
            loss = compute_grpo_loss(
                logits=logits,
                labels=labels,
                advantages=advantage.unsqueeze(0).to(device),
                attention_mask=response_mask,
            )

            # Backpropagate to get gradients
            loss.backward()

            # Extract gradient with respect to the logits (the latent-thought activation)
            # This is ∂L/∂z where z is the pre-softmax logit vector
            # Shape: [1, seq, vocab]
            logit_grad = logits.grad

            if logit_grad is not None:
                # For GSNR, we focus on the response tokens only (not prompt)
                # Extract gradients for response positions and flatten
                # Shape: [1, seq, vocab] -> [response_len, vocab] -> [response_len * vocab]
                response_logit_grad = logit_grad[:, prompt_len:, :]  # [1, response_len, vocab]

                # Flatten the gradient for this rollout
                # This gives us g_{p,r} = ∇_{z_{p,r}} L_{p,r}
                grad_flat = response_logit_grad.detach().clone().flatten()
                gradients.append(grad_flat)

            # Clear the retained grad to free memory
            if logits.grad is not None:
                logits.grad = None

        model.zero_grad()

    if len(gradients) == 0:
        return {"gsnr": 0.0, "valid": False, "reason": "no_gradients"}

    # Pad gradients to same length (responses may have different lengths)
    # Find max gradient dimension
    max_grad_len = max(g.shape[0] for g in gradients)

    # Pad shorter gradients with zeros
    padded_gradients = []
    for g in gradients:
        if g.shape[0] < max_grad_len:
            padding = torch.zeros(max_grad_len - g.shape[0], device=g.device, dtype=g.dtype)
            g = torch.cat([g, padding], dim=0)
        padded_gradients.append(g)

    # Stack gradients: [R, grad_dim]
    gradients = torch.stack(padded_gradients, dim=0)

    # Compute mean gradient μ_p
    mu_p = gradients.mean(dim=0)  # [grad_dim]

    # Compute noise (variance across rollouts)
    # noise_p = (1/(R-1)) * sum_r ||g_r - μ_p||²
    if R > 1:
        diff = gradients - mu_p.unsqueeze(0)  # [R, grad_dim]
        noise_p = (diff ** 2).sum(dim=1).mean()  # scalar
    else:
        noise_p = torch.tensor(0.0, device=device)

    # Compute GSNR
    signal = (mu_p ** 2).sum()  # ||μ_p||²
    epsilon = 1e-8
    gsnr = signal / (noise_p + epsilon)

    return {
        "gsnr": gsnr.item(),
        "signal": signal.item(),
        "noise": noise_p.item(),
        "mean_reward": mean_reward.item(),
        "std_reward": std_reward.item(),
        "num_rollouts": R,
        "valid": True,
    }


def generate_responses_vllm(
    llm,
    tokenizer,
    prompts: List[str],
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float = 0.2,
    batch_size: int = 32,
) -> List[List[str]]:
    """
    Generate multiple responses per prompt using vLLM for fast inference.

    Args:
        llm: vLLM LLM instance
        tokenizer: The tokenizer (for reference, vLLM uses its own)
        prompts: List of prompt strings
        num_rollouts: Number of responses per prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (default: 0.2)
        batch_size: Number of prompts to process at a time (default: 32)

    Returns:
        List of lists of response strings
    """
    from vllm import SamplingParams

    # Create sampling parameters for diverse rollouts
    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    all_responses = []

    # Process prompts in batches to avoid OOM
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        logger.info(f"  Generating batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)...")

        # Generate responses for this batch
        outputs = llm.generate(batch_prompts, sampling_params)

        for output in outputs:
            responses = [out.text for out in output.outputs]
            all_responses.append(responses)

    return all_responses


def generate_responses_hf(
    model,
    tokenizer,
    prompts: List[str],
    num_rollouts: int,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.2,
) -> List[List[str]]:
    """
    Generate multiple responses per prompt using HuggingFace generation.
    Fallback when vLLM is not available.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        num_rollouts: Number of responses per prompt
        max_new_tokens: Maximum tokens to generate
        device: CUDA device
        temperature: Sampling temperature (default: 0.2)

    Returns:
        List of lists of response strings
    """
    all_responses = []

    # For GSNR, we need diverse samples, so use temperature > 0
    # Temperature controls diversity in rollouts for gradient variance measurement
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.95,
        "num_return_sequences": num_rollouts,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config,
            )

        # Extract responses (remove prompt)
        responses = []
        for output in outputs:
            response_tokens = output[prompt_len:]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response_text)

        all_responses.append(responses)

    return all_responses


def evaluate_gsnr_with_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    num_rollouts: int,
    max_new_tokens: int,
    max_prompt_length: int,
    temperature: float = 0.2,
    num_prompts: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 32,
    include_intermediate_loops: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate GSNR metric using vLLM for generation and HuggingFace for gradients.

    This function uses a two-phase approach:
    1. Phase 1: Use vLLM to generate all rollouts quickly (batched)
    2. Phase 2: Use HuggingFace to compute gradients for GSNR

    Args:
        model_path: Path to the model
        test_data: List of test examples
        num_rollouts: Number of rollouts per prompt (R in the paper)
        max_new_tokens: Max tokens to generate
        max_prompt_length: Max prompt length
        temperature: Sampling temperature for generation (default: 0.2)
        num_prompts: Number of prompts to evaluate (None for all)
        gpu_memory_utilization: GPU memory fraction for vLLM
        batch_size: Number of prompts to process at a time (default: 32)
        include_intermediate_loops: If True, compute GSNR w.r.t. all loop logits

    Returns:
        - Summary metrics dict
        - List of per-prompt results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Limit number of prompts if specified
    if num_prompts is not None:
        test_data = test_data[:num_prompts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer first (shared between vLLM and HF)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # =========================================================================
    # PHASE 1: Generate all rollouts using vLLM (fast batched generation)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: Generating rollouts with vLLM")
    logger.info("=" * 70)

    # Format all prompts
    formatted_prompts = []
    for example in test_data:
        problem = example.get("problem", "")
        prompt_text = format_math_prompt(problem, tokenizer, use_few_shot=False)
        formatted_prompts.append(prompt_text)

    # Load vLLM model
    logger.info(f"Loading vLLM model from {model_path}...")
    from vllm import LLM

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
    )

    # Generate all responses in batches
    logger.info(f"Generating {num_rollouts} rollouts for {len(formatted_prompts)} prompts (batch_size={batch_size})...")
    all_responses = generate_responses_vllm(
        llm, tokenizer, formatted_prompts, num_rollouts, max_new_tokens, temperature, batch_size
    )

    # Grade all responses
    logger.info("Grading responses...")
    all_rewards = []
    all_response_details = []
    for idx, (example, responses) in enumerate(zip(test_data, all_responses)):
        gold_answer = get_gold_answer(example)
        rewards = []
        response_details = []
        for response in responses:
            pred_answer = extract_boxed_answer(response)
            is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)
            response_details.append({
                "response": response[:500],  # Truncate for storage
                "extracted_answer": pred_answer,
                "correct": is_correct,
            })
        all_rewards.append(rewards)
        all_response_details.append(response_details)

    # Unload vLLM to free GPU memory for HuggingFace
    logger.info("Unloading vLLM model...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Compute gradients using HuggingFace
    # =========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Computing gradients with HuggingFace")
    logger.info("=" * 70)

    # Load HuggingFace model for gradient computation
    logger.info(f"Loading HuggingFace model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Compute GSNR for each prompt
    prompt_results = []
    gsnr_values = []

    logger.info(f"Computing GSNR for {len(test_data)} prompts...")

    for idx, (example, prompt_text, responses, rewards, response_details) in enumerate(
        zip(test_data, formatted_prompts, all_responses, all_rewards, all_response_details)
    ):
        problem = example.get("problem", "")
        gold_answer = get_gold_answer(example)
        level = example.get("level", "unknown")
        subject = example.get("subject") or example.get("type", "Unknown")

        # Compute GSNR for this prompt
        gsnr_result = compute_gsnr_for_prompt(
            model, tokenizer, prompt_text, responses, rewards, device,
            max_length=max_prompt_length + max_new_tokens,
            include_intermediate_loops=include_intermediate_loops,
        )

        prompt_result = {
            "idx": idx,
            "problem": problem[:200],  # Truncate for storage
            "level": level,
            "subject": subject,
            "gold_answer": gold_answer,
            "num_correct": sum(rewards),
            "accuracy": sum(rewards) / len(rewards),
            "gsnr": gsnr_result["gsnr"],
            "signal": gsnr_result.get("signal", 0),
            "noise": gsnr_result.get("noise", 0),
            "valid": gsnr_result.get("valid", False),
            "responses": response_details,
        }
        prompt_results.append(prompt_result)

        if gsnr_result.get("valid", False):
            gsnr_values.append(gsnr_result["gsnr"])

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_data):
            if gsnr_values:
                log_gsnr_values = [np.log(g + 1e-8) for g in gsnr_values]
                mean_log_gsnr = np.mean(log_gsnr_values)
                logger.info(
                    f"Processed {idx + 1}/{len(test_data)} | "
                    f"Valid GSNRs: {len(gsnr_values)} | "
                    f"Mean log(GSNR): {mean_log_gsnr:.4f}"
                )

    # Compute overall GSNR (mean of log-transformed values)
    if gsnr_values:
        epsilon = 1e-8
        log_gsnr_values = [np.log(g + epsilon) for g in gsnr_values]
        overall_gsnr = np.mean(log_gsnr_values)
        std_log_gsnr = np.std(log_gsnr_values)
    else:
        overall_gsnr = float('-inf')
        std_log_gsnr = 0.0
        log_gsnr_values = []

    summary = {
        "overall_gsnr": overall_gsnr,
        "std_log_gsnr": std_log_gsnr,
        "num_valid_prompts": len(gsnr_values),
        "num_total_prompts": len(test_data),
        "num_rollouts": num_rollouts,
        "raw_gsnr_values": gsnr_values,
        "log_gsnr_values": log_gsnr_values,
        "mean_raw_gsnr": np.mean(gsnr_values) if gsnr_values else 0.0,
        "std_raw_gsnr": np.std(gsnr_values) if gsnr_values else 0.0,
    }

    return summary, prompt_results


def evaluate_gsnr_hf_only(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    num_rollouts: int,
    max_new_tokens: int,
    max_prompt_length: int,
    device: torch.device,
    temperature: float = 0.2,
    num_prompts: Optional[int] = None,
    include_intermediate_loops: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate GSNR metric using HuggingFace only (fallback when vLLM unavailable).

    Args:
        model: The model
        tokenizer: The tokenizer
        test_data: List of test examples
        num_rollouts: Number of rollouts per prompt (R in the paper)
        max_new_tokens: Max tokens to generate
        max_prompt_length: Max prompt length
        device: CUDA device
        temperature: Sampling temperature for generation (default: 0.2)
        num_prompts: Number of prompts to evaluate (None for all)
        include_intermediate_loops: If True, compute GSNR w.r.t. all loop logits

    Returns:
        - Summary metrics dict
        - List of per-prompt results
    """
    # Limit number of prompts if specified
    if num_prompts is not None:
        test_data = test_data[:num_prompts]

    prompt_results = []
    gsnr_values = []

    logger.info(f"Evaluating GSNR on {len(test_data)} prompts with R={num_rollouts} rollouts each")

    for idx, example in enumerate(test_data):
        problem = example.get("problem", "")
        gold_answer = get_gold_answer(example)
        level = example.get("level", "unknown")
        subject = example.get("subject") or example.get("type", "Unknown")

        # Format prompt
        prompt_text = format_math_prompt(problem, tokenizer, use_few_shot=False)

        # Generate R responses
        responses_list = generate_responses_hf(
            model, tokenizer, [prompt_text], num_rollouts, max_new_tokens, device, temperature
        )
        responses = responses_list[0]

        # Grade each response
        rewards = []
        response_details = []
        for response in responses:
            pred_answer = extract_boxed_answer(response)
            is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)
            response_details.append({
                "response": response[:500],  # Truncate for storage
                "extracted_answer": pred_answer,
                "correct": is_correct,
            })

        # Compute GSNR for this prompt
        gsnr_result = compute_gsnr_for_prompt(
            model, tokenizer, prompt_text, responses, rewards, device,
            max_length=max_prompt_length + max_new_tokens,
            include_intermediate_loops=include_intermediate_loops,
        )

        prompt_result = {
            "idx": idx,
            "problem": problem[:200],  # Truncate for storage
            "level": level,
            "subject": subject,
            "gold_answer": gold_answer,
            "num_correct": sum(rewards),
            "accuracy": sum(rewards) / len(rewards),
            "gsnr": gsnr_result["gsnr"],
            "signal": gsnr_result.get("signal", 0),
            "noise": gsnr_result.get("noise", 0),
            "valid": gsnr_result.get("valid", False),
            "responses": response_details,
        }
        prompt_results.append(prompt_result)

        if gsnr_result.get("valid", False):
            gsnr_values.append(gsnr_result["gsnr"])

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_data):
            if gsnr_values:
                log_gsnr_values = [np.log(g + 1e-8) for g in gsnr_values]
                mean_log_gsnr = np.mean(log_gsnr_values)
                logger.info(
                    f"Processed {idx + 1}/{len(test_data)} | "
                    f"Valid GSNRs: {len(gsnr_values)} | "
                    f"Mean log(GSNR): {mean_log_gsnr:.4f}"
                )

    # Compute overall GSNR (mean of log-transformed values)
    if gsnr_values:
        epsilon = 1e-8
        log_gsnr_values = [np.log(g + epsilon) for g in gsnr_values]
        overall_gsnr = np.mean(log_gsnr_values)
        std_log_gsnr = np.std(log_gsnr_values)
    else:
        overall_gsnr = float('-inf')
        std_log_gsnr = 0.0
        log_gsnr_values = []

    summary = {
        "overall_gsnr": overall_gsnr,
        "std_log_gsnr": std_log_gsnr,
        "num_valid_prompts": len(gsnr_values),
        "num_total_prompts": len(test_data),
        "num_rollouts": num_rollouts,
        "raw_gsnr_values": gsnr_values,
        "log_gsnr_values": log_gsnr_values,
        "mean_raw_gsnr": np.mean(gsnr_values) if gsnr_values else 0.0,
        "std_raw_gsnr": np.std(gsnr_values) if gsnr_values else 0.0,
    }

    return summary, prompt_results


def main():
    parser = argparse.ArgumentParser(
        description="GSNR evaluation for GRPO checkpoints"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory (e.g., grpo_output/3374119)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gsm8k", "aime24", "beyondaime"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="Number of rollouts per prompt (R in paper)")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for generation (default: 0.2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of prompts to process at a time for vLLM generation (default: 32)")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for fast rollout generation (default: True)")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Disable vLLM and use HuggingFace only")
    parser.add_argument("--include_intermediate_loops", action="store_true",
                        help="Compute GSNR w.r.t. all loop logits (intermediate + final), "
                             "not just the final loop. This measures gradient signal quality "
                             "across the entire Universal Transformer computation.")
    args = parser.parse_args()

    # Handle vLLM flag
    use_vllm = args.use_vllm and not args.no_vllm

    # Validate inputs
    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    # Get benchmark config
    bench_config = BENCHMARK_CONFIGS[args.benchmark]

    if not os.path.isfile(bench_config["test_file"]):
        raise ValueError(f"Test file not found: {bench_config['test_file']}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = "/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results/grpo"

    # Create evaluation output directory
    exp_id = os.path.basename(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(
        args.output_dir,
        f"gsnr_{args.benchmark}_{exp_id}_{args.checkpoint}_{timestamp}"
    )
    os.makedirs(eval_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GSNR Evaluation for GRPO")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Num rollouts (R): {args.num_rollouts}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Use vLLM: {use_vllm}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Include intermediate loops: {args.include_intermediate_loops}")
    logger.info(f"Output directory: {eval_dir}")
    logger.info("")

    # Prepare model
    model_path = prepare_model(
        args.experiment_dir,
        args.checkpoint,
        eval_dir,
    )

    # Load test data
    test_data = load_dataset(bench_config["test_file"])

    # Check if vLLM is available and should be used
    vllm_available = False
    if use_vllm:
        try:
            import vllm
            vllm_available = True
            logger.info("vLLM is available, using accelerated rollout generation")
        except ImportError:
            logger.warning("vLLM not available, falling back to HuggingFace generation")
            vllm_available = False

    # Run evaluation
    if vllm_available and use_vllm:
        # Use vLLM for fast generation, HuggingFace for gradients
        summary, prompt_results = evaluate_gsnr_with_vllm(
            model_path=model_path,
            test_data=test_data,
            num_rollouts=args.num_rollouts,
            max_new_tokens=bench_config["max_new_tokens"],
            max_prompt_length=bench_config["max_prompt_length"],
            temperature=args.temperature,
            num_prompts=args.num_prompts,
            gpu_memory_utilization=args.gpu_memory_utilization,
            batch_size=args.batch_size,
            include_intermediate_loops=args.include_intermediate_loops,
        )
    else:
        # Fallback to HuggingFace only
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        summary, prompt_results = evaluate_gsnr_hf_only(
            model=model,
            tokenizer=tokenizer,
            test_data=test_data,
            num_rollouts=args.num_rollouts,
            max_new_tokens=bench_config["max_new_tokens"],
            max_prompt_length=bench_config["max_prompt_length"],
            device=device,
            temperature=args.temperature,
            num_prompts=args.num_prompts,
            include_intermediate_loops=args.include_intermediate_loops,
        )

    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info("GSNR EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall GSNR (mean log): {summary['overall_gsnr']:.4f}")
    logger.info(f"Std log(GSNR): {summary['std_log_gsnr']:.4f}")
    logger.info(f"Mean raw GSNR: {summary['mean_raw_gsnr']:.6f}")
    logger.info(f"Std raw GSNR: {summary['std_raw_gsnr']:.6f}")
    logger.info(f"Valid prompts: {summary['num_valid_prompts']}/{summary['num_total_prompts']}")

    # Save results
    results_file = os.path.join(eval_dir, "gsnr_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "method": "grpo",
            "experiment_id": exp_id,
            "experiment_dir": args.experiment_dir,
            "checkpoint": args.checkpoint,
            "benchmark": args.benchmark,
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "use_vllm": vllm_available and use_vllm,
            "include_intermediate_loops": args.include_intermediate_loops,
            "timestamp": timestamp,
            "summary": summary,
            "prompt_results": prompt_results,
        }, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save summary JSON (for aggregation)
    summary_file = os.path.join(eval_dir, "gsnr_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "method": "grpo",
            "experiment_id": exp_id,
            "checkpoint": args.checkpoint,
            "benchmark": args.benchmark,
            "num_rollouts": args.num_rollouts,
            "temperature": args.temperature,
            "include_intermediate_loops": args.include_intermediate_loops,
            "overall_gsnr": summary["overall_gsnr"],
            "std_log_gsnr": summary["std_log_gsnr"],
            "mean_raw_gsnr": summary["mean_raw_gsnr"],
            "std_raw_gsnr": summary["std_raw_gsnr"],
            "num_valid_prompts": summary["num_valid_prompts"],
            "num_total_prompts": summary["num_total_prompts"],
            "timestamp": timestamp,
        }, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    # Save summary text
    summary_txt = os.path.join(eval_dir, "gsnr_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("GSNR Evaluation Summary (GRPO)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {args.experiment_dir}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Num rollouts (R): {args.num_rollouts}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Used vLLM: {vllm_available and use_vllm}\n")
        f.write(f"Include intermediate loops: {args.include_intermediate_loops}\n\n")
        f.write(f"Overall GSNR (mean log): {summary['overall_gsnr']:.4f}\n")
        f.write(f"Std log(GSNR): {summary['std_log_gsnr']:.4f}\n")
        f.write(f"Mean raw GSNR: {summary['mean_raw_gsnr']:.6f}\n")
        f.write(f"Std raw GSNR: {summary['std_raw_gsnr']:.6f}\n")
        f.write(f"Valid prompts: {summary['num_valid_prompts']}/{summary['num_total_prompts']}\n")
    logger.info(f"Text summary saved to: {summary_txt}")

    logger.info("\nGSNR evaluation complete!")


if __name__ == "__main__":
    main()
