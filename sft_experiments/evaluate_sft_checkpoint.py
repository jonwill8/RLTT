#!/usr/bin/env python3
"""
Evaluate SFT LoRA checkpoint on MATH-500 test set.

This script evaluates a trained SFT LoRA checkpoint by:
1. Loading the base model and LoRA adapter
2. Optionally merging weights for faster inference
3. Generating solutions for MATH-500 problems
4. Computing accuracy metrics (overall and by category)

Features:
- Supports LoRA adapters from SFT training
- Option to merge weights for vLLM compatibility
- Configurable prompt length (zero-shot vs few-shot)
- Configurable total_ut_steps for Ouro model
- Greedy or sampled generation
- Detailed metrics by question type and difficulty

Usage:
    python evaluate_sft_checkpoint.py \
        --checkpoint_path /path/to/checkpoint-best \
        --output_dir ./eval_output \
        --max_new_tokens 3072

    # With configurable prompt length and few-shot:
    python evaluate_sft_checkpoint.py \
        --checkpoint_path /path/to/checkpoint-best \
        --max_prompt_length 4096 \
        --use_few_shot

    # With different total_ut_steps:
    python evaluate_sft_checkpoint.py \
        --checkpoint_path /path/to/checkpoint-best \
        --total_ut_steps 6
"""
import os
import sys
import json
import re
import argparse
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from collections import defaultdict
from tqdm import tqdm

# Set offline mode before any imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.flush = lambda: None

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized math utilities
from math_utils import (
    extract_boxed_answer,
    check_math_answer,
    get_gold_answer,
    format_math_prompt,
    INSTRUCTION,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT LoRA checkpoint on MATH-500")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to SFT LoRA checkpoint directory (e.g., checkpoint-best)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to base model (default: read from adapter_config.json)"
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=4,
        help="Number of recurrent loops for Ouro model (default: 4)"
    )
    parser.add_argument(
        "--merge_weights",
        action="store_true",
        help="Merge LoRA weights into base model (required for vLLM)"
    )
    parser.add_argument(
        "--full_model",
        action="store_true",
        help="Checkpoint is a full model (not LoRA adapter). Use this for checkpoints trained with full parameter updates."
    )

    # Data arguments
    parser.add_argument(
        "--test_file",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
        help="Path to MATH-500 test file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    # Generation arguments
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt length in tokens (increase for few-shot)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3072,
        help="Maximum tokens to generate per problem"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use 5-shot Chain-of-Thought prompting"
    )

    # vLLM arguments
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for HuggingFace generation"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: checkpoint_path/eval_results)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def truncate_after_first_boxed(text: str) -> str:
    """Truncate response after the first complete \\boxed{} expression."""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)

    if not match:
        return text

    start = match.end()
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        return text[:pos]

    return text


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load MATH-500 test data."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    if max_samples is not None and max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    logger.info(f"Loaded {len(data)} test examples from {file_path}")
    return data


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is a LoRA adapter (has adapter_config.json)."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    return os.path.exists(config_path)


def get_base_model_path(checkpoint_path: str) -> str:
    """Extract base model path from adapter_config.json."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_path}\n"
            f"This checkpoint appears to be a full model (not a LoRA adapter).\n"
            f"Use --full_model flag to evaluate full model checkpoints."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")

    return base_model


def prepare_full_model(args) -> str:
    """Prepare a full model checkpoint for vLLM evaluation.

    For full model checkpoints, we just need to ensure total_ut_steps is set
    in the config and return the path.

    Returns path to the model directory (may be copied if config needs updating).
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    checkpoint_path = args.checkpoint_path

    logger.info(f"Loading full model checkpoint from {checkpoint_path}")
    sys.stderr.flush()

    # Check if we need to update total_ut_steps in the config
    config = AutoConfig.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    current_ut_steps = getattr(config, "total_ut_steps", None)

    if current_ut_steps == args.total_ut_steps:
        # Config already has correct total_ut_steps, can use checkpoint directly
        logger.info(f"Using checkpoint directly (total_ut_steps={args.total_ut_steps})")
        return checkpoint_path

    # Need to update config - copy to output dir
    model_dir = os.path.join(args.output_dir, "model_with_config")
    logger.info(f"Copying model to {model_dir} with total_ut_steps={args.total_ut_steps}")
    sys.stderr.flush()

    os.makedirs(model_dir, exist_ok=True)

    # Copy all files from checkpoint
    for item in os.listdir(checkpoint_path):
        src = os.path.join(checkpoint_path, item)
        dst = os.path.join(model_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Update config
    config.total_ut_steps = args.total_ut_steps
    config.save_pretrained(model_dir)

    logger.info(f"Model prepared with total_ut_steps={args.total_ut_steps}")
    sys.stderr.flush()

    return model_dir


def load_and_merge_model(args) -> str:
    """Load LoRA model, merge weights, and save to temp directory.

    Returns path to merged model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_path = args.checkpoint_path
    base_model_path = args.base_model_path or get_base_model_path(checkpoint_path)

    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    sys.stderr.flush()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",  # Load on CPU for merging
    )

    # Set total_ut_steps before loading adapter
    if hasattr(base_model.config, "total_ut_steps"):
        base_model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

    # Load and merge LoRA adapter
    logger.info("Loading LoRA adapter...")
    sys.stderr.flush()
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        local_files_only=True,
    )

    logger.info("Merging LoRA weights...")
    sys.stderr.flush()
    model = model.merge_and_unload()

    # Update config with total_ut_steps
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps

    # Save merged model to output directory
    merged_dir = os.path.join(args.output_dir, "merged_model")
    logger.info(f"Saving merged model to {merged_dir}")
    sys.stderr.flush()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Copy tokenizer files from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.save_pretrained(merged_dir)

    # Copy modeling_ouro.py from base model if it exists
    modeling_file = os.path.join(base_model_path, "modeling_ouro.py")
    if os.path.exists(modeling_file):
        shutil.copy(modeling_file, merged_dir)
        logger.info("Copied modeling_ouro.py to merged model")

    logger.info(f"Merged model saved successfully")
    sys.stderr.flush()

    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()

    return merged_dir


def load_peft_model(args):
    """Load LoRA model without merging (for HuggingFace generation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_path = args.checkpoint_path
    base_model_path = args.base_model_path or get_base_model_path(checkpoint_path)

    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    sys.stderr.flush()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    # Set total_ut_steps
    if hasattr(base_model.config, "total_ut_steps"):
        base_model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        local_files_only=True,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def load_full_model_hf(args):
    """Load full model checkpoint (for HuggingFace generation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = args.checkpoint_path

    logger.info(f"Loading full model from {checkpoint_path}")
    sys.stderr.flush()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    # Set total_ut_steps
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def evaluate_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Evaluate using vLLM for fast inference."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}...")
    sys.stderr.flush()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info("Tokenizer loaded successfully.")
    sys.stderr.flush()

    # Calculate max model length
    max_model_len = args.max_prompt_length + args.max_new_tokens

    logger.info(f"Initializing vLLM engine (this may take a few minutes)...")
    logger.info(f"  Max model length: {max_model_len} (prompt: {args.max_prompt_length} + generation: {args.max_new_tokens})")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    sys.stderr.flush()

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("vLLM engine initialized successfully.")
    sys.stderr.flush()

    # Prepare prompts
    logger.info("Preparing prompts...")
    sys.stderr.flush()
    prompts = []
    gold_answers = []
    problems = []
    levels = []
    types = []
    prompt_lengths = []

    for example in test_data:
        problem = example.get("problem", "")
        problems.append(problem)
        gold_answers.append(get_gold_answer(example))
        prompt = format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot)
        prompts.append(prompt)
        levels.append(example.get("level", "unknown"))
        types.append(example.get("type") or example.get("subject", "Unknown"))
        prompt_lengths.append(len(tokenizer.encode(prompt)))

    # Report prompt length statistics
    avg_len = sum(prompt_lengths) / len(prompt_lengths)
    max_len = max(prompt_lengths)
    min_len = min(prompt_lengths)
    logger.info(f"Prompt token lengths - min: {min_len}, avg: {avg_len:.1f}, max: {max_len}")

    over_limit = sum(1 for l in prompt_lengths if l > args.max_prompt_length)
    if over_limit > 0:
        logger.warning(f"{over_limit} prompts exceed max_prompt_length ({args.max_prompt_length})")
    sys.stderr.flush()

    # Setup sampling
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else 0.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    logger.info(f"Generating responses for {len(prompts)} problems...")
    sys.stderr.flush()

    outputs = llm.generate(prompts, sampling_params)

    logger.info(f"Generation complete. Processing {len(outputs)} outputs...")
    sys.stderr.flush()

    # Process results
    results = []
    correct_count = 0
    for i, (output, gold_answer, problem, level, ptype) in enumerate(
        zip(outputs, gold_answers, problems, levels, types)
    ):
        response = output.outputs[0].text
        response = truncate_after_first_boxed(response)
        pred_answer = extract_boxed_answer(response)
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)
        if is_correct:
            correct_count += 1

        results.append({
            "problem": problem,
            "level": level,
            "type": ptype,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(outputs):
            acc = correct_count / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{len(outputs)} samples | Running accuracy: {acc:.2f}% ({correct_count}/{i + 1})")
            sys.stderr.flush()

    return results


def evaluate_hf(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Evaluate using HuggingFace transformers (slower fallback)."""
    logger.info(f"Starting HuggingFace generation...")
    logger.info(f"  Max prompt length: {args.max_prompt_length}")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    sys.stderr.flush()

    results = []

    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Evaluating"):
        batch = test_data[i:i+args.batch_size]

        prompts = []
        gold_answers = []
        problems = []
        levels = []
        types = []

        for example in batch:
            problem = example.get("problem", "")
            problems.append(problem)
            gold_answers.append(get_gold_answer(example))
            prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot))
            levels.append(example.get("level", "unknown"))
            types.append(example.get("type") or example.get("subject", "Unknown"))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, (output, gold_answer, problem, level, ptype) in enumerate(
            zip(outputs, gold_answers, problems, levels, types)
        ):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            response = truncate_after_first_boxed(response)
            pred_answer = extract_boxed_answer(response)
            is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)

            results.append({
                "problem": problem,
                "level": level,
                "type": ptype,
                "gold_answer": gold_answer,
                "model_response": response,
                "extracted_answer": pred_answer,
                "correct": is_correct,
            })

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy metrics by level, type, and overall."""
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        level = r.get("level", "unknown")
        ptype = r.get("type", "Unknown")

        level_stats[level]["total"] += 1
        type_stats[ptype]["total"] += 1

        if r["correct"]:
            level_stats[level]["correct"] += 1
            type_stats[ptype]["correct"] += 1

    metrics = {
        "by_level": {},
        "by_type": {},
    }

    total_correct = 0
    total_count = 0

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics["by_level"][level] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": acc,
        }
        total_correct += stats["correct"]
        total_count += stats["total"]

    for ptype in sorted(type_stats.keys()):
        stats = type_stats[ptype]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics["by_type"][ptype] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": acc,
        }

    metrics["overall"] = {
        "correct": total_correct,
        "total": total_count,
        "accuracy": total_correct / total_count if total_count > 0 else 0,
    }

    return metrics


def main():
    args = parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, "eval_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SFT Checkpoint Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Total UT steps: {args.total_ut_steps}")
    logger.info(f"Max prompt length: {args.max_prompt_length}")
    logger.info(f"Use few-shot: {args.use_few_shot}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Output dir: {args.output_dir}")
    sys.stderr.flush()

    # Load test data
    test_data = load_test_data(args.test_file, args.max_samples)

    # Auto-detect checkpoint type if not explicitly specified
    if not args.full_model and not is_lora_checkpoint(args.checkpoint_path):
        logger.info("Auto-detected full model checkpoint (no adapter_config.json found)")
        logger.info("Switching to full model mode. Use --full_model to suppress this message.")
        args.full_model = True

    # Determine if we should use vLLM
    use_vllm = VLLM_AVAILABLE and not args.no_vllm

    if args.full_model:
        # Full model checkpoint (not LoRA)
        logger.info("Using full model checkpoint (not LoRA)")

        if use_vllm:
            # Prepare full model for vLLM
            model_path = prepare_full_model(args)
            results = evaluate_vllm(model_path, test_data, args)
        else:
            # Use HuggingFace with full model
            logger.info("Using HuggingFace generation (slower than vLLM)")
            model, tokenizer = load_full_model_hf(args)
            results = evaluate_hf(model, tokenizer, test_data, args)
    elif use_vllm:
        # vLLM requires merged model for LoRA
        if not args.merge_weights:
            logger.info("vLLM requires merged model. Enabling --merge_weights automatically.")
            args.merge_weights = True

        # Merge LoRA weights and save
        merged_model_path = load_and_merge_model(args)

        # Run vLLM evaluation
        results = evaluate_vllm(merged_model_path, test_data, args)
    else:
        # Use HuggingFace with PEFT
        logger.info("Using HuggingFace generation (slower than vLLM)")
        model, tokenizer = load_peft_model(args)
        results = evaluate_hf(model, tokenizer, test_data, args)

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    logger.info("")

    logger.info("Accuracy by Type:")
    for ptype, stats in sorted(metrics['by_type'].items()):
        logger.info(f"  {ptype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    logger.info("")

    logger.info("Accuracy by Level:")
    for level, stats in sorted(metrics['by_level'].items()):
        logger.info(f"  {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "args": {
                "checkpoint_path": args.checkpoint_path,
                "total_ut_steps": args.total_ut_steps,
                "max_prompt_length": args.max_prompt_length,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "use_few_shot": args.use_few_shot,
                "test_file": args.test_file,
            },
            "metrics": metrics,
        }, f, indent=2)
    logger.info(f"\nMetrics saved to: {results_file}")

    # Save detailed results
    details_file = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Detailed results saved to: {details_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write("SFT Checkpoint Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Total UT steps: {args.total_ut_steps}\n")
        f.write(f"Max prompt length: {args.max_prompt_length}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n\n")
        f.write("Accuracy by Type:\n")
        for ptype, stats in sorted(metrics['by_type'].items()):
            f.write(f"  {ptype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write("\nAccuracy by Level:\n")
        for level, stats in sorted(metrics['by_level'].items()):
            f.write(f"  {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
