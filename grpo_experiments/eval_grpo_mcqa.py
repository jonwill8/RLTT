#!/usr/bin/env python3
"""
Evaluate GRPO checkpoint on MCQA benchmarks (ARC-Challenge, MMLU-STEM, GPQA).

This script evaluates a trained GRPO checkpoint by:
1. Loading the base model and optionally LoRA adapter
2. Optionally merging weights for faster inference
3. Generating answers for MCQA problems
4. Computing accuracy metrics (overall and by subject)

Uses the shared non_math_utils for answer parsing and checking.

Usage:
    python eval_grpo_mcqa.py \
        --checkpoint_path /path/to/checkpoint-best \
        --test_file /path/to/arc_challenge.test.jsonl \
        --output_dir ./eval_output

    # With different total_ut_steps:
    python eval_grpo_mcqa.py \
        --checkpoint_path /path/to/checkpoint-best \
        --total_ut_steps 6 \
        --test_file /path/to/mmlu_stem.test.jsonl
"""
import os
import sys
import json
import argparse
import logging
import shutil
import subprocess
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

# Add parent directory to path for non_math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized MCQA utilities
from non_math_utils import (
    extract_answer,
    normalize_answer,
    is_correct,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


# ============================================================================
# Few-shot examples for MCQA (10 examples per benchmark from train/dev splits)
# ============================================================================

# Load few-shot examples from JSON file for consistency across all eval scripts
_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    logger.warning("Using empty few-shot examples. Run extract_fewshot_examples.py to generate.")
    FEW_SHOT_EXAMPLES = {"arc": [], "mmlu": [], "gpqa": []}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint on MCQA benchmarks")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to GRPO checkpoint directory (e.g., checkpoint-best)"
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
        help="Checkpoint is a full model (not LoRA adapter)."
    )

    # Data arguments
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to MCQA test file (JSONL format)"
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
        default=512,
        help="Maximum prompt length in tokens"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
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
        help="Use few-shot prompting"
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


def detect_benchmark(test_file: str) -> str:
    """Detect benchmark type from file path."""
    test_file_lower = test_file.lower()
    if "arc" in test_file_lower:
        return "arc"
    elif "mmlu" in test_file_lower:
        return "mmlu"
    elif "gpqa" in test_file_lower:
        return "gpqa"
    return "arc"


def format_mcqa_prompt(
    question: str,
    choices: List[str],
    choice_labels: List[str],
    tokenizer,
    use_few_shot: bool = False,
    benchmark: str = "arc",
) -> str:
    """Format a multiple-choice question prompt."""
    choice_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])

    if use_few_shot:
        examples = FEW_SHOT_EXAMPLES.get(benchmark, FEW_SHOT_EXAMPLES["arc"])

        few_shot_text = ""
        for ex in examples:
            ex_choice_labels = ["A", "B", "C", "D"][:len(ex["choices"])]
            ex_choice_text = "\n".join([f"{l}. {t}" for l, t in zip(ex_choice_labels, ex["choices"])])
            few_shot_text += f"""Question: {ex["question"]}

{ex_choice_text}

{ex["reasoning"]}

"""

        system_message = """You are a helpful assistant that answers multiple choice questions. Think through the problem step by step, then give your final answer as a single letter (A, B, C, or D)."""

        user_message = f"""{few_shot_text}Question: {question}

{choice_text}

Think step by step and then provide your answer. Put your final answer in \\boxed{{}}. Once you provide the final answer, stop immediately."""

    else:
        system_message = """You are a helpful assistant that answers multiple choice questions. Think through the problem step by step, then give your final answer as a single letter (A, B, C, or D)."""

        user_message = f"""Question: {question}

{choice_text}

Think step by step and then provide your answer. Put your final answer in \\boxed{{}}. Once you provide the final answer, stop immediately."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load MCQA test data from JSONL."""
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
    """Check if checkpoint is a LoRA adapter."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    return os.path.exists(config_path)


def is_fsdp_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is an FSDP checkpoint that needs merging."""
    # FSDP checkpoints have model_world_size_*.pt files
    fsdp_pattern = os.path.join(checkpoint_path, "model_world_size_*.pt")
    import glob
    return len(glob.glob(fsdp_pattern)) > 0


def merge_fsdp_checkpoint(checkpoint_path: str, output_dir: str, total_ut_steps: int) -> str:
    """Merge FSDP checkpoint into HuggingFace format.

    This handles checkpoints from GRPO/RLTT training which use FSDP sharding.
    """
    model_dir = os.path.join(output_dir, "merged_model")

    # Check if already merged
    if os.path.isdir(model_dir):
        safetensors = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        if safetensors:
            logger.info(f"Using existing merged model at: {model_dir}")
            # Update config with total_ut_steps
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                config["total_ut_steps"] = total_ut_steps
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            return model_dir

    logger.info(f"Merging FSDP checkpoint from: {checkpoint_path}")
    sys.stderr.flush()

    # Fix checkpoint files first
    ckpt_hf_dir = os.path.join(checkpoint_path, "huggingface")
    base_model_dir = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"

    # Fix config.json if it exists
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
    if not os.path.exists(modeling_path) and os.path.exists(os.path.join(base_model_dir, "modeling_ouro.py")):
        shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

    # Run merger
    os.makedirs(model_dir, exist_ok=True)
    logger.info("Running verl.model_merger...")
    sys.stderr.flush()

    result = subprocess.run([
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--trust-remote-code",
        "--local_dir", checkpoint_path,
        "--target_dir", model_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Merge failed: {result.stderr}")
        raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")

    logger.info("FSDP checkpoint merged successfully")

    # Update config.json with the specified number of loops
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["total_ut_steps"] = total_ut_steps
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    logger.info(f"Model prepared with total_ut_steps={total_ut_steps}")
    sys.stderr.flush()

    return model_dir


def get_base_model_path(checkpoint_path: str) -> str:
    """Extract base model path from adapter_config.json."""
    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_path}\n"
            f"Use --full_model flag to evaluate full model checkpoints."
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")

    return base_model


def prepare_full_model(args) -> str:
    """Prepare a full model checkpoint for vLLM evaluation."""
    from transformers import AutoConfig

    checkpoint_path = args.checkpoint_path

    logger.info(f"Loading full model checkpoint from {checkpoint_path}")
    sys.stderr.flush()

    config = AutoConfig.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    current_ut_steps = getattr(config, "total_ut_steps", None)

    if current_ut_steps == args.total_ut_steps:
        logger.info(f"Using checkpoint directly (total_ut_steps={args.total_ut_steps})")
        return checkpoint_path

    model_dir = os.path.join(args.output_dir, "model_with_config")
    logger.info(f"Copying model to {model_dir} with total_ut_steps={args.total_ut_steps}")
    sys.stderr.flush()

    os.makedirs(model_dir, exist_ok=True)

    for item in os.listdir(checkpoint_path):
        src = os.path.join(checkpoint_path, item)
        dst = os.path.join(model_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    config.total_ut_steps = args.total_ut_steps
    config.save_pretrained(model_dir)

    logger.info(f"Model prepared with total_ut_steps={args.total_ut_steps}")
    sys.stderr.flush()

    return model_dir


def load_and_merge_model(args) -> str:
    """Load LoRA model, merge weights, and save to temp directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_path = args.checkpoint_path
    base_model_path = args.base_model_path or get_base_model_path(checkpoint_path)

    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    sys.stderr.flush()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )

    if hasattr(base_model.config, "total_ut_steps"):
        base_model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

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

    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps

    merged_dir = os.path.join(args.output_dir, "merged_model")
    logger.info(f"Saving merged model to {merged_dir}")
    sys.stderr.flush()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.save_pretrained(merged_dir)

    modeling_file = os.path.join(base_model_path, "modeling_ouro.py")
    if os.path.exists(modeling_file):
        shutil.copy(modeling_file, merged_dir)
        logger.info("Copied modeling_ouro.py to merged model")

    logger.info(f"Merged model saved successfully")
    sys.stderr.flush()

    del model
    del base_model
    torch.cuda.empty_cache()

    return merged_dir


def evaluate_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Evaluate using vLLM for fast inference."""
    from transformers import AutoTokenizer

    benchmark = detect_benchmark(args.test_file)

    logger.info(f"Loading tokenizer from {model_path}...")
    sys.stderr.flush()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info("Tokenizer loaded successfully.")
    sys.stderr.flush()

    max_model_len = args.max_prompt_length + args.max_new_tokens

    logger.info(f"Initializing vLLM engine...")
    logger.info(f"  Max model length: {max_model_len}")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    logger.info(f"  Benchmark: {benchmark}")
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
    questions = []
    all_choices = []
    all_choice_labels = []
    subjects = []
    prompt_lengths = []

    for example in test_data:
        question = example.get("question", "")
        choices = example.get("choices", [])
        choice_labels = example.get("choice_labels", ["A", "B", "C", "D"][:len(choices)])
        answer = example.get("answer", "")

        questions.append(question)
        gold_answers.append(answer)
        all_choices.append(choices)
        all_choice_labels.append(choice_labels)
        subjects.append(example.get("subject", "unknown"))

        prompt = format_mcqa_prompt(
            question,
            choices,
            choice_labels,
            tokenizer,
            use_few_shot=args.use_few_shot,
            benchmark=benchmark,
        )
        prompts.append(prompt)
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
    for i, (output, gold_answer, question, choices, choice_labels, subject) in enumerate(
        zip(outputs, gold_answers, questions, all_choices, all_choice_labels, subjects)
    ):
        response = output.outputs[0].text

        # Extract predicted answer using non_math_utils
        pred_answer = extract_answer(response, choice_labels)

        # Check if correct using non_math_utils
        is_correct_answer = is_correct(response, gold_answer, choice_labels)
        if is_correct_answer:
            correct_count += 1

        results.append({
            "question": question,
            "choices": choices,
            "choice_labels": choice_labels,
            "subject": subject,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct_answer,
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
    benchmark = detect_benchmark(args.test_file)

    logger.info(f"Starting HuggingFace generation...")
    logger.info(f"  Max prompt length: {args.max_prompt_length}")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    logger.info(f"  Benchmark: {benchmark}")
    sys.stderr.flush()

    results = []

    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Evaluating"):
        batch = test_data[i:i+args.batch_size]

        prompts = []
        gold_answers = []
        questions = []
        all_choices = []
        all_choice_labels = []
        subjects = []

        for example in batch:
            question = example.get("question", "")
            choices = example.get("choices", [])
            choice_labels = example.get("choice_labels", ["A", "B", "C", "D"][:len(choices)])
            answer = example.get("answer", "")

            questions.append(question)
            gold_answers.append(answer)
            all_choices.append(choices)
            all_choice_labels.append(choice_labels)
            subjects.append(example.get("subject", "unknown"))

            prompts.append(format_mcqa_prompt(
                question,
                choices,
                choice_labels,
                tokenizer,
                use_few_shot=args.use_few_shot,
                benchmark=benchmark,
            ))

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

        for j, (output, gold_answer, question, choices, choice_labels, subject) in enumerate(
            zip(outputs, gold_answers, questions, all_choices, all_choice_labels, subjects)
        ):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)

            pred_answer = extract_answer(response, choice_labels)
            is_correct_answer = is_correct(response, gold_answer, choice_labels)

            results.append({
                "question": question,
                "choices": choices,
                "choice_labels": choice_labels,
                "subject": subject,
                "gold_answer": gold_answer,
                "model_response": response,
                "extracted_answer": pred_answer,
                "correct": is_correct_answer,
            })

    return results


def load_peft_model(args):
    """Load LoRA model without merging (for HuggingFace generation)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    checkpoint_path = args.checkpoint_path
    base_model_path = args.base_model_path or get_base_model_path(checkpoint_path)

    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA adapter from {checkpoint_path}")
    sys.stderr.flush()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    if hasattr(base_model.config, "total_ut_steps"):
        base_model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        local_files_only=True,
    )
    model.eval()

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

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps
        logger.info(f"Set total_ut_steps to {args.total_ut_steps}")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy metrics by subject and overall."""
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        subject = r.get("subject", "unknown")
        subject_stats[subject]["total"] += 1
        if r["correct"]:
            subject_stats[subject]["correct"] += 1

    metrics = {
        "by_subject": {},
    }

    total_correct = 0
    total_count = 0

    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics["by_subject"][subject] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": acc,
        }
        total_correct += stats["correct"]
        total_count += stats["total"]

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
        args.output_dir = os.path.join(args.checkpoint_path, "eval_results_mcqa")
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    benchmark = detect_benchmark(args.test_file)

    logger.info("=" * 60)
    logger.info("GRPO Checkpoint MCQA Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Benchmark: {benchmark}")
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

    # Determine if we should use vLLM
    use_vllm = VLLM_AVAILABLE and not args.no_vllm

    # Check for FSDP checkpoint first (from GRPO/RLTT training)
    if is_fsdp_checkpoint(args.checkpoint_path):
        logger.info("Detected FSDP checkpoint - merging to HuggingFace format")
        if not use_vllm:
            raise ValueError("FSDP checkpoints require vLLM for evaluation")
        model_path = merge_fsdp_checkpoint(args.checkpoint_path, args.output_dir, args.total_ut_steps)
        results = evaluate_vllm(model_path, test_data, args)
    # Auto-detect checkpoint type
    elif not args.full_model and not is_lora_checkpoint(args.checkpoint_path):
        logger.info("Auto-detected full model checkpoint (no adapter_config.json found)")
        args.full_model = True

        if use_vllm:
            model_path = prepare_full_model(args)
            results = evaluate_vllm(model_path, test_data, args)
        else:
            logger.info("Using HuggingFace generation (slower than vLLM)")
            model, tokenizer = load_full_model_hf(args)
            results = evaluate_hf(model, tokenizer, test_data, args)
    elif args.full_model:
        logger.info("Using full model checkpoint (not LoRA)")

        if use_vllm:
            model_path = prepare_full_model(args)
            results = evaluate_vllm(model_path, test_data, args)
        else:
            logger.info("Using HuggingFace generation (slower than vLLM)")
            model, tokenizer = load_full_model_hf(args)
            results = evaluate_hf(model, tokenizer, test_data, args)
    elif use_vllm:
        if not args.merge_weights:
            logger.info("vLLM requires merged model. Enabling --merge_weights automatically.")
            args.merge_weights = True

        merged_model_path = load_and_merge_model(args)
        results = evaluate_vllm(merged_model_path, test_data, args)
    else:
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
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    logger.info("")

    if len(metrics['by_subject']) > 1:
        logger.info("Accuracy by Subject:")
        for subject, stats in sorted(metrics['by_subject'].items()):
            logger.info(f"  {subject}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "args": {
                "checkpoint_path": args.checkpoint_path,
                "benchmark": benchmark,
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
        f.write("GRPO Checkpoint MCQA Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Total UT steps: {args.total_ut_steps}\n")
        f.write(f"Max prompt length: {args.max_prompt_length}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n\n")
        if len(metrics['by_subject']) > 1:
            f.write("Accuracy by Subject:\n")
            for subject, stats in sorted(metrics['by_subject'].items()):
                f.write(f"  {subject}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
