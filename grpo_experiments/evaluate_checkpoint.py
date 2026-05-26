#!/usr/bin/env python3
"""
Evaluate a GRPO checkpoint on MATH-500 test set.

Usage:
    python evaluate_checkpoint.py \
        --checkpoint_dir /path/to/checkpoint \
        --output_dir ./eval_output \
        --max_new_tokens 3072

    # With configurable prompt length and few-shot:
    python evaluate_checkpoint.py \
        --checkpoint_dir /path/to/checkpoint \
        --max_prompt_length 4096 \
        --use_few_shot
"""
import os
import sys
import json
import re
import argparse
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # Force flush to stderr so logs appear immediately in slurm .err files
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
# Ensure logs are flushed immediately
for handler in logger.handlers:
    handler.flush = lambda: None

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized math utilities (using RL training answer parsing)
from math_utils import (
    rl_extract_boxed_answer as extract_boxed_answer,
    rl_check_math_answer as check_math_answer,
    rl_get_gold_answer as get_gold_answer,
    format_math_prompt,
    INSTRUCTION,
    RL_MATH_VERIFY_AVAILABLE as MATH_VERIFY_AVAILABLE,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


# ============================================================================
# Local Helpers
# ============================================================================

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


def load_math500(filepath: str) -> List[Dict[str, Any]]:
    """Load MATH-500 test set from JSONL."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def format_prompt(problem: str, tokenizer, use_few_shot: bool = False) -> str:
    """Format a problem for generation.

    Args:
        problem: The math problem to solve.
        tokenizer: The tokenizer with apply_chat_template method.
        use_few_shot: If True, include 5-shot CoT examples before the problem.

    Returns:
        Formatted prompt string ready for model input.
    """
    # Use centralized format_math_prompt from math_utils
    return format_math_prompt(problem, tokenizer, use_few_shot=use_few_shot)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    max_new_tokens: int,
    max_prompt_length: int = 1024,
    use_few_shot: bool = False,
    gpu_memory_utilization: float = 0.9,
) -> List[Dict[str, Any]]:
    """Evaluate using vLLM for fast inference.

    Args:
        model_path: Path to the HuggingFace model checkpoint.
        test_data: List of MATH-500 test examples.
        max_new_tokens: Maximum tokens to generate per problem.
        max_prompt_length: Maximum prompt length in tokens.
        use_few_shot: If True, use 5-shot CoT prompting.
        gpu_memory_utilization: vLLM GPU memory fraction.

    Returns:
        List of result dicts with predictions and correctness.
    """
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

    # Calculate max model length based on prompt length + generation length
    max_model_len = max_prompt_length + max_new_tokens

    logger.info(f"Initializing vLLM engine (this may take a few minutes)...")
    logger.info(f"  Max model length: {max_model_len} (prompt: {max_prompt_length} + generation: {max_new_tokens})")
    logger.info(f"  Few-shot prompting: {use_few_shot}")
    sys.stderr.flush()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
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
    prompt_lengths = []

    for example in test_data:
        problem = example.get("problem", "")
        problems.append(problem)
        gold_answers.append(get_gold_answer(example))
        prompt = format_prompt(problem, tokenizer, use_few_shot=use_few_shot)
        prompts.append(prompt)
        levels.append(example.get("level", "unknown"))
        # Track token lengths
        prompt_lengths.append(len(tokenizer.encode(prompt)))

    # Report prompt length statistics
    avg_len = sum(prompt_lengths) / len(prompt_lengths)
    max_len = max(prompt_lengths)
    min_len = min(prompt_lengths)
    logger.info(f"Prompt token lengths - min: {min_len}, avg: {avg_len:.1f}, max: {max_len}")

    # Warn if prompts exceed max length
    over_limit = sum(1 for l in prompt_lengths if l > max_prompt_length)
    if over_limit > 0:
        logger.warning(f"{over_limit} prompts exceed max_prompt_length ({max_prompt_length})")
    sys.stderr.flush()

    # Setup sampling
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Greedy decoding for evaluation
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    logger.info(f"Generating responses for {len(prompts)} problems...")
    sys.stderr.flush()

    # Generate with progress tracking
    outputs = llm.generate(prompts, sampling_params)

    logger.info(f"Generation complete. Processing {len(outputs)} outputs...")
    sys.stderr.flush()

    # Process results
    results = []
    correct_count = 0
    for i, (output, gold_answer, problem, level) in enumerate(zip(outputs, gold_answers, problems, levels)):
        response = output.outputs[0].text
        response = truncate_after_first_boxed(response)
        pred_answer = extract_boxed_answer(response)
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)
        if is_correct:
            correct_count += 1

        results.append({
            "problem": problem,
            "level": level,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct,
        })

        # Log progress every 50 samples
        if (i + 1) % 50 == 0 or (i + 1) == len(outputs):
            acc = correct_count / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{len(outputs)} samples | Running accuracy: {acc:.2f}% ({correct_count}/{i + 1})")
            sys.stderr.flush()

    return results


def evaluate_hf(
    model_path: str,
    test_data: List[Dict[str, Any]],
    max_new_tokens: int,
    max_prompt_length: int = 1024,
    use_few_shot: bool = False,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Evaluate using HuggingFace transformers (slower fallback).

    Args:
        model_path: Path to the HuggingFace model checkpoint.
        test_data: List of MATH-500 test examples.
        max_new_tokens: Maximum tokens to generate per problem.
        max_prompt_length: Maximum prompt length in tokens.
        use_few_shot: If True, use 5-shot CoT prompting.
        batch_size: Batch size for generation.

    Returns:
        List of result dicts with predictions and correctness.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading model from {model_path}...")
    logger.info(f"  Max prompt length: {max_prompt_length}")
    logger.info(f"  Few-shot prompting: {use_few_shot}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model.eval()

    results = []

    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i+batch_size]

        prompts = []
        gold_answers = []
        problems = []
        levels = []

        for example in batch:
            problem = example.get("problem", "")
            problems.append(problem)
            gold_answers.append(get_gold_answer(example))
            prompts.append(format_prompt(problem, tokenizer, use_few_shot=use_few_shot))
            levels.append(example.get("level", "unknown"))

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, (output, gold_answer, problem, level) in enumerate(zip(outputs, gold_answers, problems, levels)):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            response = truncate_after_first_boxed(response)
            pred_answer = extract_boxed_answer(response)
            is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)

            results.append({
                "problem": problem,
                "level": level,
                "gold_answer": gold_answer,
                "model_response": response,
                "extracted_answer": pred_answer,
                "correct": is_correct,
            })

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy metrics by level and overall."""
    from collections import defaultdict

    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        level = r.get("level", "unknown")
        level_stats[level]["total"] += 1
        if r["correct"]:
            level_stats[level]["correct"] += 1

    metrics = {}
    total_correct = 0
    total_count = 0

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics[f"level_{level}_accuracy"] = acc
        metrics[f"level_{level}_correct"] = stats["correct"]
        metrics[f"level_{level}_total"] = stats["total"]
        total_correct += stats["correct"]
        total_count += stats["total"]

    metrics["overall_accuracy"] = total_correct / total_count if total_count > 0 else 0
    metrics["overall_correct"] = total_correct
    metrics["overall_total"] = total_count

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint on MATH-500")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to merged HuggingFace checkpoint")
    parser.add_argument("--math500_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
                        help="Path to MATH-500 test JSONL")
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="Output directory for results")
    parser.add_argument("--max_new_tokens", type=int, default=3072,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length in tokens (increase for few-shot)")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Use 5-shot Chain-of-Thought prompting instead of zero-shot")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Force HuggingFace generation instead of vLLM")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for HF generation")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    logger.info(f"Loading MATH-500 from {args.math500_file}...")
    sys.stderr.flush()
    test_data = load_math500(args.math500_file)
    logger.info(f"Loaded {len(test_data)} problems")
    sys.stderr.flush()

    # Run evaluation
    use_vllm = VLLM_AVAILABLE and not args.no_vllm
    logger.info(f"Using {'vLLM' if use_vllm else 'HuggingFace'} for generation")
    logger.info(f"Max prompt length: {args.max_prompt_length}")
    logger.info(f"Few-shot prompting: {args.use_few_shot}")
    sys.stderr.flush()

    if use_vllm:
        results = evaluate_vllm(
            args.checkpoint_dir,
            test_data,
            args.max_new_tokens,
            max_prompt_length=args.max_prompt_length,
            use_few_shot=args.use_few_shot,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        results = evaluate_hf(
            args.checkpoint_dir,
            test_data,
            args.max_new_tokens,
            max_prompt_length=args.max_prompt_length,
            use_few_shot=args.use_few_shot,
            batch_size=args.batch_size,
        )

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Overall: {metrics['overall_correct']}/{metrics['overall_total']} = {metrics['overall_accuracy']*100:.2f}%")
    logger.info("")
    for level in sorted([k for k in metrics if k.startswith("level_") and k.endswith("_accuracy")]):
        level_name = level.replace("level_", "").replace("_accuracy", "")
        correct = metrics[f"level_{level_name}_correct"]
        total = metrics[f"level_{level_name}_total"]
        acc = metrics[level]
        logger.info(f"  Level {level_name}: {correct}/{total} = {acc*100:.2f}%")

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint_dir,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_length": args.max_prompt_length,
            "use_few_shot": args.use_few_shot,
            "metrics": metrics,
            "samples": results,
        }, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint_dir}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(f"Max prompt length: {args.max_prompt_length}\n")
        f.write(f"Few-shot prompting: {args.use_few_shot}\n")
        f.write(f"Overall accuracy: {metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"Correct: {metrics['overall_correct']}/{metrics['overall_total']}\n")
    logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
