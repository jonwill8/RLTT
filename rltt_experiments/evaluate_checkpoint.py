#!/usr/bin/env python3
"""
Evaluate RLTT checkpoint on MATH-500 test set.

This script evaluates a trained RLTT checkpoint by:
1. Loading the checkpoint (LoRA adapter or full model)
2. Generating solutions for MATH-500 problems
3. Computing accuracy metrics (overall and by category)

Features:
- Supports vLLM for fast generation
- Configurable prompt length (zero-shot vs few-shot)
- Greedy or sampled generation
- Detailed metrics by question type and difficulty

Uses centralized math_utils for answer parsing and checking.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from tqdm import tqdm

# Set offline mode before any imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from centralized math_utils (using RL training answer parsing)
from math_utils import (
    rl_extract_boxed_answer as extract_boxed_answer,
    rl_check_math_answer as check_math_answer,
    rl_get_gold_answer as get_gold_answer,
    build_chat_messages,
    format_math_prompt,
    INSTRUCTION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RLTT checkpoint on MATH-500")

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=4,
        help="Number of recurrent loops for Ouro model"
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
        help="Maximum prompt length"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
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
        "--use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM for generation"
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace generation"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )

    # Other arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation"
    )
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


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load MATH-500 test data."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    if max_samples is not None and max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    logger.info(f"Loaded {len(data)} test examples from {file_path}")
    return data


def load_model_and_tokenizer(args):
    """Load model and tokenizer from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = args.checkpoint_path

    logger.info(f"Loading model from {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Set Ouro config
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_with_vllm(args, test_data: List[Dict], tokenizer) -> List[str]:
    """Generate solutions using vLLM."""
    from vllm import LLM, SamplingParams

    model_path = args.checkpoint_path

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model from {model_path}")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
    )

    # Prepare prompts
    prompts = []
    for item in test_data:
        problem = item.get("problem", "")
        prompt = format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot)
        prompts.append(prompt)

    # Set sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_p=1.0,
    )

    # Generate
    logger.info(f"Generating solutions for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)
    completions = [output.outputs[0].text for output in outputs]

    return completions


def generate_with_hf(args, model, tokenizer, test_data: List[Dict]) -> List[str]:
    """Generate solutions using HuggingFace."""
    completions = []

    model.eval()

    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Generating"):
        batch = test_data[i:i + args.batch_size]

        # Prepare prompts
        prompts = []
        for item in batch:
            problem = item.get("problem", "")
            prompt = format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot)
            prompts.append(prompt)

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            completion = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            )
            completions.append(completion)

    return completions


def evaluate_completions(test_data: List[Dict], completions: List[str]) -> Dict[str, Any]:
    """Evaluate completions and compute metrics."""
    results = []
    correct_count = 0
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    for item, completion in zip(test_data, completions):
        # Get gold answer
        gold_answer = get_gold_answer(item)

        # Extract predicted answer
        pred_answer = extract_boxed_answer(completion)

        # Check correctness
        is_correct = False
        if pred_answer is not None:
            is_correct = check_math_answer(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1

        # Get category info
        # MATH-500 uses "subject", training MATH uses "type"
        problem_type = item.get("type") or item.get("subject", "Unknown")
        level = item.get("level", "Unknown")

        # Update by-type stats
        by_type[problem_type]["total"] += 1
        if is_correct:
            by_type[problem_type]["correct"] += 1

        # Update by-level stats
        by_level[level]["total"] += 1
        if is_correct:
            by_level[level]["correct"] += 1

        # Store result
        results.append({
            "problem": item.get("problem", ""),
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "completion": completion,
            "is_correct": is_correct,
            "type": problem_type,
            "level": level,
        })

    # Compute overall accuracy
    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0.0

    # Compute per-type accuracy
    type_accuracy = {}
    for t, stats in by_type.items():
        type_accuracy[t] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
        }

    # Compute per-level accuracy
    level_accuracy = {}
    for l, stats in by_level.items():
        level_accuracy[l] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
        }

    return {
        "overall": {
            "correct": correct_count,
            "total": total,
            "accuracy": accuracy,
        },
        "by_type": type_accuracy,
        "by_level": level_accuracy,
        "results": results,
    }


def main():
    args = parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, "eval_results")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RLTT Checkpoint Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Few-shot: {args.use_few_shot}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")

    # Set seed
    torch.manual_seed(args.seed)

    # Load test data
    test_data = load_test_data(args.test_file, args.max_samples)

    # Generate completions
    use_vllm = args.use_vllm and not args.no_vllm

    if use_vllm:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.checkpoint_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            completions = generate_with_vllm(args, test_data, tokenizer)
        except Exception as e:
            logger.warning(f"vLLM generation failed: {e}")
            logger.info("Falling back to HuggingFace generation...")
            model, tokenizer = load_model_and_tokenizer(args)
            completions = generate_with_hf(args, model, tokenizer, test_data)
    else:
        model, tokenizer = load_model_and_tokenizer(args)
        completions = generate_with_hf(args, model, tokenizer, test_data)

    # Evaluate
    logger.info("Evaluating completions...")
    metrics = evaluate_completions(test_data, completions)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    logger.info("")

    logger.info("Accuracy by Type:")
    for t, stats in sorted(metrics['by_type'].items()):
        logger.info(f"  {t}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    logger.info("")

    logger.info("Accuracy by Level:")
    for l, stats in sorted(metrics['by_level'].items()):
        logger.info(f"  {l}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": {
                "overall": metrics["overall"],
                "by_type": metrics["by_type"],
                "by_level": metrics["by_level"],
            },
        }, f, indent=2)
    logger.info(f"\nMetrics saved to: {results_file}")

    # Save detailed results
    details_file = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_file, "w") as f:
        for result in metrics["results"]:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Detailed results saved to: {details_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"RLTT Checkpoint Evaluation Summary\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Test file: {args.test_file}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n\n")
        f.write(f"Accuracy by Type:\n")
        for t, stats in sorted(metrics['by_type'].items()):
            f.write(f"  {t}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write(f"\nAccuracy by Level:\n")
        for l, stats in sorted(metrics['by_level'].items()):
            f.write(f"  {l}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
