#!/usr/bin/env python3
"""
Evaluate the base Ouro-2.6B-Thinking model on competition benchmarks (AIME26, HMMT25).

This mirrors the competition evaluation workflow used for RLTT/GRPO checkpoints but
targets the vanilla baseline model.
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from tqdm import tqdm

# Set offline mode before any imports that might touch the hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ============================================================================
# Benchmark Configuration
# ============================================================================
BENCHMARK_CONFIG = {
    "aime26": {
        "name": "AIME 2026",
        "file": "/scratch/gpfs/OLGARUS/jw4199/datasets/aime26/aime2026.jsonl",
    },
    "hmmt25": {
        "name": "HMMT 2025",
        "file": "/scratch/gpfs/OLGARUS/jw4199/datasets/hmmt_2025/hmmt_2025.jsonl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Ouro baseline model on competition benchmarks (AIME26, HMMT25)"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking",
        help="Path to Ouro baseline model",
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to evaluate on (aime26 or hmmt25)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional override for test file path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )

    # Generation arguments
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3072,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use 5-shot Chain-of-Thought prompting",
    )

    # vLLM arguments
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM for generation (default: enabled)",
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace generation",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )

    # Other arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for HuggingFace generation fallback",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: ./eval_output/<benchmark>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test data from JSONL."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)

    if max_samples is not None and max_samples < len(data):
        import random

        random.seed(42)
        data = random.sample(data, max_samples)

    logger.info(f"Loaded {len(data)} test examples from {file_path}")
    return data


def load_model_and_tokenizer(args):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_with_vllm(args, test_data: List[Dict], tokenizer) -> List[str]:
    """Generate solutions using vLLM."""
    from vllm import LLM, SamplingParams

    model_path = args.model_path

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
        batch = test_data[i : i + args.batch_size]

        prompts = []
        for item in batch:
            problem = item.get("problem", "")
            prompt = format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot)
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
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

    for item, completion in zip(test_data, completions):
        gold_answer = str(get_gold_answer(item))
        pred_answer = extract_boxed_answer(completion)

        is_correct = False
        if pred_answer is not None:
            is_correct = check_math_answer(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1

        # Get category info (HMMT has problem_type, AIME has none)
        problem_type = item.get("problem_type", item.get("type", item.get("subject", "Unknown")))
        if isinstance(problem_type, list):
            problem_type = ", ".join(problem_type)

        by_type[problem_type]["total"] += 1
        if is_correct:
            by_type[problem_type]["correct"] += 1

        results.append(
            {
                "problem": item.get("problem", ""),
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "completion": completion,
                "is_correct": is_correct,
                "type": problem_type,
            }
        )

    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0.0

    type_accuracy = {}
    for t, stats in by_type.items():
        type_accuracy[t] = {
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
        "results": results,
    }


def main():
    args = parse_args()

    # Resolve benchmark config
    bench_cfg = BENCHMARK_CONFIG[args.benchmark]
    bench_name = bench_cfg["name"]
    if args.test_file is None:
        args.test_file = bench_cfg["file"]

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("./eval_output", f"{args.benchmark}")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Ouro Baseline Evaluation - {bench_name}")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Benchmark: {bench_name}")
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
                args.model_path,
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
    logger.info(f"EVALUATION RESULTS - {bench_name}")
    logger.info("=" * 60)
    logger.info(
        f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
        f"({metrics['overall']['correct']}/{metrics['overall']['total']})"
    )
    logger.info("")

    if metrics["by_type"]:
        logger.info("Accuracy by Type:")
        for t, stats in sorted(metrics["by_type"].items()):
            logger.info(f"  {t}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "benchmark": args.benchmark,
                "benchmark_name": bench_name,
                "metrics": {
                    "overall": metrics["overall"],
                    "by_type": metrics["by_type"],
                },
            },
            f,
            indent=2,
        )
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
        f.write(f"Ouro Baseline Evaluation Summary - {bench_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Benchmark: {bench_name}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Test file: {args.test_file}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(
            f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
            f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n\n"
        )
        if metrics["by_type"]:
            f.write("Accuracy by Type:\n")
            for t, stats in sorted(metrics["by_type"].items()):
                f.write(f"  {t}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
