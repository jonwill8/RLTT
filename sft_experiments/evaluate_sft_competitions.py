#!/usr/bin/env python3
"""
Evaluate SFT checkpoints on competition benchmarks (AIME26, HMMT25).

This wraps evaluate_sft_checkpoint.py, mapping benchmark names to datasets and
forwarding the rest of the arguments unchanged.
"""
import os
import sys
import argparse

import evaluate_sft_checkpoint as base_eval

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
        description="Evaluate SFT checkpoint on competition benchmarks (AIME26, HMMT25)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to SFT checkpoint (LoRA or full model)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional override for test file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=4,
        help="Number of recurrent loops for Ouro model",
    )
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
        help="Sampling temperature",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use 5-shot Chain-of-Thought prompting",
    )
    parser.add_argument(
        "--merge_weights",
        action="store_true",
        help="Merge LoRA weights into the base model for vLLM",
    )
    parser.add_argument(
        "--full_model",
        action="store_true",
        help="Checkpoint is a full model (not LoRA)",
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM and use HuggingFace generation",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for HF generation fallback",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Optional override for base model path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bench_cfg = BENCHMARK_CONFIG[args.benchmark]
    if args.test_file is None:
        args.test_file = bench_cfg["file"]

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, f"eval_{args.benchmark}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build argv for evaluate_sft_checkpoint
    cli_args = [
        "evaluate_sft_checkpoint.py",
        "--checkpoint_path",
        args.checkpoint_path,
        "--test_file",
        args.test_file,
        "--output_dir",
        args.output_dir,
        "--total_ut_steps",
        str(args.total_ut_steps),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
    ]

    if args.use_few_shot:
        cli_args.append("--use_few_shot")
    if args.merge_weights:
        cli_args.append("--merge_weights")
    if args.full_model:
        cli_args.append("--full_model")
    if args.no_vllm:
        cli_args.append("--no_vllm")
    if args.max_samples is not None:
        cli_args.extend(["--max_samples", str(args.max_samples)])
    if args.base_model_path:
        cli_args.extend(["--base_model_path", args.base_model_path])

    sys.argv = cli_args
    base_eval.main()


if __name__ == "__main__":
    main()
