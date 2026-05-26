#!/usr/bin/env python3
"""
Evaluate DeepSeek-R1-Distill-Qwen baselines on competition benchmarks (AIME26, HMMT25).

This is a thin wrapper around evaluate_baseline.py that maps benchmark names to
their test files and standardizes output metadata.
"""
import os
import json
import argparse
import logging
from typing import Dict

import torch

from evaluate_baseline import (
    ModelConfig,
    DataConfig,
    EvalConfig,
    BaselineEvalConfig,
    evaluate_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BENCHMARK_CONFIG: Dict[str, Dict[str, str]] = {
    "aime26": {
        "name": "AIME 2026",
        "file": "/scratch/gpfs/OLGARUS/jw4199/datasets/aime26/aime2026.jsonl",
    },
    "hmmt25": {
        "name": "HMMT 2025",
        "file": "/scratch/gpfs/OLGARUS/jw4199/datasets/hmmt_2025/hmmt_2025.jsonl",
    },
}

MODEL_PATHS = {
    "1.5b": "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-1.5B",
    "7b": "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-7B",
}


def normalize_model_size(size: str) -> str:
    size_norm = size.lower().replace(" ", "").replace("_", "")
    if size_norm in ("1.5b", "1.5", "1p5b", "15b"):
        return "1.5B"
    if size_norm in ("7b", "7"):
        return "7B"
    raise ValueError(f"Unsupported MODEL_SIZE: {size}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-R1-Distill-Qwen baselines on competition benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_CONFIG.keys()),
        help="Benchmark to evaluate on (aime26 or hmmt25)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7B",
        help="Model size to evaluate (1.5B or 7B)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional override for model path",
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
        help="Maximum number of samples to evaluate",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bench_cfg = BENCHMARK_CONFIG[args.benchmark]
    bench_name = bench_cfg["name"]
    test_file = args.test_file or bench_cfg["file"]

    model_size = normalize_model_size(args.model_size)
    model_path = args.model_path
    if model_path is None:
        key = "1.5b" if model_size.startswith("1.5") else "7b"
        model_path = MODEL_PATHS[key]

    few_shot_tag = "fewshot" if args.use_few_shot else "zeroshot"
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./eval_output",
            f"deepseek_r1_{model_size}_{args.benchmark}_prompt{args.max_prompt_length}_{few_shot_tag}_tokens{args.max_new_tokens}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"DeepSeek-R1 Baseline Evaluation - {bench_name}")
    logger.info("=" * 60)
    logger.info(f"Model size: {model_size}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Benchmark: {bench_name}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Few-shot: {args.use_few_shot}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Max prompt length: {args.max_prompt_length}")

    torch.manual_seed(args.seed)

    config = BaselineEvalConfig(
        model=ModelConfig(model_name_or_path=model_path),
        data=DataConfig(test_file=test_file, max_samples=args.max_samples),
        eval=EvalConfig(
            output_dir=args.output_dir,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_few_shot=args.use_few_shot,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_dtype="bfloat16",
            seed=args.seed,
        ),
    )

    results = evaluate_baseline(config)

    payload = {
        "benchmark": args.benchmark,
        "benchmark_name": bench_name,
        "model_size": model_size,
        "model_path": model_path,
        "args": {
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "use_few_shot": args.use_few_shot,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "max_samples": args.max_samples,
        },
        "metrics": results["metrics"],
        "level_breakdown": results["level_breakdown"],
        "type_breakdown": results["type_breakdown"],
        "timing": results.get("timing", {}),
    }

    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Metrics saved to: {results_file}")

    details_file = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_file, "w") as f:
        for sample in results.get("samples", []):
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Detailed results saved to: {details_file}")

    summary_file = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"DeepSeek-R1 Competition Evaluation - {bench_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: DeepSeek-R1-Distill-Qwen-{model_size}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Benchmark: {bench_name}\n")
        f.write(f"Test file: {test_file}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Max prompt length: {args.max_prompt_length}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(
            f"Overall Accuracy: {results['metrics']['overall_accuracy']*100:.2f}% "
            f"({results['metrics']['overall_correct']}/{results['metrics']['overall_total']})\n"
        )
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
