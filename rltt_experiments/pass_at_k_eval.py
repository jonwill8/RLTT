#!/usr/bin/env python3
"""
Pass@k Evaluation Script for RLTT Experiments

This script evaluates a checkpoint using pass@k metric to investigate potential
entropy collapse. For each problem, it generates k responses and checks if at
least one is correct.

The pass@k metric helps understand model diversity:
- Low pass@k with high k suggests entropy collapse (model always gives same wrong answer)
- Pass@k increasing with k suggests model has diversity in responses

Usage:
    python pass_at_k_eval.py \
        --experiment_dir /path/to/rltt_output/3374111 \
        --checkpoint step_140 \
        --benchmark math500 \
        --k 8 \
        --temperature 0.6 \
        --output_dir /path/to/pass_at_k_results
"""
import os
import sys
import json
import argparse
import logging
import random
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
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

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


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


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Find the merged model path for a given checkpoint."""
    # Check in existing eval directories
    for dirname in os.listdir(experiment_dir):
        if dirname.startswith("eval_"):
            eval_dir = os.path.join(experiment_dir, dirname)
            merged_path = os.path.join(eval_dir, checkpoint, "merged_model")
            if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
                logger.info(f"Found existing merged model at: {merged_path}")
                return merged_path
    return None


def merge_checkpoint(experiment_dir: str, checkpoint: str, output_dir: str) -> str:
    """Merge FSDP checkpoint to HuggingFace format."""
    import subprocess

    step_num = checkpoint.replace("step_", "")
    fsdp_checkpoint_dir = os.path.join(experiment_dir, f"global_step_{step_num}", "actor")
    merged_model_dir = os.path.join(output_dir, "merged_model")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    # Check if already merged
    if os.path.isdir(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
        logger.info(f"Using existing merged model at: {merged_model_dir}")
        return merged_model_dir

    logger.info(f"Merging FSDP checkpoint from: {fsdp_checkpoint_dir}")
    logger.info(f"Target directory: {merged_model_dir}")

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
        import shutil
        shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

    # Run merger
    result = subprocess.run([
        "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--trust-remote-code",
        "--local_dir", fsdp_checkpoint_dir,
        "--target_dir", merged_model_dir,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Merge failed: {result.stderr}")
        raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")

    logger.info("Checkpoint merged successfully")
    return merged_model_dir


def evaluate_pass_at_k(
    llm,
    tokenizer,
    test_data: List[Dict[str, Any]],
    k: int,
    temperature: float,
    max_new_tokens: int,
    base_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate pass@k metric.

    For each problem, generate k responses with different seeds and check if
    at least one is correct.

    Returns:
        - Summary dict with overall metrics
        - List of per-problem results with all k responses
    """
    # Generate k seeds
    set_seed(base_seed)
    seeds = [random.randint(0, 2**31 - 1) for _ in range(k)]

    # Prepare prompts
    prompts = []
    gold_answers = []
    metadata = []  # Store level, subject, etc.

    for example in test_data:
        problem = example.get("problem", "")
        prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=False))
        gold_answers.append(get_gold_answer(example))
        metadata.append({
            "level": example.get("level", "unknown"),
            "subject": example.get("subject", "unknown"),
            "problem": problem,
        })

    # Setup stop tokens
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Collect all k responses for each problem
    all_responses = [[] for _ in range(len(test_data))]  # [problem_idx][sample_idx]

    for sample_idx, seed in enumerate(seeds):
        logger.info(f"  Generating sample {sample_idx + 1}/{k} (seed: {seed})")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            seed=seed,
            stop_token_ids=stop_token_ids if stop_token_ids else None,
        )

        outputs = llm.generate(prompts, sampling_params)

        for problem_idx, output in enumerate(outputs):
            response = output.outputs[0].text
            pred_answer = extract_boxed_answer(response)
            is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answers[problem_idx])

            all_responses[problem_idx].append({
                "seed": seed,
                "response": response,
                "extracted_answer": pred_answer,
                "correct": is_correct,
            })

    # Compute pass@k metrics
    pass_at_k_count = 0
    level_stats = {}
    subject_stats = {}
    per_problem_results = []

    for problem_idx, responses in enumerate(all_responses):
        gold = gold_answers[problem_idx]
        meta = metadata[problem_idx]
        level = meta["level"]
        subject = meta["subject"]

        # Check if any response is correct
        any_correct = any(r["correct"] for r in responses)
        num_correct = sum(1 for r in responses if r["correct"])

        if any_correct:
            pass_at_k_count += 1

        # Track by level
        if level not in level_stats:
            level_stats[level] = {"pass_count": 0, "total": 0, "correct_counts": []}
        level_stats[level]["total"] += 1
        level_stats[level]["correct_counts"].append(num_correct)
        if any_correct:
            level_stats[level]["pass_count"] += 1

        # Track by subject
        if subject not in subject_stats:
            subject_stats[subject] = {"pass_count": 0, "total": 0, "correct_counts": []}
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct_counts"].append(num_correct)
        if any_correct:
            subject_stats[subject]["pass_count"] += 1

        # Store per-problem result
        per_problem_results.append({
            "problem_idx": problem_idx,
            "problem": meta["problem"],
            "level": level,
            "subject": subject,
            "gold_answer": gold,
            "pass_at_k": any_correct,
            "num_correct_of_k": num_correct,
            "responses": responses,  # All k responses with details
        })

    # Compute summary metrics
    total = len(test_data)
    pass_at_k_rate = pass_at_k_count / total if total > 0 else 0

    # Level breakdown
    level_metrics = {}
    for level, stats in level_stats.items():
        level_metrics[level] = {
            "pass_at_k": stats["pass_count"] / stats["total"] if stats["total"] > 0 else 0,
            "pass_count": stats["pass_count"],
            "total": stats["total"],
            "mean_correct_of_k": np.mean(stats["correct_counts"]),
            "std_correct_of_k": np.std(stats["correct_counts"]),
        }

    # Subject breakdown
    subject_metrics = {}
    for subject, stats in subject_stats.items():
        subject_metrics[subject] = {
            "pass_at_k": stats["pass_count"] / stats["total"] if stats["total"] > 0 else 0,
            "pass_count": stats["pass_count"],
            "total": stats["total"],
            "mean_correct_of_k": np.mean(stats["correct_counts"]),
            "std_correct_of_k": np.std(stats["correct_counts"]),
        }

    # Compute distribution of correct counts
    all_correct_counts = [sum(1 for r in responses if r["correct"]) for responses in all_responses]
    correct_count_dist = {i: all_correct_counts.count(i) for i in range(k + 1)}

    summary = {
        "pass_at_k": pass_at_k_rate,
        "pass_count": pass_at_k_count,
        "total": total,
        "k": k,
        "temperature": temperature,
        "seeds_used": seeds,
        "level_metrics": level_metrics,
        "subject_metrics": subject_metrics,
        "correct_count_distribution": correct_count_dist,
        "mean_correct_of_k": np.mean(all_correct_counts),
        "std_correct_of_k": np.std(all_correct_counts),
    }

    return summary, per_problem_results


def main():
    parser = argparse.ArgumentParser(description="Pass@k evaluation for entropy collapse investigation")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory (e.g., rltt_output/3374111)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_CONFIGS.keys()),
                        help="Benchmark to evaluate on")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of samples per problem for pass@k")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating sample seeds")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/pass_at_k_results",
                        help="Output directory for results")
    parser.add_argument("--method", type=str, default="rltt",
                        choices=["grpo", "rltt"],
                        help="Method name for output file naming")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for pass@k evaluation")

    # Get benchmark config
    bench_config = BENCHMARK_CONFIGS[args.benchmark]
    test_file = bench_config["test_file"]
    max_new_tokens = bench_config["max_new_tokens"]
    max_prompt_length = bench_config["max_prompt_length"]

    if not os.path.isfile(test_file):
        raise ValueError(f"Test file not found: {test_file}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output paths
    exp_id = os.path.basename(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.method}_{args.benchmark}_{exp_id}_{args.checkpoint}_k{args.k}_{timestamp}"
    output_json = os.path.join(args.output_dir, f"{output_prefix}.json")
    output_csv = os.path.join(args.output_dir, f"{output_prefix}.csv")

    logger.info("=" * 60)
    logger.info("Pass@k Evaluation for Entropy Collapse Investigation")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Experiment directory: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"k (samples per problem): {args.k}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output JSON: {output_json}")
    logger.info("")

    # Find or create merged model
    model_path = find_merged_model(args.experiment_dir, args.checkpoint)
    if model_path is None:
        # Need to merge the checkpoint
        temp_dir = os.path.join(args.output_dir, f"temp_{args.method}_{exp_id}_{args.checkpoint}")
        os.makedirs(temp_dir, exist_ok=True)
        model_path = merge_checkpoint(args.experiment_dir, args.checkpoint, temp_dir)

    logger.info(f"Using model at: {model_path}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Initialize vLLM
    max_model_len = max_prompt_length + max_new_tokens
    logger.info(f"Initializing vLLM engine (max_model_len={max_model_len})...")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("vLLM engine initialized")

    # Load test data
    test_data = load_dataset(test_file)

    # Run pass@k evaluation
    logger.info("")
    logger.info(f"Running pass@{args.k} evaluation...")
    logger.info("")

    summary, per_problem_results = evaluate_pass_at_k(
        llm=llm,
        tokenizer=tokenizer,
        test_data=test_data,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=max_new_tokens,
        base_seed=args.base_seed,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)
    logger.info(f"Pass@{args.k}: {summary['pass_at_k']*100:.2f}% ({summary['pass_count']}/{summary['total']})")
    logger.info(f"Mean correct per problem: {summary['mean_correct_of_k']:.2f} / {args.k}")
    logger.info("")
    logger.info("Correct count distribution:")
    for count, num_problems in sorted(summary['correct_count_distribution'].items()):
        pct = num_problems / summary['total'] * 100
        logger.info(f"  {count}/{args.k} correct: {num_problems} problems ({pct:.1f}%)")

    # Save full results JSON
    full_results = {
        "method": args.method,
        "benchmark": args.benchmark,
        "experiment_dir": args.experiment_dir,
        "checkpoint": args.checkpoint,
        "k": args.k,
        "temperature": args.temperature,
        "base_seed": args.base_seed,
        "timestamp": timestamp,
        "summary": summary,
        "per_problem_results": per_problem_results,
    }

    with open(output_json, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Full results saved to: {output_json}")

    # Save summary CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["problem_idx", "level", "subject", "gold_answer", "pass_at_k", "num_correct_of_k"]
        for i in range(args.k):
            header.extend([f"sample_{i+1}_correct", f"sample_{i+1}_answer"])
        writer.writerow(header)

        # Data rows
        for result in per_problem_results:
            row = [
                result["problem_idx"],
                result["level"],
                result["subject"],
                result["gold_answer"],
                1 if result["pass_at_k"] else 0,
                result["num_correct_of_k"],
            ]
            for resp in result["responses"]:
                row.extend([1 if resp["correct"] else 0, resp["extracted_answer"]])
            writer.writerow(row)

    logger.info(f"Summary CSV saved to: {output_csv}")

    # Print level breakdown
    logger.info("")
    logger.info("Pass@k by level:")
    for level, metrics in sorted(summary["level_metrics"].items()):
        logger.info(f"  Level {level}: {metrics['pass_at_k']*100:.2f}% ({metrics['pass_count']}/{metrics['total']})")


if __name__ == "__main__":
    main()
