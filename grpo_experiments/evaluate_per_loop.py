#!/usr/bin/env python3
"""
Per-Loop Evaluation Script for GRPO Checkpoints

This script evaluates a GRPO checkpoint with a fixed number of recursive loops
(total_ut_steps) to analyze how model performance changes with loop count.

Features:
- Evaluates with configurable loop counts (1, 2, or 3 loops)
- Supports MATH-500, GSM8K, AIME24, and BeyondAIME benchmarks
- Logs both overall metrics and individual model responses
- Uses vLLM for fast inference

Usage:
    python evaluate_per_loop.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --num_loops 1 \
        --benchmark math500

    # Evaluate with 2 loops on GSM8K:
    python evaluate_per_loop.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --num_loops 2 \
        --benchmark gsm8k

    # Evaluate on AIME24:
    python evaluate_per_loop.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --num_loops 3 \
        --benchmark aime24
"""
import os
import sys
import json
import argparse
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set offline mode
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
    # Check in existing eval directories
    for dirname in os.listdir(experiment_dir):
        if dirname.startswith("eval_"):
            eval_dir = os.path.join(experiment_dir, dirname)
            merged_path = os.path.join(eval_dir, checkpoint, "merged_model")
            if os.path.isdir(merged_path) and os.path.exists(os.path.join(merged_path, "config.json")):
                logger.info(f"Found existing merged model at: {merged_path}")
                return merged_path
    return None


def prepare_model_with_loops(
    experiment_dir: str,
    checkpoint: str,
    num_loops: int,
    output_dir: str
) -> str:
    """Prepare model checkpoint with specified number of loops.

    This function:
    1. Finds or creates the merged HuggingFace model
    2. Updates the config.json to set total_ut_steps
    3. Returns the path to the prepared model
    """
    step_num = checkpoint.replace("step_", "")
    fsdp_checkpoint_dir = os.path.join(experiment_dir, f"global_step_{step_num}", "actor")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    # Create output directory for this loop count
    model_dir = os.path.join(output_dir, f"model_loops_{num_loops}")

    # Check if already prepared
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        # Verify the loop count is correct
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("total_ut_steps") == num_loops:
            logger.info(f"Using existing model at: {model_dir} (loops={num_loops})")
            return model_dir

    # First check if there's an existing merged model we can copy from
    existing_merged = find_merged_model(experiment_dir, checkpoint)

    if existing_merged:
        # Copy existing merged model and update config
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

    # Update config.json with the specified number of loops
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["total_ut_steps"] = num_loops

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model prepared with total_ut_steps={num_loops}")
    return model_dir


def evaluate_model(
    model_path: str,
    test_data: List[Dict[str, Any]],
    max_new_tokens: int,
    max_prompt_length: int,
    gpu_memory_utilization: float = 0.9,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Evaluate model on benchmark using vLLM."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Calculate max model length
    max_model_len = max_prompt_length + max_new_tokens

    logger.info(f"Initializing vLLM engine...")
    logger.info(f"  Max model length: {max_model_len} (prompt: {max_prompt_length} + generation: {max_new_tokens})")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("vLLM engine initialized")

    # Prepare prompts
    prompts = []
    gold_answers = []
    problems = []
    levels = []
    subjects = []

    for example in test_data:
        problem = example.get("problem", "")
        problems.append(problem)
        gold_answers.append(get_gold_answer(example))
        prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=False))
        levels.append(example.get("level", "unknown"))
        subjects.append(example.get("subject") or example.get("type", "Unknown"))

    # Setup sampling
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    logger.info(f"Generating responses for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    results = []
    correct_count = 0

    for i, (output, gold_answer, problem, level, subject) in enumerate(
        zip(outputs, gold_answers, problems, levels, subjects)
    ):
        response = output.outputs[0].text
        pred_answer = extract_boxed_answer(response)
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1

        results.append({
            "problem": problem,
            "level": level,
            "subject": subject,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(outputs):
            acc = correct_count / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{len(outputs)} | Running accuracy: {acc:.2f}%")

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy metrics by level, subject, and overall."""
    from collections import defaultdict

    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        level = r.get("level", "unknown")
        subject = r.get("subject", "Unknown")

        level_stats[level]["total"] += 1
        subject_stats[subject]["total"] += 1

        if r["correct"]:
            level_stats[level]["correct"] += 1
            subject_stats[subject]["correct"] += 1

    metrics = {
        "by_level": {},
        "by_subject": {},
    }

    total_correct = 0
    total_count = 0

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics["by_level"][str(level)] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": acc,
        }
        total_correct += stats["correct"]
        total_count += stats["total"]

    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        metrics["by_subject"][subject] = {
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
    parser = argparse.ArgumentParser(
        description="Per-loop evaluation of GRPO checkpoints"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory (e.g., grpo_output/3374119)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--num_loops", type=int, required=True, choices=[1, 2, 3],
                        help="Number of recursive loops (1, 2, or 3)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["math500", "gsm8k", "aime24", "beyondaime"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: experiment_dir/per_loop_eval)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for evaluation")

    # Get benchmark config
    bench_config = BENCHMARK_CONFIGS[args.benchmark]

    if not os.path.isfile(bench_config["test_file"]):
        raise ValueError(f"Test file not found: {bench_config['test_file']}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "per_loop_eval")

    # Create evaluation output directory
    exp_id = os.path.basename(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(
        args.output_dir,
        f"{args.benchmark}_loops{args.num_loops}_{args.checkpoint}_{timestamp}"
    )
    os.makedirs(eval_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"GRPO Per-Loop Evaluation")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Number of loops: {args.num_loops}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Output directory: {eval_dir}")
    logger.info("")

    # Prepare model with specified loop count
    model_path = prepare_model_with_loops(
        args.experiment_dir,
        args.checkpoint,
        args.num_loops,
        eval_dir,
    )

    # Load test data
    test_data = load_dataset(bench_config["test_file"])

    # Run evaluation
    results = evaluate_model(
        model_path,
        test_data,
        max_new_tokens=bench_config["max_new_tokens"],
        max_prompt_length=bench_config["max_prompt_length"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
    )

    # Compute metrics
    metrics = compute_metrics(results)

    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EVALUATION RESULTS (Loops: {args.num_loops})")
    logger.info("=" * 70)
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    logger.info("")

    logger.info("Accuracy by Level:")
    for level, stats in sorted(metrics['by_level'].items()):
        logger.info(f"  Level {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    logger.info("")

    logger.info("Accuracy by Subject:")
    for subject, stats in sorted(metrics['by_subject'].items()):
        logger.info(f"  {subject}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save comprehensive results JSON
    results_file = os.path.join(eval_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "method": "grpo",
            "experiment_id": exp_id,
            "experiment_dir": args.experiment_dir,
            "checkpoint": args.checkpoint,
            "num_loops": args.num_loops,
            "benchmark": args.benchmark,
            "temperature": args.temperature,
            "timestamp": timestamp,
            "metrics": metrics,
            "samples": results,
        }, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")

    # Save summary text file
    summary_file = os.path.join(eval_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write("GRPO Per-Loop Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment: {args.experiment_dir}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of loops: {args.num_loops}\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"Temperature: {args.temperature}\n\n")
        f.write(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
                f"({metrics['overall']['correct']}/{metrics['overall']['total']})\n\n")
        f.write("Accuracy by Level:\n")
        for level, stats in sorted(metrics['by_level'].items()):
            f.write(f"  Level {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write("\nAccuracy by Subject:\n")
        for subject, stats in sorted(metrics['by_subject'].items()):
            f.write(f"  {subject}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
