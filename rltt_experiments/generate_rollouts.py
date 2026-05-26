#!/usr/bin/env python3
"""
Generate Model Rollouts for RLTT Experiments

This script generates greedy (temperature=0.0) model responses for a specified
benchmark and saves them to the centralized rollout_comparison_results directory.

Usage:
    python generate_rollouts.py \
        --experiment_dir /path/to/rltt_output/3374111 \
        --checkpoint step_140 \
        --benchmark math500

Benchmarks supported: math500, gsm8k, aime24, beyondaime

Output format: JSON file with all model responses and their correctness scores.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

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


# Benchmark configurations - token limits from run_repeated_eval.slurm
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

# Default output directory
DEFAULT_OUTPUT_DIR = "/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rollout_comparison_results"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
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

    # Check in pass_at_k_results temp directory
    pass_at_k_dir = "/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/pass_at_k_results"
    exp_id = os.path.basename(experiment_dir)
    temp_dir = os.path.join(pass_at_k_dir, f"temp_rltt_{exp_id}_{checkpoint}", "merged_model")
    if os.path.isdir(temp_dir) and os.path.exists(os.path.join(temp_dir, "config.json")):
        logger.info(f"Found existing merged model at: {temp_dir}")
        return temp_dir

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


def generate_rollouts(
    llm,
    tokenizer,
    test_data: List[Dict[str, Any]],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Generate greedy rollouts for all problems.

    Returns a list of results, one per problem, with model response and score.
    """
    # Prepare prompts
    prompts = []
    gold_answers = []
    metadata = []

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

    # Greedy decoding: temperature=0.0
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Greedy decoding
        top_p=1.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    logger.info(f"Generating responses for {len(prompts)} problems (greedy decoding)...")
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    results = []
    correct_count = 0

    for idx, (output, gold, meta) in enumerate(zip(outputs, gold_answers, metadata)):
        response = output.outputs[0].text
        pred_answer = extract_boxed_answer(response)
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold)

        if is_correct:
            correct_count += 1

        results.append({
            "problem_idx": idx,
            "problem": meta["problem"],
            "level": meta["level"],
            "subject": meta["subject"],
            "gold_answer": gold,
            "response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct,
        })

    logger.info(f"Generated {len(results)} responses, {correct_count} correct ({correct_count/len(results)*100:.2f}%)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate greedy rollouts for RLTT experiments")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory (e.g., rltt_output/3374111)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_CONFIGS.keys()),
                        help="Benchmark to evaluate on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--method", type=str, default="rltt",
                        choices=["grpo", "rltt"],
                        help="Method name for output file naming")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Validate inputs
    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for rollout generation")

    # Get benchmark config
    bench_config = BENCHMARK_CONFIGS[args.benchmark]
    test_file = bench_config["test_file"]
    max_new_tokens = bench_config["max_new_tokens"]
    max_prompt_length = bench_config["max_prompt_length"]

    if not os.path.isfile(test_file):
        raise ValueError(f"Test file not found: {test_file}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output path
    exp_id = os.path.basename(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(
        args.output_dir,
        f"{args.method}_{args.benchmark}_{exp_id}_{args.checkpoint}_greedy_{timestamp}.json"
    )

    logger.info("=" * 60)
    logger.info("RLTT Greedy Rollout Generation")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Experiment directory: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Temperature: 0.0 (greedy)")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Max prompt length: {max_prompt_length}")
    logger.info(f"Output: {output_json}")
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

    # Generate rollouts
    logger.info("")
    logger.info("Generating greedy rollouts...")
    logger.info("")

    results = generate_rollouts(
        llm=llm,
        tokenizer=tokenizer,
        test_data=test_data,
        max_new_tokens=max_new_tokens,
    )

    # Compute summary stats
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0

    # Level breakdown
    level_stats = {}
    for r in results:
        level = r["level"]
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if r["correct"]:
            level_stats[level]["correct"] += 1

    # Subject breakdown
    subject_stats = {}
    for r in results:
        subject = r["subject"]
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if r["correct"]:
            subject_stats[subject]["correct"] += 1

    # Save results
    output_data = {
        "method": args.method,
        "benchmark": args.benchmark,
        "experiment_dir": args.experiment_dir,
        "checkpoint": args.checkpoint,
        "model_path": model_path,
        "temperature": 0.0,
        "seed": args.seed,
        "timestamp": timestamp,
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "level_stats": level_stats,
            "subject_stats": subject_stats,
        },
        "results": results,
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total problems: {total}")
    logger.info(f"Correct: {correct} ({accuracy*100:.2f}%)")
    logger.info(f"Results saved to: {output_json}")
    logger.info("")
    logger.info("By Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        logger.info(f"  Level {level}: {acc:.1f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
