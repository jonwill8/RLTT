#!/usr/bin/env python3
"""
Repeated Evaluation Script for GSM8K Benchmark (GRPO)

This script evaluates a checkpoint N times under different random seeds
to enable statistical significance testing (t-test) between GRPO and RLTT.

Following the methodology from HRPO paper (Table 9):
- Uses sampling-based evaluation (temperature > 0) instead of greedy decoding
- Runs N independent evaluations with different seeds
- Saves results to CSV for subsequent t-test analysis

Usage:
    python repeated_eval_gsm8k.py \
        --experiment_dir /path/to/grpo_output/3374119 \
        --checkpoint step_140 \
        --num_runs 10 \
        --output_dir /path/to/repeated_eval_results
"""
import os
import sys
import json
import argparse
import logging
import random
import re
import csv
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
    format_math_prompt,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load GSM8K test set from JSONL."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)
    logger.info(f"Loaded {len(data)} problems from {filepath}")
    return data


def get_gsm8k_answer(item: Dict[str, Any]) -> str:
    """Extract the gold answer from a GSM8K example."""
    # GSM8K format: answer field contains the final number after ####
    # Or the solution field contains reasoning with #### answer at the end
    if "answer" in item:
        answer = item["answer"]
        # If answer is already just the number
        if isinstance(answer, (int, float)):
            return str(answer)
        # If answer contains #### delimiter
        if "####" in str(answer):
            return str(answer).split("####")[-1].strip()
        return str(answer).strip()

    # Fallback: check solution field
    if "solution" in item:
        solution = item["solution"]
        if "####" in solution:
            return solution.split("####")[-1].strip()

    return ""


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """Extract the predicted answer from model response.

    GSM8K answers are typically integers. We look for:
    1. \\boxed{} format (if model uses it)
    2. #### delimiter format
    3. Last number in the response
    """
    # Try boxed format first
    boxed_answer = extract_boxed_answer(response)
    if boxed_answer:
        # Clean the answer - remove commas, extract number
        cleaned = re.sub(r'[,$]', '', boxed_answer)
        match = re.search(r'-?\d+\.?\d*', cleaned)
        if match:
            return match.group()
        return boxed_answer

    # Try #### format
    if "####" in response:
        after_delimiter = response.split("####")[-1].strip()
        match = re.search(r'-?\d+\.?\d*', after_delimiter)
        if match:
            return match.group()

    # Fallback: find the last number in the response
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return numbers[-1]

    return None


def check_gsm8k_answer(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer for GSM8K."""
    if pred is None or gold is None:
        return False

    try:
        # Clean both answers
        pred_clean = re.sub(r'[,$]', '', str(pred)).strip()
        gold_clean = re.sub(r'[,$]', '', str(gold)).strip()

        # Try numeric comparison
        pred_num = float(pred_clean)
        gold_num = float(gold_clean)

        # For integers, check exact match
        if pred_num == int(pred_num) and gold_num == int(gold_num):
            return int(pred_num) == int(gold_num)

        # For floats, allow small tolerance
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        # Fallback to string comparison
        return pred_clean.lower() == gold_clean.lower()


def find_merged_model(experiment_dir: str, checkpoint: str) -> Optional[str]:
    """Find the merged model path for a given checkpoint."""
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

    if os.path.isdir(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
        logger.info(f"Using existing merged model at: {merged_model_dir}")
        return merged_model_dir

    logger.info(f"Merging FSDP checkpoint from: {fsdp_checkpoint_dir}")

    # Fix checkpoint files
    ckpt_hf_dir = os.path.join(fsdp_checkpoint_dir, "huggingface")
    base_model_dir = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"

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

    modeling_path = os.path.join(ckpt_hf_dir, "modeling_ouro.py")
    if not os.path.exists(modeling_path):
        import shutil
        shutil.copy(os.path.join(base_model_dir, "modeling_ouro.py"), modeling_path)

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


def evaluate_single_run(
    model_path: str,
    test_data: List[Dict[str, Any]],
    tokenizer,
    llm,
    seed: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run a single evaluation with the given seed."""
    set_seed(seed)

    # Prepare prompts
    prompts = []
    gold_answers = []

    for example in test_data:
        # GSM8K can use "problem" or "question" field
        problem = example.get("problem") or example.get("question", "")
        prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=False))
        gold_answers.append(get_gsm8k_answer(example))

    # Setup sampling
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    correct_count = 0

    for output, gold_answer in zip(outputs, gold_answers):
        response = output.outputs[0].text
        pred_answer = extract_gsm8k_answer(response)
        is_correct = check_gsm8k_answer(pred_answer, gold_answer)

        if is_correct:
            correct_count += 1

    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated evaluation on GSM8K for statistical significance testing")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of evaluation runs with different seeds")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating run seeds")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--test_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/gsm8k/gsm8k.test.jsonl",
                        help="Path to GSM8K test file")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/repeated_eval_results",
                        help="Output directory for results")
    parser.add_argument("--method", type=str, default="grpo",
                        choices=["grpo", "rltt"],
                        help="Method name for output file naming")
    args = parser.parse_args()

    if not os.path.isdir(args.experiment_dir):
        raise ValueError(f"Experiment directory not found: {args.experiment_dir}")

    if not os.path.isfile(args.test_file):
        raise ValueError(f"Test file not found: {args.test_file}")

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is required for repeated evaluation")

    os.makedirs(args.output_dir, exist_ok=True)

    exp_id = os.path.basename(args.experiment_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(
        args.output_dir,
        f"{args.method}_gsm8k_{exp_id}_{args.checkpoint}_n{args.num_runs}_{timestamp}.csv"
    )

    logger.info("=" * 60)
    logger.info("GSM8K Repeated Evaluation")
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info("")

    model_path = find_merged_model(args.experiment_dir, args.checkpoint)
    if model_path is None:
        temp_dir = os.path.join(args.output_dir, f"temp_{args.method}_{exp_id}_{args.checkpoint}")
        os.makedirs(temp_dir, exist_ok=True)
        model_path = merge_checkpoint(args.experiment_dir, args.checkpoint, temp_dir)

    logger.info(f"Using model at: {model_path}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    max_model_len = args.max_prompt_length + args.max_new_tokens
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

    test_data = load_dataset(args.test_file)

    set_seed(args.base_seed)
    run_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_runs)]

    results = []

    for run_idx, seed in enumerate(run_seeds):
        logger.info("")
        logger.info(f"[Run {run_idx + 1}/{args.num_runs}] Seed: {seed}")

        run_result = evaluate_single_run(
            model_path=model_path,
            test_data=test_data,
            tokenizer=tokenizer,
            llm=llm,
            seed=seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        results.append(run_result)
        logger.info(f"  Accuracy: {run_result['accuracy']*100:.2f}% ({run_result['correct']}/{run_result['total']})")

    # Write results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "seed", "accuracy", "correct", "total"])
        for run_idx, r in enumerate(results):
            writer.writerow([run_idx + 1, r["seed"], r["accuracy"], r["correct"], r["total"]])

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    accuracies = [r["accuracy"] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    logger.info(f"Mean accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    logger.info(f"Min: {min(accuracies)*100:.2f}%, Max: {max(accuracies)*100:.2f}%")
    logger.info(f"Results saved to: {output_csv}")

    summary = {
        "method": args.method,
        "benchmark": "gsm8k",
        "experiment_dir": args.experiment_dir,
        "checkpoint": args.checkpoint,
        "num_runs": args.num_runs,
        "temperature": args.temperature,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "accuracies": accuracies,
        "run_seeds": run_seeds,
        "output_csv": output_csv,
    }

    summary_json = output_csv.replace(".csv", "_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_json}")


# Qwen3 checkpoints use the standard verl merger; override the GRPO/Ouro
# helpers copied from grpo_experiments.
from repeated_eval_qwen3_utils import find_merged_model, merge_checkpoint


if __name__ == "__main__":
    main()
