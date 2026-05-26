#!/usr/bin/env python3
"""
Repeated Evaluation Script for GPQA Benchmark (GRPO)

This script evaluates a checkpoint N times under different random seeds
to enable statistical significance testing (t-test) between GRPO and RLTT.

Following the methodology from HRPO paper (Table 9):
- Uses sampling-based evaluation (temperature > 0) instead of greedy decoding
- Runs N independent evaluations with different seeds
- Saves results to CSV for subsequent t-test analysis

Usage:
    python repeated_eval_gpqa.py \
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
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

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

# Add parent directory to path for non_math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_math_utils import (
    extract_answer,
    is_correct,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


# Load few-shot examples
_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    FEW_SHOT_EXAMPLES = {"arc": [], "mmlu": [], "gpqa": []}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load GPQA test set from JSONL."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} problems from {filepath}")
    return data


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
    import shutil

    step_num = checkpoint.replace("step_", "")
    fsdp_checkpoint_dir = os.path.join(experiment_dir, f"global_step_{step_num}", "actor")
    merged_model_dir = os.path.join(output_dir, "merged_model")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    if os.path.isdir(merged_model_dir) and os.path.exists(os.path.join(merged_model_dir, "config.json")):
        logger.info(f"Using existing merged model at: {merged_model_dir}")
        return merged_model_dir

    logger.info(f"Merging FSDP checkpoint from: {fsdp_checkpoint_dir}")
    logger.info(f"Target directory: {merged_model_dir}")

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


def format_mcqa_prompt(
    question: str,
    choices: List[str],
    choice_labels: List[str],
    tokenizer,
    use_few_shot: bool = False,
) -> str:
    """Format a multiple-choice question prompt."""
    choice_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])

    if use_few_shot:
        examples = FEW_SHOT_EXAMPLES.get("gpqa", [])

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


def evaluate_single_run(
    model_path: str,
    test_data: List[Dict[str, Any]],
    tokenizer,
    llm,
    seed: int,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    use_few_shot: bool = False,
) -> Dict[str, Any]:
    """Run a single evaluation with the given seed."""
    set_seed(seed)

    prompts = []
    gold_answers = []
    subjects = []
    all_choice_labels = []

    for example in test_data:
        question = example.get("question", "")
        choices = example.get("choices", [])
        choice_labels = example.get("choice_labels", ["A", "B", "C", "D"][:len(choices)])

        prompts.append(format_mcqa_prompt(question, choices, choice_labels, tokenizer, use_few_shot))
        gold_answers.append(example.get("answer", ""))
        subjects.append(example.get("subject", "unknown"))
        all_choice_labels.append(choice_labels)

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

    outputs = llm.generate(prompts, sampling_params)

    correct_count = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, (output, gold_answer, subject, choice_labels) in enumerate(zip(outputs, gold_answers, subjects, all_choice_labels)):
        response = output.outputs[0].text
        is_correct_answer = is_correct(response, gold_answer, choice_labels)

        if is_correct_answer:
            correct_count += 1

        subject_stats[subject]["total"] += 1
        if is_correct_answer:
            subject_stats[subject]["correct"] += 1

    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0

    subject_accuracies = {}
    for subject, stats in subject_stats.items():
        subject_accuracies[subject] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "subject_accuracies": subject_accuracies,
        "subject_stats": dict(subject_stats),
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated evaluation on GPQA for statistical significance testing")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to experiment output directory (e.g., grpo_output/3374119)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint to evaluate (e.g., step_140)")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of evaluation runs with different seeds")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generating run seeds")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (>0 for variance)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--test_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/gpqa.test.jsonl",
                        help="Path to GPQA test file")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/repeated_eval_results_non-math",
                        help="Output directory for results")
    parser.add_argument("--method", type=str, default="grpo",
                        choices=["grpo", "rltt"],
                        help="Method name for output file naming")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Use few-shot prompting")
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
        f"{args.method}_gpqa_{exp_id}_{args.checkpoint}_n{args.num_runs}_{timestamp}.csv"
    )

    logger.info("=" * 60)
    logger.info("GPQA Repeated Evaluation")
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
            use_few_shot=args.use_few_shot,
        )

        results.append(run_result)
        logger.info(f"  Accuracy: {run_result['accuracy']*100:.2f}% ({run_result['correct']}/{run_result['total']})")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["run", "seed", "accuracy", "correct", "total"]
        all_subjects = sorted(set(
            subject for r in results for subject in r["subject_accuracies"].keys()
        ))
        for subject in all_subjects:
            header.append(f"subject_{subject}_accuracy")
        writer.writerow(header)

        for run_idx, r in enumerate(results):
            row = [run_idx + 1, r["seed"], r["accuracy"], r["correct"], r["total"]]
            for subject in all_subjects:
                row.append(r["subject_accuracies"].get(subject, 0))
            writer.writerow(row)

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
        "benchmark": "gpqa",
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
