#!/usr/bin/env python3
"""
Repeated Evaluation Script for MBPP Benchmark (GRPO)

This script evaluates a checkpoint N times under different random seeds
to enable statistical significance testing (t-test) between GRPO and RLTT.

Following the methodology from HRPO paper:
- Uses sampling-based evaluation (temperature > 0) instead of greedy decoding
- Runs N independent evaluations with different seeds
- Saves results to CSV for subsequent t-test analysis

Usage:
    python repeated_eval_mbpp.py \
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_math_code_utils import (
    extract_code_from_response,
    check_mbpp_correctness,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


# Load few-shot examples
_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/code/code_fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    FEW_SHOT_EXAMPLES = {"humaneval": [], "mbpp": []}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load MBPP test set from JSONL."""
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


def format_mbpp_prompt(
    prompt: str,
    test_list: List[str],
    tokenizer,
    use_few_shot: bool = False,
) -> str:
    """Format an MBPP prompt."""
    system_message = """You are an expert Python programmer. Write a Python function that solves the given task. Make sure your code passes all the provided test cases."""

    test_str = "\n".join(test_list[:3])

    if use_few_shot:
        examples = FEW_SHOT_EXAMPLES.get("mbpp", [])
        few_shot_text = ""
        for ex in examples:
            ex_tests = "\n".join(ex.get('test_list', [])[:2])
            few_shot_text += f"""Task: {ex.get('prompt', '')}

Test cases:
{ex_tests}

Solution:
```python
{ex.get('code', '')}
```

"""
        user_message = f"""{few_shot_text}Task: {prompt}

Test cases:
{test_str}

Write a Python function that solves this task. Put your solution in a Python code block."""
    else:
        user_message = f"""Task: {prompt}

Test cases that your solution should pass:
{test_str}

Write a Python function that solves this task. Put your solution in a Python code block."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted_prompt


def evaluate_single_run(
    model_path: str,
    test_data: List[Dict[str, Any]],
    tokenizer,
    llm,
    seed: int,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    execution_timeout: float = 5.0,
    use_few_shot: bool = False,
) -> Dict[str, Any]:
    """Run a single evaluation with the given seed."""
    set_seed(seed)

    prompts = []
    original_prompts = []
    test_lists = []
    test_imports_list = []
    task_ids = []

    for example in test_data:
        orig_prompt = example.get("prompt", example.get("text", ""))
        test_list = example.get("test_list", [])
        test_imports = example.get("test_imports", [])
        task_id = example.get("task_id", example.get("id", ""))

        prompts.append(format_mbpp_prompt(orig_prompt, test_list, tokenizer, use_few_shot))
        original_prompts.append(orig_prompt)
        test_lists.append(test_list)
        test_imports_list.append(test_imports)
        task_ids.append(task_id)

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Stop sequences for code generation
    stop_strings = ["```\n", "\n\n\n"]

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        stop=stop_strings,
    )

    outputs = llm.generate(prompts, sampling_params)

    correct_count = 0
    results_details = []

    for i, (output, orig_prompt, test_list, test_imports, task_id) in enumerate(
        zip(outputs, original_prompts, test_lists, test_imports_list, task_ids)
    ):
        response = output.outputs[0].text

        # Extract code from response
        code = extract_code_from_response(response)

        passed, error, _ = check_mbpp_correctness(code, test_list, test_imports, execution_timeout)

        if passed:
            correct_count += 1

        results_details.append({
            "task_id": task_id,
            "passed": passed,
            "error": error if not passed else "",
        })

    total = len(test_data)
    accuracy = correct_count / total if total > 0 else 0

    return {
        "seed": seed,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "details": results_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated evaluation on MBPP for statistical significance testing")
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
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="Maximum prompt length")
    parser.add_argument("--execution_timeout", type=float, default=5.0,
                        help="Timeout for code execution in seconds")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--test_file", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/datasets/code/mbpp.test.jsonl",
                        help="Path to MBPP test file")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/repeated_eval_results_code",
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
        f"{args.method}_mbpp_{exp_id}_{args.checkpoint}_n{args.num_runs}_{timestamp}.csv"
    )

    logger.info("=" * 60)
    logger.info("MBPP Repeated Evaluation")
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {args.experiment_dir}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Execution timeout: {args.execution_timeout}s")
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
            execution_timeout=args.execution_timeout,
            use_few_shot=args.use_few_shot,
        )

        results.append(run_result)
        logger.info(f"  Pass@1: {run_result['accuracy']*100:.2f}% ({run_result['correct']}/{run_result['total']})")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run", "seed", "pass_at_1", "correct", "total"]
        writer.writerow(header)

        for run_idx, r in enumerate(results):
            row = [run_idx + 1, r["seed"], r["accuracy"], r["correct"], r["total"]]
            writer.writerow(row)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)

    accuracies = [r["accuracy"] for r in results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    logger.info(f"Mean Pass@1: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    logger.info(f"Min: {min(accuracies)*100:.2f}%, Max: {max(accuracies)*100:.2f}%")
    logger.info(f"Results saved to: {output_csv}")

    summary = {
        "method": args.method,
        "benchmark": "mbpp",
        "experiment_dir": args.experiment_dir,
        "checkpoint": args.checkpoint,
        "num_runs": args.num_runs,
        "temperature": args.temperature,
        "execution_timeout": args.execution_timeout,
        "mean_pass_at_1": mean_acc,
        "std_pass_at_1": std_acc,
        "min_pass_at_1": min(accuracies),
        "max_pass_at_1": max(accuracies),
        "pass_at_1_values": accuracies,
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
