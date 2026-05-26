#!/usr/bin/env python3
"""
Baseline Evaluation Script for Qwen3 models on math benchmarks.

Evaluates Qwen3-1.7B and Qwen3-4B on MATH-500, GSM8K, AIME24, and BeyondAIME.

Uses the same offline loading, answer checking (math-verify), vLLM acceleration,
and prompting as in grpo_experiments and rltt_experiments.

Usage:
    python evaluate_baseline.py \
        --model_path /path/to/Qwen3-4B \
        --test_file /path/to/test.jsonl \
        --output_dir ./eval_output
"""
import os
import sys
import json
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Disable HF hub access for offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized math utilities
from math_utils import (
    extract_boxed_answer,
    check_math_answer,
    get_gold_answer,
    format_math_prompt,
    MATH_VERIFY_AVAILABLE,
)

# Import vLLM (required for this script)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)

if not MATH_VERIFY_AVAILABLE:
    print("WARNING: math_verify not available. Using fallback answer matching.")
    print("Install with: pip install math-verify")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Qwen3-4B"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    max_samples: Optional[int] = None  # None = all samples


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_output"

    # Generation settings
    max_prompt_length: int = 1024
    max_new_tokens: int = 2048
    temperature: float = 0.0  # Greedy decoding
    use_few_shot: bool = False  # Zero-shot by default

    # vLLM settings
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"

    # Misc
    seed: int = 42


@dataclass
class BaselineEvalConfig:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ============================================================================
# Dataset loading
# ============================================================================

def load_test_dataset(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test dataset from JSONL file."""
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)

    if max_samples is not None and max_samples < len(data):
        # Use deterministic sampling for reproducibility
        import random
        rng = random.Random(42)
        data = rng.sample(data, max_samples)

    return data


def truncate_after_first_boxed(text: str) -> str:
    """Truncate response after the first complete \\boxed{} expression."""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)

    if not match:
        return text

    start = match.end()
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        return text[:pos]

    return text


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate_baseline(config: BaselineEvalConfig) -> Dict[str, Any]:
    """Evaluate Qwen3 model on test dataset using vLLM.

    Returns:
        Dictionary containing metrics and detailed results.
    """
    logger.info("=" * 60)
    logger.info("Qwen3 Baseline Evaluation")
    logger.info("=" * 60)

    # Load dataset
    logger.info(f"Loading test data from: {config.data.test_file}")
    dataset = load_test_dataset(
        config.data.test_file,
        config.data.max_samples
    )
    total = len(dataset)
    logger.info(f"Loaded {total} samples")

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {config.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=True,
    )
    logger.info("Tokenizer loaded successfully")

    # Initialize vLLM engine
    logger.info("Initializing vLLM engine...")
    logger.info(f"  Model: {config.model.model_name_or_path}")
    logger.info(f"  GPU memory utilization: {config.eval.vllm_gpu_memory_utilization}")
    logger.info(f"  Tensor parallel size: {config.eval.vllm_tensor_parallel_size}")
    logger.info(f"  dtype: {config.eval.vllm_dtype}")

    max_model_len = config.eval.max_prompt_length + config.eval.max_new_tokens
    logger.info(f"  max_model_len: {max_model_len}")

    vllm_engine = LLM(
        model=config.model.model_name_or_path,
        tokenizer=config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
        dtype=config.eval.vllm_dtype,
        gpu_memory_utilization=config.eval.vllm_gpu_memory_utilization,
        tensor_parallel_size=config.eval.vllm_tensor_parallel_size,
        max_model_len=max_model_len,
        seed=config.eval.seed,
    )
    logger.info("vLLM engine initialized successfully")

    # Prepare all prompts using tokenizer's chat template
    logger.info(f"Preparing prompts (few-shot: {config.eval.use_few_shot})...")
    prompts = []
    gold_answers = []
    problems = []
    prompt_lengths = []

    for example in dataset:
        problem = example.get("problem", "")
        problems.append(problem)
        gold_answer = get_gold_answer(example)
        gold_answers.append("" if gold_answer is None else str(gold_answer))
        prompt = format_math_prompt(problem, tokenizer, use_few_shot=config.eval.use_few_shot)
        prompts.append(prompt)
        prompt_lengths.append(len(tokenizer.encode(prompt)))

    # Report prompt length statistics
    avg_len = sum(prompt_lengths) / len(prompt_lengths)
    max_len = max(prompt_lengths)
    min_len = min(prompt_lengths)
    logger.info(f"Prompt token lengths - min: {min_len}, avg: {avg_len:.1f}, max: {max_len}")

    # Warn if prompts exceed max length
    over_limit = sum(1 for l in prompt_lengths if l > config.eval.max_prompt_length)
    if over_limit > 0:
        logger.warning(f"{over_limit} prompts exceed max_prompt_length ({config.eval.max_prompt_length})")

    # Configure sampling parameters
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=config.eval.max_new_tokens,
        temperature=config.eval.temperature,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
    )

    logger.info("Generation settings:")
    logger.info(f"  max_tokens: {config.eval.max_new_tokens}")
    logger.info(f"  temperature: {sampling_params.temperature}")
    logger.info(f"  stop_token_ids: {stop_token_ids}")

    # Generate all responses
    logger.info(f"\nGenerating responses for {total} problems...")
    start_time = time.time()

    outputs = vllm_engine.generate(prompts, sampling_params)

    generation_time = time.time() - start_time
    logger.info(f"Generation complete in {generation_time:.2f}s ({total/generation_time:.2f} samples/sec)")

    # Process results
    logger.info("\nProcessing results...")
    correct = 0
    all_outputs = []

    # Track by level and type if available
    level_stats = {}
    type_stats = {}

    for i, (output, gold_answer, problem, example) in enumerate(
        tqdm(zip(outputs, gold_answers, problems, dataset), total=total, desc="Evaluating")
    ):
        response = output.outputs[0].text

        # Truncate response after the first \boxed{} to remove any extra generation
        response = truncate_after_first_boxed(response)

        # Extract predicted answer
        pred_answer = extract_boxed_answer(response)

        # Check if correct using math_verify
        is_correct = pred_answer is not None and check_math_answer(pred_answer, gold_answer)
        if is_correct:
            correct += 1

        # Track by level
        level = example.get("level", "unknown")
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1

        # Track by type
        prob_type = example.get("type", "unknown")
        if prob_type not in type_stats:
            type_stats[prob_type] = {"correct": 0, "total": 0}
        type_stats[prob_type]["total"] += 1
        if is_correct:
            type_stats[prob_type]["correct"] += 1

        # Store detailed output
        all_outputs.append({
            "index": i,
            "problem": problem,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct,
            "level": level,
            "type": prob_type,
        })

        # Log progress every 50 samples
        if (i + 1) % 50 == 0 or (i + 1) == total:
            acc = correct / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{total} samples | Running accuracy: {acc:.2f}% ({correct}/{i + 1})")

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0

    # Build metrics dict
    metrics = {
        "overall_accuracy": accuracy,
        "overall_correct": correct,
        "overall_total": total,
    }

    # Calculate accuracy by level
    level_breakdown = {}
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        level_breakdown[level] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        # Also add to metrics for CSV extraction
        metrics[f"level_{level}_accuracy"] = acc
        metrics[f"level_{level}_correct"] = stats["correct"]
        metrics[f"level_{level}_total"] = stats["total"]

    # Calculate accuracy by type
    type_breakdown = {}
    for prob_type in sorted(type_stats.keys()):
        stats = type_stats[prob_type]
        type_breakdown[prob_type] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # Build results
    results = {
        "metrics": metrics,
        "level_breakdown": level_breakdown,
        "type_breakdown": type_breakdown,
        "config": {
            "model_path": config.model.model_name_or_path,
            "test_file": config.data.test_file,
            "use_few_shot": config.eval.use_few_shot,
            "max_prompt_length": config.eval.max_prompt_length,
            "max_new_tokens": config.eval.max_new_tokens,
            "temperature": config.eval.temperature,
            "math_verify_available": MATH_VERIFY_AVAILABLE,
        },
        "timing": {
            "generation_time_sec": generation_time,
            "samples_per_sec": total / generation_time,
        },
        "samples": all_outputs,
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.model_name_or_path}")
    logger.info(f"Total Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    logger.info(f"math_verify used: {MATH_VERIFY_AVAILABLE}")

    if level_breakdown:
        logger.info("\nAccuracy by Level:")
        for level in sorted(level_breakdown.keys()):
            stats = level_breakdown[level]
            logger.info(f"  {level}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")

    if type_breakdown:
        logger.info("\nAccuracy by Type:")
        for prob_type in sorted(type_breakdown.keys()):
            stats = type_breakdown[prob_type]
            logger.info(f"  {prob_type}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")

    logger.info("=" * 60)

    return results


def main():
    """Main function."""
    import argparse

    # Load default configuration
    config = BaselineEvalConfig()

    # Parse command line overrides
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 models on math benchmarks")

    # Model args
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model weights")

    # Data args
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Number of samples to evaluate (None=all)")

    # Eval args
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="Output directory for results")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Max prompt length (tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Enable 5-shot CoT prompting (default: zero-shot)")

    # vLLM args
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory fraction for vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")

    args = parser.parse_args()

    # Apply overrides
    config.model.model_name_or_path = args.model_path
    config.data.test_file = args.test_file
    config.data.max_samples = args.max_samples
    config.eval.output_dir = args.output_dir
    config.eval.max_prompt_length = args.max_prompt_length
    config.eval.max_new_tokens = args.max_new_tokens
    config.eval.temperature = args.temperature
    config.eval.use_few_shot = args.use_few_shot
    config.eval.vllm_gpu_memory_utilization = args.gpu_memory_utilization
    config.eval.vllm_tensor_parallel_size = args.tensor_parallel_size

    # Create output directory
    os.makedirs(config.eval.output_dir, exist_ok=True)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model.model_name_or_path}")
    logger.info(f"  Test file: {config.data.test_file}")
    logger.info(f"  Max samples: {config.data.max_samples or 'all'}")
    logger.info(f"  Output dir: {config.eval.output_dir}")
    logger.info(f"  Few-shot: {config.eval.use_few_shot}")
    logger.info(f"  Max prompt length: {config.eval.max_prompt_length}")
    logger.info(f"  Max new tokens: {config.eval.max_new_tokens}")
    logger.info(f"  Temperature: {config.eval.temperature}")
    logger.info(f"  vLLM GPU memory: {config.eval.vllm_gpu_memory_utilization}")

    # Run evaluation
    results = evaluate_baseline(config)

    # Save results
    output_file = os.path.join(config.eval.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    # Also save a summary file
    summary_file = os.path.join(config.eval.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Model: {config.model.model_name_or_path}\n")
        f.write(f"Test file: {config.data.test_file}\n")
        f.write(f"Max prompt length: {config.eval.max_prompt_length}\n")
        f.write(f"Max new tokens: {config.eval.max_new_tokens}\n")
        f.write(f"Few-shot prompting: {config.eval.use_few_shot}\n")
        f.write(f"Overall accuracy: {results['metrics']['overall_accuracy']*100:.2f}%\n")
        f.write(f"Correct: {results['metrics']['overall_correct']}/{results['metrics']['overall_total']}\n")
    logger.info(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    main()
