#!/usr/bin/env python3
"""
Baseline Evaluation Script for Ouro-2.6B-Thinking on MATH-500 dataset.

Uses the same offline loading, answer checking (math-verify), vLLM acceleration,
and 5-shot CoT prompting as in sft_train.py.

Usage:
    python eval_ouro_math500.py
    # Or with custom config:
    python eval_ouro_math500.py --output_dir ./my_output --batch_size 16
"""
import os
import sys
import json
import logging
import time
from dataclasses import dataclass, field, asdict
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
    build_chat_messages,
    FEW_SHOT_EXAMPLES,
    INSTRUCTION,
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
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration (matching sft_config.py patterns)
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking" 
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    math500_samples: Optional[int] = None  # None = all 500 samples


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_output"

    # Generation settings (matching sft_train.py validation settings)
    max_prompt_length: int = 2048
    max_new_tokens: int = 8192
    temperature: float = 0.0  # Greedy decoding
    do_sample: bool = False
    use_few_shot: bool = True  # 5-shot CoT prompting (as in Ouro paper)

    # vLLM settings (matching sft_config.py ValidationConfig)
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

def load_validation_dataset(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load validation dataset from JSONL file."""
    data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    if max_samples is not None and max_samples < len(data):
        # Use deterministic sampling for reproducibility
        import random
        rng = random.Random(42)
        data = rng.sample(data, max_samples)

    return data


def truncate_after_first_boxed(text: str) -> str:
    """Truncate response after the first complete \\boxed{} expression.

    This removes any extra generation that occurs after the model provides its answer.
    """
    import re
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)

    if not match:
        return text  # No boxed answer found

    # Find the matching closing brace
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
        # Return text up to and including the closing brace
        return text[:pos]

    return text  # Couldn't find matching brace


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate_math500(config: BaselineEvalConfig) -> Dict[str, Any]:
    """Evaluate Ouro-2.6B-Thinking on MATH-500 using vLLM.

    Returns:
        Dictionary containing metrics and detailed results.
    """
    logger.info("=" * 60)
    logger.info("Baseline Evaluation: Ouro-2.6B-Thinking on MATH-500")
    logger.info("=" * 60)

    # Load dataset
    logger.info(f"Loading MATH-500 from: {config.data.math500_test_file}")
    dataset = load_validation_dataset(
        config.data.math500_test_file,
        config.data.math500_samples
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
    logger.info(f"Preparing prompts (5-shot CoT: {config.eval.use_few_shot})...")
    prompts = []
    gold_answers = []
    problems = []

    for example in dataset:
        problem = example.get("problem", "")
        problems.append(problem)
        gold_answers.append(get_gold_answer(example))
        prompts.append(format_math_prompt(problem, tokenizer, use_few_shot=config.eval.use_few_shot))

    # Configure sampling parameters
    # Use tokenizer's special tokens for stop sequences
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        max_tokens=config.eval.max_new_tokens,
        temperature=config.eval.temperature if config.eval.do_sample else 0.0,
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
        level = example.get("level", "Unknown")
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1

        # Track by type
        prob_type = example.get("type", "Unknown")
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

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0

    # Calculate accuracy by level
    level_breakdown = {}
    for level, stats in sorted(level_stats.items()):
        level_breakdown[level] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # Calculate accuracy by type
    type_breakdown = {}
    for prob_type, stats in sorted(type_stats.items()):
        type_breakdown[prob_type] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # Build results
    results = {
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "level_breakdown": level_breakdown,
            "type_breakdown": type_breakdown,
        },
        "config": {
            "model_path": config.model.model_name_or_path,
            "math500_file": config.data.math500_test_file,
            "use_few_shot": config.eval.use_few_shot,
            "max_new_tokens": config.eval.max_new_tokens,
            "temperature": config.eval.temperature,
            "prompt_format": "chatml",
            "math_verify_available": MATH_VERIFY_AVAILABLE,
        },
        "timing": {
            "generation_time_sec": generation_time,
            "samples_per_sec": total / generation_time,
        },
        "outputs": all_outputs,
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.model_name_or_path}")
    logger.info(f"Total Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    logger.info(f"math_verify used: {MATH_VERIFY_AVAILABLE}")

    logger.info("\nAccuracy by Level:")
    for level in sorted(level_breakdown.keys()):
        stats = level_breakdown[level]
        logger.info(f"  {level}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")

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
    parser = argparse.ArgumentParser(description="Evaluate Ouro-2.6B-Thinking on MATH-500")

    # Model args
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model weights")

    # Data args
    parser.add_argument("--math500_file", type=str, default=None,
                        help="Path to MATH-500 test JSONL file")
    parser.add_argument("--math500_samples", type=int, default=None,
                        help="Number of samples to evaluate (None=all)")

    # Eval args
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Max prompt length (tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="Disable 5-shot CoT prompting (use zero-shot)")

    # vLLM args
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None,
                        help="GPU memory fraction for vLLM")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism")

    args = parser.parse_args()

    # Apply overrides
    if args.model_path:
        config.model.model_name_or_path = args.model_path
    if args.math500_file:
        config.data.math500_test_file = args.math500_file
    if args.math500_samples is not None:
        config.data.math500_samples = args.math500_samples
    if args.output_dir:
        config.eval.output_dir = args.output_dir
    if args.max_prompt_length is not None:
        config.eval.max_prompt_length = args.max_prompt_length
    if args.max_new_tokens is not None:
        config.eval.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.eval.temperature = args.temperature
    if args.no_few_shot:
        config.eval.use_few_shot = False
    if args.vllm_gpu_memory_utilization is not None:
        config.eval.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
    if args.vllm_tensor_parallel_size is not None:
        config.eval.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size

    # Create output directory
    os.makedirs(config.eval.output_dir, exist_ok=True)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model.model_name_or_path}")
    logger.info(f"  MATH-500 file: {config.data.math500_test_file}")
    logger.info(f"  MATH-500 samples: {config.data.math500_samples or 'all'}")
    logger.info(f"  Output dir: {config.eval.output_dir}")
    logger.info(f"  5-shot CoT: {config.eval.use_few_shot}")
    logger.info(f"  Prompt format: ChatML")
    logger.info(f"  Max new tokens: {config.eval.max_new_tokens}")
    logger.info(f"  Temperature: {config.eval.temperature}")
    logger.info(f"  vLLM GPU memory: {config.eval.vllm_gpu_memory_utilization}")

    # Run evaluation
    results = evaluate_math500(config)

    # Save results
    output_file = os.path.join(config.eval.output_dir, "math500_eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")

    # Also save a summary file (without full outputs for quick viewing)
    summary = {
        "metrics": results["metrics"],
        "config": results["config"],
        "timing": results["timing"],
    }
    summary_file = os.path.join(config.eval.output_dir, "math500_eval_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    main()
