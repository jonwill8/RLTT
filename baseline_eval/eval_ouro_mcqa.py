#!/usr/bin/env python3
"""
Baseline Evaluation Script for Ouro-2.6B-Thinking on MCQA benchmarks.

Evaluates Ouro-2.6B-Thinking on ARC-Challenge, MMLU-STEM, and GPQA.

Uses the shared non_math_utils for answer parsing and checking.

Usage:
    python eval_ouro_mcqa.py \
        --model_path /path/to/Ouro-2.6B-Thinking \
        --test_file /path/to/test.jsonl \
        --output_dir ./eval_output
"""
import os
import sys
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Disable HF hub access for offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Add parent directory to path for non_math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized MCQA utilities
from non_math_utils import (
    extract_answer,
    normalize_answer,
    is_correct,
)

# Import vLLM (required for this script)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)


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
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/arc_challenge.test.jsonl"
    max_samples: Optional[int] = None  # None = all samples


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_output"

    # Generation settings
    max_prompt_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.0  # Greedy decoding
    use_few_shot: bool = False  # Zero-shot by default

    # vLLM settings
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"

    # Misc
    seed: int = 42


@dataclass
class BaselineMCQAEvalConfig:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ============================================================================
# Few-shot examples for MCQA (10 examples per benchmark from train/dev splits)
# ============================================================================

# Load few-shot examples from JSON file for consistency across all eval scripts
_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    logger.warning("Using empty few-shot examples. Run extract_fewshot_examples.py to generate.")
    FEW_SHOT_EXAMPLES = {"arc": [], "mmlu": [], "gpqa": []}


# ============================================================================
# Prompting
# ============================================================================

def format_mcqa_prompt(
    question: str,
    choices: List[str],
    choice_labels: List[str],
    tokenizer,
    use_few_shot: bool = False,
    benchmark: str = "arc",
) -> str:
    """
    Format a multiple-choice question prompt.

    Args:
        question: The question text
        choices: List of answer choice texts
        choice_labels: Labels for choices (A, B, C, D, ...)
        tokenizer: HuggingFace tokenizer
        use_few_shot: Whether to include few-shot examples
        benchmark: Which benchmark (for few-shot selection)

    Returns:
        Formatted prompt string
    """
    # Build choice text
    choice_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choices)])

    if use_few_shot:
        # Build few-shot examples
        examples = FEW_SHOT_EXAMPLES.get(benchmark, FEW_SHOT_EXAMPLES["arc"])

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
        # Zero-shot prompt
        system_message = """You are a helpful assistant that answers multiple choice questions. Think through the problem step by step, then give your final answer as a single letter (A, B, C, or D)."""

        user_message = f"""Question: {question}

{choice_text}

Think step by step and then provide your answer. Put your final answer in \\boxed{{}}. Once you provide the final answer, stop immediately."""

    # Use chat template
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


def detect_benchmark(test_file: str) -> str:
    """Detect benchmark type from file path."""
    test_file_lower = test_file.lower()
    if "arc" in test_file_lower:
        return "arc"
    elif "mmlu" in test_file_lower:
        return "mmlu"
    elif "gpqa" in test_file_lower:
        return "gpqa"
    return "arc"  # default


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate_baseline_mcqa(config: BaselineMCQAEvalConfig) -> Dict[str, Any]:
    """Evaluate Ouro-2.6B-Thinking on MCQA test dataset using vLLM.

    Returns:
        Dictionary containing metrics and detailed results.
    """
    logger.info("=" * 60)
    logger.info("Ouro-2.6B-Thinking Baseline Evaluation (MCQA)")
    logger.info("=" * 60)

    # Detect benchmark type
    benchmark = detect_benchmark(config.data.test_file)
    logger.info(f"Detected benchmark: {benchmark}")

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
    questions = []
    all_choices = []
    all_choice_labels = []
    prompt_lengths = []

    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", [])
        choice_labels = example.get("choice_labels", ["A", "B", "C", "D"][:len(choices)])
        answer = example.get("answer", "")

        questions.append(question)
        gold_answers.append(answer)
        all_choices.append(choices)
        all_choice_labels.append(choice_labels)

        prompt = format_mcqa_prompt(
            question,
            choices,
            choice_labels,
            tokenizer,
            use_few_shot=config.eval.use_few_shot,
            benchmark=benchmark,
        )
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

    # Track by subject if available
    subject_stats = {}

    for i, (output, gold_answer, question, choices, choice_labels, example) in enumerate(
        tqdm(
            zip(outputs, gold_answers, questions, all_choices, all_choice_labels, dataset),
            total=total,
            desc="Evaluating"
        )
    ):
        response = output.outputs[0].text

        # Extract predicted answer using non_math_utils
        pred_answer = extract_answer(response, choice_labels)

        # Check if correct using non_math_utils
        is_correct_answer = is_correct(response, gold_answer, choice_labels)
        if is_correct_answer:
            correct += 1

        # Track by subject
        subject = example.get("subject", "unknown")
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct_answer:
            subject_stats[subject]["correct"] += 1

        # Store detailed output
        all_outputs.append({
            "index": i,
            "question": question,
            "choices": choices,
            "choice_labels": choice_labels,
            "gold_answer": gold_answer,
            "model_response": response,
            "extracted_answer": pred_answer,
            "correct": is_correct_answer,
            "subject": subject,
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

    # Calculate accuracy by subject
    subject_breakdown = {}
    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        subject_breakdown[subject] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }
        # Also add to metrics for CSV extraction
        metrics[f"subject_{subject}_accuracy"] = acc
        metrics[f"subject_{subject}_correct"] = stats["correct"]
        metrics[f"subject_{subject}_total"] = stats["total"]

    # Build results
    results = {
        "metrics": metrics,
        "subject_breakdown": subject_breakdown,
        "config": {
            "model_path": config.model.model_name_or_path,
            "test_file": config.data.test_file,
            "benchmark": benchmark,
            "use_few_shot": config.eval.use_few_shot,
            "max_prompt_length": config.eval.max_prompt_length,
            "max_new_tokens": config.eval.max_new_tokens,
            "temperature": config.eval.temperature,
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
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Total Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    if subject_breakdown and len(subject_breakdown) > 1:
        logger.info("\nAccuracy by Subject:")
        for subject in sorted(subject_breakdown.keys()):
            stats = subject_breakdown[subject]
            logger.info(f"  {subject}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")

    logger.info("=" * 60)

    return results


def main():
    """Main function."""
    import argparse

    # Load default configuration
    config = BaselineMCQAEvalConfig()

    # Parse command line overrides
    parser = argparse.ArgumentParser(description="Evaluate Ouro-2.6B-Thinking on MCQA benchmarks")

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
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Max prompt length (tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Enable few-shot prompting (default: zero-shot)")

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
    results = evaluate_baseline_mcqa(config)

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
        f.write(f"Benchmark: {results['config']['benchmark']}\n")
        f.write(f"Max prompt length: {config.eval.max_prompt_length}\n")
        f.write(f"Max new tokens: {config.eval.max_new_tokens}\n")
        f.write(f"Few-shot prompting: {config.eval.use_few_shot}\n")
        f.write(f"Overall accuracy: {results['metrics']['overall_accuracy']*100:.2f}%\n")
        f.write(f"Correct: {results['metrics']['overall_correct']}/{results['metrics']['overall_total']}\n")
    logger.info(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    main()
