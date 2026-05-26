#!/usr/bin/env python3
"""
Evaluate Qwen3 models on Code Generation benchmarks (HumanEval, MBPP).

This script evaluates Qwen3 baseline models (1.7B, 4B) by:
1. Loading the model with vLLM for fast inference
2. Generating code completions
3. Executing code with test cases and computing pass@k metrics

Uses the shared non_math_code_utils for code extraction and execution.

Usage:
    python eval_qwen3_code.py \
        --model_path /path/to/Qwen3-4B \
        --test_file /path/to/humaneval.test.jsonl \
        --output_dir ./eval_output

    # Evaluate on MBPP:
    python eval_qwen3_code.py \
        --model_path /path/to/Qwen3-1.7B \
        --test_file /path/to/mbpp.test.jsonl
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# Set offline mode before any imports
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
for handler in logger.handlers:
    handler.flush = lambda: None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized code utilities
from non_math_code_utils import (
    extract_code_from_response,
    extract_code_completion,
    check_correctness,
    check_mbpp_correctness,
    compute_pass_at_k,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, will use HuggingFace generation (slower)")


# ============================================================================
# Few-shot examples for code generation
# ============================================================================

_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/code/code_fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    FEW_SHOT_EXAMPLES = {"humaneval": [], "mbpp": []}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 on code generation benchmarks")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Qwen3 model directory"
    )

    # Data arguments
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to code test file (JSONL format)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    # Generation arguments
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length in tokens"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per problem"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem (for pass@k)"
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Use few-shot prompting"
    )

    # Code execution arguments
    parser.add_argument(
        "--execution_timeout",
        type=float,
        default=5.0,
        help="Timeout for code execution in seconds"
    )

    # vLLM arguments
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace generation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for HuggingFace generation"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: model_path/eval_results_code)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def detect_benchmark(test_file: str) -> str:
    """Detect benchmark type from file path."""
    test_file_lower = test_file.lower()
    if "humaneval" in test_file_lower:
        return "humaneval"
    elif "mbpp" in test_file_lower:
        return "mbpp"
    return "humaneval"


def format_humaneval_prompt(
    prompt: str,
    entry_point: str,
    tokenizer,
    use_few_shot: bool = False,
) -> str:
    """Format a HumanEval prompt for Qwen3."""
    system_message = """You are an expert Python programmer. Complete the given function based on its signature and docstring. Only provide the function body, not the signature or docstring. Make sure your code is correct and handles edge cases."""

    if use_few_shot:
        examples = FEW_SHOT_EXAMPLES.get("humaneval", [])
        few_shot_text = ""
        for ex in examples:
            few_shot_text += f"""Function to complete:
```python
{ex.get('prompt', '')}
```

Solution:
```python
{ex.get('code', '')}
```

"""
        user_message = f"""{few_shot_text}Function to complete:
```python
{prompt}
```

Complete the function body. Put your solution in a Python code block."""
    else:
        user_message = f"""Complete the following Python function:

```python
{prompt}
```

Write only the function body (the code that goes after the docstring). Put your solution in a Python code block."""

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


def format_mbpp_prompt(
    prompt: str,
    test_list: List[str],
    tokenizer,
    use_few_shot: bool = False,
) -> str:
    """Format an MBPP prompt for Qwen3."""
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


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load code test data from JSONL."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    if max_samples is not None and max_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, max_samples)

    logger.info(f"Loaded {len(data)} test examples from {file_path}")
    return data


def evaluate_vllm(
    model_path: str,
    test_data: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Evaluate using vLLM for fast inference."""
    from transformers import AutoTokenizer

    benchmark = detect_benchmark(args.test_file)

    logger.info(f"Loading tokenizer from {model_path}...")
    sys.stderr.flush()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info("Tokenizer loaded successfully.")
    sys.stderr.flush()

    max_model_len = args.max_prompt_length + args.max_new_tokens

    logger.info(f"Initializing vLLM engine...")
    logger.info(f"  Max model length: {max_model_len}")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    logger.info(f"  Benchmark: {benchmark}")
    logger.info(f"  Num samples: {args.num_samples}")
    sys.stderr.flush()

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    logger.info("vLLM engine initialized successfully.")
    sys.stderr.flush()

    # Prepare prompts
    logger.info("Preparing prompts...")
    sys.stderr.flush()
    prompts = []
    original_prompts = []
    prompt_lengths = []

    for example in test_data:
        if benchmark == "humaneval":
            orig_prompt = example.get("prompt", "")
            entry_point = example.get("entry_point", "")
            formatted_prompt = format_humaneval_prompt(
                orig_prompt,
                entry_point,
                tokenizer,
                use_few_shot=args.use_few_shot,
            )
        else:  # mbpp
            orig_prompt = example.get("prompt", "")
            test_list = example.get("test_list", [])
            formatted_prompt = format_mbpp_prompt(
                orig_prompt,
                test_list,
                tokenizer,
                use_few_shot=args.use_few_shot,
            )

        prompts.append(formatted_prompt)
        original_prompts.append(orig_prompt)
        prompt_lengths.append(len(tokenizer.encode(formatted_prompt)))

    # Report prompt length statistics
    avg_len = sum(prompt_lengths) / len(prompt_lengths)
    max_len = max(prompt_lengths)
    min_len = min(prompt_lengths)
    logger.info(f"Prompt token lengths - min: {min_len}, avg: {avg_len:.1f}, max: {max_len}")

    over_limit = sum(1 for l in prompt_lengths if l > args.max_prompt_length)
    if over_limit > 0:
        logger.warning(f"{over_limit} prompts exceed max_prompt_length ({args.max_prompt_length})")
    sys.stderr.flush()

    # Setup sampling
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Stop sequences for code generation
    stop_strings = ["```\n", "\n\n\n"]

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.num_samples > 1 else 0.0,
        n=args.num_samples,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        stop=stop_strings,
    )

    logger.info(f"Generating responses for {len(prompts)} problems...")
    sys.stderr.flush()

    outputs = llm.generate(prompts, sampling_params)

    logger.info(f"Generation complete. Executing code and checking correctness...")
    sys.stderr.flush()

    # Process results
    results = []
    correct_count = 0

    for i, (output, orig_prompt, example) in enumerate(
        tqdm(zip(outputs, original_prompts, test_data), total=len(test_data), desc="Evaluating")
    ):
        task_id = example.get("task_id", example.get("id", i))
        sample_results = []

        for j, completion_output in enumerate(output.outputs):
            response = completion_output.text

            # Extract code from response
            if benchmark == "humaneval":
                entry_point = example.get("entry_point", "")
                code = extract_code_from_response(response, entry_point)
                # If the extracted code doesn't have the function definition, prepend prompt
                if f"def {entry_point}" not in code:
                    code = orig_prompt + code

                test_code = example.get("test", "")
                passed, error = check_correctness(
                    code, test_code, args.execution_timeout
                )
            else:  # mbpp
                code = extract_code_from_response(response)
                test_list = example.get("test_list", [])
                test_imports = example.get("test_imports", [])

                passed, error, _ = check_mbpp_correctness(
                    code, test_list, test_imports, args.execution_timeout
                )

            sample_results.append({
                "sample_idx": j,
                "response": response,
                "extracted_code": code,
                "passed": passed,
                "error": error if not passed else "",
            })

        # For pass@1, count if the first sample passes
        if sample_results and sample_results[0]["passed"]:
            correct_count += 1

        results.append({
            "index": i,
            "task_id": task_id,
            "original_prompt": orig_prompt,
            "num_samples": len(sample_results),
            "num_passed": sum(1 for s in sample_results if s["passed"]),
            "samples": sample_results,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            acc = correct_count / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{len(test_data)} | Pass@1: {acc:.2f}% ({correct_count}/{i + 1})")
            sys.stderr.flush()

    return results


def evaluate_hf(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Evaluate using HuggingFace transformers (slower fallback)."""
    benchmark = detect_benchmark(args.test_file)

    logger.info(f"Starting HuggingFace generation...")
    logger.info(f"  Max prompt length: {args.max_prompt_length}")
    logger.info(f"  Few-shot prompting: {args.use_few_shot}")
    logger.info(f"  Benchmark: {benchmark}")
    sys.stderr.flush()

    results = []
    correct_count = 0

    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Generating"):
        batch = test_data[i:i+args.batch_size]

        prompts = []
        original_prompts = []

        for example in batch:
            if benchmark == "humaneval":
                orig_prompt = example.get("prompt", "")
                entry_point = example.get("entry_point", "")
                formatted_prompt = format_humaneval_prompt(
                    orig_prompt,
                    entry_point,
                    tokenizer,
                    use_few_shot=args.use_few_shot,
                )
            else:
                orig_prompt = example.get("prompt", "")
                test_list = example.get("test_list", [])
                formatted_prompt = format_mbpp_prompt(
                    orig_prompt,
                    test_list,
                    tokenizer,
                    use_few_shot=args.use_few_shot,
                )

            prompts.append(formatted_prompt)
            original_prompts.append(orig_prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_prompt_length,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, (output, orig_prompt, example) in enumerate(
            zip(outputs, original_prompts, batch)
        ):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)

            task_id = example.get("task_id", example.get("id", i + j))

            if benchmark == "humaneval":
                entry_point = example.get("entry_point", "")
                code = extract_code_from_response(response, entry_point)
                if f"def {entry_point}" not in code:
                    code = orig_prompt + code

                test_code = example.get("test", "")
                passed, error = check_correctness(
                    code, test_code, args.execution_timeout
                )
            else:
                code = extract_code_from_response(response)
                test_list = example.get("test_list", [])
                test_imports = example.get("test_imports", [])

                passed, error, _ = check_mbpp_correctness(
                    code, test_list, test_imports, args.execution_timeout
                )

            if passed:
                correct_count += 1

            results.append({
                "index": i + j,
                "task_id": task_id,
                "original_prompt": orig_prompt,
                "num_samples": 1,
                "num_passed": 1 if passed else 0,
                "samples": [{
                    "sample_idx": 0,
                    "response": response,
                    "extracted_code": code,
                    "passed": passed,
                    "error": error if not passed else "",
                }],
            })

    return results


def load_model_hf(args):
    """Load model for HuggingFace generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path

    logger.info(f"Loading model from {model_path}")
    sys.stderr.flush()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def compute_metrics(results: List[Dict[str, Any]], k_values: List[int] = [1]) -> Dict[str, Any]:
    """Compute pass@k metrics."""
    total = len(results)
    total_passed = sum(1 for r in results if r["samples"][0]["passed"])

    # Calculate pass@k for different k values
    pass_at_k_results = {}
    for k in k_values:
        task_pass_rates = []
        for item in results:
            n = item["num_samples"]
            c = item["num_passed"]
            if n >= k:
                task_pass_rates.append(compute_pass_at_k(n, c, k))
        if task_pass_rates:
            pass_at_k_results[f"pass@{k}"] = sum(task_pass_rates) / len(task_pass_rates)

    metrics = {
        "pass@1": total_passed / total if total > 0 else 0.0,
        **pass_at_k_results,
        "total_problems": total,
        "total_passed": total_passed,
    }

    return metrics


def main():
    args = parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "eval_results_code")
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    benchmark = detect_benchmark(args.test_file)

    # Extract model name from path
    model_name = os.path.basename(args.model_path)

    logger.info("=" * 60)
    logger.info("Qwen3 Baseline Code Generation Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Max prompt length: {args.max_prompt_length}")
    logger.info(f"Use few-shot: {args.use_few_shot}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Execution timeout: {args.execution_timeout}s")
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Output dir: {args.output_dir}")
    sys.stderr.flush()

    # Load test data
    test_data = load_test_data(args.test_file, args.max_samples)

    # Determine if we should use vLLM
    use_vllm = VLLM_AVAILABLE and not args.no_vllm

    if use_vllm:
        results = evaluate_vllm(args.model_path, test_data, args)
    else:
        logger.info("Using HuggingFace generation (slower than vLLM)")
        model, tokenizer = load_model_hf(args)
        results = evaluate_hf(model, tokenizer, test_data, args)

    # Compute metrics
    k_values = [1]
    if args.num_samples >= 5:
        k_values.append(5)
    if args.num_samples >= 10:
        k_values.append(10)

    metrics = compute_metrics(results, k_values)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Pass@1: {metrics['pass@1']:.2%} ({metrics['total_passed']}/{metrics['total_problems']})")
    for k, v in metrics.items():
        if k.startswith('pass@') and k != 'pass@1':
            logger.info(f"{k}: {v:.2%}")
    logger.info("=" * 60)

    # Save results
    results_file = os.path.join(args.output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "args": {
                "model_path": args.model_path,
                "model_name": model_name,
                "benchmark": benchmark,
                "max_prompt_length": args.max_prompt_length,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "num_samples": args.num_samples,
                "use_few_shot": args.use_few_shot,
                "execution_timeout": args.execution_timeout,
                "test_file": args.test_file,
            },
            "metrics": metrics,
        }, f, indent=2)
    logger.info(f"\nMetrics saved to: {results_file}")

    # Save detailed results
    details_file = os.path.join(args.output_dir, "eval_details.jsonl")
    with open(details_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Detailed results saved to: {details_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Qwen3 Baseline Code Generation Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Max prompt length: {args.max_prompt_length}\n")
        f.write(f"Max new tokens: {args.max_new_tokens}\n")
        f.write(f"Few-shot: {args.use_few_shot}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Num samples: {args.num_samples}\n")
        f.write(f"Execution timeout: {args.execution_timeout}s\n\n")
        f.write(f"Pass@1: {metrics['pass@1']:.2%} ({metrics['total_passed']}/{metrics['total_problems']})\n")
        for k, v in metrics.items():
            if k.startswith('pass@') and k != 'pass@1':
                f.write(f"{k}: {v:.2%}\n")
    logger.info(f"Summary saved to: {summary_file}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
