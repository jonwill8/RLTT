#!/usr/bin/env python3
"""
Baseline Evaluation Script for Ouro-2.6B-Thinking on Code Generation benchmarks.

Evaluates Ouro-2.6B-Thinking on HumanEval and MBPP.

Uses the shared non_math_code_utils for code extraction and execution.

Usage:
    python eval_ouro_code.py \
        --model_path /path/to/Ouro-2.6B-Thinking \
        --test_file /path/to/humaneval.test.jsonl \
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
    test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/code/humaneval.test.jsonl"
    max_samples: Optional[int] = None  # None = all samples


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_output"

    # Generation settings
    max_prompt_length: int = 2048
    max_new_tokens: int = 1024
    temperature: float = 0.0  # Greedy decoding for pass@1
    num_samples: int = 1  # Number of samples per problem (for pass@k)
    use_few_shot: bool = False  # Zero-shot by default

    # Code execution settings
    execution_timeout: float = 5.0  # Timeout for code execution in seconds

    # vLLM settings
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"

    # Misc
    seed: int = 42


@dataclass
class BaselineCodeEvalConfig:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ============================================================================
# Few-shot examples for code generation
# ============================================================================

# Load few-shot examples from JSON file
_FEWSHOT_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/code/code_fewshot_examples.json"
try:
    with open(_FEWSHOT_FILE, "r") as _f:
        FEW_SHOT_EXAMPLES = json.load(_f)
except FileNotFoundError:
    logger.warning(f"Few-shot examples file not found: {_FEWSHOT_FILE}")
    logger.warning("Using empty few-shot examples. Run prepare_code_train_splits.py to generate.")
    FEW_SHOT_EXAMPLES = {"humaneval": [], "mbpp": []}


# ============================================================================
# Prompting
# ============================================================================

def format_humaneval_prompt(
    prompt: str,
    entry_point: str,
    tokenizer,
    use_few_shot: bool = False,
) -> str:
    """
    Format a HumanEval prompt.

    HumanEval provides a function signature with docstring, and expects
    the model to complete the function body.

    Args:
        prompt: The function signature and docstring
        entry_point: The function name
        tokenizer: HuggingFace tokenizer
        use_few_shot: Whether to include few-shot examples

    Returns:
        Formatted prompt string
    """
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
    """
    Format an MBPP prompt.

    MBPP provides a task description and test cases.

    Args:
        prompt: The task description
        test_list: List of assertion tests
        tokenizer: HuggingFace tokenizer
        use_few_shot: Whether to include few-shot examples

    Returns:
        Formatted prompt string
    """
    system_message = """You are an expert Python programmer. Write a Python function that solves the given task. Make sure your code passes all the provided test cases."""

    # Format test cases for the prompt
    test_str = "\n".join(test_list[:3])  # Show first 3 test cases

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
        import random
        rng = random.Random(42)
        data = rng.sample(data, max_samples)

    return data


def detect_benchmark(test_file: str) -> str:
    """Detect benchmark type from file path."""
    test_file_lower = test_file.lower()
    if "humaneval" in test_file_lower:
        return "humaneval"
    elif "mbpp" in test_file_lower:
        return "mbpp"
    return "humaneval"  # default


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate_baseline_code(config: BaselineCodeEvalConfig) -> Dict[str, Any]:
    """Evaluate Ouro-2.6B-Thinking on code generation test dataset using vLLM.

    Returns:
        Dictionary containing metrics and detailed results.
    """
    logger.info("=" * 60)
    logger.info("Ouro-2.6B-Thinking Baseline Evaluation (Code Generation)")
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

    # Prepare all prompts
    logger.info(f"Preparing prompts (few-shot: {config.eval.use_few_shot})...")
    prompts = []
    original_prompts = []  # Store original code prompts for completion extraction
    prompt_lengths = []

    for example in dataset:
        if benchmark == "humaneval":
            orig_prompt = example.get("prompt", "")
            entry_point = example.get("entry_point", "")
            formatted_prompt = format_humaneval_prompt(
                orig_prompt,
                entry_point,
                tokenizer,
                use_few_shot=config.eval.use_few_shot,
            )
            original_prompts.append(orig_prompt)
        else:  # mbpp
            orig_prompt = example.get("prompt", "")
            test_list = example.get("test_list", [])
            formatted_prompt = format_mbpp_prompt(
                orig_prompt,
                test_list,
                tokenizer,
                use_few_shot=config.eval.use_few_shot,
            )
            original_prompts.append(orig_prompt)

        prompts.append(formatted_prompt)
        prompt_lengths.append(len(tokenizer.encode(formatted_prompt)))

    # Report prompt length statistics
    avg_len = sum(prompt_lengths) / len(prompt_lengths)
    max_len = max(prompt_lengths)
    min_len = min(prompt_lengths)
    logger.info(f"Prompt token lengths - min: {min_len}, avg: {avg_len:.1f}, max: {max_len}")

    # Configure sampling parameters
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # For code generation, also stop at certain tokens
    stop_strings = ["```\n", "\n\n\n"]

    sampling_params = SamplingParams(
        max_tokens=config.eval.max_new_tokens,
        temperature=config.eval.temperature if config.eval.num_samples > 1 else 0.0,
        n=config.eval.num_samples,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        stop=stop_strings,
    )

    logger.info("Generation settings:")
    logger.info(f"  max_tokens: {config.eval.max_new_tokens}")
    logger.info(f"  temperature: {sampling_params.temperature}")
    logger.info(f"  num_samples: {config.eval.num_samples}")

    # Generate all responses
    logger.info(f"\nGenerating responses for {total} problems...")
    start_time = time.time()

    outputs = vllm_engine.generate(prompts, sampling_params)

    generation_time = time.time() - start_time
    logger.info(f"Generation complete in {generation_time:.2f}s ({total/generation_time:.2f} samples/sec)")

    # Process results and check correctness
    logger.info("\nExecuting code and checking correctness...")
    correct = 0
    all_outputs = []

    for i, (output, orig_prompt, example) in enumerate(
        tqdm(
            zip(outputs, original_prompts, dataset),
            total=total,
            desc="Evaluating"
        )
    ):
        task_id = example.get("task_id", example.get("id", i))
        sample_results = []

        for j, completion_output in enumerate(output.outputs):
            response = completion_output.text

            # Extract code from response
            if benchmark == "humaneval":
                entry_point = example.get("entry_point", "")
                # For HumanEval, we need to combine prompt + completion
                code = extract_code_from_response(response, entry_point)
                # If the extracted code doesn't have the function definition, prepend prompt
                if f"def {entry_point}" not in code:
                    code = orig_prompt + code

                # Get test code
                test_code = example.get("test", "")

                # Check correctness
                passed, error = check_correctness(
                    code, test_code, config.eval.execution_timeout
                )
            else:  # mbpp
                code = extract_code_from_response(response)
                test_list = example.get("test_list", [])
                test_imports = example.get("test_imports", [])

                passed, error, _ = check_mbpp_correctness(
                    code, test_list, test_imports, config.eval.execution_timeout
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
            correct += 1

        all_outputs.append({
            "index": i,
            "task_id": task_id,
            "original_prompt": orig_prompt,
            "num_samples": len(sample_results),
            "num_passed": sum(1 for s in sample_results if s["passed"]),
            "samples": sample_results,
        })

        # Log progress every 50 samples
        if (i + 1) % 50 == 0 or (i + 1) == total:
            acc = correct / (i + 1) * 100
            logger.info(f"Processed {i + 1}/{total} | Pass@1: {acc:.2f}% ({correct}/{i + 1})")

    # Calculate metrics
    pass_at_1 = correct / total if total > 0 else 0.0

    # Calculate pass@k for different k values
    k_values = [1]
    if config.eval.num_samples >= 5:
        k_values.append(5)
    if config.eval.num_samples >= 10:
        k_values.append(10)

    pass_at_k_results = {}
    for k in k_values:
        task_pass_rates = []
        for item in all_outputs:
            n = item["num_samples"]
            c = item["num_passed"]
            if n >= k:
                task_pass_rates.append(compute_pass_at_k(n, c, k))
        if task_pass_rates:
            pass_at_k_results[f"pass@{k}"] = sum(task_pass_rates) / len(task_pass_rates)

    metrics = {
        "pass@1": pass_at_1,
        **pass_at_k_results,
        "total_problems": total,
        "total_passed": correct,
    }

    # Build results
    results = {
        "metrics": metrics,
        "config": {
            "model_path": config.model.model_name_or_path,
            "test_file": config.data.test_file,
            "benchmark": benchmark,
            "use_few_shot": config.eval.use_few_shot,
            "max_prompt_length": config.eval.max_prompt_length,
            "max_new_tokens": config.eval.max_new_tokens,
            "temperature": config.eval.temperature,
            "num_samples": config.eval.num_samples,
            "execution_timeout": config.eval.execution_timeout,
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
    logger.info(f"Pass@1: {pass_at_1*100:.2f}% ({correct}/{total})")
    for k, v in pass_at_k_results.items():
        if k != "pass@1":
            logger.info(f"{k}: {v*100:.2f}%")
    logger.info("=" * 60)

    return results


def main():
    """Main function."""
    import argparse

    # Load default configuration
    config = BaselineCodeEvalConfig()

    # Parse command line overrides
    parser = argparse.ArgumentParser(description="Evaluate Ouro-2.6B-Thinking on code generation benchmarks")

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
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="Max prompt length (tokens)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem (for pass@k)")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Enable few-shot prompting (default: zero-shot)")
    parser.add_argument("--execution_timeout", type=float, default=5.0,
                        help="Timeout for code execution in seconds")

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
    config.eval.num_samples = args.num_samples
    config.eval.use_few_shot = args.use_few_shot
    config.eval.execution_timeout = args.execution_timeout
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
    logger.info(f"  Num samples: {config.eval.num_samples}")
    logger.info(f"  Execution timeout: {config.eval.execution_timeout}s")

    # Run evaluation
    results = evaluate_baseline_code(config)

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
        f.write(f"Num samples: {config.eval.num_samples}\n")
        f.write(f"Pass@1: {results['metrics']['pass@1']*100:.2f}%\n")
        for k, v in results['metrics'].items():
            if k.startswith('pass@') and k != 'pass@1':
                f.write(f"{k}: {v*100:.2f}%\n")
    logger.info(f"Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    main()
