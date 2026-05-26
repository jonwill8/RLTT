#!/usr/bin/env python3
"""
Code execution and correctness checking utilities for code generation benchmarks.

This module provides utilities for:
1. Extracting code from model responses
2. Safely executing code with test cases
3. Checking correctness (pass@k evaluation)

Supports both HumanEval and MBPP benchmarks.
"""

import re
import sys
import signal
import tempfile
import traceback
import multiprocessing
from io import StringIO
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager


# Default timeout for code execution (in seconds)
DEFAULT_TIMEOUT = 5.0


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Code execution timed out")


@contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time."""
    if sys.platform == "win32":
        # Windows doesn't support SIGALRM, so we just yield without timeout
        yield
    else:
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def extract_code_from_response(
    response: str,
    entry_point: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Extract code from a model response.

    Handles various formats:
    1. Code blocks with ```python ... ```
    2. Code blocks with ``` ... ```
    3. Raw code (when no markdown blocks found)
    4. Multiple code blocks (takes the most relevant one)

    Args:
        response: The model's response text
        entry_point: The function name to look for (helps identify correct block)
        prompt: The original prompt (used to strip repeated prompt from response)

    Returns:
        Extracted code string
    """
    if not response:
        return ""

    response = response.strip()

    # Remove the prompt from the beginning if it was repeated
    if prompt and response.startswith(prompt):
        response = response[len(prompt):].strip()

    # Pattern 1: Look for ```python ... ``` blocks
    python_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)

    # Pattern 2: Look for generic ``` ... ``` blocks if no python-specific ones
    if not python_blocks:
        python_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)

    if python_blocks:
        # If we have an entry_point, try to find the block containing it
        if entry_point:
            for block in python_blocks:
                if f"def {entry_point}" in block:
                    return block.strip()

        # Otherwise, return the longest block (usually the most complete)
        return max(python_blocks, key=len).strip()

    # Pattern 3: No markdown blocks - try to extract raw code
    # Look for function definitions
    lines = response.split('\n')
    code_lines = []
    in_function = False
    indent_level = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the beginning
        if not code_lines and not stripped:
            continue

        # Start of function
        if stripped.startswith('def '):
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            code_lines.append(line)
        elif in_function:
            # Check if we're still in the function
            current_indent = len(line) - len(line.lstrip()) if stripped else indent_level + 4
            if stripped and current_indent <= indent_level and not stripped.startswith(('#', '@', 'def ')):
                # End of function
                break
            code_lines.append(line)
        elif stripped.startswith('import ') or stripped.startswith('from '):
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    # Fallback: return the entire response (might be raw code)
    return response


def extract_code_completion(
    response: str,
    prompt: str,
    entry_point: Optional[str] = None,
) -> str:
    """
    Extract code completion from model response.

    For HumanEval-style tasks where the model completes a function body.

    Args:
        response: The model's response text
        prompt: The original prompt (function signature + docstring)
        entry_point: The function name being completed

    Returns:
        Complete code (prompt + completion)
    """
    # First, extract the code portion
    code = extract_code_from_response(response, entry_point, prompt)

    # If the extracted code already contains the function definition, return as-is
    if entry_point and f"def {entry_point}" in code:
        return code

    # Otherwise, prepend the prompt to get complete function
    return prompt + code


def check_correctness_simple(
    code: str,
    test_code: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """
    Check if code passes test cases (simple single-process version).

    Args:
        code: The code to test (should define the required function)
        test_code: Test code containing assertions
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (passed: bool, error_message: str)
    """
    full_code = f"{code}\n\n{test_code}"

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        with time_limit(timeout):
            exec_globals = {}
            exec(full_code, exec_globals)
        return True, ""
    except TimeoutException:
        return False, "Execution timed out"
    except AssertionError as e:
        return False, f"Assertion failed: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _execute_code_worker(code: str, test_code: str, result_queue: multiprocessing.Queue):
    """Worker function for multiprocess code execution."""
    try:
        full_code = f"{code}\n\n{test_code}"
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put((True, ""))
    except AssertionError as e:
        result_queue.put((False, f"Assertion failed: {str(e)}"))
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put((False, f"{type(e).__name__}: {str(e)}\n{tb}"))


def check_correctness(
    code: str,
    test_code: str,
    timeout: float = DEFAULT_TIMEOUT,
    use_multiprocessing: bool = True,
) -> Tuple[bool, str]:
    """
    Check if code passes test cases.

    Uses multiprocessing for safer execution (prevents infinite loops from
    blocking the main process).

    Args:
        code: The code to test (should define the required function)
        test_code: Test code containing assertions
        timeout: Maximum execution time in seconds
        use_multiprocessing: Whether to use multiprocessing (safer but slower)

    Returns:
        Tuple of (passed: bool, error_message: str)
    """
    if not use_multiprocessing:
        return check_correctness_simple(code, test_code, timeout)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_code_worker,
        args=(code, test_code, result_queue)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(1)
        if process.is_alive():
            process.kill()
        return False, "Execution timed out"

    if result_queue.empty():
        return False, "No result from execution (process may have crashed)"

    return result_queue.get()


def check_humaneval_correctness(
    completion: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """
    Check correctness for HumanEval-style problems.

    Args:
        completion: The model's completion (code after the prompt)
        prompt: The function signature and docstring
        test: The test code (contains check() function call)
        entry_point: The function name
        timeout: Execution timeout

    Returns:
        Tuple of (passed: bool, error_message: str)
    """
    # Construct full code
    full_code = prompt + completion

    # HumanEval tests typically call check(entry_point)
    # Make sure the test code can find the function
    return check_correctness(full_code, test, timeout)


def check_mbpp_correctness(
    code: str,
    test_list: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, str, List[bool]]:
    """
    Check correctness for MBPP-style problems.

    Args:
        code: The complete code (function definition)
        test_list: List of assertion strings
        test_imports: List of import statements needed for tests
        timeout: Execution timeout

    Returns:
        Tuple of (all_passed: bool, error_message: str, individual_results: List[bool])
    """
    # Build test code
    test_code = ""
    if test_imports:
        test_code = "\n".join(test_imports) + "\n\n"

    # Run each test individually to get per-test results
    individual_results = []
    first_error = ""

    for test in test_list:
        single_test_code = test_code + test
        passed, error = check_correctness(code, single_test_code, timeout)
        individual_results.append(passed)
        if not passed and not first_error:
            first_error = error

    all_passed = all(individual_results)
    return all_passed, first_error if not all_passed else "", individual_results


def compute_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int
) -> float:
    """
    Compute pass@k metric.

    pass@k estimates the probability that at least one of k samples passes.

    Uses the unbiased estimator:
    pass@k = 1 - C(n-c, k) / C(n, k)

    where n = num_samples, c = num_correct

    Args:
        num_samples: Total number of samples generated (n)
        num_correct: Number of samples that passed (c)
        k: Number of samples to consider

    Returns:
        pass@k probability
    """
    if num_samples < k:
        return float(num_correct > 0)

    if num_correct >= num_samples:
        return 1.0

    if num_correct == 0:
        return 0.0

    # Use log to avoid overflow
    import math

    # C(n-c, k) / C(n, k) = product((n-c-i) / (n-i) for i in range(k))
    log_prob_fail = 0.0
    for i in range(k):
        log_prob_fail += math.log(num_samples - num_correct - i) - math.log(num_samples - i)

    return 1.0 - math.exp(log_prob_fail)


def evaluate_code_samples(
    samples: List[Dict[str, Any]],
    k_values: List[int] = [1, 5, 10],
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Evaluate a list of code samples and compute pass@k.

    Args:
        samples: List of dicts with keys:
            - task_id: Problem identifier
            - completion: Model's code completion
            - prompt: Original prompt (for HumanEval)
            - test: Test code
            - entry_point: Function name (for HumanEval)
            - test_list: List of tests (for MBPP)
            - test_imports: Import statements (for MBPP)
            - source: 'humaneval' or 'mbpp'
        k_values: List of k values for pass@k computation
        timeout: Execution timeout per sample

    Returns:
        Dict with pass@k results and per-sample details
    """
    # Group samples by task_id
    tasks = {}
    for sample in samples:
        task_id = sample.get("task_id", sample.get("id"))
        if task_id not in tasks:
            tasks[task_id] = []
        tasks[task_id].append(sample)

    # Evaluate each task
    task_results = {}
    for task_id, task_samples in tasks.items():
        correct_count = 0
        sample_results = []

        for sample in task_samples:
            source = sample.get("source", "humaneval")

            if source == "humaneval":
                # HumanEval evaluation
                completion = sample.get("completion", "")
                prompt = sample.get("prompt", "")
                test = sample.get("test", "")
                entry_point = sample.get("entry_point", "")

                # Extract code if needed
                if "def " not in completion and prompt:
                    code = extract_code_completion(completion, prompt, entry_point)
                else:
                    code = completion

                passed, error = check_correctness(code, test, timeout)

            else:  # MBPP
                code = sample.get("completion", sample.get("code", ""))
                test_list = sample.get("test_list", [])
                test_imports = sample.get("test_imports", [])

                # Extract code if wrapped in markdown
                code = extract_code_from_response(code)

                passed, error, _ = check_mbpp_correctness(
                    code, test_list, test_imports, timeout
                )

            if passed:
                correct_count += 1

            sample_results.append({
                "passed": passed,
                "error": error,
            })

        task_results[task_id] = {
            "num_samples": len(task_samples),
            "num_correct": correct_count,
            "sample_results": sample_results,
        }

    # Compute pass@k for each k value
    pass_at_k = {}
    for k in k_values:
        task_pass_rates = []
        for task_id, result in task_results.items():
            n = result["num_samples"]
            c = result["num_correct"]
            if n >= k:
                task_pass_rates.append(compute_pass_at_k(n, c, k))

        if task_pass_rates:
            pass_at_k[f"pass@{k}"] = sum(task_pass_rates) / len(task_pass_rates)
        else:
            pass_at_k[f"pass@{k}"] = 0.0

    return {
        "pass_at_k": pass_at_k,
        "task_results": task_results,
        "total_tasks": len(tasks),
        "total_samples": len(samples),
    }
