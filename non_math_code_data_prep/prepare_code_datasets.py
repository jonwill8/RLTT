#!/usr/bin/env python3
"""
Prepare code generation datasets (HumanEval, MBPP) for evaluation.

This script loads the raw datasets from local files and creates
JSONL files in a standardized format for evaluation.

Dataset structures:
- HumanEval: Parquet file with task_id, prompt, canonical_solution, test, entry_point
- MBPP: JSONL with task_id, text (prompt), code (solution), test_list

Usage:
    python prepare_code_datasets.py --dataset humaneval --output_dir ./data
    python prepare_code_datasets.py --dataset mbpp --output_dir ./data
    python prepare_code_datasets.py --dataset all --output_dir ./data
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_humaneval_parquet(data_path: str) -> List[Dict]:
    """Load HumanEval from parquet file."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "parquet",
            data_files=data_path,
            split="train",
        )
        return list(dataset)
    except Exception as e:
        print(f"Failed to load with datasets library: {e}")
        # Fallback: try pyarrow directly
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(data_path)
            data = table.to_pydict()
            # Convert columnar to row format
            rows = []
            num_rows = len(data[list(data.keys())[0]])
            for i in range(num_rows):
                row = {k: data[k][i] for k in data.keys()}
                rows.append(row)
            return rows
        except Exception as e2:
            print(f"Failed to load with pyarrow: {e2}")
            raise


def prepare_humaneval(data_dir: str, output_dir: str) -> str:
    """
    Prepare HumanEval dataset.

    HumanEval format:
    - task_id: e.g., "HumanEval/0"
    - prompt: Function signature and docstring
    - canonical_solution: Reference solution
    - test: Test code
    - entry_point: Function name to call

    Args:
        data_dir: Directory containing HumanEval parquet file
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading HumanEval dataset...")

    # Find the parquet file
    parquet_path = os.path.join(data_dir, "openai_humaneval", "openai_humaneval", "test-00000-of-00001.parquet")
    if not os.path.exists(parquet_path):
        # Try alternative path
        parquet_path = os.path.join(data_dir, "openai_humaneval", "test-00000-of-00001.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"HumanEval parquet file not found. Searched: {parquet_path}")

    print(f"Loading from: {parquet_path}")
    data = load_humaneval_parquet(parquet_path)

    output_data = []
    for idx, item in enumerate(data):
        # Extract fields from HumanEval format
        task_id = item.get("task_id", f"HumanEval/{idx}")
        prompt = item.get("prompt", "")
        canonical_solution = item.get("canonical_solution", "")
        test = item.get("test", "")
        entry_point = item.get("entry_point", "")

        output_item = {
            "id": task_id,
            "task_id": task_id,
            "prompt": prompt,
            "canonical_solution": canonical_solution,
            "test": test,
            "entry_point": entry_point,
            "source": "humaneval",
        }
        output_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "humaneval.test.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} HumanEval examples to {output_file}")
    return output_file


def prepare_mbpp(data_dir: str, output_dir: str, use_sanitized: bool = True) -> str:
    """
    Prepare MBPP dataset.

    MBPP format (sanitized):
    - task_id: Integer ID
    - prompt: Text description of the task
    - code: Reference solution
    - test_list: List of assertion test strings
    - test_imports: List of required imports

    Args:
        data_dir: Directory containing MBPP files
        output_dir: Output directory for JSONL file
        use_sanitized: Whether to use sanitized version (default: True)

    Returns:
        Path to output file
    """
    print("Loading MBPP dataset...")

    # Choose file based on sanitized flag
    if use_sanitized:
        mbpp_path = os.path.join(data_dir, "mbpp", "data", "sanitized-mbpp.json")
    else:
        mbpp_path = os.path.join(data_dir, "mbpp", "data", "mbpp.jsonl")

    if not os.path.exists(mbpp_path):
        raise FileNotFoundError(f"MBPP file not found: {mbpp_path}")

    print(f"Loading from: {mbpp_path}")

    # Load data
    if mbpp_path.endswith(".json"):
        with open(mbpp_path, "r") as f:
            data = json.load(f)
    else:
        data = []
        with open(mbpp_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    output_data = []
    for item in data:
        task_id = item.get("task_id", 0)

        # Handle different field names
        prompt = item.get("prompt", item.get("text", ""))
        code = item.get("code", "")
        test_list = item.get("test_list", [])
        test_imports = item.get("test_imports", [])

        # Create test code from test_list
        test_code = "\n".join(test_imports) + "\n" if test_imports else ""
        test_code += "\n".join(test_list)

        output_item = {
            "id": f"mbpp_{task_id}",
            "task_id": task_id,
            "prompt": prompt,
            "canonical_solution": code,
            "test": test_code,
            "test_list": test_list,
            "test_imports": test_imports,
            "source": "mbpp",
        }
        output_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mbpp.test.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} MBPP examples to {output_file}")
    return output_file


def prepare_mbpp_train_test_split(data_dir: str, output_dir: str, use_sanitized: bool = True) -> tuple:
    """
    Prepare MBPP with proper train/test split.

    MBPP official splits:
    - Train: task_id 1-10 (few-shot examples) + 11-510 (training)
    - Test: task_id 11-510 (but we use 11-510 for eval typically)

    For sanitized MBPP:
    - We use all examples as test since sanitized is specifically for evaluation

    Args:
        data_dir: Directory containing MBPP files
        output_dir: Output directory for JSONL files
        use_sanitized: Whether to use sanitized version

    Returns:
        Tuple of (train_file, test_file) paths
    """
    print("Loading MBPP dataset for train/test split...")

    # Load sanitized MBPP
    mbpp_path = os.path.join(data_dir, "mbpp", "data", "sanitized-mbpp.json")
    if not os.path.exists(mbpp_path):
        mbpp_path = os.path.join(data_dir, "mbpp", "data", "mbpp.jsonl")

    print(f"Loading from: {mbpp_path}")

    if mbpp_path.endswith(".json"):
        with open(mbpp_path, "r") as f:
            data = json.load(f)
    else:
        data = []
        with open(mbpp_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    # Split based on task_id
    # Few-shot examples: task_id 1-10
    # Training: task_id 11-510
    # Test: task_id 511-974 (or all of sanitized for evaluation)
    train_data = []
    test_data = []

    for item in data:
        task_id = item.get("task_id", 0)
        prompt = item.get("prompt", item.get("text", ""))
        code = item.get("code", "")
        test_list = item.get("test_list", [])
        test_imports = item.get("test_imports", [])

        test_code = "\n".join(test_imports) + "\n" if test_imports else ""
        test_code += "\n".join(test_list)

        output_item = {
            "id": f"mbpp_{task_id}",
            "task_id": task_id,
            "prompt": prompt,
            "canonical_solution": code,
            "test": test_code,
            "test_list": test_list,
            "test_imports": test_imports,
            "source": "mbpp",
        }

        # For sanitized MBPP, use all as test
        # The few-shot examples (1-10) could be used separately
        if task_id <= 10:
            train_data.append(output_item)
        test_data.append(output_item)

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "mbpp.train.jsonl")
    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(train_data)} MBPP train examples to {train_file}")

    test_file = os.path.join(output_dir, "mbpp.test.jsonl")
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(test_data)} MBPP test examples to {test_file}")

    return train_file, test_file


def main():
    parser = argparse.ArgumentParser(description="Prepare code generation datasets for evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["humaneval", "mbpp", "all"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/code",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets",
        help="Directory containing raw dataset files",
    )
    parser.add_argument(
        "--use_sanitized",
        action="store_true",
        default=True,
        help="Use sanitized MBPP (default: True)",
    )

    args = parser.parse_args()

    if args.dataset == "humaneval" or args.dataset == "all":
        prepare_humaneval(args.data_dir, args.output_dir)

    if args.dataset == "mbpp" or args.dataset == "all":
        prepare_mbpp(args.data_dir, args.output_dir, args.use_sanitized)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
