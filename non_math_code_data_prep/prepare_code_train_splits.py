#!/usr/bin/env python3
"""
Prepare code generation train splits for few-shot examples.

This script loads train/validation splits from the datasets and creates
JSONL files for use as few-shot examples in evaluation.

Note:
- HumanEval only has a test split, so we don't create a train split for it
- MBPP has designated few-shot examples (task_id 1-10)

Usage:
    python prepare_code_train_splits.py --output_dir ./data
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_mbpp_fewshot(data_dir: str, output_dir: str) -> str:
    """
    Prepare MBPP few-shot examples.

    MBPP designates task_id 1-10 as few-shot examples.

    Args:
        data_dir: Directory containing MBPP files
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading MBPP for few-shot examples...")

    # Try sanitized first, then regular
    mbpp_path = os.path.join(data_dir, "mbpp", "data", "sanitized-mbpp.json")
    if not os.path.exists(mbpp_path):
        mbpp_path = os.path.join(data_dir, "mbpp", "data", "mbpp.jsonl")

    if not os.path.exists(mbpp_path):
        print(f"WARNING: MBPP file not found at {mbpp_path}")
        return None

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

    # Filter for few-shot examples (task_id 1-10)
    fewshot_data = []
    for item in data:
        task_id = item.get("task_id", 0)
        if task_id <= 10:
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
            fewshot_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mbpp.fewshot.jsonl")

    with open(output_file, "w") as f:
        for item in fewshot_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(fewshot_data)} MBPP few-shot examples to {output_file}")
    return output_file


def prepare_mbpp_train(data_dir: str, output_dir: str) -> str:
    """
    Prepare MBPP training split.

    MBPP splits:
    - Few-shot: task_id 1-10
    - Training: task_id 11-510
    - Test: task_id 511-974 (but sanitized uses different splits)

    For sanitized MBPP, we exclude the few-shot examples (1-10) for training.

    Args:
        data_dir: Directory containing MBPP files
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading MBPP for training split...")

    # Load the full MBPP (not sanitized) for training
    mbpp_path = os.path.join(data_dir, "mbpp", "data", "mbpp.jsonl")
    if not os.path.exists(mbpp_path):
        print(f"WARNING: MBPP file not found at {mbpp_path}")
        return None

    print(f"Loading from: {mbpp_path}")

    data = []
    with open(mbpp_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Filter for training examples (task_id 11-510)
    train_data = []
    for item in data:
        task_id = item.get("task_id", 0)
        # Training split: exclude few-shot (1-10) and keep 11-510
        if 11 <= task_id <= 510:
            prompt = item.get("prompt", item.get("text", ""))
            code = item.get("code", "")
            test_list = item.get("test_list", [])
            test_imports = item.get("test_imports", item.get("test_setup_code", ""))

            # Handle test_imports being a string
            if isinstance(test_imports, str) and test_imports:
                test_imports = [test_imports]
            elif not isinstance(test_imports, list):
                test_imports = []

            test_code = "\n".join(test_imports) + "\n" if test_imports else ""
            test_code += "\n".join(test_list) if isinstance(test_list, list) else test_list

            output_item = {
                "id": f"mbpp_{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": code,
                "test": test_code,
                "test_list": test_list if isinstance(test_list, list) else [test_list],
                "test_imports": test_imports,
                "source": "mbpp",
            }
            train_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mbpp.train.jsonl")

    with open(output_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(train_data)} MBPP training examples to {output_file}")
    return output_file


def select_few_shot_examples(
    data_file: str,
    num_examples: int = 3,
    seed: int = 42
) -> List[Dict]:
    """
    Select diverse few-shot examples from a data file.

    Args:
        data_file: Path to JSONL file
        num_examples: Number of examples to select
        seed: Random seed for selection

    Returns:
        List of selected examples
    """
    with open(data_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)

    if len(data) <= num_examples:
        return data

    # Select diverse examples (prefer variety in problem types)
    return random.sample(data, num_examples)


def create_fewshot_json(output_dir: str, mbpp_fewshot_file: str) -> str:
    """
    Create a JSON file with organized few-shot examples for easy loading.

    Args:
        output_dir: Output directory
        mbpp_fewshot_file: Path to MBPP few-shot JSONL file

    Returns:
        Path to output JSON file
    """
    fewshot_examples = {
        "humaneval": [],  # HumanEval doesn't have official few-shot examples
        "mbpp": [],
    }

    # Load MBPP few-shot examples
    if mbpp_fewshot_file and os.path.exists(mbpp_fewshot_file):
        examples = select_few_shot_examples(mbpp_fewshot_file, num_examples=3)
        for ex in examples:
            fewshot_examples["mbpp"].append({
                "task_id": ex.get("task_id"),
                "prompt": ex.get("prompt"),
                "code": ex.get("canonical_solution"),
                "test_list": ex.get("test_list", []),
            })

    output_file = os.path.join(output_dir, "code_fewshot_examples.json")
    with open(output_file, "w") as f:
        json.dump(fewshot_examples, f, indent=2)

    print(f"Saved few-shot examples to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prepare code generation train splits")
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

    args = parser.parse_args()

    # Prepare MBPP splits
    mbpp_fewshot_file = prepare_mbpp_fewshot(args.data_dir, args.output_dir)
    prepare_mbpp_train(args.data_dir, args.output_dir)

    # Create consolidated few-shot JSON
    if mbpp_fewshot_file:
        create_fewshot_json(args.output_dir, mbpp_fewshot_file)

    print("\nTrain split preparation complete!")

    # Print sample few-shot examples
    if mbpp_fewshot_file and os.path.exists(mbpp_fewshot_file):
        print("\n" + "=" * 60)
        print("Sample few-shot examples (MBPP):")
        print("=" * 60)
        examples = select_few_shot_examples(mbpp_fewshot_file, num_examples=3)
        for i, ex in enumerate(examples):
            print(f"\n{i+1}. Task {ex['task_id']}: {ex['prompt'][:80]}...")


if __name__ == "__main__":
    main()
