#!/usr/bin/env python3
"""
Data Preparation Script for SFT Training with verl.

Converts MATH dataset from JSONL format to Parquet format required by verl's SFTDataset.
The Parquet files contain 'prompt' and 'response' columns formatted for chat-style training.

Usage:
    python prepare_data.py
    python prepare_data.py --train_input /path/to/math_train.jsonl --train_output /path/to/output.parquet
"""

import os
import json
import argparse
from typing import Dict, Any

import pandas as pd


# Instruction used in prompts (same as training)
INSTRUCTION = (
    "Solve the following math problem. Show your reasoning step by step.\n"
    "Put your final answer in \\boxed{}. Once you provide the final answer, stop immediately."
)


def format_prompt(problem: str) -> str:
    """Format a math problem into a prompt string.

    Note: The actual chat template is applied by the tokenizer during training.
    Here we just format the raw prompt content.
    """
    return f"{INSTRUCTION}\n\nProblem: {problem}"


def load_jsonl(filepath: str) -> list:
    """Load data from JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def convert_math_train_to_parquet(
    input_path: str,
    output_path: str,
    verbose: bool = True
) -> None:
    """Convert MATH training data from JSONL to Parquet format.

    Input format (JSONL):
        {"problem": "...", "solution": "...", "level": "...", "type": "..."}

    Output format (Parquet):
        prompt: The formatted problem prompt
        response: The solution text
    """
    if verbose:
        print(f"Loading training data from: {input_path}")

    data = load_jsonl(input_path)

    if verbose:
        print(f"  Loaded {len(data)} examples")

    # Convert to prompt/response format
    records = []
    for item in data:
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        records.append({
            "prompt": format_prompt(problem),
            "response": solution,
        })

    # Create DataFrame and save to Parquet
    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_path, index=False)

    if verbose:
        print(f"  Saved to: {output_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")


def convert_math500_to_parquet(
    input_path: str,
    output_path: str,
    verbose: bool = True
) -> None:
    """Convert MATH-500 test data from JSONL to Parquet format.

    Input format (JSONL):
        {"problem": "...", "answer": "...", "level": "...", "type": "..."}

    Output format (Parquet):
        prompt: The formatted problem prompt
        response: The answer (not full solution, just for validation loss)
    """
    if verbose:
        print(f"Loading MATH-500 data from: {input_path}")

    data = load_jsonl(input_path)

    if verbose:
        print(f"  Loaded {len(data)} examples")

    # Convert to prompt/response format
    records = []
    for item in data:
        problem = item.get("problem", "")
        # MATH-500 has 'answer' field, not full solution
        answer = item.get("answer", "")
        # Create a minimal response with boxed answer for validation loss computation
        response = f"The answer is $\\boxed{{{answer}}}$."

        records.append({
            "prompt": format_prompt(problem),
            "response": response,
        })

    # Create DataFrame and save to Parquet
    df = pd.DataFrame(records)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_path, index=False)

    if verbose:
        print(f"  Saved to: {output_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert MATH dataset to Parquet format for verl SFT training")

    # Training data paths
    parser.add_argument(
        "--train_input",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl",
        help="Path to MATH training data (JSONL format)"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.parquet",
        help="Output path for training data (Parquet format)"
    )

    # Validation data paths
    parser.add_argument(
        "--val_input",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
        help="Path to MATH-500 validation data (JSONL format)"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/math500_test.parquet",
        help="Output path for validation data (Parquet format)"
    )

    # Options
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training data conversion"
    )
    parser.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip validation data conversion"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MATH Dataset Preparation for verl SFT Training")
    print("=" * 60)
    print()

    # Convert training data
    if not args.skip_train:
        if os.path.exists(args.train_input):
            print("Converting training data...")
            convert_math_train_to_parquet(args.train_input, args.train_output)
            print()
        else:
            print(f"WARNING: Training input not found: {args.train_input}")
            print("  Skipping training data conversion.")
            print()

    # Convert validation data
    if not args.skip_val:
        if os.path.exists(args.val_input):
            print("Converting validation data...")
            convert_math500_to_parquet(args.val_input, args.val_output)
            print()
        else:
            print(f"WARNING: Validation input not found: {args.val_input}")
            print("  Skipping validation data conversion.")
            print()

    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Update sft_config.yaml with the correct parquet file paths")
    print("  2. Run training with: sbatch run_sft.slurm")
    print()


if __name__ == "__main__":
    main()
