#!/usr/bin/env python3
"""
Prepare MCQA datasets (ARC-Challenge, MMLU-STEM, GPQA) for evaluation.

This script loads the raw datasets from Hugging Face cache and creates
JSONL files in a standardized format for evaluation.

Usage:
    python prepare_mcqa_datasets.py --dataset arc_c --output_dir ./data
    python prepare_mcqa_datasets.py --dataset mmlu_stem --output_dir ./data
    python prepare_mcqa_datasets.py --dataset gpqa --output_dir ./data
    python prepare_mcqa_datasets.py --dataset all --output_dir ./data
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

# MMLU STEM subjects
MMLU_STEM_SUBJECTS = [
    # Math
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    # Physics
    "astronomy",
    "college_physics",
    "conceptual_physics",
    "high_school_physics",
    # Chemistry
    "college_chemistry",
    "high_school_chemistry",
    # Biology
    "college_biology",
    "high_school_biology",
    # Computer Science
    "college_computer_science",
    "computer_security",
    "high_school_computer_science",
    "machine_learning",
    # Engineering
    "electrical_engineering",
]


def prepare_arc_challenge(cache_dir: str, output_dir: str) -> str:
    """
    Prepare ARC-Challenge dataset.

    Args:
        cache_dir: HuggingFace cache directory
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading ARC-Challenge dataset...")
    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Use test split
    test_data = dataset["test"]

    output_data = []
    for idx, item in enumerate(test_data):
        # ARC has variable number of choices (3-5)
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]

        # Find correct answer index
        correct_label = item["answerKey"]

        output_item = {
            "id": item.get("id", f"arc_c_{idx}"),
            "question": item["question"],
            "choices": choices,
            "choice_labels": labels,
            "answer": correct_label,
            "source": "arc_challenge",
        }
        output_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "arc_challenge.test.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} ARC-Challenge examples to {output_file}")
    return output_file


def prepare_mmlu_stem(cache_dir: str, output_dir: str) -> str:
    """
    Prepare MMLU-STEM dataset (subset of MMLU with STEM subjects only).

    Args:
        cache_dir: HuggingFace cache directory
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading MMLU dataset (STEM subjects)...")

    output_data = []

    for subject in MMLU_STEM_SUBJECTS:
        print(f"  Loading subject: {subject}")
        try:
            dataset = load_dataset(
                "cais/mmlu",
                subject,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            # Use test split
            test_data = dataset["test"]

            for idx, item in enumerate(test_data):
                # MMLU has exactly 4 choices (A, B, C, D)
                choices = item["choices"]

                # Answer is stored as integer (0=A, 1=B, 2=C, 3=D)
                correct_idx = item["answer"]
                correct_label = chr(ord('A') + correct_idx)

                output_item = {
                    "id": f"mmlu_{subject}_{idx}",
                    "question": item["question"],
                    "choices": choices,
                    "choice_labels": ["A", "B", "C", "D"],
                    "answer": correct_label,
                    "subject": subject,
                    "source": "mmlu_stem",
                }
                output_data.append(output_item)

        except Exception as e:
            print(f"    WARNING: Failed to load {subject}: {e}")
            continue

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mmlu_stem.test.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} MMLU-STEM examples to {output_file}")
    return output_file


def prepare_gpqa(cache_dir: str, output_dir: str) -> str:
    """
    Prepare GPQA dataset.

    Args:
        cache_dir: HuggingFace cache directory
        output_dir: Output directory for JSONL file

    Returns:
        Path to output file
    """
    print("Loading GPQA dataset...")

    # Try loading GPQA - it may require authentication
    try:
        dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",  # Use the diamond (hardest) subset
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load gpqa_diamond, trying gpqa_main: {e}")
        try:
            dataset = load_dataset(
                "Idavidrein/gpqa",
                "gpqa_main",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e2:
            print(f"Failed to load gpqa_main: {e2}")
            # Try without config name
            dataset = load_dataset(
                "Idavidrein/gpqa",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

    # Use train split (GPQA may not have a separate test split)
    if "test" in dataset:
        data = dataset["test"]
    elif "train" in dataset:
        data = dataset["train"]
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")

    output_data = []
    for idx, item in enumerate(data):
        # GPQA format may vary - handle different column names
        question = item.get("Question", item.get("question", ""))

        # Try to get choices
        if "Correct Answer" in item:
            # GPQA diamond format
            correct = item["Correct Answer"]
            choices = [
                item.get("Correct Answer", ""),
                item.get("Incorrect Answer 1", ""),
                item.get("Incorrect Answer 2", ""),
                item.get("Incorrect Answer 3", ""),
            ]
            # Remove empty choices
            choices = [c for c in choices if c]

            # Shuffle choices and track correct index
            import random
            random.seed(idx)  # Deterministic shuffle
            indices = list(range(len(choices)))
            random.shuffle(indices)
            shuffled_choices = [choices[i] for i in indices]
            correct_idx = indices.index(0)  # Original correct was at index 0
            correct_label = chr(ord('A') + correct_idx)
        else:
            # Alternative format with choices list
            choices = item.get("choices", [])
            correct_label = item.get("answer", item.get("answerKey", "A"))
            shuffled_choices = choices

        choice_labels = [chr(ord('A') + i) for i in range(len(shuffled_choices))]

        output_item = {
            "id": f"gpqa_{idx}",
            "question": question,
            "choices": shuffled_choices,
            "choice_labels": choice_labels,
            "answer": correct_label,
            "subject": item.get("Subdomain", item.get("subject", "unknown")),
            "source": "gpqa",
        }
        output_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gpqa.test.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} GPQA examples to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Prepare MCQA datasets for evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["arc_c", "mmlu_stem", "gpqa", "all"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets",
        help="HuggingFace cache directory",
    )

    args = parser.parse_args()

    if args.dataset == "arc_c" or args.dataset == "all":
        prepare_arc_challenge(args.cache_dir, args.output_dir)

    if args.dataset == "mmlu_stem" or args.dataset == "all":
        prepare_mmlu_stem(args.cache_dir, args.output_dir)

    if args.dataset == "gpqa" or args.dataset == "all":
        prepare_gpqa(args.cache_dir, args.output_dir)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
