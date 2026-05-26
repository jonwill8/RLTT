#!/usr/bin/env python3
"""
Prepare MCQA train splits for few-shot examples.

This script loads the train splits from Hugging Face cache and creates
JSONL files for use as few-shot examples in evaluation.

Usage:
    python prepare_mcqa_train_splits.py --output_dir ./data
"""

import os
import sys
import json
import argparse
import random
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


def prepare_arc_challenge_train(cache_dir: str, output_dir: str) -> str:
    """Prepare ARC-Challenge train split."""
    print("Loading ARC-Challenge train split...")
    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    train_data = dataset["train"]

    output_data = []
    for idx, item in enumerate(train_data):
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        correct_label = item["answerKey"]

        output_item = {
            "id": item.get("id", f"arc_c_train_{idx}"),
            "question": item["question"],
            "choices": choices,
            "choice_labels": labels,
            "answer": correct_label,
            "source": "arc_challenge",
        }
        output_data.append(output_item)

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "arc_challenge.train.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} ARC-Challenge train examples to {output_file}")
    return output_file


def prepare_mmlu_stem_train(cache_dir: str, output_dir: str) -> str:
    """Prepare MMLU-STEM train split (dev split in MMLU)."""
    print("Loading MMLU dataset (STEM subjects) - dev split...")

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

            # MMLU uses "dev" as the few-shot split, "validation" is also available
            if "dev" in dataset:
                data = dataset["dev"]
            elif "validation" in dataset:
                data = dataset["validation"]
            else:
                print(f"    No dev/validation split for {subject}, skipping")
                continue

            for idx, item in enumerate(data):
                choices = item["choices"]
                correct_idx = item["answer"]
                correct_label = chr(ord('A') + correct_idx)

                output_item = {
                    "id": f"mmlu_{subject}_train_{idx}",
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
    output_file = os.path.join(output_dir, "mmlu_stem.train.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} MMLU-STEM train examples to {output_file}")
    return output_file


def prepare_gpqa_train(cache_dir: str, output_dir: str) -> str:
    """Prepare GPQA train split."""
    print("Loading GPQA dataset...")

    try:
        dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",
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
            dataset = load_dataset(
                "Idavidrein/gpqa",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

    # GPQA only has train split - we'll use a portion for few-shot
    if "train" in dataset:
        data = dataset["train"]
    else:
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")

    output_data = []
    for idx, item in enumerate(data):
        question = item.get("Question", item.get("question", ""))

        if "Correct Answer" in item:
            correct = item["Correct Answer"]
            choices = [
                item.get("Correct Answer", ""),
                item.get("Incorrect Answer 1", ""),
                item.get("Incorrect Answer 2", ""),
                item.get("Incorrect Answer 3", ""),
            ]
            choices = [c for c in choices if c]

            # Shuffle choices and track correct index
            random.seed(idx + 1000)  # Different seed from test set
            indices = list(range(len(choices)))
            random.shuffle(indices)
            shuffled_choices = [choices[i] for i in indices]
            correct_idx = indices.index(0)
            correct_label = chr(ord('A') + correct_idx)
        else:
            choices = item.get("choices", [])
            correct_label = item.get("answer", item.get("answerKey", "A"))
            shuffled_choices = choices

        choice_labels = [chr(ord('A') + i) for i in range(len(shuffled_choices))]

        output_item = {
            "id": f"gpqa_train_{idx}",
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
    output_file = os.path.join(output_dir, "gpqa.train.jsonl")

    with open(output_file, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(output_data)} GPQA train examples to {output_file}")
    return output_file


def select_few_shot_examples(train_file: str, num_examples: int = 10, seed: int = 42) -> List[Dict]:
    """Select diverse few-shot examples from train file."""
    with open(train_file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)

    # Try to get diverse subjects if available
    subjects = {}
    for item in data:
        subject = item.get("subject", "default")
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(item)

    selected = []
    if len(subjects) > 1:
        # Select from different subjects
        subject_list = list(subjects.keys())
        random.shuffle(subject_list)

        idx = 0
        while len(selected) < num_examples and idx < len(subject_list) * 10:
            subject = subject_list[idx % len(subject_list)]
            if subjects[subject]:
                item = random.choice(subjects[subject])
                subjects[subject].remove(item)
                selected.append(item)
            idx += 1
    else:
        # Random selection
        selected = random.sample(data, min(num_examples, len(data)))

    return selected[:num_examples]


def main():
    parser = argparse.ArgumentParser(description="Prepare MCQA train splits")
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

    prepare_arc_challenge_train(args.cache_dir, args.output_dir)
    prepare_mmlu_stem_train(args.cache_dir, args.output_dir)
    prepare_gpqa_train(args.cache_dir, args.output_dir)

    print("\nTrain split preparation complete!")

    # Print sample few-shot selections
    print("\n" + "=" * 60)
    print("Sample few-shot examples (10 per dataset):")
    print("=" * 60)

    for dataset_name in ["arc_challenge", "mmlu_stem", "gpqa"]:
        train_file = os.path.join(args.output_dir, f"{dataset_name}.train.jsonl")
        if os.path.exists(train_file):
            examples = select_few_shot_examples(train_file, num_examples=10)
            print(f"\n{dataset_name.upper()} ({len(examples)} examples):")
            for i, ex in enumerate(examples):
                print(f"  {i+1}. {ex['question'][:80]}...")


if __name__ == "__main__":
    main()
