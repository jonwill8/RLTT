#!/usr/bin/env python3
"""
Extract few-shot examples from cached Arrow files.

This script reads the train/dev splits directly from the Arrow files
in the datasets cache and selects 10 examples for few-shot prompting.
"""

import os
import sys
import json
import random

# Try to import pyarrow or datasets
try:
    import pyarrow as pa
    USE_PYARROW = True
except ImportError:
    USE_PYARROW = False
    try:
        from datasets import Dataset
    except ImportError:
        print("Error: Neither pyarrow nor datasets is available")
        sys.exit(1)


def load_arrow_file(path):
    """Load data from an Arrow file."""
    if USE_PYARROW:
        # Try IPC stream format first (HuggingFace datasets format)
        try:
            reader = pa.ipc.open_stream(path)
            table = reader.read_all()
            return table.to_pydict()
        except pa.ArrowInvalid:
            pass
        # Try IPC file format
        try:
            reader = pa.ipc.open_file(path)
            table = reader.read_all()
            return table.to_pydict()
        except pa.ArrowInvalid:
            pass
        # Try memory-mapped file
        mmap = pa.memory_map(path)
        table = pa.ipc.open_stream(mmap).read_all()
        return table.to_pydict()
    else:
        ds = Dataset.from_file(path)
        return {col: ds[col] for col in ds.column_names}


def main():
    random.seed(42)  # For reproducibility across all scripts

    # Paths to Arrow files
    arc_train_path = "/scratch/gpfs/OLGARUS/jw4199/datasets/allenai___ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453/ai2_arc-train.arrow"
    gpqa_train_path = "/scratch/gpfs/OLGARUS/jw4199/datasets/Idavidrein___gpqa/gpqa_diamond/0.0.0/90b8e5be2b1d3d2dbfe016cdab47981150600c4a/gpqa-train.arrow"

    # MMLU dev files (one per subject)
    mmlu_base = "/scratch/gpfs/OLGARUS/jw4199/datasets/cais___mmlu"
    mmlu_subjects = [
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics", "astronomy",
        "college_physics", "conceptual_physics", "high_school_physics",
        "college_chemistry", "high_school_chemistry", "college_biology",
        "high_school_biology", "college_computer_science", "computer_security",
        "high_school_computer_science", "machine_learning", "electrical_engineering",
    ]

    # === ARC-Challenge ===
    print("Loading ARC-Challenge train...")
    arc_data = load_arrow_file(arc_train_path)
    n_arc = len(arc_data["question"])
    print(f"  Found {n_arc} examples")

    arc_indices = random.sample(range(n_arc), min(10, n_arc))
    arc_examples = []
    for idx in arc_indices:
        choices_dict = arc_data["choices"][idx]
        arc_examples.append({
            "question": arc_data["question"][idx],
            "choices": choices_dict["text"],
            "choice_labels": choices_dict["label"],
            "answer": arc_data["answerKey"][idx],
        })

    # === MMLU-STEM ===
    print("\nLoading MMLU-STEM dev splits...")
    mmlu_all = []
    for subject in mmlu_subjects:
        dev_path = f"{mmlu_base}/{subject}/0.0.0/c30699e8356da336a370243923dbaf21066bb9fe/mmlu-dev.arrow"
        if os.path.exists(dev_path):
            data = load_arrow_file(dev_path)
            n_items = len(data["question"])
            for i in range(n_items):
                mmlu_all.append({
                    "question": data["question"][i],
                    "choices": data["choices"][i],
                    "answer": chr(ord('A') + data["answer"][i]),
                    "subject": subject,
                })
    print(f"  Found {len(mmlu_all)} total MMLU-STEM dev examples")

    # Select 10 diverse examples (try to get different subjects)
    subjects_seen = set()
    mmlu_examples = []
    random.shuffle(mmlu_all)
    for item in mmlu_all:
        if item["subject"] not in subjects_seen:
            mmlu_examples.append({
                "question": item["question"],
                "choices": item["choices"],
                "choice_labels": ["A", "B", "C", "D"],
                "answer": item["answer"],
            })
            subjects_seen.add(item["subject"])
            if len(mmlu_examples) >= 10:
                break

    # If we don't have 10, fill with remaining
    if len(mmlu_examples) < 10:
        for item in mmlu_all:
            if len(mmlu_examples) >= 10:
                break
            if item not in [{"question": e["question"]} for e in mmlu_examples]:
                mmlu_examples.append({
                    "question": item["question"],
                    "choices": item["choices"],
                    "choice_labels": ["A", "B", "C", "D"],
                    "answer": item["answer"],
                })

    # === GPQA ===
    print("\nLoading GPQA train...")
    gpqa_data = load_arrow_file(gpqa_train_path)
    n_gpqa = len(gpqa_data["Question"])
    print(f"  Found {n_gpqa} examples")

    gpqa_indices = random.sample(range(n_gpqa), min(10, n_gpqa))
    gpqa_examples = []
    for idx in gpqa_indices:
        # Shuffle choices for GPQA (correct is always first in raw data)
        choices = [
            gpqa_data["Correct Answer"][idx],
            gpqa_data["Incorrect Answer 1"][idx],
            gpqa_data["Incorrect Answer 2"][idx],
            gpqa_data["Incorrect Answer 3"][idx],
        ]
        choices = [c for c in choices if c]  # Remove empty

        # Deterministic shuffle based on index
        random.seed(idx + 1000)
        indices = list(range(len(choices)))
        random.shuffle(indices)
        shuffled_choices = [choices[i] for i in indices]
        correct_idx = indices.index(0)
        correct_label = chr(ord('A') + correct_idx)

        gpqa_examples.append({
            "question": gpqa_data["Question"][idx],
            "choices": shuffled_choices,
            "choice_labels": [chr(ord('A') + i) for i in range(len(shuffled_choices))],
            "answer": correct_label,
        })

    # Reset seed after GPQA shuffling
    random.seed(42)

    # === Output ===
    print("\n" + "=" * 60)
    print("FEW-SHOT EXAMPLES")
    print("=" * 60)

    print("\n# ARC-Challenge (10 examples from train split)")
    for i, ex in enumerate(arc_examples):
        print(f"\n## Example {i+1}")
        print(f"Question: {ex['question'][:100]}...")
        print(f"Answer: {ex['answer']}")

    print("\n\n# MMLU-STEM (10 examples from dev split)")
    for i, ex in enumerate(mmlu_examples):
        print(f"\n## Example {i+1}")
        print(f"Question: {ex['question'][:100]}...")
        print(f"Answer: {ex['answer']}")

    print("\n\n# GPQA (10 examples from train split)")
    for i, ex in enumerate(gpqa_examples):
        print(f"\n## Example {i+1}")
        print(f"Question: {ex['question'][:100]}...")
        print(f"Answer: {ex['answer']}")

    # Save to JSON for use in eval scripts
    output = {
        "arc": arc_examples,
        "mmlu": mmlu_examples,
        "gpqa": gpqa_examples,
    }

    output_path = "/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/fewshot_examples.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved few-shot examples to: {output_path}")

    # Also print Python code for embedding in eval scripts
    print("\n" + "=" * 60)
    print("PYTHON CODE FOR EVAL SCRIPTS")
    print("=" * 60)
    print("\nFEW_SHOT_EXAMPLES = " + json.dumps(output, indent=4))


if __name__ == "__main__":
    main()
