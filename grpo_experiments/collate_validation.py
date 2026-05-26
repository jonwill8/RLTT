#!/usr/bin/env python3
"""
Collate validation results from GRPO training into a summary CSV.

Reads validation JSONL files and creates a CSV summarizing accuracy and reward
by difficulty level (and overall) across training steps.
"""
import os
import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


def load_validation_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a validation JSONL file."""
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_metrics_by_level(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics grouped by question level."""
    level_data = defaultdict(lambda: {"correct": 0, "total": 0, "rewards": []})

    for sample in samples:
        # Get accuracy (handle both numeric and boolean)
        acc = sample.get("acc", 0)
        if isinstance(acc, bool):
            acc = 1 if acc else 0
        acc = int(acc)

        # Get reward
        reward = sample.get("reward", 0)

        # Get question level
        level = sample.get("question_level", "unknown")
        if level is None:
            level = "unknown"

        level_data[level]["total"] += 1
        level_data[level]["correct"] += acc
        level_data[level]["rewards"].append(reward)

    # Calculate final metrics
    results = {}
    for level, data in level_data.items():
        avg_reward = sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0
        accuracy = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        results[level] = {
            "correct": data["correct"],
            "total": data["total"],
            "accuracy": accuracy,
            "avg_reward": avg_reward,
        }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collate validation results from GRPO training")
    parser.add_argument("--validation_dir", type=str, required=True,
                        help="Directory containing validation JSONL files")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output CSV file path (default: validation_summary.csv in parent of validation_dir)")
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = validation_dir.parent / "validation_summary.csv"

    # Find all validation JSONL files
    jsonl_files = list(validation_dir.glob("*.jsonl"))

    # Parse step numbers and sort
    step_files = {}
    for f in jsonl_files:
        try:
            step_num = int(f.stem)
            step_files[step_num] = f
        except ValueError:
            continue

    if not step_files:
        print("No validation files found!")
        return

    sorted_steps = sorted(step_files.keys())
    print(f"Found validation files for steps: {sorted_steps}")

    # Collect all metrics
    all_metrics = {}
    all_levels = set()

    for step in sorted_steps:
        filepath = step_files[step]
        print(f"Processing step {step}: {filepath}")

        samples = load_validation_file(str(filepath))
        metrics = compute_metrics_by_level(samples)
        all_metrics[step] = metrics
        all_levels.update(metrics.keys())

    # Sort levels (Level 1, Level 2, etc., then others)
    def level_sort_key(level):
        if isinstance(level, int):
            return (0, level)
        if str(level).startswith("Level "):
            try:
                return (0, int(str(level).replace("Level ", "")))
            except:
                pass
        return (1, str(level))

    sorted_levels = sorted(all_levels, key=level_sort_key)

    # Build CSV header
    header = ["Category"]
    for step in sorted_steps:
        header.extend([
            f"Step {step} Correct",
            f"Step {step} Total",
            f"Step {step} Accuracy (%)",
            f"Step {step} Avg Reward",
        ])

    # Build rows
    rows = []
    for level in sorted_levels:
        row = [f"Level {level}" if isinstance(level, int) else str(level)]
        for step in sorted_steps:
            metrics = all_metrics[step].get(level, {"correct": 0, "total": 0, "accuracy": 0, "avg_reward": 0})
            row.extend([
                metrics["correct"],
                metrics["total"],
                f"{metrics['accuracy']:.2f}",
                f"{metrics['avg_reward']:.4f}",
            ])
        rows.append(row)

    # Add overall row
    overall_row = ["Overall"]
    for step in sorted_steps:
        total_correct = sum(m["correct"] for m in all_metrics[step].values())
        total_samples = sum(m["total"] for m in all_metrics[step].values())
        all_rewards = []
        for level, metrics in all_metrics[step].items():
            # Reload to get individual rewards for overall average
            pass
        accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        avg_reward = (accuracy / 100 * 2) - 1  # Approximate based on accuracy
        # Better: compute overall average reward from the metrics
        weighted_reward = 0
        for level, metrics in all_metrics[step].items():
            weighted_reward += metrics["avg_reward"] * metrics["total"]
        avg_reward = weighted_reward / total_samples if total_samples > 0 else 0

        overall_row.extend([
            total_correct,
            total_samples,
            f"{accuracy:.2f}",
            f"{avg_reward:.4f}",
        ])
    rows.append(overall_row)

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved validation summary to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    for step in sorted_steps:
        total_correct = sum(m["correct"] for m in all_metrics[step].values())
        total_samples = sum(m["total"] for m in all_metrics[step].values())
        accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        print(f"Step {step}: {total_correct}/{total_samples} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
