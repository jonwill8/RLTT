#!/usr/bin/env python3
"""
Analyze SFT evaluation results to compute per-level and per-subject accuracies.

Usage:
    python analyze_sft_eval.py <step_filepath>

Example:
    python analyze_sft_eval.py /path/to/validation_outputs/step_5.json
"""

import argparse
import csv
import json
import os
from pathlib import Path


# Default path to MATH-500 dataset
DEFAULT_MATH500_PATH = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"


def load_math500_metadata(math500_path: str) -> dict:
    """Load MATH-500 dataset to get level and subject information for each problem."""
    math500_data = {}
    with open(math500_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            problem_key = item['problem'].strip()
            math500_data[problem_key] = {
                'level': item['level'],
                'subject': item['subject']
            }
    return math500_data


def analyze_eval_file(eval_filepath: str, math500_path: str = DEFAULT_MATH500_PATH) -> dict:
    """
    Analyze an SFT evaluation file and compute per-level and per-subject accuracies.

    Args:
        eval_filepath: Path to the step_X.json evaluation file
        math500_path: Path to the MATH-500 dataset file

    Returns:
        Dictionary containing all accuracy statistics
    """
    # Load MATH-500 metadata
    math500_data = load_math500_metadata(math500_path)

    # Load evaluation results
    with open(eval_filepath, 'r') as f:
        eval_data = json.load(f)

    samples = eval_data['math500']['samples']

    # Initialize statistics
    level_stats = {i: {'correct': 0, 'total': 0} for i in range(1, 6)}
    subject_stats = {}
    unmatched = 0

    # Process each sample
    for sample in samples:
        problem_key = sample['problem'].strip()

        if problem_key in math500_data:
            level = math500_data[problem_key]['level']
            subject = math500_data[problem_key]['subject']

            # Update level stats
            level_stats[level]['total'] += 1
            if sample['correct']:
                level_stats[level]['correct'] += 1

            # Update subject stats
            if subject not in subject_stats:
                subject_stats[subject] = {'correct': 0, 'total': 0}
            subject_stats[subject]['total'] += 1
            if sample['correct']:
                subject_stats[subject]['correct'] += 1
        else:
            unmatched += 1

    # Compute accuracies
    for level in level_stats:
        stats = level_stats[level]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    for subject in subject_stats:
        stats = subject_stats[subject]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # Compute overall stats
    total_correct = sum(s['correct'] for s in level_stats.values())
    total_count = sum(s['total'] for s in level_stats.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    return {
        'filepath': eval_filepath,
        'step': eval_data.get('step', 'unknown'),
        'overall': {
            'correct': total_correct,
            'total': total_count,
            'accuracy': overall_accuracy
        },
        'per_level': level_stats,
        'per_subject': subject_stats,
        'unmatched': unmatched
    }


def print_results(results: dict, show_subject: bool = True):
    """Print formatted results to console."""
    print(f"\nFile: {results['filepath']}")
    print(f"Step: {results['step']}")

    if results['unmatched'] > 0:
        print(f"Warning: {results['unmatched']} problems could not be matched to MATH-500 dataset")

    # Per-level accuracies
    print("\n" + "=" * 60)
    print("PER-LEVEL ACCURACY")
    print("=" * 60)
    print(f"{'Level':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<15}")
    print("-" * 60)

    for level in sorted(results['per_level'].keys()):
        stats = results['per_level'][level]
        acc_str = f"{stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)"
        print(f"Level {level:<4} {stats['correct']:<10} {stats['total']:<10} {acc_str}")

    print("-" * 60)
    overall = results['overall']
    acc_str = f"{overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)"
    print(f"{'Overall':<10} {overall['correct']:<10} {overall['total']:<10} {acc_str}")

    # Per-subject accuracies
    if show_subject:
        print("\n" + "=" * 60)
        print("PER-SUBJECT ACCURACY")
        print("=" * 60)
        print(f"{'Subject':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<15}")
        print("-" * 60)

        # Sort by accuracy descending
        sorted_subjects = sorted(
            results['per_subject'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for subject, stats in sorted_subjects:
            acc_str = f"{stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)"
            print(f"{subject:<25} {stats['correct']:<10} {stats['total']:<10} {acc_str}")


def save_csv(results: dict, output_path: str):
    """Save results to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header row with step, overall, and per-level columns
        header = ['step', 'overall_accuracy', 'overall_correct', 'overall_total']
        for level in range(1, 6):
            header.extend([f'level{level}_accuracy', f'level{level}_correct', f'level{level}_total'])

        # Add subject columns
        subjects = sorted(results['per_subject'].keys())
        for subject in subjects:
            safe_subject = subject.replace(' ', '_').replace('&', 'and')
            header.extend([f'{safe_subject}_accuracy', f'{safe_subject}_correct', f'{safe_subject}_total'])

        writer.writerow(header)

        # Data row
        row = [
            results['step'],
            results['overall']['accuracy'],
            results['overall']['correct'],
            results['overall']['total']
        ]

        for level in range(1, 6):
            stats = results['per_level'][level]
            row.extend([stats['accuracy'], stats['correct'], stats['total']])

        for subject in subjects:
            stats = results['per_subject'][subject]
            row.extend([stats['accuracy'], stats['correct'], stats['total']])

        writer.writerow(row)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SFT evaluation results for per-level and per-subject accuracies."
    )
    parser.add_argument(
        "eval_filepath",
        type=str,
        help="Path to the step_X.json evaluation file"
    )
    parser.add_argument(
        "--math500-path",
        type=str,
        default=DEFAULT_MATH500_PATH,
        help=f"Path to MATH-500 dataset (default: {DEFAULT_MATH500_PATH})"
    )
    parser.add_argument(
        "--no-subject",
        action="store_true",
        help="Skip per-subject accuracy output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output results to a CSV file at the specified path"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.eval_filepath):
        print(f"Error: File not found: {args.eval_filepath}")
        return 1

    if not os.path.exists(args.math500_path):
        print(f"Error: MATH-500 dataset not found: {args.math500_path}")
        return 1

    # Analyze the file
    results = analyze_eval_file(args.eval_filepath, args.math500_path)

    # Output results
    if args.csv:
        save_csv(results, args.csv)
    elif args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, show_subject=not args.no_subject)

    return 0


if __name__ == "__main__":
    exit(main())
