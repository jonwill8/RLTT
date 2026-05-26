#!/usr/bin/env python3
"""
Analyze validation outputs from GRPO training.

Computes:
- Overall accuracy
- Accuracy by question type (Algebra, Geometry, etc.)
- Accuracy by difficulty level

Usage:
    # Analyze a specific validation file
    python analyze_validation.py --validation_dir /path/to/validation_outputs

    # Analyze a specific step
    python analyze_validation.py --validation_dir /path/to/validation_outputs --step 100

    # Watch for new files and analyze them
    python analyze_validation.py --validation_dir /path/to/validation_outputs --watch
"""
import os
import sys
import json
import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_validation_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a validation JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of validation samples
    """
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_accuracy_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute detailed accuracy metrics from validation samples.

    Args:
        samples: List of validation sample dicts with 'acc', 'question_type', etc.

    Returns:
        Dict with overall accuracy and breakdown by type/level
    """
    if not samples:
        return {"error": "No samples found"}

    # Extract data
    acc_values = []
    question_types = []
    question_levels = []

    for sample in samples:
        # Handle both numeric and boolean acc values
        acc = sample.get("acc", 0)
        if isinstance(acc, bool):
            acc = 1 if acc else 0
        acc_values.append(float(acc))

        # Get question type and level
        qtype = sample.get("question_type", "unknown")
        if qtype is None:
            qtype = "unknown"
        question_types.append(qtype)

        qlevel = sample.get("question_level", "unknown")
        if qlevel is None:
            qlevel = "unknown"
        question_levels.append(qlevel)

    # Overall metrics
    acc_array = np.array(acc_values)
    metrics = {
        "overall_accuracy": float(acc_array.mean()),
        "total_samples": len(acc_array),
        "correct_samples": int(acc_array.sum()),
    }

    # Accuracy by question type
    type_breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    for acc, qtype in zip(acc_values, question_types):
        type_breakdown[qtype]["total"] += 1
        type_breakdown[qtype]["correct"] += int(acc)

    metrics["accuracy_by_type"] = {
        qtype: {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"],
        }
        for qtype, data in sorted(type_breakdown.items())
    }

    # Accuracy by difficulty level
    level_breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    for acc, level in zip(acc_values, question_levels):
        level_breakdown[level]["total"] += 1
        level_breakdown[level]["correct"] += int(acc)

    metrics["accuracy_by_level"] = {
        level: {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "correct": data["correct"],
            "total": data["total"],
        }
        for level, data in sorted(level_breakdown.items())
    }

    return metrics


def print_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Print metrics in a formatted way.

    Args:
        metrics: Dict of computed metrics
        step: Optional step number
    """
    if "error" in metrics:
        logger.error(metrics["error"])
        return

    step_str = f"Step {step}" if step is not None else "Validation Results"
    print(f"\n{'='*60}")
    print(f"{step_str}")
    print(f"{'='*60}")

    # Overall accuracy
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"  Correct: {metrics['correct_samples']}/{metrics['total_samples']}")

    # Accuracy by question type
    if "accuracy_by_type" in metrics and metrics["accuracy_by_type"]:
        print(f"\nAccuracy by Question Type:")
        for qtype, data in metrics["accuracy_by_type"].items():
            print(f"  {qtype}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    # Accuracy by difficulty level
    if "accuracy_by_level" in metrics and metrics["accuracy_by_level"]:
        print(f"\nAccuracy by Difficulty Level:")
        for level, data in metrics["accuracy_by_level"].items():
            print(f"  {level}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

    print(f"{'='*60}\n")


def analyze_validation_dir(
    validation_dir: str,
    step: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze validation outputs from a directory.

    Args:
        validation_dir: Directory containing validation JSONL files
        step: Specific step to analyze (if None, analyzes latest)
        output_file: Optional file to save results

    Returns:
        Dict of metrics for the analyzed step(s)
    """
    validation_path = Path(validation_dir)
    if not validation_path.exists():
        logger.error(f"Validation directory not found: {validation_dir}")
        return {}

    # Find validation files
    jsonl_files = list(validation_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {validation_dir}")
        return {}

    # Parse step numbers from filenames
    step_files = {}
    for f in jsonl_files:
        try:
            step_num = int(f.stem)
            step_files[step_num] = f
        except ValueError:
            continue

    if not step_files:
        logger.warning("No valid step files found")
        return {}

    # Analyze specific step or latest
    if step is not None:
        if step not in step_files:
            logger.error(f"Step {step} not found. Available steps: {sorted(step_files.keys())}")
            return {}
        steps_to_analyze = [step]
    else:
        steps_to_analyze = sorted(step_files.keys())

    all_metrics = {}
    for step_num in steps_to_analyze:
        filepath = step_files[step_num]
        logger.info(f"Analyzing {filepath}")

        samples = load_validation_file(str(filepath))
        metrics = compute_accuracy_metrics(samples)
        metrics["step"] = step_num
        metrics["file"] = str(filepath)

        all_metrics[step_num] = metrics
        print_metrics(metrics, step_num)

    # Save results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_file}")

    return all_metrics


def watch_validation_dir(validation_dir: str, poll_interval: float = 30.0):
    """Watch for new validation files and analyze them.

    Args:
        validation_dir: Directory to watch
        poll_interval: Seconds between checks
    """
    validation_path = Path(validation_dir)
    analyzed_files = set()

    logger.info(f"Watching {validation_dir} for new validation files...")
    logger.info(f"Press Ctrl+C to stop")

    while True:
        try:
            jsonl_files = list(validation_path.glob("*.jsonl"))

            for filepath in jsonl_files:
                if filepath not in analyzed_files:
                    try:
                        step_num = int(filepath.stem)
                    except ValueError:
                        continue

                    logger.info(f"New validation file found: {filepath}")
                    samples = load_validation_file(str(filepath))
                    metrics = compute_accuracy_metrics(samples)
                    print_metrics(metrics, step_num)

                    analyzed_files.add(filepath)

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopping watch...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Analyze validation outputs from GRPO training"
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        required=True,
        help="Directory containing validation JSONL files",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific step to analyze (default: analyze all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for metrics JSON",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for new files and analyze them continuously",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=30.0,
        help="Poll interval in seconds for watch mode",
    )

    args = parser.parse_args()

    if args.watch:
        watch_validation_dir(args.validation_dir, args.poll_interval)
    else:
        analyze_validation_dir(
            args.validation_dir,
            step=args.step,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
