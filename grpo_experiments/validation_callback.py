"""
Custom validation callback for GRPO training.

Computes:
- Overall accuracy
- Accuracy by question type (Algebra, Geometry, etc.)
- Accuracy by difficulty level

This callback processes the reward_extra_info returned by the reward function
and logs detailed metrics to wandb and local files.
"""
import os
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class ValidationMetricsCallback:
    """Callback to compute and log detailed validation metrics including accuracy by question type."""

    def __init__(
        self,
        output_dir: str,
        use_wandb: bool = True,
        log_to_file: bool = True,
    ):
        """Initialize the validation metrics callback.

        Args:
            output_dir: Directory to save validation metrics files
            use_wandb: Whether to log metrics to wandb
            log_to_file: Whether to save metrics to local JSON files
        """
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.log_to_file = log_to_file
        self.metrics_history = []

        # Create output directory
        self.metrics_dir = os.path.join(output_dir, "validation_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

    def compute_metrics(
        self,
        reward_extra_infos: Dict[str, List[Any]],
        global_step: int,
    ) -> Dict[str, float]:
        """Compute detailed validation metrics from reward extra info.

        Args:
            reward_extra_infos: Dict with keys like 'acc', 'question_type', 'question_level', etc.
                               Each value is a list of values for each sample.
            global_step: Current training step

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        # Get accuracy values and question types
        acc_values = reward_extra_infos.get("acc", [])
        question_types = reward_extra_infos.get("question_type", [])
        question_levels = reward_extra_infos.get("question_level", [])

        if not acc_values:
            logger.warning("No accuracy values found in reward_extra_infos")
            return metrics

        # Convert to numpy for easier computation
        acc_array = np.array(acc_values, dtype=float)

        # Overall accuracy
        overall_acc = acc_array.mean()
        metrics["val/overall_accuracy"] = overall_acc
        metrics["val/total_samples"] = len(acc_array)
        metrics["val/correct_samples"] = int(acc_array.sum())

        logger.info(f"Step {global_step} - Overall Accuracy: {overall_acc:.2%} ({int(acc_array.sum())}/{len(acc_array)})")

        # Accuracy by question type
        if question_types and len(question_types) == len(acc_values):
            type_metrics = self._compute_accuracy_by_group(
                acc_array, question_types, "type"
            )
            metrics.update(type_metrics)

            # Log type breakdown
            logger.info("Accuracy by Question Type:")
            type_acc = defaultdict(lambda: {"correct": 0, "total": 0})
            for acc, qtype in zip(acc_values, question_types):
                type_acc[qtype]["total"] += 1
                type_acc[qtype]["correct"] += acc
            for qtype in sorted(type_acc.keys()):
                data = type_acc[qtype]
                acc_pct = data["correct"] / data["total"] if data["total"] > 0 else 0
                logger.info(f"  {qtype}: {acc_pct:.2%} ({data['correct']}/{data['total']})")

        # Accuracy by difficulty level
        if question_levels and len(question_levels) == len(acc_values):
            level_metrics = self._compute_accuracy_by_group(
                acc_array, question_levels, "level"
            )
            metrics.update(level_metrics)

            # Log level breakdown
            logger.info("Accuracy by Difficulty Level:")
            level_acc = defaultdict(lambda: {"correct": 0, "total": 0})
            for acc, level in zip(acc_values, question_levels):
                level_acc[level]["total"] += 1
                level_acc[level]["correct"] += acc
            for level in sorted(level_acc.keys()):
                data = level_acc[level]
                acc_pct = data["correct"] / data["total"] if data["total"] > 0 else 0
                logger.info(f"  {level}: {acc_pct:.2%} ({data['correct']}/{data['total']})")

        return metrics

    def _compute_accuracy_by_group(
        self,
        acc_array: np.ndarray,
        groups: List[str],
        group_name: str,
    ) -> Dict[str, float]:
        """Compute accuracy metrics grouped by a category.

        Args:
            acc_array: Array of accuracy values (0 or 1)
            groups: List of group labels for each sample
            group_name: Name of the grouping (e.g., "type", "level")

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        # Group samples by category
        group_accs = defaultdict(list)
        for acc, group in zip(acc_array, groups):
            group_accs[group].append(acc)

        # Compute accuracy for each group
        for group, accs in group_accs.items():
            acc_mean = np.mean(accs)
            # Sanitize group name for metric key (replace spaces, special chars)
            safe_group = str(group).replace(" ", "_").replace("/", "_")
            metrics[f"val/acc_by_{group_name}/{safe_group}"] = acc_mean
            metrics[f"val/count_by_{group_name}/{safe_group}"] = len(accs)

        return metrics

    def log_metrics(
        self,
        metrics: Dict[str, float],
        global_step: int,
    ):
        """Log metrics to wandb and/or local file.

        Args:
            metrics: Dict of metric names to values
            global_step: Current training step
        """
        # Add step to metrics record
        metrics_record = {
            "step": global_step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.metrics_history.append(metrics_record)

        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=global_step)
            except ImportError:
                logger.warning("wandb not available, skipping wandb logging")
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        # Save to file
        if self.log_to_file:
            # Save individual step metrics
            step_file = os.path.join(self.metrics_dir, f"step_{global_step}.json")
            with open(step_file, "w") as f:
                json.dump(metrics_record, f, indent=2)

            # Update cumulative metrics file
            history_file = os.path.join(self.metrics_dir, "metrics_history.json")
            with open(history_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

            logger.info(f"Saved validation metrics to {step_file}")

    def __call__(
        self,
        reward_extra_infos: Dict[str, List[Any]],
        global_step: int,
    ) -> Dict[str, float]:
        """Process validation results and log metrics.

        Args:
            reward_extra_infos: Dict with reward extra info from validation
            global_step: Current training step

        Returns:
            Dict of computed metrics
        """
        metrics = self.compute_metrics(reward_extra_infos, global_step)
        self.log_metrics(metrics, global_step)
        return metrics


def process_validation_with_type_breakdown(
    data_sources: List[str],
    sample_uids: List[str],
    reward_extra_infos: Dict[str, List[Any]],
    output_dir: str,
    global_step: int,
    use_wandb: bool = True,
) -> Dict[str, float]:
    """Process validation metrics with accuracy breakdown by question type.

    This is a convenience function that can be called after verl's validation
    to add detailed metrics.

    Args:
        data_sources: List of data source identifiers for each sample
        sample_uids: List of sample UIDs
        reward_extra_infos: Dict from verl containing acc, question_type, etc.
        output_dir: Output directory for saving metrics
        global_step: Current training step
        use_wandb: Whether to log to wandb

    Returns:
        Dict of detailed validation metrics
    """
    callback = ValidationMetricsCallback(
        output_dir=output_dir,
        use_wandb=use_wandb,
        log_to_file=True,
    )

    return callback(reward_extra_infos, global_step)


def compute_accuracy_summary(
    reward_extra_infos: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """Compute a summary of accuracy metrics.

    Args:
        reward_extra_infos: Dict with acc, question_type, question_level lists

    Returns:
        Dict with overall accuracy and breakdown by type/level
    """
    acc_values = reward_extra_infos.get("acc", [])
    question_types = reward_extra_infos.get("question_type", [])
    question_levels = reward_extra_infos.get("question_level", [])

    if not acc_values:
        return {"overall_accuracy": 0.0, "total_samples": 0}

    summary = {
        "overall_accuracy": np.mean(acc_values),
        "total_samples": len(acc_values),
        "correct_samples": sum(acc_values),
    }

    # Accuracy by type
    if question_types:
        type_breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
        for acc, qtype in zip(acc_values, question_types):
            type_breakdown[qtype]["total"] += 1
            type_breakdown[qtype]["correct"] += acc

        summary["accuracy_by_type"] = {
            qtype: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "correct": data["correct"],
                "total": data["total"],
            }
            for qtype, data in type_breakdown.items()
        }

    # Accuracy by level
    if question_levels:
        level_breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
        for acc, level in zip(acc_values, question_levels):
            level_breakdown[level]["total"] += 1
            level_breakdown[level]["correct"] += acc

        summary["accuracy_by_level"] = {
            level: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "correct": data["correct"],
                "total": data["total"],
            }
            for level, data in level_breakdown.items()
        }

    return summary
