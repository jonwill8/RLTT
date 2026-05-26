"""
Checkpoint manager for GRPO training with save-best-only behavior.

This module provides a CheckpointManager class that monitors validation outputs
and manages checkpoints to only keep those that improve validation accuracy.

Usage:
    After verl training completes, run the cleanup to remove non-best checkpoints:

    python checkpoint_manager.py --output_dir /path/to/grpo_output --cleanup

    Or use programmatically:

    from checkpoint_manager import CheckpointManager
    manager = CheckpointManager(output_dir)
    manager.cleanup_non_best_checkpoints()
"""
import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints to keep only the best ones based on validation accuracy."""

    def __init__(
        self,
        output_dir: str,
        metric_key: str = "overall_accuracy",
        keep_last: bool = True,
        keep_top_k: int = 1,
    ):
        """Initialize the checkpoint manager.

        Args:
            output_dir: GRPO output directory containing checkpoints and validation outputs
            metric_key: Metric to use for determining best checkpoint (default: overall_accuracy)
            keep_last: Whether to always keep the last checkpoint (default: True)
            keep_top_k: Number of best checkpoints to keep (default: 1)
        """
        self.output_dir = Path(output_dir)
        self.metric_key = metric_key
        self.keep_last = keep_last
        self.keep_top_k = keep_top_k

        # Paths
        self.validation_dir = self.output_dir / "validation_outputs"
        self.best_checkpoint_dir = self.output_dir / "best_checkpoint"
        self.checkpoint_registry_file = self.output_dir / "checkpoint_registry.json"

    def get_validation_accuracy(self, step: int) -> Optional[float]:
        """Get validation accuracy for a specific step from validation outputs.

        Args:
            step: Training step number

        Returns:
            Accuracy value or None if not found
        """
        # Check for validation JSONL file
        val_file = self.validation_dir / f"{step}.jsonl"
        if not val_file.exists():
            logger.warning(f"No validation file found for step {step}: {val_file}")
            return None

        # Parse validation JSONL to compute accuracy
        correct = 0
        total = 0

        try:
            with open(val_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    acc = sample.get("acc", 0)
                    if isinstance(acc, bool):
                        acc = 1 if acc else 0
                    correct += int(acc)
                    total += 1

            if total == 0:
                return None

            return correct / total

        except Exception as e:
            logger.error(f"Error reading validation file {val_file}: {e}")
            return None

    def find_checkpoints(self) -> List[Tuple[int, Path]]:
        """Find all checkpoint directories.

        Returns:
            List of (step, path) tuples sorted by step
        """
        checkpoints = []

        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("global_step_"):
                try:
                    step = int(item.name.replace("global_step_", ""))
                    # Verify it's a valid checkpoint (has actor directory)
                    if (item / "actor").exists():
                        checkpoints.append((step, item))
                except ValueError:
                    continue

        return sorted(checkpoints, key=lambda x: x[0])

    def get_checkpoint_accuracies(self) -> Dict[int, float]:
        """Get validation accuracies for all checkpoints.

        Returns:
            Dict mapping step -> accuracy
        """
        checkpoints = self.find_checkpoints()
        accuracies = {}

        for step, path in checkpoints:
            acc = self.get_validation_accuracy(step)
            if acc is not None:
                accuracies[step] = acc
                logger.info(f"Step {step}: accuracy = {acc:.2%}")
            else:
                logger.warning(f"Step {step}: no validation accuracy found")

        return accuracies

    def find_best_checkpoints(self) -> List[Tuple[int, float]]:
        """Find the best checkpoints based on validation accuracy.

        Returns:
            List of (step, accuracy) tuples for best checkpoints
        """
        accuracies = self.get_checkpoint_accuracies()

        if not accuracies:
            logger.warning("No checkpoint accuracies found")
            return []

        # Sort by accuracy descending
        sorted_checkpoints = sorted(
            accuracies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_checkpoints[:self.keep_top_k]

    def cleanup_non_best_checkpoints(self, dry_run: bool = False) -> Dict[str, any]:
        """Remove checkpoints that are not the best.

        Args:
            dry_run: If True, only report what would be deleted without deleting

        Returns:
            Dict with cleanup summary
        """
        checkpoints = self.find_checkpoints()
        accuracies = self.get_checkpoint_accuracies()

        if not checkpoints:
            logger.info("No checkpoints found")
            return {"status": "no_checkpoints"}

        # Find checkpoints to keep
        keep_steps = set()

        # Keep top-k by accuracy
        best = self.find_best_checkpoints()
        for step, acc in best:
            keep_steps.add(step)
            logger.info(f"Keeping step {step} (accuracy: {acc:.2%}) - best checkpoint")

        # Optionally keep last checkpoint
        if self.keep_last and checkpoints:
            last_step = checkpoints[-1][0]
            if last_step not in keep_steps:
                keep_steps.add(last_step)
                acc = accuracies.get(last_step)
                acc_str = f"{acc:.2%}" if acc else "unknown"
                logger.info(f"Keeping step {last_step} (accuracy: {acc_str}) - last checkpoint")

        # Delete non-best checkpoints
        deleted = []
        kept = []

        for step, path in checkpoints:
            if step in keep_steps:
                kept.append(step)
            else:
                acc = accuracies.get(step)
                acc_str = f"{acc:.2%}" if acc else "unknown"

                if dry_run:
                    logger.info(f"Would delete step {step} (accuracy: {acc_str})")
                else:
                    logger.info(f"Deleting step {step} (accuracy: {acc_str})")
                    try:
                        shutil.rmtree(path)
                        deleted.append(step)
                    except Exception as e:
                        logger.error(f"Failed to delete {path}: {e}")

        # Copy best checkpoint to best_checkpoint directory
        if best and not dry_run:
            best_step, best_acc = best[0]
            best_path = self.output_dir / f"global_step_{best_step}"

            if self.best_checkpoint_dir.exists():
                shutil.rmtree(self.best_checkpoint_dir)

            logger.info(f"Copying best checkpoint (step {best_step}, {best_acc:.2%}) to {self.best_checkpoint_dir}")
            shutil.copytree(best_path, self.best_checkpoint_dir)

            # Save metadata
            metadata = {
                "best_step": best_step,
                "best_accuracy": best_acc,
                "source_path": str(best_path),
            }
            with open(self.best_checkpoint_dir / "best_checkpoint_info.json", "w") as f:
                json.dump(metadata, f, indent=2)

        summary = {
            "total_checkpoints": len(checkpoints),
            "kept": kept,
            "deleted": deleted,
            "best_step": best[0][0] if best else None,
            "best_accuracy": best[0][1] if best else None,
            "dry_run": dry_run,
        }

        # Save registry
        if not dry_run:
            self._save_registry(summary)

        return summary

    def _save_registry(self, summary: Dict):
        """Save checkpoint registry to file."""
        with open(self.checkpoint_registry_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved checkpoint registry to {self.checkpoint_registry_file}")


def cleanup_checkpoints(
    output_dir: str,
    keep_top_k: int = 1,
    keep_last: bool = True,
    dry_run: bool = False,
) -> Dict[str, any]:
    """Convenience function to cleanup non-best checkpoints.

    Args:
        output_dir: GRPO output directory
        keep_top_k: Number of best checkpoints to keep
        keep_last: Whether to keep the last checkpoint
        dry_run: If True, only report what would be deleted

    Returns:
        Cleanup summary dict
    """
    manager = CheckpointManager(
        output_dir=output_dir,
        keep_top_k=keep_top_k,
        keep_last=keep_last,
    )
    return manager.cleanup_non_best_checkpoints(dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Manage GRPO checkpoints - keep only best by validation accuracy"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="GRPO output directory containing checkpoints and validation outputs",
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=1,
        help="Number of best checkpoints to keep (default: 1)",
    )
    parser.add_argument(
        "--keep_last",
        action="store_true",
        default=True,
        help="Always keep the last checkpoint (default: True)",
    )
    parser.add_argument(
        "--no_keep_last",
        action="store_true",
        help="Don't keep the last checkpoint",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would be deleted, don't actually delete",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoints with their accuracies",
    )

    args = parser.parse_args()

    keep_last = args.keep_last and not args.no_keep_last

    manager = CheckpointManager(
        output_dir=args.output_dir,
        keep_top_k=args.keep_top_k,
        keep_last=keep_last,
    )

    if args.list:
        # Just list checkpoints and accuracies
        print("\nCheckpoint Accuracies:")
        print("=" * 50)
        accuracies = manager.get_checkpoint_accuracies()
        if not accuracies:
            print("No checkpoints with validation data found")
        else:
            for step, acc in sorted(accuracies.items()):
                print(f"  Step {step}: {acc:.2%}")

            best = max(accuracies.items(), key=lambda x: x[1])
            print(f"\nBest: Step {best[0]} ({best[1]:.2%})")
        print("=" * 50)
    else:
        # Cleanup
        summary = manager.cleanup_non_best_checkpoints(dry_run=args.dry_run)

        print("\nCheckpoint Cleanup Summary:")
        print("=" * 50)
        print(f"  Total checkpoints: {summary['total_checkpoints']}")
        print(f"  Kept: {summary['kept']}")
        print(f"  Deleted: {summary['deleted']}")
        if summary['best_step'] is not None:
            print(f"  Best step: {summary['best_step']} ({summary['best_accuracy']:.2%})")
        if args.dry_run:
            print("\n  (Dry run - no files were deleted)")
        print("=" * 50)


if __name__ == "__main__":
    main()
