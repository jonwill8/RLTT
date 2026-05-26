"""
Rollout logging for GRPO training with verl.

This module provides a reward function wrapper that logs rollouts to a JSONL file
while computing rewards using the math_dapo reward function.
"""
import os
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Thread-safe file writing
_file_lock = threading.Lock()
_rollout_file = None
_step_counter = 0


def init_rollout_logger(output_dir: str):
    """Initialize the rollout logger with output directory."""
    global _rollout_file
    rollout_path = os.path.join(output_dir, "rollouts.jsonl")
    _rollout_file = rollout_path

    # Create/clear the file
    with open(rollout_path, "w") as f:
        # Write header comment
        f.write(json.dumps({
            "type": "header",
            "timestamp": datetime.now().isoformat(),
            "description": "GRPO training rollouts"
        }) + "\n")

    return rollout_path


def log_rollout(
    step: int,
    prompt: str,
    response: str,
    reward: float,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
):
    """Log a single rollout to the JSONL file."""
    global _rollout_file, _file_lock

    if _rollout_file is None:
        return

    entry = {
        "type": "rollout",
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "ground_truth": ground_truth,
    }

    if extra_info:
        entry["extra"] = extra_info

    with _file_lock:
        with open(_rollout_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


def log_step_summary(
    step: int,
    num_samples: int,
    mean_reward: float,
    num_correct: int
):
    """Log a step summary."""
    global _rollout_file, _file_lock

    if _rollout_file is None:
        return

    entry = {
        "type": "step_summary",
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "num_samples": num_samples,
        "mean_reward": mean_reward,
        "num_correct": num_correct,
        "accuracy": num_correct / num_samples if num_samples > 0 else 0.0
    }

    with _file_lock:
        with open(_rollout_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# Global step tracker for the reward function
_current_step = 0


def set_current_step(step: int):
    """Set the current training step for logging."""
    global _current_step
    _current_step = step


def get_current_step() -> int:
    """Get the current training step."""
    global _current_step
    return _current_step


def increment_step():
    """Increment the training step counter."""
    global _current_step
    _current_step += 1
    return _current_step
