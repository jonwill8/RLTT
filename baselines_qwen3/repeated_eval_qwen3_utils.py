#!/usr/bin/env python3
"""Qwen3 checkpoint helpers for repeated-evaluation scripts."""
import logging
import os
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


def _is_hf_model(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def _step_num(checkpoint: str) -> str:
    if checkpoint.startswith("step_"):
        return checkpoint.replace("step_", "", 1)
    if checkpoint.startswith("global_step_"):
        return checkpoint.replace("global_step_", "", 1)
    return checkpoint


def _checkpoint_dir(experiment_dir: str, checkpoint: str) -> str:
    if os.path.basename(experiment_dir).startswith("global_step_"):
        return experiment_dir
    return os.path.join(experiment_dir, f"global_step_{_step_num(checkpoint)}")


def _actor_dir(experiment_dir: str, checkpoint: str) -> str:
    if os.path.basename(experiment_dir) == "actor":
        return experiment_dir
    return os.path.join(_checkpoint_dir(experiment_dir, checkpoint), "actor")


def find_merged_model(experiment_dir: str, checkpoint: str) -> Optional[str]:
    """Find an existing HF-format model for a Qwen3 checkpoint, if available."""
    if _is_hf_model(experiment_dir):
        logger.info(f"Using HF-format model directory: {experiment_dir}")
        return experiment_dir

    checkpoint_dir = _checkpoint_dir(experiment_dir, checkpoint)
    if _is_hf_model(checkpoint_dir):
        logger.info(f"Using HF-format checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir

    merged_in_checkpoint = os.path.join(checkpoint_dir, "merged_model")
    if _is_hf_model(merged_in_checkpoint):
        logger.info(f"Found existing merged model at: {merged_in_checkpoint}")
        return merged_in_checkpoint

    try:
        dirnames = os.listdir(experiment_dir)
    except OSError:
        return None

    for dirname in dirnames:
        if dirname.startswith("eval_"):
            eval_dir = os.path.join(experiment_dir, dirname)
            merged_path = os.path.join(eval_dir, checkpoint, "merged_model")
            if _is_hf_model(merged_path):
                logger.info(f"Found existing merged model at: {merged_path}")
                return merged_path

    return None


def merge_checkpoint(experiment_dir: str, checkpoint: str, output_dir: str) -> str:
    """Merge a Qwen3 FSDP actor checkpoint to HuggingFace format."""
    existing = find_merged_model(experiment_dir, checkpoint)
    if existing is not None:
        return existing

    fsdp_checkpoint_dir = _actor_dir(experiment_dir, checkpoint)
    merged_model_dir = os.path.join(output_dir, "merged_model")

    if not os.path.isdir(fsdp_checkpoint_dir):
        raise ValueError(f"FSDP checkpoint not found: {fsdp_checkpoint_dir}")

    if _is_hf_model(merged_model_dir):
        logger.info(f"Using existing merged model at: {merged_model_dir}")
        return merged_model_dir

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Merging Qwen3 FSDP checkpoint from: {fsdp_checkpoint_dir}")
    logger.info(f"Target directory: {merged_model_dir}")

    result = subprocess.run(
        [
            "python", "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--trust-remote-code",
            "--local_dir", fsdp_checkpoint_dir,
            "--target_dir", merged_model_dir,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Merge stdout: {result.stdout}")
        logger.error(f"Merge stderr: {result.stderr}")
        raise RuntimeError(f"Failed to merge checkpoint: {result.stderr}")

    logger.info("Checkpoint merged successfully")
    return merged_model_dir
