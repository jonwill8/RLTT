"""
Custom verl components for GRPO experiments.

This module provides a custom ActorRolloutRefWorker that logs
trainable vs frozen parameter counts during initialization.

Usage:
    from verl_grpo import GRPOActorRolloutRefWorker

    # Use GRPOActorRolloutRefWorker instead of ActorRolloutRefWorker
    # in your verl training configuration
"""

from .grpo_fsdp_workers import GRPOActorRolloutRefWorker

__all__ = [
    'GRPOActorRolloutRefWorker',
]
