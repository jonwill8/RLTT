"""
Custom verl components for RLTT (Reward Latent Thought Trajectories).

This module provides custom actor workers and algorithms that implement
the full RLTT objective with multi-loop log-probability computation,
enabling multi-GPU training with verl while preserving the complete
RLTT loss function.

Usage:
    from verl_rltt import RLTTActorRolloutRefWorker

    # Use RLTTActorRolloutRefWorker instead of ActorRolloutRefWorker
    # in your verl training configuration
"""

from .rltt_actor import RLTTDataParallelPPOActor
from .rltt_algos import (
    compute_rltt_policy_loss,
    compute_loop_weights,
    aggregate_multi_loop_log_probs,
    compute_rltt_kl_penalty,
    LearnedLoopWeights,
)
from .rltt_fsdp_workers import RLTTActorRolloutRefWorker

__all__ = [
    # Actor
    'RLTTDataParallelPPOActor',
    # Algorithms
    'compute_rltt_policy_loss',
    'compute_loop_weights',
    'aggregate_multi_loop_log_probs',
    'compute_rltt_kl_penalty',
    'LearnedLoopWeights',
    # Workers
    'RLTTActorRolloutRefWorker',
]
