"""
Custom FSDP Workers for GRPO that log trainable parameter counts.

This module provides a custom ActorRolloutRefWorker that adds
parameter logging to verl's standard worker.

IMPORTANT: The monkey-patch for convert_to_regular_types MUST happen
before importing verl.workers.fsdp_workers, because fsdp_workers imports
convert_to_regular_types at module load time. If we patch after the import,
the local binding in fsdp_workers is already set to the original function.
"""

import logging
import os
import sys

# =============================================================================
# CRITICAL: Apply monkey-patch BEFORE importing verl.workers.fsdp_workers
# =============================================================================
# This patch must happen before any verl worker imports because fsdp_workers.py
# does `from verl.utils.py_functional import convert_to_regular_types` at the
# top of the module. If we patch after that import, the local binding is already
# set and our patch won't take effect.

import verl.utils.py_functional as verl_py_functional

_original_convert_to_regular_types = verl_py_functional.convert_to_regular_types


def _patched_convert_to_regular_types(obj):
    """Patched version that splits comma-separated strings for target_modules.

    This is needed because:
    1. verl's HFModelConfig expects target_modules as a string (for OmegaConf validation)
    2. PEFT expects target_modules as a list for specific module names
    3. Only "all-linear" works as a string in PEFT

    This patch converts comma-separated strings like "q_proj,k_proj,v_proj,o_proj"
    into lists ["q_proj", "k_proj", "v_proj", "o_proj"] when convert_to_regular_types is called.
    """
    result = _original_convert_to_regular_types(obj)

    # If result is a comma-separated string (not "all-linear"), split it into a list
    if isinstance(result, str) and ',' in result and result != "all-linear":
        return [m.strip() for m in result.split(',')]

    return result


# Apply the monkey-patch to py_functional module BEFORE fsdp_workers is imported
verl_py_functional.convert_to_regular_types = _patched_convert_to_regular_types

# =============================================================================
# NOW we can safely import verl.workers.fsdp_workers
# The import will pick up our patched function
# =============================================================================
import ray
import torch
from omegaconf import DictConfig

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch

# Also patch the already-imported reference in fsdp_workers module
import verl.workers.fsdp_workers as verl_fsdp_workers
verl_fsdp_workers.convert_to_regular_types = _patched_convert_to_regular_types

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

__all__ = ['GRPOActorRolloutRefWorker']


@ray.remote
class GRPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """Custom ActorRolloutRefWorker that logs trainable parameter counts.

    This worker extends the standard ActorRolloutRefWorker to add
    logging of trainable vs frozen parameters after model initialization.

    Note: This module also monkey-patches verl's convert_to_regular_types
    to handle comma-separated LoRA target_modules strings.
    """

    def __init__(self, config: DictConfig, role: str):
        """Initialize GRPO worker.

        Args:
            config: Worker configuration
            role: Worker role ('actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref')
        """
        super().__init__(config, role)

        if self.rank == 0:
            print(f"GRPOActorRolloutRefWorker initialized (role={role})")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model and log parameter counts.

        This calls the parent init_model() and then logs trainable vs frozen
        parameter counts on rank 0.

        Note: The @register decorator is required for verl's RayWorkerGroup
        dispatch system to recognize this method. It overrides the parent's
        registered method.
        """
        print(f"[GRPO WORKER DEBUG] init_model() called on rank {self.rank}", flush=True)
        sys.stdout.flush()

        # Call parent implementation directly (bypass decorator by accessing the underlying method)
        # Note: target_modules conversion is handled by the monkey-patched
        # convert_to_regular_types function at module level
        ActorRolloutRefWorker.init_model(self)

        # Log trainable parameters (all ranks participate in all_reduce, only rank 0 prints)
        if self._is_actor:
            self._log_trainable_parameters()

        print(f"[GRPO WORKER DEBUG] init_model() completed on rank {self.rank}", flush=True)
        sys.stdout.flush()

    def _log_trainable_parameters(self):
        """Log trainable vs frozen parameter counts.

        Computes the TOTAL parameter count across all FSDP shards by summing
        local counts from each rank. This gives the true model size before sharding.

        All ranks must call this method (for all_reduce), but only rank 0 prints.
        """
        # Count local (sharded) parameters
        local_total = sum(p.numel() for p in self.actor_module_fsdp.parameters())
        local_trainable = sum(p.numel() for p in self.actor_module_fsdp.parameters() if p.requires_grad)

        # Sum across all ranks to get global totals (FSDP shards parameters across ranks)
        world_size = torch.distributed.get_world_size()

        # Create tensors for all_reduce
        total_tensor = torch.tensor([local_total], dtype=torch.long, device='cuda')
        trainable_tensor = torch.tensor([local_trainable], dtype=torch.long, device='cuda')

        # Sum across all ranks - ALL ranks must participate
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(trainable_tensor, op=torch.distributed.ReduceOp.SUM)

        # Only rank 0 prints the result
        if self.rank == 0:
            global_total = total_tensor.item()
            global_trainable = trainable_tensor.item()
            global_frozen = global_total - global_trainable

            if global_total > 0:
                trainable_pct = 100 * global_trainable / global_total
                frozen_pct = 100 * global_frozen / global_total
            else:
                trainable_pct = frozen_pct = 0.0

            print(f"Parameter summary (summed across {world_size} FSDP shards):")
            print(f"  Total params:     {global_total:,}")
            print(f"  Trainable params: {global_trainable:,} ({trainable_pct:.2f}%)")
            print(f"  Frozen params:    {global_frozen:,} ({frozen_pct:.2f}%)")
