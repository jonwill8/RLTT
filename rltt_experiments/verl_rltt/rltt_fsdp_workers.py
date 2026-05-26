"""
RLTT-aware FSDP Workers for verl.

This module provides custom worker classes that integrate the RLTT actor
into verl's distributed training framework.
"""

import logging
import os
import sys

import ray
import torch
from omegaconf import DictConfig

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
import verl.utils.py_functional as verl_py_functional

from .rltt_actor import RLTTDataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


# Monkey-patch convert_to_regular_types to handle comma-separated target_modules
# We need to patch both the module AND the imported reference in fsdp_workers
import verl.workers.fsdp_workers as verl_fsdp_workers

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


# Apply the monkey-patch to both locations
verl_py_functional.convert_to_regular_types = _patched_convert_to_regular_types
verl_fsdp_workers.convert_to_regular_types = _patched_convert_to_regular_types

__all__ = ['RLTTActorRolloutRefWorker']


@ray.remote
class RLTTActorRolloutRefWorker(ActorRolloutRefWorker):
    """RLTT-aware version of ActorRolloutRefWorker.

    This worker uses RLTTDataParallelPPOActor instead of the standard
    DataParallelPPOActor, enabling multi-loop log-probability computation
    for the full RLTT objective.
    """

    def __init__(self, config: DictConfig, role: str):
        """Initialize RLTT worker.

        Args:
            config: Worker configuration (should include RLTT-specific options)
            role: Worker role ('actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref')
        """
        super().__init__(config, role)

        # Extract RLTT-specific configuration
        self.loop_weighting = config.get('loop_weighting', 'uniform')
        self.progressive_alpha = config.get('progressive_alpha', 1.0)
        self.total_ut_steps = config.get('total_ut_steps', 4)

        # Learned weights configuration
        self.learned_weights_init = config.get('learned_weights_init', 'uniform')
        self.learned_weights_temp = config.get('learned_weights_temp', 1.0)
        self.learned_weights_lr = config.get('learned_weights_lr', None)

        if self.rank == 0:
            print(f"RLTT Worker initialized:")
            print(f"  loop_weighting: {self.loop_weighting}")
            print(f"  progressive_alpha: {self.progressive_alpha}")
            print(f"  total_ut_steps: {self.total_ut_steps}")
            if self.loop_weighting == "learned":
                print(f"  learned_weights_init: {self.learned_weights_init}")
                print(f"  learned_weights_temp: {self.learned_weights_temp}")
                print(f"  learned_weights_lr: {self.learned_weights_lr}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with RLTT actor instead of standard actor.

        This calls the parent init_model() and then replaces the standard
        DataParallelPPOActor with RLTTDataParallelPPOActor.
        """
        print(f"[RLTT WORKER DEBUG] init_model() called on rank {self.rank}", flush=True)
        sys.stdout.flush()

        # Call parent implementation - this handles all the model loading,
        # optimizer creation, rollout/ref setup, etc. with proper trust_remote_code
        # Note: target_modules conversion is handled by the monkey-patched
        # convert_to_regular_types function at module level
        super().init_model()

        # Replace standard actor with RLTT actor
        if self._is_actor:
            print(f"[RLTT WORKER DEBUG] Replacing actor with RLTTDataParallelPPOActor on rank {self.rank}", flush=True)
            sys.stdout.flush()

            # Use RLTTDataParallelPPOActor instead of DataParallelPPOActor
            self.actor = RLTTDataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                loop_weighting=self.loop_weighting,
                progressive_alpha=self.progressive_alpha,
                total_ut_steps=self.total_ut_steps,
                learned_weights_init=self.learned_weights_init,
                learned_weights_temp=self.learned_weights_temp,
                learned_weights_lr=self.learned_weights_lr,
            )

            if self.rank == 0:
                print(f"Created RLTTDataParallelPPOActor with:")
                print(f"  loop_weighting={self.loop_weighting}")
                print(f"  progressive_alpha={self.progressive_alpha}")
                print(f"  total_ut_steps={self.total_ut_steps}")
                if self.loop_weighting == "learned":
                    print(f"  learned_weights_init={self.learned_weights_init}")
                    print(f"  learned_weights_temp={self.learned_weights_temp}")
                    print(f"  learned_weights_lr={self.learned_weights_lr}")

            # Log trainable vs frozen parameters (all ranks participate in all_reduce)
            self._log_trainable_parameters()

        print(f"[RLTT WORKER DEBUG] init_model() completed on rank {self.rank}", flush=True)
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
