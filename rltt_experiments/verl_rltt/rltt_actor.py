"""
Custom RLTT Actor for verl that computes per-loop log-probabilities.

This actor extends DataParallelPPOActor to support the full RLTT objective
by computing log-probabilities for each recurrent loop and aggregating
them with configurable weighting strategies.
"""

import itertools
from typing import Tuple, List, Optional, Dict, Any
import inspect

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from .rltt_algos import (
    compute_loop_weights,
    compute_rltt_policy_loss,
    compute_rltt_kl_penalty,
    aggregate_multi_loop_log_probs,
)

__all__ = ['RLTTDataParallelPPOActor']


class RLTTDataParallelPPOActor(DataParallelPPOActor):
    """RLTT-aware Actor that computes per-loop log-probabilities.

    This actor extends the standard DataParallelPPOActor to:
    1. Detect if the model supports per-loop logit computation
    2. Compute log-probabilities for each recurrent loop
    3. Aggregate them using RLTT's weighted sum objective
    4. Compute the policy gradient loss with multi-loop awareness
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        loop_weighting: str = "uniform",
        progressive_alpha: float = 1.0,
        total_ut_steps: int = 4,
        learned_weights_init: str = "uniform",
        learned_weights_temp: float = 1.0,
        learned_weights_lr: Optional[float] = None,
    ):
        """Initialize RLTT actor.

        Args:
            config: Actor configuration from verl
            actor_module: The model (Ouro or RLTT-modified Ouro)
            actor_optimizer: Optimizer for training
            loop_weighting: Strategy for weighting loops ("uniform", "progressive", "exit_pdf", "learned")
            progressive_alpha: Alpha exponent for progressive weighting
            total_ut_steps: Number of recurrent loops (T_max)
            learned_weights_init: Initialization for learned weights ("uniform" or "progressive")
            learned_weights_temp: Softmax temperature for learned weights
            learned_weights_lr: Learning rate for learned weights (default: 10x model LR)
        """
        super().__init__(config, actor_module, actor_optimizer)

        self.loop_weighting = loop_weighting
        self.progressive_alpha = progressive_alpha
        self.total_ut_steps = total_ut_steps
        self.use_learned_weights = (loop_weighting == "learned")

        # Check if model supports native per-loop logits
        self.model_supports_per_loop = self._check_model_rltt_support()

        # Initialize learned weights if using them
        self.learned_weights_module = None
        if self.use_learned_weights:
            from .rltt_algos import LearnedLoopWeights
            self.learned_weights_module = LearnedLoopWeights(
                num_loops=total_ut_steps,
                init_strategy=learned_weights_init,
                init_alpha=progressive_alpha,
                temperature=learned_weights_temp,
                device=torch.cuda.current_device(),
            )
            # Add learned weights to optimizer
            if actor_optimizer is not None:
                # Get current LR from optimizer
                current_lr = actor_optimizer.param_groups[0]['lr']
                weight_lr = learned_weights_lr if learned_weights_lr else current_lr * 10
                actor_optimizer.add_param_group({
                    'params': self.learned_weights_module.parameters(),
                    'lr': weight_lr,
                    'weight_decay': 0.0,
                })
                print(f"RLTT Actor: Added learned weights to optimizer (lr={weight_lr})")
            print(f"RLTT Actor: Using LEARNED weights (init={learned_weights_init}, temp={learned_weights_temp})")
            print(f"  Initial weights: {self.learned_weights_module.get_weights_for_logging()}")

        if self.model_supports_per_loop:
            print(f"RLTT Actor: Model supports native per-loop logits")
        else:
            print(f"RLTT Actor: Model does NOT support per-loop logits, using fallback")

        print(f"RLTT Actor: loop_weighting={loop_weighting}, alpha={progressive_alpha}, T_max={total_ut_steps}")

    def _check_model_rltt_support(self) -> bool:
        """Check if model supports native per-loop logit computation."""
        # Get the underlying module (may be wrapped in FSDP/DDP/PEFT)
        model = self.actor_module
        print(f"[RLTT DEBUG] _check_model_rltt_support: actor_module type = {type(model)}", flush=True)
        print(f"[RLTT DEBUG] hasattr checks: _fsdp_wrapped_module={hasattr(model, '_fsdp_wrapped_module')}, _orig_mod={hasattr(model, '_orig_mod')}, module={hasattr(model, 'module')}", flush=True)

        # Handle various FSDP and PEFT wrapping patterns - need to unwrap recursively
        # Order: FSDP -> PEFT -> actual model
        max_unwrap = 10
        for _ in range(max_unwrap):
            unwrapped = False
            # FSDP wrapping
            if hasattr(model, '_fsdp_wrapped_module'):
                model = model._fsdp_wrapped_module
                print(f"[RLTT DEBUG] Unwrapped _fsdp_wrapped_module -> {type(model)}", flush=True)
                unwrapped = True
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                print(f"[RLTT DEBUG] Unwrapped _orig_mod -> {type(model)}", flush=True)
                unwrapped = True
            # For FSDP, .module gives the wrapped module
            if hasattr(model, 'module') and not isinstance(getattr(model, 'module', None), type(None)):
                inner = model.module
                # Check it's not a method or the same object
                if inner is not model and not callable(inner):
                    model = inner
                    print(f"[RLTT DEBUG] Unwrapped module -> {type(model)}", flush=True)
                    unwrapped = True
            # PEFT/LoRA wrapping - PeftModelForCausalLM has base_model.model
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                model = model.base_model.model
                print(f"[RLTT DEBUG] Unwrapped PEFT base_model.model -> {type(model)}", flush=True)
                unwrapped = True
            elif hasattr(model, 'get_base_model'):
                try:
                    base = model.get_base_model()
                    if base is not model:
                        model = base
                        print(f"[RLTT DEBUG] Unwrapped via get_base_model() -> {type(model)}", flush=True)
                        unwrapped = True
                except Exception:
                    pass
            if not unwrapped:
                break

        print(f"[RLTT DEBUG] Final unwrapped model type = {type(model)}", flush=True)

        # Check for RLTT-specific forward parameters
        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters
            param_names = list(params.keys())
            print(f"[RLTT DEBUG] Forward params (first 10): {param_names[:10]}", flush=True)
            # Check for the new memory-efficient option first, fall back to old option
            has_log_probs = 'return_per_loop_log_probs' in params
            has_logits = 'return_per_loop_logits' in params
            has_exit_pdf = 'return_exit_pdf' in params
            result = (has_log_probs or has_logits) and has_exit_pdf
            print(f"[RLTT DEBUG] Forward params check: log_probs={has_log_probs}, logits={has_logits}, exit_pdf={has_exit_pdf} -> {result}", flush=True)
            return result
        except Exception as e:
            print(f"[RLTT DEBUG] Exception checking forward signature: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _model_supports_per_loop_log_probs(self) -> bool:
        """Check if model supports the memory-efficient return_per_loop_log_probs option."""
        model = self.actor_module
        # Handle various FSDP and PEFT wrapping patterns - unwrap recursively
        max_unwrap = 10
        for _ in range(max_unwrap):
            unwrapped = False
            # FSDP wrapping
            if hasattr(model, '_fsdp_wrapped_module'):
                model = model._fsdp_wrapped_module
                unwrapped = True
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                unwrapped = True
            if hasattr(model, 'module') and not isinstance(getattr(model, 'module', None), type(None)):
                inner = model.module
                if inner is not model and not callable(inner):
                    model = inner
                    unwrapped = True
            # PEFT/LoRA wrapping - PeftModelForCausalLM has base_model.model
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                model = model.base_model.model
                unwrapped = True
            elif hasattr(model, 'get_base_model'):
                try:
                    base = model.get_base_model()
                    if base is not model:
                        model = base
                        unwrapped = True
                except Exception:
                    pass
            if not unwrapped:
                break

        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters
            result = 'return_per_loop_log_probs' in params
            print(f"[RLTT DEBUG] _model_supports_per_loop_log_probs: model_type={type(model).__name__}, result={result}", flush=True)
            return result
        except Exception as e:
            print(f"[RLTT DEBUG] _model_supports_per_loop_log_probs exception: {e}", flush=True)
            return False

    def _forward_micro_batch_rltt(
        self,
        micro_batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass that returns per-loop log-probabilities.

        Returns:
            entropy: (bs, response_length)
            log_probs: (bs, response_length) - final loop log-probs (for compatibility)
            per_loop_log_probs: List of T tensors, each (bs, response_length)
            exit_pdf: Optional (bs, response_length, T) if requested and model supports it
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch['multi_modal_inputs']], dim=0
                )

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            # For RLTT, we need per-loop log-probs
            # Try memory-efficient option first (return_per_loop_log_probs)
            # Fall back to return_per_loop_logits if not available
            if self.model_supports_per_loop:
                use_memory_efficient = self._model_supports_per_loop_log_probs()

                if use_memory_efficient:
                    # Memory-efficient path: model computes log-probs internally
                    # and deletes logits immediately after each loop
                    # We need to provide the labels (response tokens) for log-prob computation
                    # The model expects labels aligned with input_ids, so we need the full sequence
                    # but shifted by 1 for next-token prediction

                    # Create labels tensor: shift input_ids left by 1, response portion only matters
                    # For log-prob computation, we want P(token_t | tokens_<t)
                    # The model will compute log-probs for all positions, we slice later
                    labels_for_log_prob = input_ids[:, 1:]  # Shift left by 1
                    # Pad to match original length
                    labels_for_log_prob = torch.nn.functional.pad(
                        labels_for_log_prob, (0, 1), value=0
                    )

                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_per_loop_log_probs=True,
                        return_exit_pdf=(self.loop_weighting == "exit_pdf"),
                        log_prob_labels=labels_for_log_prob,
                        log_prob_temperature=temperature,
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    # Get per-loop log-probs and entropies directly from model (computed efficiently)
                    per_loop_log_probs_full = output.per_loop_log_probs  # List of T tensors
                    per_loop_entropies_full = output.per_loop_entropies  # List of T tensors
                    exit_pdf = getattr(output, 'exit_pdf', None)

                    # Clear output reference
                    del output

                    # Slice log-probs and entropies to response portion
                    per_loop_log_probs = []
                    per_loop_entropies = []
                    for loop_log_probs_full, loop_entropy_full in zip(per_loop_log_probs_full, per_loop_entropies_full):
                        # Slice to response portion: last response_length tokens (shifted by 1)
                        loop_log_probs = loop_log_probs_full[:, -response_length - 1:-1]
                        loop_entropy = loop_entropy_full[:, -response_length - 1:-1]
                        per_loop_log_probs.append(loop_log_probs)
                        per_loop_entropies.append(loop_entropy)

                    # Final loop values for compatibility
                    log_probs = per_loop_log_probs[-1]
                    entropy = per_loop_entropies[-1]

                    # Process exit_pdf if available
                    if exit_pdf is not None:
                        # Slice to response portion
                        exit_pdf = exit_pdf[:, -response_length - 1:-1, :]  # (bs, response_length, T)

                else:
                    # Fallback: use return_per_loop_logits (memory-intensive)
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_per_loop_logits=True,
                        return_exit_pdf=(self.loop_weighting == "exit_pdf"),
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    # Get per-loop logits from output (will process and delete sequentially)
                    per_loop_logits_list = output.per_loop_logits  # List of T tensors
                    exit_pdf = getattr(output, 'exit_pdf', None)

                    # Clear output reference to allow GC of other tensors
                    del output

                    # Compute per-loop log-probs sequentially to save memory
                    # Process each loop's logits immediately and delete to free GPU memory
                    per_loop_log_probs = []
                    per_loop_entropies = []

                    num_loops = len(per_loop_logits_list)
                    for i in range(num_loops):
                        # Pop the first logits tensor (removes from list, allowing GC)
                        loop_logits = per_loop_logits_list[i]
                        per_loop_logits_list[i] = None  # Allow GC of this tensor

                        # Apply temperature and slice to response portion
                        loop_logits = loop_logits / temperature
                        loop_logits = loop_logits[:, -response_length - 1:-1, :]  # (bs, response_length, vocab)

                        loop_log_probs = logprobs_from_logits(loop_logits, micro_batch['responses'])
                        loop_entropy = verl_F.entropy_from_logits(loop_logits)

                        # Delete logits immediately after computing log-probs
                        del loop_logits

                        per_loop_log_probs.append(loop_log_probs)
                        per_loop_entropies.append(loop_entropy)

                    # Clear the list entirely
                    del per_loop_logits_list

                    # Final loop values for compatibility
                    log_probs = per_loop_log_probs[-1]
                    entropy = per_loop_entropies[-1]

                    # Process exit_pdf if available
                    if exit_pdf is not None:
                        # Slice to response portion
                        exit_pdf = exit_pdf[:, -response_length - 1:-1, :]  # (bs, response_length, T)

            else:
                # Fallback: Run model T times with different total_ut_steps
                per_loop_log_probs = []
                per_loop_entropies = []
                exit_pdf = None

                for t in range(1, self.total_ut_steps + 1):
                    # Temporarily set total_ut_steps
                    if hasattr(self.actor_module, 'config'):
                        original_ut = getattr(self.actor_module.config, 'total_ut_steps', self.total_ut_steps)
                        self.actor_module.config.total_ut_steps = t

                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                    )

                    # Restore original
                    if hasattr(self.actor_module, 'config'):
                        self.actor_module.config.total_ut_steps = original_ut

                    logits = output.logits / temperature
                    logits = logits[:, -response_length - 1:-1, :]

                    loop_log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                    loop_entropy = verl_F.entropy_from_logits(logits)

                    per_loop_log_probs.append(loop_log_probs)
                    per_loop_entropies.append(loop_entropy)

                log_probs = per_loop_log_probs[-1]
                entropy = per_loop_entropies[-1]

            return entropy, log_probs, per_loop_log_probs, exit_pdf

    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> torch.Tensor:
        """Compute log probabilities with optional per-loop computation.

        For inference/rollout, we use the standard final-loop log-probs.
        Per-loop computation is only needed during training (update_policy).
        """
        import sys
        print(f"[RLTT DEBUG] compute_log_prob() called, calculate_entropy={calculate_entropy}", flush=True)
        sys.stdout.flush()
        # Use parent implementation for inference
        result = super().compute_log_prob(data, calculate_entropy=calculate_entropy)
        print(f"[RLTT DEBUG] compute_log_prob() completed", flush=True)
        sys.stdout.flush()
        return result

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        """Update policy using RLTT's multi-loop loss (Equation 5 from paper).

        L_RLTT(θ) = J_RLTT_PG(θ) + β * D_KL(π_θ || π_ref)

        Where:
        - J_RLTT_PG uses weighted sum of per-loop log-probs (Equation 6)
        - D_KL uses final loop only with DeepSeek/GRPO approximation (Equation 7)
        """
        import sys
        print(f"[RLTT DEBUG] update_policy() called, batch_size={data.batch.batch_size}", flush=True)
        sys.stdout.flush()

        self.actor_module.train()

        temperature = data.meta_info['temperature']

        # Select keys needed for RLTT objective
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'advantages', 'old_log_probs'  # old_log_probs is always available
        ]

        # ref_log_prob is only available when use_kl_loss is enabled in config
        # If not available, we'll use old_log_probs as the reference
        use_ref_log_prob = 'ref_log_prob' in data.batch.keys()
        if use_ref_log_prob:
            select_keys.append('ref_log_prob')

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Create dataloader
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}

        # Compute static loop weights once if not using exit_pdf or learned
        if self.loop_weighting not in ("exit_pdf", "learned"):
            static_loop_weights = compute_loop_weights(
                num_loops=self.total_ut_steps,
                strategy=self.loop_weighting,
                alpha=self.progressive_alpha,
                device=torch.cuda.current_device(),
            )
        else:
            static_loop_weights = None

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                # Split into micro-batches
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch_data.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch_data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch_data, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    # Prepare data
                    if isinstance(micro_data, DataProto):
                        micro_data = {**micro_data.batch.to(torch.cuda.current_device()), **micro_data.non_tensor_batch}
                    else:
                        micro_data = micro_data.to(torch.cuda.current_device())

                    responses = micro_data['responses']
                    response_length = responses.size(1)
                    attention_mask = micro_data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    advantages = micro_data['advantages']
                    entropy_coeff = self.config.entropy_coeff

                    # Forward pass with RLTT per-loop computation
                    entropy, final_log_prob, per_loop_log_probs, exit_pdf = self._forward_micro_batch_rltt(
                        micro_batch=micro_data, temperature=temperature
                    )

                    # Determine loop weights
                    if self.use_learned_weights and self.learned_weights_module is not None:
                        # Get learned weights (differentiable through softmax)
                        loop_weights = self.learned_weights_module()
                    elif self.loop_weighting == "exit_pdf" and exit_pdf is not None:
                        loop_weights = exit_pdf
                    else:
                        loop_weights = static_loop_weights

                    # Compute RLTT policy loss (pure REINFORCE, no probability ratios)
                    # This matches the paper's Equation 6:
                    # J_RLTT_PG(θ) = -E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t * log P^(t)(y_{i,j}) * Â_i ]
                    pg_loss, rltt_metrics = compute_rltt_policy_loss(
                        per_loop_log_probs=per_loop_log_probs,
                        advantages=advantages,
                        eos_mask=response_mask,
                        loop_weights=loop_weights,
                    )

                    # Compute entropy loss
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # Compute policy loss (starts with J_RLTT_PG from Equation 6)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    # KL regularization term (Equation 5 & 7 from RLTT paper)
                    # L_RLTT(θ) = J_RLTT_PG(θ) + β * D_KL(π_θ || π_ref)
                    #
                    # π_ref = frozen initial policy (NOT the rollout policy)
                    # This prevents drift from the SFT checkpoint
                    #
                    # From Equation 7: KL uses final loop P^(T_max) only
                    # Use ref_log_prob if available, otherwise fall back to old_log_probs
                    if use_ref_log_prob:
                        ref_log_prob = micro_data['ref_log_prob']
                    else:
                        ref_log_prob = micro_data['old_log_probs']

                    # Low-variance KL estimator (DeepSeek/HRPO style):
                    # D_KL(π_θ || π_ref) ≈ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
                    # This is equivalent to: ratio - log(ratio) - 1 where ratio = π_ref/π_θ
                    log_ratio = ref_log_prob - final_log_prob  # log(π_ref/π_θ)
                    per_token_kl = torch.exp(log_ratio) - log_ratio - 1
                    kl_loss = verl_F.masked_mean(per_token_kl, response_mask)

                    # Add KL penalty to loss (β * D_KL term from Equation 5)
                    policy_loss = policy_loss + self.config.kl_loss_coef * kl_loss

                    # Log KL metrics
                    rltt_metrics['actor/kl_loss'] = kl_loss.detach().item()
                    rltt_metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    # Scale loss for gradient accumulation
                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    # Log metrics
                    step_metrics = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                    }
                    step_metrics.update(rltt_metrics)

                    # Log learned weights if using them
                    if self.learned_weights_module is not None:
                        weight_dict = self.learned_weights_module.get_weights_for_logging()
                        for k, v in weight_dict.items():
                            step_metrics[f'rltt/{k}'] = v
                    append_to_dict(metrics, step_metrics)

                # Optimizer step
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
