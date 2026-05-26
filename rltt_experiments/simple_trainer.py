"""
Simple RLTT (Reward Latent Thought Trajectories) Trainer for single-GPU training.

This implements the RLTT algorithm from the paper, which distributes credit
across the entire latent thought trajectory of looped language models.

Key differences from GRPO:
- Instead of only using log P^(T_max) (final loop), RLTT uses a weighted sum
  of log-probabilities across ALL loops: Σ ω_t * log P^(t)
- This provides denser credit assignment to intermediate latent computations

RLTT Loss (Equation 5 from paper):
    L_RLTT(θ) = J_RLTT_PG(θ) + β * D_KL(π_θ || π_ref)

where:
    J_RLTT_PG(θ) = -E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t * log P^(t)(y_{i,j} | x, y_{<j}) * Â_i ]

This trainer supports three modes for obtaining per-loop information:
1. "native": Use the RLTT-modified Ouro model that exposes per-loop logits
2. "exit_pdf": Use the native exit probability distribution as loop weights
3. "fallback": Fall back to final loop only (equivalent to GRPO)

This is a fallback trainer when verl is not available or for single-GPU setups.
It uses vLLM for accelerated generation when available.
"""
import os
import json
import logging
import time
from typing import Optional, List, Callable, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Import vLLM (optional)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ParquetDataset(Dataset):
    """Dataset for loading parquet files."""

    def __init__(self, parquet_path: str, tokenizer, max_prompt_length: int = 1024):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
            "data_source": row.get("data_source", "math"),
            "index": row.get("extra_info", {}).get("index", idx) if isinstance(row.get("extra_info"), dict) else idx,
        }


def compute_loop_weights(total_ut_steps: int, weighting: str = "uniform", alpha: float = 1.0) -> torch.Tensor:
    """Compute loop weights for RLTT.

    Args:
        total_ut_steps: Total number of loops (T_max)
        weighting: "uniform", "progressive", "exit_pdf" (dynamic), or "learned"
        alpha: Exponent for progressive weighting (only used if weighting="progressive")

    Returns:
        Tensor of shape (total_ut_steps,) with weights summing to 1
    """
    if weighting == "uniform":
        # Option A from paper: ω_t = 1/T_max
        weights = torch.ones(total_ut_steps) / total_ut_steps
    elif weighting == "progressive":
        # Option B from paper: ω_t = t^α / Σ s^α
        t = torch.arange(1, total_ut_steps + 1, dtype=torch.float32)
        weights = t ** alpha
        weights = weights / weights.sum()
    elif weighting == "exit_pdf":
        # Dynamic weighting based on exit probability - computed per-sample
        # Return uniform as placeholder; actual weights come from model
        weights = torch.ones(total_ut_steps) / total_ut_steps
    elif weighting == "learned":
        # Learned weights - return uniform as initialization
        # Actual learned weights are managed by LearnedLoopWeights module
        weights = torch.ones(total_ut_steps) / total_ut_steps
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting}")

    return weights


class LearnedLoopWeights(torch.nn.Module):
    """Learnable loop weights for RLTT.

    This module maintains trainable logits that are converted to normalized
    weights via softmax. The weights are jointly optimized with the policy.

    Initialization options:
    - "uniform": Start with equal weights (default)
    - "progressive": Start with progressive weights favoring later loops
    """

    def __init__(
        self,
        total_ut_steps: int,
        init_strategy: str = "uniform",
        init_alpha: float = 1.0,
        temperature: float = 1.0,
    ):
        """Initialize learned loop weights.

        Args:
            total_ut_steps: Number of loops (T_max)
            init_strategy: How to initialize weights ("uniform" or "progressive")
            init_alpha: Alpha for progressive initialization
            temperature: Softmax temperature (higher = more uniform)
        """
        super().__init__()
        self.total_ut_steps = total_ut_steps
        self.temperature = temperature

        # Initialize logits based on strategy
        if init_strategy == "uniform":
            # Uniform initialization: all logits equal
            init_logits = torch.zeros(total_ut_steps)
        elif init_strategy == "progressive":
            # Progressive initialization: favor later loops
            t = torch.arange(1, total_ut_steps + 1, dtype=torch.float32)
            target_weights = t ** init_alpha
            target_weights = target_weights / target_weights.sum()
            # Convert to logits (inverse softmax)
            init_logits = torch.log(target_weights + 1e-8)
        else:
            init_logits = torch.zeros(total_ut_steps)

        # Learnable logits
        self.logits = torch.nn.Parameter(init_logits)

    def forward(self) -> torch.Tensor:
        """Get normalized weights via softmax.

        Returns:
            weights: Tensor of shape (total_ut_steps,) summing to 1
        """
        return torch.softmax(self.logits / self.temperature, dim=0)

    def get_weights_for_logging(self) -> Dict[str, float]:
        """Get current weights as a dict for logging."""
        with torch.no_grad():
            weights = self.forward()
            return {f"loop_{t+1}_weight": w.item() for t, w in enumerate(weights)}


class SimpleRLTTTrainer:
    """Simple RLTT trainer for single-GPU training.

    Implements the RLTT algorithm which distributes credit across all
    latent thought loops of a looped language model.

    Supports four loop weighting modes:
    - "uniform": Equal weight for all loops (Option A from paper)
    - "progressive": Later loops get more weight (Option B from paper)
    - "exit_pdf": Use the model's native exit probability distribution
    - "learned": Jointly learn the weights during training
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        train_parquet: str,
        reward_func: Callable,
        args,
        log_rollouts: Optional[Callable] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.args = args
        self.log_rollouts = log_rollouts

        # Device
        self.device = next(model.parameters()).device

        # Training params
        self.beta = args.beta
        self.num_generations = args.num_generations
        self.num_prompts_per_batch = args.num_prompts_per_batch
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature

        # RLTT-specific params
        self.total_ut_steps = args.total_ut_steps
        self.loop_weighting = getattr(args, 'loop_weighting', 'uniform')
        self.progressive_alpha = getattr(args, 'progressive_alpha', 1.0)
        self.use_exit_pdf = (self.loop_weighting == "exit_pdf")
        self.use_learned_weights = (self.loop_weighting == "learned")

        # Check if model supports RLTT native per-loop outputs
        self.model_supports_rltt = self._check_model_rltt_support()

        # Initialize loop weights based on strategy
        self.learned_weights_module = None
        if self.use_learned_weights:
            # Create learnable weights module
            learned_init = getattr(args, 'learned_weights_init', 'uniform')
            learned_temp = getattr(args, 'learned_weights_temp', 1.0)
            self.learned_weights_module = LearnedLoopWeights(
                total_ut_steps=self.total_ut_steps,
                init_strategy=learned_init,
                init_alpha=self.progressive_alpha,
                temperature=learned_temp,
            ).to(self.device)
            self.loop_weights = None  # Will be computed dynamically
            logger.info(f"RLTT using LEARNED weights (init={learned_init}, temp={learned_temp})")
            logger.info(f"  Initial weights: {self.learned_weights_module.get_weights_for_logging()}")
        elif self.use_exit_pdf:
            self.loop_weights = None
            logger.info("RLTT using dynamic exit_pdf weights from model")
        else:
            # Static weights (uniform or progressive)
            self.loop_weights = compute_loop_weights(
                self.total_ut_steps,
                self.loop_weighting,
                self.progressive_alpha
            ).to(self.device)
            logger.info(f"RLTT static loop weights ({self.loop_weighting}): {self.loop_weights.tolist()}")

        # Create dataset
        self.train_dataset = ParquetDataset(
            train_parquet,
            tokenizer,
            max_prompt_length=args.max_prompt_length,
        )

        # Create dataloader (one prompt at a time, we'll generate multiple samples)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.num_prompts_per_batch,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer - include learned weights if using learned weighting
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        if self.learned_weights_module is not None:
            # Add learned weights with potentially different learning rate
            learned_weights_lr = getattr(args, 'learned_weights_lr', args.learning_rate * 10)
            param_groups = [
                {'params': params_to_optimize, 'lr': args.learning_rate},
                {'params': self.learned_weights_module.parameters(), 'lr': learned_weights_lr, 'weight_decay': 0.0},
            ]
            self.optimizer = AdamW(
                param_groups,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.99),
            )
            logger.info(f"Optimizer includes learned weights (lr={learned_weights_lr})")
        else:
            self.optimizer = AdamW(
                params_to_optimize,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.99),
            )

        # Scheduler
        total_steps = len(self.train_dataloader) * args.num_train_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        # Initialize vLLM if available
        self.vllm_engine = None
        if VLLM_AVAILABLE and not args.no_vllm:
            self._init_vllm()

        # Metrics
        self.global_step = 0
        self.metrics = defaultdict(list)

        # Wandb
        self.use_wandb = WANDB_AVAILABLE and not args.no_wandb
        if self.use_wandb:
            wandb.init(
                project=getattr(args, 'wandb_project', 'rltt-ouro-math'),
                name=f"simple-rltt-{os.path.basename(args.output_dir)}",
                config=vars(args),
                mode="offline" if os.environ.get("WANDB_MODE") == "offline" else "online",
            )

    def _check_model_rltt_support(self) -> bool:
        """Check if the model supports RLTT native per-loop output."""
        # Check if the model has the RLTT-specific forward signature
        # by looking for return_per_loop_logits parameter
        import inspect

        model_to_check = self.model
        if hasattr(model_to_check, 'base_model'):
            model_to_check = model_to_check.base_model

        if hasattr(model_to_check, 'forward'):
            sig = inspect.signature(model_to_check.forward)
            if 'return_per_loop_logits' in sig.parameters:
                logger.info("Model supports RLTT native per-loop logits")
                return True

        logger.info("Model does not support native per-loop logits, will attempt extraction")
        return False

    def _init_vllm(self):
        """Initialize vLLM engine."""
        logger.info("Initializing vLLM engine...")
        try:
            self.vllm_engine = LLM(
                model=self.args.model_path,
                tokenizer=self.args.model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                max_model_len=self.max_prompt_length + self.max_completion_length,
                tensor_parallel_size=self.args.vllm_tensor_parallel_size,
                enforce_eager=self.args.vllm_enforce_eager,
            )
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM: {e}")
            logger.warning("Falling back to HuggingFace generation")
            self.vllm_engine = None

    def _generate_completions(self, prompts: List[str]) -> List[str]:
        """Generate completions for prompts."""
        if self.vllm_engine is not None:
            return self._generate_with_vllm(prompts)
        else:
            return self._generate_with_hf(prompts)

    def _generate_with_vllm(self, prompts: List[str]) -> List[str]:
        """Generate completions using vLLM."""
        # Repeat each prompt num_generations times
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.num_generations)

        sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=1.0,
        )

        outputs = self.vllm_engine.generate(expanded_prompts, sampling_params)
        completions = [output.outputs[0].text for output in outputs]

        return completions

    def _generate_with_hf(self, prompts: List[str]) -> List[str]:
        """Generate completions using HuggingFace."""
        # Repeat each prompt num_generations times
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.num_generations)

        completions = []
        batch_size = 4  # Small batch size for HF generation

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(expanded_prompts), batch_size):
                batch_prompts = expanded_prompts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                ).to(self.device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode
                for j, output in enumerate(outputs):
                    input_len = inputs["input_ids"][j].shape[0]
                    completion = self.tokenizer.decode(
                        output[input_len:],
                        skip_special_tokens=True,
                    )
                    completions.append(completion)

        self.model.train()
        return completions

    def _compute_multi_loop_log_probs_native(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        is_ref: bool = False,
    ) -> tuple:
        """Compute log probabilities using native RLTT model support.

        Uses the model's return_per_loop_logits=True to get logits from each loop.

        Returns:
            Tuple of:
            - per_loop_log_probs: [batch_size, seq_len-1, total_ut_steps]
            - final_log_probs: [batch_size, seq_len-1]
            - exit_pdf: [batch_size, seq_len, total_ut_steps] or None
        """
        context = torch.no_grad() if is_ref else torch.enable_grad()

        with context:
            # Call with RLTT-specific flags
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_per_loop_logits=True,
                return_exit_pdf=self.use_exit_pdf,
            )

            target_ids = input_ids[:, 1:]
            completion_mask_shifted = completion_mask[:, 1:]

            # Get per-loop log probs
            per_loop_logits = outputs.per_loop_logits  # List of [batch, seq, vocab]

            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] - 1
            total_ut_steps = len(per_loop_logits)

            per_loop_log_probs = torch.zeros(
                batch_size, seq_len, total_ut_steps,
                device=input_ids.device, dtype=torch.float32
            )

            for t, logits_t in enumerate(per_loop_logits):
                logits_t_shifted = logits_t[:, :-1, :]
                log_probs_t = F.log_softmax(logits_t_shifted, dim=-1)
                token_log_probs_t = log_probs_t.gather(
                    dim=-1, index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                per_loop_log_probs[:, :, t] = token_log_probs_t * completion_mask_shifted

            # Final loop log probs for KL
            final_log_probs = per_loop_log_probs[:, :, -1]

            # Exit PDF if requested
            exit_pdf = getattr(outputs, 'exit_pdf', None)

            return per_loop_log_probs, final_log_probs, exit_pdf

    def _compute_multi_loop_log_probs_fallback(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        is_ref: bool = False,
    ) -> tuple:
        """Compute log probabilities using fallback method.

        For models that don't support native RLTT, this attempts to extract
        per-loop information from hidden states, or falls back to final loop only.

        Returns:
            Tuple of:
            - per_loop_log_probs: [batch_size, seq_len-1, total_ut_steps]
            - final_log_probs: [batch_size, seq_len-1]
            - exit_pdf: None (not available in fallback mode)
        """
        context = torch.no_grad() if is_ref else torch.enable_grad()

        with context:
            # Standard forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            target_ids = input_ids[:, 1:]
            completion_mask_shifted = completion_mask[:, 1:]

            # Get final logits
            logits = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            final_log_probs = token_log_probs * completion_mask_shifted

            if is_ref:
                # Reference model only needs final loop
                return final_log_probs.unsqueeze(-1), final_log_probs, None

            # Try to extract per-loop hidden states for policy model
            per_loop_hidden = None

            # Check if the Ouro model's internal structure is accessible
            # The OuroModel.forward returns (outputs, hidden_states_list, gate_list)
            # but through OuroForCausalLM it's wrapped differently

            # Try accessing hidden_states from different locations
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                all_hidden_states = outputs.hidden_states

                # Get model config
                config = model.config if hasattr(model, 'config') else None
                if config is None and hasattr(model, 'base_model'):
                    config = model.base_model.config

                if config is not None:
                    total_ut_steps = getattr(config, 'total_ut_steps', self.total_ut_steps)
                    num_layers = getattr(config, 'num_hidden_layers', 1)

                    # Check if hidden states match expected structure
                    # hidden_states[0] = embeddings
                    # For looped model: hidden_states has 1 + num_layers * total_ut_steps entries
                    expected_len = 1 + num_layers * total_ut_steps
                    if len(all_hidden_states) == expected_len:
                        per_loop_hidden = []
                        for t in range(total_ut_steps):
                            # Get the final layer's output for loop t
                            layer_idx = (t + 1) * num_layers
                            per_loop_hidden.append(all_hidden_states[layer_idx])

            if per_loop_hidden is not None:
                # Get lm_head
                lm_head = None
                if hasattr(model, 'lm_head'):
                    lm_head = model.lm_head
                elif hasattr(model, 'base_model') and hasattr(model.base_model, 'lm_head'):
                    lm_head = model.base_model.lm_head

                if lm_head is not None:
                    batch_size = input_ids.shape[0]
                    seq_len = input_ids.shape[1] - 1
                    total_ut_steps = len(per_loop_hidden)

                    per_loop_log_probs = torch.zeros(
                        batch_size, seq_len, total_ut_steps,
                        device=input_ids.device, dtype=torch.float32
                    )

                    for t, hidden_t in enumerate(per_loop_hidden):
                        logits_t = lm_head(hidden_t)[:, :-1, :]
                        log_probs_t = F.log_softmax(logits_t, dim=-1)
                        token_log_probs_t = log_probs_t.gather(
                            dim=-1, index=target_ids.unsqueeze(-1)
                        ).squeeze(-1)
                        per_loop_log_probs[:, :, t] = token_log_probs_t * completion_mask_shifted

                    return per_loop_log_probs, per_loop_log_probs[:, :, -1], None

            # Fallback: use final loop for all
            logger.warning(
                "Could not extract per-loop hidden states. "
                "RLTT will use final loop logits only (equivalent to GRPO). "
                "For full RLTT, use the RLTT-modified Ouro model."
            )
            per_loop_log_probs = final_log_probs.unsqueeze(-1).expand(-1, -1, self.total_ut_steps)
            return per_loop_log_probs, final_log_probs, None

    def _compute_multi_loop_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        is_ref: bool = False,
    ) -> tuple:
        """Compute log probabilities for all loops in the latent thought trajectory.

        This is the core of RLTT: instead of only computing log P^(T_max),
        we compute log P^(t) for t = 1, ..., T_max and return all of them.

        Returns:
            Tuple of:
            - per_loop_log_probs: [batch_size, seq_len-1, total_ut_steps]
            - final_log_probs: [batch_size, seq_len-1]
            - exit_pdf: [batch_size, seq_len, total_ut_steps] or None
        """
        if self.model_supports_rltt:
            return self._compute_multi_loop_log_probs_native(
                model, input_ids, attention_mask, completion_mask, is_ref
            )
        else:
            return self._compute_multi_loop_log_probs_fallback(
                model, input_ids, attention_mask, completion_mask, is_ref
            )

    def _compute_rltt_loss(
        self,
        per_loop_log_probs: torch.Tensor,
        policy_final_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        exit_pdf: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute RLTT loss.

        RLTT loss (Equation 5 from paper):
            L_RLTT(θ) = -J_RLTT_PG(θ) + β * D_KL(π_θ || π_ref)

        where J_RLTT_PG (Equation 6) uses weighted sum over loops:
            J_RLTT_PG = E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t * log P^(t)(y_{i,j}) * Â_i ]

        And D_KL (Equation 7) uses only the final loop:
            D_KL = E[ (1/g) Σ_i (1/|y_i|) Σ_j KL[P^(T_max) || P^(T_max)_ref] ]

        Args:
            per_loop_log_probs: [batch_size, seq_len-1, total_ut_steps] log probs for each loop
            policy_final_log_probs: [batch_size, seq_len-1] final loop log probs for KL
            ref_log_probs: [batch_size, seq_len-1] reference model log probs
            rewards: [batch_size] rewards for each sample
            completion_mask: [batch_size, seq_len] mask for completion tokens
            exit_pdf: [batch_size, seq_len, total_ut_steps] exit probability distribution (optional)

        Returns:
            Tuple of (loss, kl_value)
        """
        completion_mask_shifted = completion_mask[:, 1:]

        # Determine loop weights
        if self.use_learned_weights and self.learned_weights_module is not None:
            # Use learned weights (differentiable through softmax)
            learned_weights = self.learned_weights_module()  # [T_max]
            loop_weights = learned_weights.view(1, 1, -1)  # [1, 1, T_max]
        elif self.use_exit_pdf and exit_pdf is not None:
            # Use per-sample, per-token exit PDF as weights
            # exit_pdf: [batch, seq, T_max]
            # Need to align with completion_mask_shifted: [batch, seq-1]
            exit_pdf_shifted = exit_pdf[:, :-1, :]  # [batch, seq-1, T_max]
            loop_weights = exit_pdf_shifted  # Dynamic per-token weights
        else:
            # Use static weights
            loop_weights = self.loop_weights.view(1, 1, -1)  # [1, 1, T_max]

        # Compute weighted sum of log probs across loops (Equation 6)
        # per_loop_log_probs: [batch, seq-1, T_max]
        # loop_weights: [1, 1, T_max] or [batch, seq-1, T_max]
        weighted_log_probs = (per_loop_log_probs * loop_weights).sum(dim=-1)

        # Sum over sequence (completion tokens only)
        policy_sum = (weighted_log_probs * completion_mask_shifted).sum(dim=-1)

        # Reference log probs sum (for KL, uses final loop)
        ref_sum = (ref_log_probs * completion_mask_shifted).sum(dim=-1)

        # Final loop policy sum (for KL)
        policy_final_sum = (policy_final_log_probs * completion_mask_shifted).sum(dim=-1)

        # Compute KL divergence per sample (using final loop only, as in Equation 7)
        kl = ref_sum - policy_final_sum

        # Normalize rewards within groups (GRPO-style)
        batch_size = rewards.shape[0]
        num_groups = batch_size // self.num_generations
        rewards_reshaped = rewards.view(num_groups, self.num_generations)

        # Compute group mean and std
        group_mean = rewards_reshaped.mean(dim=1, keepdim=True)
        group_std = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8

        # Normalize advantages
        advantages = (rewards_reshaped - group_mean) / group_std
        advantages = advantages.view(-1)

        # RLTT loss: -advantage * weighted_log_prob + beta * KL
        loss = -(advantages * policy_sum).mean() + self.beta * kl.mean()

        return loss, kl.mean().item()

    def train(self):
        """Run RLTT training."""
        logger.info("Starting RLTT training...")
        logger.info(f"  Total epochs: {self.args.num_train_epochs}")
        logger.info(f"  Prompts per batch: {self.num_prompts_per_batch}")
        logger.info(f"  Generations per prompt: {self.num_generations}")
        logger.info(f"  Total samples per step: {self.num_prompts_per_batch * self.num_generations}")
        logger.info(f"  Loop weighting: {self.loop_weighting}")
        logger.info(f"  Total UT steps: {self.total_ut_steps}")
        logger.info(f"  Model supports native RLTT: {self.model_supports_rltt}")
        if not self.use_exit_pdf:
            logger.info(f"  Static loop weights: {self.loop_weights.tolist()}")
        else:
            logger.info(f"  Using dynamic exit_pdf weights")

        for epoch in range(self.args.num_train_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_train_epochs}")

            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_kl = 0.0
            epoch_rollout_time = 0.0
            epoch_backprop_time = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                loss, metrics = self._training_step(batch)

                epoch_loss += loss
                epoch_reward += metrics["reward"]
                epoch_kl += metrics["kl"]
                epoch_rollout_time += metrics["rollout_time"]
                epoch_backprop_time += metrics["backprop_time"]
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "reward": f"{metrics['reward']:.2%}",
                    "rollout": f"{metrics['rollout_time']:.1f}s",
                    "backprop": f"{metrics['backprop_time']:.1f}s",
                })

                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        "train/loss": loss,
                        "train/reward": metrics["reward"],
                        "train/kl": metrics["kl"],
                        "train/step": self.global_step,
                        "timing/rollout_time": metrics["rollout_time"],
                        "timing/backprop_time": metrics["backprop_time"],
                    }
                    # Log learned weights if using them
                    if self.learned_weights_module is not None:
                        weight_dict = self.learned_weights_module.get_weights_for_logging()
                        for k, v in weight_dict.items():
                            log_dict[f"rltt/{k}"] = v
                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_reward = epoch_reward / num_batches
            avg_kl = epoch_kl / num_batches
            total_rollout_time = epoch_rollout_time
            total_backprop_time = epoch_backprop_time

            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            logger.info(f"  Average reward: {avg_reward:.2%}")
            logger.info(f"  Average KL: {avg_kl:.4f}")
            logger.info(f"  Total rollout time: {total_rollout_time:.1f}s")
            logger.info(f"  Total backprop time: {total_backprop_time:.1f}s")
            if self.learned_weights_module is not None:
                logger.info(f"  Learned weights: {self.learned_weights_module.get_weights_for_logging()}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_steps == 0:
                self._save_checkpoint(epoch + 1)

        # Final save
        self._save_checkpoint("final")
        logger.info("Training complete!")

    def _training_step(self, batch: Dict[str, Any]) -> tuple:
        """Run a single training step."""
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]

        # Generate completions (rollout)
        rollout_start = time.perf_counter()
        completions = self._generate_completions(prompts)
        rollout_time = time.perf_counter() - rollout_start

        # Expand ground truths to match completions
        expanded_ground_truths = []
        expanded_prompts = []
        for prompt, gt in zip(prompts, ground_truths):
            expanded_ground_truths.extend([gt] * self.num_generations)
            expanded_prompts.extend([prompt] * self.num_generations)

        # Compute rewards
        rewards = self.reward_func(
            prompts=expanded_prompts,
            completions=completions,
            answer=expanded_ground_truths,
        )
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Log rollouts
        if self.log_rollouts is not None:
            self.log_rollouts(
                step=self.global_step,
                prompts=expanded_prompts,
                completions=completions,
                rewards=rewards.tolist(),
                answers=expanded_ground_truths,
            )

        # Tokenize prompt + completion
        full_texts = [p + c for p, c in zip(expanded_prompts, completions)]

        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length + self.max_completion_length,
        ).to(self.device)

        # Create completion mask (1 for completion tokens, 0 for prompt tokens)
        prompt_inputs = self.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )

        completion_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.float32)
        for i in range(len(expanded_prompts)):
            prompt_len = prompt_inputs["attention_mask"][i].sum().item()
            total_len = inputs["attention_mask"][i].sum().item()
            completion_mask[i, prompt_len:total_len] = 1.0

        # Start timing backprop (forward passes + loss + backward + optimizer)
        backprop_start = time.perf_counter()

        # Compute multi-loop log probs for policy model
        self.model.train()
        per_loop_log_probs, policy_final_log_probs, exit_pdf = self._compute_multi_loop_log_probs(
            self.model,
            inputs["input_ids"],
            inputs["attention_mask"],
            completion_mask,
            is_ref=False,
        )

        # Compute log probs for reference model (final loop only)
        with torch.no_grad():
            ref_log_probs, _, _ = self._compute_multi_loop_log_probs(
                self.ref_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                completion_mask,
                is_ref=True,
            )

        # Compute RLTT loss
        loss, kl = self._compute_rltt_loss(
            per_loop_log_probs,
            policy_final_log_probs,
            ref_log_probs,
            rewards,
            completion_mask,
            exit_pdf=exit_pdf,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.args.max_grad_norm,
        )
        self.optimizer.step()
        self.scheduler.step()

        backprop_time = time.perf_counter() - backprop_start

        metrics = {
            "reward": rewards.mean().item(),
            "kl": kl,
            "rollout_time": rollout_time,
            "backprop_time": backprop_time,
        }

        return loss.item(), metrics

    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model (LoRA adapter if using PEFT)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        # Save learned weights if using them
        if self.learned_weights_module is not None:
            state["learned_weights_state_dict"] = self.learned_weights_module.state_dict()
            state["learned_weights_values"] = self.learned_weights_module.get_weights_for_logging()
            logger.info(f"  Learned weights: {state['learned_weights_values']}")

        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

        logger.info(f"Saved checkpoint to: {checkpoint_dir}")
