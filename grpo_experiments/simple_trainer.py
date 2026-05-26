"""
Simple GRPO Trainer for single-GPU training.

This is a fallback trainer when verl is not available or for single-GPU setups.
It uses vLLM for accelerated generation when available.
"""
import os
import json
import logging
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


class SimpleGRPOTrainer:
    """Simple GRPO trainer for single-GPU training."""

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

        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
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
                project=args.wandb_project,
                name=f"simple-{'ppo' if getattr(args, 'use_ppo', False) else 'grpo'}-{os.path.basename(args.output_dir)}",
                config=vars(args),
                mode="offline" if os.environ.get("WANDB_MODE") == "offline" else "online",
            )

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

    def _compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for completions."""
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, :-1, :]  # Shift logits

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]  # Shift targets

            # Gather log probs for target tokens
            token_log_probs = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Mask to only completion tokens
            completion_mask_shifted = completion_mask[:, 1:]
            token_log_probs = token_log_probs * completion_mask_shifted

            return token_log_probs

    def _compute_grpo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        old_log_probs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute GRPO or PPO loss depending on args.use_ppo.

        GRPO: loss = -E[advantage * log_prob] + beta * KL
        PPO:  loss = -E[min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)] + beta * KL

        Where advantage is normalized within each prompt group (GRPO) or computed
        via GAE with a simple value baseline (PPO).
        """
        # Sum log probs over completion tokens
        completion_mask_shifted = completion_mask[:, 1:]
        policy_sum = (policy_log_probs * completion_mask_shifted).sum(dim=-1)
        ref_sum = (ref_log_probs * completion_mask_shifted).sum(dim=-1)

        # Compute KL divergence per sample
        kl = ref_sum - policy_sum

        use_ppo = getattr(self.args, "use_ppo", False)

        if use_ppo:
            # PPO: clipped surrogate objective
            # Use group-mean as a simple value baseline for GAE-like advantage
            batch_size = rewards.shape[0]
            num_groups = batch_size // self.num_generations
            rewards_reshaped = rewards.view(num_groups, self.num_generations)
            group_mean = rewards_reshaped.mean(dim=1, keepdim=True)
            group_std = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8
            advantages = ((rewards_reshaped - group_mean) / group_std).view(-1)

            clip_ratio = getattr(self.args, "ppo_clip_ratio", 0.2)

            if old_log_probs is not None:
                # Compute importance sampling ratio
                old_sum = (old_log_probs * completion_mask_shifted).sum(dim=-1)
                log_ratio = policy_sum - old_sum
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                policy_loss = torch.min(surr1, surr2).mean()
            else:
                # First epoch or no old_log_probs: standard policy gradient
                policy_loss = (advantages * policy_sum).mean()

            # Entropy bonus
            entropy_coeff = getattr(self.args, "entropy_coeff", 0.01)
            # Approximate entropy from policy log probs
            entropy = -(policy_log_probs * completion_mask_shifted).sum(dim=-1).mean()

            loss = -policy_loss + self.beta * kl.mean() - entropy_coeff * entropy
        else:
            # Standard GRPO: subtract mean and divide by std
            batch_size = rewards.shape[0]
            num_groups = batch_size // self.num_generations
            rewards_reshaped = rewards.view(num_groups, self.num_generations)
            group_mean = rewards_reshaped.mean(dim=1, keepdim=True)
            group_std = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8
            advantages = ((rewards_reshaped - group_mean) / group_std).view(-1)

            # GRPO loss: -advantage * log_prob + beta * KL
            loss = -(advantages * policy_sum).mean() + self.beta * kl.mean()

        return loss, kl.mean().item()

    def train(self):
        """Run GRPO training."""
        logger.info("Starting GRPO training...")

        # Log training mode prominently
        if self.args.no_lora:
            logger.info("=" * 60)
            logger.info("TRAINING MODE: FULL PARAMETER FINE-TUNING")
            logger.info("All model parameters will be updated during training")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("TRAINING MODE: LoRA FINE-TUNING")
            logger.info(f"LoRA rank: {self.args.lora_r}, alpha: {self.args.lora_alpha}")
            logger.info(f"Target modules: {self.args.lora_target_modules}")
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()
            logger.info("=" * 60)

        logger.info(f"  Total epochs: {self.args.num_train_epochs}")
        logger.info(f"  Prompts per batch: {self.num_prompts_per_batch}")
        logger.info(f"  Generations per prompt: {self.num_generations}")
        logger.info(f"  Total samples per step: {self.num_prompts_per_batch * self.num_generations}")

        for epoch in range(self.args.num_train_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.args.num_train_epochs}")

            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_kl = 0.0
            num_batches = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                loss, metrics = self._training_step(batch)

                epoch_loss += loss
                epoch_reward += metrics["reward"]
                epoch_kl += metrics["kl"]
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "reward": f"{metrics['reward']:.2%}",
                    "kl": f"{metrics['kl']:.4f}",
                })

                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss,
                        "train/reward": metrics["reward"],
                        "train/kl": metrics["kl"],
                        "train/step": self.global_step,
                    }, step=self.global_step)

                self.global_step += 1

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_reward = epoch_reward / num_batches
            avg_kl = epoch_kl / num_batches

            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            logger.info(f"  Average reward: {avg_reward:.2%}")
            logger.info(f"  Average KL: {avg_kl:.4f}")

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

        # Generate completions
        completions = self._generate_completions(prompts)

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

        # Compute log probs
        self.model.train()
        policy_log_probs = self._compute_log_probs(
            self.model,
            inputs["input_ids"],
            inputs["attention_mask"],
            completion_mask,
        )

        with torch.no_grad():
            ref_log_probs = self._compute_log_probs(
                self.ref_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                completion_mask,
            )

        use_ppo = getattr(self.args, "use_ppo", False)
        ppo_epochs = getattr(self.args, "ppo_epochs", 4) if use_ppo else 1

        total_loss = 0.0
        total_kl = 0.0

        # Store old log probs for PPO importance sampling (detached)
        old_log_probs = policy_log_probs.detach().clone() if use_ppo else None

        for ppo_epoch in range(ppo_epochs):
            if ppo_epoch > 0:
                # Recompute policy log probs for subsequent PPO epochs
                self.model.train()
                policy_log_probs = self._compute_log_probs(
                    self.model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    completion_mask,
                )

            # Compute loss (GRPO or PPO)
            loss, kl = self._compute_grpo_loss(
                policy_log_probs,
                ref_log_probs,
                rewards,
                completion_mask,
                old_log_probs=old_log_probs,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_kl += kl

        self.scheduler.step()

        metrics = {
            "reward": rewards.mean().item(),
            "kl": total_kl / ppo_epochs,
        }

        return total_loss / ppo_epochs, metrics

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
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))

        logger.info(f"Saved checkpoint to: {checkpoint_dir}")
