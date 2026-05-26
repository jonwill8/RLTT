"""
Simple SFT Trainer for single-process training.

This trainer uses a single process (no torchrun) to avoid conflicts between
PyTorch distributed and vLLM's distributed setup during validation.

Matches the GRPO training approach for consistency.
"""
import os
import sys
import json
import logging
import gc
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized math utilities
from math_utils import (
    extract_boxed_answer,
    check_math_answer,
    get_gold_answer,
    format_math_prompt,
    build_chat_messages,
    FEW_SHOT_EXAMPLES,
    INSTRUCTION,
    MATH_VERIFY_AVAILABLE,
)

# Optional imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Dataset
# ============================================================================

class SFTDataset(Dataset):
    """Simple SFT dataset for loading parquet files."""

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_length: int = 2048,
        prompt_key: str = "prompt",
        response_key: str = "response",
    ):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.response_key = response_key

        logger.info(f"Loaded {len(self.df)} examples from {parquet_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row[self.prompt_key]
        response = row[self.response_key]

        # Apply chat template to prompt
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_str = response + self.tokenizer.eos_token

        # Tokenize
        prompt_ids = self.tokenizer(
            prompt_str,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        response_ids = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        # Combine
        input_ids = torch.cat([prompt_ids, response_ids])
        prompt_length = len(prompt_ids)

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # Create labels (mask prompt tokens with -100)
        labels = input_ids.clone()
        labels[:prompt_length] = -100

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id: int):
    """Collate function with dynamic padding."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        # Pad on the right
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        attention_mask.append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        labels.append(F.pad(item["labels"], (0, pad_len), value=-100))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# ============================================================================
# Validation utilities
# ============================================================================

def load_validation_dataset(data_path: str, max_samples: Optional[int] = None) -> list:
    """Load validation dataset from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    return data


# ============================================================================
# Simple SFT Trainer
# ============================================================================

class SimpleSFTTrainer:
    """Simple SFT trainer for single-process training."""

    def __init__(
        self,
        model,
        tokenizer,
        train_parquet: str,
        val_parquet: str,
        args,
        validation_jsonl: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training params
        self.max_length = args.max_seq_length
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.warmup_ratio = args.warmup_ratio
        self.num_epochs = args.num_train_epochs
        self.batch_size = args.micro_batch_size
        # Use explicit gradient_accumulation_steps if provided, otherwise calculate from batch sizes
        if getattr(args, 'gradient_accumulation_steps', None) is not None:
            self.gradient_accumulation_steps = args.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = args.train_batch_size // args.micro_batch_size

        # Create datasets
        self.train_dataset = SFTDataset(
            train_parquet,
            tokenizer,
            max_length=self.max_length,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
        )

        self.val_dataset = SFTDataset(
            val_parquet,
            tokenizer,
            max_length=self.max_length,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
        )

        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            pin_memory=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            pin_memory=True,
        )

        # Calculate steps
        self.steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        self.max_steps = getattr(args, 'max_steps', None)
        if self.max_steps is not None and self.max_steps > 0:
            self.total_steps = self.max_steps
        else:
            self.total_steps = self.steps_per_epoch * self.num_epochs
            self.max_steps = None  # Disable max_steps limit
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)

        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Validation dataset (JSONL for generation evaluation)
        self.validation_data = None
        if validation_jsonl and os.path.exists(validation_jsonl):
            self.validation_data = load_validation_dataset(
                validation_jsonl,
                args.math500_samples
            )
            logger.info(f"Loaded {len(self.validation_data)} validation samples for generation eval")

        # vLLM for validation
        self.vllm_engine = None
        self.use_vllm = args.use_vllm and VLLM_AVAILABLE

        # Detect if model is using LoRA (PEFT model)
        self.is_peft_model = hasattr(model, 'peft_config') or hasattr(model, 'active_adapter')

        # Metrics
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.best_step = 0

        # Wandb
        self.use_wandb = WANDB_AVAILABLE and not args.no_wandb
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"sft-{os.path.basename(args.output_dir)}",
                config=vars(args),
                mode="offline" if os.environ.get("WANDB_MODE") == "offline" else "online",
            )

        # CSV logging
        self.csv_file = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            self.csv_file = os.path.join(args.output_dir, "training_log.csv")
            with open(self.csv_file, 'w') as f:
                f.write("step,epoch,loss,val_loss,lr\n")

        logger.info(f"SimpleSFTTrainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Train samples: {len(self.train_dataset)}")
        logger.info(f"  Val samples: {len(self.val_dataset)}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"  Total steps: {self.total_steps}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        return LambdaLR(self.optimizer, lr_lambda)

    def _init_vllm_engine(self, model_path: str = None):
        """Initialize vLLM engine for validation.

        Args:
            model_path: Path to model to load. If None, uses base model with LoRA.
                       If specified, loads the full model from that path.
        """
        if self.vllm_engine is not None:
            return

        # Determine which model to load and whether to enable LoRA
        if model_path is not None:
            # Full fine-tuning: load the saved full model directly
            load_path = model_path
            enable_lora = False
            logger.info(f"Initializing vLLM engine with full model from {load_path}...")
        else:
            # LoRA: load base model with LoRA support
            load_path = self.args.model_path
            enable_lora = True
            logger.info(f"Initializing vLLM engine with LoRA support from {load_path}...")

        vllm_kwargs = {
            "model": load_path,
            "tokenizer": load_path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
            "tensor_parallel_size": getattr(self.args, 'vllm_tensor_parallel_size', 1),
            "max_model_len": self.args.max_prompt_length + self.args.max_new_tokens,
        }

        if enable_lora:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64

        self.vllm_engine = LLM(**vllm_kwargs)
        logger.info("vLLM engine initialized")

    def _cleanup_vllm_engine(self):
        """Cleanup vLLM engine."""
        if self.vllm_engine is not None:
            logger.info("Cleaning up vLLM engine...")
            del self.vllm_engine
            self.vllm_engine = None
            torch.cuda.empty_cache()
            gc.collect()

    def train(self):
        """Run SFT training."""
        logger.info("Starting SFT training...")
        logger.info(f"  Total epochs: {self.num_epochs}")
        logger.info(f"  Max steps: {self.max_steps if self.max_steps else 'None (use epochs)'}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Learning rate: {self.learning_rate}")

        # Run baseline validation before training starts (step 0)
        skip_initial = getattr(self.args, 'no_initial_validation', False) or getattr(self.args, 'no_validation', False)
        if self.validation_data is not None and not skip_initial:
            logger.info("=" * 60)
            logger.info("Running baseline validation before training (step 0)...")
            logger.info("=" * 60)
            val_acc = self._run_validation_generation()
            if val_acc is not None:
                logger.info(f"Baseline MATH-500 accuracy: {val_acc:.2%}")
                if self.use_wandb:
                    wandb.log({"val/math500_accuracy": val_acc}, step=0)

        self.model.train()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0.0
            num_batches = 0
            self.optimizer.zero_grad()

            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                total=len(self.train_dataloader)
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

                # Optimizer step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Compute gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    grad_norm_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Update progress bar
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "grad_norm": f"{grad_norm_value:.2f}",
                    })

                    # Log to wandb
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": loss.item() * self.gradient_accumulation_steps,
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch + batch_idx / len(self.train_dataloader),
                        }, step=self.global_step)

                    # Run validation at intervals
                    if self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0:
                        val_loss = self._run_validation_loss()
                        # Skip generation validation if --no_validation is set
                        if getattr(self.args, 'no_validation', False):
                            val_acc = None
                        else:
                            val_acc = self._run_validation_generation()

                        if self.use_wandb:
                            metrics = {"val/loss": val_loss}
                            if val_acc is not None:
                                metrics["val/math500_accuracy"] = val_acc
                            wandb.log(metrics, step=self.global_step)

                        # Log to CSV
                        if self.csv_file:
                            with open(self.csv_file, 'a') as f:
                                f.write(f"{self.global_step},{epoch + batch_idx / len(self.train_dataloader):.2f},{loss.item() * self.gradient_accumulation_steps:.4f},{val_loss:.4f},{current_lr:.2e}\n")

                        # Save best checkpoint
                        if val_acc is not None and val_acc > self.best_val_accuracy:
                            self.best_val_accuracy = val_acc
                            self.best_step = self.global_step
                            self._save_checkpoint("best")

                        self.model.train()

                    # Check if max_steps reached
                    if self.max_steps is not None and self.global_step >= self.max_steps:
                        logger.info(f"Reached max_steps ({self.max_steps}), stopping training...")
                        break

            # Check if max_steps reached (break outer loop too)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                break

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")

        # Final save
        self._save_checkpoint("final")
        logger.info(f"Training complete! Best accuracy: {self.best_val_accuracy:.2%} at step {self.best_step}")

    def _run_validation_loss(self) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def _run_validation_generation(self) -> Optional[float]:
        """Run generation-based validation on MATH-500."""
        if self.validation_data is None:
            return None

        if not self.use_vllm:
            logger.info("Skipping generation validation (vLLM not enabled)")
            return None

        # Check if few-shot prompting is enabled
        use_few_shot = getattr(self.args, 'use_few_shot', False)
        logger.info(f"Running MATH-500 validation ({len(self.validation_data)} samples, few-shot: {use_few_shot})...")
        logger.info(f"Model type: {'LoRA (PEFT)' if self.is_peft_model else 'Full fine-tuning'}")

        # Save model weights for vLLM
        if self.is_peft_model:
            # LoRA mode: save adapter only
            model_path = os.path.join(self.args.output_dir, "temp_lora_for_vllm", f"step_{self.global_step}")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            vllm_model_path = None  # Will use base model + LoRA adapter
            lora_path = model_path
        else:
            # Full fine-tuning: save full model
            model_path = os.path.join(self.args.output_dir, "temp_full_model_for_vllm", f"step_{self.global_step}")
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            vllm_model_path = model_path  # Load full model directly
            lora_path = None

        # Move model to CPU to free GPU memory for vLLM
        self.model.to("cpu")
        torch.cuda.empty_cache()

        try:
            self._init_vllm_engine(model_path=vllm_model_path)

            # Prepare prompts using format_math_prompt with optional few-shot
            prompts = []
            gold_answers = []
            problems = []
            levels = []
            for example in self.validation_data:
                problem = example.get("problem", "")
                problems.append(problem)
                prompt = format_math_prompt(problem, self.tokenizer, use_few_shot=use_few_shot)
                prompts.append(prompt)
                gold_answers.append(get_gold_answer(example))
                levels.append(example.get("level", None))

            # Generate
            sampling_params = SamplingParams(
                max_tokens=self.args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )

            # Use LoRA request only if using PEFT model
            if self.is_peft_model and lora_path is not None:
                lora_request = LoRARequest(
                    lora_name=f"step_{self.global_step}",
                    lora_int_id=self.global_step + 1,
                    lora_path=lora_path,
                )
                outputs = self.vllm_engine.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                # Full fine-tuning: no LoRA request needed
                outputs = self.vllm_engine.generate(prompts, sampling_params)

            # Evaluate and collect detailed results
            correct = 0
            samples = []
            # Track per-level accuracy
            level_stats = {i: {"correct": 0, "total": 0} for i in range(1, 6)}

            for i, output in enumerate(outputs):
                response = output.outputs[0].text
                predicted = extract_boxed_answer(response)
                is_correct = predicted is not None and check_math_answer(predicted, gold_answers[i])
                if is_correct:
                    correct += 1

                # Update per-level stats
                level = levels[i]
                if level is not None and level in level_stats:
                    level_stats[level]["total"] += 1
                    if is_correct:
                        level_stats[level]["correct"] += 1

                samples.append({
                    "problem": problems[i],
                    "gold_answer": gold_answers[i],
                    "model_response": response,
                    "extracted_answer": predicted,
                    "correct": is_correct,
                    "level": level,
                })

            accuracy = correct / len(self.validation_data)

            # Log overall accuracy
            logger.info(f"MATH-500 Overall Accuracy: {accuracy:.2%} ({correct}/{len(self.validation_data)})")

            # Compute and log per-level accuracy
            logger.info("Per-Level Accuracy:")
            level_accuracies = {}
            for level in sorted(level_stats.keys()):
                stats = level_stats[level]
                if stats["total"] > 0:
                    level_acc = stats["correct"] / stats["total"]
                    level_accuracies[level] = level_acc
                    logger.info(f"  Level {level}: {level_acc:.2%} ({stats['correct']}/{stats['total']})")

            # Save detailed validation outputs with per-level stats
            self._save_validation_outputs(correct, len(self.validation_data), accuracy, samples, level_stats)

            # Log per-level metrics to wandb
            if self.use_wandb:
                level_metrics = {}
                for level, stats in level_stats.items():
                    if stats["total"] > 0:
                        level_metrics[f"val/level_{level}_accuracy"] = stats["correct"] / stats["total"]
                        level_metrics[f"val/level_{level}_correct"] = stats["correct"]
                        level_metrics[f"val/level_{level}_total"] = stats["total"]
                wandb.log(level_metrics, step=self.global_step)

            return accuracy

        except Exception as e:
            logger.error(f"Validation generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            self._cleanup_vllm_engine()
            self.model.to(self.device)

    def _save_validation_outputs(self, correct: int, total: int, accuracy: float, samples: List[Dict[str, Any]], level_stats: Optional[Dict[int, Dict[str, int]]] = None):
        """Save detailed validation outputs to JSON file."""
        # Create validation outputs directory
        validation_dir = os.path.join(self.args.output_dir, "validation_outputs")
        os.makedirs(validation_dir, exist_ok=True)

        # Build output data in the same format as sft_experiments_v0
        output_data = {
            "step": self.global_step,
            "math500": {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "samples": samples,
            },
        }

        # Add per-level stats if available
        if level_stats is not None:
            by_level = {}
            for level, stats in level_stats.items():
                if stats["total"] > 0:
                    by_level[str(level)] = {
                        "total": stats["total"],
                        "correct": stats["correct"],
                        "accuracy": stats["correct"] / stats["total"],
                    }
            output_data["math500"]["by_level"] = by_level

        # Save to JSON file
        output_file = os.path.join(validation_dir, f"step_{self.global_step}.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved validation outputs to: {output_file}")

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model (LoRA adapter if using PEFT)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_accuracy": self.best_val_accuracy,
            "best_step": self.best_step,
        }
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint to: {checkpoint_dir}")
