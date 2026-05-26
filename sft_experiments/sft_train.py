#!/usr/bin/env python3
"""
SFT Training Script for Ouro-2.6B-Thinking on MATH dataset.

Uses SimpleSFTTrainer for single-process training with vLLM validation.

Usage:
    python sft_train.py --model_path /path/to/model --train_file train.parquet --val_file val.parquet

    # With config overrides
    python sft_train.py --learning_rate 1e-5 --num_train_epochs 3
"""

import os
import sys
import json
import re
import time
import logging
import argparse
from typing import Dict, Any, Optional

# Set environment variables before imports - OFFLINE MODE for cluster without internet
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT for LoRA
from peft import LoraConfig, TaskType, get_peft_model

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_lora_target_modules(target_modules_arg: str):
    """Convert CLI argument to LoRA target modules for PEFT LoraConfig.

    Args:
        target_modules_arg: Either "all-linear" or "attention"

    Returns:
        Either "all-linear" string or list ["q_proj", "k_proj", "v_proj", "o_proj"]
    """
    if target_modules_arg == "all-linear":
        return "all-linear"
    elif target_modules_arg == "attention":
        # Attention projection layers: q_proj, k_proj, v_proj, o_proj
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        raise ValueError(f"Unknown lora_target_modules: {target_modules_arg}")


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SFT Training")

    # Model args
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to base model")

    # Data args
    parser.add_argument("--train_file", type=str, default=None,
                        help="Path to training parquet file")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to validation parquet file")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max sequence length")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Global training batch size")
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="Micro batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps (overrides train_batch_size/micro_batch_size calculation)")

    # Training args
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training steps (overrides epochs)")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=None,
                        help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=None,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear",
                        choices=["all-linear", "attention"],
                        help="LoRA target modules: 'all-linear' applies to all linear layers, "
                             "'attention' applies only to attention weights (q_proj, k_proj, v_proj, o_proj)")

    # Validation args
    parser.add_argument("--eval_steps", type=int, default=None,
                        help="Run validation every N steps")
    parser.add_argument("--math500_test_file", type=str, default=None,
                        help="Path to MATH-500 test JSONL file")
    parser.add_argument("--math500_samples", type=int, default=None,
                        help="Number of MATH-500 samples (None=all)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens for generation")
    parser.add_argument("--use_few_shot", action="store_true",
                        help="Use 5-shot CoT prompting")
    parser.add_argument("--no_validation", action="store_true",
                        help="Disable all validation (baseline + periodic)")
    parser.add_argument("--no_initial_validation", action="store_true", default=True,
                        help="Skip baseline validation at step 0 (periodic validation still runs)")

    # vLLM args
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="Use vLLM for validation")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Disable vLLM")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=None,
                        help="vLLM GPU memory utilization")

    # Logging args
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name")

    # Additional args
    parser.add_argument("--prompt_key", type=str, default="prompt",
                        help="Column name for prompts in parquet")
    parser.add_argument("--response_key", type=str, default="response",
                        help="Column name for responses in parquet")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Max prompt length for validation generation")

    # Legacy arg (ignored, kept for compatibility)
    parser.add_argument("--simple", action="store_true",
                        help="(Deprecated) Simple mode is now the default")

    return parser.parse_args()


def main():
    """Main entry point for SFT training."""
    from simple_sft_trainer import SimpleSFTTrainer

    args = parse_args()

    logger.info("=" * 60)
    logger.info("SFT Training (single-process)")
    logger.info("=" * 60)

    # Set seed
    seed = args.seed if args.seed else 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation='flash_attention_2',
    )

    # Apply LoRA if enabled, otherwise full fine-tuning
    lora_r = args.lora_r if args.lora_r else 32
    lora_alpha = args.lora_alpha if args.lora_alpha else 64
    if not args.no_lora:
        target_modules = get_lora_target_modules(args.lora_target_modules)
        logger.info("=" * 60)
        logger.info("TRAINING MODE: LoRA FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}) with target_modules={args.lora_target_modules}...")
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # PEFT's get_peft_model() automatically freezes the base model
        # and only keeps LoRA parameters trainable - no need to manually freeze

        model.print_trainable_parameters()

        # Also log trainable params in consistent format
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    else:
        logger.info("=" * 60)
        logger.info("TRAINING MODE: FULL PARAMETER FINE-TUNING")
        logger.info("=" * 60)
        logger.info("All model parameters will be updated (no LoRA)")
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing to reduce memory usage
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    # Create args namespace for simple trainer
    class TrainerArgs:
        pass

    trainer_args = TrainerArgs()
    trainer_args.model_path = args.model_path
    trainer_args.output_dir = args.output_dir if args.output_dir else "./sft_output"
    trainer_args.max_seq_length = args.max_seq_length if args.max_seq_length else 2048
    trainer_args.learning_rate = args.learning_rate if args.learning_rate else 2e-5
    trainer_args.weight_decay = args.weight_decay if args.weight_decay else 0.1
    trainer_args.max_grad_norm = args.max_grad_norm if args.max_grad_norm else 1.0
    trainer_args.warmup_ratio = args.warmup_ratio if args.warmup_ratio else 0.1
    trainer_args.num_train_epochs = args.num_train_epochs if args.num_train_epochs else 2
    trainer_args.max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    trainer_args.train_batch_size = args.train_batch_size if args.train_batch_size else 128
    trainer_args.micro_batch_size = args.micro_batch_size if args.micro_batch_size else 8
    trainer_args.gradient_accumulation_steps = args.gradient_accumulation_steps  # None means auto-calculate
    trainer_args.eval_steps = args.eval_steps if args.eval_steps else 20
    trainer_args.math500_samples = args.math500_samples
    trainer_args.max_new_tokens = args.max_new_tokens if args.max_new_tokens else 3072
    trainer_args.max_prompt_length = args.max_prompt_length if args.max_prompt_length else 1024
    trainer_args.use_vllm = not args.no_vllm
    trainer_args.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization if args.vllm_gpu_memory_utilization else 0.5
    trainer_args.no_wandb = args.no_wandb
    trainer_args.wandb_project = args.wandb_project if args.wandb_project else "sft-ouro-math"
    trainer_args.prompt_key = args.prompt_key
    trainer_args.response_key = args.response_key
    trainer_args.no_validation = args.no_validation
    trainer_args.no_initial_validation = args.no_initial_validation
    trainer_args.use_few_shot = args.use_few_shot

    # Create output directory
    os.makedirs(trainer_args.output_dir, exist_ok=True)

    # Create trainer
    trainer = SimpleSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_parquet=args.train_file,
        val_parquet=args.val_file,
        args=trainer_args,
        validation_jsonl=args.math500_test_file,
    )

    # Train
    trainer.train()

    logger.info("SFT training complete!")
    logger.info(f"Model saved to: {trainer_args.output_dir}")


if __name__ == '__main__':
    main()
