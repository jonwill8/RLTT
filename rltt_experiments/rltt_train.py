#!/usr/bin/env python3
"""
RLTT (Reward Latent Thought Trajectories) Training Script.

This script implements the RLTT algorithm for training looped language models
like Ouro. RLTT distributes credit across the entire latent thought trajectory,
not just the final loop output.

Key differences from GRPO:
- RLTT aggregates log-probabilities from ALL loops: Σ ω_t * log P^(t)
- GRPO only uses the final loop: log P^(T_max)

This provides denser credit assignment for the model's latent reasoning.

Features:
- vLLM-accelerated rollout generation (3-10x faster than HF generation)
- FSDP for multi-GPU training (via verl)
- Configurable loop weighting (uniform or progressive)
- Same CLI structure as grpo_experiments for compatibility
- Offline support for HPC environments
- Checkpoint loading from SFT experiments

Usage:
    # Single GPU (uses simple trainer)
    python rltt_train.py --model_path /path/to/model

    # Multi-GPU with verl + vLLM
    python rltt_train.py --model_path /path/to/model --n_gpus 8

    # With RLTT-specific options
    python rltt_train.py --loop_weighting progressive --progressive_alpha 2.0
"""
import os
import sys
import json
import logging
import argparse
import glob as glob_module
from typing import Optional, Dict, Any
from dataclasses import asdict

# Set offline mode before any imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # Ensure our config takes effect
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RLTT Training - Reward Latent Thought Trajectories"
    )

    # === Model arguments ===
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking",
        help="Path to base model weights"
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=4,
        help="Fixed number of recurrent loops for Ouro model"
    )

    # === RLTT-specific arguments ===
    parser.add_argument(
        "--loop_weighting",
        type=str,
        default="uniform",
        choices=["uniform", "progressive", "exit_pdf", "learned"],
        help="Loop weighting strategy: 'uniform' (Option A), 'progressive' (Option B), 'exit_pdf' (model's exit probability), or 'learned' (jointly optimized)"
    )
    parser.add_argument(
        "--progressive_alpha",
        type=float,
        default=1.0,
        help="Alpha exponent for progressive weighting (only used if loop_weighting='progressive')"
    )
    parser.add_argument(
        "--learned_weights_init",
        type=str,
        default="uniform",
        choices=["uniform", "progressive"],
        help="Initialization strategy for learned weights (only used if loop_weighting='learned')"
    )
    parser.add_argument(
        "--learned_weights_temp",
        type=float,
        default=1.0,
        help="Softmax temperature for learned weights (higher = more uniform, only used if loop_weighting='learned')"
    )
    parser.add_argument(
        "--learned_weights_lr",
        type=float,
        default=None,
        help="Learning rate for learned weights (default: 10x model learning rate, only used if loop_weighting='learned')"
    )
    parser.add_argument(
        "--use_rltt_model",
        action="store_true",
        help="Use the RLTT-modified Ouro model that exposes per-loop hidden states"
    )
    parser.add_argument(
        "--rltt_model_path",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT",
        help="Path to RLTT-modified Ouro model (used if --use_rltt_model is set)"
    )
    parser.add_argument(
        "--rltt_logprob_chunk_size",
        type=int,
        default=256,
        help="Chunk size for memory-efficient log-prob computation. Set to 0 to disable chunking. "
             "Lower values use less memory but may be slightly slower. Recommended: 256-512."
    )

    # === SFT checkpoint loading ===
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint (LoRA adapter)"
    )
    parser.add_argument(
        "--use_latest_sft",
        action="store_true",
        help="Auto-find latest SFT checkpoint"
    )
    parser.add_argument(
        "--no_sft_checkpoint",
        action="store_true",
        help="Skip SFT checkpoint loading - start from fresh MODEL_PATH weights"
    )
    # Compute default SFT output dir relative to this script
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _rltt_root = os.path.dirname(_script_dir)
    _default_sft_output = os.path.join(_rltt_root, "sft_experiments", "sft_ouro_math_output")
    parser.add_argument(
        "--sft_output_dir",
        type=str,
        default=_default_sft_output,
        help="SFT output directory for auto-discovery"
    )

    # === Data arguments ===
    parser.add_argument(
        "--train_file",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl",
        help="Training data JSONL file"
    )

    # === RLTT hyperparameters ===
    parser.add_argument(
        "--beta",
        type=float,
        default=0.001,
        help="KL penalty coefficient"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of generations per prompt (group size)"
    )
    parser.add_argument(
        "--num_prompts_per_batch",
        type=int,
        default=128,
        help="Number of unique prompts per batch"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for generation"
    )

    # === Optimizer settings ===
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.1,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (fraction of total steps for warmup). One step = one prompt batch."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],  # verl only supports these two (both include warmup)
        help="Learning rate scheduler type (both include warmup)"
    )

    # === Batch sizes ===
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--forward_batch_size",
        type=int,
        default=2,
        help="Micro-batch size per GPU for actor update. With RLTT's 4 UT loops, "
             "reduce to 2-4 to avoid OOM. This controls ppo_micro_batch_size_per_gpu. "
             "NOTE: Ignored when use_dynamic_bsz=True (use --ppo_max_token_len instead)."
    )
    parser.add_argument(
        "--ppo_max_token_len",
        type=int,
        default=8192,
        help="Max tokens per micro-batch during actor update (with dynamic batching). "
             "Controls memory usage during backprop. Lower = less memory but slower. "
             "With max_seq_len=4096: 8192 = ~2 samples, 16384 = ~4 samples per micro-batch."
    )
    parser.add_argument(
        "--log_prob_max_token_len",
        type=int,
        default=16384,
        help="Max tokens per micro-batch during log-prob computation (rollout/ref). "
             "Controls memory usage during forward pass for log-prob calculation. "
             "Lower = less memory but slower. Default: 16384."
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable verl's gradient checkpointing. Required when using exit_pdf loop weighting "
             "to avoid conflicts with model's loop-level checkpointing. May increase memory usage "
             "but the model's own loop-level checkpointing still provides memory efficiency."
    )

    # === Sequence lengths ===
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length in tokens (zero-shot prompts are ~170 tokens)"
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=4096,
        help="Maximum completion length in tokens"
    )

    # === Training duration ===
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 to use epochs)"
    )

    # === LoRA configuration ===
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA for full parameter fine-tuning. When used with --sft_checkpoint, "
             "the SFT LoRA will be merged into the base model before training all parameters."
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        choices=["all-linear", "attention"],
        help="LoRA target modules: 'all-linear' applies to all linear layers, "
             "'attention' applies only to attention weights (q_proj, k_proj, v_proj, o_proj)"
    )

    # === Validation arguments ===
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=7,
        help="Validation frequency (in epochs for verl)"
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Validation batch size"
    )
    parser.add_argument(
        "--val_max_new_tokens",
        type=int,
        default=4096,
        help="Max completion length for validation"
    )
    parser.add_argument(
        "--math500_test_file",
        type=str,
        default="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl",
        help="Path to MATH-500 test set"
    )
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Disable validation"
    )
    parser.add_argument(
        "--skip_initial_validation",
        action="store_true",
        help="Skip validation at step 0"
    )

    # === Logging ===
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Logging frequency"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5,
        help="Checkpoint save frequency (all checkpoints are saved)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rltt-ouro-math",
        help="W&B project name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    # === vLLM configuration ===
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM for generation (default: True)"
    )
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM, use HuggingFace generation"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory fraction for vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism in vLLM"
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="Disable CUDA graphs in vLLM"
    )

    # === Memory optimization ===
    parser.add_argument(
        "--ref_offload",
        action="store_true",
        help="Offload reference model parameters to CPU. Saves ~5-7GB GPU memory per GPU "
             "but slows KL computation by ~10-15%%. Recommended when facing OOM."
    )

    # === Multi-GPU settings ===
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training"
    )

    # === Other ===
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rltt_output",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader (should match or be less than --cpus-per-task in SLURM)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def merge_lora_checkpoint(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    rltt_logprob_chunk_size: Optional[int] = None,
) -> str:
    """Merge LoRA adapter into base model for verl compatibility.

    verl doesn't support LoRA natively, so we merge the adapter into
    the base model and save the merged model.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model
        rltt_logprob_chunk_size: Chunk size for memory-efficient log-prob computation

    Returns:
        Path to merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA adapter into base model...")
    logger.info(f"  Base model: {base_model_path}")
    logger.info(f"  LoRA adapter: {lora_path}")
    logger.info(f"  Output: {output_path}")

    # Check if already merged
    config_json_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_json_path):
        logger.info(f"Merged model already exists at {output_path}, skipping merge")
        # Still copy modeling files in case they were updated
        import shutil
        for fname in ["modeling_ouro.py", "configuration_ouro.py", "chat_template.jinja"]:
            src = os.path.join(base_model_path, fname)
            dst = os.path.join(output_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        # Update config.json with RLTT chunk size if specified
        if rltt_logprob_chunk_size is not None:
            with open(config_json_path, "r") as f:
                config_dict = json.load(f)
            config_dict["rltt_logprob_chunk_size"] = rltt_logprob_chunk_size
            with open(config_json_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        return output_path

    os.makedirs(output_path, exist_ok=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",  # Load on CPU for merging
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        is_trainable=False,
    )

    # Merge and unload
    model = model.merge_and_unload()

    # Set RLTT chunk size in model config before saving
    if rltt_logprob_chunk_size is not None:
        model.config.rltt_logprob_chunk_size = rltt_logprob_chunk_size

    # Save merged model
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Copy custom modeling files from base model (required for trust_remote_code)
    import shutil
    custom_files = [
        "modeling_ouro.py",
        "configuration_ouro.py",
        "chat_template.jinja",
    ]
    for fname in custom_files:
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    logger.info("LoRA merge complete!")
    return output_path


def find_best_sft_checkpoint(sft_output_dir: str) -> Optional[str]:
    """Find the best SFT checkpoint from the output directory.

    Looks for checkpoints in the format:
    sft_output_dir/<job_id>/checkpoints/best_model/
    """
    if not os.path.exists(sft_output_dir):
        logger.warning(f"SFT output directory does not exist: {sft_output_dir}")
        return None

    # Find all job directories (numeric names)
    job_dirs = []
    for item in os.listdir(sft_output_dir):
        item_path = os.path.join(sft_output_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            job_dirs.append((int(item), item_path))

    if not job_dirs:
        # Check if checkpoints exist directly in the output dir
        best_model_path = os.path.join(sft_output_dir, "checkpoints", "best_model")
        if os.path.exists(best_model_path):
            return best_model_path
        logger.warning(f"No job directories found in: {sft_output_dir}")
        return None

    # Sort by job ID (highest = most recent)
    job_dirs.sort(key=lambda x: x[0], reverse=True)

    # Find first job with a best_model checkpoint
    for job_id, job_path in job_dirs:
        best_model_path = os.path.join(job_path, "checkpoints", "best_model")
        if os.path.exists(best_model_path):
            # Verify it has adapter files
            if os.path.exists(os.path.join(best_model_path, "adapter_config.json")):
                logger.info(f"Found SFT checkpoint from job {job_id}: {best_model_path}")
                return best_model_path

    logger.warning(f"No valid SFT checkpoints found in: {sft_output_dir}")
    return None


def get_lora_target_modules_for_verl(target_modules_arg: str):
    """Convert CLI argument to LoRA target modules for verl config.

    For "all-linear", returns the string which PEFT recognizes as a special value.
    For "attention", returns a comma-separated string that verl's convert_to_regular_types
    will convert to a list for PEFT.

    Note: verl's HFModelConfig.target_modules is typed as Optional[str], so we must
    pass a string. The verl utility function convert_to_regular_types() handles
    converting comma-separated strings to lists for PEFT.

    Args:
        target_modules_arg: Either "all-linear" or "attention"

    Returns:
        Either "all-linear" string or "q_proj,k_proj,v_proj,o_proj" comma-separated string
    """
    if target_modules_arg == "all-linear":
        return "all-linear"
    elif target_modules_arg == "attention":
        # Attention projection layers as comma-separated string for verl
        # verl's convert_to_regular_types() will convert this to a list for PEFT
        return "q_proj,k_proj,v_proj,o_proj"
    else:
        raise ValueError(f"Unknown lora_target_modules: {target_modules_arg}")


def get_lora_target_modules_for_peft(target_modules_arg: str):
    """Convert CLI argument to LoRA target modules for PEFT LoraConfig.

    PEFT's LoraConfig accepts either a string or list for target_modules.

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


def build_verl_config(args) -> Dict[str, Any]:
    """Build verl configuration from command line arguments.

    Note: verl doesn't natively support RLTT's multi-loop loss.
    For multi-GPU training with verl, we fall back to GRPO-style loss
    (final loop only) but log a warning. Full RLTT requires the
    simple trainer which has access to per-loop hidden states.
    """
    # Determine number of GPUs
    n_gpus = args.n_gpus or torch.cuda.device_count()
    if n_gpus == 0:
        n_gpus = 1
        logger.warning("No GPUs detected, using CPU (not recommended)")

    # Determine vLLM tensor parallel size
    tp_size = args.vllm_tensor_parallel_size
    tp_size = min(tp_size, n_gpus)
    if tp_size > 1 and n_gpus % tp_size != 0:
        tp_size = 1
        logger.warning(f"TP size {args.vllm_tensor_parallel_size} doesn't divide {n_gpus} GPUs, using TP=1")

    # Compute total batch size
    train_batch_size = args.num_prompts_per_batch * args.num_generations

    # Determine rollout method
    use_vllm = args.use_vllm and not args.no_vllm

    # Max prompt for model
    max_prompt_for_model = args.max_prompt_length

    config = {
        # Algorithm - use GRPO as base since verl doesn't support RLTT natively
        "algorithm": {
            "adv_estimator": "grpo",
            "kl_penalty": "kl",
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": args.beta,
            }
        },

        # Data
        "data": {
            "train_batch_size": train_batch_size,
            "val_batch_size": train_batch_size,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_completion_length,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "sampler": None,
            "shuffle": True,
            "dataloader_num_workers": args.dataloader_num_workers,
        },

        # Actor/Rollout/Ref combined
        "actor_rollout_ref": {
            "model": {
                "path": args.model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": not args.no_gradient_checkpointing,
                "trust_remote_code": True,
                # LoRA config - enables ref_in_actor for KL regularization
                # The base model (without LoRA) serves as the frozen reference policy
                "lora_rank": 0 if args.no_lora else args.lora_r,
                "lora_alpha": args.lora_alpha if not args.no_lora else 0,
                "target_modules": get_lora_target_modules_for_verl(args.lora_target_modules),
                "exclude_modules": None,
            },
            "actor": {
                "_target_": "verl.workers.config.FSDPActorConfig",
                "strategy": "fsdp",
                "ppo_epochs": 1,
                "entropy_coeff": 0.0,
                "use_kl_loss": True,  # RLTT Equation 5: includes KL regularization term
                "kl_loss_coef": args.beta,  # β coefficient for KL term
                "kl_loss_type": "low_var_kl",
                "ppo_mini_batch_size": min(128, train_batch_size),
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": args.forward_batch_size,
                "use_dynamic_bsz": True,
                "ppo_max_token_len_per_gpu": args.ppo_max_token_len,
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "loss_agg_mode": "token-mean",
                "grad_clip": args.max_grad_norm,  # Gradient clipping threshold
                "optim": {
                    "_target_": "verl.workers.config.FSDPOptimizerConfig",
                    "optimizer": "AdamW8bit",
                    "optimizer_impl": "bitsandbytes.optim",
                    "lr": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "betas": [0.9, 0.999],
                    "override_optimizer_config": None,
                },
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": False,
                    "optimizer_offload": False,
                    "fsdp_size": -1,
                    "dtype": "bfloat16",
                },
                "checkpoint": {
                    "_target_": "verl.trainer.config.CheckpointConfig",
                    "save_contents": ["model", "optimizer", "extra"],
                    "load_contents": ["model", "optimizer", "extra"],
                    "async_save": False,
                },
            },
            "rollout": {
                "_target_": "verl.workers.config.rollout.RolloutConfig",
                "name": "vllm" if use_vllm else "hf",
                "mode": "sync",
                "n": args.num_generations,
                "tensor_model_parallel_size": tp_size,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "load_format": "auto",
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "enforce_eager": args.vllm_enforce_eager,
                "free_cache_engine": False,
                "enable_prefix_caching": True,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": None,
                "log_prob_max_token_len_per_gpu": args.log_prob_max_token_len,
                "log_prob_use_dynamic_bsz": True,
                "temperature": args.temperature,
                "max_model_len": max_prompt_for_model + args.max_completion_length,
                "enable_chunked_prefill": True,
                "prompt_length": args.max_prompt_length,
                "response_length": args.max_completion_length,
                "multi_turn": {
                    "enable": False,
                },
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": None,
                "log_prob_max_token_len_per_gpu": args.log_prob_max_token_len,
                "log_prob_use_dynamic_bsz": True,
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": args.ref_offload,  # Offload ref model to CPU to save GPU memory
                },
            },
            "hybrid_engine": True,
        },

        # Critic (disabled for GRPO-style training)
        "critic": {
            "enable": False,
        },

        # Algorithm settings
        "algorithm": {
            "adv_estimator": "grpo",  # Advantage computation method (used for weighting gradients)
            "use_kl_in_reward": False,
            "kl_coef": args.beta,
            "gamma": 1.0,  # Discount factor for GAE (1.0 = no discounting)
            "lam": 1.0,    # Lambda for GAE
        },

        # Trainer
        "trainer": {
            "experiment_name": f"RLTT-{os.path.basename(args.model_path)}-{args.loop_weighting}",
            "project_name": args.wandb_project,
            "logger": ["wandb"] if not args.no_wandb else [],
            "n_gpus_per_node": n_gpus,
            "nnodes": args.nnodes,
            "total_epochs": args.num_train_epochs,
            "save_freq": args.save_steps,
            "test_freq": -1 if args.no_validation else args.eval_steps,
            "val_before_train": False if args.no_validation else (not args.skip_initial_validation),
            "critic_warmup": 0,
            "validation_data_dir": os.path.join(args.output_dir, "validation_outputs"),
            "device": "cuda",
            "total_training_steps": None,
            "resume_mode": "disable",
            "esi_redundant_time": 0,  # ESI checkpoint timing buffer
            "balance_batch": False,
            "default_local_dir": args.output_dir,
            "default_hdfs_dir": None,  # HDFS directory for checkpoints (not used)
            # Gradient accumulation: accumulate N rollout batches before actor update
            # Effective batch size = num_prompts_per_batch * gradient_accumulation_steps
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },

        # Custom reward function with rollout logging (centralized in math_utils)
        "custom_reward_function": {
            "path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "math_utils", "reward.py")),
            "name": "compute_score",
        },

        # Global profiler (disabled)
        "global_profiler": {
            "steps": None,
            "profile_continuous_steps": None,
        },

        # Reward model config (using custom reward function, not a neural RM)
        "reward_model": {
            "enable": False,
            "launch_reward_fn_async": False,
        },
    }

    return config


def create_rollout_logger(output_dir: str):
    """Create rollout logger for logging completions."""
    rollout_log_dir = os.path.join(output_dir, "rollout_logs")
    os.makedirs(rollout_log_dir, exist_ok=True)
    rollout_log_file = os.path.join(rollout_log_dir, "rollouts.jsonl")

    def log_rollouts(step: int, prompts: list, completions: list, rewards: list, answers: list):
        """Log rollouts to JSONL file."""
        with open(rollout_log_file, "a") as f:
            for idx, (prompt, completion, reward, answer) in enumerate(zip(prompts, completions, rewards, answers)):
                entry = {
                    "step": step,
                    "index": idx,
                    "prompt": prompt if isinstance(prompt, str) else str(prompt),
                    "completion": completion if isinstance(completion, str) else str(completion),
                    "ground_truth": answer,
                    "reward": float(reward),
                    "correct": reward > 0,
                }
                f.write(json.dumps(entry) + "\n")

    return log_rollouts


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RLTT Training - Reward Latent Thought Trajectories")
    logger.info("=" * 60)

    # Determine and log training mode (LoRA vs Full Fine-tuning)
    training_mode = "FULL PARAMETER FINE-TUNING" if args.no_lora else "LoRA FINE-TUNING"
    logger.info(f"")
    logger.info(f"*** TRAINING MODE: {training_mode} ***")
    if args.no_lora:
        logger.info(f"*** All model parameters will be updated during training ***")
    else:
        logger.info(f"*** Only LoRA adapter parameters will be updated ***")
        logger.info(f"*** LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}, target: {args.lora_target_modules} ***")
    logger.info(f"")

    # Log configuration
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Beta (KL coef): {args.beta}")
    logger.info(f"Num generations: {args.num_generations}")
    logger.info(f"Num prompts per batch: {args.num_prompts_per_batch}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    effective_batch = args.num_prompts_per_batch * args.gradient_accumulation_steps
    logger.info(f"Effective prompts per update: {effective_batch} ({args.num_prompts_per_batch} x {args.gradient_accumulation_steps})")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max prompt length: {args.max_prompt_length}")
    logger.info(f"Max completion length: {args.max_completion_length}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"vLLM enabled: {args.use_vllm and not args.no_vllm}")
    logger.info(f"PPO max token len: {args.ppo_max_token_len}")
    logger.info(f"Log-prob max token len: {args.log_prob_max_token_len}")
    logger.info(f"Ref model offload: {args.ref_offload}")
    logger.info(f"")
    logger.info(f"RLTT-specific settings:")
    logger.info(f"  Loop weighting: {args.loop_weighting}")
    logger.info(f"  Progressive alpha: {args.progressive_alpha}")
    logger.info(f"  Total UT steps: {args.total_ut_steps}")
    logger.info(f"  Use RLTT model: {args.use_rltt_model}")
    if args.use_rltt_model:
        logger.info(f"  RLTT model path: {args.rltt_model_path}")
    if args.loop_weighting == "learned":
        logger.info(f"  Learned weights init: {args.learned_weights_init}")
        logger.info(f"  Learned weights temp: {args.learned_weights_temp}")
        lr_display = args.learned_weights_lr if args.learned_weights_lr else f"{args.learning_rate * 10} (10x model LR)"
        logger.info(f"  Learned weights LR: {lr_display}")

    # Check for SFT checkpoint
    sft_checkpoint = None
    if args.no_sft_checkpoint:
        logger.info("SFT checkpoint loading disabled - starting from fresh model weights")
        sft_checkpoint = None
    elif args.sft_checkpoint:
        sft_checkpoint = args.sft_checkpoint
    elif args.use_latest_sft:
        sft_checkpoint = find_best_sft_checkpoint(args.sft_output_dir)

    if sft_checkpoint:
        logger.info(f"SFT checkpoint: {sft_checkpoint}")
    elif not args.no_sft_checkpoint:
        logger.info("No SFT checkpoint specified - starting from fresh model weights")

    # Build verl configuration
    verl_config = build_verl_config(args)

    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "args": vars(args),
            "verl_config": verl_config,
        }, f, indent=2)
    logger.info(f"Saved configuration to: {config_path}")

    # Check if we should use verl or fallback to simple training
    n_gpus = args.n_gpus or torch.cuda.device_count()

    if n_gpus <= 1 or args.debug:
        # Single GPU or debug mode - use simplified training with full RLTT
        logger.info("Using simplified single-GPU training with full RLTT loss")
        run_simple_training(args, sft_checkpoint)
    else:
        # Multi-GPU - use verl with custom RLTT worker
        logger.info(f"Using verl for {n_gpus}-GPU training with full RLTT objective")
        logger.info("RLTTActorRolloutRefWorker enables multi-loop loss computation in verl")
        run_verl_training(args, verl_config, sft_checkpoint)


def run_simple_training(args, sft_checkpoint: Optional[str] = None):
    """Run simplified single-GPU training with full RLTT loss.

    This uses the SimpleRLTTTrainer which implements the full RLTT algorithm
    with multi-loop log-probability computation.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel, LoraConfig, get_peft_model
    import copy

    from data_utils import load_jsonl, prepare_train_data
    from reward_utils import create_math_reward_func

    set_seed(args.seed)

    # Determine model path (use RLTT model if requested)
    model_path = args.model_path
    if args.use_rltt_model:
        model_path = args.rltt_model_path
        logger.info(f"Using RLTT-modified Ouro model from: {model_path}")
        if args.loop_weighting == "exit_pdf":
            logger.info("exit_pdf loop weighting enabled - will use model's native exit probability distribution")
    elif args.loop_weighting == "exit_pdf":
        logger.warning(
            "exit_pdf loop weighting requested but --use_rltt_model not set. "
            "The standard Ouro model may not expose per-loop logits. "
            "Consider using --use_rltt_model for optimal performance."
        )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    # Set Ouro model config
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps
    # Set RLTT log-prob chunk size for memory-efficient computation
    if args.rltt_logprob_chunk_size is not None:
        model.config.rltt_logprob_chunk_size = args.rltt_logprob_chunk_size

    # Create reference model (before LoRA)
    logger.info("Creating reference model...")
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load or create LoRA, or set up full fine-tuning
    if args.no_lora:
        # FULL FINE-TUNING MODE
        logger.info("=" * 60)
        logger.info("FULL PARAMETER FINE-TUNING MODE")
        logger.info("=" * 60)

        if sft_checkpoint:
            # Check if SFT checkpoint is a LoRA adapter
            adapter_config_path = os.path.join(sft_checkpoint, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("SFT checkpoint is a LoRA adapter - merging into base model...")
                # Load the LoRA adapter and merge
                model = PeftModel.from_pretrained(
                    model,
                    sft_checkpoint,
                    is_trainable=False,
                )
                model = model.merge_and_unload()
                logger.info("LoRA adapter merged into base model successfully")
            else:
                logger.info(f"SFT checkpoint is a full model, loading weights...")
                # Load full model weights
                model = AutoModelForCausalLM.from_pretrained(
                    sft_checkpoint,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto",
                )
                if hasattr(model.config, "total_ut_steps"):
                    model.config.total_ut_steps = args.total_ut_steps
                if args.rltt_logprob_chunk_size is not None:
                    model.config.rltt_logprob_chunk_size = args.rltt_logprob_chunk_size

        # All parameters are trainable for full fine-tuning
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info("All model parameters will be updated during training")

        # Create a fresh reference model from the merged weights
        logger.info("Creating reference model from merged weights...")
        ref_model = copy.deepcopy(model)
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        # LoRA FINE-TUNING MODE
        logger.info("=" * 60)
        logger.info("LoRA FINE-TUNING MODE")
        logger.info("=" * 60)

        if sft_checkpoint:
            logger.info(f"Loading LoRA from SFT checkpoint: {sft_checkpoint}")
            model = PeftModel.from_pretrained(
                model,
                sft_checkpoint,
                is_trainable=True,
            )
            # PEFT's PeftModel.from_pretrained() with is_trainable=True automatically
            # freezes the base model and only keeps LoRA parameters trainable
        else:
            target_modules = get_lora_target_modules_for_peft(args.lora_target_modules)
            logger.info(f"Creating new LoRA adapter with target_modules={target_modules}...")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        # PEFT's get_peft_model() automatically freezes the base model
        # and only keeps LoRA parameters trainable - no need to manually freeze

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        model.print_trainable_parameters()

    # Prepare data
    logger.info("Preparing training data...")
    data_dir = os.path.join(args.output_dir, "data")
    train_parquet = prepare_train_data(
        args.train_file,
        data_dir,
        tokenizer=tokenizer,
        use_few_shot=False,
    )

    # Create reward function
    reward_func = create_math_reward_func()

    # Create rollout logger
    log_rollouts = create_rollout_logger(args.output_dir)

    logger.info("Starting RLTT training with full multi-loop loss...")
    logger.info("Note: For multi-GPU training with vLLM acceleration, use --n_gpus > 1")

    # Import and run RLTT training loop
    from simple_trainer import SimpleRLTTTrainer

    trainer = SimpleRLTTTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_parquet=train_parquet,
        reward_func=reward_func,
        args=args,
        log_rollouts=log_rollouts,
    )

    trainer.train()


def run_verl_training(args, verl_config: Dict[str, Any], sft_checkpoint: Optional[str] = None):
    """Run verl-based multi-GPU training with full RLTT objective.

    This uses the custom RLTTActorRolloutRefWorker which implements
    the full RLTT loss with multi-loop log-probability computation.
    """
    import time

    # Import verl components
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, ResourcePoolManager
        from verl.single_controller.ray import RayWorkerGroup
        from omegaconf import OmegaConf
    except ImportError as e:
        logger.error(f"Failed to import verl: {e}")
        logger.error("Please install verl: pip install verl")
        logger.error("Falling back to simple training...")
        run_simple_training(args, sft_checkpoint)
        return

    # Create custom trainer with timing logging
    class TimedRayPPOTrainer(RayPPOTrainer):
        """RayPPOTrainer with prominent timing logging for rollout vs backprop."""

        def fit(self):
            """Training loop with timing logging and gradient accumulation across rollout batches."""
            from omegaconf import OmegaConf
            from verl.utils.tracking import Tracking
            from tqdm import tqdm
            import numpy as np
            import uuid
            from verl import DataProto
            from verl.trainer.ppo.core_algos import agg_loss
            from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics, compute_throughout_metrics
            from verl.trainer.ppo.reward import compute_reward
            from verl.trainer.ppo.ray_trainer import (
                compute_response_mask, reduce_metrics, should_save_ckpt_esi,
                AdvantageEstimator, compute_advantage, apply_kl_penalty,
            )
            from verl.utils.debug import marked_timer
            from copy import deepcopy

            tracking_logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

            self.global_steps = 0
            self._load_checkpoint()

            # Gradient accumulation config
            grad_accum_steps = self.config.trainer.get("gradient_accumulation_steps", 1)
            if grad_accum_steps > 1:
                logger.info(f"Gradient accumulation enabled: accumulating {grad_accum_steps} rollout batches per update")

            # Initial validation
            if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = self._validate()
                if val_metrics:
                    logger.info(f"Initial validation metrics: {val_metrics}")
                    tracking_logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    return

            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
            self.global_steps += 1
            last_val_metrics = None
            self.max_steps_duration = 0

            # Epoch-level timing accumulators
            epoch_rollout_time = 0.0
            epoch_backprop_time = 0.0
            epoch_steps = 0
            current_epoch = 0

            # Gradient accumulation buffer
            accumulated_batches = []
            accumulated_timing = {"gen": 0.0, "reward": 0.0, "old_log_prob": 0.0, "ref": 0.0, "adv": 0.0}
            accum_count = 0

            for epoch in range(self.config.trainer.total_epochs):
                if epoch != current_epoch:
                    # Log epoch summary
                    if epoch_steps > 0:
                        logger.info(f"Epoch {current_epoch + 1} Timing Summary:")
                        logger.info(f"  Total rollout time: {epoch_rollout_time:.1f}s")
                        logger.info(f"  Total backprop time: {epoch_backprop_time:.1f}s")
                        logger.info(f"  Avg rollout per step: {epoch_rollout_time / epoch_steps:.1f}s")
                        logger.info(f"  Avg backprop per step: {epoch_backprop_time / epoch_steps:.1f}s")
                    # Reset for new epoch
                    epoch_rollout_time = 0.0
                    epoch_backprop_time = 0.0
                    epoch_steps = 0
                    current_epoch = epoch

                for batch_dict in self.train_dataloader:
                    metrics = {}
                    timing_raw = {}

                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        # Generate (rollout)
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        # Log rollout time immediately
                        if grad_accum_steps > 1:
                            logger.info(f"Step {self.global_steps} (accum {accum_count + 1}/{grad_accum_steps}): rollout done in {timing_raw.get('gen', 0.0):.1f}s")
                        else:
                            logger.info(f"Step {self.global_steps}: rollout done in {timing_raw.get('gen', 0.0):.1f}s")

                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # Reward
                        with marked_timer("reward", timing_raw, color="yellow"):
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        # Old log prob
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics["actor/entropy"] = entropy_agg.detach().item()
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        # Reference log prob
                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # Critic values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        # Advantage
                        with marked_timer("adv", timing_raw, color="brown"):
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Note: We compute advantage per-batch first, then re-normalize after concatenation
                            # This is required because GRPO normalizes by group
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # Update critic (if enabled, do it per batch)
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # Accumulate timing
                        accumulated_timing["gen"] += timing_raw.get("gen", 0.0)
                        accumulated_timing["reward"] += timing_raw.get("reward", 0.0)
                        accumulated_timing["old_log_prob"] += timing_raw.get("old_log_prob", 0.0)
                        accumulated_timing["adv"] += timing_raw.get("adv", 0.0)

                        # Accumulate batch for gradient accumulation
                        accumulated_batches.append(batch)
                        accum_count += 1

                        # Update actor only when we have accumulated enough batches
                        should_update = (accum_count >= grad_accum_steps) or is_last_step
                        if should_update and self.config.trainer.critic_warmup <= self.global_steps:
                            # Concatenate accumulated batches
                            if len(accumulated_batches) > 1:
                                # Collect global_token_num from each batch before concat
                                # (DataProto.concat fails on conflicting meta_info values)
                                all_global_token_nums = []
                                for b in accumulated_batches:
                                    gtn = b.meta_info.pop("global_token_num", [])
                                    if isinstance(gtn, list):
                                        all_global_token_nums.extend(gtn)
                                    else:
                                        all_global_token_nums.append(gtn)

                                combined_batch = DataProto.concat(accumulated_batches)
                                # Restore global_token_num as combined list
                                combined_batch.meta_info["global_token_num"] = all_global_token_nums
                            else:
                                combined_batch = accumulated_batches[0]

                            with marked_timer("update_actor", timing_raw, color="red"):
                                combined_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(combined_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                            # Log backprop time
                            total_rollout_time = accumulated_timing["gen"]
                            backprop_time_val = timing_raw.get("update_actor", 0.0)
                            if grad_accum_steps > 1:
                                logger.info(f"Step {self.global_steps}: backprop done in {backprop_time_val:.1f}s "
                                           f"(accumulated {accum_count} batches, {len(combined_batch.batch['input_ids'])} samples, "
                                           f"total rollout time: {total_rollout_time:.1f}s)")
                            else:
                                logger.info(f"Step {self.global_steps}: backprop done in {backprop_time_val:.1f}s")

                            # Use combined_batch for metrics when accumulation happened
                            batch = combined_batch

                            # Reset accumulation buffer
                            accumulated_batches = []
                            accumulated_timing = {"gen": 0.0, "reward": 0.0, "old_log_prob": 0.0, "ref": 0.0, "adv": 0.0}
                            accum_count = 0

                    # Track timing
                    rollout_time = timing_raw.get("gen", 0.0)
                    backprop_time = timing_raw.get("update_actor", 0.0)
                    epoch_rollout_time += rollout_time
                    epoch_backprop_time += backprop_time
                    epoch_steps += 1

                    # Validation
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Checkpoint
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    steps_duration = timing_raw.get("step", 0.0)
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # Metrics
                    metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "timing/rollout_time": rollout_time,
                        "timing/backprop_time": backprop_time,
                        "training/grad_accum_steps": grad_accum_steps,
                    })
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    tracking_logger.log(data=metrics, step=self.global_steps)

                    # Update progress bar with timing
                    progress_bar.set_postfix({
                        "rollout": f"{rollout_time:.1f}s",
                        "backprop": f"{backprop_time:.1f}s",
                        "accum": f"{accum_count}/{grad_accum_steps}" if grad_accum_steps > 1 else "off",
                    })
                    progress_bar.update(1)
                    self.global_steps += 1

                    if is_last_step:
                        # Final epoch summary
                        if epoch_steps > 0:
                            logger.info(f"Epoch {current_epoch + 1} Timing Summary:")
                            logger.info(f"  Total rollout time: {epoch_rollout_time:.1f}s")
                            logger.info(f"  Total backprop time: {epoch_backprop_time:.1f}s")
                        logger.info(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

            # Final epoch summary
            if epoch_steps > 0:
                logger.info(f"Epoch {current_epoch + 1} Timing Summary:")
                logger.info(f"  Total rollout time: {epoch_rollout_time:.1f}s")
                logger.info(f"  Total backprop time: {epoch_backprop_time:.1f}s")
            progress_bar.close()

    # Import custom RLTT worker
    try:
        from verl_rltt import RLTTActorRolloutRefWorker
    except ImportError:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from verl_rltt import RLTTActorRolloutRefWorker

    from transformers import AutoTokenizer
    from data_utils import prepare_train_data, prepare_test_data
    from math_utils import compute_score as compute_math_reward

    # Set output directory environment variable for rollout logging in Ray workers
    os.environ["RLTT_OUTPUT_DIR"] = os.path.abspath(args.output_dir)

    logger.info("Setting up verl training with RLTT objective...")
    logger.info("Using RLTTActorRolloutRefWorker for multi-loop loss computation")

    # Log training mode prominently
    if args.no_lora:
        logger.info("=" * 60)
        logger.info("FULL PARAMETER FINE-TUNING MODE")
        logger.info("All model parameters will be trained (no LoRA adapters)")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("LoRA FINE-TUNING MODE")
        logger.info(f"Training LoRA adapters with rank={args.lora_r}, alpha={args.lora_alpha}")
        logger.info(f"Target modules: {args.lora_target_modules}")
        logger.info("=" * 60)

    # Determine model path
    model_path = args.model_path
    if args.use_rltt_model:
        model_path = args.rltt_model_path
        logger.info(f"Using RLTT-modified Ouro model: {model_path}")

    model_path = os.path.abspath(model_path)

    # Handle SFT checkpoint
    if sft_checkpoint:
        # Check if it's a LoRA checkpoint (has adapter_config.json)
        adapter_config_path = os.path.join(sft_checkpoint, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            if args.no_lora:
                # FULL FINE-TUNING: Merge SFT LoRA into base model, then train all parameters
                logger.info("=" * 60)
                logger.info("FULL FINE-TUNING WORKFLOW:")
                logger.info("  1. Loading SFT checkpoint (LoRA adapter)")
                logger.info("  2. Merging LoRA weights into base model")
                logger.info("  3. Training ALL model parameters with RLTT")
                logger.info("=" * 60)
                merged_model_path = os.path.abspath(os.path.join(args.output_dir, "merged_model"))
                model_path = merge_lora_checkpoint(
                    base_model_path=model_path,
                    lora_path=sft_checkpoint,
                    output_path=merged_model_path,
                    rltt_logprob_chunk_size=args.rltt_logprob_chunk_size,
                )
                model_path = os.path.abspath(model_path)
                logger.info(f"Merged model saved to: {model_path}")
                logger.info("Training will update ALL parameters (full fine-tuning)")
            else:
                # LoRA enabled - merge SFT checkpoint into base model, then verl will apply fresh LoRA
                logger.info("=" * 60)
                logger.info("LoRA FINE-TUNING WORKFLOW:")
                logger.info("  1. Loading SFT checkpoint (LoRA adapter)")
                logger.info("  2. Merging LoRA weights into base model")
                logger.info("  3. Applying fresh LoRA adapters for RLTT training")
                logger.info("=" * 60)
                merged_model_path = os.path.abspath(os.path.join(args.output_dir, "merged_model"))
                model_path = merge_lora_checkpoint(
                    base_model_path=model_path,
                    lora_path=sft_checkpoint,
                    output_path=merged_model_path,
                    rltt_logprob_chunk_size=args.rltt_logprob_chunk_size,
                )
                model_path = os.path.abspath(model_path)
                logger.info(f"Merged model saved to: {model_path}")
                logger.info("Training will update only LoRA adapter parameters")
        else:
            # It's a full model checkpoint (not LoRA adapter)
            if args.use_rltt_model:
                # SFT checkpoint is a full model, but we need RLTT model architecture for exit_pdf
                # We need to use RLTT model's Python files with SFT checkpoint's weights
                logger.info("=" * 60)
                logger.info("RLTT MODEL ARCHITECTURE WITH FULL SFT CHECKPOINT")
                logger.info("Combining SFT weights with RLTT model architecture for exit_pdf support...")
                logger.info("=" * 60)

                # Create a new directory that combines:
                # - SFT checkpoint's weights (*.safetensors, *.bin)
                # - RLTT model's Python files (modeling_ouro.py, configuration_ouro.py, etc.)
                # - RLTT model's config.json (has RLTT-specific settings)
                rltt_sft_model_path = os.path.abspath(os.path.join(args.output_dir, "rltt_sft_model"))

                # Check if already exists
                config_json_path = os.path.join(rltt_sft_model_path, "config.json")
                if not os.path.exists(config_json_path):
                    os.makedirs(rltt_sft_model_path, exist_ok=True)
                    import shutil
                    import glob

                    # Copy weight files from SFT checkpoint
                    for weight_file in glob.glob(os.path.join(sft_checkpoint, "*.safetensors")) + \
                                       glob.glob(os.path.join(sft_checkpoint, "*.bin")) + \
                                       glob.glob(os.path.join(sft_checkpoint, "model.safetensors.index.json")):
                        shutil.copy2(weight_file, rltt_sft_model_path)
                        logger.info(f"  Copied weight file: {os.path.basename(weight_file)}")

                    # Copy tokenizer files from SFT checkpoint
                    for tok_file in glob.glob(os.path.join(sft_checkpoint, "tokenizer*")) + \
                                    glob.glob(os.path.join(sft_checkpoint, "special_tokens*")) + \
                                    glob.glob(os.path.join(sft_checkpoint, "vocab*")) + \
                                    glob.glob(os.path.join(sft_checkpoint, "merges*")):
                        shutil.copy2(tok_file, rltt_sft_model_path)

                    # Copy RLTT model's Python files (contain exit_pdf functionality)
                    rltt_base = args.rltt_model_path
                    for py_file in glob.glob(os.path.join(rltt_base, "*.py")):
                        shutil.copy2(py_file, rltt_sft_model_path)
                        logger.info(f"  Copied RLTT model file: {os.path.basename(py_file)}")

                    # Copy config.json from RLTT model (has RLTT-specific config)
                    rltt_config_path = os.path.join(rltt_base, "config.json")
                    if os.path.exists(rltt_config_path):
                        shutil.copy2(rltt_config_path, rltt_sft_model_path)
                        logger.info(f"  Copied RLTT config.json")

                    # Copy chat template if exists
                    for template_file in glob.glob(os.path.join(sft_checkpoint, "*.jinja")) + \
                                         glob.glob(os.path.join(rltt_base, "*.jinja")):
                        shutil.copy2(template_file, rltt_sft_model_path)

                # Update rltt_logprob_chunk_size in config if needed
                if args.rltt_logprob_chunk_size is not None:
                    with open(config_json_path, "r") as f:
                        config_dict = json.load(f)
                    if config_dict.get("rltt_logprob_chunk_size") != args.rltt_logprob_chunk_size:
                        config_dict["rltt_logprob_chunk_size"] = args.rltt_logprob_chunk_size
                        with open(config_json_path, "w") as f:
                            json.dump(config_dict, f, indent=2)

                model_path = rltt_sft_model_path
                logger.info(f"Created RLTT-compatible model at: {model_path}")
            else:
                # No RLTT model needed, use SFT checkpoint directly
                model_path = os.path.abspath(sft_checkpoint)
                logger.info(f"Using full model checkpoint: {model_path}")

            if args.no_lora:
                logger.info("Training will update ALL parameters (full fine-tuning)")
            else:
                logger.info("Training will update only LoRA adapter parameters")

    # Update config with model path
    verl_config["actor_rollout_ref"]["model"]["path"] = model_path

    # Update the model's config.json with rltt_logprob_chunk_size
    # This is needed when the model wasn't merged (i.e., using base model or full checkpoint directly)
    if args.rltt_logprob_chunk_size is not None:
        model_config_path = os.path.join(model_path, "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                model_config_dict = json.load(f)
            if model_config_dict.get("rltt_logprob_chunk_size") != args.rltt_logprob_chunk_size:
                model_config_dict["rltt_logprob_chunk_size"] = args.rltt_logprob_chunk_size
                with open(model_config_path, "w") as f:
                    json.dump(model_config_dict, f, indent=2)

    # Add RLTT-specific config to verl config
    # NOTE: These must be inside actor_rollout_ref because verl passes config.actor_rollout_ref to workers
    verl_config["actor_rollout_ref"]["loop_weighting"] = args.loop_weighting
    verl_config["actor_rollout_ref"]["progressive_alpha"] = args.progressive_alpha
    verl_config["actor_rollout_ref"]["total_ut_steps"] = args.total_ut_steps
    verl_config["actor_rollout_ref"]["rltt_logprob_chunk_size"] = args.rltt_logprob_chunk_size

    # Add learned weights config if using learned weighting
    if args.loop_weighting == "learned":
        verl_config["actor_rollout_ref"]["learned_weights_init"] = args.learned_weights_init
        verl_config["actor_rollout_ref"]["learned_weights_temp"] = args.learned_weights_temp
        verl_config["actor_rollout_ref"]["learned_weights_lr"] = args.learned_weights_lr

    # Load tokenizer for data preparation
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data in parquet format
    data_dir = os.path.abspath(os.path.join(args.output_dir, "data"))
    train_parquet = prepare_train_data(
        args.train_file,
        data_dir,
        tokenizer=tokenizer,
        use_few_shot=False,
    )
    train_parquet = os.path.abspath(train_parquet)

    # Calculate total training steps for LR scheduler
    # One step = one prompt batch
    if args.max_steps > 0:
        # Use max_steps directly if specified
        total_training_steps = args.max_steps
    else:
        # Compute from epochs: steps_per_epoch * num_epochs
        import pyarrow.parquet as pq
        train_table = pq.read_table(train_parquet)
        num_train_samples = len(train_table)
        steps_per_epoch = (num_train_samples + args.num_prompts_per_batch - 1) // args.num_prompts_per_batch
        total_training_steps = steps_per_epoch * args.num_train_epochs
        logger.info(f"  Training samples: {num_train_samples}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")

    # Calculate warmup steps (one step = one prompt batch)
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    logger.info(f"LR Schedule:")
    logger.info(f"  Scheduler type: {args.lr_scheduler_type}")
    logger.info(f"  Total steps (batches): {total_training_steps}")
    logger.info(f"  Warmup steps (batches): {warmup_steps} ({args.warmup_ratio*100:.0f}%)")
    logger.info(f"  Prompts per batch: {args.num_prompts_per_batch}")

    # Update trainer config with total_training_steps
    verl_config["trainer"]["total_training_steps"] = total_training_steps

    # Update optim config with scheduler settings (verl format)
    verl_config["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps_ratio"] = args.warmup_ratio
    verl_config["actor_rollout_ref"]["actor"]["optim"]["warmup_style"] = args.lr_scheduler_type
    verl_config["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = total_training_steps
    verl_config["actor_rollout_ref"]["actor"]["optim"]["min_lr_ratio"] = 0.0  # decay to 0 for cosine
    verl_config["actor_rollout_ref"]["actor"]["optim"]["lr_scheduler_type"] = args.lr_scheduler_type

    # Prepare test data if validation is enabled
    test_files = []
    if not args.no_validation:
        test_parquet = prepare_test_data(
            args.math500_test_file,
            data_dir,
            tokenizer=tokenizer,
            use_few_shot=False,
        )
        test_parquet = os.path.abspath(test_parquet)
        test_files.append(test_parquet)

    # Update config with data paths
    verl_config["data"]["train_files"] = [train_parquet]
    if test_files:
        verl_config["data"]["val_files"] = test_files
    else:
        verl_config["data"]["val_files"] = [train_parquet]

    # Save final config
    config_path = os.path.join(args.output_dir, "verl_config.json")
    with open(config_path, "w") as f:
        json.dump(verl_config, f, indent=2)

    logger.info(f"Saved verl config to: {config_path}")
    logger.info(f"RLTT Configuration:")
    logger.info(f"  Loop weighting: {args.loop_weighting}")
    logger.info(f"  Progressive alpha: {args.progressive_alpha}")
    logger.info(f"  Total UT steps: {args.total_ut_steps}")
    logger.info(f"  Log-prob chunk size: {args.rltt_logprob_chunk_size}")

    # Convert to OmegaConf for verl
    config = OmegaConf.create(verl_config)

    # Determine number of GPUs
    n_gpus = args.n_gpus or torch.cuda.device_count()

    # Set up resource pool
    # All roles colocated on same GPUs for hybrid engine
    resource_pool_spec = {
        "actor_rollout": [n_gpus] * args.nnodes,
    }
    mapping = {
        Role.ActorRollout: "actor_rollout",
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    # Create role-worker mapping with RLTT worker
    role_worker_mapping = {
        Role.ActorRollout: RLTTActorRolloutRefWorker,
    }

    # Create reward function that matches verl's interface
    class MathRewardManager:
        """Reward manager for math problems compatible with verl."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, data, return_dict=False):
            """Compute rewards for a batch of math problem responses.

            Args:
                data: DataProto object containing batched data
                return_dict: If True, return dict with reward_tensor and extra info

            Returns:
                Reward tensor or dict with reward_tensor
            """
            import torch

            # Initialize reward tensor with same shape as responses
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            for i in range(len(data)):
                data_item = data[i]

                # Get response tokens and decode
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode the response
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                # Get ground truth from non_tensor_batch
                ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
                if not ground_truth:
                    # Fallback: try to get from 'answer' field
                    ground_truth = data_item.non_tensor_batch.get("answer", "")

                # Get extra_info for logging (contains problem text, level, type, etc.)
                extra_info = data_item.non_tensor_batch.get("extra_info", {})

                # Compute reward using our math reward function
                reward = compute_math_reward(
                    data_source="math",
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                # Place reward at the last valid response token position
                reward_tensor[i, valid_response_length - 1] = reward

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor

    reward_fn = MathRewardManager(tokenizer)

    logger.info("Starting verl training with RLTT worker...")

    # Initialize Ray if not already done
    import ray
    if not ray.is_initialized():
        ray.init()

    # Create resource pool
    resource_pool_manager.create_resource_pool()

    # Create trainer with timing logging
    trainer = TimedRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
    )

    # Initialize workers
    trainer.init_workers()

    # Run training
    trainer.fit()

    logger.info("RLTT verl training completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
