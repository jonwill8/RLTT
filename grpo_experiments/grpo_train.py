#!/usr/bin/env python3
"""
GRPO Training Script using verl + vLLM for accelerated rollout generation.

This script provides the same CLI interface as grpo_experiments_v0 but uses
the verl framework for multi-GPU training with vLLM-accelerated rollouts.

Features:
- vLLM-accelerated rollout generation (3-10x faster than HF generation)
- FSDP for multi-GPU training
- Same CLI arguments as grpo_experiments_v0 for compatibility
- Offline support for HPC environments
- Checkpoint loading from SFT experiments
- Rollout logging to JSONL and wandb

Usage:
    # Single GPU (uses HF fallback)
    python grpo_train.py --model_path /path/to/model

    # Multi-GPU with verl + vLLM
    python grpo_train.py --model_path /path/to/model --n_gpus 8

    # With custom hyperparameters
    python grpo_train.py --beta 0.001 --num_generations 8 --learning_rate 1e-6
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
)
logger = logging.getLogger(__name__)


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


def patch_ouro_model_compatibility(model_path: str) -> None:
    """Apply compatibility fixes for Ouro custom model code.

    Newer transformers versions can fail with:
    `TypeError: check_model_inputs.<locals>.wrapped_fn() got an unexpected
    keyword argument 'input_ids'` when OuroModel.forward is decorated with
    `@check_model_inputs`. Removing that decorator restores the expected forward
    signature for FSDP/verl actor calls.
    """
    modeling_path = os.path.join(model_path, "modeling_ouro.py")
    if not os.path.exists(modeling_path):
        return

    try:
        with open(modeling_path, "r", encoding="utf-8") as f:
            content = f.read()

        old_block = "    @check_model_inputs\n    @auto_docstring\n    def forward("
        new_block = "    @auto_docstring\n    def forward("

        if old_block not in content:
            return

        content = content.replace(old_block, new_block, 1)

        with open(modeling_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            "Applied Ouro compatibility patch: removed @check_model_inputs from "
            f"{modeling_path}"
        )
    except Exception as e:
        logger.warning(f"Failed to apply Ouro compatibility patch: {e}")


def parse_args():
    """Parse command line arguments - same interface as grpo_experiments_v0."""
    parser = argparse.ArgumentParser(
        description="GRPO Training with verl + vLLM acceleration"
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

    # === GRPO-specific hyperparameters ===
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
    parser.add_argument(
        "--use_ppo",
        action="store_true",
        help="Use PPO algorithm instead of GRPO. This enables a critic (value function) "
             "for GAE-based advantage estimation and uses a clipped surrogate objective. "
             "Requires more GPU memory due to the critic model."
    )
    parser.add_argument(
        "--ppo_clip_ratio",
        type=float,
        default=0.2,
        help="PPO clipping ratio for the surrogate objective (only used with --use_ppo)"
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per batch (only used with --use_ppo)"
    )
    parser.add_argument(
        "--gae_gamma",
        type=float,
        default=1.0,
        help="GAE discount factor (only used with --use_ppo)"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation (only used with --use_ppo)"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=1e-5,
        help="Learning rate for the critic model (only used with --use_ppo)"
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (only used with --use_ppo)"
    )
    parser.add_argument(
        "--critic_warmup",
        type=int,
        default=0,
        help="Number of steps to train critic before updating actor (only used with --use_ppo)"
    )
    parser.add_argument(
        "--cliprange_value",
        type=float,
        default=0.5,
        help="Value function clipping range (only used with --use_ppo)"
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
        help="Warmup ratio"
    )

    # === Batch sizes ===
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )

    # === Dynamic batch size token limits ===
    parser.add_argument(
        "--ppo_max_token_len",
        type=int,
        default=16384,
        help="Max token length per GPU for actor update (dynamic batching). "
             "Lower values reduce GPU memory usage during backprop."
    )
    parser.add_argument(
        "--log_prob_max_token_len",
        type=int,
        default=16384,
        help="Max token length per GPU for log-prob computation (dynamic batching). "
             "Lower values reduce GPU memory during rollout log-prob calculation."
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
        help="Checkpoint save frequency (in training steps, where 1 step = backprop over num_prompts_per_batch × num_generations rollouts)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="grpo-ouro-math-verl",
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
        default="./grpo_output",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader (reduce to save CPU RAM if OOM)"
    )

    return parser.parse_args()


def merge_lora_checkpoint(base_model_path: str, lora_path: str, output_path: str) -> str:
    """Merge LoRA adapter into base model for verl compatibility.

    verl doesn't support LoRA natively, so we merge the adapter into
    the base model and save the merged model.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model

    Returns:
        Path to merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA adapter into base model...")
    logger.info(f"  Base model: {base_model_path}")
    logger.info(f"  LoRA adapter: {lora_path}")
    logger.info(f"  Output: {output_path}")

    # Check if already merged (must have config.json AND custom model files)
    import glob as glob_module
    base_py_files = glob_module.glob(os.path.join(base_model_path, "*.py"))
    output_has_py_files = all(
        os.path.exists(os.path.join(output_path, os.path.basename(f)))
        for f in base_py_files
    ) if base_py_files else True

    if os.path.exists(os.path.join(output_path, "config.json")) and output_has_py_files:
        logger.info(f"Merged model already exists at {output_path}, skipping merge")
        return output_path
    elif os.path.exists(os.path.join(output_path, "config.json")) and not output_has_py_files:
        # Model exists but missing custom files - copy them
        logger.info(f"Merged model exists but missing custom files, copying...")
        import shutil
        for src_file in base_py_files:
            dst_file = os.path.join(output_path, os.path.basename(src_file))
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied custom model file: {os.path.basename(src_file)}")
        return output_path

    os.makedirs(output_path, exist_ok=True)

    # Load base model
    logger.info("Loading base model...")
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
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        is_trainable=False,
    )

    # Merge and unload
    logger.info("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Copy custom model files (modeling_*.py, configuration_*.py) for trust_remote_code models
    import shutil
    import glob
    custom_files = glob.glob(os.path.join(base_model_path, "*.py"))
    for src_file in custom_files:
        dst_file = os.path.join(output_path, os.path.basename(src_file))
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied custom model file: {os.path.basename(src_file)}")

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


def build_verl_config(args) -> Dict[str, Any]:
    """Build verl configuration from command line arguments."""

    # Determine number of GPUs
    n_gpus = args.n_gpus or torch.cuda.device_count()
    if n_gpus == 0:
        n_gpus = 1
        logger.warning("No GPUs detected, using CPU (not recommended)")

    # Determine vLLM tensor parallel size
    # Use the explicitly provided TP size (no auto-enabling)
    tp_size = args.vllm_tensor_parallel_size
    tp_size = min(tp_size, n_gpus)
    if tp_size > 1 and n_gpus % tp_size != 0:
        tp_size = 1
        logger.warning(f"TP size {args.vllm_tensor_parallel_size} doesn't divide {n_gpus} GPUs, using TP=1")

    # Compute total batch size
    train_batch_size = args.num_prompts_per_batch * args.num_generations

    # Determine rollout method
    use_vllm = args.use_vllm and not args.no_vllm

    # Use single max_prompt_length for vLLM model length
    # Run validation separately on checkpoints if different prompt length needed
    max_prompt_for_model = args.max_prompt_length

    config = {
        # Algorithm - see verl/trainer/config/ppo_trainer.yaml for all fields
        "algorithm": {
            "gamma": args.gae_gamma if args.use_ppo else 1.0,
            "lam": args.gae_lambda if args.use_ppo else 1.0,
            "adv_estimator": "gae" if args.use_ppo else "grpo",
            "norm_adv_by_std_in_grpo": True,  # Standard GRPO std normalization
            "use_kl_in_reward": True,  # Add KL penalty to reward signal
            "kl_penalty": "kl",
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": args.beta,
            },
        },

        # Data - see verl/trainer/config/data/legacy_data.yaml for all fields
        "data": {
            "train_batch_size": train_batch_size,
            "val_batch_size": args.val_batch_size,
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
                "enable_gradient_checkpointing": True,
                "trust_remote_code": True,  # Required for custom model code (e.g., Ouro)
                # Note: dtype is set in the saved model config (torch_dtype=bfloat16)
                # verl reads this from the model's config.json, not from CLI args
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
                # PPO uses multiple epochs per batch with clipping; GRPO uses single epoch
                "ppo_epochs": args.ppo_epochs if args.use_ppo else 1,
                "entropy_coeff": args.entropy_coeff if args.use_ppo else 0.0,
                "use_kl_loss": False,
                "kl_loss_coef": args.beta,
                "kl_loss_type": "low_var_kl",
                "clip_ratio": args.ppo_clip_ratio if args.use_ppo else 0.2,
                "ppo_mini_batch_size": min(128, train_batch_size),
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": None,  # Use dynamic batching instead
                "use_dynamic_bsz": True,
                # Token limits for dynamic batching during actor update (backprop)
                "ppo_max_token_len_per_gpu": args.ppo_max_token_len,
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "loss_agg_mode": "token-mean",
                "optim": {
                    "_target_": "verl.workers.config.FSDPOptimizerConfig",
                    # Use 8-bit Adam to save ~50% optimizer memory
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
                "free_cache_engine": False,  # Keep cache for throughput (like RLVR-Decomposed)
                "enable_prefix_caching": True,  # Cache prompt prefixes across generations
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": None,
                # Token limits for dynamic batching during log-prob computation
                "log_prob_max_token_len_per_gpu": args.log_prob_max_token_len,
                "log_prob_use_dynamic_bsz": True,  # Dynamic batching for rollout log probs
                "temperature": args.temperature,
                "max_model_len": max_prompt_for_model + args.max_completion_length,
                "enable_chunked_prefill": True,  # Higher throughput (like RLVR-Decomposed)
                "prompt_length": args.max_prompt_length,
                "response_length": args.max_completion_length,
                "multi_turn": {
                    "enable": False,
                },
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": None,
                # Token limits for dynamic batching during ref log-prob computation
                "log_prob_max_token_len_per_gpu": args.log_prob_max_token_len,
                "log_prob_use_dynamic_bsz": True,  # Dynamic batching for ref log probs
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": False,
                "fsdp_config": {
                    "_target_": "verl.workers.config.FSDPEngineConfig",
                    "param_offload": False,
                    "dtype": "bfloat16",
                },
            },
            "hybrid_engine": True,  # Required by verl RayPPOTrainer
        },

        # Critic (enabled for PPO with GAE, disabled for GRPO)
        "critic": {
            "enable": args.use_ppo,
            **(
                {
                    "_target_": "verl.workers.config.FSDPCriticConfig",
                    "strategy": "fsdp",
                    "ppo_mini_batch_size": min(128, train_batch_size),
                    "ppo_micro_batch_size": None,
                    "ppo_micro_batch_size_per_gpu": None,
                    "use_dynamic_bsz": True,
                    "ppo_max_token_len_per_gpu": args.ppo_max_token_len,
                    "ppo_epochs": args.ppo_epochs,
                    "cliprange_value": args.cliprange_value,
                    "loss_agg_mode": "token-mean",
                    "grad_clip": args.max_grad_norm,
                    "model": {
                        "_target_": "verl.workers.config.FSDPCriticModelCfg",
                        "path": args.model_path,
                        "tokenizer_path": args.model_path,
                        "lora_rank": 0,  # Critic uses full parameters
                        "enable_gradient_checkpointing": True,
                        "use_remove_padding": False,
                        "trust_remote_code": True,
                        "fsdp_config": {
                            "_target_": "verl.workers.config.FSDPEngineConfig",
                            "param_offload": False,
                            "optimizer_offload": False,
                            "fsdp_size": -1,
                            "dtype": "bfloat16",
                        },
                    },
                    "optim": {
                        "_target_": "verl.workers.config.FSDPOptimizerConfig",
                        "optimizer": "AdamW8bit",
                        "optimizer_impl": "bitsandbytes.optim",
                        "lr": args.critic_lr,
                        "weight_decay": args.weight_decay,
                        "betas": [0.9, 0.999],
                        "override_optimizer_config": None,
                    },
                    "checkpoint": {
                        "_target_": "verl.trainer.config.CheckpointConfig",
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": ["model", "optimizer", "extra"],
                        "async_save": False,
                    },
                }
                if args.use_ppo
                else {}
            ),
        },

        # Trainer
        "trainer": {
            "experiment_name": f"{'PPO' if args.use_ppo else 'GRPO'}-{os.path.basename(args.model_path)}",
            "project_name": args.wandb_project,
            "logger": ["wandb"] if not args.no_wandb else [],
            "n_gpus_per_node": n_gpus,
            "nnodes": args.nnodes,
            "total_epochs": args.num_train_epochs,
            "save_freq": args.save_steps,
            "test_freq": -1 if args.no_validation else args.eval_steps,  # -1 disables validation in verl
            "val_before_train": False if args.no_validation else (not args.skip_initial_validation),
            "critic_warmup": args.critic_warmup if args.use_ppo else 0,
            "validation_data_dir": os.path.join(args.output_dir, "validation_outputs"),
            "device": "cuda",
            "total_training_steps": None,
            "resume_mode": "disable",
            "esi_redundant_time": 0,  # ESI checkpoint timing buffer
            "balance_batch": False,
            "default_local_dir": args.output_dir,  # Directory for checkpoints
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
    algo_name = "PPO" if args.use_ppo else "GRPO"
    logger.info(f"{algo_name} Training with verl + vLLM Acceleration")
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
    if args.use_ppo:
        logger.info(f"")
        logger.info(f"*** ALGORITHM: PPO (Proximal Policy Optimization) ***")
        logger.info(f"*** Critic-based GAE advantage estimation ***")
        logger.info(f"*** Clipped surrogate objective (clip={args.ppo_clip_ratio}) ***")
        logger.info(f"*** PPO epochs: {args.ppo_epochs}, Entropy coeff: {args.entropy_coeff} ***")
        logger.info(f"*** GAE gamma={args.gae_gamma}, lambda={args.gae_lambda} ***")
        logger.info(f"")
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
    logger.info(f"PPO max token len: {args.ppo_max_token_len}")
    logger.info(f"Log-prob max token len: {args.log_prob_max_token_len}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"vLLM enabled: {args.use_vllm and not args.no_vllm}")

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
        # Single GPU or debug mode - use simplified training
        logger.info("Using simplified single-GPU training (no verl)")
        run_simple_training(args, sft_checkpoint)
    else:
        # Multi-GPU - use verl
        logger.info(f"Using verl for {n_gpus}-GPU training")
        run_verl_training(args, verl_config, sft_checkpoint)


def run_simple_training(args, sft_checkpoint: Optional[str] = None):
    """Run simplified single-GPU training (fallback when verl is not needed).

    This uses a similar approach to grpo_experiments_v0 but with vLLM support.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel, LoraConfig, get_peft_model
    import copy

    from data_utils import load_jsonl, prepare_train_data
    from reward_utils import create_math_reward_func, process_math_answer

    set_seed(args.seed)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )

    # Set Ouro model config
    if hasattr(model.config, "total_ut_steps"):
        model.config.total_ut_steps = args.total_ut_steps

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
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    sft_checkpoint,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map="auto",
                )
                if hasattr(model.config, "total_ut_steps"):
                    model.config.total_ut_steps = args.total_ut_steps

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

    logger.info("Starting simple GRPO training...")
    logger.info("Note: For multi-GPU training with vLLM acceleration, use --n_gpus > 1")

    # Import and run simple training loop
    from simple_trainer import SimpleGRPOTrainer

    trainer = SimpleGRPOTrainer(
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
    """Run verl-based multi-GPU training with vLLM acceleration."""
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

    # Create custom trainer with timing logging (adapted from rltt_train.py)
    class TimedRayPPOTrainer(RayPPOTrainer):
        """RayPPOTrainer with prominent timing logging for rollout vs backprop."""

        def fit(self):
            """Training loop with timing logging."""
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
                compute_response_mask, reduce_metrics,
                compute_advantage, apply_kl_penalty,
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
                    })
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    tracking_logger.log(data=metrics, step=self.global_steps)

                    # Update progress bar with timing
                    postfix = {
                        "rollout": f"{rollout_time:.1f}s",
                        "backprop": f"{backprop_time:.1f}s",
                    }
                    if grad_accum_steps > 1:
                        postfix["accum"] = f"{accum_count}/{grad_accum_steps}"
                    progress_bar.set_postfix(postfix)
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

    # Import custom GRPO worker with parameter logging
    try:
        from verl_grpo import GRPOActorRolloutRefWorker
        logger.info("Using custom GRPOActorRolloutRefWorker with parameter logging")
    except ImportError:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from verl_grpo import GRPOActorRolloutRefWorker
        logger.info("Using custom GRPOActorRolloutRefWorker with parameter logging")

    from transformers import AutoTokenizer
    from data_utils import prepare_train_data, prepare_test_data
    from math_utils import compute_score as compute_math_reward

    logger.info("Setting up verl training...")

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

    # Handle SFT checkpoint
    # IMPORTANT: verl's tokenizer loading validates paths as HuggingFace repo IDs
    # and rejects relative paths like "./foo". We must use absolute paths.
    model_path = os.path.abspath(args.model_path)
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
                logger.info(f"  3. Training ALL model parameters with {'PPO' if args.use_ppo else 'GRPO'}")
                logger.info("=" * 60)
                merged_model_path = os.path.abspath(os.path.join(args.output_dir, "merged_model"))
                model_path = merge_lora_checkpoint(
                    base_model_path=args.model_path,
                    lora_path=sft_checkpoint,
                    output_path=merged_model_path,
                )
                model_path = os.path.abspath(model_path)
                verl_config["actor_rollout_ref"]["model"]["path"] = model_path
                logger.info(f"Merged model saved to: {model_path}")
                logger.info("Training will update ALL parameters (full fine-tuning)")
            else:
                # LoRA enabled - merge SFT checkpoint into base model, then verl will apply fresh LoRA
                # This initializes from SFT weights but trains with new LoRA adapters
                logger.info("=" * 60)
                logger.info("LoRA FINE-TUNING WORKFLOW:")
                logger.info("  1. Loading SFT checkpoint (LoRA adapter)")
                logger.info("  2. Merging LoRA weights into base model")
                logger.info(f"  3. Applying fresh LoRA adapters for {'PPO' if args.use_ppo else 'GRPO'} training")
                logger.info("=" * 60)
                merged_model_path = os.path.abspath(os.path.join(args.output_dir, "merged_model"))
                model_path = merge_lora_checkpoint(
                    base_model_path=args.model_path,
                    lora_path=sft_checkpoint,
                    output_path=merged_model_path,
                )
                model_path = os.path.abspath(model_path)
                verl_config["actor_rollout_ref"]["model"]["path"] = model_path
                logger.info(f"Merged model saved to: {model_path}")
                logger.info("Training will update only LoRA adapter parameters")
        else:
            # It's a full model checkpoint (not LoRA adapter)
            # Check if it has custom model files - if not, we need to copy them from base model
            import glob as glob_module
            base_py_files = glob_module.glob(os.path.join(args.model_path, "*.py"))
            sft_has_py_files = all(
                os.path.exists(os.path.join(sft_checkpoint, os.path.basename(f)))
                for f in base_py_files
            ) if base_py_files else True

            if not sft_has_py_files:
                # SFT checkpoint is missing custom model files - create a combined directory
                logger.info("=" * 60)
                logger.info("FULL SFT CHECKPOINT WITH CUSTOM MODEL FILES")
                logger.info("Combining SFT weights with base model's custom files...")
                logger.info("=" * 60)

                combined_model_path = os.path.abspath(os.path.join(args.output_dir, "combined_sft_model"))

                # Check if already exists
                config_json_path = os.path.join(combined_model_path, "config.json")
                if not os.path.exists(config_json_path):
                    os.makedirs(combined_model_path, exist_ok=True)
                    import shutil

                    # Copy weight files from SFT checkpoint
                    for weight_file in glob_module.glob(os.path.join(sft_checkpoint, "*.safetensors")) + \
                                       glob_module.glob(os.path.join(sft_checkpoint, "*.bin")) + \
                                       glob_module.glob(os.path.join(sft_checkpoint, "model.safetensors.index.json")):
                        shutil.copy2(weight_file, combined_model_path)
                        logger.info(f"  Copied weight file: {os.path.basename(weight_file)}")

                    # Copy tokenizer files from SFT checkpoint
                    for tok_file in glob_module.glob(os.path.join(sft_checkpoint, "tokenizer*")) + \
                                    glob_module.glob(os.path.join(sft_checkpoint, "special_tokens*")) + \
                                    glob_module.glob(os.path.join(sft_checkpoint, "vocab*")) + \
                                    glob_module.glob(os.path.join(sft_checkpoint, "merges*")):
                        shutil.copy2(tok_file, combined_model_path)

                    # Copy config.json from SFT checkpoint (has trained model config)
                    sft_config_path = os.path.join(sft_checkpoint, "config.json")
                    if os.path.exists(sft_config_path):
                        shutil.copy2(sft_config_path, combined_model_path)
                        logger.info(f"  Copied config.json from SFT checkpoint")

                    # Copy custom Python files from base model (modeling_*.py, configuration_*.py, etc.)
                    for py_file in base_py_files:
                        shutil.copy2(py_file, combined_model_path)
                        logger.info(f"  Copied custom model file: {os.path.basename(py_file)}")

                    # Copy chat template if exists
                    for template_file in glob_module.glob(os.path.join(sft_checkpoint, "*.jinja")) + \
                                         glob_module.glob(os.path.join(args.model_path, "*.jinja")):
                        dst = os.path.join(combined_model_path, os.path.basename(template_file))
                        if not os.path.exists(dst):
                            shutil.copy2(template_file, combined_model_path)

                model_path = combined_model_path
                logger.info(f"Created combined model at: {model_path}")
            else:
                # SFT checkpoint has all needed files, use directly
                model_path = os.path.abspath(sft_checkpoint)
                logger.info(f"Using full model checkpoint: {model_path}")

            verl_config["actor_rollout_ref"]["model"]["path"] = model_path
            if args.no_lora:
                logger.info("Training will update ALL parameters (full fine-tuning)")
            else:
                logger.info("Training will update only LoRA adapter parameters")

    # Sync critic model path with actor model path (may have changed due to SFT merge)
    if args.use_ppo and "model" in verl_config.get("critic", {}):
        verl_config["critic"]["model"]["path"] = verl_config["actor_rollout_ref"]["model"]["path"]
        verl_config["critic"]["model"]["tokenizer_path"] = verl_config["actor_rollout_ref"]["model"]["path"]

    # Compatibility fix for Ouro custom modeling code under newer transformers.
    patch_ouro_model_compatibility(model_path)

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
    # Use absolute paths to avoid issues when verl changes working directory
    data_dir = os.path.abspath(os.path.join(args.output_dir, "data"))
    train_parquet = prepare_train_data(
        args.train_file,
        data_dir,
        tokenizer=tokenizer,
        use_few_shot=False,
    )
    # Convert to absolute path
    train_parquet = os.path.abspath(train_parquet)

    # Prepare test data if validation is enabled
    test_files = []
    if not args.no_validation:
        test_parquet = prepare_test_data(
            args.math500_test_file,
            data_dir,
            tokenizer=tokenizer,
            use_few_shot=False,  # Always zero-shot; run few-shot validation separately on checkpoints
        )
        # Convert to absolute path
        test_parquet = os.path.abspath(test_parquet)
        test_files.append(test_parquet)

    # Update config with data paths (must be absolute paths for verl)
    verl_config["data"]["train_files"] = [train_parquet]
    if test_files:
        verl_config["data"]["val_files"] = test_files
    else:
        # verl unconditionally creates a validation dataset, even with test_freq=-1
        # We must provide the training data as a dummy val_files to avoid errors
        # Since test_freq=-1, validation won't actually run
        verl_config["data"]["val_files"] = [train_parquet]

    # Calculate total training steps for LR scheduler
    # One step = one prompt batch (before gradient accumulation)
    if args.max_steps > 0:
        # Use max_steps directly if specified
        total_training_steps = args.max_steps
    else:
        # Compute from epochs: ceil(num_samples / num_prompts_per_batch) * num_epochs
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
    logger.info(f"  Scheduler type: constant (with warmup)")
    logger.info(f"  Total steps (batches): {total_training_steps}")
    logger.info(f"  Warmup steps (batches): {warmup_steps} ({args.warmup_ratio*100:.0f}%)")
    logger.info(f"  Prompts per batch: {args.num_prompts_per_batch}")

    # Update trainer config with total_training_steps
    verl_config["trainer"]["total_training_steps"] = total_training_steps

    # Update optim config with scheduler settings (verl format)
    verl_config["actor_rollout_ref"]["actor"]["optim"]["lr_warmup_steps_ratio"] = args.warmup_ratio
    verl_config["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = total_training_steps
    verl_config["actor_rollout_ref"]["actor"]["optim"]["min_lr_ratio"] = 0.0  # decay to 0 for cosine
    verl_config["actor_rollout_ref"]["actor"]["optim"]["lr_scheduler_type"] = "constant"

    # Save final config
    config_path = os.path.join(args.output_dir, "verl_config.json")
    with open(config_path, "w") as f:
        json.dump(verl_config, f, indent=2)

    logger.info(f"Saved verl config to: {config_path}")
    logger.info("Starting verl training with custom GRPOActorRolloutRefWorker...")

    # Convert to OmegaConf for verl
    config = OmegaConf.create(verl_config)

    # Determine number of GPUs
    n_gpus = args.n_gpus or torch.cuda.device_count()

    # Set up resource pool - all roles colocated on same GPUs for hybrid engine
    # RefPolicy is included to enable KL regularization (ref_in_actor=True when using LoRA)
    resource_pool_spec = {
        "actor_rollout": [n_gpus] * args.nnodes,
    }
    mapping = {
        Role.ActorRollout: "actor_rollout",
        Role.RefPolicy: "actor_rollout",  # Colocated with actor for ref_in_actor
    }

    # PPO requires a Critic role for value function estimation
    if args.use_ppo:
        mapping[Role.Critic] = "actor_rollout"  # Colocate critic on same GPUs

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    # Create role-worker mapping with custom GRPO worker (includes parameter logging)
    # RefPolicy uses same worker - when using LoRA, ref_in_actor=True so ref log probs
    # are computed by the actor worker using the base model (without LoRA adapters)
    role_worker_mapping = {
        Role.ActorRollout: GRPOActorRolloutRefWorker,
        Role.RefPolicy: GRPOActorRolloutRefWorker,
    }

    # PPO needs a CriticWorker for value function
    if args.use_ppo:
        import ray
        from verl.workers.fsdp_workers import CriticWorker
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    # Create reward function that matches verl's interface
    class MathRewardManager:
        """Reward manager for math problems compatible with verl."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, data, return_dict=False):
            """Compute rewards for a batch of math problem responses."""
            import torch

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            for i in range(len(data)):
                data_item = data[i]

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", "")
                if not ground_truth:
                    ground_truth = data_item.non_tensor_batch.get("answer", "")

                reward = compute_math_reward(
                    data_source="math",
                    solution_str=response_str,
                    ground_truth=ground_truth,
                )

                reward_tensor[i, valid_response_length - 1] = reward

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor

    reward_fn = MathRewardManager(tokenizer)

    # Set environment variables for custom reward function (before Ray workers start)
    os.environ["GRPO_OUTPUT_DIR"] = os.path.abspath(args.output_dir)
    os.environ["GRPO_NUM_GENERATIONS"] = str(verl_config.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 8))
    logger.info(f"Set GRPO_OUTPUT_DIR={os.environ['GRPO_OUTPUT_DIR']}")

    # Initialize Ray if not already done
    import ray
    if not ray.is_initialized():
        ray.init()

    # Create resource pool
    resource_pool_manager.create_resource_pool()

    # Create trainer with custom worker and timing logging
    trainer = TimedRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
    )

    # Initialize workers (this is where parameter logging happens)
    trainer.init_workers()

    # Run training
    trainer.fit()

    logger.info(f"{'PPO' if args.use_ppo else 'GRPO'} verl training completed successfully!")

    # Post-training: Analyze validation outputs for accuracy by question type
    validation_outputs_dir = os.path.join(args.output_dir, "validation_outputs")
    if os.path.exists(validation_outputs_dir):
        logger.info("Analyzing validation outputs for accuracy by question type...")
        try:
            from analyze_validation import analyze_validation_dir
            metrics_output = os.path.join(args.output_dir, "validation_metrics_summary.json")
            analyze_validation_dir(
                validation_outputs_dir,
                output_file=metrics_output,
            )
            logger.info(f"Validation metrics saved to: {metrics_output}")
        except Exception as e:
            logger.warning(f"Failed to analyze validation outputs: {e}")
    else:
        logger.info("No validation outputs found to analyze")


if __name__ == "__main__":
    main()
