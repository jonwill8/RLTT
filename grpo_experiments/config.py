"""
GRPO Training Configuration for verl + vLLM accelerated training.

Based on grpo_experiments_v0 but using verl framework for multi-GPU
training with vLLM-accelerated rollout generation.

Features:
- vLLM-accelerated rollout generation
- FSDP for multi-GPU training
- Same CLI arguments as grpo_experiments_v0 for compatibility
- Offline support for HPC environments
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# Resolve paths relative to this file's location
_THIS_DIR = Path(__file__).parent.resolve()
_RLTT_ROOT = _THIS_DIR.parent


def get_default_sft_checkpoint() -> str:
    """Get default SFT checkpoint path relative to RLTT root."""
    return str(_RLTT_ROOT / "sft_experiments" / "sft_ouro_math_output")


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"

    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_cache: bool = False  # Disable for training

    # Fixed recurrent loops for Ouro model
    total_ut_steps: int = 4

    # Initialize from SFT checkpoint (LoRA adapter)
    sft_checkpoint_path: Optional[str] = None
    # Default: relative to RLTT root (../sft_experiments/sft_ouro_math_output from grpo_experiments/)
    default_sft_checkpoint: str = ""  # Computed at runtime via get_default_sft_checkpoint()


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    preprocessing_num_workers: int = 4

    # Data format for verl
    prompt_key: str = "prompt"
    answer_key: str = "answer"


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # Use "all-linear" to apply LoRA to all linear layers for maximum expressiveness
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"


@dataclass
class VLLMRolloutConfig:
    """vLLM rollout configuration for accelerated generation."""
    # Use vLLM for generation (recommended)
    name: str = "vllm"  # "vllm" or "hf"

    # vLLM specific settings
    tensor_model_parallel_size: int = 1  # Number of GPUs for TP during inference
    gpu_memory_utilization: float = 0.7  # GPU memory fraction for vLLM
    enforce_eager: bool = False  # False = enable CUDA graphs
    free_cache_engine: bool = False  # Keep cache between batches
    enable_prefix_caching: bool = True  # Enable prefix caching

    # Generation parameters
    n: int = 8  # Number of samples per prompt (GRPO group size)
    temperature: float = 0.9
    top_p: float = 1.0
    max_response_length: int = 4096


@dataclass
class ActorConfig:
    """Actor (policy) model training configuration."""
    # GRPO-specific
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"

    # PPO mini-batch size (for gradient updates)
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = 8

    # Dynamic batch size optimization
    use_dynamic_bsz: bool = True
    ppo_max_token_len_per_gpu: int = 48000

    # Optimizer
    optim_lr: float = 1e-6
    optim_weight_decay: float = 0.1
    optim_betas: tuple = (0.9, 0.99)
    max_grad_norm: float = 0.1

    # Schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

    # FSDP settings
    fsdp_param_offload: bool = False
    fsdp_optimizer_offload: bool = False

    # Gradient checkpointing
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = True


@dataclass
class RefConfig:
    """Reference model configuration."""
    log_prob_max_token_len_per_gpu: int = 64000
    fsdp_param_offload: bool = True  # Offload ref model to save memory


@dataclass
class GRPOTrainingConfig:
    """GRPO Training hyperparameters."""
    output_dir: str = "./grpo_output"

    # === GRPO-specific hyperparameters ===
    beta: float = 0.001  # KL coefficient
    num_generations: int = 8  # Samples per prompt (same as rollout.n)
    num_prompts_per_batch: int = 128  # Unique prompts per batch

    # Total batch = num_prompts_per_batch * num_generations
    # e.g., 128 * 8 = 1024 samples per batch

    # === Sequence lengths ===
    max_prompt_length: int = 1024
    max_completion_length: int = 4096

    # === Training duration ===
    total_epochs: int = 20

    # === Precision ===
    bf16: bool = True
    fp16: bool = False

    # === Saving and Validation ===
    save_freq: int = 7  # Save every N epochs
    test_freq: int = 7  # Validate every N epochs
    val_before_train: bool = True  # Run validation before training

    # === Random seed ===
    seed: int = 42

    # === Data filtering ===
    filter_overlong_prompts: bool = True
    truncation: str = "error"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    # Multi-GPU / Multi-node settings
    n_gpus_per_node: int = 8
    nnodes: int = 1

    # Resource pool configuration
    # Default: all GPUs in single pool for actor+rollout+ref
    use_separate_pools: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = True
    wandb_project: str = "grpo-ouro-math-verl"
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = True

    experiment_name: str = "GRPO-Ouro-MATH"

    # Rollout logging
    log_rollouts: bool = True
    rollout_log_freq: int = 1  # Log rollouts every N steps


@dataclass
class ValidationConfig:
    """Validation configuration."""
    # Validation frequency (in epochs, aligned with verl)
    eval_epochs: int = 7

    # Generation settings for validation
    max_prompt_length: int = 1024
    max_new_tokens: int = 4096
    temperature: float = 0.0  # Greedy decoding for evaluation
    do_sample: bool = False
    use_few_shot: bool = False

    # MATH-500 test file
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
    math500_samples: Optional[int] = None  # None = all 500

    # Batch size for evaluation
    eval_batch_size: int = 64


@dataclass
class GRPOConfig:
    """Combined GRPO configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rollout: VLLMRolloutConfig = field(default_factory=VLLMRolloutConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
