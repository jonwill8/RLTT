"""
SFT Training Configuration for Ouro-2.6B-Thinking on MATH dataset.

Uses SimpleSFTTrainer for single-process training with vLLM validation.
Hyperparameters based on Ouro paper Section 4 (SFT settings).
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    use_cache: bool = False  # Disable for training

    # Gradient checkpointing
    enable_gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Parquet files (verl requires parquet format)
    train_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.parquet"
    val_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/math500_test.parquet"

    # Column names in parquet files
    prompt_key: str = "prompt"
    response_key: str = "response"

    # Batch sizes
    train_batch_size: int = 128  # Global batch size
    micro_batch_size_per_gpu: int = 8  # For gradient accumulation

    # Sequence length
    max_seq_length: int = 2048

    # Truncation strategy: 'error', 'left', or 'right'
    truncation: str = "right"


@dataclass
class ValidationConfig:
    """Validation configuration."""
    # Enable validation during training
    enabled: bool = True

    # Validation dataset (JSONL format for generation evaluation)
    math500_test_file: str = "/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"

    # Validation frequency (in training steps)
    eval_steps: int = 30

    # Generation settings for validation
    max_prompt_length: int = 2048
    max_new_tokens: int = 4096
    temperature: float = 0.0  # Greedy decoding
    do_sample: bool = False
    use_few_shot: bool = False  # Use 5-shot CoT prompting

    # Number of samples to evaluate (None = all 500)
    math500_samples: Optional[int] = None

    # vLLM acceleration for validation inference
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.7
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"


@dataclass
class LoggingConfig:
    """Offline logging configuration."""
    # Weights & Biases (offline mode)
    use_wandb: bool = True
    wandb_project: str = "sft-ouro-math"
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = True

    # CSV logging
    use_csv: bool = True
    csv_filename: str = "training_log.csv"

    # Console logging
    use_console: bool = True

    # Log validation metrics
    log_validation_to_wandb: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # LoRA target modules: "all-linear" applies to all linear layers,
    # "attention" applies only to attention weights (q_proj, k_proj, v_proj, o_proj)
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters based on Ouro paper Section 4 (SFT settings)."""
    output_dir: str = "./sft_ouro_math_output"

    # Optimizer settings (Ouro paper: AdamW with β=(0.9, 0.95), LR=2e-5 for SFT)
    learning_rate: float = 2e-5
    weight_decay: float = 0.1  # Ouro paper default
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Ouro paper: β2=0.95 (not 0.99)
    max_grad_norm: float = 1.0

    # Schedule
    warmup_ratio: float = 0.1

    # Training duration
    num_train_epochs: int = 5
    max_steps: int = -1  # -1 means use num_train_epochs

    # Random seed
    seed: int = 42


@dataclass
class SFTConfig:
    """Combined configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
