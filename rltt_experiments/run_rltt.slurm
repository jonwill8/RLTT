#!/bin/bash
#SBATCH --job-name=rltt_train
#SBATCH --output=logs/rltt_train_%j.out
#SBATCH --error=logs/rltt_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# Adjust partition/account as needed
#SBATCH --partition=ailab

# ============================================================================
# RLTT HYPERPARAMETERS - TUNE THESE VIA SLURM SUBMISSION
# ============================================================================
# Usage examples:
#   sbatch run_rltt.slurm                                           # Default settings (LoRA fine-tuning)
#   sbatch --export=ALL,USE_LORA=false run_rltt.slurm               # Full parameter fine-tuning
#   sbatch --export=ALL,LOOP_WEIGHTING=progressive run_rltt.slurm   # Progressive weighting
#   sbatch --export=ALL,LOOP_WEIGHTING=exit_pdf,USE_RLTT_MODEL=true run_rltt.slurm   # Exit PDF weighting
#   sbatch --export=ALL,BETA=0.001,PROGRESSIVE_ALPHA=2.0 run_rltt.slurm
#
# Training Modes:
#   USE_LORA=true  -> LoRA fine-tuning (default, ~2-3 GB per GPU)
#   USE_LORA=false -> Full parameter fine-tuning (~40-60 GB per GPU for 2.6B model)
#
# Full Fine-tuning Workflow:
#   1. Loads SFT checkpoint (LoRA adapter)
#   2. Merges LoRA into base model
#   3. Trains ALL parameters with multi-GPU FSDP
#
# Multi-GPU Training:
#   This script uses verl with a custom RLTTActorRolloutRefWorker for multi-GPU training.
#   The full RLTT objective (multi-loop log-probability aggregation) is preserved
#   across all GPUs via FSDP sharding and the custom actor implementation.
#   Single-GPU training (N_GPUS=1) uses a simplified trainer instead.

# === RLTT-specific hyperparameters ===
# Loop weighting strategy:
#   "uniform"     - Option A: equal weight for all loops (ω_t = 1/T_max)
#   "progressive" - Option B: later loops get more weight (ω_t = t^α / Σs^α)
#   "exit_pdf"    - Option C: use model's native exit probability distribution
#   "learned"     - Option D: jointly optimize weights during training
LOOP_WEIGHTING=${LOOP_WEIGHTING:-"exit_pdf"}
# Alpha exponent for progressive weighting (only used if LOOP_WEIGHTING=progressive)
# α = 0: uniform weights
# α = 1: linear weights (1, 2, 3, 4)
# α = 2: quadratic weights (1, 4, 9, 16)
PROGRESSIVE_ALPHA=${PROGRESSIVE_ALPHA:-1.0}
# Learned weights configuration (only used if LOOP_WEIGHTING=learned)
LEARNED_WEIGHTS_INIT=${LEARNED_WEIGHTS_INIT:-uniform}  # "uniform" or "progressive"
LEARNED_WEIGHTS_TEMP=${LEARNED_WEIGHTS_TEMP:-1.0}      # Softmax temperature (higher = more uniform)
LEARNED_WEIGHTS_LR=${LEARNED_WEIGHTS_LR:-}             # Learning rate (default: 10x model LR)
# Use RLTT-modified Ouro model that exposes per-loop hidden states and exit PDF
# Required for exit_pdf loop weighting, recommended for optimal RLTT performance
USE_RLTT_MODEL=${USE_RLTT_MODEL:-true}
RLTT_MODEL_PATH=${RLTT_MODEL_PATH:-"/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking-RLTT"}
# Chunk size for memory-efficient log-prob computation (reduces OOM risk)
# Lower values = less memory but slightly slower. Set to 0 to disable chunking.
RLTT_LOGPROB_CHUNK_SIZE=${RLTT_LOGPROB_CHUNK_SIZE:-2048}
# Offload reference model to CPU to save GPU memory (~5-7GB per GPU)
# Set to true if facing OOM. Slows KL computation
REF_OFFLOAD=${REF_OFFLOAD:-true}
# Disable verl's gradient checkpointing (controls verl's layer-level checkpointing)
# With exit_pdf mode, the model's loop-level checkpointing is auto-disabled via
# DISABLE_LOOP_CHECKPOINTING env var, allowing verl's checkpointing to work safely.
# Set to true only if you want to disable ALL checkpointing (not recommended - OOM likely)
NO_GRADIENT_CHECKPOINTING=${NO_GRADIENT_CHECKPOINTING:-false}

# === KL penalty and RL hyperparameters ===
BETA=${BETA:-0.001}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
NUM_PROMPTS_PER_BATCH=${NUM_PROMPTS_PER_BATCH:-32}
TEMPERATURE=${TEMPERATURE:-1}

# === Optimizer hyperparameters ===
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}  # cosine or constant (both include warmup)

# === Batch sizes ===
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
# Max tokens per micro-batch during actor update (with dynamic batching)
# Controls memory usage during backprop. Lower = less memory but slower.
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-8192} #change back to 16384 for 8 GPU
# Max tokens per micro-batch during log-prob computation (rollout/ref)
# Controls memory usage during forward pass for log-prob calculation.
# Lower = less memory but slower.
LOG_PROB_MAX_TOKEN_LEN=${LOG_PROB_MAX_TOKEN_LEN:-8192} #change back to 16384 for 8 GPU
# Number of dataloader workers (reduce to save CPU RAM if OOM)
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

# === Sequence lengths ===
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}

# === Training duration ===
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:--1}

# === Training Mode: LoRA vs Full Fine-tuning ===
# USE_LORA=true  -> LoRA fine-tuning (default, memory efficient)
# USE_LORA=false -> Full parameter fine-tuning (requires more GPU memory)
#
# Full Fine-tuning Workflow:
#   1. Load SFT checkpoint (which contains LoRA adapter)
#   2. Merge LoRA weights into base model
#   3. Train ALL model parameters with RLTT
#
# Example usage:
#   sbatch --export=ALL,USE_LORA=false run_rltt.slurm  # Full fine-tuning
#   sbatch --export=ALL,USE_LORA=true run_rltt.slurm   # LoRA fine-tuning (default)
#
USE_LORA=${USE_LORA:-false}

# LoRA configuration (only used when USE_LORA=true)
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
# LORA_TARGET_MODULES: "all-linear" applies to all linear layers,
#                      "attention" applies only to q_proj, k_proj, v_proj, o_proj
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-attention}

# === Ouro model configuration ===
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}

# === Validation configuration ===
EVAL_STEPS=${EVAL_STEPS:-5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_MAX_NEW_TOKENS=${VAL_MAX_NEW_TOKENS:-3072}
NO_VALIDATION=${NO_VALIDATION:-true}
SKIP_INITIAL_VALIDATION=${SKIP_INITIAL_VALIDATION:-true}
MATH500_TEST_FILE=${MATH500_TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"}

# === SFT checkpoint initialization ===
# Set USE_SFT_CHECKPOINT=false to start from fresh MODEL_PATH weights (no SFT initialization)
# Example: sbatch --export=ALL,USE_SFT_CHECKPOINT=false run_rltt.slurm
USE_SFT_CHECKPOINT=${USE_SFT_CHECKPOINT:-false}
SFT_CHECKPOINT_PATH=${SFT_CHECKPOINT_PATH:-""}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-""}

# === Logging ===
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-10}
WANDB_PROJECT=${WANDB_PROJECT:-rltt-ouro-math}

# === vLLM configuration ===
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.45}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-1}
USE_VLLM=${USE_VLLM:-true}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-false}

# === Multi-GPU settings ===
if [ -n "$N_GPUS" ]; then
    N_GPUS=$N_GPUS
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    N_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    N_GPUS=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    N_GPUS=2
fi
NNODES=${NNODES:-${SLURM_NNODES:-1}}

# === Random seed ===
SEED=${SEED:-42}

# ============================================================================
# PATHS - Adjust if needed
# ============================================================================
MODEL_PATH=${MODEL_PATH:-"/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking"}
TRAIN_FILE=${TRAIN_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./rltt_output/${SLURM_JOB_ID}"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Training - Reward Latent Thought Trajectories"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "=========================================="
if [ "$USE_LORA" = "true" ]; then
    echo "*** TRAINING MODE: LoRA FINE-TUNING ***"
else
    echo "*** TRAINING MODE: FULL PARAMETER FINE-TUNING ***"
fi
echo "=========================================="
echo ""

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set wandb to offline mode
export WANDB_MODE=offline

# Set cache directories
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Ray settings for verl
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR

# Disable model's loop-level checkpointing when using exit_pdf with FSDP
# This prevents the FSDP state machine conflict with torch.utils.checkpoint
# verl's own checkpointing will handle memory optimization instead
if [ "$LOOP_WEIGHTING" = "exit_pdf" ]; then
    export DISABLE_LOOP_CHECKPOINTING=1
    echo "DISABLE_LOOP_CHECKPOINTING=1 (exit_pdf mode with FSDP)"
else
    export DISABLE_LOOP_CHECKPOINTING=0
fi

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments}"
RLTT_ROOT="$(dirname "$WORK_DIR")"
cd $WORK_DIR

# Set default SFT paths relative to RLTT root if not specified
if [ -z "$SFT_OUTPUT_DIR" ]; then
    SFT_OUTPUT_DIR="$RLTT_ROOT/sft_experiments/sft_ouro_math_output"
fi

# Create logs directory
mkdir -p logs

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Load CUDA toolkit matching PyTorch's CUDA version (12.8)
module load cudatoolkit/12.8

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Check for vLLM
if python -c "import vllm" 2>/dev/null; then
    echo "vLLM: available"
    VLLM_AVAILABLE=true
else
    echo "vLLM: not available (will use HF generation)"
    VLLM_AVAILABLE=false
fi

# Check for verl
if python -c "import verl" 2>/dev/null; then
    echo "verl: available"
    VERL_AVAILABLE=true
else
    echo "verl: not available (will use simple trainer)"
    VERL_AVAILABLE=false
fi
echo ""

# Print configuration
echo "=========================================="
echo "RLTT Configuration"
echo "=========================================="
echo ""
echo "RLTT-Specific Settings:"
echo "  Loop weighting strategy: $LOOP_WEIGHTING"
echo "  Progressive alpha: $PROGRESSIVE_ALPHA"
echo "  Total UT steps: $TOTAL_UT_STEPS"
echo "  Log-prob chunk size: $RLTT_LOGPROB_CHUNK_SIZE"
echo "  Reference model CPU offload: $REF_OFFLOAD"
echo "  Verl gradient checkpointing: $([ "$NO_GRADIENT_CHECKPOINTING" = "true" ] && echo "disabled" || echo "enabled")"
echo ""
echo "Model Settings:"
echo "  Model path: $MODEL_PATH"
echo ""
echo "Data Settings:"
echo "  Train file: $TRAIN_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "RL Hyperparameters:"
echo "  Beta (KL coef): $BETA"
echo "  Num generations per prompt: $NUM_GENERATIONS"
echo "  Num prompts per batch: $NUM_PROMPTS_PER_BATCH"
echo "  Temperature: $TEMPERATURE"
echo ""
COMPUTED_BATCH_SIZE=$((NUM_PROMPTS_PER_BATCH * NUM_GENERATIONS))
echo "Training Hyperparameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  LR scheduler: $LR_SCHEDULER_TYPE"
echo "  Warmup ratio: $WARMUP_RATIO"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Max grad norm: $MAX_GRAD_NORM"
echo "  Epochs: $NUM_EPOCHS"
echo "  Total batch size: $COMPUTED_BATCH_SIZE ($NUM_PROMPTS_PER_BATCH prompts x $NUM_GENERATIONS gens)"
echo "  Dynamic batch sizing: enabled (ppo_max_token_len=$PPO_MAX_TOKEN_LEN)"
echo "  Log-prob max token len: $LOG_PROB_MAX_TOKEN_LEN"
echo ""
echo "Sequence Lengths:"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Max completion length: $MAX_COMPLETION_LENGTH"
echo ""
echo "=========================================="
echo "Training Mode Configuration:"
echo "=========================================="
if [ "$USE_LORA" = "true" ]; then
    echo "  *** TRAINING MODE: LoRA FINE-TUNING ***"
    echo "  Only LoRA adapter parameters will be updated"
    echo "  LoRA r: $LORA_R"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA target modules: $LORA_TARGET_MODULES"
else
    echo "  *** TRAINING MODE: FULL PARAMETER FINE-TUNING ***"
    echo "  All model parameters will be updated"
    echo "  Note: Requires more GPU memory than LoRA"
    if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
        echo "  Workflow: SFT LoRA -> Merge into base -> Train all params"
    fi
fi
echo ""
echo "Multi-GPU Configuration:"
echo "  Number of GPUs: $N_GPUS"
echo "  Number of nodes: $NNODES"
echo "  vLLM TP size: $VLLM_TP_SIZE"
echo ""
echo "vLLM Configuration:"
echo "  Use vLLM: $USE_VLLM"
echo "  GPU memory utilization: $VLLM_GPU_MEM"
echo ""
echo "SFT Checkpoint:"
echo "  Use SFT checkpoint: $USE_SFT_CHECKPOINT"
if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
    echo "  SFT checkpoint path: $SFT_CHECKPOINT_PATH"
else
    echo "  Starting from fresh model weights (no SFT initialization)"
fi
echo ""
echo "=========================================="
echo ""

# Explain RLTT loop weighting
echo "RLTT Loop Weighting Explanation:"
echo "================================"
if [ "$LOOP_WEIGHTING" = "uniform" ]; then
    echo "  Using UNIFORM weighting (Option A from paper)"
    echo "  ω_t = 1/T_max for all loops"
    echo "  All loops receive equal credit"
    echo "  Encourages correct distribution early in the trajectory"
elif [ "$LOOP_WEIGHTING" = "progressive" ]; then
    echo "  Using PROGRESSIVE weighting (Option B from paper)"
    echo "  ω_t = t^α / Σ s^α where α = $PROGRESSIVE_ALPHA"
    echo "  Later loops receive more weight"
    echo "  Reflects that later refinements are closer to final decision"
elif [ "$LOOP_WEIGHTING" = "exit_pdf" ]; then
    echo "  Using EXIT_PDF weighting (Option C - model-native)"
    echo "  ω_t = P(exit at loop t) from model's exit gate"
    echo "  Dynamic per-token weighting based on model's exit probability distribution"
    echo "  Requires RLTT-modified Ouro model (USE_RLTT_MODEL=true)"
elif [ "$LOOP_WEIGHTING" = "learned" ]; then
    echo "  Using LEARNED weighting (Option D - jointly optimized)"
    echo "  ω_t = softmax(logits_t / temp) where logits are learnable parameters"
    echo "  Weights are optimized jointly with the policy during training"
    echo "  Initialization: $LEARNED_WEIGHTS_INIT, Temperature: $LEARNED_WEIGHTS_TEMP"
    if [ -n "$LEARNED_WEIGHTS_LR" ]; then
        echo "  Learning rate: $LEARNED_WEIGHTS_LR"
    else
        echo "  Learning rate: 10x model LR (default)"
    fi
fi
echo ""
if [ "$USE_RLTT_MODEL" = "true" ]; then
    echo "RLTT Model Configuration:"
    echo "========================"
    echo "  Using RLTT-modified Ouro model: $RLTT_MODEL_PATH"
    echo "  This model exposes per-loop hidden states and exit PDF"
    echo ""
fi

# Calculate steps based on dataset size
DATASET_SIZE=7500
STEPS_PER_EPOCH=$((DATASET_SIZE / NUM_PROMPTS_PER_BATCH))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
echo "Training Progress Guide:"
echo "========================"
echo "  Dataset size: $DATASET_SIZE prompts"
echo "  Prompts per step: $NUM_PROMPTS_PER_BATCH"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total steps: $TOTAL_STEPS (over $NUM_EPOCHS epoch(s))"
echo ""
echo "Starting RLTT training..."
echo ""

# Build command line arguments
LORA_ARGS="--lora_target_modules $LORA_TARGET_MODULES"
if [ "$USE_LORA" = "false" ]; then
    LORA_ARGS="$LORA_ARGS --no_lora"
fi

# Build SFT checkpoint arguments
SFT_ARGS=""
if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
    if [ -n "$SFT_CHECKPOINT_PATH" ] && [ -d "$SFT_CHECKPOINT_PATH" ]; then
        SFT_ARGS="--sft_checkpoint $SFT_CHECKPOINT_PATH"
    else
        SFT_ARGS="--use_latest_sft --sft_output_dir $SFT_OUTPUT_DIR"
    fi
else
    # Explicitly skip SFT checkpoint - start from fresh MODEL_PATH weights
    SFT_ARGS="--no_sft_checkpoint"
fi

# Build validation arguments
VAL_ARGS="--eval_steps $EVAL_STEPS --val_batch_size $VAL_BATCH_SIZE --val_max_new_tokens $VAL_MAX_NEW_TOKENS --math500_test_file $MATH500_TEST_FILE"
if [ "$NO_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --no_validation"
fi
if [ "$SKIP_INITIAL_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --skip_initial_validation"
fi

# Build vLLM arguments
VLLM_ARGS=""
if [ "$USE_VLLM" = "false" ]; then
    VLLM_ARGS="--no_vllm"
else
    VLLM_ARGS="--vllm_gpu_memory_utilization $VLLM_GPU_MEM --vllm_tensor_parallel_size $VLLM_TP_SIZE"
    if [ "$VLLM_ENFORCE_EAGER" = "true" ]; then
        VLLM_ARGS="$VLLM_ARGS --vllm_enforce_eager"
    fi
fi

# Build RLTT model arguments
RLTT_MODEL_ARGS="--rltt_logprob_chunk_size $RLTT_LOGPROB_CHUNK_SIZE"
if [ "$USE_RLTT_MODEL" = "true" ]; then
    RLTT_MODEL_ARGS="$RLTT_MODEL_ARGS --use_rltt_model --rltt_model_path $RLTT_MODEL_PATH"
fi
if [ "$NO_GRADIENT_CHECKPOINTING" = "true" ]; then
    RLTT_MODEL_ARGS="$RLTT_MODEL_ARGS --no_gradient_checkpointing"
fi
if [ "$REF_OFFLOAD" = "true" ]; then
    RLTT_MODEL_ARGS="$RLTT_MODEL_ARGS --ref_offload"
fi

# Build learned weights arguments
LEARNED_WEIGHTS_ARGS=""
if [ "$LOOP_WEIGHTING" = "learned" ]; then
    LEARNED_WEIGHTS_ARGS="--learned_weights_init $LEARNED_WEIGHTS_INIT --learned_weights_temp $LEARNED_WEIGHTS_TEMP"
    if [ -n "$LEARNED_WEIGHTS_LR" ]; then
        LEARNED_WEIGHTS_ARGS="$LEARNED_WEIGHTS_ARGS --learned_weights_lr $LEARNED_WEIGHTS_LR"
    fi
fi

# Run training
python rltt_train.py \
    --model_path "$MODEL_PATH" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
    --loop_weighting "$LOOP_WEIGHTING" \
    --progressive_alpha "$PROGRESSIVE_ALPHA" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" \
    --num_generations "$NUM_GENERATIONS" \
    --num_prompts_per_batch "$NUM_PROMPTS_PER_BATCH" \
    --temperature "$TEMPERATURE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --ppo_max_token_len "$PPO_MAX_TOKEN_LEN" \
    --log_prob_max_token_len "$LOG_PROB_MAX_TOKEN_LEN" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --wandb_project "$WANDB_PROJECT" \
    --seed "$SEED" \
    --n_gpus "$N_GPUS" \
    --nnodes "$NNODES" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    $LORA_ARGS \
    $SFT_ARGS \
    $VAL_ARGS \
    $VLLM_ARGS \
    $RLTT_MODEL_ARGS \
    $LEARNED_WEIGHTS_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  Model checkpoint: $OUTPUT_DIR/"
echo "  Rollout logs: $OUTPUT_DIR/rollout_logs/"
echo "  wandb logs: $OUTPUT_DIR/wandb/"
echo ""
echo "To sync wandb (when internet available):"
echo "  wandb sync $OUTPUT_DIR/wandb/offline-run-*"
echo ""
echo "=========================================="
