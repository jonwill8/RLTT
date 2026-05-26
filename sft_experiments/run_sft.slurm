#!/bin/bash
#SBATCH --job-name=sft_ouro_verl
#SBATCH --output=logs/sft_ouro_%j.out
#SBATCH --error=logs/sft_ouro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

# Adjust partition/account as needed
#SBATCH --partition=ailab
# #SBATCH --constraint=mig
# #SBATCH --partition=gpu80

# ============================================================================
# SFT HYPERPARAMETERS - TUNE THESE VIA SLURM SUBMISSION
# ============================================================================
# Usage examples:
#   sbatch run_sft.slurm                                    # Default settings (LoRA fine-tuning)
#   sbatch --export=ALL,USE_LORA=false run_sft.slurm        # Full parameter fine-tuning
#   sbatch --export=ALL,LEARNING_RATE=1e-5 run_sft.slurm    # Custom LR
#   sbatch --export=ALL,NUM_EPOCHS=3,LORA_R=64 run_sft.slurm
#
# Training Modes:
#   USE_LORA=true  -> LoRA fine-tuning (default, memory efficient)
#   USE_LORA=false -> Full parameter fine-tuning (requires more GPU memory)

# === Optimizer hyperparameters ===
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}

# === Batch sizes ===
# For 32*140 = 4480 total samples experiment: batch_size=32, steps=140
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}

# === Sequence lengths ===
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-1024}
MAX_GENERATION_TOKENS=${MAX_GENERATION_TOKENS:-2048}
# Compute max_seq_length as sum of prompt + generation tokens
MAX_SEQ_LENGTH=$((MAX_PROMPT_TOKENS + MAX_GENERATION_TOKENS))

# === Training duration ===
NUM_EPOCHS=${NUM_EPOCHS:-10}
# Set to 140 for 32*140 = 4480 total samples experiment
MAX_STEPS=${MAX_STEPS:-140}

# === Training Mode: LoRA vs Full Fine-tuning ===
# USE_LORA=true  -> LoRA fine-tuning (default, memory efficient ~2-3 GB per GPU)
# USE_LORA=false -> Full parameter fine-tuning (requires more GPU memory)
#
# Example usage:
#   sbatch --export=ALL,USE_LORA=false run_sft.slurm  # Full fine-tuning
#   sbatch --export=ALL,USE_LORA=true run_sft.slurm   # LoRA fine-tuning (default)
#
USE_LORA=${USE_LORA:-false}

# LoRA configuration (only used when USE_LORA=true)
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
# LORA_TARGET_MODULES: "all-linear" (all linear layers) or "attention" (q_proj, k_proj, v_proj, o_proj only)
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"attention"}

# === Validation configuration ===
EVAL_STEPS=${EVAL_STEPS:-5}
MATH500_SAMPLES=${MATH500_SAMPLES:-500}
VAL_USE_FEW_SHOT=${VAL_USE_FEW_SHOT:-false}
NO_VALIDATION=${NO_VALIDATION:-true}
NO_INITIAL_VALIDATION=${NO_INITIAL_VALIDATION:-true}

# === vLLM configuration ===
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.4}
USE_VLLM=${USE_VLLM:-true}

# === Logging ===
WANDB_PROJECT=${WANDB_PROJECT:-sft-ouro-math}

# === Random seed ===
SEED=${SEED:-42}

# ============================================================================
# PATHS - Adjust if needed
# ============================================================================
MODEL_PATH=${MODEL_PATH:-"/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking"}
# Input files (JSONL format - will be auto-converted to parquet)
TRAIN_JSONL=${TRAIN_JSONL:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"}
VAL_JSONL=${VAL_JSONL:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./sft_ouro_math_output/${SLURM_JOB_ID}"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "SFT Training: Ouro-2.6B-Thinking on MATH Dataset (verl)"
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

# Set offline mode for HuggingFace (no internet access)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set wandb to offline mode (will sync later when internet available)
export WANDB_MODE=offline

# Set cache directories (use local paths)
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# Reduce CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Working directory (use SLURM_SUBMIT_DIR since BASH_SOURCE doesn't work in SLURM)
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments}"
cd $WORK_DIR

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

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
else
    echo "vLLM: not available (will use HF generation for validation)"
fi

# Check for verl
if python -c "import verl" 2>/dev/null; then
    echo "verl: available"
else
    echo "verl: not available"
    exit 1
fi
echo ""

# ============================================================================
# DATA PREPARATION - Convert JSONL to Parquet
# ============================================================================
echo "=========================================="
echo "Preparing Data (JSONL -> Parquet)"
echo "=========================================="

# Set output paths for parquet files
TRAIN_PARQUET="$OUTPUT_DIR/train.parquet"
VAL_PARQUET="$OUTPUT_DIR/val.parquet"

echo "Converting training data..."
echo "  Input:  $TRAIN_JSONL"
echo "  Output: $TRAIN_PARQUET"

python -c "
import sys
sys.path.insert(0, '.')
from prepare_data import convert_math_train_to_parquet
convert_math_train_to_parquet('$TRAIN_JSONL', '$TRAIN_PARQUET', verbose=True)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert training data"
    exit 1
fi

echo ""
echo "Converting validation data..."
echo "  Input:  $VAL_JSONL"
echo "  Output: $VAL_PARQUET"

python -c "
import sys
sys.path.insert(0, '.')
from prepare_data import convert_math500_to_parquet
convert_math500_to_parquet('$VAL_JSONL', '$VAL_PARQUET', verbose=True)
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert validation data"
    exit 1
fi

echo ""
echo "Data preparation complete!"
echo ""

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================
echo "=========================================="
echo "SFT Configuration"
echo "=========================================="
echo ""
echo "Model Settings:"
echo "  Model path: $MODEL_PATH"
echo ""
echo "Data Settings:"
echo "  Train file (JSONL): $TRAIN_JSONL"
echo "  Train file (parquet): $TRAIN_PARQUET"
echo "  Val file (JSONL): $VAL_JSONL"
echo "  Val file (parquet): $VAL_PARQUET"
echo "  Output dir: $OUTPUT_DIR"
echo ""
echo "Training Hyperparameters:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Max grad norm: $MAX_GRAD_NORM"
echo "  Warmup ratio: $WARMUP_RATIO"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max steps: $MAX_STEPS (-1 = use epochs)"
echo "  Micro batch size: $MICRO_BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch size: $((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Max prompt tokens: $MAX_PROMPT_TOKENS"
echo "  Max generation tokens: $MAX_GENERATION_TOKENS"
echo "  Max sequence length: $MAX_SEQ_LENGTH (computed: prompt + generation)"
echo "  Random seed: $SEED"
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
fi
echo ""
echo "Validation Configuration:"
echo "  Validation enabled: $([ "$NO_VALIDATION" = "false" ] && echo 'yes' || echo 'no')"
echo "  Initial validation: $([ "$NO_INITIAL_VALIDATION" = "false" ] && echo 'yes' || echo 'no')"
echo "  Eval steps: $EVAL_STEPS"
echo "  MATH-500 samples: $MATH500_SAMPLES"
echo "  Use few-shot: $VAL_USE_FEW_SHOT"
echo ""
echo "vLLM Configuration:"
echo "  Use vLLM: $USE_VLLM"
echo "  GPU memory utilization: $VLLM_GPU_MEM"
echo ""
echo "Logging Configuration:"
echo "  wandb project: $WANDB_PROJECT"
echo ""

# Calculate training steps
DATASET_SIZE=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))")
EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
STEPS_PER_EPOCH=$((DATASET_SIZE / EFFECTIVE_BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))

echo "=========================================="
echo "Training Progress Guide:"
echo "=========================================="
echo "  Dataset size: $DATASET_SIZE examples"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Steps per epoch: $STEPS_PER_EPOCH"
echo "  Total steps: $TOTAL_STEPS (over $NUM_EPOCHS epoch(s))"
echo ""
echo "=========================================="
echo ""
echo "Starting SFT training..."
echo ""

# ============================================================================
# BUILD COMMAND LINE ARGUMENTS
# ============================================================================
LORA_ARGS="--lora_target_modules $LORA_TARGET_MODULES"
if [ "$USE_LORA" = "false" ]; then
    LORA_ARGS="$LORA_ARGS --no_lora"
fi

VAL_ARGS="--eval_steps $EVAL_STEPS --math500_samples $MATH500_SAMPLES --max_new_tokens $MAX_GENERATION_TOKENS --math500_test_file $VAL_JSONL"
if [ "$VAL_USE_FEW_SHOT" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --use_few_shot"
fi
if [ "$NO_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --no_validation"
fi
if [ "$NO_INITIAL_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --no_initial_validation"
fi

VLLM_ARGS=""
if [ "$USE_VLLM" = "false" ]; then
    VLLM_ARGS="--no_vllm"
else
    VLLM_ARGS="--vllm_gpu_memory_utilization $VLLM_GPU_MEM"
fi

# ============================================================================
# RUN TRAINING
# ============================================================================
# Single-process training with vLLM validation

python sft_train.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_PARQUET" \
    --val_file "$VAL_PARQUET" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_prompt_length $MAX_PROMPT_TOKENS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --wandb_project "$WANDB_PROJECT" \
    --seed $SEED \
    $LORA_ARGS \
    $VAL_ARGS \
    $VLLM_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  Model checkpoint: $OUTPUT_DIR/"
echo "  wandb logs:       $OUTPUT_DIR/wandb/"
echo "  Training CSV log: $OUTPUT_DIR/training_log.csv"
echo "  Validation JSON:  $OUTPUT_DIR/validation_results.json"
echo ""
echo "To sync wandb (when internet available):"
echo "  wandb sync $OUTPUT_DIR/wandb/offline-run-*"
echo "=========================================="
