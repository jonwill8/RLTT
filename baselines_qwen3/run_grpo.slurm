#!/bin/bash
#SBATCH --job-name=qwen3_grpo_train
#SBATCH --output=logs/qwen3_grpo_train_%j.out
#SBATCH --error=logs/qwen3_grpo_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=224G
#SBATCH --gres=gpu:4
#SBATCH --time=5:30:00

# Adjust partition/account as needed
#SBATCH --account=karthikn
#SBATCH --partition=ailab
# #SBATCH --constraint=gpu80

# ============================================================================
# QWEN3 GRPO TRAINING (GRPO-ONLY)
# ============================================================================
# Usage examples:
#   sbatch run_grpo.slurm                                      # Default settings (full-parameter fine-tuning)
#   sbatch --export=ALL,MODEL_SIZE=1.7B run_grpo.slurm        # Use Qwen3-1.7B
#   sbatch --export=ALL,USE_LORA=true run_grpo.slurm          # LoRA fine-tuning
#   sbatch --export=ALL,BETA=0.001 run_grpo.slurm             # Custom beta
#   sbatch --export=ALL,N_GPUS=8 run_grpo.slurm               # Use 8 GPUs
#   sbatch --export=ALL,LR=5e-7,BETA=0.002,NUM_GENERATIONS=4 run_grpo.slurm
#   sbatch --export=ALL,LOG_PROB_MAX_TOKEN_LEN=8192 run_grpo.slurm  # OOM fix for rollout log-prob
#
# Training Modes:
#   USE_LORA=false -> Full parameter fine-tuning (default)
#   USE_LORA=true  -> LoRA fine-tuning

# === Model selection ===
# MODEL_SIZE can be 1.7B or 4B. Ignored if MODEL_PATH is explicitly provided.
MODEL_SIZE=${MODEL_SIZE:-"1.7B"}
MODEL_PATH=${MODEL_PATH:-""}
if [ -z "$MODEL_PATH" ]; then
    if [ "$MODEL_SIZE" = "1.7B" ] || [ "$MODEL_SIZE" = "1.7b" ]; then
        MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Qwen3-1.7B"
    elif [ "$MODEL_SIZE" = "4B" ] || [ "$MODEL_SIZE" = "4b" ]; then
        MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Qwen3-4B"
    else
        echo "ERROR: MODEL_SIZE must be 1.7B or 4B (or set MODEL_PATH directly). Got: $MODEL_SIZE"
        exit 1
    fi
fi

# === GRPO-specific hyperparameters ===
BETA=${BETA:-0.001}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
NUM_PROMPTS_PER_BATCH=${NUM_PROMPTS_PER_BATCH:-16}
TEMPERATURE=${TEMPERATURE:-1}

# === Optimizer hyperparameters ===
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}

# === Batch sizes ===
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

# === Dynamic batch size token limits ===
LOG_PROB_MAX_TOKEN_LEN=${LOG_PROB_MAX_TOKEN_LEN:-16384}

# === Sequence lengths ===
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}

# === Training duration ===
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_STEPS=${MAX_STEPS:--1}

# === Training Mode: LoRA vs Full Fine-tuning ===
USE_LORA=${USE_LORA:-true}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"attention"}

# === Optional recurrent model configuration (used by models that support it) ===
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}

# === Validation configuration ===
EVAL_STEPS=${EVAL_STEPS:-5}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_MAX_NEW_TOKENS=${VAL_MAX_NEW_TOKENS:-3072}
NO_VALIDATION=${NO_VALIDATION:-true}
SKIP_INITIAL_VALIDATION=${SKIP_INITIAL_VALIDATION:-true}
MATH500_TEST_FILE=${MATH500_TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"}

# === SFT checkpoint initialization ===
USE_SFT_CHECKPOINT=${USE_SFT_CHECKPOINT:-false}
SFT_CHECKPOINT_PATH=${SFT_CHECKPOINT_PATH:-""}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-""}

# === Logging ===
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-10}
WANDB_PROJECT=${WANDB_PROJECT:-grpo-qwen3-math-verl}

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
# PATHS
# ============================================================================
TRAIN_FILE=${TRAIN_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./grpo_output/qwen3_${MODEL_SIZE}_${SLURM_JOB_ID}"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "Qwen3 GRPO Training with verl + vLLM Acceleration"
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

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR

WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3}"
RLTT_ROOT="$(dirname "$WORK_DIR")"
cd $WORK_DIR

if [ -z "$SFT_OUTPUT_DIR" ]; then
    SFT_OUTPUT_DIR="$RLTT_ROOT/sft_experiments/sft_qwen3_math_output"
fi

mkdir -p logs

source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm
module load cudatoolkit/12.8

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

if python -c "import vllm" 2>/dev/null; then
    echo "vLLM: available"
else
    echo "vLLM: not available (will use HF generation)"
fi

if python -c "import verl" 2>/dev/null; then
    echo "verl: available"
else
    echo "verl: not available (will use simple trainer)"
fi
echo ""

echo "=========================================="
echo "Qwen3 GRPO Configuration"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Model size: $MODEL_SIZE"
echo "Train file: $TRAIN_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Beta: $BETA"
echo "Num generations: $NUM_GENERATIONS"
echo "Num prompts per batch: $NUM_PROMPTS_PER_BATCH"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Use LoRA: $USE_LORA"
echo "n_gpus: $N_GPUS"
echo "nnodes: $NNODES"
echo "=========================================="
echo ""

LORA_ARGS="--lora_target_modules $LORA_TARGET_MODULES"
if [ "$USE_LORA" = "false" ]; then
    LORA_ARGS="$LORA_ARGS --no_lora"
fi

SFT_ARGS=""
if [ "$USE_SFT_CHECKPOINT" = "true" ]; then
    if [ -n "$SFT_CHECKPOINT_PATH" ] && [ -d "$SFT_CHECKPOINT_PATH" ]; then
        SFT_ARGS="--sft_checkpoint $SFT_CHECKPOINT_PATH"
    else
        SFT_ARGS="--use_latest_sft --sft_output_dir $SFT_OUTPUT_DIR"
    fi
else
    SFT_ARGS="--no_sft_checkpoint"
fi

VAL_ARGS="--eval_steps $EVAL_STEPS --val_batch_size $VAL_BATCH_SIZE --val_max_new_tokens $VAL_MAX_NEW_TOKENS --math500_test_file $MATH500_TEST_FILE"
if [ "$NO_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --no_validation"
fi
if [ "$SKIP_INITIAL_VALIDATION" = "true" ]; then
    VAL_ARGS="$VAL_ARGS --skip_initial_validation"
fi

VLLM_ARGS=""
if [ "$USE_VLLM" = "false" ]; then
    VLLM_ARGS="--no_vllm"
else
    VLLM_ARGS="--vllm_gpu_memory_utilization $VLLM_GPU_MEM --vllm_tensor_parallel_size $VLLM_TP_SIZE"
    if [ "$VLLM_ENFORCE_EAGER" = "true" ]; then
        VLLM_ARGS="$VLLM_ARGS --vllm_enforce_eager"
    fi
fi

python grpo_train.py \
    --model_path "$MODEL_PATH" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
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
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --log_prob_max_token_len "$LOG_PROB_MAX_TOKEN_LEN" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
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
    $VLLM_ARGS

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "Output files:"
echo "  Model checkpoint: $OUTPUT_DIR/"
echo "  Rollout logs: $OUTPUT_DIR/rollout_logs/"
echo "  wandb logs: $OUTPUT_DIR/wandb/"
echo "=========================================="
