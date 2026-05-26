#!/bin/bash
#SBATCH --job-name=qwen3_rl_eval_beyondaime
#SBATCH --output=logs/qwen3_rl_eval_beyondaime_%j.out
#SBATCH --error=logs/qwen3_rl_eval_beyondaime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:55:00

# Adjust partition as needed
#SBATCH --account=karthikn
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# QWEN3 RL-TRAINED EVALUATION - BeyondAIME
# ============================================================================
# This script evaluates GRPO-trained Qwen3 checkpoints on the BeyondAIME benchmark.
#
# Usage:
#   sbatch run_eval_beyondaime_rl_trained.slurm
#   sbatch --export=ALL,MODEL_SIZE=1.7B run_eval_beyondaime_rl_trained.slurm
#   sbatch --export=ALL,MODEL_SIZE=4B run_eval_beyondaime_rl_trained.slurm
#   sbatch --export=ALL,MODEL_PATH=/path/to/grpo_output/.../global_step_140 run_eval_beyondaime_rl_trained.slurm
#
# Optional parameters:
#   MODEL_SIZE=4B                  - Built-in checkpoint to evaluate: 1.7B or 4B
#   MODEL_PATH=/path/to/checkpoint - GRPO checkpoint directory to evaluate (overrides MODEL_SIZE)
#   USE_FEW_SHOT=true              - Use 5-shot CoT prompting (default: false)
#   MAX_NEW_TOKENS=4096            - Max tokens to generate
#   MAX_PROMPT_LENGTH=1024         - Max prompt length
#   GPU_MEMORY_UTILIZATION=0.9     - vLLM GPU memory fraction

# ============================================================================
# CONFIGURATION
# ============================================================================
QWEN3_1_7B_MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/grpo_output/qwen3_1.7B_8720883/global_step_140"
QWEN3_4B_MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/grpo_output/qwen3_4B_8720874/global_step_140"
MODEL_SIZE=${MODEL_SIZE:-"1.7B"}

if [ -z "${MODEL_PATH:-}" ]; then
    case "$MODEL_SIZE" in
        1.7B|1.7b|1_7B|1_7b)
            MODEL_SIZE="1.7B"
            MODEL_PATH="$QWEN3_1_7B_MODEL_PATH"
            ;;
        4B|4b)
            MODEL_SIZE="4B"
            MODEL_PATH="$QWEN3_4B_MODEL_PATH"
            ;;
        *)
            echo "ERROR: MODEL_SIZE must be 1.7B or 4B, got: $MODEL_SIZE"
            exit 1
            ;;
    esac
else
    MODEL_SIZE="custom"
fi

# Prompt configuration
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# Generation settings - BeyondAIME needs more tokens for complex reasoning
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
TEMPERATURE=0.0

# Data settings - BeyondAIME dataset
TEST_FILE=${TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/BeyondAIME/beyondaime.test.jsonl"}

# Output settings
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
CHECKPOINT_TAG=$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/qwen3_rltrained_beyondaime_${CHECKPOINT_TAG}_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}_${SLURM_JOB_ID}"}
CLEANUP_MERGED_MODEL=${CLEANUP_MERGED_MODEL:-true}

# ============================================================================
# CHECK MODEL PATH EXISTS
# ============================================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

EVAL_MODEL_PATH="$MODEL_PATH"

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "Qwen3 RL-Trained Evaluation - BeyondAIME"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Model selection: $MODEL_SIZE"
echo "Checkpoint path: $MODEL_PATH"
echo "Eval model path: $EVAL_MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Configuration:"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Use few-shot: $USE_FEW_SHOT"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Temperature: $TEMPERATURE"
echo ""

# Set offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3}"
cd $WORK_DIR

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Load CUDA toolkit
module load cudatoolkit/12.8

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# ============================================================================
# PREPARE EVALUATION MODEL
# ============================================================================
if [ -d "$MODEL_PATH/actor" ]; then
    FSDP_CHECKPOINT_DIR="$MODEL_PATH/actor"
    MERGED_MODEL_DIR="${OUTPUT_DIR}/merged_model"
    echo "Detected FSDP checkpoint at: $FSDP_CHECKPOINT_DIR"
    echo "Merging checkpoint for inference..."
    python -m verl.model_merger merge \
        --backend fsdp \
        --trust-remote-code \
        --local_dir "$FSDP_CHECKPOINT_DIR" \
        --target_dir "$MERGED_MODEL_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to merge FSDP checkpoint"
        exit 1
    fi
    EVAL_MODEL_PATH="$MERGED_MODEL_DIR"
    echo "Merged model ready at: $EVAL_MODEL_PATH"
fi

# ============================================================================
# RUN EVALUATION
# ============================================================================
echo "Starting evaluation..."
echo ""

# Build few-shot argument
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

python evaluate_baseline.py \
    --model_path "$EVAL_MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --temperature "$TEMPERATURE" \
    $FEW_SHOT_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

if [ -d "${OUTPUT_DIR}/merged_model" ] && [ "$CLEANUP_MERGED_MODEL" = "true" ]; then
    echo "Cleaning up merged model directory..."
    rm -rf "${OUTPUT_DIR}/merged_model"
fi

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "=========================================="
echo "BeyondAIME RL-Trained Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/eval_results.json"
echo "  cat $OUTPUT_DIR/eval_summary.txt"
echo "=========================================="
