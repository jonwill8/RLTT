#!/bin/bash
#SBATCH --job-name=deepseek_r1_eval_beyondaime
#SBATCH --output=logs/deepseek_r1_eval_beyondaime_%j.out
#SBATCH --error=logs/deepseek_r1_eval_beyondaime_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

# Adjust partition as needed
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# DEEPSEEK-R1-DISTILL-QWEN BASELINE EVALUATION - BeyondAIME
# ============================================================================
# This script evaluates DeepSeek-R1-Distill-Qwen models (1.5B, 7B) on BeyondAIME.
#
# Usage:
#   sbatch --export=ALL,MODEL_SIZE=7B run_eval_beyondaime.slurm
#
# Example:
#   sbatch --export=ALL,MODEL_SIZE=1.5B run_eval_beyondaime.slurm
#   sbatch --export=ALL,MODEL_SIZE=7B run_eval_beyondaime.slurm
#
# Optional parameters:
#   USE_FEW_SHOT=true         - Use 5-shot CoT prompting (default: false)
#   MAX_NEW_TOKENS=4096       - Max tokens to generate
#   MAX_PROMPT_LENGTH=1024    - Max prompt length
#   GPU_MEMORY_UTILIZATION=0.9 - vLLM GPU memory fraction

# ============================================================================
# CONFIGURATION
# ============================================================================
# MODEL_SIZE - which DeepSeek-R1-Distill-Qwen model to evaluate (1.5B or 7B)
MODEL_SIZE=${MODEL_SIZE:-"7B"}

# Map model size to model path
case "$MODEL_SIZE" in
    "1.5B"|"1.5b")
        MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-1.5B"
        ;;
    "7B"|"7b")
        MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-7B"
        ;;
    *)
        echo "ERROR: Unknown MODEL_SIZE: $MODEL_SIZE"
        echo "Valid options: 1.5B, 7B"
        exit 1
        ;;
esac

# Prompt configuration
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# Generation settings - BeyondAIME needs more tokens for complex reasoning
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# Data settings - BeyondAIME dataset
TEST_FILE=${TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/BeyondAIME/beyondaime.test.jsonl"}

# Output settings
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/deepseek_r1_${MODEL_SIZE}_beyondaime_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}_${SLURM_JOB_ID}"}

# ============================================================================
# CHECK MODEL PATH EXISTS
# ============================================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen Baseline Evaluation - BeyondAIME"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Model: DeepSeek-R1-Distill-Qwen-$MODEL_SIZE"
echo "Model path: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Configuration:"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Use few-shot: $USE_FEW_SHOT"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
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
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_deepseek_r1}"
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
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    $FEW_SHOT_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "=========================================="
echo "BeyondAIME Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/eval_results.json"
echo "  cat $OUTPUT_DIR/eval_summary.txt"
echo "=========================================="
