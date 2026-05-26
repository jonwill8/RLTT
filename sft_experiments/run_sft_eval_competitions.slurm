#!/bin/bash
#SBATCH --job-name=sft_eval_comp
#SBATCH --output=logs/sft_eval_comp_%j.out
#SBATCH --error=logs/sft_eval_comp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Adjust partition as needed
# #SBATCH --partition=ailab
#SBATCH --constraint=gpu80

# ============================================================================
# SFT CHECKPOINT EVALUATION ON COMPETITION BENCHMARKS (AIME26, HMMT25)
# ============================================================================
# Usage examples:
#   sbatch --export=ALL,BENCHMARK=aime26 run_sft_eval_competitions.slurm
#   sbatch --export=ALL,BENCHMARK=hmmt25,CHECKPOINT_PATH=/path/to/ckpt run_sft_eval_competitions.slurm
#
# Optional parameters:
#   TOTAL_UT_STEPS=4             - Number of recurrent loops
#   USE_FEW_SHOT=false           - Enable 5-shot CoT prompting
#   MAX_PROMPT_LENGTH=1024       - Prompt length
#   MAX_NEW_TOKENS=3072          - Generation length
#   TEMPERATURE=0.0              - Sampling temperature
#   GPU_MEMORY_UTILIZATION=0.9   - vLLM GPU memory fraction
#   MERGE_WEIGHTS=true           - Merge LoRA weights for vLLM
#   FULL_MODEL=true              - Set true if checkpoint is a full model
#   USE_VLLM=true                - Set false to force HF generation
#   OUTPUT_DIR=...               - Custom output directory
#   MAX_SAMPLES=...              - Limit number of samples
#   BASE_MODEL_PATH=...          - Override base model path if needed

# ============================================================================
# REQUIRED PARAMETERS
# ============================================================================
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments/sft_ouro_math_output/3706435/checkpoint-final"}

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"aime26"} # aime26 or hmmt25
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}
MERGE_WEIGHTS=${MERGE_WEIGHTS:-true}
FULL_MODEL=${FULL_MODEL:-true}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
USE_VLLM=${USE_VLLM:-true}
MAX_SAMPLES=${MAX_SAMPLES:-""}
BATCH_SIZE=${BATCH_SIZE:-4}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-""}

OUTPUT_DIR=${OUTPUT_DIR:-"./eval_outputs/${BENCHMARK}_${SLURM_JOB_ID}"}

# ============================================================================
# CHECK REQUIRED ARGUMENTS
# ============================================================================
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH is required"
    exit 1
fi

if [ "$BENCHMARK" != "aime26" ] && [ "$BENCHMARK" != "hmmt25" ]; then
    echo "ERROR: BENCHMARK must be 'aime26' or 'hmmt25', got: $BENCHMARK"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "SFT Checkpoint Evaluation - ${BENCHMARK}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Benchmark: $BENCHMARK"
echo "Output dir: $OUTPUT_DIR"
echo "Total UT steps: $TOTAL_UT_STEPS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
echo "Few-shot (5-shot CoT): $USE_FEW_SHOT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Merge weights: $MERGE_WEIGHTS"
echo "Full model: $FULL_MODEL"
echo "Use vLLM: $USE_VLLM"
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

# Fix for MIG-enabled nodes: vLLM can't parse MIG UUIDs in CUDA_VISIBLE_DEVICES
if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]]; then
    echo "Detected MIG GPU, setting CUDA_VISIBLE_DEVICES=0"
    export CUDA_VISIBLE_DEVICES=0
fi

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments}"
cd $WORK_DIR

# Create directories
mkdir -p logs
mkdir -p $OUTPUT_DIR

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
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

MERGE_ARG=""
if [ "$MERGE_WEIGHTS" = "true" ]; then
    MERGE_ARG="--merge_weights"
fi

FULL_MODEL_ARG=""
if [ "$FULL_MODEL" = "true" ]; then
    FULL_MODEL_ARG="--full_model"
fi

NO_VLLM_ARG=""
if [ "$USE_VLLM" = "false" ]; then
    NO_VLLM_ARG="--no_vllm"
fi

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

BASE_MODEL_ARG=""
if [ -n "$BASE_MODEL_PATH" ]; then
    BASE_MODEL_ARG="--base_model_path $BASE_MODEL_PATH"
fi

python evaluate_sft_competitions.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --batch_size "$BATCH_SIZE" \
    $FEW_SHOT_ARG \
    $MERGE_ARG \
    $FULL_MODEL_ARG \
    $NO_VLLM_ARG \
    $MAX_SAMPLES_ARG \
    $BASE_MODEL_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Competition Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
