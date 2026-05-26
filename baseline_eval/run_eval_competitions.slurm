#!/bin/bash
#SBATCH --job-name=eval_ouro_comp
#SBATCH --output=logs/eval_ouro_comp_%j.out
#SBATCH --error=logs/eval_ouro_comp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Adjust partition as needed
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# BASELINE OURO EVALUATION - Competition Benchmarks (AIME26, HMMT25)
# ============================================================================
# Usage:
#   sbatch --export=ALL,BENCHMARK=aime26,MODEL_SIZE=1.4B run_eval_competitions.slurm
#   sbatch --export=ALL,BENCHMARK=hmmt25,MODEL_SIZE=1.4B run_eval_competitions.slurm
#
# Optional parameters:
#   MODEL_SIZE=1.4B                - Ouro model size to evaluate (1.4B or 2.6B; default: 2.6B)
#   MODEL_PATH=...                 - Exact model path; overrides MODEL_SIZE
#   MAX_PROMPT_LENGTH=1024         - Max prompt length
#   USE_FEW_SHOT=false             - Use 5-shot CoT prompting
#   MAX_NEW_TOKENS=3072            - Max tokens to generate
#   GPU_MEMORY_UTILIZATION=0.9     - vLLM GPU memory fraction
#   VLLM_TENSOR_PARALLEL_SIZE=1    - vLLM tensor parallel size
#   USE_VLLM=true                  - Enable vLLM (set false to force HF)
#   OUTPUT_DIR=...                 - Custom output directory
#   BATCH_SIZE=32                  - HF fallback batch size

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"aime26"} # aime26 or hmmt25

source /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval/resolve_ouro_model.sh

# Prompt configuration
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# Generation settings
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
TEMPERATURE=${TEMPERATURE:-0.0}
BATCH_SIZE=${BATCH_SIZE:-32}

# vLLM toggle
USE_VLLM=${USE_VLLM:-true}

# Output settings
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/${MODEL_TAG}_${BENCHMARK}_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}_${SLURM_JOB_ID}"}

# ============================================================================
# VALIDATE BENCHMARK
# ============================================================================
if [ "$BENCHMARK" != "aime26" ] && [ "$BENCHMARK" != "hmmt25" ]; then
    echo "ERROR: BENCHMARK must be 'aime26' or 'hmmt25', got: $BENCHMARK"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "Ouro Baseline Evaluation - ${BENCHMARK}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "Prompt length: $MAX_PROMPT_LENGTH"
echo "Few-shot (5-shot CoT): $USE_FEW_SHOT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "vLLM GPU mem util: $GPU_MEMORY_UTILIZATION"
echo "vLLM tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
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
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval}"
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
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

VLLM_ARG=""
if [ "$USE_VLLM" = "false" ]; then
    VLLM_ARG="--no_vllm"
fi

python evaluate_competitions.py \
    --benchmark "$BENCHMARK" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --vllm_gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    $FEW_SHOT_ARG \
    $VLLM_ARG

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
