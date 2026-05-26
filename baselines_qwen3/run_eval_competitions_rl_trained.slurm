#!/bin/bash
#SBATCH --job-name=qwen3_rl_eval_comp
#SBATCH --output=logs/qwen3_rl_eval_comp_%j.out
#SBATCH --error=logs/qwen3_rl_eval_comp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00

# Adjust partition as needed
#SBATCH --account=karthikn
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# QWEN3 RL-TRAINED COMPETITION EVALUATION (AIME26, HMMT25)
# ============================================================================
# Usage examples:
#   sbatch --export=ALL,BENCHMARK=aime26 run_eval_competitions_rl_trained.slurm
#   sbatch --export=ALL,BENCHMARK=hmmt25 run_eval_competitions_rl_trained.slurm
#   sbatch --export=ALL,BENCHMARK=aime26,MODEL_SIZE=1.7B run_eval_competitions_rl_trained.slurm
#   sbatch --export=ALL,BENCHMARK=aime26,MODEL_SIZE=4B run_eval_competitions_rl_trained.slurm
#   sbatch --export=ALL,BENCHMARK=aime26,MODEL_PATH=/path/to/global_step_140 run_eval_competitions_rl_trained.slurm
#
# Optional parameters:
#   BENCHMARK=aime26              - aime26 or hmmt25
#   MODEL_SIZE=4B                 - Built-in checkpoint to evaluate: 1.7B or 4B
#   MODEL_PATH=...                - GRPO checkpoint path (overrides MODEL_SIZE)
#   MAX_PROMPT_LENGTH=1024        - Prompt length
#   USE_FEW_SHOT=false            - Enable 5-shot CoT prompting
#   MAX_NEW_TOKENS=3072           - Max tokens
#   GPU_MEMORY_UTILIZATION=0.9    - vLLM GPU memory fraction
#   VLLM_TENSOR_PARALLEL_SIZE=1   - vLLM tensor parallel size
#   OUTPUT_DIR=...                - Custom output directory
#   MAX_SAMPLES=...               - Limit number of samples
#   CLEANUP_MERGED_MODEL=true     - Remove merged model after eval

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"aime26"} # aime26 or hmmt25
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

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
TEMPERATURE=0.0
MAX_SAMPLES=${MAX_SAMPLES:-""}
CLEANUP_MERGED_MODEL=${CLEANUP_MERGED_MODEL:-true}

CHECKPOINT_TAG=$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/qwen3_rltrained_${BENCHMARK}_${CHECKPOINT_TAG}_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}_${SLURM_JOB_ID}"}

# ============================================================================
# VALIDATE BENCHMARK / CHECKPOINT
# ============================================================================
if [ "$BENCHMARK" != "aime26" ] && [ "$BENCHMARK" != "hmmt25" ]; then
    echo "ERROR: BENCHMARK must be 'aime26' or 'hmmt25', got: $BENCHMARK"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

EVAL_MODEL_PATH="$MODEL_PATH"

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "Qwen3 RL-Trained Evaluation - ${BENCHMARK}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Model selection: $MODEL_SIZE"
echo "Checkpoint path: $MODEL_PATH"
echo "Eval model path: $EVAL_MODEL_PATH"
echo "Benchmark: $BENCHMARK"
echo "Output dir: $OUTPUT_DIR"
echo "Prompt length: $MAX_PROMPT_LENGTH"
echo "Few-shot (5-shot CoT): $USE_FEW_SHOT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "vLLM GPU mem util: $GPU_MEMORY_UTILIZATION"
echo "vLLM tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
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
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

python evaluate_competitions.py \
    --benchmark "$BENCHMARK" \
    --model_path "$EVAL_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --vllm_gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    $FEW_SHOT_ARG \
    $MAX_SAMPLES_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

if [ -d "${OUTPUT_DIR}/merged_model" ] && [ "$CLEANUP_MERGED_MODEL" = "true" ]; then
    echo "Cleaning up merged model directory..."
    rm -rf "${OUTPUT_DIR}/merged_model"
fi

echo ""
echo "=========================================="
echo "RL-Trained Competition Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
