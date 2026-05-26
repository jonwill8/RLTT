#!/bin/bash
#SBATCH --job-name=rltt_eval_gpqa
#SBATCH --output=logs/rltt_eval_gpqa_%j.out
#SBATCH --error=logs/rltt_eval_gpqa_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Adjust partition as needed
#SBATCH --account=olgarus
#SBATCH --partition=ailab
# #SBATCH --constraint=gpu80

# ============================================================================
# RLTT CHECKPOINT EVALUATION ON GPQA
# ============================================================================
# Usage examples:
#   # Basic evaluation with default RLTT checkpoint:
#   sbatch run_rltt_eval_gpqa.slurm
#
#   # Evaluate with a different checkpoint:
#   sbatch --export=ALL,EXPERIMENT_DIR=/path/to/rltt_output/3374111,CHECKPOINT=step_140 run_rltt_eval_gpqa.slurm
#
#   # Evaluate with different max tokens:
#   sbatch --export=ALL,MAX_NEW_TOKENS=4096 run_rltt_eval_gpqa.slurm

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
# Experiment directory and checkpoint step (mirrors run_per_loop_eval.slurm)
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/6221068"}
CHECKPOINT=${CHECKPOINT:-"step_140"}

# Construct full checkpoint path (convert step_X to global_step_X/actor)
STEP_NUM=${CHECKPOINT#step_}
CHECKPOINT_PATH="${EXPERIMENT_DIR}/global_step_${STEP_NUM}/actor"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Total recurrent steps for Ouro model (controls thinking depth)
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}

# Whether to merge LoRA weights into base model (faster inference, uses more memory)
MERGE_WEIGHTS=${MERGE_WEIGHTS:-true}

# Whether checkpoint is a full model (not LoRA). Set to true for full param checkpoints.
FULL_MODEL=${FULL_MODEL:-true}

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================
# Max prompt length in tokens - GPQA has long prompts
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}

# Use few-shot prompting instead of zero-shot
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
# Maximum tokens to generate per problem - GPQA may need more tokens for reasoning
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

# Sampling temperature (0.0 = greedy/deterministic)
TEMPERATURE=${TEMPERATURE:-0.0}

# vLLM GPU memory utilization
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# ============================================================================
# DATA SETTINGS
# ============================================================================
# Path to GPQA test file
TEST_FILE=${TEST_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/gpqa.test.jsonl"}

# Maximum samples to evaluate (empty = all)
MAX_SAMPLES=${MAX_SAMPLES:-""}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_outputs/gpqa_${FEW_SHOT_TAG}_${SLURM_JOB_ID}"}

# ============================================================================
# CHECK REQUIRED ARGUMENTS
# ============================================================================
# Validate checkpoint path exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo "Usage:"
    echo "  sbatch --export=ALL,EXPERIMENT_DIR=/path/to/rltt_output/3374111,CHECKPOINT=step_140 run_rltt_eval_gpqa.slurm"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Checkpoint Evaluation on GPQA"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
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

# Fix for MIG-enabled nodes
if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]]; then
    echo "Detected MIG GPU, setting CUDA_VISIBLE_DEVICES=0"
    export CUDA_VISIBLE_DEVICES=0
fi

# Working directory
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments}"
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
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Total UT steps: $TOTAL_UT_STEPS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
echo "Use few-shot: $USE_FEW_SHOT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Merge LoRA weights: $MERGE_WEIGHTS"
echo "Full model (not LoRA): $FULL_MODEL"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

echo "Starting evaluation..."
echo ""

# Build optional arguments
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

MERGE_ARG=""
if [ "$MERGE_WEIGHTS" = "true" ]; then
    MERGE_ARG="--merge_weights"
fi

FULL_MODEL_ARG=""
if [ "$FULL_MODEL" = "true" ]; then
    FULL_MODEL_ARG="--full_model"
fi

python eval_rltt_mcqa.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --temperature "$TEMPERATURE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    $FEW_SHOT_ARG \
    $MAX_SAMPLES_ARG \
    $MERGE_ARG \
    $FULL_MODEL_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "GPQA Evaluation complete!"
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
