#!/bin/bash
#SBATCH --job-name=grpo_eval_single
#SBATCH --output=logs/grpo_eval_single_%j.out
#SBATCH --error=logs/grpo_eval_single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Adjust partition as needed
# #SBATCH --constraint=gpu80
#SBATCH --account=olgarus
#SBATCH --partition=ailab

# ============================================================================
# SINGLE CHECKPOINT EVALUATION (GRPO)
# ============================================================================
# This script evaluates a SINGLE GRPO checkpoint on MATH-500.
#
# Usage:
#   sbatch --export=ALL,CHECKPOINT_PATH=/path/to/global_step_X run_eval_single_ckpt_grpo.slurm
#
# Example:
#   sbatch --export=ALL,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3692220/global_step_30 run_eval_single_ckpt_grpo.slurm
#
# Optional parameters:
#   USE_FEW_SHOT=true         - Use 5-shot CoT prompting (default: false)
#   MAX_NEW_TOKENS=3072       - Max tokens to generate (default: 2048)
#   MAX_PROMPT_LENGTH=1024    - Max prompt length (default: 1024)
#   GPU_MEMORY_UTILIZATION=0.9 - vLLM GPU memory fraction (default: 0.9)
#   OUTPUT_DIR=...            - Custom output directory (auto-generated if not set)

# ============================================================================
# CONFIGURATION
# ============================================================================
# CHECKPOINT_PATH - path to the global_step_X directory
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/6246436/global_step_140"}

# Prompt configuration
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# Generation settings
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# Data settings
MATH500_FILE=${MATH500_FILE:-"/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"}

# ============================================================================
# CHECK REQUIRED ARGUMENTS
# ============================================================================
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH is required"
    echo "Usage:"
    echo "  sbatch --export=ALL,CHECKPOINT_PATH=/path/to/global_step_X run_eval_single_ckpt_grpo.slurm"
    echo ""
    echo "Example:"
    echo "  sbatch --export=ALL,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3692220/global_step_30 run_eval_single_ckpt_grpo.slurm"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Extract step number and run directory from checkpoint path
STEP_NAME=$(basename "$CHECKPOINT_PATH")
STEP_NUM=$(echo "$STEP_NAME" | sed 's/global_step_//')
RUN_DIR=$(dirname "$CHECKPOINT_PATH")
RUN_ID=$(basename "$RUN_DIR")

# Output settings - results go in the checkpoint directory or custom location
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"${CHECKPOINT_PATH}/eval_math500_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "GRPO Single Checkpoint Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Step: $STEP_NUM"
echo "Run ID: $RUN_ID"
echo "Output dir: $OUTPUT_DIR"
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
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments}"
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
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo ""

# ============================================================================
# BASE MODEL CONFIGURATION
# ============================================================================
BASE_MODEL_DIR="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"

# ============================================================================
# SETUP PATHS
# ============================================================================
FSDP_CHECKPOINT_DIR="${CHECKPOINT_PATH}/actor"
MERGED_MODEL_DIR="${OUTPUT_DIR}/merged_model"

# Check if FSDP checkpoint exists
if [ ! -d "$FSDP_CHECKPOINT_DIR" ]; then
    echo "ERROR: FSDP checkpoint not found at $FSDP_CHECKPOINT_DIR"
    exit 1
fi

# ============================================================================
# FIX CHECKPOINT FILES (verl saves broken PEFT files)
# ============================================================================
CKPT_HF_DIR="${FSDP_CHECKPOINT_DIR}/huggingface"

echo "Fixing checkpoint files in: $CKPT_HF_DIR"

# Fix config.json - point to OuroForCausalLM, not PeftModelForCausalLM
if [ -f "$CKPT_HF_DIR/config.json" ]; then
    sed -i 's/"AutoModelForCausalLM": "peft_model.PeftModelForCausalLM"/"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"/g' \
        "$CKPT_HF_DIR/config.json"
    echo "  - Fixed config.json auto_map"
fi

# Copy modeling_ouro.py if missing
if [ ! -f "$CKPT_HF_DIR/modeling_ouro.py" ]; then
    cp "$BASE_MODEL_DIR/modeling_ouro.py" "$CKPT_HF_DIR/"
    echo "  - Copied modeling_ouro.py"
fi

# Fix utils.py imports (change relative to absolute peft imports)
if [ -f "$CKPT_HF_DIR/utils.py" ]; then
    sed -i 's/from \.integrations/from peft.utils.integrations/g' "$CKPT_HF_DIR/utils.py"
    sed -i 's/from \.loftq_utils/from peft.utils.loftq_utils/g' "$CKPT_HF_DIR/utils.py"
    sed -i 's/from \.other/from peft.utils.other/g' "$CKPT_HF_DIR/utils.py"
    sed -i 's/from \.peft_types/from peft.utils.peft_types/g' "$CKPT_HF_DIR/utils.py"
    sed -i 's/from \.save_and_load/from peft.utils.save_and_load/g' "$CKPT_HF_DIR/utils.py"
    sed -i 's/from \.warning/from peft.utils.warning/g' "$CKPT_HF_DIR/utils.py"
    echo "  - Fixed utils.py imports"
fi

# Fix peft_model.py imports
if [ -f "$CKPT_HF_DIR/peft_model.py" ]; then
    sed -i 's/from \. import __version__/from peft import __version__/g' "$CKPT_HF_DIR/peft_model.py"
    echo "  - Fixed peft_model.py imports"
fi

# Clear transformers cache to avoid stale cached modules
rm -rf /scratch/gpfs/OLGARUS/jw4199/model_weights_path/transformers_modules/huggingface/ 2>/dev/null
echo "  - Cleared transformers cache"
echo ""

# ============================================================================
# MERGE FSDP CHECKPOINT
# ============================================================================
echo "Merging FSDP checkpoint..."
echo "  Source: ${FSDP_CHECKPOINT_DIR}"
echo "  Target: ${MERGED_MODEL_DIR}"

python -m verl.model_merger merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir "$FSDP_CHECKPOINT_DIR" \
    --target_dir "$MERGED_MODEL_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to merge FSDP checkpoint"
    exit 1
fi

echo "Checkpoint merged successfully"
echo ""

# ============================================================================
# RUN EVALUATION
# ============================================================================
echo "Running evaluation..."

# Build few-shot argument
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

python evaluate_checkpoint.py \
    --checkpoint_dir "$MERGED_MODEL_DIR" \
    --math500_file "$MATH500_FILE" \
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
# DISPLAY RESULTS
# ============================================================================
RESULTS_JSON="${OUTPUT_DIR}/eval_results.json"

if [ -f "$RESULTS_JSON" ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation Results"
    echo "=========================================="
    python -c "
import json
with open('$RESULTS_JSON', 'r') as f:
    data = json.load(f)
metrics = data.get('metrics', {})
print(f\"Overall Accuracy: {metrics.get('overall_accuracy', 0)*100:.2f}%\")
print(f\"Correct: {metrics.get('overall_correct', 0)}/{metrics.get('overall_total', 0)}\")
print('')
print('Per-Level Results:')
for i in range(1, 6):
    level_key = f'Level {i}'
    acc = metrics.get(f'level_{level_key}_accuracy', 0)
    correct = metrics.get(f'level_{level_key}_correct', 0)
    total = metrics.get(f'level_{level_key}_total', 0)
    print(f'  Level {i}: {acc*100:.2f}% ({correct}/{total})')
"
fi

# ============================================================================
# CLEANUP MERGED MODEL TO SAVE SPACE
# ============================================================================
echo ""
echo "Cleaning up merged model directory to save space..."
rm -rf "$MERGED_MODEL_DIR"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Results JSON: $RESULTS_JSON"
echo ""
echo "To view results:"
echo "  cat $RESULTS_JSON"
echo "=========================================="
