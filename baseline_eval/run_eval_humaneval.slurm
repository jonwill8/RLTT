#!/bin/bash
#SBATCH --job-name=eval_ouro_humaneval
#SBATCH --output=logs/eval_ouro_humaneval_%j.out
#SBATCH --error=logs/eval_ouro_humaneval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Adjust partition/account as needed
#SBATCH --partition=ailab
# #SBATCH --constraint=mig
# #SBATCH --partition=gpu80

# ============================================================================
# OURO THINKING BASELINE EVALUATION - HUMANEVAL
# ============================================================================
# This script evaluates Ouro thinking models on HumanEval code generation.
#
# Usage:
#   sbatch --export=ALL,MODEL_SIZE=1.4B run_eval_humaneval.slurm
#
# Optional parameters (via --export):
#   MODEL_SIZE=1.4B          - Ouro model size to evaluate (1.4B or 2.6B; default: 2.6B)
#   MODEL_PATH=...           - Exact model path; overrides MODEL_SIZE
#   USE_FEW_SHOT=true         - Use few-shot prompting (default: false)
#   MAX_NEW_TOKENS=512        - Max tokens to generate
#   MAX_PROMPT_LENGTH=2048    - Max prompt length
#   NUM_SAMPLES=1             - Number of samples per problem (for pass@k)
#   EXECUTION_TIMEOUT=5.0     - Code execution timeout in seconds
#   GPU_MEMORY_UTILIZATION=0.9 - vLLM GPU memory fraction

source /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval/resolve_ouro_model.sh

echo "=========================================="
echo "Baseline Evaluation: ${MODEL_NAME} on HumanEval"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Set offline mode for HuggingFace (no internet access)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories (use local paths)
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Fix for MIG-enabled nodes: vLLM can't parse MIG UUIDs in CUDA_VISIBLE_DEVICES
# Override with simple device index if we detect a MIG UUID
if [[ "$CUDA_VISIBLE_DEVICES" == *"MIG"* ]]; then
    echo "Detected MIG GPU, setting CUDA_VISIBLE_DEVICES=0"
    export CUDA_VISIBLE_DEVICES=0
fi

# Working directory (use SLURM_SUBMIT_DIR since BASH_SOURCE doesn't work in SLURM)
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval}"
cd $WORK_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Load CUDA toolkit
module load cudatoolkit/12.8

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Configuration
TEST_FILE="/scratch/gpfs/OLGARUS/jw4199/datasets/code/humaneval.test.jsonl"

# Evaluation parameters (with defaults)
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.0}
NUM_SAMPLES=${NUM_SAMPLES:-1}
EXECUTION_TIMEOUT=${EXECUTION_TIMEOUT:-5.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}

# Output settings
FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/${MODEL_TAG}_humaneval_prompt${MAX_PROMPT_LENGTH}_${FEW_SHOT_TAG}_tokens${MAX_NEW_TOKENS}_${SLURM_JOB_ID}"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model path: $MODEL_PATH"
echo "  Dataset: $TEST_FILE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Evaluation Settings:"
echo "  Few-shot prompting: $USE_FEW_SHOT"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Num samples: $NUM_SAMPLES"
echo "  Execution timeout: $EXECUTION_TIMEOUT"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo ""
echo "Starting evaluation..."
echo ""

# Build optional arguments
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

# Run evaluation
python eval_ouro_code.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --num_samples "$NUM_SAMPLES" \
    --execution_timeout "$EXECUTION_TIMEOUT" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    $FEW_SHOT_ARG

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "HumanEval Evaluation Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/eval_results.json"
echo "  cat $OUTPUT_DIR/eval_summary.txt"
echo "=========================================="
