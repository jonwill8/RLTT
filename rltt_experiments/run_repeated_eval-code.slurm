#!/bin/bash
#SBATCH --job-name=rltt_repeated_eval_code
#SBATCH --output=logs/rltt_repeated_eval_code_%j.out
#SBATCH --error=logs/rltt_repeated_eval_code_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# Adjust partition as needed
#SBATCH --account=olgarus
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# RLTT Repeated Evaluation for Code Benchmarks (Statistical Significance)
# ============================================================================
# This script runs repeated evaluations on code benchmarks (HumanEval, MBPP)
# to generate data for t-test analysis comparing GRPO vs RLTT.
#
# Usage:
#   sbatch --export=ALL,BENCHMARK=humaneval,CHECKPOINT=step_140 run_repeated_eval-code.slurm
#
# Required parameters:
#   BENCHMARK - One of: humaneval, mbpp
#   CHECKPOINT - Checkpoint to evaluate (e.g., step_140)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: rltt_output/3692225)
#   NUM_RUNS - Number of evaluation runs (default: 10)
#   TEMPERATURE - Sampling temperature (default: 0.2)
#   OUTPUT_DIR - Output directory (default: repeated_eval_results_code)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"mbpp"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/6221068"}
NUM_RUNS=${NUM_RUNS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results_code"}

# Benchmark-specific settings
case $BENCHMARK in
    mbpp)
        SCRIPT="repeated_eval_mbpp.py"
        TEST_FILE="/scratch/gpfs/OLGARUS/jw4199/datasets/code/mbpp.test.jsonl"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        EXECUTION_TIMEOUT=${EXECUTION_TIMEOUT:-5.0}
        ;;
    humaneval)
        SCRIPT="repeated_eval_humaneval.py"
        TEST_FILE="/scratch/gpfs/OLGARUS/jw4199/datasets/code/humaneval.test.jsonl"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
        EXECUTION_TIMEOUT=${EXECUTION_TIMEOUT:-5.0}
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCHMARK"
        echo "Valid options: humaneval, mbpp"
        exit 1
        ;;
esac

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Repeated Evaluation (Code)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Checkpoint: $CHECKPOINT"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Number of runs: $NUM_RUNS"
echo "Temperature: $TEMPERATURE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
echo "Execution timeout: $EXECUTION_TIMEOUT"
echo "Test file: $TEST_FILE"
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
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments}"
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
echo "=========================================="
echo "Running repeated evaluation..."
echo "=========================================="

python $SCRIPT \
    --experiment_dir "$EXPERIMENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_runs $NUM_RUNS \
    --temperature $TEMPERATURE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --execution_timeout $EXECUTION_TIMEOUT \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --method rltt

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
