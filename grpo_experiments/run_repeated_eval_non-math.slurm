#!/bin/bash
#SBATCH --job-name=grpo_repeated_eval_nonmath
#SBATCH --output=logs/grpo_repeated_eval_nonmath_%j.out
#SBATCH --error=logs/grpo_repeated_eval_nonmath_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Adjust partition as needed
#SBATCH --account=olgarus
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab


# ============================================================================
# GRPO Repeated Evaluation for Non-Math Benchmarks (Statistical Significance)
# ============================================================================
# This script runs repeated evaluations on non-math benchmarks to generate
# data for t-test analysis comparing GRPO vs RLTT.
#
# Usage:
#   sbatch --export=ALL,BENCHMARK=arc_c,CHECKPOINT=step_140 run_repeated_eval_non-math.slurm
#
# Required parameters:
#   BENCHMARK - One of: arc_c, mmlu_st, gpqa
#   CHECKPOINT - Checkpoint to evaluate (e.g., step_140)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: grpo_output/3692220)
#   NUM_RUNS - Number of evaluation runs (default: 10)
#   TEMPERATURE - Sampling temperature (default: 0.2)
#   OUTPUT_DIR - Output directory (default: repeated_eval_results_non-math)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"gpqa"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/6246436"} #3692220 | 6246436
NUM_RUNS=${NUM_RUNS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results_non-math"}

# Benchmark-specific settings
case $BENCHMARK in
    arc_c)
        SCRIPT="repeated_eval_arc_c.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
        ;;
    mmlu_st)
        SCRIPT="repeated_eval_mmlu_st.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
        ;;
    gpqa)
        SCRIPT="repeated_eval_gpqa.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCHMARK"
        echo "Valid options: arc_c, mmlu_st, gpqa"
        exit 1
        ;;
esac

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "GRPO Repeated Evaluation (Non-Math)"
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
    --output_dir "$OUTPUT_DIR" \
    --method grpo

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
