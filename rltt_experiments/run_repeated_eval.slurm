#!/bin/bash
#SBATCH --job-name=rltt_repeated_eval
#SBATCH --output=logs/rltt_repeated_eval_%j.out
#SBATCH --error=logs/rltt_repeated_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Adjust partition as needed
#SBATCH --account=olgarus
# SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# RLTT Repeated Evaluation for Statistical Significance Testing
# ============================================================================
# This script runs repeated evaluations on a specific benchmark to generate
# data for t-test analysis comparing GRPO vs RLTT.
#
# Usage:
#   sbatch --export=ALL,BENCHMARK=math500,CHECKPOINT=step_140 run_repeated_eval.slurm
#
# Required parameters:
#   BENCHMARK - One of: math500, gsm8k, aime24, aime26, beyondaime
#   CHECKPOINT - Checkpoint to evaluate (e.g., step_140)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: rltt_output/3374111)
#   NUM_RUNS - Number of evaluation runs (default: 10)
#   TEMPERATURE - Sampling temperature (default: 0.7)
#   OUTPUT_DIR - Output directory (default: repeated_eval_results)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"aime26"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3692225"} #3692225 |6221068
NUM_RUNS=${NUM_RUNS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results"}

# Benchmark-specific settings
case $BENCHMARK in
    math500)
        SCRIPT="repeated_eval_math500.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    gsm8k)
        SCRIPT="repeated_eval_gsm8k.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
        ;;
    aime24)
        SCRIPT="repeated_eval_aime24.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    aime26)
        SCRIPT="repeated_eval_aime26.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    beyondaime)
        SCRIPT="repeated_eval_beyondaime.py"
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCHMARK"
        echo "Valid options: math500, gsm8k, aime24, aime26, beyondaime"
        exit 1
        ;;
esac

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Repeated Evaluation"
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
