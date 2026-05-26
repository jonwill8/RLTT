#!/bin/bash
#SBATCH --job-name=grpo_rollouts
#SBATCH --output=logs/grpo_rollouts_%j.out
#SBATCH --error=logs/grpo_rollouts_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Adjust partition as needed
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# GRPO Greedy Rollout Generation
# ============================================================================
# This script generates greedy (temperature=0.0) model responses for a specified
# benchmark and saves them to the centralized rollout_comparison_results directory.
#
# Usage:
#   sbatch --export=ALL,BENCHMARK=math500,CHECKPOINT=step_140 run_generate_rollouts.slurm
#
# Required parameters:
#   BENCHMARK - One of: math500, gsm8k, aime24, beyondaime
#   CHECKPOINT - Checkpoint to evaluate (e.g., step_140)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: grpo_output/3374119)
#   OUTPUT_DIR - Output directory (default: rollout_comparison_results)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"gsm8k"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3374119"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rollout_comparison_results"}

# Benchmark-specific settings (from run_repeated_eval.slurm)
case $BENCHMARK in
    math500)
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    gsm8k)
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
        ;;
    aime24)
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    beyondaime)
        MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
        MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
        ;;
    *)
        echo "ERROR: Unknown benchmark: $BENCHMARK"
        echo "Valid options: math500, gsm8k, aime24, beyondaime"
        exit 1
        ;;
esac

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "GRPO Greedy Rollout Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Checkpoint: $CHECKPOINT"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Max prompt length: $MAX_PROMPT_LENGTH"
echo "Temperature: 0.0 (greedy)"
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
# RUN ROLLOUT GENERATION
# ============================================================================
echo "=========================================="
echo "Generating greedy rollouts..."
echo "=========================================="

python generate_rollouts.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR" \
    --method grpo

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Generation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
