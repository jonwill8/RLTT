#!/bin/bash
#SBATCH --job-name=rltt_pass_at_k
#SBATCH --output=logs/rltt_pass_at_k_%j.out
#SBATCH --error=logs/rltt_pass_at_k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

# Adjust partition as needed
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# RLTT Pass@k Evaluation for Entropy Collapse Investigation
# ============================================================================
# This script runs pass@k evaluation to investigate potential entropy collapse.
# For each problem, k responses are generated and pass@k metric computed.
#
# Usage:
#   sbatch --export=ALL,BENCHMARK=math500,CHECKPOINT=step_140,K=8 run_pass_at_k_eval.slurm
#
# Required parameters:
#   BENCHMARK - One of: math500, gsm8k, aime24, beyondaime
#   CHECKPOINT - Checkpoint to evaluate (e.g., step_140)
#   K - Number of samples per problem (e.g., 2, 4, 8, 16)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: rltt_output/3374111)
#   TEMPERATURE - Sampling temperature (default: 0.6)
#   OUTPUT_DIR - Output directory (default: pass_at_k_results)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"math500"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
K=${K:-32}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3692225"}
TEMPERATURE=${TEMPERATURE:-0.6}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/pass_at_k_results"}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Pass@k Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmark: $BENCHMARK"
echo "Checkpoint: $CHECKPOINT"
echo "K (samples per problem): $K"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Temperature: $TEMPERATURE"
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
# RUN PASS@K EVALUATION
# ============================================================================
echo "=========================================="
echo "Running pass@$K evaluation..."
echo "=========================================="

python pass_at_k_eval.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --benchmark "$BENCHMARK" \
    --k $K \
    --temperature $TEMPERATURE \
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
