#!/bin/bash
#SBATCH --job-name=grpo_gsnr_eval
#SBATCH --output=logs/grpo_gsnr_eval_%j.out
#SBATCH --error=logs/grpo_gsnr_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

# Adjust partition as needed
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# GRPO GSNR Evaluation
# ============================================================================
# This script computes the Gradient Signal-to-Noise Ratio (GSNR) metric
# for GRPO checkpoints as described in the GSNR paper.
#
# GSNR measures the quality of the learning signal by computing:
# 1. For each prompt, sample R responses
# 2. Compute gradients for each response
# 3. Measure signal (mean gradient) vs noise (variance across gradients)
#
# Usage:
#   # Evaluate on MATH-500:
#   sbatch --export=ALL,BENCHMARK=math500 run_gsnr_eval.slurm
#
#   # Evaluate on AIME24 with custom checkpoint:
#   sbatch --export=ALL,BENCHMARK=aime24,CHECKPOINT=step_140 run_gsnr_eval.slurm
#
#   # Run all benchmarks:
#   for bench in math500 gsm8k aime24 beyondaime; do
#     sbatch --export=ALL,BENCHMARK=$bench run_gsnr_eval.slurm
#   done
#
# Required parameters:
#   BENCHMARK   - Benchmark to evaluate (math500, gsm8k, aime24, beyondaime)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: grpo_output/3374119)
#   CHECKPOINT     - Checkpoint to evaluate (default: step_140)
#   NUM_ROLLOUTS   - Number of rollouts per prompt (default: 8)
#   NUM_PROMPTS    - Number of prompts to evaluate (default: all)
#   TEMPERATURE    - Sampling temperature for generation (default: 0.2)
#   BATCH_SIZE     - Number of prompts to process at a time (default: 32)
#   OUTPUT_DIR     - Output directory (default: gsnr_results/grpo)
#   INCLUDE_INTERMEDIATE_LOOPS - Set to 1 to compute GSNR w.r.t. all loop logits (default: 0)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK=${BENCHMARK:-"gsm8k"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3692220"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-8}
NUM_PROMPTS=${NUM_PROMPTS:-""}
TEMPERATURE=${TEMPERATURE:-0.2}
BATCH_SIZE=${BATCH_SIZE:-32}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results/grpo"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
INCLUDE_INTERMEDIATE_LOOPS=${INCLUDE_INTERMEDIATE_LOOPS:-1}

# Validate BENCHMARK
if [[ ! "$BENCHMARK" =~ ^(math500|gsm8k|aime24|beyondaime)$ ]]; then
    echo "ERROR: BENCHMARK must be math500, gsm8k, aime24, or beyondaime (got: $BENCHMARK)"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "GRPO GSNR Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Benchmark: $BENCHMARK"
echo "  Num rollouts (R): $NUM_ROLLOUTS"
echo "  Num prompts: ${NUM_PROMPTS:-all}"
echo "  Temperature: $TEMPERATURE"
echo "  Batch size: $BATCH_SIZE"
echo "  Include intermediate loops: $INCLUDE_INTERMEDIATE_LOOPS"
echo "  Output dir: $OUTPUT_DIR"
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
echo "Running GSNR evaluation..."
echo "=========================================="

# Build command
CMD="python gsnr_eval.py \
    --experiment_dir \"$EXPERIMENT_DIR\" \
    --checkpoint \"$CHECKPOINT\" \
    --benchmark \"$BENCHMARK\" \
    --num_rollouts $NUM_ROLLOUTS \
    --temperature $TEMPERATURE \
    --batch_size $BATCH_SIZE \
    --output_dir \"$OUTPUT_DIR\" \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"

# Add optional num_prompts argument
if [ -n "$NUM_PROMPTS" ]; then
    CMD="$CMD --num_prompts $NUM_PROMPTS"
fi

# Add include_intermediate_loops flag if enabled
if [ "$INCLUDE_INTERMEDIATE_LOOPS" = "1" ]; then
    CMD="$CMD --include_intermediate_loops"
fi

echo "Running command: $CMD"
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "GSNR Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
