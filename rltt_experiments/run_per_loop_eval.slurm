#!/bin/bash
#SBATCH --job-name=rltt_per_loop_eval
#SBATCH --output=logs/rltt_per_loop_eval_%j.out
#SBATCH --error=logs/rltt_per_loop_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

# Adjust partition as needed
#SBATCH --constraint=gpu80
# #SBATCH --partition=ailab

# ============================================================================
# RLTT Per-Loop Evaluation
# ============================================================================
# This script evaluates an RLTT checkpoint with a fixed number of recursive loops
# to analyze how model performance changes with loop count.
#
# Usage:
#   # Evaluate with 1 loop on MATH-500:
#   sbatch --export=ALL,EXPERIMENT_DIR=/path/to/rltt_output/3374111,CHECKPOINT=step_140,NUM_LOOPS=1,BENCHMARK=math500 run_per_loop_eval.slurm
#
#   # Evaluate with 2 loops on GSM8K:
#   sbatch --export=ALL,CHECKPOINT=step_140,NUM_LOOPS=2,BENCHMARK=gsm8k run_per_loop_eval.slurm
#
#   # Run all loop counts (1, 2, 3) on MATH-500:
#   for loops in 1 2 3; do
#     sbatch --export=ALL,NUM_LOOPS=$loops,BENCHMARK=math500 run_per_loop_eval.slurm
#   done
#
# Required parameters:
#   NUM_LOOPS   - Number of recursive loops (1, 2, or 3)
#   BENCHMARK   - Benchmark to evaluate on (math500, gsm8k, aime24, or beyondaime)
#
# Optional parameters:
#   EXPERIMENT_DIR - Path to experiment output (default: rltt_output/3374111)
#   CHECKPOINT     - Checkpoint to evaluate (default: step_140)
#   OUTPUT_DIR     - Output directory (default: per_loop_results)
#   TEMPERATURE    - Sampling temperature (default: 0.0 for greedy)

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_LOOPS=${NUM_LOOPS:-1}
BENCHMARK=${BENCHMARK:-"beyondaime"}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3692225"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/per_loop_results"}
TEMPERATURE=${TEMPERATURE:-0.0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

# Validate NUM_LOOPS
if [[ ! "$NUM_LOOPS" =~ ^[123]$ ]]; then
    echo "ERROR: NUM_LOOPS must be 1, 2, or 3 (got: $NUM_LOOPS)"
    exit 1
fi

# Validate BENCHMARK
if [[ ! "$BENCHMARK" =~ ^(math500|gsm8k|aime24|beyondaime)$ ]]; then
    echo "ERROR: BENCHMARK must be math500, gsm8k, aime24, or beyondaime (got: $BENCHMARK)"
    exit 1
fi

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "RLTT Per-Loop Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Experiment dir: $EXPERIMENT_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Number of loops: $NUM_LOOPS"
echo "  Benchmark: $BENCHMARK"
echo "  Temperature: $TEMPERATURE"
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
echo "Running per-loop evaluation..."
echo "Evaluating with ${NUM_LOOPS} loop(s) on ${BENCHMARK}"
echo "=========================================="

python evaluate_per_loop.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_loops $NUM_LOOPS \
    --benchmark "$BENCHMARK" \
    --output_dir "$OUTPUT_DIR/rltt" \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --temperature $TEMPERATURE

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR/rltt"

exit $EXIT_CODE
