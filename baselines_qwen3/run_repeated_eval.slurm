#!/bin/bash
#SBATCH --job-name=qwen3_repeated_eval
#SBATCH --output=logs/qwen3_repeated_eval_%j.out
#SBATCH --error=logs/qwen3_repeated_eval_%j.err
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
# Qwen3 GRPO Repeated Evaluation for Statistical Significance Testing
# ============================================================================
# This script runs repeated evaluations on math benchmarks to generate
# data for t-test analysis comparing Qwen3 GRPO-trained checkpoints.
#
# Usage:
#   sbatch --export=ALL,MODEL_SIZE=4B run_repeated_eval.slurm
#   sbatch --export=ALL,MODEL_SIZE=1.7B,BENCHMARKS="math500 gsm8k" run_repeated_eval.slurm
#   sbatch --export=ALL,MODEL_PATH=/path/to/global_step_140,BENCHMARK=aime24 run_repeated_eval.slurm
#
# Optional parameters:
#   BENCHMARKS - Space-separated list to run sequentially
#                (default: math500 gsm8k aime24 aime26 beyondaime)
#   BENCHMARK - Single benchmark to run; ignored if BENCHMARKS is set
#   MODEL_SIZE - Qwen3 checkpoint size: 1.7B or 4B (default: 4B)
#   MODEL_PATH - Exact global_step checkpoint path; overrides MODEL_SIZE/EXPERIMENT_DIR
#   EXPERIMENT_DIR - Parent experiment output directory
#   CHECKPOINT - Checkpoint to evaluate (default: step_140)
#   NUM_RUNS - Number of evaluation runs (default: 10)
#   TEMPERATURE - Sampling temperature (default: 0.2)
#   GPU_MEMORY_UTILIZATION - vLLM GPU memory fraction (default: 0.9)
#   OUTPUT_DIR - Output directory (default: repeated_eval_results)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_BENCHMARKS=(math500 gsm8k aime24 aime26 beyondaime)
if [ -n "${BENCHMARKS:-}" ]; then
    read -r -a EVAL_BENCHMARKS <<< "$BENCHMARKS"
elif [ -n "${BENCHMARK:-}" ]; then
    EVAL_BENCHMARKS=("$BENCHMARK")
else
    EVAL_BENCHMARKS=("${DEFAULT_BENCHMARKS[@]}")
fi

MODEL_SIZE=${MODEL_SIZE:-"4B"}
CHECKPOINT=${CHECKPOINT:-"step_140"}
NUM_RUNS=${NUM_RUNS:-10}
TEMPERATURE=${TEMPERATURE:-0.2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/repeated_eval_results"}
USER_MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
USER_MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-}

# Qwen3 checkpoint selection
case "$MODEL_SIZE" in
    1.7B|1.7b|1.7|1_7B|1_7b|1_7|qwen3_1.7B|qwen3_1.7b)
        DEFAULT_EXPERIMENT_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/grpo_output/qwen3_1.7B_6243864"
        ;;
    4B|4b|4|qwen3_4B|qwen3_4b)
        DEFAULT_EXPERIMENT_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3/grpo_output/qwen3_4B_6243862"
        ;;
    *)
        echo "ERROR: Unknown MODEL_SIZE: $MODEL_SIZE"
        echo "Valid options: 1.7B, 4B"
        exit 1
        ;;
esac

EXPERIMENT_DIR=${EXPERIMENT_DIR:-"$DEFAULT_EXPERIMENT_DIR"}

# MODEL_PATH may point directly at a global_step_* checkpoint directory.
if [ -n "${MODEL_PATH:-}" ]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
        exit 1
    fi
    STEP_DIR=$(basename "$MODEL_PATH")
    if [[ "$STEP_DIR" != global_step_* ]]; then
        echo "ERROR: MODEL_PATH must point to a global_step_* checkpoint directory"
        exit 1
    fi
    STEP_NUM=${STEP_DIR#global_step_}
    CHECKPOINT="step_${STEP_NUM}"
    EXPERIMENT_DIR=$(dirname "$MODEL_PATH")
fi

if [[ "$(basename "$EXPERIMENT_DIR")" == global_step_* ]]; then
    STEP_DIR=$(basename "$EXPERIMENT_DIR")
    STEP_NUM=${STEP_DIR#global_step_}
    CHECKPOINT="step_${STEP_NUM}"
    CHECKPOINT_PATH="$EXPERIMENT_DIR"
    EXPERIMENT_DIR=$(dirname "$EXPERIMENT_DIR")
else
    STEP_NUM=${CHECKPOINT#step_}
    STEP_NUM=${STEP_NUM#global_step_}
    CHECKPOINT="step_${STEP_NUM}"
    CHECKPOINT_PATH="${EXPERIMENT_DIR}/global_step_${STEP_NUM}"
fi

case "$EXPERIMENT_DIR" in
    *qwen3_1.7B*) MODEL_SIZE="1.7B" ;;
    *qwen3_4B*) MODEL_SIZE="4B" ;;
esac

set_benchmark_config() {
    local benchmark="$1"
    case $benchmark in
        math500)
            SCRIPT="repeated_eval_math500.py"
            DEFAULT_MAX_NEW_TOKENS=3072
            DEFAULT_MAX_PROMPT_LENGTH=1024
            ;;
        gsm8k)
            SCRIPT="repeated_eval_gsm8k.py"
            DEFAULT_MAX_NEW_TOKENS=3072
            DEFAULT_MAX_PROMPT_LENGTH=512
            ;;
        aime24)
            SCRIPT="repeated_eval_aime24.py"
            DEFAULT_MAX_NEW_TOKENS=3072
            DEFAULT_MAX_PROMPT_LENGTH=1024
            ;;
        aime26)
            SCRIPT="repeated_eval_aime26.py"
            DEFAULT_MAX_NEW_TOKENS=3072
            DEFAULT_MAX_PROMPT_LENGTH=1024
            ;;
        beyondaime)
            SCRIPT="repeated_eval_beyondaime.py"
            DEFAULT_MAX_NEW_TOKENS=3072
            DEFAULT_MAX_PROMPT_LENGTH=1024
            ;;
        *)
            echo "ERROR: Unknown benchmark: $benchmark"
            echo "Valid options: math500, gsm8k, aime24, aime26, beyondaime"
            return 1
            ;;
    esac

    MAX_NEW_TOKENS=${USER_MAX_NEW_TOKENS:-$DEFAULT_MAX_NEW_TOKENS}
    MAX_PROMPT_LENGTH=${USER_MAX_PROMPT_LENGTH:-$DEFAULT_MAX_PROMPT_LENGTH}
}

# ============================================================================
# JOB SETUP
# ============================================================================
echo "=========================================="
echo "Qwen3 GRPO Repeated Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""
echo "Benchmarks: ${EVAL_BENCHMARKS[*]}"
echo "Model size: $MODEL_SIZE"
echo "Checkpoint: $CHECKPOINT"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Number of runs: $NUM_RUNS"
echo "Temperature: $TEMPERATURE"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "Output dir: $OUTPUT_DIR"
echo ""

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

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
WORK_DIR="${WORK_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3}"
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
echo "Running repeated evaluations..."
echo "=========================================="

for BENCHMARK in "${EVAL_BENCHMARKS[@]}"; do
    if ! set_benchmark_config "$BENCHMARK"; then
        exit 1
    fi

    echo ""
    echo "------------------------------------------"
    echo "Benchmark: $BENCHMARK"
    echo "Script: $SCRIPT"
    echo "Max new tokens: $MAX_NEW_TOKENS"
    echo "Max prompt length: $MAX_PROMPT_LENGTH"
    echo "Start time: $(date)"
    echo "------------------------------------------"

    python "$SCRIPT" \
        --experiment_dir "$EXPERIMENT_DIR" \
        --checkpoint "$CHECKPOINT" \
        --num_runs "$NUM_RUNS" \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_prompt_length "$MAX_PROMPT_LENGTH" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --output_dir "$OUTPUT_DIR" \
        --method grpo

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Evaluation failed for benchmark: $BENCHMARK"
        echo "Exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi

    echo "Completed benchmark: $BENCHMARK"
done

EXIT_CODE=0

echo ""
echo "=========================================="
echo "All Evaluations Complete"
echo "=========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"

exit $EXIT_CODE
