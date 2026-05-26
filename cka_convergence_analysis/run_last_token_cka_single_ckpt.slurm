#!/bin/bash
#SBATCH --job-name=cka_last_token
#SBATCH --output=logs/cka_last_token_%j.out
#SBATCH --error=logs/cka_last_token_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# LAST-TOKEN-ACROSS-PROMPTS CKA (AIME26)
# ============================================================================
# Usage examples:
#   sbatch --export=ALL,METHOD=rltt,CHECKPOINT_PATH=/.../RLTT/rltt_experiments/rltt_output/<run>/global_step_140 run_last_token_cka_single_ckpt.slurm
#   sbatch --export=ALL,METHOD=grpo,CHECKPOINT_PATH=/.../RLTT/grpo_experiments/grpo_output/<run>/global_step_140 run_last_token_cka_single_ckpt.slurm
#
# Optional exports:
#   TOTAL_UT_STEPS=4
#   USE_FEW_SHOT=true|false
#   MAX_SAMPLES=30
#   MAX_PROMPT_LENGTH=2048
#   BATCH_SIZE=8
#   OUTPUT_DIR=/custom/output/path  (default: cka_convergence_analysis/runs/<run_name>)
#   CLEANUP_MERGED_MODEL=true|false

set -euo pipefail

METHOD=${METHOD:-"grpo"} # rltt | grpo
BENCHMARK=${BENCHMARK:-"aime26"}

if [ "$METHOD" = "rltt" ]; then
    CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3699742/global_step_140"}
elif [ "$METHOD" = "grpo" ]; then
    CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3692220/global_step_140"}
else
    echo "ERROR: METHOD must be rltt or grpo, got: $METHOD"
    exit 1
fi

TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}
MAX_SAMPLES=${MAX_SAMPLES:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-8}
CLEANUP_MERGED_MODEL=${CLEANUP_MERGED_MODEL:-false}

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH"
    exit 1
fi

FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
RUN_TAG=$(date +%Y%m%d_%H%M%S)
STEP_NAME=$(basename "$CHECKPOINT_PATH")
CKA_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/cka_convergence_analysis"
OUTPUT_DIR=${OUTPUT_DIR:-"${CKA_DIR}/runs/${METHOD}_${BENCHMARK}_${STEP_NAME}_loops${TOTAL_UT_STEPS}_${FEW_SHOT_TAG}_samples${MAX_SAMPLES}_${RUN_TAG}"}

echo "=========================================="
echo "Last-token Across Prompts CKA Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Start time: $(date)"
echo ""
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Benchmark: $BENCHMARK"
echo "UT steps: $TOTAL_UT_STEPS"
echo "Output dir: $OUTPUT_DIR"
echo ""

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TOKENIZERS_PARALLELISM=false

WORK_DIR="${SLURM_SUBMIT_DIR:-$CKA_DIR}"
cd "$WORK_DIR"
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm
module load cudatoolkit/12.8

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
    FEW_SHOT_ARG="--use_few_shot"
fi

CLEANUP_ARG=""
if [ "$CLEANUP_MERGED_MODEL" = "true" ]; then
    CLEANUP_ARG="--cleanup_merged_model"
fi

python last_token_cka_competitions.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --method "$METHOD" \
    --benchmark "$BENCHMARK" \
    --total_ut_steps "$TOTAL_UT_STEPS" \
    --max_samples "$MAX_SAMPLES" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    $FEW_SHOT_ARG \
    $CLEANUP_ARG

echo ""
echo "=========================================="
echo "CKA analysis complete"
echo "=========================================="
echo "End time: $(date)"
echo "Output dir: $OUTPUT_DIR"
echo "Main JSON: $OUTPUT_DIR/cka_results.json"
echo "Run config: $OUTPUT_DIR/run_config.json"
echo "Heatmap: $OUTPUT_DIR/cka_heatmap.png"
echo "Curve: $OUTPUT_DIR/convergence_to_final_curve.png"
