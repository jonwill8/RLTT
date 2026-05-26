#!/bin/bash
#SBATCH --job-name=flops_deepseek
#SBATCH --output=logs/flops_deepseek_%j.out
#SBATCH --error=logs/flops_deepseek_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# DEEPSEEK-R1 FLOPs PROFILING (HF generate + DeepSpeed FLOPs profiler)
# ============================================================================
# Usage:
#   sbatch --export=ALL,MODEL_SIZE=1.5B run_flops_deepseek.slurm
#   sbatch --export=ALL,MODEL_SIZE=7B run_flops_deepseek.slurm
#   sbatch --export=ALL,MODEL_PATH=/custom/deepseek/path run_flops_deepseek.slurm

set -euo pipefail

BENCHMARK="aime26"
MODEL_SIZE=${MODEL_SIZE:-"1.5B"}
MODEL_PATH=${MODEL_PATH:-""}
MAX_SAMPLES=${MAX_SAMPLES:-30}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.0}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}
SEED=${SEED:-42}

WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/flop_analysis}"
cd "$WORK_DIR"
mkdir -p logs

source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm
module load cudatoolkit/12.8

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TOKENIZERS_PARALLELISM=false

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *"MIG"* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

python -c "import deepspeed" >/dev/null 2>&1 || {
  echo "ERROR: deepspeed is not installed in env ouro_vllm"
  echo "Install with: pip install deepspeed"
  exit 1
}

if [ -z "$MODEL_PATH" ]; then
  case "${MODEL_SIZE,,}" in
    "1.5b"|"15b"|"1.5")
      MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-1.5B"
      ;;
    "7b"|"7")
      MODEL_PATH="/scratch/gpfs/OLGARUS/jw4199/model_weights_path/DeepSeek-R1-Distill-Qwen-7B"
      ;;
    *)
      echo "ERROR: MODEL_SIZE must be 1.5B or 7B (or pass MODEL_PATH)"
      exit 1
      ;;
  esac
fi

FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/deepseek_${MODEL_SIZE}_${BENCHMARK}_samples${MAX_SAMPLES}_${FEW_SHOT_TAG}_${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_DIR"

FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
  FEW_SHOT_ARG="--use_few_shot"
fi

echo "=========================================="
echo "DeepSeek FLOPs Profiling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Benchmark: $BENCHMARK"
echo "Model size: $MODEL_SIZE"
echo "Model path: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Start: $(date)"
echo "=========================================="

python profile_generation_flops.py \
  --model_path "$MODEL_PATH" \
  --max_samples "$MAX_SAMPLES" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  $FEW_SHOT_ARG

echo "=========================================="
echo "FLOPs profiling complete"
echo "Summary: ${OUTPUT_DIR}/flop_profile_summary.json"
echo "Details: ${OUTPUT_DIR}/flop_profile_details.jsonl"
echo "Live rollouts: ${OUTPUT_DIR}/rollouts.jsonl"
echo "Responses: ${OUTPUT_DIR}/sample_responses.jsonl"
echo "Per-sample FLOPs: ${OUTPUT_DIR}/per_sample_flops.jsonl"
echo "End: $(date)"
echo "=========================================="

