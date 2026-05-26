#!/bin/bash
#SBATCH --job-name=flops_ouro
#SBATCH --output=logs/flops_ouro_%j.out
#SBATCH --error=logs/flops_ouro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
# #SBATCH --constraint=gpu80
#SBATCH --partition=ailab

# ============================================================================
# OURO FLOPs PROFILING (HF generate + DeepSpeed FLOPs profiler)
# ============================================================================
# Usage examples:
#   sbatch --export=ALL,EVAL_MODE=rltt,CHECKPOINT_PATH=/.../rltt_output/.../global_step_140 run_flops_ouro.slurm
#   sbatch --export=ALL,EVAL_MODE=grpo,CHECKPOINT_PATH=/.../grpo_output/.../global_step_140 run_flops_ouro.slurm
#   sbatch --export=ALL,EVAL_MODE=base run_flops_ouro.slurm
#   sbatch --export=ALL,EVAL_MODE=sft,CHECKPOINT_PATH=/.../checkpoint-final run_flops_ouro.slurm
#
# Optional:
#   EVAL_MODE=base|sft|rltt|grpo (default: sft)
#   MODEL_PATH=/custom/model/path (overrides BASE_MODEL_PATH for direct model eval)
#   BASE_MODEL_PATH=/scratch/.../Ouro-2.6B-Thinking-RLTT
#   MAX_SAMPLES=16
#   MAX_PROMPT_LENGTH=1024
#   MAX_NEW_TOKENS=3072
#   TEMPERATURE=0.0
#   USE_FEW_SHOT=true|false
#   TOTAL_UT_STEPS=4
#   OUTPUT_DIR=/custom/path

set -euo pipefail

BENCHMARK="aime26"
EVAL_MODE=${EVAL_MODE:-"sft"} # base | sft | rltt | grpo
MODEL_PATH=${MODEL_PATH:-""}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-2.6B-Thinking-RLTT"}

if [ "$EVAL_MODE" != "base" ] && [ "$EVAL_MODE" != "sft" ] && [ "$EVAL_MODE" != "rltt" ] && [ "$EVAL_MODE" != "grpo" ]; then
  echo "ERROR: EVAL_MODE must be one of: base, sft, rltt, grpo. Got: $EVAL_MODE"
  exit 1
fi

# Default checkpoint selection when evaluating checkpoint-based modes.
if [ -z "$MODEL_PATH" ] && [ -z "$CHECKPOINT_PATH" ]; then
  if [ "$EVAL_MODE" = "rltt" ]; then
    CHECKPOINT_PATH="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3699742/global_step_140"
  elif [ "$EVAL_MODE" = "grpo" ]; then
    CHECKPOINT_PATH="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3692220/global_step_140"
  elif [ "$EVAL_MODE" = "sft" ]; then
    CHECKPOINT_PATH="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments/sft_ouro_math_output/3706435/checkpoint-final"
  fi
fi

# In base mode, always run direct model eval (no checkpoint merge path).
if [ "$EVAL_MODE" = "base" ]; then
  CHECKPOINT_PATH=""
fi

MAX_SAMPLES=${MAX_SAMPLES:-30}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
TEMPERATURE=${TEMPERATURE:-0.0}
USE_FEW_SHOT=${USE_FEW_SHOT:-false}
TOTAL_UT_STEPS=${TOTAL_UT_STEPS:-4}
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

python -c "import deepspeed" >/dev/null 2>&1 || {
  echo "ERROR: deepspeed is not installed in env ouro_vllm"
  echo "Install with: pip install deepspeed"
  exit 1
}

FEW_SHOT_TAG=$([ "$USE_FEW_SHOT" = "true" ] && echo "fewshot" || echo "zeroshot")
if [ -n "$CHECKPOINT_PATH" ]; then
  STEP_NAME=$(basename "$CHECKPOINT_PATH")
else
  STEP_NAME="base"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/ouro_${EVAL_MODE}_${BENCHMARK}_${STEP_NAME}_samples${MAX_SAMPLES}_${FEW_SHOT_TAG}_${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Ouro FLOPs Profiling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Eval mode: ${EVAL_MODE}"
echo "Benchmark: $BENCHMARK"
echo "Checkpoint path: ${CHECKPOINT_PATH:-N/A}"
echo "Model path override: ${MODEL_PATH:-N/A}"
echo "Base model path: ${BASE_MODEL_PATH}"
echo "Output dir: $OUTPUT_DIR"
echo "Start: $(date)"
echo "=========================================="

PROFILE_MODEL_PATH=""
MERGED_MODEL_DIR=""

if [ -n "$CHECKPOINT_PATH" ]; then
  if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH does not exist: $CHECKPOINT_PATH"
    exit 1
  fi

  FSDP_CHECKPOINT_DIR="${CHECKPOINT_PATH}/actor"
  if [ -d "$FSDP_CHECKPOINT_DIR" ]; then
    MERGED_MODEL_DIR="${OUTPUT_DIR}/merged_model"
    CKPT_HF_DIR="${FSDP_CHECKPOINT_DIR}/huggingface"
    BASE_MODEL_DIR="${MODEL_PATH:-$BASE_MODEL_PATH}"

    echo "Fixing checkpoint files in: $CKPT_HF_DIR"
    if [ -f "$CKPT_HF_DIR/config.json" ]; then
      sed -i 's/"AutoModelForCausalLM": "peft_model.PeftModelForCausalLM"/"AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM"/g' "$CKPT_HF_DIR/config.json"
    fi
    if [ ! -f "$CKPT_HF_DIR/modeling_ouro.py" ]; then
      cp "$BASE_MODEL_DIR/modeling_ouro.py" "$CKPT_HF_DIR/"
    fi
    if [ -f "$CKPT_HF_DIR/utils.py" ]; then
      sed -i 's/from \.integrations/from peft.utils.integrations/g' "$CKPT_HF_DIR/utils.py"
      sed -i 's/from \.loftq_utils/from peft.utils.loftq_utils/g' "$CKPT_HF_DIR/utils.py"
      sed -i 's/from \.other/from peft.utils.other/g' "$CKPT_HF_DIR/utils.py"
      sed -i 's/from \.peft_types/from peft.utils.peft_types/g' "$CKPT_HF_DIR/utils.py"
      sed -i 's/from \.save_and_load/from peft.utils.save_and_load/g' "$CKPT_HF_DIR/utils.py"
      sed -i 's/from \.warning/from peft.utils.warning/g' "$CKPT_HF_DIR/utils.py"
    fi
    if [ -f "$CKPT_HF_DIR/peft_model.py" ]; then
      sed -i 's/from \. import __version__/from peft import __version__/g' "$CKPT_HF_DIR/peft_model.py"
    fi

    rm -rf /scratch/gpfs/OLGARUS/jw4199/model_weights_path/transformers_modules/huggingface/ 2>/dev/null || true

    echo "Merging FSDP checkpoint to HF format..."
    python -m verl.model_merger merge \
      --backend fsdp \
      --trust-remote-code \
      --local_dir "$FSDP_CHECKPOINT_DIR" \
      --target_dir "$MERGED_MODEL_DIR"

    PROFILE_MODEL_PATH="$MERGED_MODEL_DIR"
  elif [ "$EVAL_MODE" = "sft" ]; then
    echo "No actor dir found; using SFT checkpoint as direct HF model."
    PROFILE_MODEL_PATH="$CHECKPOINT_PATH"
  else
    echo "ERROR: FSDP checkpoint not found at $FSDP_CHECKPOINT_DIR"
    echo "Expected CHECKPOINT_PATH to be a training step directory containing /actor for EVAL_MODE=$EVAL_MODE."
    exit 1
  fi
elif [ -n "$MODEL_PATH" ]; then
  PROFILE_MODEL_PATH="$MODEL_PATH"
elif [ "$EVAL_MODE" = "base" ]; then
  PROFILE_MODEL_PATH="$BASE_MODEL_PATH"
else
  echo "ERROR: EVAL_MODE=$EVAL_MODE requires CHECKPOINT_PATH (or set MODEL_PATH for direct model eval)."
  exit 1
fi

FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "true" ]; then
  FEW_SHOT_ARG="--use_few_shot"
fi

python profile_generation_flops.py \
  --model_path "$PROFILE_MODEL_PATH" \
  --max_samples "$MAX_SAMPLES" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --total_ut_steps "$TOTAL_UT_STEPS" \
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

