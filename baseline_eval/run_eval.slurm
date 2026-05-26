#!/bin/bash
#SBATCH --job-name=eval_ouro_math500
#SBATCH --output=logs/eval_ouro_%j.out
#SBATCH --error=logs/eval_ouro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

#SBATCH --account=olgarus
# #SBATCH --partition=pli-c
#SBATCH --partition=ailab
# #SBATCH --partition=gpu80

source /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval/resolve_ouro_model.sh

echo "=========================================="
echo "Baseline Evaluation: ${MODEL_NAME} on MATH-500"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Set offline mode for HuggingFace (no internet access)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories (use local paths)
export HF_HOME=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export HF_DATASETS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path
export TRANSFORMERS_CACHE=/scratch/gpfs/OLGARUS/jw4199/model_weights_path

# Disable tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false

# Working directory (use SLURM_SUBMIT_DIR since BASH_SOURCE doesn't work in SLURM)
WORK_DIR="${SLURM_SUBMIT_DIR:-/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval}"
cd $WORK_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo "math_verify available: $(python -c 'import math_verify; print("yes")' 2>/dev/null || echo 'no')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# Configuration
MATH500_FILE="/scratch/gpfs/OLGARUS/jw4199/datasets/MATH-500/MATH-500.test.jsonl"
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_output/${MODEL_TAG}_math500_${SLURM_JOB_ID}"}

# Evaluation parameters
MAX_PROMPT_LENGTH=1024
MAX_NEW_TOKENS=3072
TEMPERATURE=0.0
VLLM_GPU_MEMORY_UTIL=0.9
VLLM_TENSOR_PARALLEL_SIZE=1
USE_FEW_SHOT=false  # controls zero-shot prompting
MATH500_SAMPLES=""  # Leave empty for all samples, or set to a number

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Model path: $MODEL_PATH"
echo "  Dataset: $MATH500_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: ${MATH500_SAMPLES:-all}"
echo ""
echo "Evaluation Settings:"
echo "  5-shot CoT prompting: $USE_FEW_SHOT"
echo "  Max prompt length: $MAX_PROMPT_LENGTH"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  vLLM GPU memory utilization: $VLLM_GPU_MEMORY_UTIL"
echo "  vLLM tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
echo ""
echo "Starting evaluation..."
echo ""

# Build optional arguments
FEW_SHOT_ARG=""
if [ "$USE_FEW_SHOT" = "false" ]; then
    FEW_SHOT_ARG="--no_few_shot"
fi

SAMPLES_ARG=""
if [ -n "$MATH500_SAMPLES" ]; then
    SAMPLES_ARG="--math500_samples $MATH500_SAMPLES"
fi

# Run evaluation
python eval_ouro_math500.py \
    --model_path "$MODEL_PATH" \
    --math500_file "$MATH500_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTIL" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    $FEW_SHOT_ARG \
    $SAMPLES_ARG

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  Full results:  $OUTPUT_DIR/math500_eval_results.json"
echo "  Summary:       $OUTPUT_DIR/math500_eval_summary.json"
echo "=========================================="
