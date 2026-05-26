#!/bin/bash
# ============================================================================
# Run all Qwen3 baseline evaluations
# ============================================================================
# This script submits evaluation jobs for all Qwen3 model sizes across
# all benchmark datasets.
#
# Usage:
#   ./run_all_evals.sh           # Submit all jobs
#   ./run_all_evals.sh --dry-run # Show commands without submitting
#
# This will submit 12 jobs total:
#   - 2 models (1.7B, 4B) x 4 datasets (MATH-500, GSM8K, AIME24, BeyondAIME)

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "DRY RUN - Commands will be shown but not executed"
    echo ""
fi

cd "$(dirname "$0")"

MODELS=("1.7B" "4B")
DATASETS=("math500" "gsm8k" "aime24" "beyondaime")

echo "=========================================="
echo "Qwen3 Baseline Evaluation Submission"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Total jobs: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo ""

JOB_IDS=()

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        CMD="sbatch --export=ALL,MODEL_SIZE=${model} run_eval_${dataset}.slurm"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] $CMD"
        else
            echo "Submitting: Qwen3-${model} on ${dataset}..."
            OUTPUT=$($CMD)
            JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+')
            JOB_IDS+=("$JOB_ID")
            echo "  -> $OUTPUT"
        fi
    done
done

echo ""
echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete. Use without --dry-run to submit jobs."
else
    echo "All jobs submitted!"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel all with: scancel ${JOB_IDS[*]}"
fi
echo "=========================================="
