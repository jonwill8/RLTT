#!/bin/bash
# ============================================================================
# Run All GSNR Evaluations
# ============================================================================
# This script submits SLURM jobs to evaluate GSNR for both RLTT and GRPO
# across all benchmarks.
#
# Usage:
#   ./run_all_gsnr_evals.sh
#
#   # With custom checkpoint:
#   CHECKPOINT=step_100 ./run_all_gsnr_evals.sh
#
#   # With custom experiment directories:
#   RLTT_EXPERIMENT_DIR=/path/to/rltt_output/XXXXXX \
#   GRPO_EXPERIMENT_DIR=/path/to/grpo_output/XXXXXX \
#   ./run_all_gsnr_evals.sh

set -e

# Configuration
CHECKPOINT=${CHECKPOINT:-"step_140"}
RLTT_EXPERIMENT_DIR=${RLTT_EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/3374111"}
GRPO_EXPERIMENT_DIR=${GRPO_EXPERIMENT_DIR:-"/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/3374119"}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-8}
NUM_PROMPTS=${NUM_PROMPTS:-""}  # Empty means all prompts

RLTT_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments"
GRPO_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments"

BENCHMARKS=("math500" "gsm8k" "aime24" "beyondaime")

echo "=========================================="
echo "GSNR Evaluation Job Submission"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "RLTT experiment: $RLTT_EXPERIMENT_DIR"
echo "GRPO experiment: $GRPO_EXPERIMENT_DIR"
echo "Num rollouts: $NUM_ROLLOUTS"
echo "Num prompts: ${NUM_PROMPTS:-all}"
echo ""

# Track submitted jobs
RLTT_JOBS=()
GRPO_JOBS=()

# Submit RLTT jobs
echo "Submitting RLTT GSNR evaluation jobs..."
cd "$RLTT_DIR"

for bench in "${BENCHMARKS[@]}"; do
    echo "  Submitting RLTT ${bench}..."

    EXPORT_ARGS="ALL,BENCHMARK=${bench},CHECKPOINT=${CHECKPOINT},EXPERIMENT_DIR=${RLTT_EXPERIMENT_DIR},NUM_ROLLOUTS=${NUM_ROLLOUTS}"
    if [ -n "$NUM_PROMPTS" ]; then
        EXPORT_ARGS="${EXPORT_ARGS},NUM_PROMPTS=${NUM_PROMPTS}"
    fi

    JOB_ID=$(sbatch --export="$EXPORT_ARGS" --parsable run_gsnr_eval.slurm)
    RLTT_JOBS+=("${bench}:${JOB_ID}")
    echo "    Submitted job $JOB_ID for RLTT ${bench}"
done

# Submit GRPO jobs
echo ""
echo "Submitting GRPO GSNR evaluation jobs..."
cd "$GRPO_DIR"

for bench in "${BENCHMARKS[@]}"; do
    echo "  Submitting GRPO ${bench}..."

    EXPORT_ARGS="ALL,BENCHMARK=${bench},CHECKPOINT=${CHECKPOINT},EXPERIMENT_DIR=${GRPO_EXPERIMENT_DIR},NUM_ROLLOUTS=${NUM_ROLLOUTS}"
    if [ -n "$NUM_PROMPTS" ]; then
        EXPORT_ARGS="${EXPORT_ARGS},NUM_PROMPTS=${NUM_PROMPTS}"
    fi

    JOB_ID=$(sbatch --export="$EXPORT_ARGS" --parsable run_gsnr_eval.slurm)
    GRPO_JOBS+=("${bench}:${JOB_ID}")
    echo "    Submitted job $JOB_ID for GRPO ${bench}"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "RLTT jobs:"
for job in "${RLTT_JOBS[@]}"; do
    echo "  $job"
done
echo ""
echo "GRPO jobs:"
for job in "${GRPO_JOBS[@]}"; do
    echo "  $job"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs complete, aggregate results with:"
echo "  cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results"
echo "  python aggregate_gsnr_results.py"
