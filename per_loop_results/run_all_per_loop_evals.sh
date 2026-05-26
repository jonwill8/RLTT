#!/bin/bash
# ============================================================================
# Run All Per-Loop Evaluations
# ============================================================================
# This script submits SLURM jobs for all combinations of:
#   - Methods: GRPO, RLTT
#   - Loop counts: 1, 2, 3
#   - Benchmarks: math500, gsm8k, aime24, beyondaime (configurable)
#
# Usage:
#   # Run all evaluations for MATH-500:
#   ./run_all_per_loop_evals.sh math500
#
#   # Run all evaluations for GSM8K:
#   ./run_all_per_loop_evals.sh gsm8k
#
#   # Run all evaluations for AIME24:
#   ./run_all_per_loop_evals.sh aime24
#
#   # Run all evaluations for BeyondAIME:
#   ./run_all_per_loop_evals.sh beyondaime
#
#   # Run all evaluations for all benchmarks:
#   ./run_all_per_loop_evals.sh all
#
#   # Custom checkpoint:
#   CHECKPOINT=step_100 ./run_all_per_loop_evals.sh math500
# ============================================================================

BENCHMARK=${1:-"math500"}
CHECKPOINT=${CHECKPOINT:-"step_140"}

# Directories
RLTT_DIR="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT"
GRPO_EXPERIMENT_DIR="${GRPO_EXPERIMENT_DIR:-$RLTT_DIR/grpo_experiments/grpo_output/3374119}"
RLTT_EXPERIMENT_DIR="${RLTT_EXPERIMENT_DIR:-$RLTT_DIR/rltt_experiments/rltt_output/3374111}"
OUTPUT_DIR="${OUTPUT_DIR:-$RLTT_DIR/per_loop_results}"

echo "=========================================="
echo "Per-Loop Evaluation Submission Script"
echo "=========================================="
echo "Benchmark: $BENCHMARK"
echo "Checkpoint: $CHECKPOINT"
echo "GRPO experiment: $GRPO_EXPERIMENT_DIR"
echo "RLTT experiment: $RLTT_EXPERIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Determine benchmarks to run
if [ "$BENCHMARK" == "all" ]; then
    BENCHMARKS="math500 gsm8k aime24 beyondaime"
else
    BENCHMARKS="$BENCHMARK"
fi

# Track submitted jobs
SUBMITTED_JOBS=""

for bench in $BENCHMARKS; do
    echo "Submitting jobs for benchmark: $bench"
    echo "-" * 40

    for loops in 1 2 3; do
        # Submit GRPO job
        echo "  Submitting GRPO loops=$loops $bench..."
        GRPO_JOB=$(sbatch --parsable \
            --export=ALL,EXPERIMENT_DIR=$GRPO_EXPERIMENT_DIR,CHECKPOINT=$CHECKPOINT,NUM_LOOPS=$loops,BENCHMARK=$bench,OUTPUT_DIR=$OUTPUT_DIR \
            $RLTT_DIR/grpo_experiments/run_per_loop_eval.slurm)
        echo "    GRPO job ID: $GRPO_JOB"
        SUBMITTED_JOBS="$SUBMITTED_JOBS $GRPO_JOB"

        # Submit RLTT job
        echo "  Submitting RLTT loops=$loops $bench..."
        RLTT_JOB=$(sbatch --parsable \
            --export=ALL,EXPERIMENT_DIR=$RLTT_EXPERIMENT_DIR,CHECKPOINT=$CHECKPOINT,NUM_LOOPS=$loops,BENCHMARK=$bench,OUTPUT_DIR=$OUTPUT_DIR \
            $RLTT_DIR/rltt_experiments/run_per_loop_eval.slurm)
        echo "    RLTT job ID: $RLTT_JOB"
        SUBMITTED_JOBS="$SUBMITTED_JOBS $RLTT_JOB"
    done
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Submitted jobs:$SUBMITTED_JOBS"
echo ""
echo "To monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, run aggregation:"
echo "  python $OUTPUT_DIR/aggregate_per_loop_results.py --benchmark $BENCHMARK"
echo ""
echo "Or for all benchmarks:"
echo "  python $OUTPUT_DIR/aggregate_per_loop_results.py --all"
