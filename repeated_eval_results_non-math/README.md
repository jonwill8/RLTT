# Repeated Evaluation Results: Non-Math

Stores repeated evaluation outputs for MCQA benchmarks and provides paired
t-test analysis between GRPO and RLTT.

## Producing Results

RLTT:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=gpqa,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval_non-math.slurm
```

GRPO:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=gpqa,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval_non-math.slurm
```

Supported `BENCHMARK` values are `arc_c`, `mmlu_st`, and `gpqa`.

## T-Test Analysis

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results_non-math
python run_ttest_analysis.py --benchmark gpqa
python run_ttest_analysis.py --all
```
