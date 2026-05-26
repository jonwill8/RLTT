# Repeated Evaluation Results: Code

Stores repeated evaluation outputs for HumanEval and MBPP and provides paired
t-test analysis between GRPO and RLTT.

## Producing Results

RLTT:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=mbpp,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval-code.slurm
```

GRPO:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=mbpp,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval-code.slurm
```

Supported `BENCHMARK` values are `humaneval` and `mbpp`.

## T-Test Analysis

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results_code
python run_ttest_analysis.py --benchmark mbpp
python run_ttest_analysis.py --all
```
