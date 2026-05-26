# Repeated Evaluation Results: Math

Stores repeated evaluation outputs for math benchmarks and provides paired
t-test analysis between GRPO and RLTT.

## Producing Results

RLTT:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval.slurm
```

GRPO:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140,NUM_RUNS=10 run_repeated_eval.slurm
```

Supported math `BENCHMARK` values include `math500`, `gsm8k`, `aime24`,
`aime26`, and `beyondaime`.

## T-Test Analysis

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results
python run_ttest_analysis.py --benchmark math500
python run_ttest_analysis.py --all
```

The script auto-detects the latest matching RLTT and GRPO CSVs unless explicit
CSV paths are supplied.
