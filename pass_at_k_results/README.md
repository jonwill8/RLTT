# pass@k Results

Stores pass@k evaluation outputs and aggregation utilities for RLTT and GRPO.

## Producing Results

Run from the method-specific experiment directory:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,K=8 run_pass_at_k_eval.slurm
```

GRPO uses the matching wrapper in `grpo_experiments/`:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140,K=8 run_pass_at_k_eval.slurm
```

## Aggregation

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/pass_at_k_results
python aggregate_pass_at_k.py
python aggregate_pass_at_k.py --benchmark math500
python aggregate_pass_at_k.py --k 8
```

The aggregator scans JSON outputs in this directory and writes CSV/JSON summary
files comparing methods by benchmark and `k`.
