# Per-Loop Results

Stores per-loop evaluation outputs for RLTT, GRPO, and SFT checkpoints. These
runs measure how accuracy changes when the model is evaluated after a selected
number of recurrent loops.

## Producing Results

RLTT:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,NUM_LOOPS=1 run_per_loop_eval.slurm
```

GRPO:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140,NUM_LOOPS=1 run_per_loop_eval.slurm
```

SFT:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments
sbatch --export=ALL,BENCHMARK=math500,CHECKPOINT_PATH=/scratch/.../checkpoint,NUM_LOOPS=1 run_per_loop_eval.slurm
```

## Aggregation

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/per_loop_results
python aggregate_per_loop_results.py
```

Outputs are organized by method subdirectories such as `rltt/`, `grpo/`, and
`sft/`.
