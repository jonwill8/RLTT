# Rollout Comparisons

Stores generated rollout comparison artifacts for qualitative RLTT/GRPO
analysis.

## Producing Rollouts

Use the method-specific rollout launchers:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=gsm8k,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_generate_rollouts.slurm
```

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=gsm8k,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_generate_rollouts.slurm
```

## Helper Script

`create_both_correct.py` filters/organizes rollout cases where both compared
methods answer correctly. Use it after the relevant rollout JSONL files exist.

## Contents

Subdirectories are typically named by method and evaluation setting, for example
`rltt_eval_math500_prompt1024_zeroshot_tokens2048/`.
