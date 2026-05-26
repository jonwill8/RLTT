# RLTT verl Extensions

Custom verl integration code for RLTT training.

## Files

| File | Purpose |
| --- | --- |
| `rltt_algos.py` | RLTT objective and loop-weighted log-probability logic. |
| `rltt_actor.py` | RLTT actor implementation used during rollout/update. |
| `rltt_fsdp_workers.py` | FSDP worker classes and hooks for multi-GPU RLTT training. |
| `__init__.py` | Package exports. |

## Usage

These modules are imported by `../rltt_train.py` and the multi-GPU training
path launched by `../run_rltt.slurm`. They are not usually called directly.

Changes here affect the RL objective and distributed worker behavior, so verify
with a short Slurm run before launching a full experiment:

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,MAX_STEPS=2,SAVE_STEPS=1,NUM_PROMPTS_PER_BATCH=4,NUM_GENERATIONS=2 run_rltt.slurm
```
