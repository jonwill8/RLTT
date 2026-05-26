# GRPO Experiments

This directory mirrors the RLTT workflow for GRPO baselines. Use it when you
need matched training, checkpoint evaluation, repeated evaluation, pass@k,
GSNR, per-loop, or rollout-comparison runs for GRPO.

## Main Entry Points

| File | Purpose |
| --- | --- |
| `grpo_train.py` | Main GRPO trainer. |
| `simple_trainer.py` | Single-GPU fallback trainer. |
| `config.py`, `data_utils.py` | Training configuration and dataset helpers. |
| `evaluate_checkpoint.py` | Math checkpoint evaluation. |
| `eval_grpo_mcqa.py` | MCQA checkpoint evaluation. |
| `eval_grpo_code.py` | Code benchmark evaluation. |
| `evaluate_checkpoint_competitions.py` | Competition benchmark evaluation. |
| `repeated_eval_*.py` | Repeated eval scripts for statistical testing. |
| `pass_at_k_eval.py`, `gsnr_eval.py`, `evaluate_per_loop.py` | Analysis scripts. |
| `verl_grpo/` | Custom verl/GRPO support code. |

## Slurm Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
mkdir -p logs
```

Train GRPO:

```bash
sbatch run_grpo.slurm
```

Evaluate one checkpoint:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/<run_id>/global_step_140 run_eval_single_ckpt_grpo.slurm
```

Evaluate MCQA/code benchmarks:

```bash
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_grpo_eval_arc_c.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_grpo_eval_gpqa.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_grpo_eval_mbpp.slurm
```

Repeated evaluation:

```bash
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval.slurm
sbatch --export=ALL,BENCHMARK=arc_c,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval_non-math.slurm
sbatch --export=ALL,BENCHMARK=mbpp,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval-code.slurm
```

## Outputs

Training writes to `grpo_output/<slurm_job_id>/`. Logs are in `logs/`.
Repeated-evaluation outputs are written to the top-level
`repeated_eval_results*` directories so they can be compared directly with RLTT.
