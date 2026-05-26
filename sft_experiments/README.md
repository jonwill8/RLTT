# SFT Experiments

This directory contains the supervised fine-tuning baseline used as a reference
and, optionally, as initialization for RLTT/GRPO runs.

## Main Entry Points

| File | Purpose |
| --- | --- |
| `prepare_data.py` | Convert raw training data into SFT-ready format. |
| `sft_train.py` | Main SFT trainer. |
| `simple_sft_trainer.py` | Single-GPU fallback trainer. |
| `sft_config.py` | SFT configuration defaults. |
| `evaluate_sft_checkpoint.py` | Math checkpoint evaluation. |
| `eval_sft_mcqa.py` | MCQA checkpoint evaluation. |
| `eval_sft_code.py` | Code benchmark evaluation. |
| `evaluate_sft_competitions.py` | AIME26/HMMT-style competition evaluation. |
| `evaluate_per_loop.py` | Per-loop evaluation for SFT checkpoints. |

## Slurm Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/sft_experiments
mkdir -p logs
```

Train SFT:

```bash
sbatch run_sft.slurm
```

Evaluate a checkpoint:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../sft_ouro_math_output/<checkpoint> run_sft_eval.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../sft_ouro_math_output/<checkpoint> run_sft_eval_gsm8k.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../sft_ouro_math_output/<checkpoint> run_sft_eval_arc_c.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../sft_ouro_math_output/<checkpoint> run_sft_eval_mbpp.slurm
```

Competition evaluation:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../sft_ouro_math_output/<checkpoint>,BENCHMARK=aime26 run_sft_eval_competitions.slurm
```

## Outputs

Training outputs default to `sft_ouro_math_output/`. Logs are in `logs/`, and
some evaluation wrappers write summaries under `eval_outputs/`.
