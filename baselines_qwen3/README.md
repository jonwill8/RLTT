# Qwen3 Baselines

This directory evaluates Qwen3 baselines and contains optional Qwen3 GRPO
training/repeated-evaluation scripts. It is useful for non-Ouro comparisons
using the same benchmark and statistical-evaluation structure as RLTT/GRPO.

## Entry Points

| File | Purpose |
| --- | --- |
| `evaluate_baseline.py` | Math benchmark evaluation. |
| `evaluate_baseline_mcqa.py` | ARC-Challenge, GPQA, and MMLU-STEM evaluation. |
| `eval_qwen3_code.py` | HumanEval and MBPP evaluation. |
| `evaluate_competitions.py` | AIME26/HMMT-style competition evaluation. |
| `grpo_train.py`, `run_grpo.slurm` | Qwen3 GRPO training. |
| `repeated_eval_*.py` | Repeated evaluation scripts. |
| `run_eval_*_rl_trained.slurm` | Evaluation launchers for RL-trained Qwen3 checkpoints. |
| `run_all_evals.sh` | Convenience launcher for the baseline suite. |

## Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_qwen3
mkdir -p logs
```

Evaluate base Qwen3:

```bash
sbatch run_eval_math500.slurm
sbatch run_eval_gsm8k.slurm
sbatch run_eval_aime24.slurm
sbatch run_eval_beyondaime.slurm
sbatch run_eval_arc_c.slurm
sbatch run_eval_gpqa.slurm
sbatch run_eval_mmlu_stem.slurm
sbatch run_eval_humaneval.slurm
sbatch run_eval_mbpp.slurm
```

Run repeated evaluation:

```bash
sbatch --export=ALL,BENCHMARK=math500 run_repeated_eval.slurm
sbatch --export=ALL,BENCHMARK=gpqa run_repeated_eval_non-math.slurm
sbatch --export=ALL,BENCHMARK=mbpp run_repeated_eval-code.slurm
```

Train or evaluate an RL-trained Qwen3 checkpoint:

```bash
sbatch run_grpo.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../grpo_output/<run_id>,CHECKPOINT=step_140 run_eval_math500_rl_trained.slurm
```

## Outputs

Logs are written to `logs/`. Baseline evaluation outputs default to
`eval_output/`; Qwen3 GRPO training outputs default to `grpo_output/`.
