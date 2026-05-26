# DeepSeek-R1 Baselines

This directory evaluates DeepSeek-R1-Distill baselines on the same benchmark
families used for Ouro, RLTT, GRPO, and SFT comparisons.

## Entry Points

| File | Purpose |
| --- | --- |
| `evaluate_baseline.py` | Math benchmark evaluation. |
| `evaluate_baseline_mcqa.py` | ARC-Challenge, GPQA, and MMLU-STEM evaluation. |
| `eval_deepseek_r1_code.py` | HumanEval and MBPP evaluation. |
| `evaluate_competitions.py` | Competition benchmark evaluation. |
| `run_eval_*.slurm` | Cluster launchers for individual benchmarks. |
| `run_all_evals.sh` | Convenience launcher for the baseline suite. |

## Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baselines_deepseek_r1
mkdir -p logs
```

Submit individual benchmark jobs:

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

Competition evaluation:

```bash
sbatch --export=ALL,BENCHMARK=aime26 run_eval_competitions.slurm
```

Override `MODEL_PATH`, `MODEL_SIZE`, `OUTPUT_DIR`, or token limits with
`sbatch --export=ALL,...` when comparing a different DeepSeek checkpoint.

## Outputs

Logs are written to `logs/`. Evaluation artifacts default to `eval_output/` or
the `OUTPUT_DIR` supplied to the Slurm job.
