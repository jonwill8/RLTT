# Baseline Evaluation

This directory evaluates base Ouro models before RLTT, GRPO, or SFT training.
Use it to establish reference accuracy on math, MCQA, code, and competition
benchmarks.

## Main Entry Points

| File | Purpose |
| --- | --- |
| `eval_ouro_math500.py` | MATH-500, GSM8K, AIME, and BeyondAIME-style math evaluation. |
| `eval_ouro_mcqa.py` | ARC-Challenge, GPQA, and MMLU-STEM evaluation. |
| `eval_ouro_code.py` | HumanEval and MBPP evaluation. |
| `evaluate_competitions.py` | AIME26/HMMT-style competition evaluation. |
| `resolve_ouro_model.sh` | Shared model-size/path resolver for Slurm scripts. |

## Slurm Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/baseline_eval
mkdir -p logs
```

Default baseline model size is usually `2.6B`. Override `MODEL_SIZE` or
`MODEL_PATH` when needed:

```bash
sbatch --export=ALL,MODEL_SIZE=1.4B run_eval.slurm
sbatch --export=ALL,MODEL_PATH=/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking run_eval.slurm
```

Benchmark-specific launchers:

```bash
sbatch run_eval.slurm
sbatch run_eval_gsm8k.slurm
sbatch run_eval_aime24.slurm
sbatch run_eval_aime26.slurm
sbatch run_eval_beyondaime.slurm
sbatch run_eval_arc_c.slurm
sbatch run_eval_gpqa.slurm
sbatch run_eval_mmlu_stem.slurm
sbatch run_eval_humaneval.slurm
sbatch run_eval_mbpp.slurm
```

Competition launcher:

```bash
sbatch --export=ALL,BENCHMARK=aime26 run_eval_competitions.slurm
```

## Outputs

Slurm logs are written to `logs/`. Evaluation outputs are written under
`eval_output/` or the `OUTPUT_DIR` passed to the launcher.
