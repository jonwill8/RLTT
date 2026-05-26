# GSNR (Gradient Signal-to-Noise Ratio) Evaluation Results

This directory contains GSNR evaluation results for RLTT and GRPO experiments, implementing the metric described in the GSNR paper.

## Overview

GSNR measures the quality of the learning signal produced by the RLTT/GRPO objective by computing:

1. For each prompt `p`, sample R=8 independent responses using the current policy
2. Each response is graded to compute advantages (binary reward: correct/incorrect)
3. For each rollout, compute gradient of loss w.r.t. latent-thought logits
4. Compute mean gradient `μ_p` and noise variance across rollouts
5. `GSNR_p = ||μ_p||²₂ / (noise_p + ε)`
6. Overall `GSNR = mean(log(GSNR_p + ε))` across prompts

Higher GSNR indicates a more consistent learning signal across rollouts.

## Directory Structure

```
gsnr_results/
├── README.md                    # This file
├── aggregate_gsnr_results.py    # Aggregation script
├── run_all_gsnr_evals.sh        # Convenience script to run all evals
├── grpo/                        # GRPO results
│   └── gsnr_{benchmark}_{exp_id}_{checkpoint}_{timestamp}/
│       ├── gsnr_results.json    # Full results with per-prompt data
│       ├── gsnr_summary.json    # Summary metrics for aggregation
│       ├── gsnr_summary.txt     # Human-readable summary
│       └── merged_model/        # Merged model checkpoint
├── rltt/                        # RLTT results
│   └── gsnr_{benchmark}_{exp_id}_{checkpoint}_{timestamp}/
│       └── ...
├── gsnr_all_results_latest.csv       # Latest aggregated results
├── gsnr_comparison_latest.csv        # Latest RLTT vs GRPO comparison
├── gsnr_aggregated_latest.json       # Latest JSON with all data
└── gsnr_summary_report_latest.txt    # Latest summary report
```

## Usage

### Running Evaluations

**Run all evaluations:**
```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results
./run_all_gsnr_evals.sh
```

**Run with custom checkpoint:**
```bash
CHECKPOINT=step_100 ./run_all_gsnr_evals.sh
```

**Run individual benchmark (RLTT):**
```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
sbatch --export=ALL,BENCHMARK=math500 run_gsnr_eval.slurm
```

**Run individual benchmark (GRPO):**
```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments
sbatch --export=ALL,BENCHMARK=math500 run_gsnr_eval.slurm
```

### Aggregating Results

After evaluations complete:
```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results
python aggregate_gsnr_results.py
```

This generates:
- `gsnr_all_results_{timestamp}.csv` - All results in CSV format
- `gsnr_comparison_{timestamp}.csv` - RLTT vs GRPO comparison
- `gsnr_aggregated_{timestamp}.json` - Full JSON data
- `gsnr_summary_report_{timestamp}.txt` - Human-readable report

## Evaluation Scripts

### RLTT: `rltt_experiments/gsnr_eval.py`

```bash
python gsnr_eval.py \
    --experiment_dir /path/to/rltt_output/3374111 \
    --checkpoint step_140 \
    --benchmark math500 \
    --num_rollouts 8 \
    --output_dir /path/to/gsnr_results/rltt
```

### GRPO: `grpo_experiments/gsnr_eval.py`

```bash
python gsnr_eval.py \
    --experiment_dir /path/to/grpo_output/3374119 \
    --checkpoint step_140 \
    --benchmark math500 \
    --num_rollouts 8 \
    --output_dir /path/to/gsnr_results/grpo
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--experiment_dir` | Required | Path to experiment output directory |
| `--checkpoint` | Required | Checkpoint to evaluate (e.g., step_140) |
| `--benchmark` | Required | Benchmark: math500, gsm8k, aime24, beyondaime |
| `--num_rollouts` | 8 | Number of rollouts per prompt (R in paper) |
| `--num_prompts` | all | Number of prompts to evaluate |
| `--output_dir` | auto | Output directory for results |

## Benchmarks

| Benchmark | Dataset Size | Max Tokens |
|-----------|-------------|------------|
| math500 | 500 problems | 2048 |
| gsm8k | 1319 problems | 512 |
| aime24 | 30 problems | 3072 |
| beyondaime | 110 problems | 3072 |

## Interpreting Results

**Overall GSNR (mean log)**: Primary metric - higher is better
- Positive values indicate strong learning signal
- Negative values indicate noisy/weak learning signal
- Compare RLTT vs GRPO on same benchmark

**Std log(GSNR)**: Variance in GSNR across prompts
- Lower variance suggests more consistent signal

**Mean/Std raw GSNR**: Non-log-transformed values for reference

**Valid prompts**: Number of prompts with computable GSNR
- Some prompts may have all same rewards (no gradient variance)

## Expected Results

Based on the paper:
- RLTT should show higher GSNR than GRPO due to denser credit assignment
- Both methods should show higher GSNR on easier benchmarks (GSM8K > MATH-500)
- AIME/BeyondAIME may show lower GSNR due to difficulty

## Notes

- Temperature 0.7 is used for sampling to ensure diversity in rollouts
- Evaluation uses HuggingFace generation (not vLLM) for gradient computation
- GPU memory: ~64GB recommended for full model with gradient computation
- Time estimate: ~4 hours per benchmark depending on size
