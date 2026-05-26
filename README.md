# RLTT: Reinforcement Learning for Latent Thought Trajectories

This repository contains training, evaluation, and analysis code for RLTT on
looped Ouro-style language models. RLTT assigns reward credit across latent
thought loops instead of using only the final loop, so most day-to-day work
happens in `rltt_experiments/`.

The repo also includes GRPO, SFT, baseline model evaluations, repeated
statistical evaluation, pass@k, GSNR, per-loop, CKA, and FLOPs analysis flows.

## Start Here

Use the Slurm wrappers from `rltt_experiments/` on the cluster. They set the
expected conda environment, CUDA module, offline Hugging Face caches, model
paths, output directories, and checkpoint merge steps.

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
mkdir -p logs
```

Run RLTT training:

```bash
sbatch run_rltt.slurm
```

Run a tuned RLTT job:

```bash
sbatch --export=ALL,LOOP_WEIGHTING=exit_pdf,USE_RLTT_MODEL=true,BETA=0.001,NUM_PROMPTS_PER_BATCH=32 run_rltt.slurm
```

Evaluate a saved RLTT checkpoint on MATH-500:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/<run_id>/global_step_140 run_eval_single_ckpt_rltt.slurm
```

Evaluate the same checkpoint on other benchmark families:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_gsm8k.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_aime24.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140,BENCHMARK=aime26 run_eval_single_ckpt_competitions.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_arc_c.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_gpqa.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_mbpp.slurm
```

See `rltt_experiments/README.md` for the full RLTT training, evaluation, and
analysis runbook.

## Cluster Assumptions

The Slurm scripts assume:

- Conda environment: `ouro_vllm`
- CUDA module: `cudatoolkit/12.8`
- Offline model/data cache root: `/scratch/gpfs/OLGARUS/jw4199/model_weights_path`
- Dataset root: `/scratch/gpfs/OLGARUS/jw4199/datasets`
- Main repo path: `/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT`

If you are running outside that environment, override script variables with
`sbatch --export=ALL,NAME=value,...` or call the Python entry points directly
after setting equivalent paths.

## Main Workflow

1. Train RLTT from `rltt_experiments/run_rltt.slurm`.
2. Check outputs under `rltt_experiments/rltt_output/<slurm_job_id>/`.
3. Evaluate single checkpoints with `run_eval_single_ckpt_*.slurm`.
4. Run repeated evaluations with `run_repeated_eval*.slurm` for statistical
   comparison against GRPO.
5. Use `pass_at_k_results/`, `gsnr_results/`, `per_loop_results/`,
   `cka_convergence_analysis/`, and `flop_analysis/` for analysis artifacts.

## Repository Map

| Path | Purpose |
| --- | --- |
| `rltt_experiments/` | Primary RLTT training, checkpoint evaluation, repeated evaluation, and analysis launchers. Start here. |
| `grpo_experiments/` | GRPO training and evaluation baseline mirroring the RLTT workflow. |
| `sft_experiments/` | Supervised fine-tuning baseline and SFT checkpoint evaluation. |
| `baseline_eval/` | Base Ouro model evaluations before RL training. |
| `baselines_deepseek_r1/` | DeepSeek-R1-Distill baseline evaluations. |
| `baselines_qwen3/` | Qwen3 baseline, repeated evaluation, and optional Qwen3 GRPO workflows. |
| `math_utils/` | Math answer parsing, prompting, and reward helpers shared by training/eval scripts. |
| `non_math_utils/` | Multiple-choice answer parsing helpers for ARC-C, GPQA, and MMLU-STEM. |
| `non_math_code_utils/` | Code benchmark execution and checking helpers for HumanEval/MBPP. |
| `non_math_data_prep/` | MCQA dataset preparation scripts. |
| `non_math_code_data_prep/` | HumanEval/MBPP dataset preparation scripts. |
| `repeated_eval_results/` | Math repeated-evaluation outputs and paired t-test utilities. |
| `repeated_eval_results_non-math/` | Non-math repeated-evaluation outputs and paired t-test utilities. |
| `repeated_eval_results_code/` | Code repeated-evaluation outputs and paired t-test utilities. |
| `pass_at_k_results/` | pass@k result files and aggregation utilities. |
| `per_loop_results/` | Per-loop evaluation outputs and aggregation utilities. |
| `gsnr_results/` | Gradient signal-to-noise ratio results and aggregation utilities. |
| `cka_convergence_analysis/` | Last-token-across-prompts CKA analysis for RLTT/GRPO checkpoints. |
| `flop_analysis/` | FLOPs profiling wrappers for Ouro, Qwen3, and DeepSeek baselines. |
| `rollout_comparisons/` | Rollout comparison artifacts and helper scripts. |
| `reference_pdfs/` | Local reference PDFs used while developing analyses and writeups. |

Each non-hidden top-level directory has README coverage with local entry points,
expected inputs, and outputs.

## Benchmarks

The active scripts cover:

- Math: MATH-500, GSM8K, AIME24, AIME26, BeyondAIME, HMMT25
- MCQA: ARC-Challenge, GPQA, MMLU-STEM
- Code: HumanEval, MBPP

## RLTT Loop Weighting

`run_rltt.slurm` exposes the main RLTT objective choices through environment
variables:

| Variable | Values | Notes |
| --- | --- | --- |
| `LOOP_WEIGHTING` | `uniform`, `progressive`, `exit_pdf`, `learned` | Default in the Slurm wrapper is `exit_pdf`. |
| `PROGRESSIVE_ALPHA` | float | Used only with `LOOP_WEIGHTING=progressive`. |
| `USE_RLTT_MODEL` | `true`, `false` | Required for `exit_pdf`; uses the RLTT-modified Ouro model. |
| `USE_LORA` | `true`, `false` | Current Slurm default is full-parameter fine-tuning with `false`. |
| `BETA` | float | KL coefficient. |
| `NUM_GENERATIONS` | integer | Rollouts per prompt. |
| `NUM_PROMPTS_PER_BATCH` | integer | Unique prompts per optimization step. |

## Outputs

Training outputs are written under:

```text
rltt_experiments/rltt_output/<slurm_job_id>/
```

Typical checkpoints look like:

```text
rltt_experiments/rltt_output/<slurm_job_id>/global_step_140/actor/
```

Evaluation wrappers either write under the checkpoint directory or under
workflow-specific result directories such as `repeated_eval_results/`,
`pass_at_k_results/`, `gsnr_results/`, and `per_loop_results/`.

## Notes

- The Slurm wrappers are the preferred interface on the cluster.
- Direct Python commands are useful for local debugging, but the wrappers handle
  FSDP checkpoint merge and environment setup.
- Generated logs, merged checkpoints, and large evaluation outputs should remain
  out of Git unless they are intentionally curated summaries.

## Citation

If you use this code, please cite the RLTT work.
