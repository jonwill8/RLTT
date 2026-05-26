# Last-token-across-prompts CKA (AIME26)

This directory contains a single-checkpoint CKA analysis flow for RLTT/GRPO checkpoints on AIME26 (30 prompts), following `cka_analysis_idea.pdf`.

## Files

- `last_token_cka_competitions.py`
  - Loads AIME26 prompts.
  - Extracts **last prompt token hidden state** for each UT loop.
  - Builds `Z(i)` matrices (prompts x hidden_dim), computes:
    - loop-loop CKA heatmap
    - convergence-to-final CKA curve
- `run_last_token_cka_single_ckpt.slurm`
  - SLURM wrapper modeled after existing single-checkpoint competition evaluation jobs.

## Submit (RLTT example)

```bash
sbatch --export=ALL,METHOD=rltt,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/<run_id>/global_step_140 run_last_token_cka_single_ckpt.slurm
```

## Submit (GRPO example)

```bash
sbatch --export=ALL,METHOD=grpo,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/grpo_experiments/grpo_output/<run_id>/global_step_140 run_last_token_cka_single_ckpt.slurm
```

## Main outputs

- `run_config.json` - full run configuration (CLI args + resolved model/output paths + runtime metadata)
- `cka_results.json` - CKA heatmap values + convergence curve
- `cka_heatmap.png` - 4x4 (or `total_ut_steps` x `total_ut_steps`) CKA heatmap
- `convergence_to_final_curve.png` - `CKA(Z(i), Z(final))` curve
- `last_token_loop_embeddings.npz` - saved loop-wise embeddings
- `prompt_metadata.jsonl` - prompt/gold metadata

By default, outputs are written inside this directory under `runs/` for easy local visualization.
