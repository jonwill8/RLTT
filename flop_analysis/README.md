# FLOPs Profiling (Established Library)

This directory profiles rollout FLOPs using:

- Hugging Face `generate()`
- DeepSpeed FLOPs profiler (`deepspeed.profiling.flops_profiler`)

## Files

- `profile_generation_flops.py`: shared profiler script
- `run_flops_ouro.slurm`: Ouro profiling (supports RLTT/GRPO checkpoint merge or direct model path)
- `run_flops_qwen3.slurm`: Qwen3 profiling
- `run_flops_deepseek.slurm`: DeepSeek-R1-Distill-Qwen profiling

## Install requirement

DeepSpeed must be installed in your job environment:

```bash
pip install deepspeed
```

## Typical submissions

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../rltt_output/.../global_step_140,MAX_SAMPLES=16 RLTT/flop_analysis/run_flops_ouro.slurm
```

```bash
sbatch --export=ALL,MODEL_SIZE=1.7B,MAX_SAMPLES=16 RLTT/flop_analysis/run_flops_qwen3.slurm
```

```bash
sbatch --export=ALL,MODEL_SIZE=7B,MAX_SAMPLES=16 RLTT/flop_analysis/run_flops_deepseek.slurm
```

Outputs are written under each run directory:

- `flop_profile_summary.json`
- `flop_profile_details.jsonl`
