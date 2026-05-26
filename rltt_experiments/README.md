# RLTT Experiments

This is the primary working directory for RLTT training and evaluation. Use the
Slurm scripts here as the main interface on the cluster; they set the conda
environment, CUDA module, cache paths, default model/data paths, and FSDP merge
steps used by the Python scripts.

## Quick Start

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments
mkdir -p logs
```

Train RLTT:

```bash
sbatch run_rltt.slurm
```

Evaluate a checkpoint on MATH-500:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/<run_id>/global_step_140 run_eval_single_ckpt_rltt.slurm
```

Run repeated evaluation for a paper-style statistical comparison:

```bash
sbatch --export=ALL,BENCHMARK=aime26,EXPERIMENT_DIR=/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/rltt_experiments/rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval.slurm
```

## Directory Contents

| Path | Purpose |
| --- | --- |
| `rltt_train.py` | Main RLTT trainer. Uses simple single-GPU code when appropriate and verl/FSDP for multi-GPU runs. |
| `simple_trainer.py` | Single-GPU fallback trainer. |
| `config.py` | Dataclass configuration defaults used by the trainer. |
| `data_utils.py` | Dataset loading and preprocessing helpers. |
| `verl_rltt/` | Custom verl RLTT actor, algorithms, and FSDP worker hooks. |
| `run_rltt.slurm` | Main RLTT training launcher. |
| `run_eval_single_ckpt_*.slurm` | Single-checkpoint math/competition evaluation launchers. |
| `run_rltt_eval_*.slurm` | MCQA and code benchmark evaluation launchers. |
| `run_repeated_eval*.slurm` | Repeated evaluation launchers for math, non-math, and code benchmarks. |
| `run_pass_at_k_eval.slurm` | pass@k launcher. |
| `run_gsnr_eval.slurm` | GSNR launcher. |
| `run_per_loop_eval.slurm` | Per-loop evaluation launcher. |
| `run_generate_rollouts.slurm` | Rollout generation launcher for qualitative comparisons. |
| `logs/` | Slurm stdout/stderr logs. |
| `rltt_output/` | Training outputs and checkpoints. |
| `eval_outputs/` | Evaluation outputs for some non-math/code wrappers. |

## Environment

The Slurm scripts assume:

```bash
source /home/jw4199/miniconda3/etc/profile.d/conda.sh
conda activate ouro_vllm
module load cudatoolkit/12.8
```

They also set offline Hugging Face mode and point caches at:

```text
/scratch/gpfs/OLGARUS/jw4199/model_weights_path
```

## Training

Primary launcher:

```bash
sbatch run_rltt.slurm
```

Common overrides:

```bash
sbatch --export=ALL,USE_LORA=true run_rltt.slurm
sbatch --export=ALL,USE_LORA=false run_rltt.slurm
sbatch --export=ALL,LOOP_WEIGHTING=uniform run_rltt.slurm
sbatch --export=ALL,LOOP_WEIGHTING=progressive,PROGRESSIVE_ALPHA=2.0 run_rltt.slurm
sbatch --export=ALL,LOOP_WEIGHTING=exit_pdf,USE_RLTT_MODEL=true run_rltt.slurm
sbatch --export=ALL,LOOP_WEIGHTING=learned,LEARNED_WEIGHTS_INIT=progressive run_rltt.slurm
```

Short debug run:

```bash
sbatch --export=ALL,MAX_STEPS=2,SAVE_STEPS=1,NUM_PROMPTS_PER_BATCH=4,NUM_GENERATIONS=2 run_rltt.slurm
```

Important `run_rltt.slurm` defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking` | Base model path used by the Slurm wrapper. |
| `RLTT_MODEL_PATH` | `/scratch/gpfs/OLGARUS/jw4199/model_weights_path/Ouro-1.4B-Thinking-RLTT` | RLTT-modified Ouro model exposing loop states and exit probabilities. |
| `TRAIN_FILE` | `/scratch/gpfs/OLGARUS/jw4199/datasets/MATH/math_train.jsonl` | Training data. |
| `OUTPUT_DIR` | `./rltt_output/${SLURM_JOB_ID}` | Training output directory. |
| `LOOP_WEIGHTING` | `exit_pdf` | Loop credit weighting strategy. |
| `USE_RLTT_MODEL` | `true` | Use RLTT-modified model. |
| `USE_LORA` | `false` | Full-parameter fine-tuning by default in the current wrapper. |
| `BETA` | `0.001` | KL coefficient. |
| `NUM_GENERATIONS` | `8` | Rollouts per prompt. |
| `NUM_PROMPTS_PER_BATCH` | `32` | Unique prompts per step. |
| `MAX_PROMPT_LENGTH` | `1024` | Prompt token limit. |
| `MAX_COMPLETION_LENGTH` | `2048` | Completion token limit. |
| `SAVE_STEPS` | `10` | Checkpoint save frequency. |

The script auto-detects `N_GPUS` from Slurm when possible. Set it explicitly
when you need to force a value:

```bash
sbatch --export=ALL,N_GPUS=4 run_rltt.slurm
```

## Checkpoints

Training runs write to:

```text
rltt_output/<slurm_job_id>/
```

A typical checkpoint path is:

```text
rltt_output/<slurm_job_id>/global_step_140/
```

The model weights used by merge/evaluation live under:

```text
rltt_output/<slurm_job_id>/global_step_140/actor/
```

When a Slurm evaluation script asks for `CHECKPOINT_PATH`, pass the
`global_step_*` directory, not the nested `actor` directory, unless that script
explicitly says otherwise.

## Single-Checkpoint Evaluation

Math-style evaluation scripts merge the FSDP checkpoint, run vLLM evaluation,
write `eval_results.json`, and remove the temporary merged model.

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_rltt.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_gsm8k.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_aime24.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140 run_eval_single_ckpt_beyondaime.slurm
```

Competition benchmarks use one launcher with a `BENCHMARK` override:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140,BENCHMARK=aime26 run_eval_single_ckpt_competitions.slurm
sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../global_step_140,BENCHMARK=hmmt25 run_eval_single_ckpt_competitions.slurm
```

Common evaluation overrides:

| Variable | Meaning |
| --- | --- |
| `MAX_NEW_TOKENS` | Completion token budget. |
| `MAX_PROMPT_LENGTH` | Prompt truncation/token budget. |
| `USE_FEW_SHOT` | Set `true` for few-shot prompting where supported. |
| `GPU_MEMORY_UTILIZATION` | vLLM memory fraction. |
| `OUTPUT_DIR` | Custom output location. |

## MCQA Evaluation

Use these wrappers for multiple-choice benchmarks:

```bash
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_arc_c.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_gpqa.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_mmlu_stem.slurm
```

The MCQA scripts use `eval_rltt_mcqa.py` and default to datasets under
`/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/`.

## Code Evaluation

Use these wrappers for code-generation benchmarks:

```bash
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_humaneval.slurm
sbatch --export=ALL,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_rltt_eval_mbpp.slurm
```

The code scripts use `eval_rltt_code.py` and write benchmark summaries under
`eval_outputs/` unless `OUTPUT_DIR` is overridden.

## Repeated Evaluation

Repeated evaluation generates multiple independent accuracy samples for t-tests
against GRPO. These wrappers all take `EXPERIMENT_DIR`, `CHECKPOINT`,
`NUM_RUNS`, `TEMPERATURE`, and `OUTPUT_DIR`.

Math:

```bash
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval.slurm
sbatch --export=ALL,BENCHMARK=gsm8k,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval.slurm
sbatch --export=ALL,BENCHMARK=aime26,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval.slurm
```

Non-math:

```bash
sbatch --export=ALL,BENCHMARK=arc_c,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval_non-math.slurm
sbatch --export=ALL,BENCHMARK=gpqa,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval_non-math.slurm
```

Code:

```bash
sbatch --export=ALL,BENCHMARK=humaneval,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval-code.slurm
sbatch --export=ALL,BENCHMARK=mbpp,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_repeated_eval-code.slurm
```

Default output directories:

| Wrapper | Output directory |
| --- | --- |
| `run_repeated_eval.slurm` | `../repeated_eval_results/` |
| `run_repeated_eval_non-math.slurm` | `../repeated_eval_results_non-math/` |
| `run_repeated_eval-code.slurm` | `../repeated_eval_results_code/` |

After RLTT and GRPO repeated evals finish, run the relevant
`run_ttest_analysis.py` script from the corresponding result directory.

## Analysis Jobs

pass@k:

```bash
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,K=8 run_pass_at_k_eval.slurm
```

Per-loop evaluation:

```bash
sbatch --export=ALL,BENCHMARK=beyondaime,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140,NUM_LOOPS=2 run_per_loop_eval.slurm
```

GSNR:

```bash
sbatch --export=ALL,BENCHMARK=math500,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_gsnr_eval.slurm
```

Rollout generation:

```bash
sbatch --export=ALL,BENCHMARK=gsm8k,EXPERIMENT_DIR=/scratch/.../rltt_output/<run_id>,CHECKPOINT=step_140 run_generate_rollouts.slurm
```

## Direct Python Entry Points

Direct Python commands are most useful for debugging. On the cluster, prefer
Slurm wrappers for real runs because they handle checkpoint merge and setup.

```bash
python rltt_train.py --model_path /path/to/model --train_file /path/to/train.jsonl
python evaluate_checkpoint.py --checkpoint_path /path/to/merged/model --test_file /path/to/test.jsonl
python eval_rltt_mcqa.py --checkpoint_path /path/to/merged/model --test_file /path/to/mcqa.jsonl --benchmark arc_c
python eval_rltt_code.py --checkpoint_path /path/to/merged/model --test_file /path/to/code.jsonl --benchmark mbpp
```

## Troubleshooting

- `CHECKPOINT_PATH` should usually be `.../global_step_<n>`, not
  `.../global_step_<n>/actor`.
- If evaluation fails on missing Ouro files, use the Slurm wrapper; it patches
  and merges FSDP checkpoint files before evaluation.
- If CUDA memory is tight, lower `PPO_MAX_TOKEN_LEN`, `LOG_PROB_MAX_TOKEN_LEN`,
  `NUM_PROMPTS_PER_BATCH`, or `VLLM_GPU_MEM`.
- `LOOP_WEIGHTING=exit_pdf` requires `USE_RLTT_MODEL=true` and a valid
  `RLTT_MODEL_PATH`.
- Logs are in `logs/<job_name>_<job_id>.out` and `.err`.
