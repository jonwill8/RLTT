# Non-Math Data Preparation

Scripts for converting MCQA datasets into the JSONL format consumed by the
evaluation wrappers.

## Files

| File | Purpose |
| --- | --- |
| `prepare_mcqa_datasets.py` | Build standardized test JSONL files for ARC-Challenge, MMLU-STEM, and GPQA. |
| `prepare_mcqa_train_splits.py` | Build train/dev-style MCQA splits. |
| `extract_fewshot_examples.py` | Extract reusable few-shot examples. |

## Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/non_math_data_prep
python prepare_mcqa_datasets.py --dataset all --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/mcqa
```

Individual datasets:

```bash
python prepare_mcqa_datasets.py --dataset arc_c --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/mcqa
python prepare_mcqa_datasets.py --dataset mmlu_stem --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/mcqa
python prepare_mcqa_datasets.py --dataset gpqa --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/mcqa
```

The evaluation launchers expect files such as:

```text
/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/arc_challenge.test.jsonl
/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/mmlu_stem.test.jsonl
/scratch/gpfs/OLGARUS/jw4199/datasets/mcqa/gpqa.test.jsonl
```
