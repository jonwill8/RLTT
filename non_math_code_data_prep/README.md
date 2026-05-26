# Code Data Preparation

Scripts for converting HumanEval and MBPP into the JSONL format used by the
code evaluation wrappers.

## Files

| File | Purpose |
| --- | --- |
| `prepare_code_datasets.py` | Build standardized HumanEval and MBPP test JSONL files. |
| `prepare_code_train_splits.py` | Build code training/evaluation splits. |

## Usage

```bash
cd /scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/non_math_code_data_prep
python prepare_code_datasets.py --dataset all --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/code
```

Individual datasets:

```bash
python prepare_code_datasets.py --dataset humaneval --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/code
python prepare_code_datasets.py --dataset mbpp --output_dir /scratch/gpfs/OLGARUS/jw4199/datasets/code
```

The evaluation launchers expect:

```text
/scratch/gpfs/OLGARUS/jw4199/datasets/code/humaneval.test.jsonl
/scratch/gpfs/OLGARUS/jw4199/datasets/code/mbpp.test.jsonl
```
