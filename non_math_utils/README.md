# Non-Math Utilities

Shared multiple-choice answer parsing utilities for ARC-Challenge, GPQA, and
MMLU-STEM style evaluations.

## Files

| File | Purpose |
| --- | --- |
| `answer_parsing.py` | Extract and normalize MCQA choices from model completions. |
| `__init__.py` | Package exports. |

## Used By

- `baseline_eval/eval_ouro_mcqa.py`
- `rltt_experiments/eval_rltt_mcqa.py`
- `grpo_experiments/eval_grpo_mcqa.py`
- `sft_experiments/eval_sft_mcqa.py`
- `baselines_*/*mcqa.py`

Keep choice-label behavior consistent here so MCQA comparisons remain aligned
across methods.
