# Code Benchmark Utilities

Shared helpers for executing and checking generated code on HumanEval and MBPP.

## Files

| File | Purpose |
| --- | --- |
| `code_execution.py` | Sandboxed execution/checking helpers and timeout handling. |
| `__init__.py` | Package exports. |

## Used By

- `baseline_eval/eval_ouro_code.py`
- `rltt_experiments/eval_rltt_code.py`
- `grpo_experiments/eval_grpo_code.py`
- `sft_experiments/eval_sft_code.py`
- `baselines_qwen3/eval_qwen3_code.py`
- `baselines_deepseek_r1/eval_deepseek_r1_code.py`

Code execution utilities affect benchmark correctness directly. Keep changes
small and rerun at least one HumanEval or MBPP smoke test after editing.
