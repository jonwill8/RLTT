# Math Utilities

Shared helpers for math benchmark prompting, answer extraction, answer checking,
and reward computation. These modules are imported by RLTT, GRPO, SFT, baseline,
and repeated-evaluation scripts.

## Files

| File | Purpose |
| --- | --- |
| `answer_parsing.py` | General math answer extraction and equivalence checks. |
| `rl_trained_answer_parsing.py` | Parsing helpers tuned for RL-trained model outputs. |
| `reward.py` | Reward functions used during RL training/evaluation. |
| `prompting.py` | Prompt formatting helpers for math-style datasets. |
| `__init__.py` | Package exports. |

## Notes

- Prefer importing these utilities instead of duplicating answer parsing logic
  inside experiment scripts.
- Changes here can affect training rewards and evaluation accuracy across
  multiple benchmark families, so run a focused evaluation after edits.
