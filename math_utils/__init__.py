"""
Centralized math answer parsing utilities.

This module provides robust answer extraction and comparison for math problems,
used consistently across baseline_eval, sft_experiments, grpo_experiments, and rltt_experiments.

Two answer parsing modules are available:
- answer_parsing.py: Base parsing functions
- rl_trained_answer_parsing.py: RL-trained model parsing (used by all evaluation scripts)
"""

from .answer_parsing import (
    extract_boxed_answer,
    normalize_answer,
    normalize_for_math_verify,
    try_math_verify,
    check_math_answer,
    get_gold_answer,
    MATH_VERIFY_AVAILABLE,
)

# RL-trained answer parsing (used by all evaluation scripts)
from .rl_trained_answer_parsing import (
    rl_extract_boxed_answer,
    rl_normalize_answer,
    rl_normalize_for_math_verify,
    rl_try_math_verify,
    rl_check_math_answer,
    rl_get_gold_answer,
    RL_MATH_VERIFY_AVAILABLE,
)

from .prompting import (
    FEW_SHOT_EXAMPLES,
    INSTRUCTION,
    build_chat_messages,
    format_math_prompt,
)

from .reward import (
    compute_score,
    compute_reward,
    compute_math_reward,
    create_math_reward_func,
)

__all__ = [
    # Base answer parsing functions
    "extract_boxed_answer",
    "normalize_answer",
    "normalize_for_math_verify",
    "try_math_verify",
    "check_math_answer",
    "get_gold_answer",
    "MATH_VERIFY_AVAILABLE",
    # RL-trained answer parsing (used by all evaluation scripts)
    "rl_extract_boxed_answer",
    "rl_normalize_answer",
    "rl_normalize_for_math_verify",
    "rl_try_math_verify",
    "rl_check_math_answer",
    "rl_get_gold_answer",
    "RL_MATH_VERIFY_AVAILABLE",
    # Prompting
    "FEW_SHOT_EXAMPLES",
    "INSTRUCTION",
    "build_chat_messages",
    "format_math_prompt",
    # Reward functions
    "compute_score",
    "compute_reward",
    "compute_math_reward",
    "create_math_reward_func",
]
