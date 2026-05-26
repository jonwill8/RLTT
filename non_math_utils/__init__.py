"""
Shared utilities for non-math benchmarks (ARC, MMLU, GPQA).
"""

from .answer_parsing import (
    extract_answer,
    normalize_answer,
    parse_multiple_choice,
    is_correct,
)

__all__ = [
    "extract_answer",
    "normalize_answer",
    "parse_multiple_choice",
    "is_correct",
]
