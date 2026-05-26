"""
RL-Trained Answer Parsing - Answer parsing for RL-trained model evaluation.

This module provides answer parsing functions for evaluating RL-trained models
(GRPO/RLTT). It uses the FIRST \\boxed{} occurrence for answer extraction.

Usage:
    from math_utils.rl_trained_answer_parsing import (
        rl_extract_boxed_answer,
        rl_check_math_answer,
        rl_get_gold_answer,
    )
"""

import re
from typing import Optional

# Try to import math_verify for mathematical equivalence checking
try:
    from math_verify import parse as math_parse, verify as math_verify
    from math_verify import LatexExtractionConfig, ExprExtractionConfig
    RL_MATH_VERIFY_AVAILABLE = True
except ImportError:
    RL_MATH_VERIFY_AVAILABLE = False
    math_parse = None
    math_verify = None
    LatexExtractionConfig = None
    ExprExtractionConfig = None


def rl_extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} in the solution text.

    Handles nested braces properly. Takes the FIRST \\boxed{} occurrence.

    Args:
        text: The full solution text containing \\boxed{answer}.

    Returns:
        The extracted answer string, or None if no boxed answer found.
    """
    if not text:
        return None

    # Find the first \boxed{ occurrence
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)

    if not match:
        return None

    # Take the first \boxed{}
    start = match.end()

    # Count braces to find matching closing brace
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        return text[start:pos-1].strip()
    return None


def rl_normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles common LaTeX variations:
    - \\dfrac vs \\frac
    - \\text{...} wrappers
    - \\left and \\right delimiters
    - Compact fractions like \\frac43
    - Currency symbols (\\$)
    - Percentage symbols
    - Multiple choice format variations
    - Degree symbols (^\\circ)

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized lowercase answer with whitespace removed.
    """
    if answer is None:
        return ""

    # Remove whitespace
    answer = answer.strip()

    # Normalize dfrac to frac (must be done before frac processing)
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")

    # Remove \text{} wrapper but keep content
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

    # Remove \textbf{} and \textit{} wrappers
    answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\textit\{([^}]*)\}', r'\1', answer)

    # Remove \mathrm{} wrapper
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)

    # Remove common LaTeX formatting/spacing commands
    answer = answer.replace("\\!", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\;", "")
    answer = answer.replace("\\:", "")
    answer = answer.replace("\\ ", " ")
    answer = answer.replace("\\quad", " ")
    answer = answer.replace("\\qquad", " ")

    # Remove \left and \right (keep the delimiter)
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")

    # Remove currency symbol
    answer = answer.replace("\\$", "")

    # Remove percentage symbol for comparison
    answer = answer.replace("\\%", "")
    answer = re.sub(r'(\d)\s*%', r'\1', answer)  # Remove % after numbers

    # Remove degree symbol (^\circ or \circ)
    answer = re.sub(r'\^\\circ', '', answer)
    answer = answer.replace("\\circ", "")

    # Remove units like "inches", "cm", etc. wrapped in \mbox or \text
    answer = re.sub(r'\\mbox\{[^}]*\}', '', answer)

    # Expand compact fractions: \frac43 -> \frac{4}{3}
    # Match \frac followed by two single characters (not braces)
    answer = re.sub(r'\\frac([^{])([^{])', r'\\frac{\1}{\2}', answer)

    # Normalize fractions: \frac{a}{b} -> (a)/(b)
    # Also handle negative fractions: -\frac{a}{b} -> (-a)/(b)
    def replace_frac(match):
        prefix = match.group(1) if match.group(1) else ""
        numer = match.group(2)
        denom = match.group(3)
        if prefix == "-":
            return f"(-{numer})/({denom})"
        return f"({numer})/({denom})"

    # Pattern with optional leading minus sign
    frac_pattern = r'(-)?\\frac\{([^{}]*)\}\{([^{}]*)\}'
    # Apply multiple times to handle nested fracs
    for _ in range(5):
        new_answer = re.sub(frac_pattern, replace_frac, answer)
        if new_answer == answer:
            break
        answer = new_answer

    # Also normalize simple fractions like -1/3 to (-1)/(3) for consistency
    # But be careful not to match things like dates or already normalized fracs
    # Match: optional minus, digits, slash, digits (not already in parens)
    answer = re.sub(r'(?<!\()(-?\d+)/(\d+)(?!\))', r'(\1)/(\2)', answer)

    # Remove $ signs
    answer = answer.replace("$", "")

    # Remove parentheses from multiple choice answers like (C) -> C
    answer = re.sub(r'^\(([A-E])\)$', r'\1', answer)

    # Normalize whitespace (remove all extra spaces)
    answer = re.sub(r'\s+', '', answer)

    return answer.lower()


def rl_normalize_for_math_verify(answer: str) -> str:
    """Light normalization to help math_verify parse the answer.

    Only normalizes LaTeX command variations without changing the mathematical structure.
    This is less aggressive than normalize_answer() to preserve structure for math_verify.

    Args:
        answer: The answer string to lightly normalize.

    Returns:
        Lightly normalized answer string.
    """
    if answer is None:
        return ""

    answer = answer.strip()

    # Normalize fraction commands (dfrac/tfrac -> frac)
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")

    # Remove \text{} wrapper but keep content
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\textit\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)

    # Remove \left and \right
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")

    # Remove currency/percentage symbols
    answer = answer.replace("\\$", "")
    answer = answer.replace("\\%", "")
    answer = re.sub(r'(\d)\s*%', r'\1', answer)

    # Expand compact fractions: \frac43 -> \frac{4}{3}
    answer = re.sub(r'\\frac([^{])([^{])', r'\\frac{\1}{\2}', answer)

    return answer


def rl_try_math_verify(pred: str, gold: str) -> bool:
    """Try to verify mathematical equivalence using math_verify.

    Args:
        pred: Predicted answer string.
        gold: Gold/expected answer string.

    Returns:
        True if math_verify confirms equivalence, False otherwise.
    """
    if not RL_MATH_VERIFY_AVAILABLE:
        return False

    if not pred or not gold:
        return False

    try:
        # Wrap answers in $ for LaTeX parsing if not already wrapped
        pred_latex = pred if pred.startswith("$") or pred.startswith("\\") else f"${pred}$"
        gold_latex = gold if gold.startswith("$") or gold.startswith("\\") else f"${gold}$"

        # Parse both answers with both LaTeX and expression extraction
        extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        pred_parsed = math_parse(pred_latex, extraction_config=extraction_config)
        gold_parsed = math_parse(gold_latex, extraction_config=extraction_config)

        # Verify equivalence (order matters: gold first, then prediction)
        return math_verify(gold_parsed, pred_parsed)
    except Exception:
        return False


def rl_check_math_answer(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer.

    Uses a multi-stage approach:
    1. Try math_verify on raw answers (handles mathematical equivalence)
    2. Try math_verify on lightly normalized answers (handles LaTeX variations)
    3. Fall back to fully normalized string comparison
    4. Try numeric comparison for simple numbers

    Args:
        pred: Predicted answer string.
        gold: Gold/expected answer string.

    Returns:
        True if answers match, False otherwise.
    """
    if not pred or not gold:
        return False

    # Stage 1: Try math_verify on raw answers first
    if rl_try_math_verify(pred, gold):
        return True

    # Stage 2: Try math_verify on lightly normalized answers (handles dfrac vs frac, etc.)
    pred_light = rl_normalize_for_math_verify(pred)
    gold_light = rl_normalize_for_math_verify(gold)
    if rl_try_math_verify(pred_light, gold_light):
        return True

    # Stage 3: Fallback to fully normalized string comparison
    pred_norm = rl_normalize_answer(pred)
    gold_norm = rl_normalize_answer(gold)

    # Direct match
    if pred_norm == gold_norm:
        return True

    # Stage 4: Try numeric comparison for simple numbers
    try:
        pred_num = float(pred_norm.replace(",", ""))
        gold_num = float(gold_norm.replace(",", ""))
        if abs(pred_num - gold_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def rl_get_gold_answer(example: dict) -> str:
    """Extract gold answer from example (handles both MATH and MATH-500 formats).

    Args:
        example: A dataset example dict with either 'answer' field (MATH-500)
                 or 'solution' field containing \\boxed{answer} (MATH).

    Returns:
        The gold answer string, or empty string if not found.
    """
    # MATH-500 has explicit 'answer' field
    if "answer" in example:
        return example["answer"]

    # MATH test: extract from solution's \boxed{}
    solution = example.get("solution", "")
    boxed_answer = rl_extract_boxed_answer(solution)
    if boxed_answer is not None:
        return boxed_answer

    return ""
