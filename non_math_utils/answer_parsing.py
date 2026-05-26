"""
Shared answer parsing utilities for multiple-choice benchmarks (ARC, MMLU, GPQA).

This module uses STRICT answer extraction - only explicit final answer statements
are accepted. Intermediate reasoning mentions (e.g., "Option C seems correct")
are NOT counted as answers. This prevents truncated responses from being
incorrectly marked as correct.
"""

import re
from typing import Optional


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string by stripping whitespace and converting to uppercase.

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer (uppercase, stripped)
    """
    if answer is None:
        return ""
    return answer.strip().upper()


def extract_answer(response: str, valid_choices: Optional[list[str]] = None) -> Optional[str]:
    """
    Extract a multiple-choice answer (A, B, C, D, etc.) from a model response.

    STRICT MODE: Only extracts explicit final answer statements. Does NOT extract
    intermediate reasoning mentions like "Option C seems correct" or "So C might work".
    This prevents truncated responses from being incorrectly marked as correct.

    Accepted formats (explicit final answers only):
    - "**Answer: A**" or "**Answer: (A)**"
    - "Answer: B" or "The answer is B"
    - "\\boxed{C}" (LaTeX format)
    - Final line with just the letter: "D" or "(D)"

    NOT accepted (intermediate reasoning):
    - "Option C seems correct"
    - "So the answer might be B"
    - "I think A is right"

    Args:
        response: The model's response text
        valid_choices: List of valid answer choices (default: A, B, C, D)

    Returns:
        Extracted answer letter (uppercase) or None if not found
    """
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D"]

    valid_choices = [c.upper() for c in valid_choices]
    choices_pattern = "|".join(valid_choices)

    # Clean up the response
    response = response.strip()

    # Check if response appears truncated (ends mid-sentence without punctuation)
    # This helps detect incomplete generations
    # Note: } is included for \boxed{} endings
    last_char = response[-1] if response else ""
    appears_truncated = last_char not in ".!?\"'`*)\n}" and len(response) > 100

    # STRICT patterns - only explicit final answer statements
    # These are checked in order of priority (most definitive first)

    # Pattern 1a: Markdown bold answer format "**Answer: X**" or "**Answer: (X)**"
    # Also handles "**Answer: A. True, True**" where answer text follows the letter
    match = re.search(rf"\*\*[Aa]nswer\s*:\s*[\(\[]?\s*({choices_pattern})\b[^*]*\*\*", response)
    if match:
        return match.group(1).upper()

    # Pattern 1b: Markdown bold answer with letter outside: "**Answer:** X" or "**Answer:** (X)"
    # This handles cases like "**Answer:** A. Both carry energy."
    match = re.search(rf"\*\*[Aa]nswer\s*:\s*\*\*\s*[\(\[]?\s*({choices_pattern})\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 1b2: Markdown bold closes before colon: "**Answer**: X"
    # This handles cases like "**Answer**: B. 10⁻⁴ eV."
    match = re.search(rf"\*\*[Aa]nswer\s*\*\*\s*:\s*[\(\[]?\s*({choices_pattern})\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 1c: "**Final Answer**:" variant with letter on same line or next line
    # Handles "**Final Answer**:\nA. 4.12 MeV..." or "**Final Answer**: A"
    match = re.search(rf"\*\*[Ff]inal\s+[Aa]nswer\s*\**\s*:\s*\**\s*[\n\s]*[\(\[]?\s*({choices_pattern})\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 1d: "**Answer**:" with answer on next line (handles newline between label and answer)
    # Handles "**Answer**:\nB. 10⁻⁴ eV"
    match = re.search(rf"\*\*[Aa]nswer\s*\**\s*:\s*\**\s*\n\s*[\(\[]?\s*({choices_pattern})\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 2: LaTeX boxed format \boxed{X}
    # Also handles "**Answer:** \boxed{X}" combination
    match = re.search(rf"\\boxed\s*\{{\s*({choices_pattern})\s*\}}", response)
    if match:
        return match.group(1).upper()

    # Pattern 2b: Answer with boxed: "**Answer:** \boxed{X}" or "Answer: \boxed{X}"
    match = re.search(rf"[Aa]nswer\s*:?\s*\**\s*\\boxed\s*\{{\s*({choices_pattern})\s*\}}", response)
    if match:
        return match.group(1).upper()

    # Pattern 3: "Answer: X" at the start of a line (definitive statement)
    # Handles both "Answer: D" and "Answer: D. the independent variable"
    match = re.search(rf"^[Aa]nswer\s*:\s*[\(\[]?\s*({choices_pattern})\b", response, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Pattern 4: "The answer is X" followed by end of sentence or response
    # Must be a complete statement, not "The answer is X because..." continuing
    match = re.search(rf"[Tt]he\s+answer\s+is\s*[\(\[]?\s*({choices_pattern})\s*[\)\]]?\s*[.\n]?\s*$", response)
    if match:
        return match.group(1).upper()

    # Pattern 5: "The correct answer is X" - very explicit
    # Also handles "the correct answer is:\n\nD. text"
    match = re.search(rf"[Tt]he\s+correct\s+answer\s+is\s*:?\s*[\n\s]*[\(\[]?\s*({choices_pattern})\b", response)
    if match:
        return match.group(1).upper()

    # Pattern 6: Standalone answer on final line "(X)" or "X" or "X."
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Check for "(X)" or "[X]" format
        match = re.match(rf"^\s*[\(\[]\s*({choices_pattern})\s*[\)\]]\s*\.?\s*$", last_line)
        if match:
            return match.group(1).upper()
        # Check for just "X" or "X." as the entire last line
        match = re.match(rf"^\s*({choices_pattern})\s*\.?\s*$", last_line)
        if match:
            return match.group(1).upper()

    # Pattern 7: "Therefore, X" or "Thus, X" or "So, X" as final answer
    match = re.search(rf"(?:[Tt]herefore|[Tt]hus|[Ss]o),?\s*[\(\[]?\s*({choices_pattern})\s*[\)\]]?\s*[.\n]?\s*$", response)
    if match:
        return match.group(1).upper()

    # Pattern 8: "I choose X" or "I'll go with X" or "My answer is X"
    match = re.search(rf"(?:[Ii]\s+choose|[Ii]'ll\s+go\s+with|[Mm]y\s+answer\s+is)\s*[\(\[]?\s*({choices_pattern})\s*[\)\]]?", response)
    if match:
        return match.group(1).upper()

    # If response appears truncated, do NOT fall back to extracting intermediate mentions
    if appears_truncated:
        return None

    # Pattern 9 (fallback for non-truncated responses):
    # "The answer is X" anywhere (but only if response is complete)
    match = re.search(rf"[Tt]he\s+answer\s+is\s*[\(\[]?\s*({choices_pattern})\s*[\)\]]?", response)
    if match:
        return match.group(1).upper()

    # Pattern 10 (fallback): If valid_choices are numeric (1,2,3,4) but model outputs A/B/C/D,
    # try to extract A/B/C/D and map to the corresponding numeric choice
    if all(c.isdigit() for c in valid_choices):
        letter_pattern = "A|B|C|D"
        # Try boxed format with letters
        match = re.search(rf"\\boxed\s*\{{\s*({letter_pattern})\s*\}}", response)
        if match:
            letter = match.group(1).upper()
            # Map A->1, B->2, C->3, D->4 (or to the corresponding valid_choice)
            letter_idx = ord(letter) - ord('A')
            if letter_idx < len(valid_choices):
                return valid_choices[letter_idx]

        # Try other common patterns with letters
        match = re.search(rf"[Aa]nswer\s*:?\s*\**\s*[\(\[]?\s*({letter_pattern})\b", response)
        if match:
            letter = match.group(1).upper()
            letter_idx = ord(letter) - ord('A')
            if letter_idx < len(valid_choices):
                return valid_choices[letter_idx]

    return None


def parse_multiple_choice(
    response: str,
    choices: list[str],
    choice_labels: Optional[list[str]] = None
) -> tuple[Optional[str], Optional[int]]:
    """
    Parse a multiple-choice response and return both the label and index.

    Args:
        response: The model's response text
        choices: List of answer choice texts (the actual content)
        choice_labels: Labels for choices (default: A, B, C, D...)

    Returns:
        Tuple of (answer_label, answer_index) or (None, None) if not found
    """
    if choice_labels is None:
        choice_labels = [chr(ord('A') + i) for i in range(len(choices))]

    answer = extract_answer(response, choice_labels)

    if answer is None:
        return None, None

    try:
        index = choice_labels.index(answer.upper())
        return answer.upper(), index
    except ValueError:
        return None, None


def is_correct(
    response: str,
    correct_answer: str,
    valid_choices: Optional[list[str]] = None
) -> bool:
    """
    Check if a model response contains the correct answer.

    Args:
        response: The model's response text
        correct_answer: The correct answer (letter or index)
        valid_choices: List of valid answer choices (default: A, B, C, D)

    Returns:
        True if the extracted answer matches the correct answer
    """
    extracted = extract_answer(response, valid_choices)

    if extracted is None:
        return False

    # Normalize correct answer
    correct_normalized = normalize_answer(correct_answer)

    # Handle numeric correct answers (convert 0 -> A, 1 -> B, etc.)
    if correct_normalized.isdigit():
        correct_normalized = chr(ord('A') + int(correct_normalized))

    return extracted == correct_normalized
