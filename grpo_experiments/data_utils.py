"""
Data loading and processing utilities for GRPO training with verl.

Handles:
- JSONL to Parquet conversion for verl compatibility
- Dataset preprocessing with chat templates
- Validation dataset loading
"""
import os
import sys
import json
import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from centralized math_utils
from math_utils import (
    extract_boxed_answer,
    get_gold_answer,
    build_chat_messages,
    INSTRUCTION,
)

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def convert_math_to_parquet(
    jsonl_path: str,
    output_path: str,
    tokenizer=None,
    use_few_shot: bool = False,
) -> str:
    """Convert MATH JSONL dataset to Parquet format for verl.

    verl expects parquet files with specific columns:
    - prompt: List of message dicts with 'role' and 'content' keys
             (verl applies chat template internally)
    - data_source: Source identifier (e.g., "math")
    - reward_model: Dict with reward model config including ground_truth
    - extra_info: Dict with additional metadata

    Args:
        jsonl_path: Path to input JSONL file
        output_path: Path for output parquet file
        tokenizer: Optional tokenizer (not used, kept for API compatibility)
        use_few_shot: Whether to use few-shot prompting

    Returns:
        Path to the created parquet file
    """
    logger.info(f"Loading JSONL from {jsonl_path}")
    data = load_jsonl(jsonl_path)
    logger.info(f"Loaded {len(data)} examples")

    processed_data = []
    for idx, item in enumerate(data):
        problem = item.get("problem", "")

        # Extract gold answer using centralized function
        gold_content = get_gold_answer(item)

        # Build prompt as list of message dicts (verl format)
        # verl expects this format and applies chat template internally
        if use_few_shot:
            # For few-shot, build full messages list
            messages = build_chat_messages(problem, use_few_shot=True)
        else:
            # Simple single-turn format
            messages = [{"role": "user", "content": f"{INSTRUCTION}\n\nProblem: {problem}"}]

        # verl expects reward_model with ground_truth as a string
        # The built-in math_dapo reward function calls ground_truth.split("=")
        reward_model_config = {
            "style": "rule",
            "ground_truth": gold_content,  # Must be a string, not a dict
        }

        processed_data.append({
            "prompt": messages,  # List of message dicts, NOT a string
            "data_source": "math",
            "ability": "math",
            "reward_model": reward_model_config,
            "extra_info": {
                "index": idx,
                "problem": problem,
                "level": item.get("level", ""),
                # MATH-500 uses "subject", training MATH uses "type"
                "type": item.get("type") or item.get("subject", ""),
            }
        })

    # Create DataFrame and save as parquet
    df = pd.DataFrame(processed_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Save to parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    logger.info(f"Saved {len(df)} examples to {output_path}")
    return output_path


def prepare_train_data(
    train_file: str,
    output_dir: str,
    tokenizer=None,
    use_few_shot: bool = False,
    force_regenerate: bool = True,
) -> str:
    """Prepare training data in parquet format.

    Args:
        train_file: Path to training JSONL file
        output_dir: Directory to save parquet file
        tokenizer: Tokenizer for chat template (kept for API compatibility)
        use_few_shot: Whether to use few-shot prompting
        force_regenerate: Always regenerate parquet file (default True for verl format changes)

    Returns:
        Path to the parquet file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.parquet")

    if os.path.exists(output_path) and not force_regenerate:
        logger.info(f"Using existing parquet file: {output_path}")
        return output_path

    return convert_math_to_parquet(
        train_file,
        output_path,
        tokenizer=tokenizer,
        use_few_shot=use_few_shot,
    )


def prepare_test_data(
    test_file: str,
    output_dir: str,
    tokenizer=None,
    use_few_shot: bool = False,
    force_regenerate: bool = True,
) -> str:
    """Prepare test data in parquet format.

    Args:
        test_file: Path to test JSONL file
        output_dir: Directory to save parquet file
        tokenizer: Tokenizer for chat template (kept for API compatibility)
        use_few_shot: Whether to use few-shot prompting
        force_regenerate: Always regenerate parquet file (default True for verl format changes)

    Returns:
        Path to the parquet file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use filename from test_file
    base_name = os.path.basename(test_file).replace(".jsonl", ".parquet")
    output_path = os.path.join(output_dir, base_name)

    if os.path.exists(output_path) and not force_regenerate:
        logger.info(f"Using existing parquet file: {output_path}")
        return output_path

    return convert_math_to_parquet(
        test_file,
        output_path,
        tokenizer=tokenizer,
        use_few_shot=use_few_shot,
    )


def load_validation_dataset(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load validation dataset from JSONL file."""
    data = load_jsonl(data_path)

    if max_samples is not None and max_samples < len(data):
        import random
        rng = random.Random(42)
        data = rng.sample(data, max_samples)

    return data


def format_validation_prompt(problem: str, tokenizer, use_few_shot: bool = False) -> str:
    """Format a problem for validation generation."""
    messages = build_chat_messages(problem, use_few_shot=use_few_shot)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt
