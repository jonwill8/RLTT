#!/usr/bin/env python3
"""
Qwen3 GRPO training entrypoint.

This wrapper reuses the canonical GRPO implementation in RLTT/grpo_experiments
so baselines_qwen3 can launch training locally without duplicating trainer code.
"""

import runpy
import sys
import os
from pathlib import Path


def main() -> None:
    # Force this entrypoint to be GRPO-only.
    if os.environ.get("USE_PPO", "").lower() == "true":
        raise ValueError("PPO is disabled for baselines_qwen3 GRPO training (USE_PPO=true detected).")
    if "--use_ppo" in sys.argv:
        raise ValueError("PPO is disabled for baselines_qwen3 GRPO training (--use_ppo detected).")

    this_dir = Path(__file__).resolve().parent
    shared_train = this_dir.parent / "grpo_experiments" / "grpo_train.py"

    if not shared_train.exists():
        raise FileNotFoundError(f"Shared trainer not found: {shared_train}")

    # Ensure imports like data_utils/simple_trainer resolve from grpo_experiments.
    sys.path.insert(0, str(shared_train.parent))
    sys.argv[0] = str(shared_train)
    runpy.run_path(str(shared_train), run_name="__main__")


if __name__ == "__main__":
    main()
