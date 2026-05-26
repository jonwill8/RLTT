#!/usr/bin/env python3
"""
Profile generation FLOPs on math benchmarks with Hugging Face generate().

This script uses DeepSpeed's established FLOPs profiler:
  deepspeed.profiling.flops_profiler.FlopsProfiler

It computes FLOPs per prompt/response rollout and reports aggregate statistics.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional

# Keep behavior consistent with the rest of RLTT scripts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for math_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_utils import format_math_prompt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


AIME26_NAME = "AIME 2026"
AIME26_FILE = "/scratch/gpfs/OLGARUS/jw4199/datasets/aime26/aime2026.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile rollout FLOPs with DeepSpeed FLOPs profiler + HF generate",
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF model path")
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional AIME26 test JSONL override",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=16,
        help="Number of random samples to profile (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Maximum prompt tokens",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3072,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Enable 5-shot CoT prompting",
    )
    parser.add_argument(
        "--total_ut_steps",
        type=int,
        default=None,
        help="Override Ouro recurrent steps if config has total_ut_steps",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for profiling outputs",
    )
    parser.add_argument(
        "--aggregate_every_rollouts",
        type=int,
        default=2,
        help="Emit running aggregate FLOP stats to stderr every N samples",
    )
    return parser.parse_args()


def load_test_data(file_path: str, max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if max_samples is not None and max_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)

    return data


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def safe_pstdev(values: List[float]) -> float:
    """Population std with stable behavior for small lists."""
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def main() -> None:
    args = parse_args()

    try:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed FLOPs profiler is required. Install deepspeed in this env."
        ) from exc

    bench_name = AIME26_NAME
    test_file = args.test_file or AIME26_FILE

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Generation FLOPs Profiling (HF generate + DeepSpeed profiler)")
    logger.info("=" * 72)
    logger.info("Model path: %s", args.model_path)
    logger.info("Benchmark: %s (fixed)", bench_name)
    logger.info("Test file: %s", test_file)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Max samples: %s", args.max_samples)
    logger.info("Few-shot: %s", args.use_few_shot)
    logger.info("Max prompt length: %s", args.max_prompt_length)
    logger.info("Max new tokens: %s", args.max_new_tokens)
    logger.info("Temperature: %s", args.temperature)

    torch.manual_seed(args.seed)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model.eval()

    if args.total_ut_steps is not None and hasattr(model.config, "total_ut_steps"):
        logger.info("Setting model.config.total_ut_steps = %d", args.total_ut_steps)
        model.config.total_ut_steps = args.total_ut_steps

    logger.info("Loading benchmark samples...")
    dataset = load_test_data(test_file, args.max_samples, args.seed)
    logger.info("Loaded %d samples", len(dataset))

    details_path = os.path.join(args.output_dir, "flop_profile_details.jsonl")
    rollouts_live_path = os.path.join(args.output_dir, "rollouts.jsonl")
    responses_path = os.path.join(args.output_dir, "sample_responses.jsonl")
    per_sample_flops_path = os.path.join(args.output_dir, "per_sample_flops.jsonl")
    summary_path = os.path.join(args.output_dir, "flop_profile_summary.json")

    sample_records: List[Dict[str, Any]] = []
    response_records: List[Dict[str, Any]] = []
    per_sample_flops_records: List[Dict[str, Any]] = []
    flops_values: List[float] = []
    prefill_lens: List[int] = []
    gen_lens: List[int] = []
    runtimes: List[float] = []

    # Stream rollouts as they are generated (live logging).
    with open(rollouts_live_path, "w") as rollouts_f:
        for idx, item in enumerate(dataset):
            problem = item.get("problem", "")
            prompt = format_math_prompt(problem, tokenizer, use_few_shot=args.use_few_shot)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = int(inputs["input_ids"].shape[1])
            prefill_lens.append(prompt_len)

            profiler = FlopsProfiler(model)
            profiler.start_profile()

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            elapsed = time.time() - start_time
            runtimes.append(elapsed)

            total_flops = float(profiler.get_total_flops(as_string=False))
            total_macs = float(profiler.get_total_macs(as_string=False))
            total_params = float(profiler.get_total_params(as_string=False))
            profiler.end_profile()

            generated_ids = outputs[0][prompt_len:]
            gen_len = int(generated_ids.shape[0])
            gen_lens.append(gen_len)

            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            flops_values.append(total_flops)
            record = {
                "index": idx,
                "prompt_tokens": prompt_len,
                "generated_tokens": gen_len,
                "runtime_sec": elapsed,
                "total_flops": total_flops,
                "total_macs": total_macs,
                "total_params": total_params,
                "flops_per_generated_token": (total_flops / max(gen_len, 1)),
                "problem": problem,
                "response": response,
            }
            sample_records.append(record)
            rollout_record = {
                "index": idx,
                "problem": problem,
                "response": response,
                "prompt_tokens": prompt_len,
                "generated_tokens": gen_len,
            }
            response_records.append(rollout_record)
            per_sample_flops_record = {
                "index": idx,
                "prompt_tokens": prompt_len,
                "generated_tokens": gen_len,
                "runtime_sec": elapsed,
                "total_flops": total_flops,
                "total_macs": total_macs,
                "total_params": total_params,
                "flops_per_generated_token": (total_flops / max(gen_len, 1)),
            }
            per_sample_flops_records.append(per_sample_flops_record)

            # Print per-sample FLOPs in stderr (.err) in machine-readable format.
            logger.info(
                "PER_SAMPLE_FLOPS_JSON %s",
                json.dumps(per_sample_flops_record, sort_keys=True),
            )
            logger.info(
                "[%d/%d] prompt=%d gen=%d flops=%.3e runtime=%.2fs",
                idx + 1,
                len(dataset),
                prompt_len,
                gen_len,
                total_flops,
                elapsed,
            )
            sys.stderr.flush()

            # Emit running aggregate stats to stderr every N rollouts.
            if (
                args.aggregate_every_rollouts > 0
                and ((idx + 1) % args.aggregate_every_rollouts == 0 or (idx + 1) == len(dataset))
            ):
                running_flops_per_token = [
                    f / max(t, 1) for f, t in zip(flops_values, gen_lens)
                ]
                running_agg = {
                    "num_samples": idx + 1,
                    "mean_total_flops": mean(flops_values),
                    "std_total_flops": safe_pstdev(flops_values),
                    "mean_flops_per_generated_token": mean(running_flops_per_token),
                    "std_flops_per_generated_token": safe_pstdev(running_flops_per_token),
                    "mean_generated_tokens": mean(gen_lens),
                    "std_generated_tokens": safe_pstdev([float(x) for x in gen_lens]),
                }
                logger.info(
                    "RUNNING_FLOPS_AGG_JSON %s",
                    json.dumps(running_agg, sort_keys=True),
                )
                sys.stderr.flush()

            # Live-write rollout records so users can inspect progress while running.
            rollouts_f.write(json.dumps(rollout_record) + "\n")
            rollouts_f.flush()

    with open(details_path, "w") as f:
        for row in sample_records:
            f.write(json.dumps(row) + "\n")
    with open(responses_path, "w") as f:
        for row in response_records:
            f.write(json.dumps(row) + "\n")
    with open(per_sample_flops_path, "w") as f:
        for row in per_sample_flops_records:
            f.write(json.dumps(row) + "\n")

    summary = {
        "benchmark": "aime26",
        "benchmark_name": bench_name,
        "model_path": args.model_path,
        "test_file": test_file,
        "max_samples": len(dataset),
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "use_few_shot": args.use_few_shot,
        "torch_dtype": args.torch_dtype,
        "total_ut_steps": args.total_ut_steps,
        "stats": {
            "mean_total_flops": mean(flops_values) if flops_values else 0.0,
            "median_total_flops": median(flops_values) if flops_values else 0.0,
            "min_total_flops": min(flops_values) if flops_values else 0.0,
            "max_total_flops": max(flops_values) if flops_values else 0.0,
            "mean_prompt_tokens": mean(prefill_lens) if prefill_lens else 0.0,
            "mean_generated_tokens": mean(gen_lens) if gen_lens else 0.0,
            "mean_runtime_sec": mean(runtimes) if runtimes else 0.0,
            "mean_flops_per_generated_token": (
                mean([f / max(t, 1) for f, t in zip(flops_values, gen_lens)])
                if flops_values
                else 0.0
            ),
        },
        "files": {
            "details_jsonl": details_path,
            "rollouts_jsonl": rollouts_live_path,
            "responses_jsonl": responses_path,
            "per_sample_flops_jsonl": per_sample_flops_path,
            "summary_json": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 72)
    logger.info("Mean total FLOPs per sample: %.3e", summary["stats"]["mean_total_flops"])
    logger.info("Mean generated tokens per sample: %.2f", summary["stats"]["mean_generated_tokens"])
    logger.info("Mean FLOPs/generated token: %.3e", summary["stats"]["mean_flops_per_generated_token"])
    logger.info("Wrote details: %s", details_path)
    logger.info("Wrote live rollouts: %s", rollouts_live_path)
    logger.info("Wrote responses: %s", responses_path)
    logger.info("Wrote per-sample FLOPs: %s", per_sample_flops_path)
    logger.info("Wrote summary: %s", summary_path)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
