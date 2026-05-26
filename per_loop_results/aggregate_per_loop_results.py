#!/usr/bin/env python3
"""
Per-Loop Results Aggregation Script

This script aggregates per-loop evaluation results from GRPO and RLTT experiments
into CSV files for easy comparison and analysis.

Output files:
- per_loop_comparison_{benchmark}.csv: Side-by-side GRPO vs RLTT for each loop count
- grpo_per_loop_{benchmark}.csv: GRPO results across all loop counts
- rltt_per_loop_{benchmark}.csv: RLTT results across all loop counts
- per_loop_analysis_{benchmark}.json: Detailed analysis with level breakdowns

Usage:
    # Aggregate results for MATH-500:
    python aggregate_per_loop_results.py --benchmark math500

    # Aggregate all benchmarks:
    python aggregate_per_loop_results.py --all

    # Aggregate with custom directories:
    python aggregate_per_loop_results.py \
        --benchmark math500 \
        --grpo_dir /path/to/per_loop_results/grpo \
        --rltt_dir /path/to/per_loop_results/rltt
"""
import os
import sys
import json
import argparse
import glob
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def find_latest_eval_result(
    results_dir: str,
    benchmark: str,
    num_loops: int,
    checkpoint: str = "step_140"
) -> Optional[str]:
    """Find the most recent evaluation result for given parameters."""
    # Pattern: {benchmark}_loops{num_loops}_{checkpoint}_{timestamp}/eval_results.json
    pattern = os.path.join(
        results_dir,
        f"{benchmark}_loops{num_loops}_{checkpoint}_*",
        "eval_results.json"
    )
    matches = glob.glob(pattern)

    if not matches:
        return None

    # Sort by directory timestamp (embedded in name) and return latest
    matches.sort(reverse=True)
    return matches[0]


def load_eval_result(filepath: str) -> Optional[Dict[str, Any]]:
    """Load evaluation result from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def aggregate_method_results(
    results_dir: str,
    benchmark: str,
    method: str,
    checkpoint: str = "step_140"
) -> List[Dict[str, Any]]:
    """Aggregate results for a single method across all loop counts."""
    results = []

    for num_loops in [1, 2, 3]:
        result_path = find_latest_eval_result(
            results_dir, benchmark, num_loops, checkpoint
        )

        if result_path is None:
            print(f"  No results found for {method} loops={num_loops}")
            continue

        data = load_eval_result(result_path)
        if data is None:
            continue

        # Extract relevant information
        metrics = data.get("metrics", {})
        overall = metrics.get("overall", {})

        result_entry = {
            "method": method,
            "benchmark": benchmark,
            "checkpoint": checkpoint,
            "num_loops": num_loops,
            "accuracy": overall.get("accuracy", 0),
            "correct": overall.get("correct", 0),
            "total": overall.get("total", 0),
            "source_file": result_path,
            "timestamp": data.get("timestamp", "unknown"),
        }

        # Add level breakdowns
        by_level = metrics.get("by_level", {})
        for level, stats in by_level.items():
            result_entry[f"level_{level}_accuracy"] = stats.get("accuracy", 0)
            result_entry[f"level_{level}_correct"] = stats.get("correct", 0)
            result_entry[f"level_{level}_total"] = stats.get("total", 0)

        results.append(result_entry)
        print(f"  Loaded {method} loops={num_loops}: {overall.get('accuracy', 0):.2%}")

    return results


def write_method_csv(
    results: List[Dict[str, Any]],
    output_path: str
):
    """Write results for a single method to CSV."""
    if not results:
        print(f"  No results to write for {output_path}")
        return

    # Determine all columns
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Order columns sensibly
    base_cols = ["method", "benchmark", "checkpoint", "num_loops",
                 "accuracy", "correct", "total"]
    level_cols = sorted([k for k in all_keys if k.startswith("level_")])
    other_cols = sorted([k for k in all_keys if k not in base_cols + level_cols])

    columns = base_cols + level_cols + other_cols

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("num_loops", 0)):
            writer.writerow({k: r.get(k, "") for k in columns})

    print(f"  Wrote: {output_path}")


def write_comparison_csv(
    grpo_results: List[Dict[str, Any]],
    rltt_results: List[Dict[str, Any]],
    output_path: str
):
    """Write side-by-side comparison CSV."""
    # Index results by loop count
    grpo_by_loops = {r["num_loops"]: r for r in grpo_results}
    rltt_by_loops = {r["num_loops"]: r for r in rltt_results}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "num_loops",
            "grpo_accuracy", "grpo_correct", "grpo_total",
            "rltt_accuracy", "rltt_correct", "rltt_total",
            "diff_accuracy", "rltt_better"
        ])

        # Data rows
        for num_loops in [1, 2, 3]:
            grpo = grpo_by_loops.get(num_loops, {})
            rltt = rltt_by_loops.get(num_loops, {})

            grpo_acc = grpo.get("accuracy", 0)
            rltt_acc = rltt.get("accuracy", 0)
            diff = rltt_acc - grpo_acc

            writer.writerow([
                num_loops,
                f"{grpo_acc:.4f}" if grpo else "",
                grpo.get("correct", ""),
                grpo.get("total", ""),
                f"{rltt_acc:.4f}" if rltt else "",
                rltt.get("correct", ""),
                rltt.get("total", ""),
                f"{diff:+.4f}" if grpo and rltt else "",
                "Yes" if diff > 0 else "No" if diff < 0 else "Tie"
            ])

    print(f"  Wrote: {output_path}")


def write_analysis_json(
    grpo_results: List[Dict[str, Any]],
    rltt_results: List[Dict[str, Any]],
    benchmark: str,
    output_path: str
):
    """Write detailed analysis JSON."""
    analysis = {
        "benchmark": benchmark,
        "generated_at": datetime.now().isoformat(),
        "grpo": {},
        "rltt": {},
        "comparison": {}
    }

    # Organize GRPO results
    for r in grpo_results:
        loops = r["num_loops"]
        analysis["grpo"][f"loops_{loops}"] = {
            "accuracy": r.get("accuracy", 0),
            "correct": r.get("correct", 0),
            "total": r.get("total", 0),
            "by_level": {
                k.replace("level_", "").replace("_accuracy", ""): r[k]
                for k in r.keys()
                if k.startswith("level_") and k.endswith("_accuracy")
            }
        }

    # Organize RLTT results
    for r in rltt_results:
        loops = r["num_loops"]
        analysis["rltt"][f"loops_{loops}"] = {
            "accuracy": r.get("accuracy", 0),
            "correct": r.get("correct", 0),
            "total": r.get("total", 0),
            "by_level": {
                k.replace("level_", "").replace("_accuracy", ""): r[k]
                for k in r.keys()
                if k.startswith("level_") and k.endswith("_accuracy")
            }
        }

    # Compute comparisons
    grpo_by_loops = {r["num_loops"]: r for r in grpo_results}
    rltt_by_loops = {r["num_loops"]: r for r in rltt_results}

    for num_loops in [1, 2, 3]:
        grpo = grpo_by_loops.get(num_loops, {})
        rltt = rltt_by_loops.get(num_loops, {})

        if grpo and rltt:
            grpo_acc = grpo.get("accuracy", 0)
            rltt_acc = rltt.get("accuracy", 0)
            diff = rltt_acc - grpo_acc

            analysis["comparison"][f"loops_{num_loops}"] = {
                "grpo_accuracy": grpo_acc,
                "rltt_accuracy": rltt_acc,
                "diff_accuracy": diff,
                "rltt_better": diff > 0,
                "improvement_pct": (diff / grpo_acc * 100) if grpo_acc > 0 else 0
            }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"  Wrote: {output_path}")


def print_summary_table(
    grpo_results: List[Dict[str, Any]],
    rltt_results: List[Dict[str, Any]],
    benchmark: str
):
    """Print a summary table to console."""
    grpo_by_loops = {r["num_loops"]: r for r in grpo_results}
    rltt_by_loops = {r["num_loops"]: r for r in rltt_results}

    print("")
    print("=" * 70)
    print(f"Per-Loop Comparison: {benchmark.upper()}")
    print("=" * 70)
    print("")
    print(f"{'Loops':<8} {'GRPO':<20} {'RLTT':<20} {'Diff':<12}")
    print("-" * 60)

    for num_loops in [1, 2, 3]:
        grpo = grpo_by_loops.get(num_loops, {})
        rltt = rltt_by_loops.get(num_loops, {})

        grpo_acc = grpo.get("accuracy", 0)
        rltt_acc = rltt.get("accuracy", 0)

        grpo_str = f"{grpo_acc:.2%} ({grpo.get('correct', 0)}/{grpo.get('total', 0)})" if grpo else "N/A"
        rltt_str = f"{rltt_acc:.2%} ({rltt.get('correct', 0)}/{rltt.get('total', 0)})" if rltt else "N/A"
        diff_str = f"{(rltt_acc - grpo_acc)*100:+.2f}%" if grpo and rltt else "N/A"

        print(f"{num_loops:<8} {grpo_str:<20} {rltt_str:<20} {diff_str:<12}")

    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-loop evaluation results into CSVs"
    )
    parser.add_argument("--benchmark", type=str,
                        choices=["math500", "gsm8k", "aime24", "beyondaime"],
                        help="Benchmark to aggregate results for")
    parser.add_argument("--all", action="store_true",
                        help="Aggregate all benchmarks")
    parser.add_argument("--grpo_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/per_loop_results/grpo",
                        help="Directory containing GRPO per-loop results")
    parser.add_argument("--rltt_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/per_loop_results/rltt",
                        help="Directory containing RLTT per-loop results")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/per_loop_results",
                        help="Output directory for aggregated results")
    parser.add_argument("--checkpoint", type=str, default="step_140",
                        help="Checkpoint to aggregate results for")
    args = parser.parse_args()

    # Determine which benchmarks to process
    if args.all:
        benchmarks = ["math500", "gsm8k", "aime24", "beyondaime"]
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        print("Error: Must specify --benchmark or --all")
        parser.print_help()
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each benchmark
    for benchmark in benchmarks:
        print("")
        print(f"Processing {benchmark}...")

        # Aggregate GRPO results
        print(f"  Loading GRPO results from: {args.grpo_dir}")
        grpo_results = aggregate_method_results(
            args.grpo_dir, benchmark, "grpo", args.checkpoint
        )

        # Aggregate RLTT results
        print(f"  Loading RLTT results from: {args.rltt_dir}")
        rltt_results = aggregate_method_results(
            args.rltt_dir, benchmark, "rltt", args.checkpoint
        )

        # Write individual method CSVs
        write_method_csv(
            grpo_results,
            os.path.join(args.output_dir, f"grpo_per_loop_{benchmark}.csv")
        )
        write_method_csv(
            rltt_results,
            os.path.join(args.output_dir, f"rltt_per_loop_{benchmark}.csv")
        )

        # Write comparison CSV
        write_comparison_csv(
            grpo_results,
            rltt_results,
            os.path.join(args.output_dir, f"per_loop_comparison_{benchmark}.csv")
        )

        # Write analysis JSON
        write_analysis_json(
            grpo_results,
            rltt_results,
            benchmark,
            os.path.join(args.output_dir, f"per_loop_analysis_{benchmark}.json")
        )

        # Print summary
        print_summary_table(grpo_results, rltt_results, benchmark)

    print("Aggregation complete!")


if __name__ == "__main__":
    main()
