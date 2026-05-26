#!/usr/bin/env python3
"""
Aggregate Pass@k Results

This script aggregates pass@k evaluation results from both GRPO and RLTT
experiments and produces summary CSVs and analysis JSON files.

Usage:
    python aggregate_pass_at_k.py
    python aggregate_pass_at_k.py --benchmark math500
    python aggregate_pass_at_k.py --k 8
"""
import os
import sys
import json
import csv
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Results directory (same as this script's directory)
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def find_pass_at_k_results(
    method: Optional[str] = None,
    benchmark: Optional[str] = None,
    k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Find all pass@k result JSON files matching the criteria."""
    results = []

    # Pattern: {method}_{benchmark}_{exp_id}_{checkpoint}_k{k}_{timestamp}.json
    pattern = os.path.join(RESULTS_DIR, "*.json")

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)

        # Skip non-pass@k files
        if "_k" not in filename:
            continue

        # Parse filename
        match = re.match(
            r"(grpo|rltt)_(\w+)_(\d+)_(step_\d+)_k(\d+)_(\d{8}_\d{6})\.json",
            filename
        )
        if not match:
            continue

        file_method, file_benchmark, exp_id, checkpoint, file_k, timestamp = match.groups()
        file_k = int(file_k)

        # Apply filters
        if method and file_method != method:
            continue
        if benchmark and file_benchmark != benchmark:
            continue
        if k and file_k != k:
            continue

        # Load the result
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            results.append({
                "filepath": filepath,
                "filename": filename,
                "method": file_method,
                "benchmark": file_benchmark,
                "exp_id": exp_id,
                "checkpoint": checkpoint,
                "k": file_k,
                "timestamp": timestamp,
                "data": data,
            })
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


def aggregate_by_benchmark_and_k(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Aggregate results by benchmark and k value.

    Returns nested dict: benchmark -> k -> {grpo: data, rltt: data}
    """
    aggregated = {}

    for result in results:
        benchmark = result["benchmark"]
        k = result["k"]
        method = result["method"]

        if benchmark not in aggregated:
            aggregated[benchmark] = {}
        if k not in aggregated[benchmark]:
            aggregated[benchmark][k] = {}

        # If we already have a result for this method/benchmark/k, keep the latest
        existing = aggregated[benchmark][k].get(method)
        if existing is None or result["timestamp"] > existing["timestamp"]:
            aggregated[benchmark][k][method] = result

    return aggregated


def write_comparison_csv(aggregated: Dict[str, Dict[int, Dict[str, Any]]], output_path: str):
    """Write a CSV comparing GRPO vs RLTT pass@k across all benchmarks and k values."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "benchmark",
            "k",
            "grpo_pass_at_k",
            "grpo_pass_count",
            "grpo_total",
            "grpo_mean_correct",
            "rltt_pass_at_k",
            "rltt_pass_count",
            "rltt_total",
            "rltt_mean_correct",
            "diff_pass_at_k",
            "grpo_timestamp",
            "rltt_timestamp",
        ])

        # Sort by benchmark then k
        for benchmark in sorted(aggregated.keys()):
            for k in sorted(aggregated[benchmark].keys()):
                methods = aggregated[benchmark][k]

                grpo = methods.get("grpo", {}).get("data", {}).get("summary", {})
                rltt = methods.get("rltt", {}).get("data", {}).get("summary", {})

                grpo_pass = grpo.get("pass_at_k", 0)
                rltt_pass = rltt.get("pass_at_k", 0)

                writer.writerow([
                    benchmark,
                    k,
                    f"{grpo_pass:.4f}" if grpo_pass else "",
                    grpo.get("pass_count", ""),
                    grpo.get("total", ""),
                    f"{grpo.get('mean_correct_of_k', 0):.2f}" if grpo else "",
                    f"{rltt_pass:.4f}" if rltt_pass else "",
                    rltt.get("pass_count", ""),
                    rltt.get("total", ""),
                    f"{rltt.get('mean_correct_of_k', 0):.2f}" if rltt else "",
                    f"{rltt_pass - grpo_pass:.4f}" if grpo_pass and rltt_pass else "",
                    methods.get("grpo", {}).get("timestamp", ""),
                    methods.get("rltt", {}).get("timestamp", ""),
                ])

    print(f"Comparison CSV written to: {output_path}")


def write_method_csv(results: List[Dict[str, Any]], method: str, output_path: str):
    """Write a CSV for a single method's results."""
    method_results = [r for r in results if r["method"] == method]

    if not method_results:
        print(f"No results found for method: {method}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "benchmark",
            "k",
            "checkpoint",
            "exp_id",
            "pass_at_k",
            "pass_count",
            "total",
            "mean_correct_of_k",
            "std_correct_of_k",
            "temperature",
            "timestamp",
        ])

        # Sort by benchmark, k, then timestamp (newest first)
        method_results.sort(key=lambda x: (x["benchmark"], x["k"], x["timestamp"]), reverse=False)

        for r in method_results:
            summary = r["data"].get("summary", {})
            writer.writerow([
                r["benchmark"],
                r["k"],
                r["checkpoint"],
                r["exp_id"],
                f"{summary.get('pass_at_k', 0):.4f}",
                summary.get("pass_count", ""),
                summary.get("total", ""),
                f"{summary.get('mean_correct_of_k', 0):.2f}",
                f"{summary.get('std_correct_of_k', 0):.2f}",
                summary.get("temperature", ""),
                r["timestamp"],
            ])

    print(f"{method.upper()} CSV written to: {output_path}")


def write_entropy_analysis_json(aggregated: Dict[str, Dict[int, Dict[str, Any]]], output_path: str):
    """
    Write a JSON file analyzing entropy collapse indicators.

    Key indicators of entropy collapse:
    1. Low pass@k even at high k (model stuck on wrong answers)
    2. Low mean_correct_of_k relative to pass@k (few correct samples per problem)
    3. Skewed correct_count_distribution (most problems have 0 or all-k correct)
    """
    analysis = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "benchmarks": {},
    }

    for benchmark in sorted(aggregated.keys()):
        analysis["benchmarks"][benchmark] = {
            "k_values": {},
            "entropy_indicators": {},
        }

        for k in sorted(aggregated[benchmark].keys()):
            methods = aggregated[benchmark][k]
            analysis["benchmarks"][benchmark]["k_values"][k] = {}

            for method in ["grpo", "rltt"]:
                if method not in methods:
                    continue

                data = methods[method]["data"]
                summary = data.get("summary", {})

                pass_at_k = summary.get("pass_at_k", 0)
                mean_correct = summary.get("mean_correct_of_k", 0)
                correct_dist = summary.get("correct_count_distribution", {})

                # Compute entropy indicators
                # 1. Diversity ratio: mean_correct / k (how spread out are correct answers)
                diversity_ratio = mean_correct / k if k > 0 else 0

                # 2. All-or-nothing ratio: (problems with 0 or all correct) / total
                total = summary.get("total", 1)
                zero_correct = correct_dist.get("0", correct_dist.get(0, 0))
                all_correct = correct_dist.get(str(k), correct_dist.get(k, 0))
                all_or_nothing_ratio = (zero_correct + all_correct) / total if total > 0 else 0

                # 3. Pass@k lift: how much does pass@k improve vs single sample (approximated)
                # If pass@k ~ pass@1, model has low diversity

                analysis["benchmarks"][benchmark]["k_values"][k][method] = {
                    "pass_at_k": pass_at_k,
                    "mean_correct_of_k": mean_correct,
                    "diversity_ratio": diversity_ratio,
                    "all_or_nothing_ratio": all_or_nothing_ratio,
                    "correct_count_distribution": correct_dist,
                    "level_metrics": summary.get("level_metrics", {}),
                }

        # Compute trend across k values for each method
        for method in ["grpo", "rltt"]:
            k_values = sorted(aggregated[benchmark].keys())
            pass_at_k_values = []

            for k in k_values:
                if method in aggregated[benchmark][k]:
                    pass_at_k_values.append(
                        aggregated[benchmark][k][method]["data"]["summary"].get("pass_at_k", 0)
                    )
                else:
                    pass_at_k_values.append(None)

            # Check if pass@k plateaus (sign of entropy collapse)
            analysis["benchmarks"][benchmark]["entropy_indicators"][method] = {
                "k_values": k_values,
                "pass_at_k_trend": pass_at_k_values,
            }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Entropy analysis JSON written to: {output_path}")


def write_level_breakdown_csv(results: List[Dict[str, Any]], output_path: str):
    """Write a CSV with pass@k broken down by difficulty level."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "method",
            "benchmark",
            "k",
            "checkpoint",
            "level",
            "pass_at_k",
            "pass_count",
            "total",
            "mean_correct_of_k",
        ])

        for r in results:
            summary = r["data"].get("summary", {})
            level_metrics = summary.get("level_metrics", {})

            for level, metrics in sorted(level_metrics.items()):
                writer.writerow([
                    r["method"],
                    r["benchmark"],
                    r["k"],
                    r["checkpoint"],
                    level,
                    f"{metrics.get('pass_at_k', 0):.4f}",
                    metrics.get("pass_count", ""),
                    metrics.get("total", ""),
                    f"{metrics.get('mean_correct_of_k', 0):.2f}",
                ])

    print(f"Level breakdown CSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate pass@k results")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter by benchmark (math500, gsm8k, aime24, beyondaime)")
    parser.add_argument("--k", type=int, default=None,
                        help="Filter by k value")
    parser.add_argument("--method", type=str, default=None,
                        choices=["grpo", "rltt"],
                        help="Filter by method")
    args = parser.parse_args()

    print("=" * 60)
    print("Aggregating Pass@k Results")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    # Find all results
    results = find_pass_at_k_results(
        method=args.method,
        benchmark=args.benchmark,
        k=args.k,
    )

    if not results:
        print("No pass@k results found matching criteria.")
        return

    print(f"Found {len(results)} result files")
    print("")

    # Aggregate by benchmark and k
    aggregated = aggregate_by_benchmark_and_k(results)

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write comparison CSV
    comparison_csv = os.path.join(RESULTS_DIR, f"pass_at_k_comparison_{timestamp}.csv")
    write_comparison_csv(aggregated, comparison_csv)

    # Write per-method CSVs
    for method in ["grpo", "rltt"]:
        method_csv = os.path.join(RESULTS_DIR, f"{method}_pass_at_k_{timestamp}.csv")
        write_method_csv(results, method, method_csv)

    # Write entropy analysis JSON
    entropy_json = os.path.join(RESULTS_DIR, f"entropy_analysis_{timestamp}.json")
    write_entropy_analysis_json(aggregated, entropy_json)

    # Write level breakdown CSV
    level_csv = os.path.join(RESULTS_DIR, f"pass_at_k_by_level_{timestamp}.csv")
    write_level_breakdown_csv(results, level_csv)

    print("")
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    # Print quick summary
    for benchmark in sorted(aggregated.keys()):
        print(f"\n{benchmark}:")
        for k in sorted(aggregated[benchmark].keys()):
            grpo_data = aggregated[benchmark][k].get("grpo", {}).get("data", {}).get("summary", {})
            rltt_data = aggregated[benchmark][k].get("rltt", {}).get("data", {}).get("summary", {})

            grpo_pass = grpo_data.get("pass_at_k", 0) * 100
            rltt_pass = rltt_data.get("pass_at_k", 0) * 100

            grpo_str = f"GRPO: {grpo_pass:.1f}%" if grpo_data else "GRPO: N/A"
            rltt_str = f"RLTT: {rltt_pass:.1f}%" if rltt_data else "RLTT: N/A"

            print(f"  k={k}: {grpo_str}, {rltt_str}")


if __name__ == "__main__":
    main()
