#!/usr/bin/env python3
"""
Aggregate GSNR Results Across Methods and Benchmarks

This script aggregates GSNR evaluation results from both RLTT and GRPO experiments
across all benchmarks (MATH-500, GSM8K, AIME24, BeyondAIME) and generates:
1. A unified CSV file with all results
2. A comparison JSON with statistical summaries
3. A human-readable summary report

Usage:
    python aggregate_gsnr_results.py

    # Specify custom directories:
    python aggregate_gsnr_results.py \
        --rltt_dir /path/to/gsnr_results/rltt \
        --grpo_dir /path/to/gsnr_results/grpo \
        --output_dir /path/to/gsnr_results
"""
import os
import sys
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import csv

# Default paths
DEFAULT_GSNR_ROOT = "/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results"
DEFAULT_RLTT_DIR = os.path.join(DEFAULT_GSNR_ROOT, "rltt")
DEFAULT_GRPO_DIR = os.path.join(DEFAULT_GSNR_ROOT, "grpo")

BENCHMARKS = ["math500", "gsm8k", "aime24", "beyondaime"]


def find_latest_result(results_dir: str, benchmark: str) -> Optional[Dict[str, Any]]:
    """
    Find the latest GSNR result for a given benchmark.

    Args:
        results_dir: Directory containing GSNR results (rltt or grpo)
        benchmark: Benchmark name

    Returns:
        Dict with result data or None if not found
    """
    # Pattern: gsnr_{benchmark}_{exp_id}_{checkpoint}_{timestamp}/gsnr_summary.json
    pattern = os.path.join(results_dir, f"gsnr_{benchmark}_*", "gsnr_summary.json")
    matches = glob.glob(pattern)

    if not matches:
        return None

    # Sort by modification time to get latest
    matches.sort(key=os.path.getmtime, reverse=True)
    latest_file = matches[0]

    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
        data["result_file"] = latest_file
        data["result_dir"] = os.path.dirname(latest_file)
        return data
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")
        return None


def aggregate_results(rltt_dir: str, grpo_dir: str) -> Dict[str, Any]:
    """
    Aggregate GSNR results from RLTT and GRPO directories.

    Args:
        rltt_dir: Path to RLTT GSNR results
        grpo_dir: Path to GRPO GSNR results

    Returns:
        Dict with aggregated results
    """
    results = {
        "generated_at": datetime.now().isoformat(),
        "rltt": {},
        "grpo": {},
        "comparison": {},
    }

    for benchmark in BENCHMARKS:
        # Get RLTT result
        rltt_result = find_latest_result(rltt_dir, benchmark)
        if rltt_result:
            results["rltt"][benchmark] = {
                "overall_gsnr": rltt_result.get("overall_gsnr"),
                "std_log_gsnr": rltt_result.get("std_log_gsnr"),
                "mean_raw_gsnr": rltt_result.get("mean_raw_gsnr"),
                "std_raw_gsnr": rltt_result.get("std_raw_gsnr"),
                "num_valid_prompts": rltt_result.get("num_valid_prompts"),
                "num_total_prompts": rltt_result.get("num_total_prompts"),
                "checkpoint": rltt_result.get("checkpoint"),
                "experiment_id": rltt_result.get("experiment_id"),
                "timestamp": rltt_result.get("timestamp"),
                "num_rollouts": rltt_result.get("num_rollouts"),
            }

        # Get GRPO result
        grpo_result = find_latest_result(grpo_dir, benchmark)
        if grpo_result:
            results["grpo"][benchmark] = {
                "overall_gsnr": grpo_result.get("overall_gsnr"),
                "std_log_gsnr": grpo_result.get("std_log_gsnr"),
                "mean_raw_gsnr": grpo_result.get("mean_raw_gsnr"),
                "std_raw_gsnr": grpo_result.get("std_raw_gsnr"),
                "num_valid_prompts": grpo_result.get("num_valid_prompts"),
                "num_total_prompts": grpo_result.get("num_total_prompts"),
                "checkpoint": grpo_result.get("checkpoint"),
                "experiment_id": grpo_result.get("experiment_id"),
                "timestamp": grpo_result.get("timestamp"),
                "num_rollouts": grpo_result.get("num_rollouts"),
            }

        # Compute comparison if both exist
        if benchmark in results["rltt"] and benchmark in results["grpo"]:
            rltt_gsnr = results["rltt"][benchmark]["overall_gsnr"]
            grpo_gsnr = results["grpo"][benchmark]["overall_gsnr"]

            if rltt_gsnr is not None and grpo_gsnr is not None:
                diff = rltt_gsnr - grpo_gsnr
                results["comparison"][benchmark] = {
                    "rltt_gsnr": rltt_gsnr,
                    "grpo_gsnr": grpo_gsnr,
                    "diff_gsnr": diff,
                    "rltt_better": diff > 0,
                    "improvement_pct": (diff / abs(grpo_gsnr)) * 100 if grpo_gsnr != 0 else float('inf'),
                }

    return results


def generate_csv(results: Dict[str, Any], output_path: str):
    """
    Generate a CSV file with all GSNR results.

    Args:
        results: Aggregated results dict
        output_path: Output CSV file path
    """
    rows = []

    # Header row
    header = [
        "benchmark",
        "method",
        "overall_gsnr",
        "std_log_gsnr",
        "mean_raw_gsnr",
        "std_raw_gsnr",
        "num_valid_prompts",
        "num_total_prompts",
        "checkpoint",
        "experiment_id",
        "num_rollouts",
        "timestamp",
    ]

    # Add RLTT results
    for benchmark in BENCHMARKS:
        if benchmark in results["rltt"]:
            data = results["rltt"][benchmark]
            rows.append([
                benchmark,
                "rltt",
                data.get("overall_gsnr", ""),
                data.get("std_log_gsnr", ""),
                data.get("mean_raw_gsnr", ""),
                data.get("std_raw_gsnr", ""),
                data.get("num_valid_prompts", ""),
                data.get("num_total_prompts", ""),
                data.get("checkpoint", ""),
                data.get("experiment_id", ""),
                data.get("num_rollouts", ""),
                data.get("timestamp", ""),
            ])

    # Add GRPO results
    for benchmark in BENCHMARKS:
        if benchmark in results["grpo"]:
            data = results["grpo"][benchmark]
            rows.append([
                benchmark,
                "grpo",
                data.get("overall_gsnr", ""),
                data.get("std_log_gsnr", ""),
                data.get("mean_raw_gsnr", ""),
                data.get("std_raw_gsnr", ""),
                data.get("num_valid_prompts", ""),
                data.get("num_total_prompts", ""),
                data.get("checkpoint", ""),
                data.get("experiment_id", ""),
                data.get("num_rollouts", ""),
                data.get("timestamp", ""),
            ])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_comparison_csv(results: Dict[str, Any], output_path: str):
    """
    Generate a side-by-side comparison CSV.

    Args:
        results: Aggregated results dict
        output_path: Output CSV file path
    """
    header = [
        "benchmark",
        "rltt_gsnr",
        "grpo_gsnr",
        "diff_gsnr",
        "rltt_better",
        "improvement_pct",
        "rltt_std",
        "grpo_std",
    ]

    rows = []
    for benchmark in BENCHMARKS:
        row = [benchmark]

        if benchmark in results["comparison"]:
            comp = results["comparison"][benchmark]
            row.extend([
                comp.get("rltt_gsnr", ""),
                comp.get("grpo_gsnr", ""),
                comp.get("diff_gsnr", ""),
                comp.get("rltt_better", ""),
                comp.get("improvement_pct", ""),
            ])
        else:
            row.extend(["", "", "", "", ""])

        # Add std values
        rltt_std = results["rltt"].get(benchmark, {}).get("std_log_gsnr", "")
        grpo_std = results["grpo"].get(benchmark, {}).get("std_log_gsnr", "")
        row.extend([rltt_std, grpo_std])

        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_summary_report(results: Dict[str, Any], output_path: str):
    """
    Generate a human-readable summary report.

    Args:
        results: Aggregated results dict
        output_path: Output text file path
    """
    lines = []
    lines.append("=" * 70)
    lines.append("GSNR Evaluation Summary Report")
    lines.append("=" * 70)
    lines.append(f"Generated at: {results['generated_at']}")
    lines.append("")

    # Overall comparison
    lines.append("-" * 70)
    lines.append("RLTT vs GRPO Comparison (Overall GSNR = mean log(GSNR))")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Benchmark':<15} {'RLTT GSNR':>12} {'GRPO GSNR':>12} {'Diff':>10} {'Winner':>10}")
    lines.append("-" * 60)

    for benchmark in BENCHMARKS:
        if benchmark in results["comparison"]:
            comp = results["comparison"][benchmark]
            rltt_gsnr = comp["rltt_gsnr"]
            grpo_gsnr = comp["grpo_gsnr"]
            diff = comp["diff_gsnr"]
            winner = "RLTT" if comp["rltt_better"] else "GRPO"
            lines.append(f"{benchmark:<15} {rltt_gsnr:>12.4f} {grpo_gsnr:>12.4f} {diff:>10.4f} {winner:>10}")
        else:
            lines.append(f"{benchmark:<15} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10}")

    lines.append("")

    # Detailed results
    lines.append("-" * 70)
    lines.append("Detailed Results by Method")
    lines.append("-" * 70)

    for method in ["rltt", "grpo"]:
        lines.append("")
        lines.append(f"[{method.upper()}]")
        lines.append("")

        for benchmark in BENCHMARKS:
            if benchmark in results[method]:
                data = results[method][benchmark]
                lines.append(f"  {benchmark}:")
                lines.append(f"    Overall GSNR (mean log): {data.get('overall_gsnr', 'N/A'):.4f}" if data.get('overall_gsnr') else f"    Overall GSNR: N/A")
                lines.append(f"    Std log(GSNR): {data.get('std_log_gsnr', 'N/A'):.4f}" if data.get('std_log_gsnr') else f"    Std log(GSNR): N/A")
                lines.append(f"    Mean raw GSNR: {data.get('mean_raw_gsnr', 'N/A'):.6f}" if data.get('mean_raw_gsnr') else f"    Mean raw GSNR: N/A")
                lines.append(f"    Valid prompts: {data.get('num_valid_prompts', 'N/A')}/{data.get('num_total_prompts', 'N/A')}")
                lines.append(f"    Checkpoint: {data.get('checkpoint', 'N/A')}")
                lines.append(f"    Experiment ID: {data.get('experiment_id', 'N/A')}")
                lines.append("")
            else:
                lines.append(f"  {benchmark}: No results available")
                lines.append("")

    # Interpretation guide
    lines.append("-" * 70)
    lines.append("Interpretation Guide")
    lines.append("-" * 70)
    lines.append("")
    lines.append("GSNR (Gradient Signal-to-Noise Ratio) measures learning signal quality:")
    lines.append("  - Higher GSNR = more consistent gradient signal across rollouts")
    lines.append("  - Lower GSNR = noisier gradients, harder to learn from")
    lines.append("")
    lines.append("The metric is computed as:")
    lines.append("  GSNR_p = ||mean_gradient||^2 / variance_gradient")
    lines.append("  Overall GSNR = mean(log(GSNR_p)) across prompts")
    lines.append("")
    lines.append("A higher GSNR indicates that RLTT/GRPO provides a more reliable")
    lines.append("learning signal, which typically correlates with better training.")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GSNR results across methods and benchmarks"
    )
    parser.add_argument("--rltt_dir", type=str, default=DEFAULT_RLTT_DIR,
                        help="Path to RLTT GSNR results directory")
    parser.add_argument("--grpo_dir", type=str, default=DEFAULT_GRPO_DIR,
                        help="Path to GRPO GSNR results directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_GSNR_ROOT,
                        help="Output directory for aggregated results")
    args = parser.parse_args()

    print("=" * 70)
    print("GSNR Results Aggregation")
    print("=" * 70)
    print(f"RLTT results dir: {args.rltt_dir}")
    print(f"GRPO results dir: {args.grpo_dir}")
    print(f"Output dir: {args.output_dir}")
    print("")

    # Aggregate results
    print("Aggregating results...")
    results = aggregate_results(args.rltt_dir, args.grpo_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate CSV files
    csv_path = os.path.join(args.output_dir, f"gsnr_all_results_{timestamp}.csv")
    generate_csv(results, csv_path)
    print(f"Generated CSV: {csv_path}")

    comparison_csv_path = os.path.join(args.output_dir, f"gsnr_comparison_{timestamp}.csv")
    generate_comparison_csv(results, comparison_csv_path)
    print(f"Generated comparison CSV: {comparison_csv_path}")

    # Generate JSON
    json_path = os.path.join(args.output_dir, f"gsnr_aggregated_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Generated JSON: {json_path}")

    # Generate summary report
    report_path = os.path.join(args.output_dir, f"gsnr_summary_report_{timestamp}.txt")
    generate_summary_report(results, report_path)
    print(f"Generated summary report: {report_path}")

    # Also create a 'latest' symlink/copy for easy access
    latest_csv = os.path.join(args.output_dir, "gsnr_all_results_latest.csv")
    latest_comparison = os.path.join(args.output_dir, "gsnr_comparison_latest.csv")
    latest_json = os.path.join(args.output_dir, "gsnr_aggregated_latest.json")
    latest_report = os.path.join(args.output_dir, "gsnr_summary_report_latest.txt")

    import shutil
    shutil.copy(csv_path, latest_csv)
    shutil.copy(comparison_csv_path, latest_comparison)
    shutil.copy(json_path, latest_json)
    shutil.copy(report_path, latest_report)

    print("")
    print("Latest results also available at:")
    print(f"  {latest_csv}")
    print(f"  {latest_comparison}")
    print(f"  {latest_json}")
    print(f"  {latest_report}")

    # Print summary to console
    print("")
    print("=" * 70)
    print("Quick Summary")
    print("=" * 70)
    print(f"{'Benchmark':<15} {'RLTT GSNR':>12} {'GRPO GSNR':>12} {'Diff':>10} {'Winner':>10}")
    print("-" * 60)

    for benchmark in BENCHMARKS:
        if benchmark in results["comparison"]:
            comp = results["comparison"][benchmark]
            rltt_gsnr = comp["rltt_gsnr"]
            grpo_gsnr = comp["grpo_gsnr"]
            diff = comp["diff_gsnr"]
            winner = "RLTT" if comp["rltt_better"] else "GRPO"
            print(f"{benchmark:<15} {rltt_gsnr:>12.4f} {grpo_gsnr:>12.4f} {diff:>10.4f} {winner:>10}")
        else:
            rltt_val = results["rltt"].get(benchmark, {}).get("overall_gsnr", "N/A")
            grpo_val = results["grpo"].get(benchmark, {}).get("overall_gsnr", "N/A")
            if isinstance(rltt_val, float):
                rltt_val = f"{rltt_val:.4f}"
            if isinstance(grpo_val, float):
                grpo_val = f"{grpo_val:.4f}"
            print(f"{benchmark:<15} {str(rltt_val):>12} {str(grpo_val):>12} {'N/A':>10} {'N/A':>10}")

    print("")
    print("Aggregation complete!")


if __name__ == "__main__":
    main()
