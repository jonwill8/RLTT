#!/usr/bin/env python3
"""
Statistical Significance Analysis Script (T-Test) for Non-Math Benchmarks

This script performs paired t-tests to determine if differences between
GRPO and RLTT results are statistically significant on non-math benchmarks.

Following the methodology from HRPO paper (Table 9):
- Uses paired t-test with p < 0.05 threshold for significance
- Reports mean, std, and p-values for each benchmark

Usage:
    python run_ttest_analysis.py \
        --grpo_csv grpo_arc_c_*.csv \
        --rltt_csv rltt_arc_c_*.csv

    # Or auto-detect latest results for a benchmark:
    python run_ttest_analysis.py --benchmark arc_c

    # Run analysis for all benchmarks:
    python run_ttest_analysis.py --all
"""
import os
import sys
import argparse
import glob
import json
import csv
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from scipy import stats


def load_csv_results(csv_path: str) -> List[float]:
    """Load accuracy values from a repeated evaluation CSV file."""
    accuracies = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accuracies.append(float(row["accuracy"]))
    return accuracies


def find_latest_csv(results_dir: str, method: str, benchmark: str) -> Optional[str]:
    """Find the most recent CSV file for a given method and benchmark."""
    pattern = os.path.join(results_dir, f"{method}_{benchmark}_*.csv")
    files = glob.glob(pattern)

    # Filter out summary files
    files = [f for f in files if not f.endswith("_summary.json")]

    if not files:
        return None

    # Sort by modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def paired_ttest(grpo_accuracies: List[float], rltt_accuracies: List[float]) -> Dict:
    """Perform paired t-test between GRPO and RLTT results.

    Returns statistics including p-value, t-statistic, and effect size.
    """
    grpo = np.array(grpo_accuracies)
    rltt = np.array(rltt_accuracies)

    # Basic statistics
    grpo_mean = np.mean(grpo)
    grpo_std = np.std(grpo, ddof=1)
    rltt_mean = np.mean(rltt)
    rltt_std = np.std(rltt, ddof=1)

    # Difference statistics
    diff = rltt - grpo
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)

    # Paired t-test
    # H0: mean(RLTT) = mean(GRPO)
    # H1: mean(RLTT) != mean(GRPO) (two-tailed)
    t_stat, p_value = stats.ttest_rel(rltt, grpo)

    # One-tailed p-value (RLTT > GRPO)
    p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Effect size (Cohen's d for paired samples)
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0

    # Confidence interval for the difference (95%)
    n = len(grpo)
    se = diff_std / np.sqrt(n)
    ci_lower = diff_mean - 1.96 * se
    ci_upper = diff_mean + 1.96 * se

    return {
        "grpo_mean": grpo_mean,
        "grpo_std": grpo_std,
        "rltt_mean": rltt_mean,
        "rltt_std": rltt_std,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "t_statistic": t_stat,
        "p_value": p_value,
        "p_value_one_tailed": p_value_one_tailed,
        "cohens_d": cohens_d,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_samples": n,
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
        "rltt_better": rltt_mean > grpo_mean,
    }


def format_results(benchmark: str, results: Dict, grpo_csv: str, rltt_csv: str) -> str:
    """Format t-test results for display."""
    output = []
    output.append("=" * 70)
    output.append(f"Statistical Significance Analysis: {benchmark.upper()}")
    output.append("=" * 70)
    output.append("")
    output.append("Data Sources:")
    output.append(f"  GRPO: {os.path.basename(grpo_csv)}")
    output.append(f"  RLTT: {os.path.basename(rltt_csv)}")
    output.append(f"  N samples: {results['n_samples']}")
    output.append("")
    output.append("Performance:")
    output.append(f"  GRPO:  {results['grpo_mean']*100:.2f}% ± {results['grpo_std']*100:.2f}%")
    output.append(f"  RLTT:  {results['rltt_mean']*100:.2f}% ± {results['rltt_std']*100:.2f}%")
    output.append(f"  Diff:  {results['diff_mean']*100:+.2f}% ± {results['diff_std']*100:.2f}%")
    output.append("")
    output.append("Paired T-Test Results:")
    output.append(f"  t-statistic: {results['t_statistic']:.4f}")
    output.append(f"  p-value (two-tailed): {results['p_value']:.6f}")
    output.append(f"  p-value (one-tailed, RLTT > GRPO): {results['p_value_one_tailed']:.6f}")
    output.append(f"  Cohen's d (effect size): {results['cohens_d']:.4f}")
    output.append(f"  95% CI for difference: [{results['ci_95_lower']*100:.2f}%, {results['ci_95_upper']*100:.2f}%]")
    output.append("")
    output.append("Conclusion:")

    if results['significant_0.01']:
        sig_level = "p < 0.01 (highly significant)"
    elif results['significant_0.05']:
        sig_level = "p < 0.05 (significant)"
    else:
        sig_level = "p >= 0.05 (not significant)"

    if results['rltt_better']:
        direction = "RLTT outperforms GRPO"
    else:
        direction = "GRPO outperforms RLTT"

    output.append(f"  Direction: {direction}")
    output.append(f"  Statistical Significance: {sig_level}")

    if results['significant_0.05']:
        output.append(f"  The difference IS statistically significant at alpha=0.05")
    else:
        output.append(f"  The difference is NOT statistically significant at alpha=0.05")

    # Effect size interpretation
    d = abs(results['cohens_d'])
    if d < 0.2:
        effect = "negligible"
    elif d < 0.5:
        effect = "small"
    elif d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    output.append(f"  Effect size: {effect} (|d| = {d:.3f})")

    output.append("")
    return "\n".join(output)


def run_analysis_for_benchmark(
    results_dir: str,
    benchmark: str,
    grpo_csv: Optional[str] = None,
    rltt_csv: Optional[str] = None,
) -> Optional[Dict]:
    """Run t-test analysis for a specific benchmark."""

    # Find CSV files if not provided
    if grpo_csv is None:
        grpo_csv = find_latest_csv(results_dir, "grpo", benchmark)
    if rltt_csv is None:
        rltt_csv = find_latest_csv(results_dir, "rltt", benchmark)

    if grpo_csv is None:
        print(f"Warning: No GRPO results found for {benchmark}")
        return None
    if rltt_csv is None:
        print(f"Warning: No RLTT results found for {benchmark}")
        return None

    # Load data
    grpo_accuracies = load_csv_results(grpo_csv)
    rltt_accuracies = load_csv_results(rltt_csv)

    if len(grpo_accuracies) != len(rltt_accuracies):
        print(f"Warning: Mismatched sample sizes for {benchmark}")
        print(f"  GRPO: {len(grpo_accuracies)}, RLTT: {len(rltt_accuracies)}")
        # Use minimum length
        min_len = min(len(grpo_accuracies), len(rltt_accuracies))
        grpo_accuracies = grpo_accuracies[:min_len]
        rltt_accuracies = rltt_accuracies[:min_len]

    # Run t-test
    results = paired_ttest(grpo_accuracies, rltt_accuracies)
    results["benchmark"] = benchmark
    results["grpo_csv"] = grpo_csv
    results["rltt_csv"] = rltt_csv

    # Print formatted results
    print(format_results(benchmark, results, grpo_csv, rltt_csv))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run t-test analysis comparing GRPO vs RLTT results on non-math benchmarks"
    )
    parser.add_argument("--grpo_csv", type=str, help="Path to GRPO results CSV")
    parser.add_argument("--rltt_csv", type=str, help="Path to RLTT results CSV")
    parser.add_argument("--benchmark", type=str,
                        choices=["arc_c", "mmlu_st", "gpqa"],
                        help="Benchmark to analyze (auto-detects latest results)")
    parser.add_argument("--all", action="store_true",
                        help="Run analysis for all benchmarks with available results")
    parser.add_argument("--results_dir", type=str,
                        default="/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/repeated_eval_results_non-math",
                        help="Directory containing repeated evaluation results")
    parser.add_argument("--output_json", type=str,
                        help="Save results to JSON file")
    args = parser.parse_args()

    all_results = {}

    if args.grpo_csv and args.rltt_csv:
        # Explicit CSV files provided
        benchmark = args.benchmark or "unknown"
        result = run_analysis_for_benchmark(
            args.results_dir, benchmark, args.grpo_csv, args.rltt_csv
        )
        if result:
            all_results[benchmark] = result

    elif args.all:
        # Run for all benchmarks
        benchmarks = ["arc_c", "mmlu_st", "gpqa"]
        for benchmark in benchmarks:
            result = run_analysis_for_benchmark(args.results_dir, benchmark)
            if result:
                all_results[benchmark] = result

    elif args.benchmark:
        # Run for specific benchmark
        result = run_analysis_for_benchmark(args.results_dir, args.benchmark)
        if result:
            all_results[args.benchmark] = result

    else:
        print("Error: Must specify --benchmark, --all, or both --grpo_csv and --rltt_csv")
        parser.print_help()
        sys.exit(1)

    # Print summary table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY TABLE (Following HRPO Paper Table 9 Format)")
        print("=" * 70)
        print("")
        print(f"{'Benchmark':<15} {'GRPO':<20} {'RLTT':<20} {'p-value':<12} {'Sig.':<8}")
        print("-" * 75)

        for benchmark, result in all_results.items():
            grpo_str = f"{result['grpo_mean']*100:.2f}% +/- {result['grpo_std']*100:.2f}%"
            rltt_str = f"{result['rltt_mean']*100:.2f}% +/- {result['rltt_std']*100:.2f}%"
            p_str = f"{result['p_value']:.4f}"
            sig_str = "Yes" if result['significant_0.05'] else ""

            # Bold the winner
            if result['rltt_better'] and result['significant_0.05']:
                rltt_str = f"**{rltt_str}**"
            elif not result['rltt_better'] and result['significant_0.05']:
                grpo_str = f"**{grpo_str}**"

            print(f"{benchmark:<15} {grpo_str:<20} {rltt_str:<20} {p_str:<12} {sig_str:<8}")

        print("")
        print("Note: ** indicates statistically significant improvement (p < 0.05)")
        print("")

    # Save to JSON if requested
    if args.output_json:
        output_path = args.output_json
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.results_dir, f"ttest_analysis_{timestamp}.json")

    with open(output_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {}
            for key, val in v.items():
                if isinstance(val, (np.floating, np.integer)):
                    serializable_results[k][key] = float(val)
                elif isinstance(val, np.bool_):
                    serializable_results[k][key] = bool(val)
                elif isinstance(val, np.ndarray):
                    serializable_results[k][key] = val.tolist()
                else:
                    serializable_results[k][key] = val
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
