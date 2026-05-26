#!/usr/bin/env python3
"""
Statistical comparison of GSNR between GRPO and RLTT across benchmarks.
Runs paired t-tests and Wilcoxon signed-rank tests (non-parametric alternative).
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

GSNR_DIR = Path("/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/gsnr_results")
BENCHMARKS = ["math500", "gsm8k", "aime24", "beyondaime"]

def load_gsnr_values(method: str, benchmark: str) -> np.ndarray:
    """Load raw GSNR values for a method/benchmark combination."""
    method_dir = GSNR_DIR / method

    # Find the results file
    for subdir in method_dir.iterdir():
        if benchmark in subdir.name and subdir.is_dir():
            results_file = subdir / "gsnr_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                return np.array(data["summary"]["raw_gsnr_values"])

    raise FileNotFoundError(f"No GSNR results found for {method}/{benchmark}")

def run_statistical_tests(grpo_values: np.ndarray, rltt_values: np.ndarray):
    """Run statistical tests comparing GRPO vs RLTT."""
    results = {}

    # Basic statistics
    results["grpo_mean"] = float(np.mean(grpo_values))
    results["grpo_std"] = float(np.std(grpo_values))
    results["rltt_mean"] = float(np.mean(rltt_values))
    results["rltt_std"] = float(np.std(rltt_values))
    results["n_samples"] = len(grpo_values)

    # Difference
    results["mean_diff"] = results["rltt_mean"] - results["grpo_mean"]
    results["rltt_higher"] = results["rltt_mean"] > results["grpo_mean"]

    # Paired t-test (parametric)
    t_stat, t_pvalue = stats.ttest_rel(rltt_values, grpo_values)
    results["paired_ttest"] = {
        "t_statistic": float(t_stat),
        "p_value": float(t_pvalue),
        "significant_0.05": t_pvalue < 0.05,
        "significant_0.01": t_pvalue < 0.01,
        "significant_0.001": t_pvalue < 0.001,
    }

    # Wilcoxon signed-rank test (non-parametric, good for non-normal data)
    # Only on non-zero differences
    diffs = rltt_values - grpo_values
    non_zero_diffs = diffs[diffs != 0]
    if len(non_zero_diffs) > 10:
        w_stat, w_pvalue = stats.wilcoxon(non_zero_diffs)
        results["wilcoxon"] = {
            "w_statistic": float(w_stat),
            "p_value": float(w_pvalue),
            "n_nonzero_pairs": len(non_zero_diffs),
            "significant_0.05": w_pvalue < 0.05,
            "significant_0.01": w_pvalue < 0.01,
        }
    else:
        results["wilcoxon"] = {"note": "Too few non-zero differences for Wilcoxon test"}

    # Effect size (Cohen's d for paired samples)
    diff_std = np.std(rltt_values - grpo_values)
    if diff_std > 0:
        cohens_d = results["mean_diff"] / diff_std
        results["effect_size"] = {
            "cohens_d": float(cohens_d),
            "interpretation": interpret_cohens_d(cohens_d)
        }

    # Normality test on differences (Shapiro-Wilk)
    if len(grpo_values) <= 5000:
        _, shapiro_p = stats.shapiro(rltt_values - grpo_values)
        results["normality_test"] = {
            "shapiro_p": float(shapiro_p),
            "is_normal_0.05": shapiro_p > 0.05
        }

    return results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def main():
    print("=" * 70)
    print("GSNR Statistical Comparison: GRPO vs RLTT")
    print("=" * 70)

    all_results = {}

    for benchmark in BENCHMARKS:
        print(f"\n{'='*70}")
        print(f"Benchmark: {benchmark.upper()}")
        print("=" * 70)

        try:
            grpo_values = load_gsnr_values("grpo", benchmark)
            rltt_values = load_gsnr_values("rltt", benchmark)

            if len(grpo_values) != len(rltt_values):
                print(f"  Warning: Mismatched sample sizes ({len(grpo_values)} vs {len(rltt_values)})")
                min_len = min(len(grpo_values), len(rltt_values))
                grpo_values = grpo_values[:min_len]
                rltt_values = rltt_values[:min_len]

            results = run_statistical_tests(grpo_values, rltt_values)
            all_results[benchmark] = results

            # Print results
            print(f"\n  Descriptive Statistics:")
            print(f"    GRPO:  mean={results['grpo_mean']:.6f}, std={results['grpo_std']:.6f}")
            print(f"    RLTT:  mean={results['rltt_mean']:.6f}, std={results['rltt_std']:.6f}")
            print(f"    Diff:  {results['mean_diff']:.6f} ({'RLTT higher' if results['rltt_higher'] else 'GRPO higher'})")
            print(f"    N:     {results['n_samples']}")

            print(f"\n  Paired t-test:")
            t = results["paired_ttest"]
            print(f"    t-statistic: {t['t_statistic']:.4f}")
            print(f"    p-value:     {t['p_value']:.2e}")
            sig_level = "***" if t['significant_0.001'] else "**" if t['significant_0.01'] else "*" if t['significant_0.05'] else "ns"
            print(f"    Significance: {sig_level} (*** p<0.001, ** p<0.01, * p<0.05, ns=not significant)")

            if "wilcoxon" in results and "p_value" in results["wilcoxon"]:
                print(f"\n  Wilcoxon signed-rank test (non-parametric):")
                w = results["wilcoxon"]
                print(f"    W-statistic: {w['w_statistic']:.4f}")
                print(f"    p-value:     {w['p_value']:.2e}")
                sig_level = "**" if w['significant_0.01'] else "*" if w['significant_0.05'] else "ns"
                print(f"    Significance: {sig_level}")
                print(f"    N non-zero pairs: {w['n_nonzero_pairs']}")

            if "effect_size" in results:
                print(f"\n  Effect Size:")
                print(f"    Cohen's d: {results['effect_size']['cohens_d']:.4f} ({results['effect_size']['interpretation']})")

            if "normality_test" in results:
                print(f"\n  Normality of Differences (Shapiro-Wilk):")
                n = results["normality_test"]
                print(f"    p-value: {n['shapiro_p']:.2e}")
                print(f"    Normal (p>0.05): {'Yes' if n['is_normal_0.05'] else 'No'}")
                if not n['is_normal_0.05']:
                    print(f"    Note: Data is non-normal, prefer Wilcoxon test results")

        except FileNotFoundError as e:
            print(f"  Error: {e}")
            all_results[benchmark] = {"error": str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Benchmark':<15} {'GRPO Mean':<12} {'RLTT Mean':<12} {'Diff':<12} {'t-test p':<12} {'Wilcoxon p':<12} {'Sig':<6}")
    print("-" * 80)

    for benchmark in BENCHMARKS:
        if benchmark in all_results and "error" not in all_results[benchmark]:
            r = all_results[benchmark]
            t_p = r["paired_ttest"]["p_value"]
            w_p = r.get("wilcoxon", {}).get("p_value", float('nan'))
            sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
            print(f"{benchmark:<15} {r['grpo_mean']:<12.6f} {r['rltt_mean']:<12.6f} {r['mean_diff']:<+12.6f} {t_p:<12.2e} {w_p:<12.2e} {sig:<6}")

    # Save results to JSON (convert numpy types to python types)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_file = GSNR_DIR / "gsnr_ttest_results.json"
    with open(output_file, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
