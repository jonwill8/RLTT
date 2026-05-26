#!/usr/bin/env python3
"""Build a PDF report comparing rollout FLOP metrics across model runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


RUN_LABELS = {
    "ouro_rltt_aime26_global_step_140_samples30_zeroshot_6179190": "Ouro 2.6B RLTT",
    "ouro_grpo_aime26_global_step_140_samples30_zeroshot_6179189": "Ouro 2.6B GRPO",
    "qwen3_1.7B_aime26_samples30_zeroshot_6189114": "Qwen3 1.7B",
    "qwen3_4B_aime26_samples30_zeroshot_6179201": "Qwen3 4B",
    "deepseek_1.5B_aime26_samples30_zeroshot_6189084": "DeepSeek 1.5B",
    "deepseek_7B_aime26_samples30_zeroshot_6179191": "DeepSeek 7B",
}


@dataclass
class RunStats:
    name: str
    run_dir: str
    samples: int
    total_params: float
    total_flops: np.ndarray
    flops_per_token: np.ndarray
    runtime_sec: np.ndarray
    generated_tokens: np.ndarray

    @property
    def mean_total_flops(self) -> float:
        return float(np.mean(self.total_flops))

    @property
    def median_total_flops(self) -> float:
        return float(np.median(self.total_flops))

    @property
    def p90_total_flops(self) -> float:
        return float(np.percentile(self.total_flops, 90))

    @property
    def mean_flops_per_token(self) -> float:
        return float(np.mean(self.flops_per_token))

    @property
    def mean_runtime_sec(self) -> float:
        return float(np.mean(self.runtime_sec))

    @property
    def mean_generated_tokens(self) -> float:
        return float(np.mean(self.generated_tokens))

    @property
    def mean_tokens_per_sec(self) -> float:
        return float(np.mean(self.generated_tokens / self.runtime_sec))

    @property
    def mean_tflops_per_sec(self) -> float:
        return float(np.mean(self.total_flops / self.runtime_sec) / 1e12)


def _human_b(value: float) -> str:
    return f"{value / 1e9:.2f}B"


def _human_t(value: float) -> str:
    return f"{value / 1e12:.2f}T"


def _human_g(value: float) -> str:
    return f"{value / 1e9:.2f}G"


def read_run(run_file: Path) -> RunStats:
    records: List[dict] = []
    with run_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    total_flops = np.array([r["total_flops"] for r in records], dtype=np.float64)
    flops_per_token = np.array(
        [r["flops_per_generated_token"] for r in records], dtype=np.float64
    )
    runtime_sec = np.array([r["runtime_sec"] for r in records], dtype=np.float64)
    generated_tokens = np.array([r["generated_tokens"] for r in records], dtype=np.float64)
    total_params = float(mean([r["total_params"] for r in records]))

    run_dir = run_file.parent.name
    name = RUN_LABELS.get(run_dir, run_dir)
    return RunStats(
        name=name,
        run_dir=run_dir,
        samples=len(records),
        total_params=total_params,
        total_flops=total_flops,
        flops_per_token=flops_per_token,
        runtime_sec=runtime_sec,
        generated_tokens=generated_tokens,
    )


def draw_cover(pdf: PdfPages, runs: List[RunStats], output: Path) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        "Rollout FLOP Comparison",
        "",
        "Compared models:",
        *[f"- {r.name}" for r in runs],
        "",
        f"Samples per run: {runs[0].samples if runs else 0}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output file: {output.name}",
    ]
    ax.text(
        0.03,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=16,
        family="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_summary_table(pdf: PdfPages, runs: List[RunStats]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    headers = [
        "Model",
        "Params",
        "Mean Total FLOPs",
        "Median Total FLOPs",
        "P90 Total FLOPs",
        "Mean FLOPs/Token",
        "Mean Runtime (s)",
        "Mean Gen Tokens",
        "Tokens/s",
        "TFLOPs/s",
    ]

    rows = []
    for r in runs:
        rows.append(
            [
                r.name,
                _human_b(r.total_params),
                _human_t(r.mean_total_flops),
                _human_t(r.median_total_flops),
                _human_t(r.p90_total_flops),
                _human_g(r.mean_flops_per_token),
                f"{r.mean_runtime_sec:.1f}",
                f"{r.mean_generated_tokens:.0f}",
                f"{r.mean_tokens_per_sec:.2f}",
                f"{r.mean_tflops_per_sec:.2f}",
            ]
        )

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    ax.set_title("Critical Rollout FLOP Metrics (means across 30 samples)", fontsize=16, pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_bar_panels(pdf: PdfPages, runs: List[RunStats]) -> None:
    names = [r.name for r in runs]
    x = np.arange(len(runs))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    metrics = [
        ("Mean Total FLOPs (T)", [r.mean_total_flops / 1e12 for r in runs]),
        ("Mean FLOPs / Generated Token (G)", [r.mean_flops_per_token / 1e9 for r in runs]),
        ("Mean Runtime per Sample (s)", [r.mean_runtime_sec for r in runs]),
        ("Mean Throughput (Tokens/s)", [r.mean_tokens_per_sec for r in runs]),
    ]

    for ax, (title, vals) in zip(axs, metrics):
        ax.bar(x, vals)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Metric-by-Metric Comparison", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_distribution_plots(pdf: PdfPages, runs: List[RunStats]) -> None:
    names = [r.name for r in runs]

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    axs[0].boxplot([r.total_flops / 1e12 for r in runs], tick_labels=names, vert=True)
    axs[0].set_title("Per-Sample Total FLOPs Distribution")
    axs[0].set_ylabel("Total FLOPs (T)")
    axs[0].tick_params(axis="x", rotation=30)
    axs[0].grid(axis="y", alpha=0.3)

    axs[1].boxplot([r.flops_per_token / 1e9 for r in runs], tick_labels=names, vert=True)
    axs[1].set_title("Per-Sample FLOPs/Token Distribution")
    axs[1].set_ylabel("FLOPs / Generated Token (G)")
    axs[1].tick_params(axis="x", rotation=30)
    axs[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution View Across 30 Rollouts", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report(runs_dir: Path, out_pdf: Path) -> None:
    run_files = sorted(runs_dir.glob("*/flop_profile_details.jsonl"))
    chosen = [p for p in run_files if p.parent.name in RUN_LABELS]
    if not chosen:
        raise RuntimeError(f"No known run folders found in {runs_dir}")

    stats = [read_run(p) for p in chosen]
    stats.sort(key=lambda r: r.total_params)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        draw_cover(pdf, stats, out_pdf)
        draw_summary_table(pdf, stats)
        draw_bar_panels(pdf, stats)
        draw_distribution_plots(pdf, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/flop_analysis/runs"),
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path(
            "/scratch/gpfs/OLGARUS/jw4199/looped_models/RLTT/flop_analysis/reports/rollout_flop_comparison.pdf"
        ),
        help="Output PDF path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.runs_dir, args.out_pdf)
    print(f"Wrote report to: {args.out_pdf}")


if __name__ == "__main__":
    main()
