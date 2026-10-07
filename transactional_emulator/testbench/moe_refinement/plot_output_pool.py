#!/usr/bin/env python3
"""Plot the predeclared work-conserving comparison from consolidated points."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.points.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["phase"] == "WC"]
    cases = [("qwen_full_decode_b8", "B8: 17 active experts"),
             ("qwen_full_decode_b32", "B32: 23 active experts")]
    controls = [("single", "legacy_n3", "Single N3"),
                ("single", "pool_q32", "Single + output pool"),
                ("homogeneous", "pool_q32", "Equal cores + output pool"),
                ("heterogeneous", "pool_q32", "Large/small + output pool")]
    values = []
    for case, _ in cases:
        times = []
        for organization, mode, _ in controls:
            match = [r for r in rows if (r["window"], r["organization"], r["mode"])
                     == (case, organization, mode)]
            if len(match) != 1:
                raise ValueError("missing or duplicate frozen comparison point")
            times.append(float(match[0]["total_ms"]))
        values.append(times)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    colors = ["#26394a", "#7356a8", "#337daf", "#c36e2e"]
    for ax, (_, title), times in zip(axes, cases, values):
        ax.barh(range(4), times, height=0.56, color=colors, zorder=3)
        for y, value in enumerate(times):
            ax.text(value + 0.018, y, f"{value:.3f}", va="center", fontsize=11)
        ax.set_title(title, loc="left", pad=13, fontweight="bold")
        ax.set_xlabel("Simulated operator time (ms) — lower is better", fontsize=10)
        ax.set_yticks(range(4), [c[2] for c in controls])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.grid(axis="x", alpha=0.18, zorder=0)
        ax.set_xlim(0, max(max(v) for v in values) * 1.17)
    axes[0].invert_yaxis()
    fig.suptitle("Normal MoE: single and paired cores under matched budgets", x=0.02,
                 ha="left", y=0.98, fontsize=15, fontweight="bold")
    fig.text(0.02, 0.89, "Rust + Ramulator · fixed expert bank · equal aggregate modeled resources",
             fontsize=11, color="#475569")
    fig.text(0.02, 0.02, "Pool: Q32 output records, 3 weight slots, 2 operand stages per core. "
             "Synthetic values + archived routes; no full-model or silicon claim.", fontsize=9,
             color="#475569")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.76, bottom=0.19, wspace=0.15)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output_dir / ("work_conserving_latency." + extension), dpi=180,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
