#!/usr/bin/env python3
"""Publication-exportable plots of measured finite-interface simulation cycles."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    args = parser.parse_args()
    ev = args.evaluation
    with (ev / "selection_validation.csv").open() as f:
        selections = [r for r in csv.DictReader(f) if r["interface"] == "packet"]
    with (ev / "paired_sensitivities.csv").open() as f:
        changes = list(csv.DictReader(f))
    with (ev / "control_granularity.csv").open() as f:
        controls = list(csv.DictReader(f))
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = ["#305b91", "#d47a32"]
    x = np.arange(len(selections))
    for i, (key, label) in enumerate([("selection_cycles", "B8 selection"),
                                    ("validation_cycles", "24-token validation")]):
        vals = [int(r[key]) / 1000 for r in selections]
        bars = axes[0].bar(x + (i - .5) * .36, vals, .36, color=colors[i], label=label)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[0].set_xticks(x, ["Single\n6", "Uniform\n3+3", "Uniform\n2+2+2", "Asymmetric\n2+4"])
    axes[0].set_ylabel("Simulated cycles (thousands), lower is better")
    axes[0].set_ylim(0, 125)
    axes[0].set_title("(a) Equal 12,288 multipliers")
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(range(3), [int(r["cycles"])/1000 for r in controls],
                       color=["#8c8c8c", "#6d99bf", "#305b91"])
    axes[1].bar_label(bars, fmt="%.1f", padding=3)
    axes[1].set_xticks(range(3), ["Per\ninvocation", "Same-round\ncohort", "Persistent\ntile cohort"])
    axes[1].set_title("(b) Control granularity: six uniform cores")
    axes[1].set_ylabel("B8 simulated cycles (thousands)")
    axes[1].set_ylim(0, 190)
    axes[1].text(.5, .91, "Same mapping; 34 MiB source weights", transform=axes[1].transAxes,
                 ha="center", fontsize=9)

    variants = ["weight256", "weight4096", "control2ports", "oracle_zero_control"]
    for i, (window, label) in enumerate([("b8_full", "B8"), ("b32_tokens8_31", "Validation")]):
        vals = []
        for variant in variants:
            row = next(r for r in changes if r["name"] == f"winner_main_uniform_{window}__{variant}")
            vals.append(int(row["cycles"]) / int(row["baseline_cycles"]))
        bars = axes[2].bar(np.arange(4) + (i - .5)*.36, vals, .36, color=colors[i], label=label)
        axes[2].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[2].axhline(1, color="black", lw=1, linestyle="--")
    axes[2].set_xticks(range(4), ["Weight BW\n0.25x", "Weight BW\n4x", "Control\n2 ports", "Zero control\nORACLE"])
    axes[2].set_ylabel("Elapsed cycles / same-configuration baseline")
    axes[2].set_title("(c) Fixed 2+2+2 configuration")
    axes[2].set_ylim(0, 2.7)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Finite decoded-BF16 interface model — packet ports; not native HBM or silicon measurements", fontsize=12)
    dest = ev.parent / "figures"
    dest.mkdir(exist_ok=True)
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(dest / ("finite_fabric_results." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
