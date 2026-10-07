#!/usr/bin/env python3
"""Static scientific figure from measured CSV; all bars use the same budget."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.DictReader((args.results / "mechanism.csv").open()))
    configs = [("6", "pinned_expert", "Single 6\npinned"),
               ("3+3", "pinned_expert", "Uniform 3+3\npinned"),
               ("4+2", "pinned_expert", "Asymmetric 4+2\npinned"),
               ("1+1+1+1+1+1", "tile_stealing", "Uniform 6 x 1\ntile stealing")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    colors = ["#778899", "#4682b4", "#d87c36", "#389875"]
    for ax, workload, title in zip(axes, ["4_2", "3_3"], ["Positive control: expert Me = 4, 2", "Counterexample: expert Me = 3, 3"]):
        values = [int(next(r for r in rows if r["family"] == "sustained" and r["workload"] == workload
                           and r["shape"] == shape and r["ownership"] == policy)["cycles"])
                  for shape, policy, _ in configs]
        bars = ax.bar(range(4), values, color=colors, width=.68)
        ax.bar_label(bars, padding=4)
        ax.set_xticks(range(4), [label for _, _, label in configs], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(values) * 1.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Simulated cycles (lower is better)")
    fig.suptitle("Shape matching helps conditionally; fine uniform engines are a strong challenger", fontsize=12)
    fig.text(.5, .025, "N=512, K=2048 GEMMs | 12,288 physical multipliers | assumed L=25, II=1 | ideal operand interfaces", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(args.results.parent / f"mechanism_controls.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
