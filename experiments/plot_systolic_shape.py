#!/usr/bin/env python3
"""
Plot grouped bar chart for systolic shape comparison using matplotlib.

Usage:
    python plot_latex_table.py
    python plot_latex_table.py --output /path/to/output
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# All values are relative to the (256,256) baseline.
DATA = [
    {'shape': '(256,256)', 'power_rel': 1.00, 'area_rel': 1.00, 'ffn_energy_rel': 1.00, 'fa_energy_rel': 1.00},
    {'shape': '(128,512)', 'power_rel': 1.03, 'area_rel': 1.04, 'ffn_energy_rel': 0.81, 'fa_energy_rel': 0.77},
    {'shape': '(64,1024)', 'power_rel': 1.07, 'area_rel': 1.09, 'ffn_energy_rel': 0.65, 'fa_energy_rel': 0.60},
    {'shape': '(32,2048)', 'power_rel': 1.11, 'area_rel': 1.11, 'ffn_energy_rel': 0.52, 'fa_energy_rel': 0.48},
]

BAR_KEYS   = ['power_rel', 'area_rel', 'ffn_energy_rel', 'fa_energy_rel']
BAR_LABELS = ['Power', 'Area', 'FFN Energy', 'FA Energy']
BAR_COLORS = ['#7b3294', '#c2a5cf', '#a6dba0', '#008837']


def plot(data, output_path):
    shapes = [row['shape'] for row in data]
    values = {key: [row[key] for row in data] for key in BAR_KEYS}

    n_groups = len(shapes)
    n_bars   = len(BAR_KEYS)
    x        = np.arange(n_groups)
    total_w  = 0.7
    bar_w    = total_w / n_bars
    offsets  = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_bars)

    fig, ax = plt.subplots(figsize=(9, 4))

    for j, (key, label, color, offset) in enumerate(zip(BAR_KEYS, BAR_LABELS, BAR_COLORS, offsets)):
        vals = values[key]
        ax.bar(x + offset, vals, width=bar_w, label=label, color=color,
               edgecolor='black', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(shapes, fontsize=11)
    ax.set_xlabel('Systolic Shape (BLEN, MLEN)', fontsize=12, labelpad=8)
    ax.set_ylabel('Relative Value', fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}x'))
    ax.set_ylim(0.4, max(v for vals in values.values() for v in vals) * 1.15)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot systolic shape comparison bar chart')
    parser.add_argument(
        '--output', '-o',
        default='/home/hw1020/PLENA_Simulator/experiments/systolic_shape_comparison.png',
        help='Output PNG path',
    )
    args = parser.parse_args()

    print("Data:")
    for row in DATA:
        print(
            f"  {row['shape']:10s}  power={row['power_rel']:.2f}x  area={row['area_rel']:.2f}x  "
            f"FFN={row['ffn_energy_rel']:.2f}x  FA={row['fa_energy_rel']:.2f}x"
        )

    plot(DATA, args.output)
