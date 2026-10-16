"""Standalone figures from the completed full Shared + MoE frozen tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BATCHES = (2, 4, 8, 16, 64, 96, 128)
DESIGNS = ("original", "extended_single", "extended_homogeneous", "extended_heterogeneous")
LABELS = ("Paper original", "DSE single", "DSE homogeneous", "DSE heterogeneous")
COLORS = ("#6b7280", "#2563eb", "#059669", "#e07811")
MARKERS = ("x", "o", "s", "^")
FLOW_COLORS = {
    "OS/OS": "#2563eb", "OS/WS": "#818cf8", "OS/IS": "#22d3ee",
    "WS/OS": "#059669", "WS/WS": "#84cc16", "WS/IS": "#14b8a6",
    "IS/OS": "#dc2626", "IS/WS": "#e07811", "IS/IS": "#a855f7",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Cross-dataflow results are empty")
    return rows


def os_rows(rows, design):
    points = sorted([row for row in rows if row["design"] == design and row["split"] == "heldout"
                     and row["row_type"] == "batch_GM"
                     and all(flow == "OS" for flow in row["dataflow"].split("/"))], key=lambda r: int(r["batch"]))
    if tuple(int(row["batch"]) for row in points) != BATCHES:
        raise ValueError(f"Missing or duplicated seven Batch heldout results: {design}")
    return points


def setup(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, [f"B{batch}" for batch in BATCHES])
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Batch / Shared tokens; routed assignments = 6 × Batch")


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def compute_figure(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7))
    source = {name: os_rows(rows, name) for name in DESIGNS}
    base = [float(row["compute_ms"]) for row in source["extended_single"]]
    for index, (name, label) in enumerate(zip(DESIGNS, LABELS)):
        ys = [float(row["compute_ms"]) for row in source[name]]
        axes[0].plot(BATCHES, ys, color=COLORS[index], marker=MARKERS[index], linewidth=1.8,
                     markersize=7, markerfacecolor="none" if index > 1 else COLORS[index],
                     linestyle="--" if index == 0 else "-", label=label)
        if index > 0:
            speedup = np.array(base) / np.array(ys)
            axes[1].plot(BATCHES, speedup, color=COLORS[index], marker=MARKERS[index], linewidth=1.8,
                         markersize=7, markerfacecolor="none" if index > 1 else COLORS[index], label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Complete Shared + routed GEMM makespan (ms)")
    axes[0].set_title("A. Complete workload under frozen hardware", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Compute speedup vs frozen DSE single (×)")
    axes[1].axhline(1, color="#9ca3af", linewidth=1)
    axes[1].set_title("B. Difference among the optimized organizations", loc="left", fontsize=12, fontweight="bold")
    for ax in axes:
        setup(ax)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
    frames = [f"{label}: {source[name][0]['physical_PM_PN_G']}" for name, label in zip(DESIGNS, LABELS)]
    fig.suptitle("Shared + true Top-6 MoE · 65,536 multipliers · BF16 / FP32", fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.065, 0.13, "\n".join(frames), fontsize=9)
    fig.text(0.065, 0.035,
             "135 heldout windows; window GM within Batch, then equal-weight seven-Batch GM. "
             "Geometry selected on 18 development windows.\n"
             "Analytical 1 GHz compute ceiling: exact whole-expert assignment; ideal operand/context ports. "
             "G = R / b; area and Fmax are uncalibrated. No full-inference timing claim.", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.86, bottom=0.28, wspace=0.27)
    save(fig, out, "moe_compute_frozen_architectures")


def fetch_figure(rows, out):
    fig, ax = plt.subplots(figsize=(11.8, 6.7))
    for index, (name, label) in enumerate(zip(DESIGNS, LABELS)):
        source = os_rows(rows, name)
        ax.plot(BATCHES, [float(row["compute_ms"]) for row in source],
                color=COLORS[index], marker=MARKERS[index], linewidth=1.8, markersize=7,
                markerfacecolor="none" if index > 1 else COLORS[index],
                linestyle="--" if index == 0 else "-", label=label + " compute")
    source = os_rows(rows, "original")
    weight_service = [float(row["unique_weight_fetch_ms_at256"]) for row in source]
    ax.plot(BATCHES, weight_service, color="#b91c1c", marker="D", linestyle=":", linewidth=2.1,
            label="Active unique W ÷ 256 GB/s (Batch GM)")
    windows = [row for row in rows if row["design"] == "original" and row["split"] == "heldout"
               and row["row_type"] == "window" and row["dataflow"] == "OS"]
    ranges = [[float(row["unique_weight_fetch_ms_at256"]) for row in windows if int(row["batch"]) == batch]
              for batch in BATCHES]
    ax.fill_between(BATCHES, [min(values) for values in ranges], [max(values) for values in ranges],
                    color="#b91c1c", alpha=0.13, label="Unique-W service min/max over real windows")
    setup(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms; logarithmic axis)")
    ax.set_title("Complete Shared + MoE compute vs necessary cold-weight service\n"
                 "Frozen hardware · captured true Top-6 routes · 65,536 multipliers", loc="left", fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.text(0.105, 0.04,
             "Unique W = 33 MiB Shared + 16.5 MiB × active routed experts; inactive experts contribute zero. "
             "All 64 active routed experts give 1,089 MiB.\n"
             "Service time is bytes / bandwidth, not native HBM latency. It is compared with compute "
             "and never added to it. Local refills are separate traffic.\n"
             "Analytical 1 GHz GEMM compute; routing, gather/scatter and non-GEMM arithmetic are not timed.", fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.82, bottom=0.2)
    save(fig, out, "moe_compute_vs_256GBs_weight_service")


def gm(values):
    values = list(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def flow_figure(rows, out):
    source = [row for row in rows if row["design"] == "extended_heterogeneous" and row["split"] == "heldout"]
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
    flows = sorted(FLOW_COLORS)
    for pair, batch in zip(axes, (2, 128)):
        aggregates = {row["dataflow"]: row for row in source if row["row_type"] == "batch_GM" and int(row["batch"]) == batch}
        if sorted(aggregates) != flows:
            raise ValueError("All nine OS/WS/IS dual combinations are required")
        if len({row["compute_ms"] for row in aggregates.values()}) != 1:
            raise AssertionError("Retention loops invented a compute speedup under ideal ports")
        capacities = [float(aggregates[flow]["local_scratch_tree_upper_KiB"]) / 1024 for flow in flows]
        rates = []
        for flow in flows:
            windows = [row for row in source if row["row_type"] == "window"
                       and int(row["batch"]) == batch and row["dataflow"] == flow]
            total_rates = [sum(float(row[key]) for key in ("pool_weight_refill_required_GB_s",
                                                          "pool_X_refill_required_GB_s",
                                                          "pool_context_required_GB_s")) / 1000 for row in windows]
            rates.append(gm(total_rates))
        for ax, values, caption in zip(pair, (capacities, rates), (
                "Local / grouped partial / tree storage upper bound (MiB)",
                "Pool → local W + X + grouped-context rate (TB/s; window GM)")):
            ax.barh(range(len(flows)), values, color=[FLOW_COLORS[flow] for flow in flows], height=0.66)
            ax.set_yticks(range(len(flows)), flows)
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlim(min(values) / 1.6, max(values) * 2)
            ax.grid(True, axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel(caption, fontsize=9)
            ax.set_ylabel("Core 0 / Core 1 retention loop")
            ax.set_title(f"B{batch}: same {float(aggregates[flows[0]]['compute_ms']):.6f} ms compute",
                         loc="left", fontweight="bold", fontsize=10.5)
            for index, value in enumerate(values):
                ax.annotate(f"{value:.3f}", (value, index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)
    fig.suptitle("Full Shared + MoE retention-loop storage / supply tradeoff", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.075, 0.923, "Frozen heterogeneous PM × PN × G: " + source[0]["physical_PM_PN_G"], fontsize=10)
    fig.text(0.075, 0.027,
             "Capacity takes the maximum over actual windows: fixed PE/shadow/operand/tree storage once "
             "+ grouped scratch under scheduled job overlap.\n"
             "PEs remain OS; WS/IS are declared operand-loop extensions with full G partial contexts charged. "
             "Unlimited ports keep pure compute equal.\n"
             "These are internal pool rates, not HBM bandwidth. Resident active weights and all-expert "
             "activation payload are reported separately in the workbook.", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.16, wspace=0.24, hspace=0.43)
    save(fig, out, "moe_retention_capacity_refill_tradeoff")


def main(out):
    table, frozen = out / "cross_dataflow.csv", out / "FROZEN_SELECTION.json"
    source_hashes = {path.name: sha(path) for path in (table, frozen)}
    rows = load_rows(table)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlepad": 13})
    compute_figure(rows, out)
    fetch_figure(rows, out)
    flow_figure(rows, out)
    stems = ("moe_compute_frozen_architectures", "moe_compute_vs_256GBs_weight_service",
             "moe_retention_capacity_refill_tradeoff")
    files = []
    for stem in stems:
        for suffix in ("png", "pdf"):
            path = out / f"{stem}.{suffix}"
            data = path.read_bytes()
            signature = b"\x89PNG\r\n\x1a\n" if suffix == "png" else b"%PDF"
            if not data.startswith(signature) or len(data) < 10000:
                raise AssertionError(f"Invalid standalone figure {path}")
            item = {"path": str(path), "bytes": len(data), "sha256": sha(path)}
            if suffix == "png":
                from PIL import Image
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    item["dimensions"] = list(im.size)
            files.append(item)
    if source_hashes != {path.name: sha(path) for path in (table, frozen)}:
        raise AssertionError("Frozen sources changed during plotting")
    (out / "PLOT_AUDIT.json").write_text(json.dumps({
        "status": "PASS", "full_workload_scope": "Shared + true Top6 routed MoE",
        "source_sha256": source_hashes, "source_rows": len(rows), "files": files,
        "script_sha256": sha(Path(__file__)), "no_dataflow_compute_speedup_invented": True,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "figures": 3, "files": 6}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
