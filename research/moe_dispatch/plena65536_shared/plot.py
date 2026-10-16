"""Standalone figures from frozen Shared-only DSE and retention-loop results.

Plots read the result CSV; no performance numbers are synthesized here.
OS/WS/IS retention loops share the same ideal-port compute model, so only
their storage/refill tradeoff is plotted as a dataflow difference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BATCHES = (2, 4, 8, 16, 64, 96, 128)
PROFILE = "pipelined_tree1"
COLORS = ("#6b7280", "#2563eb", "#059669", "#e07811")
MARKERS = ("x", "o", "s", "^")
FLOW_COLORS = {
    "OS/OS": "#2563eb", "OS/WS": "#818cf8", "OS/IS": "#22d3ee",
    "WS/OS": "#059669", "WS/WS": "#84cc16", "WS/IS": "#14b8a6",
    "IS/OS": "#dc2626", "IS/WS": "#e07811", "IS/IS": "#a855f7",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Cross-dataflow result table is empty")
    return rows


def os_rows(rows, design):
    selected = [r for r in rows if r["design"] == design
                and r["row_type"] == "batch" and r["profile"] == PROFILE
                and all(flow == "OS" for flow in r["dataflow"].split("/"))]
    selected = sorted(selected, key=lambda r: int(r["batch"]))
    if tuple(int(r["batch"]) for r in selected) != BATCHES:
        raise ValueError(f"Missing or duplicated seven-batch OS timing for {design}")
    return selected


def setup_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BATCHES, [f"B{batch}" for batch in BATCHES])
    ax.grid(True, which="major", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Shared token count / Batch")


def frame_summary(row):
    frames = row["native_frames"].split(" + ")
    simple = [",".join(frame.split(",")[:2]) for frame in frames]
    return " + ".join(simple)


def series_label(rows, family):
    row = rows[0]
    return f"{family}: {row['physical_PM_PN_G']}\n{frame_summary(row)}"


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def compute_figure(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.3), sharey=True)
    for ax, scope, title in zip(axes, ("native", "extended"),
                                ("A. Native BLEN / MLEN domain",
                                 "B. Output-replicated topology extension")):
        names = ("paper_original", scope + "_single",
                 scope + "_homogeneous", scope + "_heterogeneous")
        families = ("Paper original", "DSE single", "DSE homogeneous", "DSE heterogeneous")
        for index, (name, family) in enumerate(zip(names, families)):
            source = os_rows(rows, name)
            ys = [float(r["compute_ms"]) for r in source]
            ax.plot(BATCHES, ys, color=COLORS[index], marker=MARKERS[index],
                    markerfacecolor="none" if index > 1 else COLORS[index],
                    markersize=7, linewidth=1.8, linestyle="--" if index == 0 else "-",
                    label=series_label(source, family))
        setup_axis(ax)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False,
                  fontsize=8.5, ncol=2, columnspacing=1.6, handlelength=2.4)
    axes[0].set_ylabel("Pure GEMM compute time (ms; logarithmic axis)")
    fig.suptitle("Shared Expert · 65,536 multipliers · BF16 / FP32 · analytical model",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.065, 0.032,
             "Physical size = PM × PN × G; G = R / b. 1 GHz; ideal operand/context ports. "
             "Frozen hardware; runtime may select row or paired-GU / Down column partitions.\n"
             "PE dataflow remains OS. Shadow partial banks and a pipelined M_SUM tree are "
             "declared assumptions; timings are not native RTL measurements.", fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.32, left=0.065, right=0.98, wspace=0.08)
    save(fig, out, "shared_compute_native_vs_replicated")


def fetch_figure(rows, out):
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    names = ("paper_original", "extended_single", "extended_homogeneous", "extended_heterogeneous")
    labels = ("Paper original compute", "DSE extended single compute",
              "DSE extended homogeneous compute", "DSE extended heterogeneous compute")
    service = []
    for index, (name, label) in enumerate(zip(names, labels)):
        source = os_rows(rows, name)
        ys = [float(r["compute_ms"]) for r in source]
        service += [float(r["unique_weight_fetch_ms_at256"]) for r in source]
        ax.plot(BATCHES, ys, color=COLORS[index], marker=MARKERS[index],
                markerfacecolor="none" if index > 1 else COLORS[index],
                markersize=7, linewidth=1.8, linestyle="--" if index == 0 else "-", label=label)
    if not np.allclose(service, service[0], rtol=0, atol=1e-12):
        raise AssertionError("Unique 33-MiB cold weight service must be constant")
    ax.axhline(service[0], color="#b91c1c", linewidth=2, linestyle=":",
               label=f"33 MiB ÷ 256 GB/s = {service[0]:.6f} ms")
    setup_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms; logarithmic axis)")
    ax.set_title("Shared compute vs necessary cold-weight service\n"
                 "BF16 / FP32 analytical model · 1 GHz · 65,536 multipliers",
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="best", frameon=True, framealpha=0.94, fontsize=9)
    fig.text(0.105, 0.045,
             "The service line is a bytes / bandwidth estimate, not simulated HBM latency. "
             "All shared weights are unique and loaded once.\n"
             "Compute and fetch are not added here. Full pre-residency changes the external "
             "weight requirement; on-chip operand ports still need provisioning.", fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.83, bottom=0.19)
    save(fig, out, "shared_compute_vs_256GBs_weight_service")


def flow_figure(rows, out):
    source = [r for r in rows if r["design"] == "extended_heterogeneous"
              and r["row_type"] == "batch" and r["profile"] == PROFILE]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    for ax_pair, batch in zip(axes, (2, 128)):
        points = [r for r in source if int(r["batch"]) == batch]
        if len(points) != 9 or len({r["dataflow"] for r in points}) != 9:
            raise ValueError("Expected every two-core OS/WS/IS retention combination")
        computes = {float(r["compute_ms"]) for r in points}
        if len(computes) != 1:
            raise AssertionError("Ideal-port retention loops must not invent compute speedup")
        points = sorted(points, key=lambda point: point["dataflow"])
        flows = [r["dataflow"] for r in points]
        capacities = [float(r["local_scratch_tree_upper_KiB"]) / 1024 for r in points]
        refill = [(float(r["pool_weight_refill_required_GB_s"])
                   + float(r["pool_X_refill_required_GB_s"])
                   + float(r["pool_context_required_GB_s"])) / 1000 for r in points]
        metrics = (capacities, refill)
        captions = ("Local / partial / M_SUM storage upper envelope (MiB)",
                    "Pool→local W + X + partial-context rate (TB/s)")
        for ax, values, caption in zip(ax_pair, metrics, captions):
            ax.barh(range(len(flows)), values, color=[FLOW_COLORS[f] for f in flows],
                    height=0.65, alpha=0.9)
            ax.set_yticks(range(len(flows)), flows)
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlim(min(values) / 1.7, max(values) * 2.1)
            ax.grid(True, axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel(caption)
            ax.set_ylabel("Core 0 / Core 1 retention loop")
            for index, value in enumerate(values):
                ax.annotate(f"{value:.3f}", (value, index), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=8.5)
            ax.set_title(f"B{batch}: all flows share {next(iter(computes)):.6f} ms compute",
                         loc="left", fontsize=10.5, fontweight="bold")
    geometry = source[0]["physical_PM_PN_G"]
    frames = frame_summary(source[0])
    fig.suptitle("Retention-loop storage / refill tradeoff · frozen extended heterogeneous geometry",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.075, 0.919, f"PM × PN × G: {geometry}; native frames: {frames}", fontsize=10)
    fig.text(0.07, 0.025,
             "BF16 operands / FP32 partials; analytical byte accounting. Capacity is a conservative "
             "upper envelope, not synthesized SRAM area.\n"
             "WS / IS are operand-retention loop extensions on OS PEs; complete grouped partial "
             "contexts are charged. These rates are not HBM bandwidth.\n"
             "Unlimited context / operand ports keep compute equal across flows; a finite-port "
             "implementation must validate each point separately.", fontsize=9)
    fig.subplots_adjust(top=0.85, bottom=0.155, left=0.075, right=0.98, wspace=0.23, hspace=0.43)
    save(fig, out, "shared_retention_capacity_refill_tradeoff")


def main(out: Path):
    table = out / "cross_dataflow.csv"
    frozen = out / "FROZEN_SELECTION.json"
    if not table.exists() or not frozen.exists():
        raise FileNotFoundError("Complete FROZEN_SELECTION.json and cross_dataflow.csv first")
    source_hashes = {path.name: sha(path) for path in (table, frozen)}
    rows = load_rows(table)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": 10, "axes.titlepad": 13})
    compute_figure(rows, out)
    fetch_figure(rows, out)
    flow_figure(rows, out)
    stems = ("shared_compute_native_vs_replicated", "shared_compute_vs_256GBs_weight_service",
             "shared_retention_capacity_refill_tradeoff")
    files = []
    for stem in stems:
        for suffix in ("png", "pdf"):
            path = out / f"{stem}.{suffix}"
            data = path.read_bytes()
            signature = b"\x89PNG\r\n\x1a\n" if suffix == "png" else b"%PDF"
            if not data.startswith(signature) or len(data) < 10_000:
                raise AssertionError(f"Invalid standalone {suffix.upper()}: {path}")
            item = {"path": str(path), "bytes": len(data), "sha256": sha(path)}
            if suffix == "png":
                from PIL import Image
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    item["dimensions"] = list(im.size)
            files.append(item)
    if source_hashes != {path.name: sha(path) for path in (table, frozen)}:
        raise AssertionError("Source results changed during plotting")
    audit = {"status": "PASS", "source_sha256": source_hashes,
             "source_rows": len(rows), "files": files,
             "model": "BF16/FP32 analytical; native and output-replicated domains separate",
             "no_dataflow_compute_speedup_invented": True,
             "script_sha256": sha(Path(__file__))}
    (out / "PLOT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": len(stems), "files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
