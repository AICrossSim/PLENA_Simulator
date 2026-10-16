"""Export scientific figures from completed compute-ceiling artifacts.

These are analytical ideal-operand requirements, not native-HBM measurements.
Source geometry labels and per-batch means are read from the frozen result
files; the plotting code never searches a configuration or changes timings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BATCHES = (2, 4, 8, 16, 64, 96, 128)
LABELS = ("original_single", "best_single", "best_homogeneous", "best_heterogeneous")
STYLES = {
    "original_single": {"name": "Original single", "color": "#666666", "marker": "x", "linestyle": "--"},
    "best_single": {"name": "Optimized single", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "best_homogeneous": {"name": "Optimized homogeneous", "color": "#E69F00", "marker": "s", "linestyle": "-"},
    "best_heterogeneous": {"name": "Optimized heterogeneous", "color": "#009E73", "marker": "^", "linestyle": "-"},
}


def _gm(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("a geometric mean needs positive finite observations")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _number(value: float, _position: Any) -> str:
    return f"{value:,.3g}"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry_label(label: str, geometry: str) -> str:
    # Explicit MxNxK shapes rather than opaque B/H abbreviations.  Plus
    # sign denotes independent cores; each repeated shape is printed.
    return f"{STYLES[label]['name']}: {geometry.replace('+', ' + ')}"


def _draw(ax: Any, label: str, values: list[float], geometry: str) -> None:
    style = STYLES[label]
    ax.plot(range(len(BATCHES)), values, color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], linewidth=1.65, markersize=5,
            label=_geometry_label(label, geometry))


def _axes(ax: Any, *, logarithmic: bool, ylabel: str) -> None:
    ax.set_xticks(range(len(BATCHES)), [f"B{batch}" for batch in BATCHES])
    ax.set_xlabel("Tokens per window (batch)")
    ax.set_ylabel(ylabel)
    if logarithmic:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(_number))
    ax.grid(axis="y", which="major", color="#dadada", linewidth=.7)
    ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig: Any, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.png", out / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(paths[1], facecolor="white", bbox_inches="tight",
                metadata={"Description": "Ideal-operand analytical compute DSE; assumed 1 GHz; not native hardware measurement."})
    plt.close(fig)
    return paths


def _pooled_figures(out: Path) -> tuple[list[Path], dict[str, str]]:
    """Plot independently completed RR/serial-fallback pooled results."""
    pooled = out / "pooled"
    completion = pooled / "COMPLETION.json"
    reference_completion = pooled / "COMPLETION_REFERENCE.json"
    if not completion.exists() or not reference_completion.exists():
        return [], {}
    paths = [completion, reference_completion,
             pooled / "compute_by_batch.csv", pooled / "original_single_by_batch.csv"]
    source_hashes = {str(path): _hash(path) for path in paths}
    source = []
    for path in paths[2:]:
        with path.open(newline="") as stream:
            source.extend(csv.DictReader(stream))
    name_map = {"single": "best_single", "homogeneous": "best_homogeneous",
                "heterogeneous": "best_heterogeneous", "original_single": "original_single"}
    rows = [{**row, "plot_label": name_map[row["label"]]}
            for row in source if row["label"] in name_map and row["batch"] != "all"]
    lookup = {(row["plot_label"], int(row["batch"])): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {(label, batch) for label in LABELS for batch in BATCHES}:
        raise ValueError("pooled figures need four distinct configurations and all seven batches")
    geometries = {}
    for label in LABELS:
        geometry = {lookup[label, batch]["geometry"] for batch in BATCHES}
        if len(geometry) != 1:
            raise ValueError("pooled hardware geometry must remain frozen across batches")
        geometries[label] = geometry.pop()
    latency = {label: [float(lookup[label, batch]["compute_ms_geomean"]) for batch in BATCHES]
               for label in LABELS}
    if any(value <= 0 or not math.isfinite(value) for series in latency.values() for value in series):
        raise ValueError("pooled latency series must be positive and finite")
    outputs = []
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    for label in LABELS:
        _draw(ax, label, latency[label], geometries[label])
        if label == "best_heterogeneous":
            # Hollow triangle preserves the homogeneous square when both
            # schedules have exactly identical latencies.
            ax.lines[-1].set_markerfacecolor("none")
            ax.lines[-1].set_markeredgewidth(1.4)
            ax.lines[-1].set_markersize(6.5)
    _axes(ax, logarithmic=True, ylabel="Compute latency (ms, geometric mean; log scale)")
    ax.set_title("Cross-expert ready-record interleaving\nWhole-expert ownership")
    ax.legend(loc="upper left", frameon=True, framealpha=.95)
    fig.text(.12, -.025,
             "Unlimited resident operands/contexts; BF16/FP32; 12,288 multipliers; assumed 1 GHz.\n"
             "Feasible RR with per-core serial fallback; no memory/control timing; original baseline has the same interleaving ability.",
             ha="left", fontsize=8.3, color="#444444")
    fig.tight_layout()
    outputs += _save(fig, out, "pooled_compute_latency_by_batch")

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    for label in ("best_homogeneous", "best_heterogeneous"):
        ratio = [s / c for s, c in zip(latency["best_single"], latency[label])]
        _draw(ax, label, ratio, geometries[label])
        if label == "best_heterogeneous":
            ax.lines[-1].set_markerfacecolor("none")
            ax.lines[-1].set_markeredgewidth(1.4)
            ax.lines[-1].set_markersize(6.5)
    ax.axhline(1.0, color=STYLES["best_single"]["color"], linewidth=1, linestyle=":",
               label=_geometry_label("best_single", geometries["best_single"]))
    _axes(ax, logarithmic=False, ylabel="Speedup vs pooled optimized single (higher is faster)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.3f}"))
    ax.set_title("Cross-expert ready-record interleaving / Whole-expert ownership\nFrozen geometry comparison with unlimited contexts", fontsize=10.5)
    ax.legend(loc="best", frameon=True, framealpha=.95)
    fig.text(.12, -.04,
             "Paired batch geometric-mean ratios; overlapping markers indicate equal latency.\n"
             "II=1, completion latency=21 cycles, assumed 1 GHz; no same-issue expert co-packing; not calibrated hardware speedup.",
             ha="left", fontsize=8.3, color="#444444")
    fig.tight_layout()
    outputs += _save(fig, out, "pooled_relative_speedup")
    if any(_hash(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("pooled source artifacts changed during rendering")
    return outputs, source_hashes


def generate_plots(out: str | Path) -> dict[str, Any]:
    """Render frozen primary results.  All sources must be complete already."""
    out = Path(out)
    completion_path = out / "COMPLETION.json"
    if not completion_path.exists():
        raise ValueError("COMPLETION.json is required before plotting completed results")
    compute_path = out / "compute_by_batch.csv"
    with compute_path.open(newline="") as stream:
        compute = list(csv.DictReader(stream))
    selected = [row for row in compute if row["label"] in LABELS and row["batch"] != "all"]
    lookup = {(row["label"], int(row["batch"])): row for row in selected}
    if len(lookup) != len(selected):
        raise ValueError("duplicate label/batch compute summary rows")
    if set(lookup) != {(label, batch) for label in LABELS for batch in BATCHES}:
        raise ValueError("all four configurations and seven batches are required")
    geometries = {}
    for label in LABELS:
        gs = {lookup[label, batch]["geometry"] for batch in BATCHES}
        if len(gs) != 1:
            raise ValueError("hardware geometry must remain frozen across batches")
        geometries[label] = gs.pop()
    latency = {label: [float(lookup[label, batch]["compute_ms_geomean"]) for batch in BATCHES]
               for label in LABELS}
    if any(value <= 0 or not math.isfinite(value) for series in latency.values() for value in series):
        raise ValueError("latency series must be finite and positive")

    source_paths = [completion_path, compute_path]
    supply_rates: dict[str, dict[int, dict[str, float]]] = {}
    for label in LABELS:
        path = out / f"supply_{label}.json"
        source_paths.append(path)
        source = json.loads(path.read_text())
        grouped = defaultdict(list)
        seen = set()
        for row in source:
            if row["window_id"] in seen:
                raise ValueError("duplicate window in a reverse-supply artifact")
            seen.add(row["window_id"])
            if row["label"] != label or row["geometry"] != geometries[label]:
                raise ValueError("supply and frozen compute geometry disagree")
            if row["frequency_hz"] != 1_000_000_000:
                raise ValueError("these plot annotations require the declared 1 GHz clock")
            grouped[int(row["batch"])].append(row)
        if set(grouped) != set(BATCHES):
            raise ValueError("reverse-supply data do not cover the seven primary batches")
        supply_rates[label] = {}
        for batch, rows in grouped.items():
            if len(rows) != int(lookup[label, batch]["windows"]):
                raise ValueError("compute and supply window counts differ")
            compute_gm_ms = _gm([float(row["compute_latency_ms"]) for row in rows])
            if not math.isclose(compute_gm_ms, float(lookup[label, batch]["compute_ms_geomean"]), rel_tol=1e-8):
                raise ValueError("supply and compute-only latencies disagree")
            supply_rates[label][batch] = {
                "cold_external_unique_weight_GB_s": _gm([
                    row["external"]["cold_weight_average_required_GB_s"] for row in rows]),
                "pool_to_local_weight_full_slice_GB_s": _gm([
                    row["onchip"]["os_kmajor_shared_pool_average_rates"]["weight_full_slice_average_GB_s"]
                    for row in rows]),
                "local_to_array_weight_full_slice_GB_s": _gm([
                    row["onchip"]["os_kmajor_array_average_rates"]["weight_full_slice_average_GB_s"]
                    for row in rows]),
            }
    source_hashes = {str(path): _hash(path) for path in source_paths}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 8.3, "svg.fonttype": "none"})
    outputs = []

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    for label in LABELS:
        _draw(ax, label, latency[label], geometries[label])
    _axes(ax, logarithmic=True, ylabel="Compute latency (ms, geometric mean; log scale)")
    ax.set_title("Whole-expert serial issue schedule\nCompute-only DSE on real MoE routing traces")
    ax.legend(loc="upper left", frameon=True, framealpha=.95)
    fig.text(.12, -.015,
             "BF16/FP32; 12,288 multipliers; assumed 1 GHz; Shared included.\n"
             "Whole-expert serial issue schedule; ideal ready operands; post-router FFN only; not native hardware timing.",
             ha="left", fontsize=8.5, color="#444444")
    fig.tight_layout()
    outputs += _save(fig, out, "compute_latency_by_batch")

    fields = (
        ("cold_external_unique_weight_GB_s", "Cold external weight load", "Unique BF16 weights / compute time"),
        ("pool_to_local_weight_full_slice_GB_s", "Shared pool to local W buffers", "Full physical slices; reuse across M tiles"),
        ("local_to_array_weight_full_slice_GB_s", "Local W buffers/latches to array", "Per-issue operand delivery; repeated W included"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharey=True)
    for ax, (field, title, subtitle) in zip(axes, fields):
        for label in LABELS:
            series = [supply_rates[label][batch][field] for batch in BATCHES]
            _draw(ax, label, series, geometries[label])
        _axes(ax, logarithmic=True, ylabel="Required bandwidth (GB/s, GM; log scale)" if ax is axes[0] else "")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.03), frameon=False)
    fig.suptitle("Whole-expert serial issue schedule: reverse supply requirements", fontsize=13, y=1.015)
    fig.text(.5, -.105,
             "Three distinct boundaries: do not add their rates. Cold-load HBM may be zero after full weight prewarm.\n"
             "Serial issue loop only: these pool refill rates do not apply to interleaved RR. Analytical 1 GHz; no memory timing proof.",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, .09, 1, 1))
    outputs += _save(fig, out, "required_bandwidth_by_batch")

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    for label in ("best_homogeneous", "best_heterogeneous"):
        ratios = [single / candidate for single, candidate in zip(latency["best_single"], latency[label])]
        _draw(ax, label, ratios, geometries[label])
    ax.axhline(1.0, color=STYLES["best_single"]["color"], linewidth=1, linestyle=":",
               label=_geometry_label("best_single", geometries["best_single"]))
    _axes(ax, logarithmic=False, ylabel="Speedup vs optimized single (higher is faster)")
    # Relative gains can be below one percent.  Three significant digits
    # collapse nearby tick values around 1.0, so preserve three decimal
    # places explicitly on this ratio plot.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.3f}"))
    ax.set_title("Whole-expert serial issue schedule\nPure-compute advantage after single-core geometry optimization")
    ax.legend(loc="best", frameon=True, framealpha=.95)
    fig.text(.12, -.025,
             "Ratio of paired batch geometric-mean latencies; one frozen hardware configuration across all batches.\n"
             "Whole-expert serial issue only; analytical ready-operand study; not a system-level or calibrated hardware speedup.",
             ha="left", fontsize=8.5, color="#444444")
    fig.tight_layout()
    outputs += _save(fig, out, "relative_compute_speedup_by_batch")
    pooled_outputs, pooled_sources = _pooled_figures(out)
    outputs += pooled_outputs
    source_hashes.update(pooled_sources)

    if any(_hash(Path(path)) != digest for path, digest in source_hashes.items()):
        raise AssertionError("source result artifacts changed while plotting")
    receipt = {
        "status": "PASS", "scope": "completed compute-only figures; no new simulation",
        "assumed_clock_GHz": 1, "native_calibrated": False,
        "geometry_order_M_N_K": True, "geometries": geometries,
        "primary_plot_schedule": "whole-expert serial issue schedule",
        "pooled_figures_included": bool(pooled_outputs),
        "pooled_plot_schedule": "cross-expert RR ready-record interleaving, whole-expert ownership, per-core serial fallback",
        "bandwidth_aggregation": "per-window bytes/compute-time, then geometric mean within batch",
        "source_sha256": source_hashes,
        "sources_unchanged": True,
        "plot_files": [{"path": str(path), "bytes": path.stat().st_size,
                        "sha256": _hash(path)} for path in outputs],
    }
    (out / "PLOT_AUDIT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_plots(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
