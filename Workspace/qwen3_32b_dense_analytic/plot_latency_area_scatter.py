#!/usr/bin/env python3
"""Render publication-style latency/area scatter plots for Qwen3-32B DSE runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FAMILY_STYLE = {
    "MXINT": {"color": "#2878B5"},
    "MXFP": {"color": "#D65F2E"},
    "Other": {"color": "#777777"},
}

WEIGHT_MARKERS = {
    "MXINT4": "o",
    "MXINT8": "s",
    "MXFP_E1M2": "^",
    "MXFP_E2M1": "v",
    "MXFP_E4M3": "D",
    "MXFP_E5M2": "P",
}


def load_records(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load complete trials and normalize physical area to square millimetres."""
    records: list[dict[str, Any]] = []
    with (run_dir / "trials.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("state") != "complete" or not record.get("latency_ms"):
                continue
            if record.get("area_mm2") is not None:
                area_mm2 = float(record["area_mm2"])
            elif record.get("area_um2") is not None:
                area_mm2 = float(record["area_um2"]) / 1e6
            elif record.get("area") is not None:
                area_mm2 = float(record["area"]) / 1e6
            else:
                continue
            if area_mm2 <= 0:
                continue
            record = dict(record)
            record["area_mm2"] = area_mm2
            records.append(record)
    if not records:
        raise ValueError(f"no complete records with area and latency in {run_dir / 'trials.jsonl'}")

    # Interrupted grid attempts can be retried under a new Optuna trial number.
    # Plot each design tuple once so retries do not alter density or tie breaks.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("precision_profile"),
            record.get("MLEN"),
            record.get("BLEN"),
            record.get("INT_DATA_WIDTH"),
        )
        previous = unique.get(key)
        if previous is None or int(record.get("trial", -1)) > int(
            previous.get("trial", -1)
        ):
            unique[key] = record
    records = list(unique.values())

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return records, summary


def precision_family(record: dict[str, Any]) -> str:
    profile = str(record.get("precision_profile", "")).lower()
    if "mxfp" in profile:
        return "MXFP"
    if "mxint" in profile:
        return "MXINT"
    return "Other"


def weight_precision(record: dict[str, Any]) -> str:
    """Return the GPTQ weight format used by one software profile."""
    explicit = record.get("weight_precision")
    if explicit:
        return str(explicit).upper()
    profile = str(record.get("precision_profile", ""))
    match = re.match(r"w_(.+?)__", profile, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the two-objective area/latency frontier used by this figure."""
    front: list[dict[str, Any]] = []
    best_latency = float("inf")
    for record in sorted(records, key=lambda r: (float(r["area_mm2"]), float(r["latency_ms"]))):
        latency = float(record["latency_ms"])
        if latency < best_latency:
            front.append(record)
            best_latency = latency
    return front


def log_knee(front: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the maximum-distance knee in log area/log latency space."""
    if len(front) <= 2:
        return front[len(front) // 2]
    area_point = min(front, key=lambda r: float(r["area_mm2"]))
    latency_point = min(front, key=lambda r: float(r["latency_ms"]))
    x1, y1 = math.log10(float(area_point["area_mm2"])), math.log10(float(area_point["latency_ms"]))
    x2, y2 = math.log10(float(latency_point["area_mm2"])), math.log10(float(latency_point["latency_ms"]))
    denom = math.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return area_point

    def distance(record: dict[str, Any]) -> float:
        x = math.log10(float(record["area_mm2"]))
        y = math.log10(float(record["latency_ms"]))
        return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom

    return max(front, key=distance)


def _precision_token(profile: str, prefix: str) -> str:
    match = re.search(rf"(?:^|__){prefix}_([^_]+(?:_[^_]+)?)", profile.lower())
    if not match:
        return "?"
    token = match.group(1).upper()
    return token.replace("MXINT", "I").replace("MXFP_", "F")


def short_profile(record: dict[str, Any]) -> str:
    profile = str(record.get("precision_profile", "unknown"))
    return "/".join(
        [
            f"W:{_precision_token(profile, 'w')}",
            f"A:{_precision_token(profile, 'act')}",
            f"KV:{_precision_token(profile, 'kv')}",
            f"FP:{_precision_token(profile, 'fp')}",
        ]
    )


def knob_summary(name: str, record: dict[str, Any]) -> str:
    return (
        f"{name}: T{int(record['trial'])} | {short_profile(record)} | "
        f"M/V/B={int(record['MLEN'])}/{int(record['VLEN'])}/{int(record['BLEN'])} | "
        f"INT={int(record['INT_DATA_WIDTH'])} | acc={float(record['accuracy_score']):.2f} | "
        f"lat={float(record['latency_ms']):.1f} ms | area={float(record['area_mm2']):.2f} mm²"
    )


def annotate_point(
    ax: Any,
    record: dict[str, Any],
    label: str,
    marker: str,
    color: str,
    offset: tuple[int, int],
) -> None:
    x = float(record["area_mm2"])
    y = float(record["latency_ms"])
    ax.scatter(
        [x], [y], s=150, marker=marker, facecolor=color, edgecolor="black", linewidth=1.0, zorder=6
    )
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"},
        bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": "#777777", "lw": 0.6, "alpha": 0.95},
        zorder=7,
    )


def _model_title(summary: dict[str, Any]) -> str:
    if summary.get("compiler_cost_mode") == "off":
        return "Legacy Analytic Latency"
    if summary.get("compiler_cost_mode") == "compute-objective":
        return "Compiler CostEmitter Compute Latency"
    if summary.get("compiler_cost_mode") == "roofline-objective":
        return "Compiler Stage-Wise Roofline (rtl-v1 + V4)"
    return str(summary.get("compiler_cost_mode", "Latency Model"))


def plot(
    run_dir: Path,
    output: Path,
    title: str | None = None,
    *,
    min_matrix_k_splits: int = 1,
    max_blen: int | None = None,
) -> None:
    records, summary = load_records(run_dir)
    unfiltered_record_count = len(records)
    records = [
        record
        for record in records
        if int(record["MLEN"]) // int(record["BLEN"]) >= min_matrix_k_splits
        and (max_blen is None or int(record["BLEN"]) <= max_blen)
    ]
    if not records:
        raise ValueError(
            "no complete records remain after topology/calibration filtering"
        )
    front = pareto_front(records)

    # Resolve exact ties deterministically in favour of accuracy, then area or
    # latency. This avoids process scheduling deciding the reported points.
    min_latency = min(float(r["latency_ms"]) for r in records)
    lowest_latency = max(
        (r for r in records if math.isclose(float(r["latency_ms"]), min_latency, rel_tol=0, abs_tol=1e-9)),
        key=lambda r: (float(r["accuracy_score"]), -float(r["area_mm2"])),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
        }
    )

    fig = plt.figure(figsize=(11.4, 8.2), facecolor="white")
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.5, 1.65], hspace=0.30)
    ax = fig.add_subplot(grid[0])
    text_ax = fig.add_subplot(grid[1])
    text_ax.axis("off")

    for family in ("MXINT", "MXFP", "Other"):
        for weight, marker in WEIGHT_MARKERS.items():
            subset = [
                r
                for r in records
                if precision_family(r) == family and weight_precision(r) == weight
            ]
            if not subset:
                continue
            ax.scatter(
                [float(r["area_mm2"]) for r in subset],
                [float(r["latency_ms"]) for r in subset],
                s=20,
                alpha=0.45,
                marker=marker,
                linewidth=0.25,
                edgecolor="white",
                color=FAMILY_STYLE[family]["color"],
                rasterized=True,
            )

    front_sorted = sorted(front, key=lambda r: float(r["area_mm2"]))
    ax.plot(
        [float(r["area_mm2"]) for r in front_sorted],
        [float(r["latency_ms"]) for r in front_sorted],
        color="#202020",
        linewidth=1.35,
        alpha=0.82,
        label="2D Pareto frontier",
        zorder=4,
    )

    target_area = float(summary.get("target_area_mm2", 826.0))
    area_budget = float(summary.get("area_budget_mm2", target_area * 1.1))
    closest_distance = min(abs(float(r["area_mm2"]) - target_area) for r in records)
    closest_a100 = max(
        (
            r
            for r in records
            if math.isclose(
                abs(float(r["area_mm2"]) - target_area),
                closest_distance,
                rel_tol=0,
                abs_tol=1e-9,
            )
        ),
        key=lambda r: (float(r["accuracy_score"]), -float(r["latency_ms"])),
    )
    ax.axvline(target_area, color="#666666", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axvline(area_budget, color="#A23B3B", linestyle=":", linewidth=1.0, alpha=0.85)
    ax.text(
        target_area,
        0.08,
        "A100 826 mm²",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        rotation=90,
        fontsize=8.3,
        color="#555555",
    )
    ax.text(
        area_budget,
        0.08,
        "Area budget 908.6 mm²",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        rotation=90,
        fontsize=8.3,
        color="#8A3030",
    )

    latency_label = (
        "Lowest modeled latency (extrapolated)"
        if int(lowest_latency["BLEN"]) > 16
        else "Lowest modeled latency"
    )
    annotate_point(ax, lowest_latency, latency_label, "*", "#8A5AB5", (-180, 27))
    annotate_point(ax, closest_a100, "Closest to A100 area", "D", "#D6A514", (-150, 66))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Estimated PLENA chip area (mm²)")
    ax.set_ylabel("Qwen3-32B prefill latency (ms)")
    ax.set_title(title or f"Qwen3-32B Dense Full Sweep: {_model_title(summary)}")
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.30)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.20)
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=FAMILY_STYLE[family]["color"],
            markeredgecolor="white",
            markersize=6,
            label=f"{family} profiles",
        )
        for family in ("MXINT", "MXFP")
    ]
    weight_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#E8E8E8",
            markeredgecolor="#333333",
            markersize=6,
            label=f"GPTQ W: {weight.replace('MXFP_', '')}",
        )
        for weight, marker in WEIGHT_MARKERS.items()
    ]
    pareto_handle = Line2D([0], [0], color="#202020", linewidth=1.35, label="2D Pareto frontier")
    ax.legend(
        handles=family_handles + [pareto_handle] + weight_handles,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.99),
        ncol=3,
        columnspacing=1.15,
        handletextpad=0.55,
        frameon=True,
        framealpha=0.96,
    )

    total_trials = int(summary.get("n_trials", len(records)))
    pruned = int(summary.get("pruned", 0))
    failed = int(summary.get("failed", 0))
    text_ax.text(
        0.5,
        0.89,
        f"Plotted {len(records):,} / {unfiltered_record_count:,} complete designs  |  "
        f"grid {total_trials:,}  |  pruned {pruned:,}  |  failed attempts {failed:,}",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
    )
    text_ax.text(
        0.5,
        0.69,
        "Area: calibrated precision-aware logic + ASAP7 SRAM macros; excludes HBM stacks, PHY and package. "
        "Large MLEN/BLEN points are extrapolated. "
        f"Plot filter: MLEN/BLEN >= {min_matrix_k_splits}"
        + ("" if max_blen is None else f", BLEN <= {max_blen}"),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
    )
    for index, line in enumerate(
        [
            knob_summary("Lowest latency", lowest_latency),
            knob_summary("Closest to A100", closest_a100),
        ]
    ):
        text_ax.text(
            0.015,
            0.43 - index * 0.22,
            line,
            ha="left",
            va="center",
            fontsize=8.8,
            family="DejaVu Sans Mono",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--min-matrix-k-splits",
        type=int,
        default=1,
        help="Filter designs with fewer flattened MatrixMachine K-splits",
    )
    parser.add_argument(
        "--max-blen",
        type=int,
        default=None,
        help="Optional BLEN ceiling for a calibration-domain view",
    )
    args = parser.parse_args()
    output = args.output or args.run_dir / "qwen3_32b_latency_area_scatter.png"
    plot(
        args.run_dir,
        output,
        args.title,
        min_matrix_k_splits=args.min_matrix_k_splits,
        max_blen=args.max_blen,
    )
    print(output)


if __name__ == "__main__":
    main()
