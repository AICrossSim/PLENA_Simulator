"""Export two source-grounded prospective analytical 3D DSE figures.

Run after study.py finishes:
  python -m geometry3d.figures --input RESULTS_DIR --out FIGURE_DIR

Only recorded CSV scores are plotted.  The first figure combines the full
development space with frozen-hardware timing sensitivity.  The second
compares selected and fixed geometries on the recorded heldout batches.
No area, PPA, physical frequency, or silicon result is inferred.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

FAMILIES = ("single", "homogeneous", "heterogeneous")
FAMILY_NAMES = {"single": "Single core", "homogeneous": "Homogeneous dual", "heterogeneous": "Heterogeneous dual"}
FAMILY_COLORS = {"single": "#245987", "homogeneous": "#358160", "heterogeneous": "#bd6830"}
FIXED_LABELS = {"single": "fixed_6", "homogeneous": "fixed_3+3", "heterogeneous": "fixed_4+2"}
BATCHES = (2, 4, 8, 16, 64, 96, 128)
TIMINGS = ("conservative_flat20", "log2_stage1", "log2_stage2", "log2_stage4")
TIMING_NAMES = ("Flat 20", "Log₂ slope 1", "Log₂ slope 2\n(nominal)", "Log₂ slope 4")
SCOPE = "Analytical / prospective / hypothetical 1 GHz"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _true(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"unexpected boolean CSV value {value!r}")
    return value.lower() in {"true", "1"}


def _pk_config(geometry: str) -> tuple[int, ...]:
    parts = [tuple(map(int, p.split("x"))) for p in geometry.split("+")]
    if not parts or any(len(p) != 3 or min(p) < 1 for p in parts):
        raise ValueError(f"invalid geometry {geometry!r}")
    return tuple(sorted({p[2] for p in parts}))


def _finite_positive(value: str, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a recorded positive finite value")
    return result


def load_inputs(directory: Path) -> dict[str, Any]:
    """Validate recorded inputs and frozen geometry identity before plotting."""
    paths = {"development": directory / "all_geometry_development.csv",
             "heldout": directory / "heldout_summary.csv",
             "sensitivity": directory / "timing_sensitivity.csv",
             "frozen": directory / "FROZEN_SELECTION.json"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"study output not ready: {path}")
    development = read_csv(paths["development"])
    heldout = read_csv(paths["heldout"])
    sensitivity = read_csv(paths["sensitivity"])
    frozen = json.loads(paths["frozen"].read_text())
    points = {p["label"]: p for p in frozen["points"]}
    if len(points) != len(frozen["points"]):
        raise ValueError("duplicate frozen selection labels")
    labels = {"selected_" + family for family in FAMILIES} | set(FIXED_LABELS.values())
    if not labels <= points.keys():
        raise ValueError("all three selected families and fixed comparators are required")
    development_ids = set()
    legal_count = 0
    for row in development:
        if row["geometry"] in development_ids:
            raise ValueError("duplicate geometry in fullspace development CSV")
        development_ids.add(row["geometry"])
        if row["family"] not in FAMILIES:
            raise ValueError("unexpected geometry family")
        config = _pk_config(row["geometry"])
        if _true(row["different_PK"]) != (len(config) > 1):
            raise ValueError("unequal-PK marker metadata disagrees with geometry")
        if _true(row["legal"]):
            legal_count += 1
            _finite_positive(row["development_total_ms"], "development_total_ms")
    if frozen.get("legal_geometries", legal_count) != legal_count:
        raise ValueError("frozen legal count disagrees with development CSV")
    for rows in (heldout, sensitivity):
        for row in rows:
            if row["label"] not in points or row["geometry"] != points[row["label"]]["geometry"]:
                raise ValueError("reported geometry differs from frozen selection")
            _finite_positive(row["total_ms"], "total_ms")
            _finite_positive(row["layers"], "layers")
    selected_sensitivity = {(r["label"], r["timing"]): r for r in sensitivity if r["label"] in labels}
    if len(selected_sensitivity) != sum(r["label"] in labels for r in sensitivity):
        raise ValueError("duplicate selected timing-sensitivity record")
    for family in FAMILIES:
        for timing in TIMINGS:
            if ("selected_" + family, timing) not in selected_sensitivity:
                raise ValueError(f"missing selected {family}/{timing} sensitivity")
    by_batch = {(r["label"], int(r["batch"])): r for r in heldout
                if r["label"] in labels and r["batch"] != "all"}
    if len(by_batch) != sum(r["label"] in labels and r["batch"] != "all" for r in heldout):
        raise ValueError("duplicate selected/fixed heldout batch record")
    for batch in BATCHES:
        for label in labels:
            if (label, batch) not in by_batch:
                raise ValueError(f"missing heldout {label}/B{batch}")
    return {"paths": paths, "development": development, "heldout": heldout,
            "sensitivity": sensitivity, "frozen": frozen, "points": points,
            "by_batch": by_batch, "by_timing": selected_sensitivity,
            "legal_count": legal_count}


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    paths = [directory / (stem + suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _jitter(geometry: str) -> float:
    unit = int(hashlib.sha256(geometry.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 0.42


def development_and_timing(data: dict[str, Any], directory: Path) -> list[Path]:
    fig = plt.figure(figsize=(17.2, 10.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.45, 1.0), hspace=0.46, wspace=0.27)
    legal = [r for r in data["development"] if _true(r["legal"])]
    for column, family in enumerate(FAMILIES):
        ax = fig.add_subplot(grid[0, column])
        rows = [r for r in legal if r["family"] == family]
        if not rows:
            raise ValueError(f"no legal fullspace designs in family {family}")
        configs = sorted({_pk_config(r["geometry"]) for r in rows},
                         key=lambda k: (max(k), min(k), len(k)))
        x = {config: idx for idx, config in enumerate(configs)}
        for unequal, marker in ((False, "o"), (True, "^")):
            group = [r for r in rows if (len(_pk_config(r["geometry"])) > 1) == unequal]
            if group:
                ax.scatter([x[_pk_config(r["geometry"])] + _jitter(r["geometry"]) for r in group],
                           [float(r["development_total_ms"]) for r in group],
                           s=11 if not unequal else 16, marker=marker, alpha=0.24,
                           color=FAMILY_COLORS[family], linewidths=0, rasterized=True)
        selected = data["points"]["selected_" + family]["geometry"]
        selected_rows = [r for r in rows if r["geometry"] == selected]
        if len(selected_rows) != 1:
            raise ValueError("frozen selected hardware absent from legal primary screen")
        row = selected_rows[0]
        ax.scatter([x[_pk_config(selected)] + _jitter(selected)], [float(row["development_total_ms"])],
                   marker="*", s=150, c="#111111", edgecolors="white", linewidths=.8, zorder=5)
        ax.set_xticks(range(len(configs)), ["+".join(map(str, p)) for p in configs],
                      rotation=65 if len(configs) > 8 else 0, ha="right" if len(configs) > 8 else "center")
        ax.set_xlabel("Physical PK configuration (elements)")
        ax.set_title(f"{FAMILY_NAMES[family]} — {len(rows):,} legal designs", fontsize=12)
        ax.set_ylabel("Development total estimated latency (ms)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[1, :])
    positions = np.arange(len(TIMINGS), dtype=float)
    width = .23
    for idx, family in enumerate(FAMILIES):
        values = [float(data["by_timing"][("selected_" + family, timing)]["total_ms"])
                  for timing in TIMINGS]
        ax.bar(positions + (idx - 1) * width, values, width=width,
               color=FAMILY_COLORS[family], label=FAMILY_NAMES[family])
    ax.set_xticks(positions, TIMING_NAMES)
    ax.set_ylabel("Heldout total estimated latency (ms)")
    ax.set_title("Timing sensitivity of frozen selected hardware — no fullspace reranking", fontsize=12)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Full declared 3D design space and timing sensitivity\n" + SCOPE,
                 fontsize=16, y=.98)
    legend = [Line2D([0], [0], marker="o", linestyle="", color="#555555", markersize=5,
                     label="One PK value"),
              Line2D([0], [0], marker="^", linestyle="", color="#555555", markersize=6,
                     label="Two different PK values"),
              Line2D([0], [0], marker="*", linestyle="", color="#111111", markersize=10,
                     label="Frozen selected hardware; primary-screen score")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .018), ncol=3, frameon=False)
    excluded = len(data["development"]) - data["legal_count"]
    fig.text(.5, -.006,
             f"{len(data['development']):,} declared geometries; {data['legal_count']:,} legal; {excluded:,} excluded. "
             "Horizontal jitter separates recorded points. Equal multipliers; no area/PPA result.",
             ha="center", fontsize=9, color="#555555")
    return _save(fig, directory, "fullspace_development_and_timing")


def heldout_by_batch(data: dict[str, Any], directory: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.0))
    selected_patch = Patch(facecolor="#555555", label="Selected on development")
    fixed_patch = Patch(facecolor="#555555", alpha=.32, hatch="///", label="Fixed comparator")
    for ax, batch in zip(axes.flat, BATCHES):
        positions = np.arange(3, dtype=float)
        fixed_values, selected_values = [], []
        for family in FAMILIES:
            fixed = data["by_batch"][(FIXED_LABELS[family], batch)]
            selected = data["by_batch"][("selected_" + family, batch)]
            if fixed["layers"] != selected["layers"]:
                raise ValueError("paired batch comparisons require the same layer count")
            fixed_values.append(float(fixed["total_ms"]) / float(fixed["layers"]))
            selected_values.append(float(selected["total_ms"]) / float(selected["layers"]))
        colors = [FAMILY_COLORS[f] for f in FAMILIES]
        ax.bar(positions - .19, fixed_values, width=.36, color=colors, alpha=.32,
               hatch="///", edgecolor=colors, linewidth=.7)
        ax.bar(positions + .19, selected_values, width=.36, color=colors)
        ax.set_xticks(positions, ("Single", "Homo dual", "Hetero dual"), fontsize=9)
        ax.set_title(f"B{batch}", fontsize=13)
        ax.set_ylabel("Mean estimated layer latency (ms)", fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles=(fixed_patch, selected_patch), loc="upper left", frameon=False, fontsize=10)
    lines = ["Frozen geometries (PM × PN × PK)"]
    for family in FAMILIES:
        selected = data["points"]["selected_" + family]
        fixed = data["points"][FIXED_LABELS[family]]
        lines.extend(["", FAMILY_NAMES[family],
                      "Fixed: " + fixed["geometry"].replace("x", "×").replace("+", " + "),
                      "Selected: " + selected["geometry"].replace("x", "×").replace("+", " + ")])
    legend_ax.text(.02, .77, "\n".join(lines), transform=legend_ax.transAxes,
                   fontsize=9, va="top", linespacing=1.35)
    fig.suptitle("Heldout batch comparison: fixed and development-selected geometries\n" + SCOPE,
                 fontsize=16, y=1.02)
    fig.text(.5, -.025,
             "Bars use recorded total_ms / layers; each panel has its own latency scale. "
             "Fixed comparators are rerun in the same analytical model. Historically exposed captures are not a pristine blind test.",
             fontsize=9, color="#555555", ha="center")
    fig.tight_layout(h_pad=2.0, w_pad=1.3)
    return _save(fig, directory, "heldout_batch_selected_vs_fixed")


def create_figures(input_directory: Path, output_directory: Path) -> dict[str, Any]:
    data = load_inputs(input_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    paths = (development_and_timing(data, output_directory)
             + heldout_by_batch(data, output_directory))
    manifest = {"scope": SCOPE, "figures": [str(p.resolve()) for p in paths],
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in data["paths"].values()},
                "metrics": {"development": "development_total_ms under primary screening settings",
                            "timing": "total_ms for frozen selected families across recorded hypotheses",
                            "heldout": "total_ms/layers per recorded batch"},
                "not_claimed": ["area Pareto frontier", "PPA", "silicon measurements", "physical frequency validation"]}
    (output_directory / "figure_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_figures(args.input, args.out)
    print(json.dumps({"figures": manifest["figures"]}, sort_keys=True))


if __name__ == "__main__":
    main()
