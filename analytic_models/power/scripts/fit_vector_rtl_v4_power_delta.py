#!/usr/bin/env python3
"""Fit compact-stat and reduction-overwrite energy from focused VCD replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPACT_FAMILIES = (
    "compact_stats_mul",
    "compact_stats_add",
    "compact_stats_rsqrt",
)
REDUCTION_PAIRS = {
    "reduce_sum_ovr": "reduce_sum",
    "reduce_max_ovr": "reduce_max",
    "reduce_sum_seg_ovr": "reduce_sum_seg",
    "reduce_max_seg_ovr": "reduce_max_seg",
}


def _latest_rows(path: Path) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest[(row["point_key"], row["scenario"])] = row
    return [row for row in latest.values() if row.get("status") == "complete"]


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 3 or len(set(xs)) < 3:
        raise ValueError("at least three distinct action counts are required")
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    intercept = ybar - slope * xbar
    residual = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)
    )
    total = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return slope, intercept, r2


def _action_dynamic_energy(row: dict[str, str]) -> tuple[float, str]:
    """Return non-clock energy so clock-network drift cannot bias the slope."""

    nonclock = row.get("nonclock_dynamic_energy_pj", "")
    if nonclock not in {"", None}:
        return float(nonclock), "nonclock_dynamic_energy_pj"
    return float(row["window_dynamic_energy_pj"]), "window_dynamic_energy_pj_fallback"


def fit(input_csv: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _latest_rows(input_csv)
    idle: dict[tuple[str, int], tuple[float, str]] = {}
    for row in rows:
        if row["microkernel"] == "idle":
            idle[(row["point_key"], int(row["repeat_count"]))] = (
                _action_dynamic_energy(row)
            )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        microkernel = row["microkernel"]
        if microkernel == "idle":
            continue
        repeats = int(row["repeat_count"])
        idle_sample = idle.get((row["point_key"], repeats))
        if idle_sample is None:
            continue
        idle_energy, idle_basis = idle_sample
        actions = int(row.get("accepted_actions") or repeats)
        active_energy, active_basis = _action_dynamic_energy(row)
        if active_basis != idle_basis:
            continue
        incremental = active_energy - idle_energy
        normalized.append(
            {
                "point_id": row["point_id"],
                "scenario": row["scenario"],
                "pattern": row["pattern"],
                "microkernel": microkernel,
                "repeat_count": repeats,
                "accepted_actions": actions,
                "active_energy_pj": active_energy,
                "matched_idle_energy_pj": idle_energy,
                "incremental_energy_pj": incremental,
                "energy_per_action_pj": incremental / max(1, actions),
                "energy_basis": active_basis,
            }
        )

    nominal: dict[str, float] = {}
    envelope: dict[str, dict[str, float]] = {}
    fit_metrics: dict[str, Any] = {}
    failures: list[str] = []
    for family in COMPACT_FAMILIES:
        selected = [
            row
            for row in normalized
            if row["microkernel"] == family
            and row["pattern"] == "representative-qwen"
            and row["repeat_count"] in {32, 128, 512}
        ]
        try:
            slope, intercept, r2 = _linear_fit(
                [float(row["accepted_actions"]) for row in selected],
                [float(row["incremental_energy_pj"]) for row in selected],
            )
        except ValueError as exc:
            failures.append(f"{family}: {exc}")
            continue
        fit_metrics[family] = {
            "slope_pj_per_action": slope,
            "startup_pj": intercept,
            "r2": r2,
        }
        if slope <= 0 or r2 < 0.95:
            failures.append(f"{family}: slope={slope:.6g}, R2={r2:.6f}")
            continue
        nominal[family] = slope
        qwen128 = next(
            row["energy_per_action_pj"]
            for row in selected
            if row["repeat_count"] == 128
        )
        ratios: dict[str, float] = {}
        for pattern, label in (("low-toggle", "low"), ("random", "high")):
            candidates = [
                row
                for row in normalized
                if row["microkernel"] == family
                and row["pattern"] == pattern
                and row["repeat_count"] == 128
            ]
            ratios[label] = (
                1.0
                if not candidates or qwen128 == 0
                else max(0.0, candidates[0]["energy_per_action_pj"] / qwen128)
            )
        envelope[family] = {
            "low": min(ratios["low"], 1.0),
            "nominal": 1.0,
            "high": max(ratios["high"], 1.0),
        }

    qwen128_by_kernel = {
        row["microkernel"]: float(row["energy_per_action_pj"])
        for row in normalized
        if row["pattern"] == "representative-qwen"
        and row["repeat_count"] == 128
    }
    overwrite_delta: dict[str, float] = {}
    overwrite_status: dict[str, str] = {}
    for overwrite, accumulate in REDUCTION_PAIRS.items():
        if overwrite not in qwen128_by_kernel or accumulate not in qwen128_by_kernel:
            failures.append(f"missing paired reduction scenario {accumulate}/{overwrite}")
            continue
        delta = qwen128_by_kernel[overwrite] - qwen128_by_kernel[accumulate]
        resolution = max(1e-6, 0.01 * abs(qwen128_by_kernel[accumulate]))
        if abs(delta) <= resolution:
            overwrite_delta[overwrite] = 0.0
            overwrite_status[overwrite] = "below_measurement_resolution"
        elif delta < 0:
            overwrite_delta[overwrite] = 0.0
            overwrite_status[overwrite] = "nonpositive_paired_delta_clamped"
        else:
            overwrite_delta[overwrite] = delta
            overwrite_status[overwrite] = "measured_positive_delta"

    status = (
        "rtl_activity_calibrated_rtl_v4_delta"
        if not failures and len(nominal) == len(COMPACT_FAMILIES)
        else "rtl_v4_power_calibration_failed"
    )
    artifact = {
        "schema_version": "vector_rtl_v4_power_delta_v1",
        "calibration_status": status,
        "measurement_semantics": (
            "matched-idle non-clock RTL VCD replay on mapped DC netlist"
        ),
        "fp_setting": "FP_E6M5",
        "dynamic_nominal_pj": nominal,
        "activity_envelope": envelope,
        "reduction_overwrite_delta_pj": overwrite_delta,
        "reduction_overwrite_status": overwrite_status,
        "fit_metrics": fit_metrics,
        "failures": failures,
        "exclusions": [
            "gate-level activity",
            "CTS",
            "routed parasitics",
        ],
    }
    return artifact, normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    artifact, rows = fit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "point_id",
        "scenario",
        "pattern",
        "microkernel",
        "repeat_count",
        "accepted_actions",
        "active_energy_pj",
        "matched_idle_energy_pj",
        "incremental_energy_pj",
        "energy_per_action_pj",
        "energy_basis",
    )
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["calibration_status"].startswith("rtl_activity_calibrated") else 2


if __name__ == "__main__":
    raise SystemExit(main())
