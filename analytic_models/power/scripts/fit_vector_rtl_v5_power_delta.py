#!/usr/bin/env python3
"""Fit per-lane compact-stat energy from focused 32/64-lane replays."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from analytic_models.power.scripts.fit_vector_rtl_v4_power_delta import (
    COMPACT_FAMILIES,
    _action_dynamic_energy,
    _linear_fit,
)


def _latest_rows(path: Path) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest[(row["point_key"], row["scenario"])] = row
    return [
        row
        for row in latest.values()
        if row.get("status") == "complete"
    ]


def _lane_tier(point_id: str) -> int:
    match = re.search(r"_lanes(32|64)_", point_id)
    if not match:
        raise ValueError(f"cannot infer compact lane tier from {point_id!r}")
    return int(match.group(1))


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
        family = row["microkernel"]
        if family not in COMPACT_FAMILIES:
            continue
        repeats = int(row["repeat_count"])
        idle_sample = idle.get((row["point_key"], repeats))
        if idle_sample is None:
            continue
        active_energy, active_basis = _action_dynamic_energy(row)
        idle_energy, idle_basis = idle_sample
        if active_basis != idle_basis:
            continue
        actions = int(row.get("accepted_actions") or repeats)
        lanes = _lane_tier(row["point_id"])
        incremental = active_energy - idle_energy
        normalized.append(
            {
                "point_id": row["point_id"],
                "lane_tier": lanes,
                "scenario": row["scenario"],
                "pattern": row["pattern"],
                "microkernel": family,
                "repeat_count": repeats,
                "accepted_actions": actions,
                "incremental_energy_pj": incremental,
                "energy_per_action_pj": incremental / max(1, actions),
                "energy_per_lane_action_pj": incremental
                / max(1, actions * lanes),
                "energy_basis": active_basis,
            }
        )

    fits: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    failures: list[str] = []
    for family in COMPACT_FAMILIES:
        for lanes in (32, 64):
            selected = [
                row
                for row in normalized
                if row["microkernel"] == family
                and row["lane_tier"] == lanes
                and row["pattern"] == "representative-qwen"
                and row["repeat_count"] in {32, 128, 512}
            ]
            try:
                slope, startup, r2 = _linear_fit(
                    [float(row["accepted_actions"]) for row in selected],
                    [float(row["incremental_energy_pj"]) for row in selected],
                )
            except ValueError as exc:
                failures.append(f"{family}/lanes{lanes}: {exc}")
                continue
            per_lane = slope / lanes
            fits[family][lanes] = {
                "slope_pj_per_action": slope,
                "slope_pj_per_lane_action": per_lane,
                "startup_pj": startup,
                "r2": r2,
            }
            if slope <= 0 or r2 < 0.95:
                failures.append(
                    f"{family}/lanes{lanes}: slope={slope:.6g}, R2={r2:.6f}"
                )

    nominal: dict[str, float] = {}
    envelope: dict[str, dict[str, float]] = {}
    max_tier_residual = 0.0
    for family in COMPACT_FAMILIES:
        tier_fits = fits.get(family, {})
        if set(tier_fits) != {32, 64}:
            failures.append(f"{family}: missing 32/64-lane fit")
            continue
        values = [
            tier_fits[lanes]["slope_pj_per_lane_action"]
            for lanes in (32, 64)
        ]
        mean = sum(values) / len(values)
        nominal[family] = mean
        if mean > 0:
            max_tier_residual = max(
                max_tier_residual,
                *(abs(value - mean) / mean for value in values),
            )

        qwen128 = {
            int(row["lane_tier"]): float(row["energy_per_lane_action_pj"])
            for row in normalized
            if row["microkernel"] == family
            and row["pattern"] == "representative-qwen"
            and row["repeat_count"] == 128
        }
        ratios: dict[str, list[float]] = {"low": [], "high": []}
        for row in normalized:
            if row["microkernel"] != family or row["repeat_count"] != 128:
                continue
            label = {
                "low-toggle": "low",
                "random": "high",
            }.get(row["pattern"])
            if label is None:
                continue
            reference = qwen128.get(int(row["lane_tier"]), 0.0)
            if reference > 0:
                ratios[label].append(
                    max(0.0, float(row["energy_per_lane_action_pj"]) / reference)
                )
        envelope[family] = {
            "low": min([1.0, *ratios["low"]]),
            "nominal": 1.0,
            "high": max([1.0, *ratios["high"]]),
        }

    if max_tier_residual > 0.20:
        failures.append(
            "per-lane 32/64 tier residual exceeds 20%: "
            f"{max_tier_residual:.6f}"
        )
    status = (
        "rtl_activity_calibrated_rtl_v5_tiers"
        if not failures and len(nominal) == len(COMPACT_FAMILIES)
        else "rtl_v5_power_calibration_failed"
    )
    artifact = {
        "schema_version": "vector_rtl_v5_power_delta_v1",
        "calibration_status": status,
        "measurement_semantics": (
            "matched-idle non-clock RTL VCD replay on mapped VectorMachine"
        ),
        "fp_setting": "FP_E6M5",
        "measured_lane_tiers": [32, 64],
        "dynamic_nominal_pj_per_lane_action": nominal,
        "activity_envelope": envelope,
        "tier_fit_metrics": {
            family: {str(lanes): metrics for lanes, metrics in tiers.items()}
            for family, tiers in fits.items()
        },
        "max_per_lane_tier_relative_residual": max_tier_residual,
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
        "lane_tier",
        "scenario",
        "pattern",
        "microkernel",
        "repeat_count",
        "accepted_actions",
        "incremental_energy_pj",
        "energy_per_action_pj",
        "energy_per_lane_action_pj",
        "energy_basis",
    )
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if status_is_calibrated(artifact) else 2


def status_is_calibrated(artifact: dict[str, Any]) -> bool:
    return str(artifact["calibration_status"]).startswith(
        "rtl_activity_calibrated"
    )


if __name__ == "__main__":
    raise SystemExit(main())
