#!/usr/bin/env python3
"""Fit six-stream loop-AGU action energy from matched RTL-activity replays."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in csv.DictReader(path.open()):
        if raw["status"] != "complete":
            continue
        row = dict(raw)
        row["energy_pj"] = float(raw["nonclock_dynamic_energy_pj"])
        row["features"] = json.loads(raw["features_json"])["dynamic_features"]
        result[raw["scenario"]] = row
    return result


def _linear_r2(samples: list[tuple[float, float]]) -> float:
    mean_y = sum(y for _, y in samples) / len(samples)
    mean_x = sum(x for x, _ in samples) / len(samples)
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in samples)
    total = sum((y - mean_y) ** 2 for _, y in samples)
    return 1.0 if total == 0 else 1.0 - residual / total


def fit(points: Path) -> dict[str, Any]:
    rows = _rows(points)
    required = {
        "qwen_boundary_6_32", "qwen_boundary_6_128", "qwen_boundary_6_512",
        "qwen_boundary_1_128", "qwen_boundary_3_128",
        "qwen_offset_read_128", "qwen_setup_1_128", "qwen_setup_6_128",
        "low_boundary_6_128", "random_boundary_6_128",
    }
    missing = sorted(required - rows.keys())
    if missing:
        raise ValueError(f"missing AGU calibration scenarios: {missing}")

    offset = rows["qwen_offset_read_128"]
    offset_read_pj = offset["energy_pj"] / offset["features"]["agu.offset_read"]

    boundary_samples: list[tuple[float, float]] = []
    for streams in (1, 3, 6):
        row = rows[f"qwen_boundary_{streams}_128"]
        boundaries = row["features"]["agu.loop_boundary"]
        residual = (
            row["energy_pj"]
            - row["features"]["agu.offset_read"] * offset_read_pj
        ) / boundaries
        boundary_samples.append((float(streams), residual))

    # Nonnegative two-term fit: y = boundary + streams * stream_step.
    n_sum = sum(x for x, _ in boundary_samples)
    y_sum = sum(y for _, y in boundary_samples)
    nn_sum = sum(x * x for x, _ in boundary_samples)
    ny_sum = sum(x * y for x, y in boundary_samples)
    determinant = len(boundary_samples) * nn_sum - n_sum * n_sum
    boundary_pj = (y_sum * nn_sum - n_sum * ny_sum) / determinant
    stream_step_pj = (
        len(boundary_samples) * ny_sum - n_sum * y_sum
    ) / determinant
    if boundary_pj < 0:
        boundary_pj = 0.0
        stream_step_pj = ny_sum / nn_sum
    if stream_step_pj < 0:
        stream_step_pj = 0.0
        boundary_pj = y_sum / len(boundary_samples)

    setup_residuals: dict[int, float] = {}
    for streams in (1, 6):
        row = rows[f"qwen_setup_{streams}_128"]
        count = row["features"]["agu.loop_setup"]
        setup_residuals[streams] = (
            row["energy_pj"] / count
            - boundary_pj
            - streams * stream_step_pj
        )
    config_pj = max(0.0, (setup_residuals[6] - setup_residuals[1]) / 5.0)
    loop_setup_pj = max(0.0, setup_residuals[1] - config_pj)

    scaling = [
        (float(count), rows[f"qwen_boundary_6_{count}"]["energy_pj"])
        for count in (32, 128, 512)
    ]
    nominal = rows["qwen_boundary_6_128"]["energy_pj"]
    low_ratio = rows["low_boundary_6_128"]["energy_pj"] / nominal
    high_ratio = rows["random_boundary_6_128"]["energy_pj"] / nominal

    return {
        "model_version": "loop_agu_action_energy_v1",
        "calibration_status": "rtl_activity_mapped_dc_candidate",
        "generated_at": datetime.now(UTC).isoformat(),
        "corner": {
            "process": "ASAP7_TT",
            "voltage_v": 0.7,
            "temperature_c": 25,
            "clock_period_ps": 1000,
        },
        "dynamic_nominal_pj": {
            "agu_config": config_pj,
            "agu_loop_setup": loop_setup_pj,
            "agu_loop_boundary": boundary_pj,
            "agu_stream_step": stream_step_pj,
            "agu_offset_read": offset_read_pj,
        },
        "activity_envelope": {
            "low": min(1.0, low_ratio),
            "nominal": 1.0,
            "high": max(1.0, high_ratio),
            "semantics": "low-toggle/qwen-like/random empirical ratios",
        },
        "validation": {
            "boundary_6_repeat_scaling_r2": _linear_r2(scaling),
            "saif_sequential_coverage_pct": min(
                float(row["saif_seq_coverage_pct"]) for row in rows.values()
            ),
            "completed_scenarios": len(rows),
            "expected_scenarios": 13,
        },
        "source_points": str(points.resolve()),
        "limitations": [
            "RTL VCD activity replayed on a mapped DC netlist; not gate-level simulation.",
            "Pre-CTS clock-network energy is excluded from dynamic action slopes.",
            "The candidate covers the 32-bit, six-stream, four-level AGU only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = fit(args.points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
