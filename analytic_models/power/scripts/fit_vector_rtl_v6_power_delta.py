#!/usr/bin/env python3
"""Fit fail-closed rtl-v6 row-engine and packed-PV logic energy.

The activity campaign measures the production module-level integration
boundary, but the ordinary SRAM access energy is already charged from ASAP7
Liberty in ``power_model.py``.  This fitter therefore uses only the mapped
``register + combinational`` power groups and deliberately excludes the clock
network and any inferred-memory power group.  An incomplete or inconsistent
campaign produces a diagnostic artifact which the runtime refuses to use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION = (
    ROOT / "analytic_models/power/calibration/vector_rtl_v6_power_delta.json"
)


ROW_FAMILIES = {
    "row_max": ("vector.reduction_max_rows", "active_elements"),
    "row_sum": ("vector.reduction_sum_rows", "active_elements"),
    "row_sub": ("vector.softmax_row_subtract", "active_elements"),
    "row_exp": ("vector.softmax_row_exp", "active_elements"),
    "row_mul_stats": ("vector.softmax_row_multiply", "active_elements"),
    "state_max": ("softmax_state.max_update", "active_rows"),
    "state_sum": ("softmax_state.sum_update", "active_rows"),
    "state_final": ("softmax_state.final_reciprocal", "active_rows"),
}
PACKED_PV_FAMILIES = {
    "packed_pv_overwrite": "packed_pv_accumulator.overwrite",
    "packed_pv_accumulate": "packed_pv_accumulator.accumulate",
}


def _latest_complete(
    inputs: Path | Iterable[Path],
) -> list[dict[str, str]]:
    paths = [inputs] if isinstance(inputs, Path) else list(inputs)
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                latest[(row["point_key"], row["scenario"])] = row
    return [row for row in latest.values() if row.get("status") == "complete"]


def _logic_energy(row: dict[str, str]) -> float:
    """Return non-clock, non-SRAM mapped logic energy for one replay."""

    required = ("register_dynamic_energy_pj", "combinational_dynamic_energy_pj")
    if any(row.get(name) in (None, "") for name in required):
        raise ValueError(
            "rtl-v6 power rows require register and combinational energy fields"
        )
    return math.fsum(float(row[name]) for name in required)


def _integer(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        params = json.loads(row.get("params_json") or "{}")
        value = params.get(key)
    if value in (None, ""):
        match = re.search(
            {"ROW_LANES": r"_r(\d+)_", "WRITE_LANES": r"_b(\d+)_"}[key],
            row["point_id"],
        )
        if match:
            value = match.group(1)
    if value in (None, ""):
        raise ValueError(f"missing {key} for {row.get('point_id')}")
    return int(value)


def _features(row: dict[str, str]) -> dict[str, float]:
    payload = json.loads(row.get("features_json") or "{}")
    if isinstance(payload.get("dynamic_features"), dict):
        payload = payload["dynamic_features"]
    return {str(name): float(value) for name, value in payload.items()}


def _matched_idle(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["microkernel"] == "idle":
            result[(row["point_key"], int(row["repeat_count"]))] = _logic_energy(row)
    return result


def _normalized_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    idle = _matched_idle(rows)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        family = row["microkernel"]
        if family == "idle":
            continue
        repeat_count = int(row["repeat_count"])
        baseline = idle.get((row["point_key"], repeat_count))
        if baseline is None:
            continue
        params = json.loads(row.get("params_json") or "{}")
        if not params:
            feature_payload = json.loads(row.get("features_json") or "{}")
            embedded_params = feature_payload.get("params", {})
            if isinstance(embedded_params, dict):
                params = embedded_params
        row_lanes = _integer(row, "ROW_LANES") if row["component"] == "softmax_v6" else 0
        write_lanes = _integer(row, "WRITE_LANES") if row["component"] == "packed_pv_v6" else 0
        vlen_match = re.search(r"_v(\d+)_(?:r|b)\d+_", row["point_id"])
        if params.get("VLEN") in (None, "") and vlen_match is None:
            raise ValueError(f"missing VLEN for {row.get('point_id')}")
        vlen = int(params.get("VLEN") or vlen_match.group(1))
        incremental = _logic_energy(row) - baseline
        normalized.append(
            {
                "point_id": row["point_id"],
                "point_key": row["point_key"],
                "component": row["component"],
                "scenario": row["scenario"],
                "pattern": row["pattern"],
                "microkernel": family,
                "repeat_count": repeat_count,
                "accepted_actions": int(row.get("accepted_actions") or repeat_count),
                "row_lanes": row_lanes,
                "write_lanes": write_lanes,
                "vlen": vlen,
                "features": _features(row),
                "incremental_logic_energy_pj": incremental,
                "logic_energy_basis": "register_plus_combinational_dynamic_energy_pj",
                "holdout": str(row.get("holdout", "0")).lower() in {"1", "true"},
            }
        )
    return normalized


def _per_command(row: dict[str, Any]) -> float:
    return float(row["incremental_logic_energy_pj"]) / max(
        1, int(row["accepted_actions"])
    )


def _fit_two_point_nonnegative(
    first_units: float,
    first_energy: float,
    second_units: float,
    second_energy: float,
) -> tuple[float, float]:
    if second_units == first_units:
        raise ValueError("training points have identical physical units")
    slope = (second_energy - first_energy) / (second_units - first_units)
    fixed = first_energy - slope * first_units
    if slope < 0 or fixed < 0:
        raise ValueError(f"negative structural coefficient fixed={fixed}, slope={slope}")
    return fixed, slope


def _relative_error(actual: float, predicted: float) -> float:
    return abs(predicted - actual) / max(abs(actual), 1e-12)


def _mixed_prediction(
    row: dict[str, Any], action_models: dict[str, dict[str, float]]
) -> float:
    features = row["features"]
    command_count = max(1, int(row["accepted_actions"]))
    total = 0.0
    for microkernel, (action_key, unit_name) in ROW_FAMILIES.items():
        units = float(features.get(f"softmax_v6.{microkernel}", 0.0))
        if units <= 0:
            continue
        model = action_models[action_key]
        # The mixed sequence carries one command for each nonzero feature
        # family. Its feature amount already includes rows/elements.
        commands = units / (
            row["row_lanes"] * row["vlen"]
            if unit_name == "active_elements"
            else row["row_lanes"]
        )
        total += commands * model["fixed_pj_per_command"]
        total += units * model["variable_pj_per_unit"]
    return total / command_count


def fit(
    input_csv: Path | Iterable[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _latest_complete(input_csv)
    normalized = _normalized_rows(rows)
    failures: list[str] = []

    expected_points = {
        "power_softmax_v6_v32_r1_e6m5",
        "power_softmax_v6_v32_r4_e6m5",
        "power_softmax_v6_v32_r8_e6m5",
        "power_packed_pv_v6_v64_b8_e6m5",
        "power_packed_pv_v6_v64_b16_e6m5",
        "power_packed_pv_v6_v64_b32_e6m5",
    }
    observed_points = {row["point_id"] for row in rows}
    missing_points = sorted(expected_points - observed_points)
    if missing_points:
        failures.append(f"missing planned points: {missing_points}")

    action_models: dict[str, dict[str, float]] = {}
    holdout_errors: dict[str, float] = {}
    for microkernel, (action_key, unit_name) in ROW_FAMILIES.items():
        expected_repeats = 8 if microkernel == "state_final" else 128
        samples = {
            int(row["row_lanes"]): row
            for row in normalized
            if row["microkernel"] == microkernel
            and row["pattern"] == "representative-qwen"
            and row["repeat_count"] == expected_repeats
        }
        if set(samples) != {1, 4, 8}:
            failures.append(
                f"{microkernel}: expected R1/R4/R8 samples, observed {sorted(samples)}"
            )
            continue
        def units(sample: dict[str, Any]) -> float:
            return float(sample["row_lanes"] * (sample["vlen"] if unit_name == "active_elements" else 1))

        try:
            fixed, slope = _fit_two_point_nonnegative(
                units(samples[1]),
                _per_command(samples[1]),
                units(samples[4]),
                _per_command(samples[4]),
            )
        except ValueError as exc:
            failures.append(f"{microkernel}: {exc}")
            continue
        predicted = fixed + slope * units(samples[8])
        error = _relative_error(_per_command(samples[8]), predicted)
        holdout_errors[action_key] = error
        if error > 0.20:
            failures.append(f"{microkernel}: R8 holdout error {error:.2%} exceeds 20%")
        action_models[action_key] = {
            "fixed_pj_per_command": fixed,
            "variable_pj_per_unit": slope,
            "unit": unit_name,
            "train_row_lanes": [1, 4],
            "holdout_row_lanes": [8],
            "holdout_relative_error": error,
        }

    packed_pv: dict[str, dict[str, float | list[int] | str]] = {}
    packed_holdout_errors: dict[str, float] = {}
    for microkernel, action_key in PACKED_PV_FAMILIES.items():
        samples = {
            int(row["write_lanes"]): row
            for row in normalized
            if row["microkernel"] == microkernel
            and row["pattern"] == "representative-qwen"
            and row["repeat_count"] == 128
            and row["vlen"] == 64
        }
        if set(samples) != {8, 16, 32}:
            failures.append(
                f"{microkernel}: expected B8/B16/B32 samples, observed {sorted(samples)}"
            )
            continue
        try:
            fixed, slope = _fit_two_point_nonnegative(
                8.0,
                _per_command(samples[8]),
                16.0,
                _per_command(samples[16]),
            )
        except ValueError as exc:
            failures.append(f"{microkernel}: {exc}")
            continue
        predicted = fixed + slope * 32.0
        error = _relative_error(_per_command(samples[32]), predicted)
        packed_holdout_errors[action_key] = error
        if error > 0.20:
            failures.append(f"{microkernel}: B32 holdout error {error:.2%} exceeds 20%")
        packed_pv[action_key] = {
            "fixed_pj_per_row": fixed,
            "variable_pj_per_active_lane": slope,
            "train_write_lanes": [8, 16],
            "holdout_write_lanes": [32],
            "holdout_relative_error": error,
            "unit": "active_write_lane",
        }

    mixed_errors: list[float] = []
    if len(action_models) == len(ROW_FAMILIES):
        for row in normalized:
            if (
                row["component"] == "softmax_v6"
                and row["microkernel"] == "mixed"
                and row["pattern"] == "representative-qwen"
            ):
                predicted = _mixed_prediction(row, action_models)
                mixed_errors.append(_relative_error(_per_command(row), predicted))
    if not mixed_errors:
        failures.append("missing representative mixed row-engine samples")
    elif max(mixed_errors) > 0.25:
        failures.append(f"mixed row-engine error {max(mixed_errors):.2%} exceeds 25%")

    for pattern in ("representative-qwen", "low-toggle", "random"):
        row_tiers = {
            int(row["row_lanes"])
            for row in normalized
            if row["component"] == "softmax_v6"
            and row["microkernel"] == "mixed"
            and row["pattern"] == pattern
            and row["repeat_count"] == 128
        }
        if row_tiers != {1, 4, 8}:
            failures.append(
                f"row-engine {pattern} mixed: expected R1/R4/R8, "
                f"observed {sorted(row_tiers)}"
            )
        packed_tiers = {
            int(row["write_lanes"])
            for row in normalized
            if row["component"] == "packed_pv_v6"
            and row["microkernel"] == "mixed"
            and row["pattern"] == pattern
            and row["repeat_count"] == 128
            and row["vlen"] == 64
        }
        if packed_tiers != {8, 16, 32}:
            failures.append(
                f"packed-PV {pattern} mixed: expected B8/B16/B32, "
                f"observed {sorted(packed_tiers)}"
            )

    nominal_mixed = {
        (row["component"], row["point_key"]): _per_command(row)
        for row in normalized
        if row["microkernel"] == "mixed"
        and row["pattern"] == "representative-qwen"
        and row["repeat_count"] == 128
    }
    ratios: dict[str, list[float]] = defaultdict(list)
    for row in normalized:
        label = {"low-toggle": "low", "random": "high"}.get(row["pattern"])
        if label is None or row["microkernel"] != "mixed" or row["repeat_count"] != 128:
            continue
        reference = nominal_mixed.get((row["component"], row["point_key"]), 0.0)
        if reference > 0:
            ratios[label].append(max(0.0, _per_command(row) / reference))
    envelope = {
        "low": min([1.0, *ratios["low"]]),
        "nominal": 1.0,
        "high": max([1.0, *ratios["high"]]),
    }

    status = (
        "rtl_activity_calibrated_rtl_v6_logic"
        if not failures
        and len(action_models) == len(ROW_FAMILIES)
        and len(packed_pv) == len(PACKED_PV_FAMILIES)
        else "rtl_v6_power_calibration_failed"
    )
    artifact = {
        "schema_version": "vector_rtl_v6_action_energy_v2",
        "calibration_status": status,
        "measurement_semantics": (
            "matched-idle register-plus-combinational mapped-DC RTL activity; "
            "clock and SRAM dynamic energy excluded"
        ),
        "fp_setting": "FP_E6M5",
        "row_action_models": action_models,
        "packed_pv_action_models": packed_pv,
        "activity_envelope": envelope,
        "validation": {
            "row_holdout_relative_error": holdout_errors,
            "packed_pv_holdout_relative_error": packed_holdout_errors,
            "mixed_max_relative_error": max(mixed_errors, default=None),
        },
        "measured_row_lanes": [1, 4, 8],
        "interpolated_row_lanes": [2],
        "measured_write_lanes": [8, 16, 32],
        "failures": failures,
        "exclusions": [
            "clock-network energy",
            "SRAM dynamic energy (charged separately from ASAP7 Liberty)",
            "CTS",
            "routed parasitics",
        ],
    }
    return artifact, normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Calibration CSV; repeat to combine resumable campaigns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument(
        "--promote-to-calibration",
        action="store_true",
        help="Install the artifact only when every validation check passed.",
    )
    args = parser.parse_args()
    artifact, rows = fit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    fields = (
        "point_id", "point_key", "component", "scenario", "pattern",
        "microkernel", "repeat_count", "accepted_actions", "row_lanes",
        "write_lanes", "vlen", "incremental_logic_energy_pj",
        "logic_energy_basis", "holdout",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if args.promote_to_calibration and status_is_calibrated(artifact):
        DEFAULT_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, DEFAULT_CALIBRATION)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if status_is_calibrated(artifact) else 2


def status_is_calibrated(artifact: dict[str, Any]) -> bool:
    return artifact.get("calibration_status") == "rtl_activity_calibrated_rtl_v6_logic"


if __name__ == "__main__":
    raise SystemExit(main())
