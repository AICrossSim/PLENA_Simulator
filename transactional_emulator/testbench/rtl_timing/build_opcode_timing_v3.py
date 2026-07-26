#!/usr/bin/env python3
"""Build the RTL-v3 timing artifact from full-Machine behavioral evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v2.json"
DEFAULT_RAW = (
    REPO_ROOT
    / "Workspace/rtl_vector_scalar_v3_calibration/full_machine/raw_measurements.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v3.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw["measurements"]:
        result[str(record["opcode"])].append(record)
    return result


def _constant(
    grouped: dict[str, list[dict[str, Any]]], opcode: str, field: str
) -> int:
    records = grouped.get(opcode, [])
    if not records:
        raise ValueError(f"raw calibration contains no {opcode} records")
    values = {int(record[field]) for record in records}
    if len(values) != 1:
        raise ValueError(f"{opcode}.{field} is not constant: {sorted(values)}")
    return values.pop()


def _fit_log2_latency(
    grouped: dict[str, list[dict[str, Any]]], opcode: str, field: str
) -> tuple[int, int]:
    """Fit an exact integer ``base + per_level*ceil(log2(VLEN))`` model."""
    records = grouped.get(opcode, [])
    if not records:
        raise ValueError(f"raw calibration contains no {opcode} records")
    points = sorted(
        {
            ((int(row["vlen"]) - 1).bit_length(), int(row[field]))
            for row in records
        }
    )
    if len(points) < 2:
        raise ValueError(f"{opcode}.{field} needs at least two VLEN values")
    level0, cycles0 = points[0]
    level1, cycles1 = points[1]
    delta_levels = level1 - level0
    if delta_levels <= 0 or (cycles1 - cycles0) % delta_levels:
        raise ValueError(f"{opcode}.{field} has no integer log2 fit: {points}")
    per_level = (cycles1 - cycles0) // delta_levels
    base = cycles0 - per_level * level0
    for level, cycles in points:
        expected = base + per_level * level
        if cycles != expected:
            raise ValueError(
                f"{opcode}.{field} violates log2 fit at level {level}: "
                f"measured={cycles}, expected={expected}"
            )
    return base, per_level


def _points(raw: dict[str, Any], harness_fragment: str) -> list[dict[str, int]]:
    values = {
        (int(row["vlen"]), int(row["fp_exp"]), int(row["fp_mant"]))
        for row in raw["measurements"]
        if harness_fragment in str(row["harness"])
    }
    return [
        {"vlen": vlen, "exponent": exponent, "mantissa": mantissa}
        for vlen, exponent, mantissa in sorted(values)
    ]


def _require_checks(records: list[dict[str, Any]]) -> None:
    failures = [
        (row.get("point_tag"), row.get("opcode"))
        for row in records
        if "result_check" in row and row["result_check"] != "pass"
    ]
    if failures:
        raise ValueError(f"RTL numerical checks failed: {failures}")


def _validate_multi_reduction(
    records: list[dict[str, Any]], *, base: int, per_level: int, writeback: int
) -> None:
    for row in records:
        expected = base + per_level * int(row["segment_log2"]) + writeback
        if int(row["done_cycles"]) != expected:
            raise ValueError(
                f"{row['opcode']} at {row.get('point_tag')} measured "
                f"{row['done_cycles']} cycles, expected {expected}"
            )
        if int(row["segment_count"]) * int(row["segment_width"]) != int(row["vlen"]):
            raise ValueError(f"invalid compact segment census: {row}")


def build(base_path: Path, raw_path: Path) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("failure") is not None:
        raise ValueError(f"raw calibration failed: {raw['failure']}")
    required = {"vector_machine_full_timing.py", "scalar_machine_full_timing.py"}
    selected = set(raw.get("selected_harnesses", []))
    if not required.issubset(selected):
        raise ValueError(f"missing full-Machine harnesses: {sorted(required - selected)}")

    grouped = _group(raw)
    _require_checks(raw["measurements"])
    result = deepcopy(base)
    result["model"] = "plena_rtl_v3_segment_parallel_scalar_rob"
    result["source"] = {
        "rtl_repository": raw["rtl_root"],
        "rtl_head": raw["rtl_head"],
        "rtl_dirty": bool(raw["rtl_dirty"]),
        "rtl_diff_sha256": raw["rtl_diff_sha256"],
        "implementation_profile": raw["implementation_profile"],
        "dc_library_profile_verified": bool(raw["dc_library_profile_verified"]),
        "raw_measurements_path": str(raw_path.resolve()),
        "raw_measurements_sha256": _sha256(raw_path),
        "supplemental_evidence": raw.get("supplemental_evidence", []),
        "note": (
            "Vector/scalar cycles are full-Machine behavioral measurements. "
            "Multi reductions select balanced-tree intermediate levels; scalar "
            "ready is ROB-forwardable and done is in-order retirement. The 1 ns "
            "conversion is an assumption, not timing closure."
        ),
    }

    vector = result["vector"]
    vector["measured_points"] = _points(raw, "vector_machine")
    vector["multi_reduce_writeback_cycles"] = 1
    multi_sum = [
        row for opcode, rows in grouped.items()
        if opcode.startswith("V_RED_SUM_SEGS_W") for row in rows
    ]
    multi_max = [
        row for opcode, rows in grouped.items()
        if opcode.startswith("V_RED_MAX_SEGS_W") for row in rows
    ]
    _validate_multi_reduction(
        multi_sum,
        base=int(vector["reduce_sum_base_cycles"]),
        per_level=int(vector["reduce_sum_per_level_cycles"]),
        writeback=int(vector["multi_reduce_writeback_cycles"]),
    )
    _validate_multi_reduction(
        multi_max,
        base=int(vector["reduce_max_base_cycles"]),
        per_level=int(vector["reduce_max_per_level_cycles"]),
        writeback=int(vector["multi_reduce_writeback_cycles"]),
    )
    vector["measured_multi_segment_widths"] = sorted(
        {int(row["segment_width"]) for row in multi_sum + multi_max}
    )
    vector["multi_reduce_initiation_interval_cycles"] = max(
        _constant(grouped, opcode, "initiation_interval_cycles")
        for opcode in grouped
        if opcode.startswith(("V_RED_SUM_SEGS_W", "V_RED_MAX_SEGS_W"))
    )
    vector["vseg_add_cycles"] = _constant(grouped, "V_ADD_VSEG_W4", "done_cycles")
    vector["vseg_sub_cycles"] = _constant(grouped, "V_SUB_VSEG_W4", "done_cycles")
    vector["vseg_mul_cycles"] = _constant(grouped, "V_MUL_VSEG_W4", "done_cycles")
    vector["vseg_initiation_interval_cycles"] = max(
        _constant(grouped, opcode, "initiation_interval_cycles")
        for opcode in ("V_ADD_VSEG_W4", "V_SUB_VSEG_W4", "V_MUL_VSEG_W4")
    )
    vector["lane_load_cycles"] = _constant(grouped, "S_LD_VLANE_FP", "done_cycles")
    vector["lane_store_cycles"] = _constant(grouped, "S_ST_VLANE_FP", "done_cycles")
    vector["lane_access_initiation_interval_cycles"] = max(
        _constant(grouped, opcode, "initiation_interval_cycles")
        for opcode in ("S_LD_VLANE_FP", "S_ST_VLANE_FP")
    )
    shift_base, shift_per_level = _fit_log2_latency(
        grouped, "V_SHIFT_V", "done_cycles"
    )
    vector["shift_implemented"] = True
    vector["shift_base_cycles"] = shift_base
    vector["shift_per_level_cycles"] = shift_per_level
    vector["shift_initiation_interval_cycles"] = _constant(
        grouped, "V_SHIFT_V_II", "initiation_interval_cycles"
    )
    vector["measured_shift_vlens"] = sorted(
        {int(row["vlen"]) for row in grouped["V_SHIFT_V"]}
    )
    # Kept for readers of the v1/v2 artifact schema. New estimators use the
    # structural fields above.
    vector["shift_conservative_cycles"] = max(
        int(row["done_cycles"]) for row in grouped["V_SHIFT_V"]
    )

    scalar = result["scalar"]
    scalar["measured_points"] = _points(raw, "scalar_machine")
    scalar["rob_depth"] = 8
    scalar["retirement_width"] = 1
    scalar["register_count"] = 16
    pairs = {
        "fp_add": "S_ADD_FP",
        "fp_sub": "S_SUB_FP",
        "fp_max": "S_MAX_FP",
        "fp_mul": "S_MUL_FP",
        "fp_exp": "S_EXP_FP",
        "fp_reciprocal": "S_RECI_FP",
        "fp_sqrt": "S_SQRT_FP",
        "fp_move": "S_MV_FP",
        "fp_rsqrt": "S_RSQRT_FP",
    }
    for prefix, opcode in pairs.items():
        scalar[f"{prefix}_ready_cycles"] = _constant(grouped, opcode, "ready_cycles")
        scalar[f"{prefix}_done_cycles"] = _constant(grouped, opcode, "done_cycles")
        pipeline_opcode = f"{opcode}_PIPELINE"
        scalar[f"{prefix}_initiation_interval_cycles"] = (
            _constant(grouped, pipeline_opcode, "initiation_interval_cycles")
            if pipeline_opcode in grouped
            else 1
        )
    required_pipeline = {
        "S_ADD_FP_PIPELINE",
        "S_MAX_FP_PIPELINE",
        "S_MUL_FP_PIPELINE",
        "S_EXP_FP_PIPELINE",
        "S_RECI_FP_PIPELINE",
        "S_SQRT_FP_PIPELINE",
        "S_MV_FP_PIPELINE",
    }
    for opcode in required_pipeline:
        if _constant(grouped, opcode, "initiation_interval_cycles") != 1:
            raise ValueError(f"{opcode} failed the II=1 acceptance criterion")
    for opcode in ("S_RAW_FORWARD_CHAIN", "S_MIXED_ROB_RETIREMENT", "S_ROB_FULL"):
        if opcode not in grouped:
            raise ValueError(f"missing scalar ROB evidence {opcode}")

    result["evidence"] = {
        "raw_measurement_count": len(raw["measurements"]),
        "vector_point_count": len(vector["measured_points"]),
        "scalar_point_count": len(scalar["measured_points"]),
        "multi_segment_widths": vector["measured_multi_segment_widths"],
        "shift_vlens": vector["measured_shift_vlens"],
        "shift_numerical_checks_passed": True,
        "shift_initiation_interval_measured": True,
        "all_numerical_checks_passed": True,
        "scalar_independent_ii_one": True,
        "scalar_raw_forwarding_passed": True,
        "scalar_mixed_retirement_passed": True,
        "scalar_rob_full_passed": True,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.base.resolve(), args.raw.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({result['evidence']['raw_measurement_count']} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
