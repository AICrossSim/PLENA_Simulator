#!/usr/bin/env python3
"""Build the reviewed vector/scalar v2 timing artifact from RTL evidence.

The matrix section remains the previously measured full-Machine calibration.
Vector and scalar fields are replaced only after every raw observation agrees
with the compact structural formulas used by the emulator and CostEmitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v1.json"
DEFAULT_RAW = REPO_ROOT / "Workspace/rtl_vector_scalar_v2_calibration/full_machine/raw_measurements.json"
DEFAULT_OUTPUT = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v2.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _records_by_opcode(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw["measurements"]:
        grouped[str(record["opcode"])].append(record)
    return grouped


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


def _measurement_points(raw: dict[str, Any], harness_fragment: str) -> list[dict[str, int]]:
    points = {
        (int(record["vlen"]), int(record["fp_exp"]), int(record["fp_mant"]))
        for record in raw["measurements"]
        if harness_fragment in str(record["harness"])
    }
    return [
        {"vlen": vlen, "exponent": exponent, "mantissa": mantissa}
        for vlen, exponent, mantissa in sorted(points)
    ]


def _validate_reduction_formula(
    records: list[dict[str, Any]], *, base: int, per_level: int
) -> None:
    for record in records:
        opcode = str(record["opcode"])
        if opcode in {"V_RED_SUM", "V_RED_MAX"}:
            levels = (int(record["vlen"]) + 1 - 1).bit_length()
        else:
            levels = int(record["segment_log2"]) + 1
        expected = base + per_level * levels
        actual = int(record["done_cycles"])
        if actual != expected:
            raise ValueError(
                f"{opcode} formula mismatch at {record.get('point_tag')}: "
                f"measured={actual}, expected={expected}"
            )
        if record.get("result_check") != "pass":
            raise ValueError(f"{opcode} lacks a passing numerical check: {record}")


def build(base_path: Path, raw_path: Path) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("failure") is not None:
        raise ValueError(f"raw calibration failed: {raw['failure']}")
    selected = set(raw.get("selected_harnesses", []))
    required = {"vector_machine_full_timing.py", "scalar_machine_full_timing.py"}
    if not required.issubset(selected):
        raise ValueError(f"raw calibration must contain {sorted(required)}, got {sorted(selected)}")

    grouped = _records_by_opcode(raw)
    result = deepcopy(base)
    result["model"] = "plena_rtl_full_machine_timing_v2_vector_scalar_extensions"
    result["source"] = {
        "rtl_repository": raw["rtl_root"],
        "rtl_head": raw["rtl_head"],
        "rtl_dirty": bool(raw["rtl_dirty"]),
        "rtl_diff_sha256": raw["rtl_diff_sha256"],
        "implementation_profile": raw["implementation_profile"],
        "dc_library_profile_verified": bool(raw["dc_library_profile_verified"]),
        "raw_measurements_path": str(raw_path.resolve()),
        "raw_measurements_sha256": _sha256(raw_path),
        "note": (
            "Vector/scalar cycles are full-Machine behavioral RTL measurements. "
            "Segment reductions reuse the existing tree and select an early level. "
            "The 1 ns reporting period is not timing closure. Matrix coefficients "
            "are retained from v1."
        ),
    }

    vector = result["vector"]
    fixed_vector = {
        "add_vv_cycles": "V_ADD_VV",
        "add_vf_cycles": "V_ADD_VF",
        "sub_vv_cycles": "V_SUB_VV",
        "sub_vf_cycles": "V_SUB_VF",
        "mul_vv_cycles": "V_MUL_VV",
        "mul_vf_cycles": "V_MUL_VF",
        "exp_cycles": "V_EXP_VV",
        "reciprocal_cycles": "V_RECI_VV",
    }
    for field, opcode in fixed_vector.items():
        vector[field] = _constant(grouped, opcode, "done_cycles")
    vector["initiation_interval_cycles"] = _constant(
        grouped, "V_ADD_VV_II", "initiation_interval_cycles"
    )
    vector["measured_points"] = _measurement_points(raw, "vector_machine")
    segment_records = [
        record
        for opcode, records in grouped.items()
        if opcode.startswith("V_RED_SUM_SEG") or opcode.startswith("V_RED_MAX_SEG")
        for record in records
    ]
    vector["measured_segment_widths"] = sorted(
        {int(record["segment_width"]) for record in segment_records}
    )
    _validate_reduction_formula(
        grouped["V_RED_SUM"]
        + [record for record in segment_records if str(record["opcode"]).startswith("V_RED_SUM")],
        base=int(vector["reduce_sum_base_cycles"]),
        per_level=int(vector["reduce_sum_per_level_cycles"]),
    )
    _validate_reduction_formula(
        grouped["V_RED_MAX"]
        + [record for record in segment_records if str(record["opcode"]).startswith("V_RED_MAX")],
        base=int(vector["reduce_max_base_cycles"]),
        per_level=int(vector["reduce_max_per_level_cycles"]),
    )

    scalar = result["scalar"]
    scalar_pairs = {
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
    for prefix, opcode in scalar_pairs.items():
        scalar[f"{prefix}_ready_cycles"] = _constant(grouped, opcode, "ready_cycles")
        scalar[f"{prefix}_done_cycles"] = _constant(grouped, opcode, "done_cycles")
    scalar["fp_max_implemented"] = True
    scalar["measured_points"] = _measurement_points(raw, "scalar_machine")

    result["evidence"] = {
        "raw_measurement_count": len(raw["measurements"]),
        "vector_point_count": len(vector["measured_points"]),
        "scalar_point_count": len(scalar["measured_points"]),
        "segment_widths": vector["measured_segment_widths"],
        "all_reduction_numerical_checks_passed": True,
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
