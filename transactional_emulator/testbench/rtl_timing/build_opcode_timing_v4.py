#!/usr/bin/env python3
"""Build the RTL-v4 timing artifact from compact-stat/overwrite evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v3.json"
DEFAULT_RAW = (
    REPO_ROOT
    / "Workspace/rtl_vector_scalar_v4_calibration/full_machine/raw_measurements.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v4.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(raw: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw["measurements"]:
        grouped[str(record["opcode"])].append(record)
    return grouped


def _constant(records: list[dict[str, Any]], field: str) -> int:
    values = {int(record[field]) for record in records}
    if not values:
        raise ValueError(f"no records for {field}")
    if len(values) != 1:
        raise ValueError(f"{field} is not constant: {sorted(values)}")
    return values.pop()


def _operation_records(
    grouped: dict[str, list[dict[str, Any]]], stem: str
) -> list[dict[str, Any]]:
    records = [
        row
        for opcode, rows in grouped.items()
        if opcode.startswith(f"{stem}_L")
        for row in rows
    ]
    if not records:
        raise ValueError(f"raw calibration contains no {stem} records")
    return records


def _point_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["vlen"]), int(row["fp_exp"]), int(row["fp_mant"])


def build(base_path: Path, raw_path: Path) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("failure") is not None:
        raise ValueError(f"raw calibration failed: {raw['failure']}")
    if "vector_machine_full_timing.py" not in set(raw.get("selected_harnesses", [])):
        raise ValueError("vector full-Machine harness is missing")
    failed_checks = [
        (row.get("point_tag"), row.get("opcode"))
        for row in raw["measurements"]
        if row.get("result_check", "pass") != "pass"
    ]
    if failed_checks:
        raise ValueError(f"RTL numerical checks failed: {failed_checks}")

    grouped = _group(raw)
    result = deepcopy(base)
    result["model"] = "plena_rtl_v4_compact_stats_overwrite"
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
            "Compact-stat and overwrite cycles are measured at the production "
            "VectorMachine boundary. The 1 ns conversion remains an architectural "
            "assumption and is not DC timing closure."
        ),
    }

    vector = result["vector"]
    compact: dict[str, Any] = {
        "implemented": True,
        "max_lanes": 16,
    }
    all_compact: list[dict[str, Any]] = []
    for name, stem in (
        ("mul", "V_STAT_MUL_F"),
        ("add", "V_STAT_ADD_F"),
        ("rsqrt", "V_STAT_RSQRT"),
    ):
        records = _operation_records(grouped, stem)
        all_compact.extend(records)
        compact[f"{name}_ready_cycles"] = _constant(records, "ready_cycles")
        compact[f"{name}_done_cycles"] = _constant(records, "done_cycles")
        compact[f"{name}_initiation_interval_cycles"] = _constant(
            records, "initiation_interval_cycles"
        )
    compact["measured_lane_counts"] = sorted(
        {int(row["compact_lanes"]) for row in all_compact}
    )
    compact["measured_points"] = [
        {"vlen": vlen, "exponent": exponent, "mantissa": mantissa}
        for vlen, exponent, mantissa in sorted({_point_key(row) for row in all_compact})
    ]
    vector["compact_stats_simd"] = compact

    overwrite_pairs: list[tuple[str, str]] = [
        ("V_RED_SUM", "V_RED_SUM_OVR"),
        ("V_RED_MAX", "V_RED_MAX_OVR"),
    ]
    for width in (4, 8, 16):
        for operation in ("SUM", "MAX"):
            legacy = f"V_RED_{operation}_SEG_W{width}"
            overwrite = f"V_RED_{operation}_SEG_OVR_W{width}"
            if legacy in grouped and overwrite in grouped:
                overwrite_pairs.append((legacy, overwrite))
    for legacy, overwrite in overwrite_pairs:
        legacy_by_point = {
            _point_key(row): (int(row["ready_cycles"]), int(row["done_cycles"]))
            for row in grouped[legacy]
        }
        overwrite_by_point = {
            _point_key(row): (int(row["ready_cycles"]), int(row["done_cycles"]))
            for row in grouped[overwrite]
        }
        missing_legacy = overwrite_by_point.keys() - legacy_by_point.keys()
        mismatched = {
            point: (legacy_by_point[point], measured)
            for point, measured in overwrite_by_point.items()
            if point in legacy_by_point and legacy_by_point[point] != measured
        }
        if not overwrite_by_point or missing_legacy or mismatched:
            raise ValueError(
                f"{overwrite} timing differs from {legacy}: "
                f"missing_legacy={sorted(missing_legacy)!r}, "
                f"mismatched={mismatched!r}"
            )
    vector["reduction_overwrite"] = {
        "implemented": True,
        "identity": {
            "sum": "positive_zero",
            "max": "compiler_constant_slot_2_negative_60000_quantized",
        },
        "timing_model": "legacy_reduction_structural_equivalent",
        "measured_opcodes": [pair[1] for pair in overwrite_pairs],
        "segment_widths": sorted(
            {
                int(overwrite.rsplit("W", 1)[1])
                for _, overwrite in overwrite_pairs
                if "_SEG_OVR_W" in overwrite
            }
        ),
        "measured_points": [
            {"vlen": vlen, "exponent": exponent, "mantissa": mantissa}
            for vlen, exponent, mantissa in sorted(
                {_point_key(row) for row in grouped["V_RED_SUM_OVR"]}
            )
        ],
    }

    result["evidence"] = {
        **result.get("evidence", {}),
        "v4_raw_measurement_count": len(raw["measurements"]),
        "compact_stats_numerical_checks_passed": True,
        "compact_stats_measured_lane_counts": compact["measured_lane_counts"],
        "compact_stats_measured_point_count": len(compact["measured_points"]),
        "overwrite_numerical_checks_passed": True,
        "overwrite_timing_equivalent": True,
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
    print(f"wrote {args.output} ({result['evidence']['v4_raw_measurement_count']} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
