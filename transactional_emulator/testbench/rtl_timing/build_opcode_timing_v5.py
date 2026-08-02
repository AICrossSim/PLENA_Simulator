#!/usr/bin/env python3
"""Build the RTL-v5 timing artifact from tiered compact-stat measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_opcode_timing_v4 import build as build_v4


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = (
    REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v4.json"
)
DEFAULT_RAW = (
    REPO_ROOT
    / "Workspace/rtl_vector_scalar_v5_calibration/full_machine/raw_measurements.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v5.json"
)


def build(base_path: Path, raw_path: Path) -> dict:
    result = build_v4(base_path, raw_path)
    compact = result["vector"]["compact_stats_simd"]
    measured = set(int(value) for value in compact["measured_lane_counts"])
    required = {1, 4, 8, 16, 32, 64}
    missing = sorted(required - measured)
    if missing:
        raise ValueError(
            f"RTL-v5 calibration is missing compact lane counts {missing}"
        )
    compact["max_lanes"] = 64
    compact["supported_lane_tiers"] = [4, 8, 16, 32, 64]
    compact["lane_scaling"] = "physically_parameterized_tier"
    result["model"] = "plena_rtl_v5_auto_tiered_compact_stats"
    result["source"]["note"] = (
        "Compact-stat ready/done/II are measured at the production "
        "VectorMachine boundary for lane counts through 64. The 1 ns "
        "conversion remains an architectural assumption and is not DC "
        "timing closure."
    )
    result["evidence"].pop("v4_raw_measurement_count", None)
    result["evidence"]["v5_raw_measurement_count"] = len(
        json.loads(raw_path.read_text(encoding="utf-8"))["measurements"]
    )
    result["evidence"]["compact_stats_measured_lane_counts"] = sorted(measured)
    result["evidence"]["compact_stats_32_64_measured"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.base.resolve(), args.raw.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} "
        f"({result['evidence']['v5_raw_measurement_count']} records)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
