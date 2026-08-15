#!/usr/bin/env python3
"""Build a compact latency/II audit from raw reduction timing measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    audit_rows = [
        row
        for row in raw.get("measurements", ())
        if row.get("measurement_kind") == "pipeline_audit"
    ]
    if not audit_rows:
        raise ValueError("raw artifact contains no pipeline_audit measurements")
    if any(int(row["independent_ii_cycles"]) != 1 for row in audit_rows):
        raise ValueError("independent reduction rows did not sustain II=1")
    for row in audit_rows:
        if int(row["dependent_issue_interval_cycles"]) < int(row["single_latency_cycles"]):
            raise ValueError("dependent reduction was accepted before its producer completed")
    return {
        "schema_version": 1,
        "model": "rtl_v6_reduction_pipeline_audit",
        "source": {
            "raw_measurements": str(raw_path),
            "raw_measurements_sha256": _sha256(raw_path),
            "rtl_head": raw.get("rtl_head"),
            "rtl_dirty": raw.get("rtl_dirty"),
            "measurement_boundary": raw.get("measurement_boundary"),
        },
        "interpretation": {
            "r1_temporal_pipeline": "one independent VLEN row accepted per cycle",
            "dependent_row_policy": "consumer waits for the producer result",
            "r_way_spatial_pipeline": "R independent VLEN rows per cycle only with R banks and R row slices",
            "state_simd_included": False,
            "full_vector_machine_integration_validated": False,
        },
        "measurements": audit_rows,
        "independent_row_ii_one": True,
        "rtl_v6_full_integration_status": "unvalidated",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
