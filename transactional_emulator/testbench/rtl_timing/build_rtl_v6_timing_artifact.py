#!/usr/bin/env python3
"""Build a fail-closed rtl-v6 production-VectorMachine timing artifact.

Synthesis WNS and functional initiation interval are deliberately separate.
The artifact is a physical shadow for ideal-II1 DSE and never claims SimTop or
full-core timing closure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED = {
    (lanes, period)
    for lanes in (1, 4, 8)
    for period in (1000, 1250, 1500)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_rows(path: Path) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "complete":
                continue
            key = str(row.get("point_key") or row.get("point_id"))
            latest[key] = row
    return list(latest.values())


def build(
    timing_csv: Path,
    *,
    pipeline_audit: Path | None = None,
) -> dict[str, Any]:
    rows = _complete_rows(timing_csv)
    by_point: dict[tuple[int, int], dict[str, Any]] = {}
    failures: list[str] = []
    for row in rows:
        lanes = int(float(row["SOFTMAX_ROW_LANES"]))
        period_ps = int(float(row["clock_period_ps"]))
        key = (lanes, period_ps)
        if key not in EXPECTED:
            continue
        wns_ns = float(row["wns_ns"])
        by_point[key] = {
            "row_lanes": lanes,
            "target_period_ps": period_ps,
            "wns_ns": wns_ns,
            "critical_period_ns": period_ps / 1000.0 - wns_ns,
            "timing_closed": wns_ns >= 0.0,
            "point_id": row.get("point_id"),
            "report_dir": row.get("report_dir"),
        }
    missing = sorted(EXPECTED - set(by_point))
    if missing:
        failures.append(f"missing production timing points: {missing}")

    functional: dict[str, Any] | None = None
    if pipeline_audit is not None:
        functional = json.loads(pipeline_audit.read_text())
        if not functional.get("independent_row_ii_one"):
            failures.append("R1 functional reduction audit does not establish II=1")

    tier_summary: dict[str, Any] = {}
    for lanes in (1, 4, 8):
        tier_rows = [
            by_point[(lanes, period)]
            for period in (1000, 1250, 1500)
            if (lanes, period) in by_point
        ]
        tier_summary[str(lanes)] = {
            "points": tier_rows,
            "minimum_closed_period_ps": min(
                (
                    int(row["target_period_ps"])
                    for row in tier_rows
                    if row["timing_closed"]
                ),
                default=None,
            ),
            "one_ns_closed": bool(
                by_point.get((lanes, 1000), {}).get("timing_closed", False)
            ),
            "functional_independent_ii_cycles": (
                1 if lanes == 1 and functional is not None else None
            ),
            "functional_spatial_rows_per_cycle": (
                lanes
                if lanes in {1, 4, 8}
                and functional is not None
                and lanes == 1
                else None
            ),
        }

    return {
        "schema_version": "rtl_v6_vector_timing_shadow_v1",
        "calibration_status": (
            "production_vector_timing_candidate"
            if not failures
            else "production_vector_timing_incomplete"
        ),
        "scope": "production VectorMachine synthesis timing; no SimTop",
        "dse_semantics": "physical shadow only; formal DSE remains architectural ideal-II1",
        "technology": "ASAP7 TT 0.7 V 25 C",
        "source": {
            "timing_csv": str(timing_csv),
            "timing_csv_sha256": _sha256(timing_csv),
            "pipeline_audit": str(pipeline_audit) if pipeline_audit else None,
            "pipeline_audit_sha256": (
                _sha256(pipeline_audit) if pipeline_audit else None
            ),
        },
        "coverage": {
            "expected_points": len(EXPECTED),
            "observed_points": len(by_point),
            "row_lane_counts": dict(
                Counter(str(lanes) for lanes, _ in by_point)
            ),
        },
        "tiers": tier_summary,
        "functional_pipeline_audit": functional,
        "failures": failures,
        "exclusions": [
            "SimTop timing",
            "full-core routing",
            "CTS",
            "post-route parasitics",
            "functional R4/R8 latency and II unless separately measured",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--pipeline-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build(args.timing_csv, pipeline_audit=args.pipeline_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["calibration_status"].endswith("candidate") else 2


if __name__ == "__main__":
    raise SystemExit(main())
