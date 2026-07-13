#!/usr/bin/env python3
"""Merge historical Workspace runs into commit-ready calibration datasets.

The merge scans standalone and unified-scheduler runs, selects the latest
successful row for each semantic hardware/precision configuration, backfills
hierarchy data from retained reports, and projects each module onto its own
schema. Failed attempts and operational logs remain in Workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from calibration_csv import read_rows, write_rows
    from run_full_chip_calibration import CSV_FIELDS as FULL_FIELDS
    from run_hbm_system_calibration import CSV_FIELDS as HBM_FIELDS
    from run_matrix_machine_calibration import CSV_FIELDS as MATRIX_FIELDS
    from run_matrix_machine_calibration import parse_hierarchy_area as parse_matrix_hierarchy_area
    from run_scalar_machine_calibration import CSV_FIELDS as SCALAR_FIELDS
    from run_vector_machine_calibration import CSV_FIELDS as VECTOR_FIELDS
    from run_vector_machine_calibration import parse_vector_hierarchy_area
    from validate_full_chip_area_proxy import parse_full_chip_hierarchy
except ModuleNotFoundError:
    from .calibration_csv import read_rows, write_rows
    from .run_full_chip_calibration import CSV_FIELDS as FULL_FIELDS
    from .run_hbm_system_calibration import CSV_FIELDS as HBM_FIELDS
    from .run_matrix_machine_calibration import CSV_FIELDS as MATRIX_FIELDS
    from .run_matrix_machine_calibration import parse_hierarchy_area as parse_matrix_hierarchy_area
    from .run_scalar_machine_calibration import CSV_FIELDS as SCALAR_FIELDS
    from .run_vector_machine_calibration import CSV_FIELDS as VECTOR_FIELDS
    from .run_vector_machine_calibration import parse_vector_hierarchy_area
    from .validate_full_chip_area_proxy import parse_full_chip_hierarchy

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = ROOT / "Workspace"
DEFAULT_CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"


def _legacy_or_runner(row: dict[str, Any], runner: str) -> bool:
    return not row.get("runner") or row.get("runner") == runner


def _all_calibration_points(workspace_root: Path) -> list[Path]:
    return sorted(workspace_root.glob("area_new_*calibration/runs/*/calibration_points.csv")) + sorted(
        workspace_root.glob("area_new_scheduler/runs/*/calibration_points.csv")
    )


def _source_row(row: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    out = dict(row)
    try:
        out["_source_csv"] = str(path.relative_to(root))
    except ValueError:
        out["_source_csv"] = str(path)
    return out


def _latest_complete(paths: list[Path], root: Path, predicate, key_fn) -> list[dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: item.stat().st_mtime if item.exists() else 0):
        for row in read_rows(path):
            if row.get("status") != "complete" or not predicate(row):
                continue
            latest[key_fn(row)] = _source_row(row, path, root)
    rows = list(latest.values())
    rows.sort(key=lambda row: (row.get("runner", ""), row.get("mode", ""), row.get("level", ""), row.get("MLEN", ""), row.get("BLEN", ""), row.get("VLEN", ""), row.get("FP_SETTING", ""), row.get("point_id", "")))
    return rows


def _resolve_report(path_text: str) -> Path:
    path = Path(path_text) / "area.rpt"
    if path.exists():
        return path
    return ROOT / path


def _backfill_matrix_hierarchy(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("hier_total_area") or row.get("level") != "matrix_machine":
            continue
        report_dir = row.get("report_dir")
        if not report_dir:
            continue
        hierarchy = parse_matrix_hierarchy_area(_resolve_report(str(report_dir)))
        for key, value in hierarchy.items():
            row[key] = value


def _backfill_vector_hierarchy(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("hier_total_area"):
            continue
        report_dir = row.get("report_dir")
        if not report_dir:
            continue
        hierarchy = parse_vector_hierarchy_area(_resolve_report(str(report_dir)))
        for key, value in hierarchy.items():
            row[key] = value


def _backfill_full_chip_hierarchy(rows: list[dict[str, Any]]) -> None:
    field_map = {
        "hier_plena_area": "plena",
        "hier_matrix_machine_area": "matrix_machine_init",
        "hier_vector_machine_area": "vector_machine_init",
        "hier_scalar_machine_area": "scalar_machine_init",
        "hier_scalar_fp_sram_wrapper_area": "scalar_machine_init/fp_scalar_sram",
        "hier_scalar_int_sram_wrapper_area": "scalar_machine_init/int_scalar_sram",
        "hier_hbm_system_area": "hbm_interface_init",
        "hier_matrix_sram_wrapper_area": "matrix_sram",
        "hier_vector_sram_wrapper_area": "vector_sram",
    }
    for row in rows:
        if row.get("hier_plena_area"):
            continue
        report_dir = row.get("report_dir")
        if not report_dir:
            continue
        hierarchy = parse_full_chip_hierarchy(_resolve_report(str(report_dir)))
        for field, hierarchy_name in field_map.items():
            row[field] = hierarchy.get(hierarchy_name, 0.0)


def _write_compact(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write module-local fields without leaking the unified raw CSV schema."""
    compact_fields = [*fields, "_source_csv"]
    projected = [
        {field: row.get(field, "") for field in compact_fields}
        for row in rows
    ]
    write_rows(path, projected, compact_fields)


def merge_calibration_csvs(workspace_root: Path = DEFAULT_WORKSPACE, calibration_dir: Path = DEFAULT_CALIBRATION_DIR) -> dict[str, int]:
    """Regenerate all compact CSV artifacts and return per-module row counts.

    Scalar points without explicit MLEN/VLEN are excluded because their RTL
    shape is ambiguous. Full-chip anchors keep point identity because the
    committed train/validation split is defined by point ID.
    """
    paths = _all_calibration_points(workspace_root)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    matrix_mxint = _latest_complete(
        paths,
        ROOT,
        lambda r: _legacy_or_runner(r, "matrix_machine") and r.get("level") == "matrix_machine" and r.get("mode") == "mxint",
        lambda r: (
            "mxint",
            r.get("MLEN"),
            r.get("BLEN"),
            r.get("ACT_WIDTH"),
            r.get("KV_WIDTH"),
            r.get("WEIGHT_WIDTH"),
            r.get("T_BITS"),
            r.get("L_BITS"),
            r.get("scale_width"),
        ),
    )
    _backfill_matrix_hierarchy(matrix_mxint)
    _write_compact(calibration_dir / "matrix_machine_mxint.csv", matrix_mxint, MATRIX_FIELDS)
    summary["matrix_machine_mxint"] = len(matrix_mxint)

    matrix_mxfp = _latest_complete(
        paths,
        ROOT,
        lambda r: _legacy_or_runner(r, "matrix_machine") and r.get("level") == "matrix_machine" and r.get("mode") == "mxfp",
        lambda r: (
            "mxfp",
            r.get("MLEN"),
            r.get("BLEN"),
            r.get("ACT_WIDTH"),
            r.get("KV_WIDTH"),
            r.get("WEIGHT_WIDTH"),
            r.get("T_EXP"),
            r.get("T_MANT"),
            r.get("L_EXP"),
            r.get("L_MANT"),
            r.get("scale_width"),
        ),
    )
    _backfill_matrix_hierarchy(matrix_mxfp)
    _write_compact(calibration_dir / "matrix_machine_mxfp.csv", matrix_mxfp, MATRIX_FIELDS)
    summary["matrix_machine_mxfp"] = len(matrix_mxfp)

    vector = _latest_complete(
        paths,
        ROOT,
        lambda r: _legacy_or_runner(r, "vector_machine") and r.get("module") == "vector_machine",
        lambda r: (r.get("VLEN"), r.get("FP_SETTING"), r.get("V_FP_EXP_WIDTH"), r.get("V_FP_MANT_WIDTH")),
    )
    _backfill_vector_hierarchy(vector)
    _write_compact(calibration_dir / "vector_machine.csv", vector, VECTOR_FIELDS)
    summary["vector_machine"] = len(vector)

    scalar = _latest_complete(
        paths,
        ROOT,
        lambda r: (
            _legacy_or_runner(r, "scalar_machine")
            and r.get("module") == "scalar_machine"
            and bool(r.get("MLEN"))
            and bool(r.get("VLEN"))
        ),
        lambda r: (
            r.get("MLEN"),
            r.get("VLEN"),
            r.get("INT_DATA_WIDTH"),
            r.get("FP_SETTING"),
            r.get("S_FP_EXP_WIDTH"),
            r.get("S_FP_MANT_WIDTH"),
        ),
    )
    _write_compact(calibration_dir / "scalar_machine.csv", scalar, SCALAR_FIELDS)
    summary["scalar_machine"] = len(scalar)

    hbm = _latest_complete(
        paths,
        ROOT,
        lambda r: _legacy_or_runner(r, "hbm_system") and r.get("module") == "hbm_sys",
        lambda r: (
            r.get("MLEN"),
            r.get("VLEN"),
            r.get("BLEN"),
            r.get("BLOCK_DIM"),
            r.get("ACT_WIDTH"),
            r.get("KV_WIDTH"),
            r.get("WEIGHT_WIDTH"),
            r.get("MX_SCALE_WIDTH"),
            r.get("HBM_M_Prefetch_Amount"),
            r.get("HBM_V_Prefetch_Amount"),
            r.get("HBM_V_Writeback_Amount"),
        ),
    )
    _write_compact(calibration_dir / "hbm_system.csv", hbm, HBM_FIELDS)
    summary["hbm_system"] = len(hbm)

    full = _latest_complete(
        paths,
        ROOT,
        lambda r: _legacy_or_runner(r, "full_chip") and r.get("module") == "plena",
        lambda r: (
            r.get("point_id"),
            r.get("MLEN"),
            r.get("VLEN"),
            r.get("BLEN"),
            r.get("HLEN"),
            r.get("ACT_WIDTH"),
            r.get("KV_WIDTH"),
            r.get("WEIGHT_WIDTH"),
            r.get("FP_SETTING"),
        ),
    )
    _backfill_full_chip_hierarchy(full)
    _write_compact(calibration_dir / "full_chip_anchors.csv", full, FULL_FIELDS)
    summary["full_chip_anchors"] = len(full)

    (calibration_dir / "merged_calibration_csv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    """Run the merge from the command line and print a JSON row summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    args = parser.parse_args()
    print(json.dumps(merge_calibration_csvs(args.workspace_root, args.calibration_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
