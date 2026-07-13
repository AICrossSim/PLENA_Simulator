#!/usr/bin/env python3
"""Calibrate VectorMachine FP-width and VLEN area scaling with DC.

The runner synthesizes ``vector_machine`` without SRAM bitcells and parses its
hierarchy into element, reduction, buffer, and top-glue targets. Nonnegative
candidate equations are compared before the selected coefficient schema is
written with diagnostics and source provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
except ModuleNotFoundError:
    from .license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
try:
    from run_matrix_machine_calibration import (
        cleanup_worker_build,
        cleanup_workers,
        copy_reports,
        create_worker_copy,
        fit_nonnegative,
        json_safe,
        parse_hierarchy_area,
        parse_area,
        parse_area_from_text,
        parse_power,
        replace_localparam,
        summarize_synth_failure,
    )
except ModuleNotFoundError:
    from .run_matrix_machine_calibration import (
        cleanup_worker_build,
        cleanup_workers,
        copy_reports,
        create_worker_copy,
        fit_nonnegative,
        json_safe,
        parse_hierarchy_area,
        parse_area,
        parse_area_from_text,
        parse_power,
        replace_localparam,
        summarize_synth_failure,
    )

ROOT = Path(__file__).resolve().parents[3]
RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_area_workers")
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"

FP_SETTINGS = {
    "FP_E3M2": (3, 2),
    "FP_E2M3": (2, 3),
    "FP_E6M5": (6, 5),
    "FP_E5M6": (5, 6),
    "FP_E4M7": (4, 7),
    "FP_E8M5": (8, 5),
}

# The vector machine instantiates fp_fix_reciprocal, which maps to the
# Synopsys DW_fp_recip IP under DC_LIB_EN.  The IP rejects very narrow
# reciprocal configurations such as E3M2/E2M3 in our 2024.09 DC setup, so
# calibration anchors avoid those points.  The fitted model can still be used
# to extrapolate to narrower FP formats if a DSE profile needs it.
VECTOR_SYNTH_FP_SETTINGS = {
    name: (exp, mant)
    for name, (exp, mant) in FP_SETTINGS.items()
    if exp >= 3 and mant >= 3
}

CSV_FIELDS = [
    "point_key",
    "point_id",
    "module",
    "top_module",
    "status",
    "worker_id",
    "elapsed_sec",
    "area_um2",
    "dynamic_power",
    "leakage_power",
    "total_power",
    "report_dir",
    "summary_log",
    "failure_reason",
    "VLEN",
    "FP_SETTING",
    "V_FP_EXP_WIDTH",
    "V_FP_MANT_WIDTH",
    "fp_width",
    "preset",
    "hier_total_area",
    "hier_element_area",
    "hier_element_lane_area",
    "hier_reduction_area",
    "hier_reduction_layer_area",
    "hier_buffer_area",
    "hier_top_glue_area",
]


@dataclass(frozen=True)
class Point:
    """One immutable VectorMachine VLEN/FP synthesis point."""
    point_id: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {"module": self.module, "top_module": self.top_module, "params": self.params}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"vector_machine_{digest}")


def fp_width(exp: int, mant: int) -> int:
    """Return sign + exponent + mantissa bits."""
    return 1 + exp + mant


def build_plan(preset: str) -> list[Point]:
    """Build a named VLEN/FP calibration or validation point set."""
    points: list[Point] = []
    if preset == "smoke":
        entries = [(16, "FP_E5M6")]
    elif preset == "minimal-v1":
        entries = [
            (16, "FP_E5M6"),
            (32, "FP_E5M6"),
            (64, "FP_E5M6"),
            (128, "FP_E5M6"),
            (16, "FP_E4M7"),
            (16, "FP_E8M5"),
        ]
    elif preset == "refine-v1":
        entries = [
            (32, "FP_E4M7"),
            (32, "FP_E8M5"),
            (64, "FP_E4M7"),
        ]
    elif preset == "e6m5-v1":
        entries = [
            (16, "FP_E6M5"),
            (32, "FP_E6M5"),
            (64, "FP_E6M5"),
        ]
    elif preset == "tiny-v1":
        entries = [
            (16, "FP_E5M6"),
            (32, "FP_E5M6"),
            (64, "FP_E5M6"),
            (128, "FP_E5M6"),
            (16, "FP_E8M5"),
            (32, "FP_E8M5"),
            (64, "FP_E8M5"),
            (16, "FP_E6M5"),
            (32, "FP_E6M5"),
            (16, "FP_E4M7"),
        ]
    elif preset == "validation":
        entries = [(1024, "FP_E5M6")]
    elif preset == "reduced-v1":
        entries = [(vlen, "FP_E5M6") for vlen in [64, 128, 256, 512]]
        entries.extend((256, setting) for setting in VECTOR_SYNTH_FP_SETTINGS)
        entries = sorted(set(entries), key=lambda item: (item[0], item[1]))
    else:
        raise ValueError(f"unknown preset: {preset}")

    for vlen, setting in entries:
        exp, mant = FP_SETTINGS[setting]
        point_id = f"area_new_vector_v{vlen}_{setting.lower()}"
        points.append(
            Point(
                point_id=point_id,
                module="vector_machine",
                top_module="vector_machine",
                params={
                    "VLEN": vlen,
                    "FP_SETTING": setting,
                    "V_FP_EXP_WIDTH": exp,
                    "V_FP_MANT_WIDTH": mant,
                    "fp_width": fp_width(exp, mant),
                    "preset": preset,
                },
            )
        )
    return points


def point_to_row(
    point: Point,
    *,
    status: str,
    worker_id: int | str = "",
    elapsed_sec: float | str = "",
    area_um2: float | str = "",
    dynamic_power: float | str | None = "",
    leakage_power: float | str | None = "",
    total_power: float | str | None = "",
    report_dir: str = "",
    summary_log: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    """Serialize one VectorMachine synthesis outcome into CSV fields."""
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "point_key": point.point_key,
            "point_id": point.point_id,
            "module": point.module,
            "top_module": point.top_module,
            "status": status,
            "worker_id": worker_id,
            "elapsed_sec": elapsed_sec,
            "area_um2": area_um2,
            "dynamic_power": "" if dynamic_power is None else dynamic_power,
            "leakage_power": "" if leakage_power is None else leakage_power,
            "total_power": "" if total_power is None else total_power,
            "report_dir": report_dir,
            "summary_log": summary_log,
            "failure_reason": failure_reason,
        }
    )
    for key, value in point.params.items():
        if key in row:
            row[key] = value
    return row


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    """Append one standalone-run outcome under a writer lock."""
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_plan_csv(points: list[Point], path: Path) -> None:
    """Persist all planned points before synthesis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point_to_row(point, status="planned"))


def read_completed_keys(path: Path) -> set[str]:
    """Return successful point keys for resumable execution."""
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                completed.add(str(row["point_key"]))
    return completed


def patch_vector_config(point: Point, rtl_root: Path) -> None:
    """Patch VLEN and vector FP widths in an isolated RTL worker."""
    p = point.params
    replace_localparam(rtl_root / "src/definitions/configuration.svh", "VLEN", int(p["VLEN"]))
    precision = rtl_root / "src/definitions/precision.svh"
    replace_localparam(precision, "V_FP_EXP_WIDTH", int(p["V_FP_EXP_WIDTH"]))
    replace_localparam(precision, "V_FP_MANT_WIDTH", int(p["V_FP_MANT_WIDTH"]))


def cleanup_vector_build(worker_rtl: Path) -> None:
    """Remove VectorMachine DC output after copying reports."""
    cleanup_worker_build(worker_rtl, Point("vector_machine", "vector_machine", "vector_machine", {}))


def run_point(
    point: Point,
    worker_id: int,
    worker_rtl: Path,
    rtl_root: Path,
    run_dir: Path,
    cleanup_builds: bool,
    license_retry_wait_sec: float,
    license_max_retries: int,
) -> dict[str, Any]:
    """Patch, synthesize, archive, and clean one VectorMachine point."""
    start = time.time()
    try:
        patch_vector_config(point, worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth vector_machine 1000 area"
        cmd = ["nix", "develop", "-c", "bash", "-lc", synth_cmd]
        log_dir = run_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            attempt += 1
            result = subprocess.run(cmd, cwd=rtl_root, text=True, capture_output=True, check=False)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stderr.log").write_text(result.stderr)
            (log_dir / f"{point.point_key}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.stderr.log").write_text(result.stderr)
            if result.returncode == 0:
                break
            text = f"{result.stdout}\n{result.stderr}"
            if not is_dc_license_unavailable_text(text):
                break
            if license_max_retries > 0 and attempt > license_max_retries:
                break
            print(
                f"[license-busy] worker={worker_id} point={point.point_id} "
                f"attempt={attempt}; retrying in {license_retry_wait_sec:g}s",
                flush=True,
            )
            if cleanup_builds:
                cleanup_vector_build(worker_rtl)
            time.sleep(license_retry_wait_sec)
        if result.returncode != 0:
            stdout_area = parse_area_from_text(result.stdout)
            if stdout_area is not None and "Status: SUCCESS" in result.stdout:
                return point_to_row(
                    point,
                    status="complete",
                    worker_id=worker_id,
                    elapsed_sec=round(time.time() - start, 3),
                    area_um2=stdout_area,
                    failure_reason="synth reported success but report copy/summary failed",
                )
            return point_to_row(
                point,
                status="failed",
                worker_id=worker_id,
                elapsed_sec=round(time.time() - start, 3),
                failure_reason=summarize_synth_failure(result),
            )
        area_report, power_report, summary_log = copy_reports(worker_rtl, point, run_dir)
        area = parse_area(area_report)
        power = parse_power(power_report) if power_report else parse_power(Path("__missing__"))
        return point_to_row(
            point,
            status="complete",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            area_um2=area,
            dynamic_power=power["dynamic_power"],
            leakage_power=power["leakage_power"],
            total_power=power["total_power"],
            report_dir=str(area_report.parent),
            summary_log=str(summary_log or ""),
        )
    except Exception as exc:  # noqa: BLE001
        return point_to_row(
            point,
            status="failed",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            failure_reason=repr(exc),
        )
    finally:
        if cleanup_builds:
            cleanup_vector_build(worker_rtl)


def load_complete_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load successful points with numeric fields normalized for fitting."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete"]


def parse_vector_hierarchy_area(report: Path) -> dict[str, float]:
    """Aggregate DC hierarchy rows into element, reduction, buffer, and glue."""
    rows = parse_hierarchy_area(report)
    # parse_hierarchy_area returns MatrixMachine buckets; reparse its stable row
    # representation by using the report text helper shape here.
    if not report.exists():
        return {}
    import re as _re

    parsed: list[tuple[str, float]] = []
    pending_name: str | None = None
    numeric_line = _re.compile(r"^\s+([0-9.]+)\s+([0-9.]+)\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+")
    inline_line = _re.compile(
        r"^(.{1,34}?)(\s+[0-9.]+\s+[0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+)"
    )
    for line in report.read_text(errors="ignore").splitlines():
        inline = inline_line.match(line)
        if inline and inline.group(1).strip():
            match = numeric_line.match(inline.group(2))
            if match:
                parsed.append((inline.group(1).strip(), float(match.group(1))))
                pending_name = None
                continue
        match = numeric_line.match(line)
        if match and pending_name:
            parsed.append((pending_name, float(match.group(1))))
            pending_name = None
            continue
        if (
            line.strip()
            and not _re.search(r"[0-9]+\.[0-9]+", line)
            and not line.startswith("-")
            and not line.startswith("Hier")
            and not line.startswith("Global")
            and not line.startswith("Total")
            and not line.startswith("Design")
        ):
            pending_name = line.strip()

    values = {name: area for name, area in parsed}
    total = values.get("vector_machine", 0.0)
    element = values.get("element_unit", 0.0)
    reduction = values.get("reduction_unit", 0.0)
    buffers = sum(
        values.get(name, 0.0)
        for name in ["elem_track_fifo", "red_track_fifo", "v_a_buffer", "v_b_buffer", "s_in_buffer", "broadcaset_scalar"]
    )
    element_lanes = sum(
        area
        for name, area in parsed
        if name.startswith("element_unit/parallel_vec_alu_") and name.count("/") == 1
    )
    reduction_layers = sum(
        area
        for name, area in parsed
        if name.startswith("reduction_unit/level_") and name.count("/") == 1
    )
    return {
        "hier_total_area": total,
        "hier_element_area": element,
        "hier_element_lane_area": element_lanes,
        "hier_reduction_area": reduction,
        "hier_reduction_layer_area": reduction_layers,
        "hier_buffer_area": buffers,
        "hier_top_glue_area": max(0.0, total - element - reduction - buffers) if total else 0.0,
    }


def backfill_hierarchy_fields(csv_path: Path) -> None:
    """Populate hierarchy columns for historical rows with retained reports."""
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    changed = False
    for row in rows:
        if row.get("status") != "complete":
            continue
        if row.get("hier_total_area"):
            continue
        report_dir = row.get("report_dir")
        if not report_dir:
            continue
        report = Path(report_dir) / "area.rpt"
        if not report.exists():
            report = ROOT / report
        hierarchy = parse_vector_hierarchy_area(report)
        if not hierarchy:
            continue
        for key, value in hierarchy.items():
            row[key] = value
        changed = True
    if changed:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def vector_mape(rows: list[dict[str, Any]], coeffs: dict[str, float]) -> float:
    """Evaluate hierarchy-composed VectorMachine mean absolute percentage error."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        pred = (
            coeffs["element_exp_lane"] * float(row["feat_exp_lane"])
            + coeffs["element_mant_lane"] * float(row["feat_mant_lane"])
            + coeffs["reduction_exp_tree"] * float(row["feat_exp_tree"])
            + coeffs["reduction_mant_tree"] * float(row["feat_mant_tree"])
            + coeffs["buffer_vlen"] * float(row["feat_vlen"])
            + coeffs["buffer_width"] * float(row["feat_width"])
            + coeffs["top_const"]
        )
        errors.append(abs((pred - actual) / max(abs(actual), 1e-9)))
    return sum(errors) / len(errors) * 100.0


def legacy_vector_mape(rows: list[dict[str, Any]], coeffs: dict[str, float]) -> float:
    """Evaluate the older undivided vector feature equation."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        pred = (
            coeffs["a_exp_lane"] * float(row["feat_exp_lane"])
            + coeffs["a_mant_lane"] * float(row["feat_mant_lane"])
            + coeffs["e_const"]
        )
        errors.append(abs((pred - actual) / max(abs(actual), 1e-9)))
    return sum(errors) / len(errors) * 100.0


DIRECT_VECTOR_FEATURES = [
    "feat_exp_lane",
    "feat_mant_lane",
    "feat_exp_tree",
    "feat_mant_tree",
    "feat_lane_quad",
    "feat_vlen",
    "feat_const",
]


def direct_vector_mape(rows: list[dict[str, Any]], coeffs: dict[str, float]) -> float:
    """Evaluate the selected direct-feature total-area equation."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        pred = (
            coeffs["direct_exp_lane"] * float(row["feat_exp_lane"])
            + coeffs["direct_mant_lane"] * float(row["feat_mant_lane"])
            + coeffs["direct_exp_tree"] * float(row["feat_exp_tree"])
            + coeffs["direct_mant_tree"] * float(row["feat_mant_tree"])
            + coeffs["direct_lane_quad"] * float(row["feat_lane_quad"])
            + coeffs["direct_vlen"] * float(row["feat_vlen"])
            + coeffs["direct_const"] * float(row["feat_const"])
        )
        errors.append(abs((pred - actual) / max(abs(actual), 1e-9)))
    return sum(errors) / len(errors) * 100.0


def write_vector_diagnostics(run_dir: Path, rows: list[dict[str, Any]], coeffs: dict[str, float]) -> None:
    """Write per-point predictions and residuals for model review."""
    if not rows:
        return
    fields = [
        "point_id",
        "actual_area_um2",
        "predicted_area_um2",
        "error_pct",
        "fitted_element_area",
        "hier_element_area",
        "fitted_reduction_area",
        "hier_reduction_area",
        "fitted_buffer_area",
        "hier_buffer_area",
        "fitted_top_glue_area",
        "hier_top_glue_area",
    ]
    path = run_dir / "vector_machine_diagnostics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            actual = float(row["area_um2"])
            if "direct_mant_lane" in coeffs:
                element = coeffs["direct_exp_lane"] * float(row["feat_exp_lane"]) + coeffs[
                    "direct_mant_lane"
                ] * float(row["feat_mant_lane"])
                reduction = coeffs["direct_exp_tree"] * float(row["feat_exp_tree"]) + coeffs[
                    "direct_mant_tree"
                ] * float(row["feat_mant_tree"])
                buffer = coeffs["direct_lane_quad"] * float(row["feat_lane_quad"]) + coeffs["direct_vlen"] * float(
                    row["feat_vlen"]
                )
                top = coeffs["direct_const"]
            elif "element_exp_lane" in coeffs:
                element = coeffs["element_exp_lane"] * float(row["feat_exp_lane"]) + coeffs["element_mant_lane"] * float(row["feat_mant_lane"])
                reduction = coeffs["reduction_exp_tree"] * float(row["feat_exp_tree"]) + coeffs["reduction_mant_tree"] * float(row["feat_mant_tree"])
                buffer = coeffs["buffer_vlen"] * float(row["feat_vlen"]) + coeffs["buffer_width"] * float(row["feat_width"])
                top = coeffs["top_const"]
            else:
                element = coeffs["a_exp_lane"] * float(row["feat_exp_lane"]) + coeffs["a_mant_lane"] * float(row["feat_mant_lane"])
                reduction = 0.0
                buffer = 0.0
                top = coeffs["e_const"]
            predicted = element + reduction + buffer + top
            writer.writerow(
                {
                    "point_id": row["point_id"],
                    "actual_area_um2": actual,
                    "predicted_area_um2": predicted,
                    "error_pct": (predicted - actual) / max(abs(actual), 1e-9) * 100.0,
                    "fitted_element_area": element,
                    "hier_element_area": row.get("hier_element_area", ""),
                    "fitted_reduction_area": reduction,
                    "hier_reduction_area": row.get("hier_reduction_area", ""),
                    "fitted_buffer_area": buffer,
                    "hier_buffer_area": row.get("hier_buffer_area", ""),
                    "fitted_top_glue_area": top,
                    "hier_top_glue_area": row.get("hier_top_glue_area", ""),
                }
            )


def fit_and_write_coefficients(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Fit candidate nonnegative equations and persist the selected model."""
    backfill_hierarchy_fields(csv_path)
    rows = load_complete_rows(csv_path)
    if rows:
        for row in rows:
            vlen = float(row["VLEN"])
            exp = float(row["V_FP_EXP_WIDTH"])
            mant = float(row["V_FP_MANT_WIDTH"])
            width = float(row["fp_width"])
            log_vlen = math.log2(max(vlen, 2.0))
            row["feat_exp_lane"] = vlen * exp
            row["feat_mant_lane"] = vlen * mant
            row["feat_exp_tree"] = vlen * exp * log_vlen
            row["feat_mant_tree"] = vlen * mant * log_vlen
            row["feat_lane_quad"] = vlen * width * width
            row["feat_vlen"] = vlen
            row["feat_width"] = width
            row["feat_const"] = 1.0
        element_rows = [row for row in rows if row.get("hier_element_area")]
        reduction_rows = [row for row in rows if row.get("hier_reduction_area")]
        buffer_rows = [row for row in rows if row.get("hier_buffer_area")]
        top_rows = [row for row in rows if row.get("hier_top_glue_area")]
        if element_rows and reduction_rows and buffer_rows:
            element_coeffs, element_mape = fit_nonnegative(element_rows, ["feat_exp_lane", "feat_mant_lane"], target="hier_element_area")
            reduction_coeffs, reduction_mape = fit_nonnegative(
                reduction_rows, ["feat_exp_tree", "feat_mant_tree"], target="hier_reduction_area"
            )
            buffer_coeffs, buffer_mape = fit_nonnegative(buffer_rows, ["feat_vlen", "feat_width"], target="hier_buffer_area")
            top_coeffs, top_mape = fit_nonnegative(top_rows or rows, ["feat_const"], target="hier_top_glue_area" if top_rows else "area_um2")
            hierarchy_coeff_dict = {
                "element_exp_lane": float(element_coeffs[0]),
                "element_mant_lane": float(element_coeffs[1]),
                "reduction_exp_tree": float(reduction_coeffs[0]),
                "reduction_mant_tree": float(reduction_coeffs[1]),
                "buffer_vlen": float(buffer_coeffs[0]),
                "buffer_width": float(buffer_coeffs[1]),
                "top_const": float(top_coeffs[0]),
            }
            hierarchy_mape = vector_mape(rows, hierarchy_coeff_dict)
            legacy_coeffs, legacy_mape = fit_nonnegative(rows, ["feat_exp_lane", "feat_mant_lane", "feat_const"])
            legacy_coeff_dict = {
                "a_exp_lane": float(legacy_coeffs[0]),
                "a_mant_lane": float(legacy_coeffs[1]),
                "e_const": float(legacy_coeffs[2]),
            }
            direct_coeffs, direct_mape = fit_nonnegative(rows, DIRECT_VECTOR_FEATURES)
            direct_coeff_dict = {
                "direct_exp_lane": float(direct_coeffs[0]),
                "direct_mant_lane": float(direct_coeffs[1]),
                "direct_exp_tree": float(direct_coeffs[2]),
                "direct_mant_tree": float(direct_coeffs[3]),
                "direct_lane_quad": float(direct_coeffs[4]),
                "direct_vlen": float(direct_coeffs[5]),
                "direct_const": float(direct_coeffs[6]),
            }
            if direct_mape <= legacy_mape and direct_mape <= hierarchy_mape:
                coeff_dict = direct_coeff_dict
                mape = direct_mape
                selected_target = "direct_feature_total_area_selected"
            elif legacy_mape < hierarchy_mape:
                coeff_dict = legacy_coeff_dict
                mape = legacy_mape
                selected_target = "total_area_selected"
            else:
                coeff_dict = hierarchy_coeff_dict
                mape = hierarchy_mape
                selected_target = "hierarchy_submodules_selected"
            hierarchy_metadata = {
                "fit_target": selected_target,
                "hierarchy_total_mape_pct": hierarchy_mape,
                "legacy_total_mape_pct": legacy_mape,
                "direct_feature_total_mape_pct": direct_mape,
                "element_mape_pct": element_mape,
                "reduction_mape_pct": reduction_mape,
                "buffer_mape_pct": buffer_mape,
                "top_mape_pct": top_mape,
            }
        else:
            coeffs, mape = fit_nonnegative(rows, DIRECT_VECTOR_FEATURES)
            coeff_dict = {
                "direct_exp_lane": float(coeffs[0]),
                "direct_mant_lane": float(coeffs[1]),
                "direct_exp_tree": float(coeffs[2]),
                "direct_mant_tree": float(coeffs[3]),
                "direct_lane_quad": float(coeffs[4]),
                "direct_vlen": float(coeffs[5]),
                "direct_const": float(coeffs[6]),
            }
            hierarchy_metadata = {"fit_target": "direct_feature_total_area_fallback"}
        write_vector_diagnostics(run_dir, rows, coeff_dict)
        status = "fitted_from_local_plena_rtl_synth"
    else:
        coeff_dict = {}
        mape = float("nan")
        hierarchy_metadata = {}
        status = "bootstrap_insufficient_data"
    out = {
        "metadata": {
            "status": status,
            "source_csv": str(csv_path),
            "rows": len(rows),
            "mape_pct": mape,
            "feature_order": [
                "element: VLEN * V_FP_EXP_WIDTH",
                "element: VLEN * V_FP_MANT_WIDTH",
                "reduction: VLEN * log2(VLEN) * V_FP_EXP_WIDTH",
                "reduction: VLEN * log2(VLEN) * V_FP_MANT_WIDTH",
                "buffer: VLEN",
                "buffer: fp_width",
                "top: 1",
            ],
            **hierarchy_metadata,
        },
        "coefficients": coeff_dict,
    }
    path = run_dir / "vector_model_coefficients.json"
    path.write_text(json.dumps(json_safe(out), indent=2, sort_keys=True))
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, CALIBRATION_DIR / path.name)


def write_complete_csv(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Export successful VectorMachine points as compact calibration data."""
    rows = load_complete_rows(csv_path)
    if not rows:
        return
    out = run_dir / "vector_machine.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, CALIBRATION_DIR / out.name)


def run_dry_run(points: list[Point], run_dir: Path, rtl_root: Path) -> None:
    """Write plan and commands without modifying RTL or invoking DC."""
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    commands = run_dir / "plans" / "commands.txt"
    commands.parent.mkdir(parents=True, exist_ok=True)
    commands.write_text(
        "\n".join(
            f"# {point.point_id}: patch VLEN={point.params['VLEN']} "
            f"V_FP=E{point.params['V_FP_EXP_WIDTH']}M{point.params['V_FP_MANT_WIDTH']}\n"
            f"cd <worker-copy> && just synth vector_machine 1000 area"
            for point in points
        )
        + "\n"
    )
    print(f"Dry run wrote {len(points)} planned VectorMachine points to {run_dir}")
    print(f"RTL root for real runs: {rtl_root}")


def parse_args() -> argparse.Namespace:
    """Parse standalone VectorMachine calibration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["smoke", "minimal-v1", "refine-v1", "e6m5-v1", "tiny-v1", "reduced-v1", "validation"],
        default="tiny-v1",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=RTL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--point-id-regex", help="only run planned points whose point_id matches this regex")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", default="auto", help="worker count or 'auto' to use available DC licenses")
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--cleanup-worker-builds", action="store_true", default=True)
    parser.add_argument("--no-cleanup-worker-builds", dest="cleanup_worker_builds", action="store_false")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--no-copy-to-calibration", action="store_true")
    parser.add_argument(
        "--license-retry-wait-sec",
        type=float,
        default=float(os.environ.get("PLENA_DC_LICENSE_RETRY_WAIT_SEC", "60")),
    )
    parser.add_argument(
        "--license-max-retries",
        type=int,
        default=int(os.environ.get("PLENA_DC_LICENSE_MAX_RETRIES", "0")),
        help="max license-busy retries per point; 0 means retry indefinitely",
    )
    return parser.parse_args()


def main() -> int:
    """Run or dry-run VectorMachine calibration."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"vector_{safe_run_name}_{os.getpid()}"

    points = build_plan(args.preset)
    if args.point_id_regex:
        pattern = re.compile(args.point_id_regex)
        points = [point for point in points if pattern.search(point.point_id)]
    if args.limit is not None:
        points = points[: args.limit]
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    if args.dry_run:
        run_dry_run(points, run_dir, args.rtl_root)
        return 0

    csv_path = run_dir / "calibration_points.csv"
    completed = read_completed_keys(csv_path) if args.resume else set()
    pending = [point for point in points if point.point_key not in completed]
    if not pending:
        print("No pending points.")
        write_complete_csv(csv_path, run_dir, not args.no_copy_to_calibration)
        fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
        return 0

    if args.worker_root.exists() and not args.keep_workers:
        shutil.rmtree(args.worker_root)
    args.worker_root.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_dc_worker_count(args.workers, repo_root=ROOT)
    worker_paths = [create_worker_copy(i, args.worker_root, args.rtl_root) for i in range(worker_count)]
    worker_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in enumerate(worker_paths):
        worker_queue.put(item)
    csv_lock = threading.Lock()

    def wrapped(point: Point) -> dict[str, Any]:
        worker_id, worker_path = worker_queue.get()
        try:
            return run_point(
                point,
                worker_id,
                worker_path,
                args.rtl_root,
                run_dir,
                args.cleanup_worker_builds,
                args.license_retry_wait_sec,
                args.license_max_retries,
            )
        finally:
            worker_queue.put((worker_id, worker_path))

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(wrapped, point): point for point in pending}
            for future in as_completed(futures):
                row = future.result()
                append_row(csv_path, row, csv_lock)
                print(f"[{row['status']}] {row['point_id']} area={row['area_um2']} reason={row['failure_reason']}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; compact rows already written remain resumable.", file=sys.stderr)
    finally:
        write_complete_csv(csv_path, run_dir, not args.no_copy_to_calibration)
        fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
        if not args.keep_workers:
            cleanup_workers(args.worker_root)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
