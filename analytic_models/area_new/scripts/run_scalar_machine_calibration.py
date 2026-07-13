#!/usr/bin/env python3
"""Calibrate ScalarMachine integer, FP, and shape-dependent logic area.

Hierarchy-supervised fitting separates integer ALU, FP ALU, FP SFU, and top
glue while excluding scalar SRAM black boxes. MLEN and VLEN are explicit point
dimensions because scalar buffering/control can depend on both.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
from run_matrix_machine_calibration import (
    cleanup_worker_build,
    cleanup_workers,
    copy_reports,
    create_worker_copy,
    fit_nonnegative,
    json_safe,
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

# scalar_machine instantiates fp_sfu, which instantiates fp_fix_reciprocal and
# maps to Synopsys DW_fp_recip under DC_LIB_EN.  The local DC/DesignWare setup
# rejects very narrow reciprocal configs such as E3M2/E2M3, so they are not used
# as synthesis anchors.  The fitted model may still extrapolate to those formats.
SCALAR_SYNTH_FP_SETTINGS = {
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
    "MLEN",
    "VLEN",
    "INT_DATA_WIDTH",
    "FP_SETTING",
    "S_FP_EXP_WIDTH",
    "S_FP_MANT_WIDTH",
    "fp_width",
    "feat_int_mul",
    "feat_int_lin",
    "feat_fp_quad",
    "feat_fp_lin",
    "feat_fp_exp",
    "feat_mlen_buffer",
    "feat_vlen_output",
    "feat_vector_buffer",
    "feat_mlen",
    "feat_vlen",
    "feat_min_vector_buffer",
    "feat_max_vector_buffer",
    "feat_min_vector",
    "feat_max_vector",
    "feat_const",
    "hier_total_area",
    "hier_int_alu_area",
    "hier_fp_alu_area",
    "hier_fp_sfu_area",
    "hier_scalar_sram_area",
    "hier_top_glue_area",
    "preset",
]


@dataclass(frozen=True)
class Point:
    """One immutable scalar width/FP/MLEN/VLEN synthesis point."""
    point_id: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {"module": self.module, "top_module": self.top_module, "params": self.params}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"scalar_machine_{digest}")


def fp_width(exp: int, mant: int) -> int:
    """Return sign + exponent + mantissa bits."""
    return 1 + exp + mant


def features(int_width: int, exp: int, mant: int, mlen: int = 16, vlen: int = 16) -> dict[str, float]:
    """Construct arithmetic and vector-shape features used by scalar fitting."""
    width = fp_width(exp, mant)
    min_vector = min(mlen, vlen)
    max_vector = max(mlen, vlen)
    return {
        "feat_int_mul": float(int_width * int_width),
        "feat_int_lin": float(int_width),
        "feat_fp_quad": float(width * width),
        "feat_fp_lin": float(width),
        "feat_fp_exp": float(exp),
        "feat_mlen_buffer": float(mlen * width),
        "feat_vlen_output": float(vlen * width),
        "feat_vector_buffer": float(mlen * width),
        "feat_mlen": float(mlen),
        "feat_vlen": float(vlen),
        "feat_min_vector_buffer": float(min_vector * width),
        "feat_max_vector_buffer": float(max_vector * width),
        "feat_min_vector": float(min_vector),
        "feat_max_vector": float(max_vector),
        "feat_const": 1.0,
    }


def build_plan(preset: str) -> list[Point]:
    """Build a named set spanning integer, FP, and MLEN/VLEN effects."""
    if preset == "smoke":
        entries = [(16, 32, "FP_E5M6")]
    elif preset == "minimal-v1":
        entries = [
            (16, 16, "FP_E5M6"),
            (16, 32, "FP_E5M6"),
            (16, 64, "FP_E5M6"),
            (16, 32, "FP_E4M7"),
            (16, 32, "FP_E6M5"),
            (16, 32, "FP_E8M5"),
            (16, 64, "FP_E8M5"),
            (16, 16, "FP_E8M5"),
        ]
    elif preset == "vlen-v1":
        entries = [
            (16, 32, "FP_E5M6"),
            (32, 32, "FP_E5M6"),
            (64, 32, "FP_E5M6"),
            (128, 32, "FP_E5M6"),
            (32, 16, "FP_E5M6"),
            (32, 64, "FP_E5M6"),
            (64, 64, "FP_E4M7"),
            (64, 64, "FP_E6M5"),
            (64, 64, "FP_E8M5"),
        ]
    elif preset == "rich-v2":
        entries = [
            # MLEN/VLEN coupled scaling.
            (16, 16, 32, "FP_E5M6"),
            (32, 32, 32, "FP_E5M6"),
            (64, 64, 32, "FP_E5M6"),
            (128, 128, 32, "FP_E5M6"),
            # Decouple MLEN buffer width from VLEN counter/output width.
            (16, 64, 32, "FP_E5M6"),
            (64, 16, 32, "FP_E5M6"),
            (32, 128, 32, "FP_E5M6"),
            (128, 32, 32, "FP_E5M6"),
            # INT datapath scaling at fixed vector shape.
            (64, 64, 16, "FP_E5M6"),
            (64, 64, 64, "FP_E5M6"),
            # FP format scaling at fixed vector shape and int width.
            (64, 64, 32, "FP_E4M7"),
            (64, 64, 32, "FP_E6M5"),
            (64, 64, 32, "FP_E8M5"),
            # Combined stress points.
            (128, 128, 64, "FP_E8M5"),
            (32, 128, 16, "FP_E4M7"),
            (128, 32, 64, "FP_E6M5"),
        ]
    else:
        raise ValueError(f"unknown preset: {preset}")

    points: list[Point] = []
    seen: set[tuple[int, int, int, str]] = set()
    for entry in entries:
        if len(entry) == 3:
            mlen = vlen = int(entry[0])
            int_width = int(entry[1])
            setting = str(entry[2])
        else:
            mlen = int(entry[0])
            vlen = int(entry[1])
            int_width = int(entry[2])
            setting = str(entry[3])
        if setting not in SCALAR_SYNTH_FP_SETTINGS:
            raise ValueError(f"{setting} is not a scalar_machine synthesis anchor")
        key = (mlen, vlen, int_width, setting)
        if key in seen:
            continue
        seen.add(key)
        exp, mant = FP_SETTINGS[setting]
        params: dict[str, Any] = {
            "MLEN": mlen,
            "VLEN": vlen,
            "INT_DATA_WIDTH": int_width,
            "FP_SETTING": setting,
            "S_FP_EXP_WIDTH": exp,
            "S_FP_MANT_WIDTH": mant,
            "fp_width": fp_width(exp, mant),
            "preset": preset,
        }
        params.update(features(int_width, exp, mant, mlen, vlen))
        points.append(
            Point(
                point_id=f"area_new_scalar_m{mlen}_v{vlen}_i{int_width}_{setting.lower()}",
                module="scalar_machine",
                top_module="scalar_machine",
                params=params,
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
    """Serialize one scalar synthesis outcome and its fitted features."""
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
    """Persist all planned scalar points before synthesis."""
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


def patch_scalar_config(point: Point, rtl_root: Path) -> None:
    """Patch integer width, scalar FP widths, MLEN, and VLEN into worker RTL."""
    p = point.params
    configuration = rtl_root / "src/definitions/configuration.svh"
    precision = rtl_root / "src/definitions/precision.svh"
    replace_localparam(configuration, "MLEN", int(p["MLEN"]))
    replace_localparam(configuration, "VLEN", int(p["VLEN"]))
    replace_localparam(precision, "INT_DATA_WIDTH", int(p["INT_DATA_WIDTH"]))
    replace_localparam(precision, "S_FP_EXP_WIDTH", int(p["S_FP_EXP_WIDTH"]))
    replace_localparam(precision, "S_FP_MANT_WIDTH", int(p["S_FP_MANT_WIDTH"]))


def cleanup_scalar_build(worker_rtl: Path) -> None:
    """Remove ScalarMachine DC output after copying reports."""
    cleanup_worker_build(worker_rtl, Point("scalar_machine", "scalar_machine", "scalar_machine", {}))


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
    """Patch, synthesize, archive, and clean one ScalarMachine point."""
    start = time.time()
    try:
        patch_scalar_config(point, worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth scalar_machine 1000 area"
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
                cleanup_scalar_build(worker_rtl)
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
            cleanup_scalar_build(worker_rtl)


def load_complete_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load successful scalar rows and normalize numeric features."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete"]


def _parse_dc_hierarchy_rows(report: Path) -> list[tuple[str, float]]:
    if not report.exists():
        return []
    parsed: list[tuple[str, float]] = []
    pending_name: str | None = None
    numeric_line = re.compile(r"^\s+([0-9.]+)\s+([0-9.]+)\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+")
    inline_line = re.compile(
        r"^(.{1,34}?)(\s+[0-9.]+\s+[0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+)"
    )
    for line in report.read_text(errors="ignore").splitlines():
        if (
            not line.strip()
            or line.startswith("-")
            or line.startswith("Hierarchical")
            or line.startswith("Global")
            or line.startswith("Total")
            or line.startswith("Design")
        ):
            continue
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
        if not re.search(r"[0-9]+\.[0-9]+", line):
            pending_name = line.strip()
    return parsed


def parse_scalar_hierarchy_area(report: Path) -> dict[str, float]:
    """Extract scalar ALU/SFU/SRAM and residual top-glue hierarchy areas."""
    values = {name: area for name, area in _parse_dc_hierarchy_rows(report)}
    total = values.get("scalar_machine", 0.0)
    int_alu = values.get("int_alu_init", 0.0)
    fp_alu = values.get("fp_alu_init", 0.0)
    fp_sfu = values.get("fp_sfu_init", 0.0)
    scalar_sram = values.get("int_scalar_sram", 0.0) + values.get("fp_scalar_sram", 0.0)
    top_glue = max(0.0, total - int_alu - fp_alu - fp_sfu - scalar_sram) if total else 0.0
    return {
        "hier_total_area": total,
        "hier_int_alu_area": int_alu,
        "hier_fp_alu_area": fp_alu,
        "hier_fp_sfu_area": fp_sfu,
        "hier_scalar_sram_area": scalar_sram,
        "hier_top_glue_area": top_glue,
    }


def backfill_hierarchy_fields(csv_path: Path) -> None:
    """Backfill derived features and hierarchy values into historical rows."""
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or CSV_FIELDS)
    changed = False
    for field in CSV_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
            changed = True
    for row in rows:
        if row.get("status") != "complete":
            continue
        if not row.get("feat_min_vector_buffer"):
            try:
                feats = features(
                    int(row["INT_DATA_WIDTH"]),
                    int(row["S_FP_EXP_WIDTH"]),
                    int(row["S_FP_MANT_WIDTH"]),
                    int(row.get("MLEN") or 16),
                    int(row.get("VLEN") or row.get("MLEN") or 16),
                )
            except (KeyError, TypeError, ValueError):
                feats = {}
            if feats:
                for key, value in feats.items():
                    row.setdefault(key, value)
                    if row.get(key, "") == "":
                        row[key] = value
                changed = True
        if not row.get("hier_total_area"):
            report_dir = row.get("report_dir")
            if not report_dir:
                continue
            report = Path(report_dir) / "area.rpt"
            if not report.exists():
                report = ROOT / report
            if not report.exists():
                continue
            hier = parse_scalar_hierarchy_area(report)
            if not hier.get("hier_total_area"):
                continue
            row.update(hier)
            changed = True
    if changed:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _mape_for_prediction(rows: list[dict[str, Any]], prediction_fn, target: str = "area_um2") -> float:
    errors = []
    for row in rows:
        actual = float(row.get(target) or 0.0)
        if actual <= 0.0:
            continue
        pred = float(prediction_fn(row))
        errors.append(abs(pred - actual) / actual)
    return float(sum(errors) / len(errors) * 100.0) if errors else float("nan")


def fit_and_write_coefficients(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Fit hierarchy targets independently and compose total scalar area."""
    backfill_hierarchy_fields(csv_path)
    rows = load_complete_rows(csv_path)
    hierarchy_rows = [row for row in rows if row.get("hier_total_area")]
    if hierarchy_rows:
        int_vals, int_mape = fit_nonnegative(hierarchy_rows, ["feat_int_mul", "feat_int_lin", "feat_const"], target="hier_int_alu_area")
        fp_alu_vals, fp_alu_mape = fit_nonnegative(
            hierarchy_rows,
            ["feat_fp_quad", "feat_fp_lin", "feat_fp_exp", "feat_const"],
            target="hier_fp_alu_area",
        )
        fp_sfu_vals, fp_sfu_mape = fit_nonnegative(hierarchy_rows, ["feat_const"], target="hier_fp_sfu_area")
        top_vals, top_mape = fit_nonnegative(
            hierarchy_rows,
            ["feat_min_vector_buffer", "feat_max_vector_buffer", "feat_const"],
            target="hier_top_glue_area",
        )
        coeff_dict = {
            "a_int_mul": int_vals[0],
            "a_int_lin": int_vals[1],
            "a_fp_quad": fp_alu_vals[0],
            "a_fp_lin": fp_alu_vals[1],
            "a_exp": fp_alu_vals[2],
            "a_fp_alu_const": fp_alu_vals[3],
            "a_fp_sfu_const": fp_sfu_vals[0],
            "a_mlen_buffer": 0.0,
            "a_vlen_output": 0.0,
            "a_mlen": 0.0,
            "a_vlen": 0.0,
            "a_min_vector_buffer": top_vals[0],
            "a_max_vector_buffer": top_vals[1],
            "a_min_vector": 0.0,
            "a_max_vector": 0.0,
            "a_const": top_vals[2],
        }

        def predict(row: dict[str, Any]) -> float:
            return (
                coeff_dict["a_int_mul"] * float(row["feat_int_mul"])
                + coeff_dict["a_int_lin"] * float(row["feat_int_lin"])
                + coeff_dict["a_fp_quad"] * float(row["feat_fp_quad"])
                + coeff_dict["a_fp_lin"] * float(row["feat_fp_lin"])
                + coeff_dict["a_exp"] * float(row["feat_fp_exp"])
                + coeff_dict["a_fp_alu_const"]
                + coeff_dict["a_fp_sfu_const"]
                + coeff_dict["a_min_vector_buffer"] * float(row["feat_min_vector_buffer"])
                + coeff_dict["a_max_vector_buffer"] * float(row["feat_max_vector_buffer"])
                + coeff_dict["a_const"]
            )

        mape = _mape_for_prediction(hierarchy_rows, predict)
        status = "fitted_from_local_plena_rtl_synth"
        model_version = "scalar_machine_hierarchy_supervised_v2"
        diagnostics = {
            "hier_int_alu_mape_pct": int_mape,
            "hier_fp_alu_mape_pct": fp_alu_mape,
            "hier_fp_sfu_mape_pct": fp_sfu_mape,
            "hier_top_glue_mape_pct": top_mape,
            "hier_total_mape_pct": mape,
            "hierarchy_rows": len(hierarchy_rows),
        }
        feature_order = [
            "IntALU: INT_DATA_WIDTH^2, INT_DATA_WIDTH",
            "FP_ALU: fp_width^2, fp_width, S_FP_EXP_WIDTH, 1",
            "FP_SFU: constant",
            "TopGlue: min(MLEN,VLEN)*fp_width, max(MLEN,VLEN)*fp_width, 1",
        ]
    elif rows:
        feature_names = [
            "feat_int_mul",
            "feat_int_lin",
            "feat_fp_quad",
            "feat_fp_lin",
            "feat_fp_exp",
            "feat_mlen_buffer",
            "feat_vlen_output",
            "feat_mlen",
            "feat_vlen",
            "feat_const",
        ]
        vals, mape = fit_nonnegative(rows, feature_names)
        coeff_dict = {
            "a_int_mul": vals[0],
            "a_int_lin": vals[1],
            "a_fp_quad": vals[2],
            "a_fp_lin": vals[3],
            "a_exp": vals[4],
            "a_fp_alu_const": 0.0,
            "a_fp_sfu_const": 0.0,
            "a_mlen_buffer": vals[5],
            "a_vlen_output": vals[6],
            "a_mlen": vals[7],
            "a_vlen": vals[8],
            "a_min_vector_buffer": 0.0,
            "a_max_vector_buffer": 0.0,
            "a_min_vector": 0.0,
            "a_max_vector": 0.0,
            "a_const": vals[9],
        }
        status = "fitted_from_local_plena_rtl_synth"
        model_version = "scalar_machine_total_nnls_v1"
        diagnostics = {}
        feature_order = [
            "INT_DATA_WIDTH^2",
            "INT_DATA_WIDTH",
            "fp_width^2",
            "fp_width",
            "S_FP_EXP_WIDTH",
            "MLEN * fp_width",
            "VLEN * fp_width",
            "MLEN",
            "VLEN",
            "1",
        ]
    else:
        coeff_dict = {}
        mape = float("nan")
        status = "bootstrap_insufficient_data"
        model_version = "bootstrap_insufficient_data"
        diagnostics = {}
        feature_order = []
    out = {
        "metadata": {
            "status": status,
            "model_version": model_version,
            "source_csv": str(csv_path),
            "rows": len(rows),
            "mape_pct": mape,
            "feature_order": feature_order,
            "diagnostics": diagnostics,
            "training_point_ids": [str(row.get("point_id", "")) for row in hierarchy_rows or rows],
        },
        "coefficients": coeff_dict,
    }
    path = run_dir / "scalar_model_coefficients.json"
    path.write_text(json.dumps(json_safe(out), indent=2, sort_keys=True))
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, CALIBRATION_DIR / path.name)


def write_complete_csv(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Export successful ScalarMachine points as compact calibration data."""
    rows = load_complete_rows(csv_path)
    if not rows:
        return
    out = run_dir / "scalar_machine.csv"
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
            f"# {point.point_id}: patch INT_DATA_WIDTH={point.params['INT_DATA_WIDTH']} "
            f"S_FP=E{point.params['S_FP_EXP_WIDTH']}M{point.params['S_FP_MANT_WIDTH']}\n"
            f"cd <worker-copy> && just synth scalar_machine 1000 area"
            for point in points
        )
        + "\n"
    )
    print(f"Dry run wrote {len(points)} planned ScalarMachine points to {run_dir}")
    print(f"RTL root for real runs: {rtl_root}")


def parse_args() -> argparse.Namespace:
    """Parse standalone ScalarMachine calibration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "minimal-v1", "vlen-v1", "rich-v2"], default="minimal-v1")
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
    """Run or dry-run ScalarMachine calibration."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"scalar_{safe_run_name}_{os.getpid()}"

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
