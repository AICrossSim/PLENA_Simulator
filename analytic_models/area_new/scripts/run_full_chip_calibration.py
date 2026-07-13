#!/usr/bin/env python3
"""Run small full-chip PLENA area anchor synths.

These anchors are intended to calibrate the residual between the sum of
module-level area proxies and a full `plena` top-level synthesis.  They are not
used to replace the module models directly.

Memory arrays remain black boxes in this flow. Consequently, ``area_um2`` is
logic plus SRAM wrapper/conversion logic, not SRAM bitcell macro area. The
runtime area proxy adds ASAP7 SRAM macro area separately.
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
        json_safe,
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

MXFP_FORMATS = {
    "MXFP_E1M2": (1, 2),
    "MXFP_E2M1": (2, 1),
    "MXFP_E4M3": (4, 3),
    "MXFP_E5M2": (5, 2),
}

FP_SETTINGS = {
    "FP_E3M2": (3, 2),
    "FP_E2M3": (2, 3),
    "FP_E4M7": (4, 7),
    "FP_E5M6": (5, 6),
    "FP_E6M5": (6, 5),
    "FP_E8M5": (8, 5),
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
    "BLEN",
    "HLEN",
    "MATRIX_SRAM_DEPTH",
    "VECTOR_SRAM_DEPTH",
    "INT_SRAM_DEPTH",
    "FP_SRAM_DEPTH",
    "ACT_WIDTH",
    "KV_WIDTH",
    "WEIGHT_WIDTH",
    "FP_SETTING",
    "INT_DATA_WIDTH",
    "MX_SCALE_WIDTH",
    "HBM_M_Prefetch_Amount",
    "HBM_V_Prefetch_Amount",
    "HBM_V_Writeback_Amount",
    "preset",
    "hier_plena_area",
    "hier_matrix_machine_area",
    "hier_vector_machine_area",
    "hier_scalar_machine_area",
    "hier_scalar_fp_sram_wrapper_area",
    "hier_scalar_int_sram_wrapper_area",
    "hier_hbm_system_area",
    "hier_matrix_sram_wrapper_area",
    "hier_vector_sram_wrapper_area",
]


@dataclass(frozen=True)
class Point:
    """One immutable full-chip hardware/precision anchor configuration."""
    point_id: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {"module": self.module, "top_module": self.top_module, "params": self.params}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"full_chip_{digest}")


def _mk_params(
    *,
    mlen: int = 16,
    vlen: int = 16,
    blen: int = 4,
    hlen: int = 8,
    act: str = "MXINT4",
    kv: str = "MXINT4",
    weight: str = "MXINT4",
    fp: str = "FP_E5M6",
    int_width: int = 32,
    preset: str,
) -> dict[str, Any]:
    return {
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "HLEN": hlen,
        "MATRIX_SRAM_DEPTH": 2 * mlen,
        "VECTOR_SRAM_DEPTH": max(32, 2 * hlen + max(1, mlen // max(vlen, 1))),
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": mlen + 4,
        "ACT_WIDTH": act,
        "KV_WIDTH": kv,
        "WEIGHT_WIDTH": weight,
        "FP_SETTING": fp,
        "INT_DATA_WIDTH": int_width,
        "MX_SCALE_WIDTH": 8,
        "HBM_M_Prefetch_Amount": mlen,
        "HBM_V_Prefetch_Amount": max(4, min(64, vlen)),
        "HBM_V_Writeback_Amount": max(4, min(64, vlen)),
        "preset": preset,
    }


def build_plan(preset: str) -> list[Point]:
    """Build a small anchor set chosen to expose shape and precision residuals."""
    if preset == "smoke":
        entries = [("baseline_tiny", _mk_params(preset=preset))]
    elif preset == "smallest-v1":
        entries = [
            ("baseline", _mk_params(preset=preset)),
            ("low_mxint", _mk_params(act="MXINT2", kv="MXINT2", weight="MXINT4", preset=preset)),
            ("high_mxint", _mk_params(act="MXINT8", kv="MXINT8", weight="MXINT8", preset=preset)),
        ]
    else:
        raise ValueError(f"unknown preset: {preset}")
    return [
        Point(
            point_id=f"area_new_full_chip_{label}",
            module="plena",
            top_module="plena",
            params=params,
        )
        for label, params in entries
    ]


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
    """Convert one trial outcome into the stable full-chip CSV schema."""
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
    """Append one completed attempt under the shared writer lock."""
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_plan_csv(points: list[Point], path: Path) -> None:
    """Persist planned anchors before launching DC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point_to_row(point, status="planned"))


def read_completed_keys(path: Path) -> set[str]:
    """Read point keys already completed by an earlier resumable run."""
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row["point_key"] for row in csv.DictReader(f) if row.get("status") == "complete"}


def _patch_precision_token(precision: Path, prefix: str, token: str) -> None:
    token = token.upper()
    if token.startswith("MXINT"):
        width = int(token.replace("MXINT", "").replace("_", ""))
        replace_localparam(precision, f"{prefix}_MX_INT_ENABLE", 1)
        replace_localparam(precision, f"{prefix}_MX_INT_WIDTH", width)
        return
    if token.startswith("MXFP"):
        if token not in MXFP_FORMATS:
            raise ValueError(f"unsupported MXFP format for full-chip calibration: {token}")
        exp, mant = MXFP_FORMATS[token]
        replace_localparam(precision, f"{prefix}_MX_INT_ENABLE", 0)
        if prefix == "ACT":
            replace_localparam(precision, "ACT_MXFP_EXP_WIDTH", exp)
            replace_localparam(precision, "ACT_MXFP_MANT_WIDTH", mant)
        else:
            replace_localparam(precision, f"{prefix}_MX_EXP_WIDTH", exp)
            replace_localparam(precision, f"{prefix}_MX_MANT_WIDTH", mant)
        return
    raise ValueError(f"unsupported precision token: {token}")


def patch_full_chip_config(point: Point, rtl_root: Path) -> None:
    """Patch hardware dimensions and software precisions into a worker RTL copy."""
    p = point.params
    configuration = rtl_root / "src/definitions/configuration.svh"
    precision = rtl_root / "src/definitions/precision.svh"
    for key in [
        "MLEN",
        "VLEN",
        "BLEN",
        "HLEN",
        "MATRIX_SRAM_DEPTH",
        "VECTOR_SRAM_DEPTH",
        "INT_SRAM_DEPTH",
        "FP_SRAM_DEPTH",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
    ]:
        replace_localparam(configuration, key, int(p[key]))
    replace_localparam(precision, "BLOCK_DIM", int(p["BLEN"]))
    replace_localparam(precision, "MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "ACT_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "KV_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "WT_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "INT_DATA_WIDTH", int(p["INT_DATA_WIDTH"]))
    exp, mant = FP_SETTINGS[str(p["FP_SETTING"]).upper()]
    for prefix in ["V", "M", "S", "ROUND"]:
        replace_localparam(precision, f"{prefix}_FP_EXP_WIDTH", exp)
        replace_localparam(precision, f"{prefix}_FP_MANT_WIDTH", mant)
    _patch_precision_token(precision, "ACT", str(p["ACT_WIDTH"]))
    _patch_precision_token(precision, "KV", str(p["KV_WIDTH"]))
    _patch_precision_token(precision, "WT", str(p["WEIGHT_WIDTH"]))


def cleanup_full_chip_build(worker_rtl: Path) -> None:
    """Remove full-chip DC build output after reports have been copied."""
    cleanup_worker_build(worker_rtl, Point("plena", "plena", "plena", {}))


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
    """Patch, synthesize, parse, archive, and clean one full-chip anchor."""
    start = time.time()
    try:
        patch_full_chip_config(point, worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth plena 1000 area"
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
                cleanup_full_chip_build(worker_rtl)
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
        power = parse_power(power_report) if power_report else parse_power(Path("__missing__"))
        return point_to_row(
            point,
            status="complete",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            area_um2=parse_area(area_report),
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
            cleanup_full_chip_build(worker_rtl)


def write_complete_csv(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Export successful anchors and a human-readable run-local summary."""
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "complete"]
    if not rows:
        return
    out = run_dir / "full_chip_anchors.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "metadata": {
            "status": "full_chip_anchor_points",
            "source_csv": str(csv_path),
            "rows": len(rows),
            "note": "Use these points to calibrate top-level residual against module-sum area proxies.",
        },
        "points": rows,
    }
    (run_dir / "full_chip_anchor_summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, CALIBRATION_DIR / out.name)


def run_dry_run(points: list[Point], run_dir: Path, rtl_root: Path) -> None:
    """Write the plan and commands without modifying RTL or invoking DC."""
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    commands = run_dir / "plans" / "commands.txt"
    commands.parent.mkdir(parents=True, exist_ok=True)
    commands.write_text(
        "\n".join(
            f"# {point.point_id}: MLEN={point.params['MLEN']} VLEN={point.params['VLEN']} "
            f"BLEN={point.params['BLEN']} HLEN={point.params['HLEN']} "
            f"ACT={point.params['ACT_WIDTH']} KV={point.params['KV_WIDTH']} "
            f"WT={point.params['WEIGHT_WIDTH']} FP={point.params['FP_SETTING']}\n"
            f"cd <worker-copy> && just synth plena 1000 area"
            for point in points
        )
        + "\n"
    )
    print(f"Dry run wrote {len(points)} planned full-chip points to {run_dir}")
    print(f"RTL root for real runs: {rtl_root}")


def parse_args() -> argparse.Namespace:
    """Parse preset, worker, resume, retry, and artifact options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "smallest-v1"], default="smallest-v1")
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
    """Run or dry-run the standalone full-chip calibration workflow."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"full_chip_{safe_run_name}_{os.getpid()}"

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
        if not args.keep_workers:
            cleanup_workers(args.worker_root)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
