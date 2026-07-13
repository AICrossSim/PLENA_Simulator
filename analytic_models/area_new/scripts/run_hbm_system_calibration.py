#!/usr/bin/env python3
"""Calibrate on-chip ``hbm_sys`` interface logic against Synopsys DC.

This runner varies precision, MLEN/VLEN/BLEN, and prefetch/writeback counters.
It does not model HBM PHYs, stacks, channel topology, or external capacity.
Each worker patches an isolated RTL copy and deletes its build after retaining
compact reports.
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
        parse_area,
        parse_area_from_text,
        parse_power,
        replace_localparam,
        summarize_synth_failure,
    )

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytic_models.area_new.hbm_model import hbm_features  # noqa: E402

RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_area_workers")
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"

MXFP_FORMATS = {
    "MXFP_E1M2": (1, 2),
    "MXFP_E2M1": (2, 1),
    "MXFP_E4M3": (4, 3),
    "MXFP_E5M2": (5, 2),
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
    "BLOCK_DIM",
    "ACT_WIDTH",
    "KV_WIDTH",
    "WEIGHT_WIDTH",
    "MX_SCALE_WIDTH",
    "HBM_M_Prefetch_Amount",
    "HBM_V_Prefetch_Amount",
    "HBM_V_Writeback_Amount",
    "feat_hbm_ele_width",
    "feat_hbm_scale_width",
    "feat_m_path",
    "feat_v_path",
    "feat_scale_path",
    "feat_addr",
    "feat_load",
    "feat_write",
    "feat_const",
    "preset",
]

FEATURE_FIELDS = [
    "feat_hbm_ele_width",
    "feat_hbm_scale_width",
    "feat_m_path",
    "feat_v_path",
    "feat_scale_path",
    "feat_addr",
    "feat_load",
    "feat_write",
    "feat_const",
]

COEFFICIENT_NAMES = [
    "a_ele",
    "a_scale",
    "a_m_path",
    "a_v_path",
    "a_scale_path",
    "a_addr",
    "a_load",
    "a_write",
    "a_const",
]


@dataclass(frozen=True)
class Point:
    """One immutable HBM interface calibration configuration."""
    point_id: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {"module": self.module, "top_module": self.top_module, "params": self.params}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"hbm_system_{digest}")


def _mk_params(
    *,
    mlen: int = 128,
    vlen: int = 128,
    blen: int = 16,
    act: str = "MXINT4",
    kv: str = "MXINT4",
    weight: str = "MXINT4",
    m_load: int | None = None,
    v_load: int = 64,
    v_write: int = 64,
    preset: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "BLOCK_DIM": blen,
        "ACT_WIDTH": act,
        "KV_WIDTH": kv,
        "WEIGHT_WIDTH": weight,
        "MX_SCALE_WIDTH": 8,
        "HBM_M_Prefetch_Amount": mlen if m_load is None else m_load,
        "HBM_V_Prefetch_Amount": v_load,
        "HBM_V_Writeback_Amount": v_write,
        "preset": preset,
    }
    features = hbm_features(params)
    params.update(
        {
            "feat_hbm_ele_width": features["hbm_ele_width"],
            "feat_hbm_scale_width": features["hbm_scale_width"],
            "feat_m_path": features["m_path"],
            "feat_v_path": features["v_path"],
            "feat_scale_path": features["scale_path"],
            "feat_addr": features["addr"],
            "feat_load": features["load"],
            "feat_write": features["write"],
            "feat_const": features["const"],
        }
    )
    return params


def build_plan(preset: str) -> list[Point]:
    """Build precision, shape, and prefetch anchors for a named preset."""
    if preset == "smoke":
        entries = [("baseline", _mk_params(preset=preset))]
    elif preset == "minimal-v1":
        entries = [
            ("baseline", _mk_params(preset=preset)),
            ("low_mxint", _mk_params(act="MXINT2", kv="MXINT2", weight="MXINT4", preset=preset)),
            ("high_mxint", _mk_params(act="MXINT8", kv="MXINT8", weight="MXINT8", preset=preset)),
            ("mxfp_e4m3", _mk_params(act="MXFP_E4M3", kv="MXFP_E4M3", weight="MXFP_E4M3", preset=preset)),
            ("shape64", _mk_params(mlen=64, vlen=64, blen=16, m_load=64, preset=preset)),
            ("shape256", _mk_params(mlen=256, vlen=256, blen=32, m_load=256, v_load=128, v_write=128, preset=preset)),
            ("high_m_load", _mk_params(m_load=256, preset=preset)),
            ("high_v_load_write", _mk_params(v_load=128, v_write=128, preset=preset)),
        ]
    else:
        raise ValueError(f"unknown preset: {preset}")

    points: list[Point] = []
    seen: set[str] = set()
    for label, params in entries:
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        points.append(
            Point(
                point_id=f"area_new_hbm_{label}",
                module="hbm_sys",
                top_module="hbm_sys",
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
    """Serialize one HBM synthesis outcome and its regression features."""
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
    """Append one outcome under the standalone runner's CSV lock."""
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def write_plan_csv(points: list[Point], path: Path) -> None:
    """Write the complete point plan before synthesis starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point_to_row(point, status="planned"))


def read_completed_keys(path: Path) -> set[str]:
    """Return successful point keys used by ``--resume``."""
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                completed.add(str(row["point_key"]))
    return completed


def _patch_precision_token(precision: Path, prefix: str, token: str) -> None:
    token = token.upper()
    if token.startswith("MXINT"):
        width = int(token.replace("MXINT", "").replace("_", ""))
        replace_localparam(precision, f"{prefix}_MX_INT_ENABLE", 1)
        replace_localparam(precision, f"{prefix}_MX_INT_WIDTH", width)
        return
    if token.startswith("MXFP"):
        if token not in MXFP_FORMATS:
            raise ValueError(f"unsupported MXFP format for HBM calibration: {token}")
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


def patch_hbm_config(point: Point, rtl_root: Path) -> None:
    """Patch dimensions, counters, and precision fields into worker RTL."""
    p = point.params
    configuration = rtl_root / "src/definitions/configuration.svh"
    precision = rtl_root / "src/definitions/precision.svh"
    replace_localparam(configuration, "MLEN", int(p["MLEN"]))
    replace_localparam(configuration, "VLEN", int(p["VLEN"]))
    replace_localparam(configuration, "BLEN", int(p["BLEN"]))
    replace_localparam(configuration, "HBM_M_Prefetch_Amount", int(p["HBM_M_Prefetch_Amount"]))
    replace_localparam(configuration, "HBM_V_Prefetch_Amount", int(p["HBM_V_Prefetch_Amount"]))
    replace_localparam(configuration, "HBM_V_Writeback_Amount", int(p["HBM_V_Writeback_Amount"]))
    replace_localparam(precision, "BLOCK_DIM", int(p["BLOCK_DIM"]))
    replace_localparam(precision, "MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "ACT_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "KV_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    replace_localparam(precision, "WT_MX_SCALE_WIDTH", int(p["MX_SCALE_WIDTH"]))
    _patch_precision_token(precision, "ACT", str(p["ACT_WIDTH"]))
    _patch_precision_token(precision, "KV", str(p["KV_WIDTH"]))
    _patch_precision_token(precision, "WT", str(p["WEIGHT_WIDTH"]))


def patch_tilelink_upsizer_for_large_widths(rtl_root: Path) -> None:
    """Apply the local width-safe TileLink elaboration fix in a worker copy."""
    """Guard worker-local TileLink upsizer elaboration for very wide HBM buses.

    The PLENA HBM controllers instantiate tl_adapter with the default
    TileLink MaxSize=6.  That is enough for small calibration points, but when
    HBM_ELE_WIDTH grows, HostDataWidth can exceed the default maximum transfer
    size.  Synopsys DC then sees a negative exponent in HostMaxBurstLen and
    elaborates HostBurstLenWidth as an illegal zero-width vector.  This patch is
    applied only to the temporary worker RTL copy used for area calibration; it
    keeps the source checkout untouched.
    """

    path = rtl_root / "src/memory/HBM/TileLink_Lib/tl_data_upsizer.sv"
    text = path.read_text()
    old = "localparam int unsigned HostMaxBurstLen = 2 ** (MaxSize - HostNonBurstSize);"
    new = (
        "localparam int unsigned HostMaxBurstLen = "
        "(MaxSize > HostNonBurstSize) ? (2 ** (MaxSize - HostNonBurstSize)) : 1;"
    )
    if old not in text:
        return
    path.write_text(text.replace(old, new))

    tracker = rtl_root / "src/memory/HBM/TileLink_Lib/tl_burst_tracker.sv"
    tracker_text = tracker.read_text()
    tracker_old = "localparam int unsigned MaxBurstLen = 2 ** (MaxSize - NonBurstSize),"
    tracker_new = (
        "localparam int unsigned MaxBurstLen = "
        "(MaxSize > NonBurstSize) ? (2 ** (MaxSize - NonBurstSize)) : 1,"
    )
    if tracker_old in tracker_text:
        tracker.write_text(tracker_text.replace(tracker_old, tracker_new))


def cleanup_hbm_build(worker_rtl: Path) -> None:
    """Delete HBM DC build output after report archival."""
    cleanup_worker_build(worker_rtl, Point("hbm_sys", "hbm_sys", "hbm_sys", {}))


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
    """Execute one HBM synthesis with license retry and quota-safe cleanup."""
    start = time.time()
    try:
        patch_hbm_config(point, worker_rtl)
        patch_tilelink_upsizer_for_large_widths(worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth hbm_sys 1000 area"
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
                cleanup_hbm_build(worker_rtl)
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
            cleanup_hbm_build(worker_rtl)


def load_complete_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Load successful rows and coerce regression features to floats."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete"]


def fit_and_write_coefficients(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Fit nonnegative HBM feature coefficients and write provenance metadata."""
    rows = load_complete_rows(csv_path)
    if rows:
        vals, mape = fit_nonnegative(rows, FEATURE_FIELDS)
        coeff_dict = {name: vals[idx] for idx, name in enumerate(COEFFICIENT_NAMES)}
        status = "fitted_from_local_plena_rtl_synth"
    else:
        coeff_dict = {}
        mape = float("nan")
        status = "bootstrap_insufficient_data"
    out = {
        "metadata": {
            "status": status,
            "source_csv": str(csv_path),
            "rows": len(rows),
            "mape_pct": mape,
            "feature_order": [
                "HBM_ELE_WIDTH",
                "HBM_SCALE_WIDTH",
                "MLEN * (WT_WIDTH + KV_WIDTH)",
                "VLEN * (ACT_WIDTH + KV_WIDTH)",
                "((MLEN + VLEN) / BLEN) * MX_SCALE_WIDTH",
                "SourceWidth * HBM_ADDR_WIDTH",
                "log2(M_LOAD + 1) + log2(V_LOAD + 1)",
                "log2(V_WRITE + 1)",
                "1",
            ],
        },
        "coefficients": coeff_dict,
    }
    path = run_dir / "hbm_model_coefficients.json"
    path.write_text(json.dumps(json_safe(out), indent=2, sort_keys=True))
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, CALIBRATION_DIR / path.name)


def write_complete_csv(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Export successful HBM points as a compact module CSV."""
    rows = load_complete_rows(csv_path)
    if not rows:
        return
    out = run_dir / "hbm_system.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, CALIBRATION_DIR / out.name)


def run_dry_run(points: list[Point], run_dir: Path, rtl_root: Path) -> None:
    """Materialize plan/commands without patching RTL or starting DC."""
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    commands = run_dir / "plans" / "commands.txt"
    commands.parent.mkdir(parents=True, exist_ok=True)
    commands.write_text(
        "\n".join(
            f"# {point.point_id}: MLEN={point.params['MLEN']} VLEN={point.params['VLEN']} "
            f"BLEN={point.params['BLEN']} ACT={point.params['ACT_WIDTH']} "
            f"KV={point.params['KV_WIDTH']} WT={point.params['WEIGHT_WIDTH']} "
            f"M_LOAD={point.params['HBM_M_Prefetch_Amount']} "
            f"V_LOAD={point.params['HBM_V_Prefetch_Amount']} "
            f"V_WRITE={point.params['HBM_V_Writeback_Amount']}\n"
            f"cd <worker-copy> && just synth hbm_sys 1000 area"
            for point in points
        )
        + "\n"
    )
    print(f"Dry run wrote {len(points)} planned HBMSystem points to {run_dir}")
    print(f"RTL root for real runs: {rtl_root}")


def parse_args() -> argparse.Namespace:
    """Parse standalone HBM calibration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "minimal-v1"], default="minimal-v1")
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
    """Run or dry-run HBM interface calibration."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"hbm_{safe_run_name}_{os.getpid()}"

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
