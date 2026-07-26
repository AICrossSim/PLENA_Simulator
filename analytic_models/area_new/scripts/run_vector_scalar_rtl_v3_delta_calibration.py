#!/usr/bin/env python3
"""Calibrate the rtl-v3 Vector/Scalar area as a paired RTL delta.

The pre-v3 VectorMachine and ScalarMachine models already have broad DC
coverage.  Repeating that sweep after adding segment-parallel vector logic and
the scalar ROB would be expensive and would discard useful data.  This driver
therefore synthesizes a small set of identical configurations from:

* baseline RTL commit ``0beb43f``; and
* the current RTL working tree containing rtl-v3.

It fits only ``new_area - baseline_area`` with nonnegative structural features,
keeps one point per machine as a holdout, and writes compact paired CSV/JSON
artifacts next to the existing area calibration files.  Worker trees and the
temporary baseline checkout live under /tmp and are removed on normal or
exceptional exit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from run_matrix_machine_calibration import fit_nonnegative, json_safe, preflight_tmp_workers
except ModuleNotFoundError:
    from .run_matrix_machine_calibration import fit_nonnegative, json_safe, preflight_tmp_workers


ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"
DEFAULT_RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_BASELINE = "0beb43f"
MODEL_VERSION = "vector_scalar_rtl_v3_paired_delta_v1"

VECTOR_KEYS = [
    "hier_total_area",
    "hier_element_area",
    "hier_element_lane_area",
    "hier_reduction_area",
    "hier_reduction_layer_area",
    "hier_buffer_area",
    "hier_top_glue_area",
]
SCALAR_KEYS = [
    "hier_total_area",
    "hier_int_alu_area",
    "hier_fp_alu_area",
    "hier_fp_sfu_area",
    "hier_scalar_sram_area",
    "hier_top_glue_area",
]


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _latest_complete(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "complete":
                continue
            rows[tuple(str(row.get(key, "")) for key in keys)] = row
    return rows


def _seed_baseline_from_compact(machine: str, destination: Path) -> int:
    """Reuse exact pre-v3 shapes already retained in the compact database.

    A fresh V16 E5M6 synthesis is always run by the smoke flow before this
    option is used.  It verifies that the historical and current DC/ASAP7 flow
    reproduce bit-for-bit area.  Reusing the remaining identical baseline
    rows then avoids spending tens of minutes on a redundant VLEN=64 compile.
    """
    if machine == "vector":
        try:
            from run_vector_machine_calibration import CSV_FIELDS, build_plan
        except ModuleNotFoundError:
            from .run_vector_machine_calibration import CSV_FIELDS, build_plan
        source = CALIBRATION_DIR / "vector_machine.csv"
        shape_keys = ("VLEN", "FP_SETTING")
    else:
        try:
            from run_scalar_machine_calibration import CSV_FIELDS, build_plan
        except ModuleNotFoundError:
            from .run_scalar_machine_calibration import CSV_FIELDS, build_plan
        source = CALIBRATION_DIR / "scalar_machine.csv"
        shape_keys = ("MLEN", "VLEN", "INT_DATA_WIDTH", "FP_SETTING")
    historical = _latest_complete(source, shape_keys)
    existing = _latest_complete(destination, ("point_key",))
    destination.parent.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for point in build_plan("rtl-v3-delta"):
        shape = tuple(str(point.params.get(key, "")) for key in shape_keys)
        old = historical.get(shape)
        if old is None or (point.point_key,) in existing:
            continue
        row = {field: old.get(field, "") for field in CSV_FIELDS}
        row.update(
            {
                "point_key": point.point_key,
                "point_id": point.point_id,
                "module": point.module,
                "top_module": point.top_module,
                "status": "complete",
                "worker_id": "historical_verified",
                "failure_reason": "",
            }
        )
        for key, value in point.params.items():
            if key in row:
                row[key] = value
        exists = destination.exists()
        with destination.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        seeded += 1
    return seeded


def _rtl_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in [
        "src/definitions/operation.svh",
        "src/definitions/configuration.svh",
        "src/vector_machine/rtl",
        "src/scalar_machine/rtl",
        "src/control/rtl/pipeline_control.sv",
        "src/core/rtl/plena.sv",
    ]:
        path = root / relative
        paths = sorted(path.rglob("*.sv")) if path.is_dir() else [path]
        for item in paths:
            if not item.exists():
                continue
            digest.update(str(item.relative_to(root)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _export_baseline(rtl_root: Path, revision: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = destination.parent / f"{destination.name}.tar"
    subprocess.run(
        ["git", "-C", str(rtl_root), "archive", "--format=tar", "-o", str(archive), revision],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    archive.unlink(missing_ok=True)
    # ASAP7 .db files are intentionally not tracked by git, but paired area
    # measurements must use the exact same target libraries.  Copy only this
    # compact local PDK payload; generated analyzed/build directories remain
    # excluded.
    source_lib = rtl_root / "tools/synopsys/lib"
    target_lib = destination / "tools/synopsys/lib"
    if not source_lib.exists():
        raise FileNotFoundError(f"ASAP7 synthesis libraries are missing: {source_lib}")
    shutil.copytree(source_lib, target_lib, dirs_exist_ok=True)


def _run_subrunner(
    *,
    runner: str,
    preset: str,
    run_dir: Path,
    rtl_root: Path,
    worker_root: Path,
    workers: int,
    resume: bool,
    dry_run: bool,
    limit: int | None,
) -> None:
    script = Path(__file__).with_name(runner)
    command = [
        sys.executable,
        str(script),
        "--preset",
        preset,
        "--run-dir",
        str(run_dir),
        "--rtl-root",
        str(rtl_root),
        "--workers",
        str(workers),
        "--worker-root",
        str(worker_root),
        "--cleanup-worker-builds",
        "--no-copy-to-calibration",
    ]
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    print("[phase]", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PLENA_RTL_NIX_ROOT"] = str(DEFAULT_RTL_ROOT)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _pair_rows(
    baseline_csv: Path,
    current_csv: Path,
    *,
    machine: str,
) -> list[dict[str, Any]]:
    if machine == "vector":
        shape_keys = ("VLEN", "FP_SETTING")
        hierarchy_keys = VECTOR_KEYS
    else:
        shape_keys = ("MLEN", "VLEN", "INT_DATA_WIDTH", "FP_SETTING")
        hierarchy_keys = SCALAR_KEYS
    baseline = _latest_complete(baseline_csv, shape_keys)
    current = _latest_complete(current_csv, shape_keys)
    missing = sorted(set(baseline) ^ set(current))
    if missing:
        raise RuntimeError(f"unpaired {machine} calibration shapes: {missing}")
    paired: list[dict[str, Any]] = []
    for key in sorted(baseline):
        old = baseline[key]
        new = current[key]
        fp_width = int(new["fp_width"])
        vlen = int(new["VLEN"])
        row: dict[str, Any] = {
            "machine": machine,
            "point_id": new["point_id"],
            "MLEN": new.get("MLEN", ""),
            "VLEN": vlen,
            "INT_DATA_WIDTH": new.get("INT_DATA_WIDTH", ""),
            "FP_SETTING": new["FP_SETTING"],
            "fp_width": fp_width,
            "baseline_area_um2": _float(old, "area_um2"),
            "rtl_v3_area_um2": _float(new, "area_um2"),
            "delta_area_um2": _float(new, "area_um2") - _float(old, "area_um2"),
            "baseline_elapsed_sec": _float(old, "elapsed_sec"),
            "rtl_v3_elapsed_sec": _float(new, "elapsed_sec"),
            "baseline_report_dir": old.get("report_dir", ""),
            "rtl_v3_report_dir": new.get("report_dir", ""),
            "feat_vlen_width": float(vlen * fp_width),
            "feat_fp_width": float(fp_width),
            "feat_const": 1.0,
        }
        for name in hierarchy_keys:
            short = name.removeprefix("hier_").removesuffix("_area")
            row[f"baseline_{short}_area"] = _float(old, name)
            row[f"rtl_v3_{short}_area"] = _float(new, name)
            row[f"delta_{short}_area"] = _float(new, name) - _float(old, name)
        paired.append(row)
    return paired


def _mape(actual: list[float], predicted: list[float]) -> float:
    if not actual:
        return float("nan")
    return sum(abs(p - a) / max(abs(a), 1e-9) for a, p in zip(actual, predicted)) / len(actual) * 100.0


def _error_percentiles(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = sorted(abs(p - a) / max(abs(a), 1e-9) * 100.0 for a, p in zip(actual, predicted))
    if not errors:
        return {"median_pct": float("nan"), "p90_pct": float("nan"), "max_pct": float("nan")}

    def quantile(q: float) -> float:
        index = min(len(errors) - 1, max(0, math.ceil(q * len(errors)) - 1))
        return errors[index]

    return {"median_pct": quantile(0.5), "p90_pct": quantile(0.9), "max_pct": errors[-1]}


def _fit_vector(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    holdout = [row for row in rows if int(row["VLEN"]) == 32 and row["FP_SETTING"] == "FP_E8M5"]
    train = [row for row in rows if row not in holdout]
    coefficients, train_delta_mape = fit_nonnegative(
        train, ["feat_vlen_width", "feat_const"], target="delta_area_um2"
    )
    coeffs = {"delta_vlen_width": coefficients[0], "delta_const": coefficients[1]}
    for row in rows:
        row["split"] = "holdout" if row in holdout else "train"
        row["predicted_delta_area_um2"] = (
            coeffs["delta_vlen_width"] * float(row["feat_vlen_width"])
            + coeffs["delta_const"]
        )
        row["predicted_rtl_v3_area_um2"] = (
            float(row["baseline_area_um2"]) + float(row["predicted_delta_area_um2"])
        )
        row["total_error_pct"] = abs(
            float(row["predicted_rtl_v3_area_um2"]) - float(row["rtl_v3_area_um2"])
        ) / max(float(row["rtl_v3_area_um2"]), 1e-9) * 100.0
    artifact = {
        "model_version": MODEL_VERSION,
        "machine": "vector_machine",
        "coefficients": coeffs,
        "equation": "delta = delta_vlen_width * VLEN * fp_width + delta_const",
        "metadata": {
            "status": "fitted_from_local_plena_rtl_synth",
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "train_delta_mape_pct": train_delta_mape,
            "train_total_mape_pct": _mape(
                [float(row["rtl_v3_area_um2"]) for row in train],
                [float(row["predicted_rtl_v3_area_um2"]) for row in train],
            ),
            "holdout_total_errors": _error_percentiles(
                [float(row["rtl_v3_area_um2"]) for row in holdout],
                [float(row["predicted_rtl_v3_area_um2"]) for row in holdout],
            ),
        },
    }
    return artifact, rows


def _fit_scalar(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    holdout = [row for row in rows if int(row["INT_DATA_WIDTH"]) == 64]
    train = [row for row in rows if row not in holdout]
    coefficients, train_delta_mape = fit_nonnegative(
        train, ["feat_vlen_width", "feat_fp_width", "feat_const"], target="delta_area_um2"
    )
    coeffs = {
        "delta_vlen_width": coefficients[0],
        "delta_fp_width": coefficients[1],
        "delta_const": coefficients[2],
    }
    for row in rows:
        row["split"] = "holdout" if row in holdout else "train"
        row["predicted_delta_area_um2"] = (
            coeffs["delta_vlen_width"] * float(row["feat_vlen_width"])
            + coeffs["delta_fp_width"] * float(row["feat_fp_width"])
            + coeffs["delta_const"]
        )
        row["predicted_rtl_v3_area_um2"] = (
            float(row["baseline_area_um2"]) + float(row["predicted_delta_area_um2"])
        )
        row["total_error_pct"] = abs(
            float(row["predicted_rtl_v3_area_um2"]) - float(row["rtl_v3_area_um2"])
        ) / max(float(row["rtl_v3_area_um2"]), 1e-9) * 100.0
    artifact = {
        "model_version": MODEL_VERSION,
        "machine": "scalar_machine",
        "coefficients": coeffs,
        "equation": "delta = delta_vlen_width * VLEN * fp_width + delta_fp_width * fp_width + delta_const",
        "metadata": {
            "status": "fitted_from_local_plena_rtl_synth",
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "train_delta_mape_pct": train_delta_mape,
            "train_total_mape_pct": _mape(
                [float(row["rtl_v3_area_um2"]) for row in train],
                [float(row["predicted_rtl_v3_area_um2"]) for row in train],
            ),
            "holdout_total_errors": _error_percentiles(
                [float(row["rtl_v3_area_um2"]) for row in holdout],
                [float(row["predicted_rtl_v3_area_um2"]) for row in holdout],
            ),
        },
    }
    return artifact, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    run_dir: Path,
    vector_rows: list[dict[str, Any]],
    scalar_rows: list[dict[str, Any]],
    *,
    baseline_revision: str,
    current_fingerprint: str,
    copy_to_calibration: bool,
) -> None:
    vector_artifact, vector_rows = _fit_vector(vector_rows)
    scalar_artifact, scalar_rows = _fit_scalar(scalar_rows)
    common = {
        "baseline_revision": baseline_revision,
        "current_rtl_fingerprint": current_fingerprint,
        "source_run_dir": str(run_dir),
        "synthesis": "Synopsys DC area mode, identical ASAP7 constraints",
    }
    vector_artifact["metadata"].update(common)
    scalar_artifact["metadata"].update(common)
    outputs = {
        "vector_rtl_v3_delta.csv": vector_rows,
        "scalar_rtl_v3_delta.csv": scalar_rows,
    }
    for name, rows in outputs.items():
        _write_csv(run_dir / name, rows)
    json_outputs = {
        "vector_rtl_v3_delta_coefficients.json": vector_artifact,
        "scalar_rtl_v3_delta_coefficients.json": scalar_artifact,
    }
    for name, artifact in json_outputs.items():
        (run_dir / name).write_text(json.dumps(json_safe(artifact), indent=2, sort_keys=True) + "\n")
    summary = {
        "model_version": MODEL_VERSION,
        "vector": vector_artifact["metadata"],
        "scalar": scalar_artifact["metadata"],
        "vector_delta_range_um2": [
            min(float(row["delta_area_um2"]) for row in vector_rows),
            max(float(row["delta_area_um2"]) for row in vector_rows),
        ],
        "scalar_delta_range_um2": [
            min(float(row["delta_area_um2"]) for row in scalar_rows),
            max(float(row["delta_area_um2"]) for row in scalar_rows),
        ],
    }
    (run_dir / "calibration_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        for name in [*outputs, *json_outputs]:
            shutil.copy2(run_dir / name, CALIBRATION_DIR / name)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument("--baseline-revision", default=DEFAULT_BASELINE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="limit each machine/revision phase for smoke testing")
    parser.add_argument("--no-copy-to-calibration", action="store_true")
    parser.add_argument(
        "--no-reuse-existing-baseline",
        action="store_true",
        help="force all baseline shapes to rerun instead of reusing verified compact rows",
    )
    parser.add_argument("--keep-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    workers, preflight = preflight_tmp_workers(args.workers)
    (args.run_dir / "tmp_preflight.json").write_text(json.dumps(json_safe(preflight), indent=2) + "\n")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_dir.name)
    temp_root = Path(f"/tmp/plena_rtl_v3_area_delta_{safe_name}_{os.getpid()}")
    baseline_root = temp_root / "baseline_rtl"
    current_fingerprint = _rtl_fingerprint(args.rtl_root)
    try:
        _export_baseline(args.rtl_root, args.baseline_revision, baseline_root)
        if not args.no_reuse_existing_baseline and not args.dry_run:
            vector_seeded = _seed_baseline_from_compact(
                "vector", args.run_dir / "raw/vector_baseline/calibration_points.csv"
            )
            scalar_seeded = _seed_baseline_from_compact(
                "scalar", args.run_dir / "raw/scalar_baseline/calibration_points.csv"
            )
            print(
                f"[baseline-reuse] vector={vector_seeded} scalar={scalar_seeded}",
                flush=True,
            )
        phases = [
            ("vector_baseline", "run_vector_machine_calibration.py", "rtl-v3-delta", baseline_root),
            ("vector_rtl_v3", "run_vector_machine_calibration.py", "rtl-v3-delta", args.rtl_root),
            ("scalar_baseline", "run_scalar_machine_calibration.py", "rtl-v3-delta", baseline_root),
            ("scalar_rtl_v3", "run_scalar_machine_calibration.py", "rtl-v3-delta", args.rtl_root),
        ]
        for phase, runner, preset, rtl_root in phases:
            _run_subrunner(
                runner=runner,
                preset=preset,
                run_dir=args.run_dir / "raw" / phase,
                rtl_root=rtl_root,
                worker_root=temp_root / "workers" / phase,
                workers=workers,
                resume=args.resume,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        if args.dry_run:
            return 0
        vector_rows = _pair_rows(
            args.run_dir / "raw/vector_baseline/calibration_points.csv",
            args.run_dir / "raw/vector_rtl_v3/calibration_points.csv",
            machine="vector",
        )
        scalar_rows = _pair_rows(
            args.run_dir / "raw/scalar_baseline/calibration_points.csv",
            args.run_dir / "raw/scalar_rtl_v3/calibration_points.csv",
            machine="scalar",
        )
        _write_artifacts(
            args.run_dir,
            vector_rows,
            scalar_rows,
            baseline_revision=args.baseline_revision,
            current_fingerprint=current_fingerprint,
            copy_to_calibration=not args.no_copy_to_calibration,
        )
    finally:
        if not args.keep_baseline:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
