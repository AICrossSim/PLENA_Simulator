#!/usr/bin/env python3
"""Expand YAML plans and run heterogeneous area calibration campaigns.

One global queue feeds MatrixMachine, VectorMachine, ScalarMachine, HBMSystem,
and full-chip adapters. This prevents separate tmux jobs from oversubscribing
the shared DC license pool. Plans may combine named presets, explicit points,
and Cartesian grids; all expand into stable resumable jobs before synthesis.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_csv import COMMON_FIELDS, union_fields, write_rows  # noqa: E402
from calibration_merge import merge_calibration_csvs  # noqa: E402
from calibration_runtime import CalibrationJob, CompactExport, RuntimeConfig, run_calibration_jobs, stable_job_key  # noqa: E402
import run_full_chip_calibration as full_runner  # noqa: E402
import run_hbm_system_calibration as hbm_runner  # noqa: E402
import run_matrix_machine_calibration as matrix_runner  # noqa: E402
import run_scalar_machine_calibration as scalar_runner  # noqa: E402
import run_vector_machine_calibration as vector_runner  # noqa: E402
from analytic_models.area_new.hbm_model import hbm_features  # noqa: E402

RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_area_workers")


def _mxint_bits(token: str) -> int:
    token = str(token).upper().replace("_", "")
    if not token.startswith("MXINT"):
        raise ValueError(f"not an MXINT token: {token}")
    return int(token.replace("MXINT", ""))


def _mxfp_parts(token: str) -> tuple[int, int]:
    token = str(token).upper()
    match = re.match(r"MXFP_E(\d+)M(\d+)$", token)
    if not match:
        raise ValueError(f"not an MXFP token: {token}")
    return int(match.group(1)), int(match.group(2))


def _precision_mode(act: str, kv: str, weight: str) -> str:
    tokens = [str(act).upper(), str(kv).upper(), str(weight).upper()]
    if all(token.startswith("MXINT") for token in tokens):
        return "mxint"
    if all(token.startswith("MXFP") for token in tokens):
        return "mxfp"
    raise ValueError(f"mixed MXINT/MXFP precision is unsupported for calibration points: {tokens}")


def _fp_setting_parts(setting: str) -> tuple[int, int]:
    setting = str(setting).upper()
    match = re.match(r"FP_E(\d+)M(\d+)$", setting)
    if not match:
        raise ValueError(f"unsupported FP_SETTING: {setting}")
    return int(match.group(1)), int(match.group(2))


@dataclass
class Adapter:
    """Bridge a common scheduler job to a legacy module-specific runner."""

    name: str
    module: Any
    row_fields: list[str]

    def build_preset(self, item: dict[str, Any]) -> list[Any]:
        """Expand one adapter-specific named preset into point objects."""
        if self.name == "matrix_machine":
            return matrix_runner.build_plan(str(item["mode"]))
        return self.module.build_plan(str(item.get("preset", "smoke")))

    def point_from_params(self, point_id: str, params: dict[str, Any], item: dict[str, Any]) -> Any:
        """Normalize explicit YAML parameters into the runner's point type."""
        if self.name == "matrix_machine":
            return _matrix_point(point_id, params, item)
        if self.name == "vector_machine":
            return _vector_point(point_id, params)
        if self.name == "scalar_machine":
            return _scalar_point(point_id, params)
        if self.name == "hbm_system":
            return _hbm_point(point_id, params)
        if self.name == "full_chip":
            return _full_chip_point(point_id, params)
        raise ValueError(f"unknown runner: {self.name}")

    def run_point(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Delegate synthesis and report parsing to the module runner."""
        return self.module.run_point(*args, **kwargs)

    def compact_exports(self) -> list[CompactExport]:
        """Describe successful-point CSVs and their semantic deduplication keys."""
        if self.name == "matrix_machine":
            return [
                CompactExport(
                    "matrix_machine_mxint.csv",
                    matrix_runner.CSV_FIELDS,
                    lambda r: r.get("level") == "matrix_machine" and r.get("mode") == "mxint",
                    lambda r: (
                        r.get("MLEN"),
                        r.get("BLEN"),
                        r.get("ACT_WIDTH"),
                        r.get("KV_WIDTH"),
                        r.get("WEIGHT_WIDTH"),
                        r.get("T_BITS"),
                        r.get("L_BITS"),
                        r.get("scale_width"),
                    ),
                ),
                CompactExport(
                    "matrix_machine_mxfp.csv",
                    matrix_runner.CSV_FIELDS,
                    lambda r: r.get("level") == "matrix_machine" and r.get("mode") == "mxfp",
                    lambda r: (
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
                ),
            ]
        if self.name == "vector_machine":
            return [
                CompactExport(
                    "vector_machine.csv",
                    vector_runner.CSV_FIELDS,
                    lambda r: True,
                    lambda r: (r.get("VLEN"), r.get("FP_SETTING"), r.get("V_FP_EXP_WIDTH"), r.get("V_FP_MANT_WIDTH")),
                )
            ]
        if self.name == "scalar_machine":
            return [
                CompactExport(
                    "scalar_machine.csv",
                    scalar_runner.CSV_FIELDS,
                    lambda r: bool(r.get("MLEN")) and bool(r.get("VLEN")),
                    lambda r: (
                        r.get("MLEN"),
                        r.get("VLEN"),
                        r.get("INT_DATA_WIDTH"),
                        r.get("FP_SETTING"),
                        r.get("S_FP_EXP_WIDTH"),
                        r.get("S_FP_MANT_WIDTH"),
                    ),
                )
            ]
        if self.name == "hbm_system":
            return [
                CompactExport(
                    "hbm_system.csv",
                    hbm_runner.CSV_FIELDS,
                    lambda r: True,
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
            ]
        if self.name == "full_chip":
            return [
                CompactExport(
                    "full_chip_anchors.csv",
                    full_runner.CSV_FIELDS,
                    lambda r: True,
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
            ]
        return []


ADAPTERS: dict[str, Adapter] = {
    "matrix_machine": Adapter("matrix_machine", matrix_runner, matrix_runner.CSV_FIELDS),
    "vector_machine": Adapter("vector_machine", vector_runner, vector_runner.CSV_FIELDS),
    "scalar_machine": Adapter("scalar_machine", scalar_runner, scalar_runner.CSV_FIELDS),
    "hbm_system": Adapter("hbm_system", hbm_runner, hbm_runner.CSV_FIELDS),
    "full_chip": Adapter("full_chip", full_runner, full_runner.CSV_FIELDS),
}


def _matrix_point(point_id: str, params: dict[str, Any], item: dict[str, Any]) -> Any:
    params = dict(params)
    mode = str(item.get("mode") or params.get("mode") or _precision_mode(params["ACT_WIDTH"], params["KV_WIDTH"], params["WEIGHT_WIDTH"]))
    params.setdefault("scale_width", 8)
    if mode == "mxint":
        params.setdefault("T_BITS", max(_mxint_bits(params["KV_WIDTH"]), _mxint_bits(params["WEIGHT_WIDTH"])))
        params.setdefault("L_BITS", _mxint_bits(params["ACT_WIDTH"]))
    elif mode == "mxfp":
        kv = _mxfp_parts(params["KV_WIDTH"])
        wt = _mxfp_parts(params["WEIGHT_WIDTH"])
        act = _mxfp_parts(params["ACT_WIDTH"])
        params.setdefault("T_EXP", max(kv[0], wt[0]))
        params.setdefault("T_MANT", max(kv[1], wt[1]))
        params.setdefault("L_EXP", act[0])
        params.setdefault("L_MANT", act[1])
    else:
        raise ValueError(f"unsupported matrix_machine mode: {mode}")
    return matrix_runner.Point(
        point_id=point_id,
        level=str(params.get("level", "matrix_machine")),
        mode=mode,
        module=str(params.get("module", "matrix_machine")),
        top_module=str(params.get("top_module", "matrix_machine")),
        params=params,
    )


def _vector_point(point_id: str, params: dict[str, Any]) -> Any:
    params = dict(params)
    exp, mant = _fp_setting_parts(str(params.get("FP_SETTING", "FP_E5M6")))
    params.setdefault("V_FP_EXP_WIDTH", exp)
    params.setdefault("V_FP_MANT_WIDTH", mant)
    params.setdefault("fp_width", 1 + int(params["V_FP_EXP_WIDTH"]) + int(params["V_FP_MANT_WIDTH"]))
    params.setdefault("preset", "explicit")
    return vector_runner.Point(point_id, "vector_machine", "vector_machine", params)


def _scalar_point(point_id: str, params: dict[str, Any]) -> Any:
    params = dict(params)
    exp, mant = _fp_setting_parts(str(params.get("FP_SETTING", "FP_E5M6")))
    params.setdefault("S_FP_EXP_WIDTH", exp)
    params.setdefault("S_FP_MANT_WIDTH", mant)
    params.setdefault("fp_width", 1 + int(params["S_FP_EXP_WIDTH"]) + int(params["S_FP_MANT_WIDTH"]))
    params.setdefault("INT_DATA_WIDTH", 32)
    params.setdefault("VLEN", params.get("MLEN", 16))
    params.setdefault("MLEN", params["VLEN"])
    params.setdefault("preset", "explicit")
    params.update(
        scalar_runner.features(
            int(params["INT_DATA_WIDTH"]),
            int(params["S_FP_EXP_WIDTH"]),
            int(params["S_FP_MANT_WIDTH"]),
            int(params["MLEN"]),
            int(params["VLEN"]),
        )
    )
    return scalar_runner.Point(point_id, "scalar_machine", "scalar_machine", params)


def _hbm_point(point_id: str, params: dict[str, Any]) -> Any:
    params = dict(params)
    params.setdefault("MLEN", 128)
    params.setdefault("VLEN", params["MLEN"])
    params.setdefault("BLEN", 16)
    params.setdefault("BLOCK_DIM", params["BLEN"])
    params.setdefault("ACT_WIDTH", "MXINT4")
    params.setdefault("KV_WIDTH", "MXINT4")
    params.setdefault("WEIGHT_WIDTH", "MXINT4")
    params.setdefault("MX_SCALE_WIDTH", 8)
    params.setdefault("HBM_M_Prefetch_Amount", params["MLEN"])
    params.setdefault("HBM_V_Prefetch_Amount", 64)
    params.setdefault("HBM_V_Writeback_Amount", 64)
    params.setdefault("preset", "explicit")
    params.update(
        {
            f"feat_{key}": value
            for key, value in hbm_features(params).items()
            if f"feat_{key}" in hbm_runner.CSV_FIELDS
        }
    )
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
    return hbm_runner.Point(point_id, "hbm_sys", "hbm_sys", params)


def _full_chip_point(point_id: str, params: dict[str, Any]) -> Any:
    explicit_params = dict(params)
    base = full_runner._mk_params(preset="explicit")  # noqa: SLF001 - adapter intentionally reuses runner defaults.
    base.update(explicit_params)
    mlen = int(base["MLEN"])
    vlen = int(base["VLEN"])
    hlen = int(base["HLEN"])
    if "MATRIX_SRAM_DEPTH" not in explicit_params and "MATRIX_SRAM_SIZE" not in explicit_params:
        base["MATRIX_SRAM_DEPTH"] = 2 * mlen
    if "VECTOR_SRAM_DEPTH" not in explicit_params and "VECTOR_SRAM_SIZE" not in explicit_params:
        base["VECTOR_SRAM_DEPTH"] = max(32, 2 * hlen + max(1, mlen // max(vlen, 1)))
    if "INT_SRAM_DEPTH" not in explicit_params:
        base["INT_SRAM_DEPTH"] = 32
    if "FP_SRAM_DEPTH" not in explicit_params:
        base["FP_SRAM_DEPTH"] = mlen + 4
    base.setdefault("preset", "explicit")
    return full_runner.Point(point_id, "plena", "plena", base)


def _expand_grid(item: dict[str, Any]) -> list[dict[str, Any]]:
    base = dict(item.get("base", {}))
    sweep = dict(item.get("sweep", {}))
    keys = list(sweep)
    values = [list(sweep[key]) for key in keys]
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        rows.append(params)
    return rows


def load_plan(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping top-level documents."""
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("scheduler plan must be a YAML mapping")
    return raw


def build_jobs(plan: dict[str, Any], source_plan: Path, *, point_regex: str | None = None, limit: int | None = None) -> tuple[list[CalibrationJob], dict[str, Adapter]]:
    """Expand presets, explicit jobs, and grids into deduplicated jobs.

    Deduplication uses the full stable job key, not ``point_id`` alone, so a
    reused human-readable name cannot silently alias different parameters.
    """
    jobs: list[CalibrationJob] = []
    adapters_used: dict[str, Adapter] = {}

    def add_point(runner: str, point: Any, source: str) -> None:
        adapter = ADAPTERS[runner]
        job = CalibrationJob(stable_job_key(runner, point), runner, point, adapter, source)
        if point_regex and not re.search(point_regex, getattr(point, "point_id", "")):
            return
        adapters_used[runner] = adapter
        jobs.append(job)

    for idx, item in enumerate(plan.get("preset_jobs", []) or []):
        runner = str(item["runner"])
        adapter = ADAPTERS[runner]
        for point in adapter.build_preset(item):
            add_point(runner, point, f"{source_plan}:preset_jobs[{idx}]")

    for idx, item in enumerate(plan.get("explicit_jobs", []) or []):
        runner = str(item["runner"])
        adapter = ADAPTERS[runner]
        point_id = str(item.get("point_id") or f"{runner}_explicit_{idx}")
        point = adapter.point_from_params(point_id, dict(item.get("params", {})), item)
        add_point(runner, point, f"{source_plan}:explicit_jobs[{idx}]")

    for idx, item in enumerate(plan.get("grids", []) or []):
        runner = str(item["runner"])
        adapter = ADAPTERS[runner]
        for grid_idx, params in enumerate(_expand_grid(item)):
            safe = json.dumps(params, sort_keys=True)
            import hashlib

            point_id = str(item.get("point_id_prefix") or f"{runner}_grid_{idx}") + "_" + hashlib.sha1(safe.encode()).hexdigest()[:10]
            point = adapter.point_from_params(point_id, params, item)
            add_point(runner, point, f"{source_plan}:grids[{idx}][{grid_idx}]")

    # Preserve order while removing duplicate job keys.
    deduped: dict[str, CalibrationJob] = {}
    for job in jobs:
        deduped[job.job_key] = job
    jobs = list(deduped.values())
    if limit is not None:
        jobs = jobs[:limit]
    return jobs, adapters_used


def write_expanded_plan(jobs: list[CalibrationJob], path: Path, adapters: dict[str, Adapter]) -> None:
    """Persist the fully expanded queue before any expensive synthesis starts."""
    rows = []
    for job in jobs:
        params = getattr(job.point, "params", {})
        rows.append(
            {
                "job_key": job.job_key,
                "point_key": getattr(job.point, "point_key", ""),
                "point_id": getattr(job.point, "point_id", ""),
                "runner": job.runner,
                "module": getattr(job.point, "module", ""),
                "top_module": getattr(job.point, "top_module", ""),
                "source_plan": job.source_plan,
                **params,
            }
        )
    fields = union_fields(COMMON_FIELDS, *(adapter.row_fields for adapter in adapters.values()), *(row.keys() for row in rows))
    write_rows(path, rows, fields)


def parse_args() -> argparse.Namespace:
    """Parse scheduler plan, worker, resume, retry, and cleanup options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--reserve", type=int)
    parser.add_argument("--worker-root", type=Path, default=Path("/tmp/plena_rtl_area_workers"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--point-id-regex")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cleanup-worker-builds", action="store_true", default=True)
    parser.add_argument("--no-cleanup-worker-builds", dest="cleanup_worker_builds", action="store_false")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--license-retry-wait-sec", type=float)
    parser.add_argument("--license-max-retries", type=int, default=0)
    parser.add_argument("--no-merge-to-calibration", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Expand the requested plan and invoke the shared calibration runtime."""
    args = parse_args()
    plan = load_plan(args.plan)
    defaults = dict(plan.get("defaults", {}) or {})
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs, adapters = build_jobs(plan, args.plan, point_regex=args.point_id_regex, limit=args.limit)
    write_expanded_plan(jobs, run_dir / "plan.expanded.csv", adapters)

    if args.dry_run:
        print(f"Dry run expanded {len(jobs)} jobs into {run_dir / 'plan.expanded.csv'}")
        print("Runners:", ", ".join(sorted(adapters)))
        return 0

    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
    worker_root = args.worker_root / f"scheduler_{safe_run_name}_{os.getpid()}"
    reserve = args.reserve if args.reserve is not None else int(defaults.get("reserve_licenses", 1))
    rtl_root = args.rtl_root or Path(defaults.get("rtl_root", RTL_ROOT))
    retry_wait = args.license_retry_wait_sec if args.license_retry_wait_sec is not None else float(defaults.get("license_retry_wait_sec", 60))
    config = RuntimeConfig(
        run_dir=run_dir,
        rtl_root=Path(rtl_root),
        worker_root=worker_root,
        workers=args.workers,
        reserve=reserve,
        cleanup_worker_builds=bool(args.cleanup_worker_builds and defaults.get("cleanup_worker_builds", True)),
        keep_workers=args.keep_workers,
        resume=args.resume,
        skip_failed=args.skip_failed,
        retry_failed=args.retry_failed,
        license_retry_wait_sec=retry_wait,
        license_max_retries=args.license_max_retries,
    )
    code = run_calibration_jobs(jobs, adapters, config)
    if not args.no_merge_to_calibration:
        summary = merge_calibration_csvs()
        (run_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
