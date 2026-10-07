#!/usr/bin/env python3
"""Fail-on-drift checker for deterministic emulator timing goldens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLENA_ROOT = REPO_ROOT.parents[1]
DEFAULT_FIXTURE = Path(__file__).with_name("golden_workloads.json")
CHECK_KEYS = ("sim_latency_cycles", "hbm_bytes_read", "hbm_bytes_written")

for _path in (
    REPO_ROOT,
    REPO_ROOT / "PLENA_Compiler",
    REPO_ROOT / "PLENA_Tools",
    REPO_ROOT / "transactional_emulator" / "testbench",
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from transactional_emulator.testbench.emulator_runner import run_emulator


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        fixture = json.load(f)
    if fixture.get("schema_version") != 1:
        raise ValueError(f"Unsupported fixture schema_version in {path}")
    workloads = fixture.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError(f"Fixture {path} must contain a non-empty workloads list")
    return fixture


def _selected_workloads(fixture: dict[str, Any], selected_ids: set[str] | None) -> list[dict[str, Any]]:
    workloads = fixture["workloads"]
    if selected_ids is None:
        return workloads
    selected = [workload for workload in workloads if workload.get("id") in selected_ids]
    missing = selected_ids - {workload.get("id") for workload in selected}
    if missing:
        raise ValueError("Unknown workload id(s): " + ", ".join(sorted(missing)))
    return selected


def _compare_metrics(workload_id: str, expected: dict[str, int], actual: dict[str, Any]) -> list[str]:
    failures = []
    for key in CHECK_KEYS:
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        status = "PASS" if actual_value == expected_value else "FAIL"
        print(f"{workload_id:28s} {key:20s} expected={expected_value} actual={actual_value} {status}")
        if actual_value != expected_value:
            failures.append(
                f"{workload_id}.{key}: expected {expected_value}, got {actual_value}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--plena-root", type=Path, default=DEFAULT_PLENA_ROOT)
    parser.add_argument("--id", dest="ids", action="append", help="Run only this workload id; repeatable.")
    parser.add_argument("--threads", type=int, default=None, help="Override fixture thread count.")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    fixture = _load_fixture(args.fixture)
    defaults = fixture.get("defaults", {})
    workloads = _selected_workloads(fixture, set(args.ids) if args.ids else None)

    failures: list[str] = []
    for workload in workloads:
        workload_id = workload["id"]
        build_dir = args.plena_root / workload["build_dir"]
        settings_path = build_dir / "plena_settings.toml"
        if not build_dir.exists():
            raise FileNotFoundError(f"{workload_id}: build_dir does not exist: {build_dir}")
        if not settings_path.exists():
            raise FileNotFoundError(f"{workload_id}: missing settings file: {settings_path}")

        os.environ["PLENA_SETTINGS_TOML"] = str(settings_path)
        threads = args.threads
        if threads is None:
            threads = int(workload.get("threads", defaults.get("threads", 1)))
        stage_profile = bool(workload.get("stage_profile", defaults.get("stage_profile", False)))
        overlap_prefetch_compute = bool(
            workload.get(
                "overlap_prefetch_compute",
                defaults.get("overlap_prefetch_compute", False),
            )
        )

        print(f"\n=== timing golden: {workload_id} ===")
        metrics = run_emulator(
            build_dir,
            threads=threads,
            stage_profile=stage_profile,
            run_label=f"golden.{workload_id}",
            overlap_prefetch_compute=overlap_prefetch_compute,
        )
        failures.extend(_compare_metrics(workload_id, workload["expected"], metrics))

    if failures:
        print("\nTiming golden drift detected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll timing goldens matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
