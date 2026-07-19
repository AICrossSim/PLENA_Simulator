#!/usr/bin/env python3
"""Run the exhaustive Qwen3 grid as independent precision-profile shards.

The regular DSE runner remains the source of truth for evaluating one design.
This driver only changes orchestration: every precision profile receives an
independent Optuna study, avoiding SQLite writer contention between unrelated
profiles.  The completed shards are then merged into the same JSONL/CSV/report
layout consumed by the existing analysis and plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DSE_SCRIPT = HERE / "run_optuna_dse.py"
DEFAULT_ACCURACY = (
    HERE
    / "software_accuracy_inputs"
    / "software_precision_profiles_accuracy_gt_0p9.json"
)


def _load_dse_module() -> Any:
    spec = importlib.util.spec_from_file_location("qwen3_dse_shard_runtime", DSE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DSE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DSE = _load_dse_module()


@dataclass(frozen=True)
class Shard:
    index: int
    profile: str
    run_dir: Path


def _profiles(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("profiles", payload.get("precision_profiles"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a profile list")
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("precision profile names are not unique")
    return names


def _link_shared_cache(shard_dir: Path, cache_root: Path | None) -> None:
    if cache_root is None:
        return
    for name in ("compiler_trace_cache", "compiler_v4_work_cache"):
        source = cache_root / name
        if not source.is_dir():
            raise FileNotFoundError(f"shared cache directory is missing: {source}")
        target = shard_dir / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            raise ValueError(f"refusing to replace existing shard cache {target}")
        target.symlink_to(source.resolve(), target_is_directory=True)


def _shard_command(shard: Shard, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(DSE_SCRIPT),
        "--sampler", "grid",
        "--n-trials", "135",
        "--workers", "1",
        "--fixed-precision-profile", shard.profile,
        "--accuracy-constraints", str(args.accuracy_constraints),
        "--compiler-cost-mode", "roofline-objective",
        "--compiler-compute-timing", "rtl-v1",
        "--compiler-v4-memory-evaluation", "one-layer-cached-occurrence-scaled",
        "--native-layout-mode", "compact",
        "--packed-attention-schedule", "direct-first-block-v1",
        "--vector-scalar-schedule", "compiler-v1",
        "--no-legacy-bandwidth-prune",
        "--area-mode", "proxy-v2",
        "--target-area-mm2", str(args.target_area_mm2),
        "--area-budget-mm2", str(args.area_budget_mm2),
        "--run-dir", str(shard.run_dir),
    ]


def _shard_complete(shard: Shard) -> bool:
    summary = shard.run_dir / "run_summary.json"
    if not summary.exists():
        return False
    payload = json.loads(summary.read_text())
    return (
        int(payload.get("unique_grid_record_count", -1)) == 135
        and int(payload.get("completed", -1)) == 117
        and int(payload.get("pruned", -1)) == 18
        and int(payload.get("failed", -1)) == 0
    )


def run_shards(shards: list[Shard], args: argparse.Namespace) -> None:
    pending = [shard for shard in shards if not _shard_complete(shard)]
    active: dict[subprocess.Popen[str], tuple[Shard, Any]] = {}
    failures: list[tuple[Shard, int]] = []

    def launch(shard: Shard) -> None:
        shard.run_dir.mkdir(parents=True, exist_ok=True)
        _link_shared_cache(shard.run_dir, args.cache_source)
        handle = (shard.run_dir / "shard.log").open("a")
        process = subprocess.Popen(
            _shard_command(shard, args),
            cwd=DSE.REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active[process] = (shard, handle)

    while pending or active:
        while pending and len(active) < args.workers:
            launch(pending.pop(0))
        finished = [process for process in active if process.poll() is not None]
        if not finished:
            time.sleep(0.2)
            continue
        for process in finished:
            shard, handle = active.pop(process)
            handle.close()
            code = int(process.returncode or 0)
            if code != 0 or not _shard_complete(shard):
                failures.append((shard, code))

    if failures:
        details = ", ".join(
            f"{shard.index}:{shard.profile} (exit {code})"
            for shard, code in failures
        )
        raise RuntimeError(f"incomplete DSE shards: {details}")


def _csv_by_trial(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["trial"]): row for row in csv.DictReader(handle)}


def _pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic three-objective frontier in O(n log n)."""

    ordered = sorted(
        records,
        key=lambda row: (
            float(row["area_mm2"]),
            float(row["latency_ms"]),
            -float(row["accuracy_score"]),
            int(row["trial"]),
        ),
    )
    latencies = sorted({float(row["latency_ms"]) for row in ordered})
    rank = {value: index + 1 for index, value in enumerate(latencies)}
    tree = [float("-inf")] * (len(latencies) + 1)

    def query(index: int) -> float:
        result = float("-inf")
        while index > 0:
            result = max(result, tree[index])
            index -= index & -index
        return result

    def update(index: int, value: float) -> None:
        while index < len(tree):
            tree[index] = max(tree[index], value)
            index += index & -index

    front: list[dict[str, Any]] = []
    for row in ordered:
        index = rank[float(row["latency_ms"])]
        accuracy = float(row["accuracy_score"])
        if query(index) < accuracy:
            front.append(row)
        update(index, accuracy)
    return front


def _copy_trial(source: Path, target: Path, record: dict[str, Any]) -> None:
    if target.exists():
        shutil.rmtree(target)
    # Shards normally live on local /tmp while the publication run lives on
    # the home filesystem.  Hard links are not portable across that boundary.
    shutil.copytree(source, target, copy_function=shutil.copy2)
    record_path = target / "trial_record.json"
    record_path.unlink()
    DSE.write_json(record_path, record)


def merge_shards(shards: list[Shard], output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()

    for shard in shards:
        canonical = _csv_by_trial(shard.run_dir / "grid_trials.csv")
        for local_trial in sorted(canonical):
            source_dir = shard.run_dir / f"trial_{local_trial:04d}"
            record_path = source_dir / "trial_record.json"
            if not record_path.exists():
                raise FileNotFoundError(record_path)
            record = json.loads(record_path.read_text())
            grid_row = canonical[local_trial]
            record.setdefault("precision_profile", shard.profile)
            for field in ("MLEN", "BLEN", "INT_DATA_WIDTH"):
                record.setdefault(field, int(grid_row[field]))
            record.setdefault("VLEN", int(record["MLEN"]))
            key = (
                str(record["precision_profile"]),
                int(record["MLEN"]),
                int(record["BLEN"]),
                int(record["INT_DATA_WIDTH"]),
            )
            if key in seen:
                raise ValueError(f"duplicate merged grid point {key}")
            seen.add(key)
            global_trial = shard.index * 135 + local_trial
            record["trial"] = global_trial
            _copy_trial(source_dir, output_dir / f"trial_{global_trial:05d}", record)
            records.append(record)

    if len(records) != 13_905 or len(seen) != 13_905:
        raise ValueError(
            f"merged grid coverage is incomplete: records={len(records)}, unique={len(seen)}"
        )
    records.sort(key=lambda row: int(row["trial"]))
    complete = [row for row in records if row.get("state") == "complete"]
    pruned = [row for row in records if row.get("state") == "pruned"]
    failed = [row for row in records if row.get("state") == "failed"]
    if len(complete) != 12_051 or len(pruned) != 1_854 or failed:
        raise ValueError(
            "unexpected merged states: "
            f"complete={len(complete)}, pruned={len(pruned)}, failed={len(failed)}"
        )

    with (output_dir / "trials.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    DSE.write_records_csv(output_dir / "all_trials.csv", records)
    DSE.write_records_csv(output_dir / "grid_trials.csv", records)
    DSE.write_best_csv(output_dir / "best_trials.csv", complete)
    DSE.write_records_csv(output_dir / "pareto_trials.csv", _pareto_front(complete))

    selections = DSE.select_area_reference_candidates(
        complete,
        target_area_mm2=args.target_area_mm2,
        area_budget_mm2=args.area_budget_mm2,
        target_area_tolerance_pct=5.0,
    )
    DSE.write_json(
        output_dir / "a100_comparison.json",
        {
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": 5.0,
            "reference": "NVIDIA A100 826 mm2 die-area reference with a 110% feasibility budget",
            "ga100_reference_area_mm2": DSE.GA100_REFERENCE_AREA_MM2,
            "note": (
                "PLENA area is a calibrated logic plus SRAM-macro proxy and excludes "
                "physical HBM stacks/package. Large MLEN/BLEN points are structural "
                "area extrapolations and must retain their fidelity warnings."
            ),
            "feasible_trial_count": len(selections["feasible"]),
            "fidelity_qualified_trial_count": len(selections["fidelity_qualified"]),
            "fastest_under_area_budget": selections["fastest"],
            "fastest_fidelity_qualified_under_area_budget": selections["fastest_fidelity_qualified"],
            "highest_accuracy_under_area_budget": selections["highest_accuracy"],
            "closest_area_to_target_mm2": selections["closest_to_target"],
            "closest_area_below_target_mm2": selections["closest_below_target"],
            "within_target_area_tolerance": selections["within_tolerance"],
            "p90_conservative_feasible_trial_count": len(selections["p90_feasible"]),
            "p90_conservative_fastest_under_area_budget": selections["p90_fastest"],
            "p90_conservative_closest_area_to_target_mm2": selections["p90_closest_to_target"],
        },
    )

    first_summary = json.loads((shards[0].run_dir / "run_summary.json").read_text())
    first_summary.update(
        {
            "run_dir": str(output_dir),
            "n_trials": 13_905,
            "workers": args.workers,
            "workers_requested": args.workers,
            "precision_profile_count": len(shards),
            "optuna_storage_backend": "profile_sharded_journal",
            "serialized_optuna_ask": False,
            "shard_count": len(shards),
            "shard_grid_size": 135,
            "completed": len(complete),
            "pruned": len(pruned),
            "failed": len(failed),
            "attempt_count": len(records),
            "unique_grid_record_count": len(seen),
        }
    )
    DSE.write_json(output_dir / "run_summary.json", first_summary)
    DSE.write_json(
        output_dir / "shard_manifest.json",
        {
            "schema_version": 1,
            "orchestration": "independent_precision_profile_optuna_studies",
            "shards": [
                {"index": shard.index, "profile": shard.profile, "run_dir": str(shard.run_dir)}
                for shard in shards
            ],
            "coverage": {
                "profiles": len(shards),
                "points_per_profile": 135,
                "total_points": len(records),
                "unique_design_keys": len(seen),
            },
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--accuracy-constraints", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--cache-source", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--target-area-mm2", type=float, default=826.0)
    parser.add_argument("--area-budget-mm2", type=float, default=908.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    names = _profiles(args.accuracy_constraints)
    shards = [
        Shard(index, name, args.scratch_dir / "shards" / f"{index:03d}_{name}")
        for index, name in enumerate(names)
    ]
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    run_shards(shards, args)
    merge_shards(shards, args.run_dir, args)
    print(f"Wrote sharded exhaustive DSE run: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
