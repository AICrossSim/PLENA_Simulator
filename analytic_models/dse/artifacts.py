"""Compact, resume-safe artifacts for large Optuna studies."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_COMPACT_TRIAL_DROP_KEYS = frozenset(
    {
        "area_metrics",
        "area_breakdown",
        "area_new_breakdown",
        "area_new_inputs",
        "area_proxy_breakdown",
        "area_proxy_inputs",
        "attention_schedule_layout",
        "bandwidth_shadow",
        "compiler_compute_validation",
        "compiler_hbm_traffic_breakdown",
        "external_hbm_energy_by_opcode",
        "external_hbm_energy_by_role",
        "external_hbm_energy_by_stage",
        "multi_chip",
        "packed_attention_metadata",
        "power_shadow",
        "vector_scalar_optimization_metadata",
    }
)


def load_json(path: Path) -> dict[str, Any]:
    handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")
    with handle as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = (
        gzip.open(path, "wt", compresslevel=1)
        if path.suffix == ".gz"
        else path.open("w")
    )
    with handle as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def compact_trial_record(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in record.items()
        if key not in _COMPACT_TRIAL_DROP_KEYS
    }
    compact["artifact_record_schema"] = "compact_trial_v1"
    compact["detail_artifact"] = "trial_detail.json.gz"
    return compact


def trial_lifecycle_record(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "trial",
        "state",
        "reason",
        "latency_ms",
        "area_mm2",
        "system_energy_nominal_mj",
        "accuracy_score",
        "precision_profile",
        "MLEN",
        "BLEN",
        "VLEN",
        "chip_count",
        "tp_degree",
        "cp_degree",
        "ep_degree",
        "nvlink_port_count",
        "matrix_sram_policy",
        "parallel_model",
        "dse_phase_telemetry_seconds",
    )
    lifecycle = {key: record.get(key) for key in keys if key in record}
    lifecycle["trial_record_path"] = (
        f"trial_{int(record.get('trial', -1)):04d}/trial_record.json"
    )
    lifecycle["artifact_record_schema"] = "worker_lifecycle_v1"
    return lifecycle


def persist_trial_record(
    trial_dir: Path,
    record: Mapping[str, Any],
    *,
    artifact_retention: str,
) -> None:
    if artifact_retention == "full":
        write_json(trial_dir / "trial_record.json", dict(record))
        return
    write_json(trial_dir / "trial_detail.json.gz", dict(record))
    write_json(trial_dir / "trial_record.json", compact_trial_record(record))


def canonical_json_sha256(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def selector_trial_summary(
    record: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if record is None:
        return None
    retained_containers = {
        "area_extrapolation_warnings",
        "candidate_fidelity_issues",
        "matrix_sram_equivalent_policies",
        "multi_chip_fidelity",
        "power_excludes",
        "power_warnings",
    }
    summary = {
        key: value
        for key, value in record.items()
        if not isinstance(value, (dict, list, tuple))
        or key in retained_containers
    }
    summary["artifact_record_schema"] = "selector_summary_v1"
    return summary


def run_artifact_bytes(run_dir: Path) -> dict[str, int]:
    totals: Counter[str] = Counter()
    files = [path for path in run_dir.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(run_dir)
        if relative.parts[0].startswith("trial_"):
            category = "trial_dirs"
        elif len(relative.parts) > 1:
            category = relative.parts[0]
        elif path.name.startswith("area_cache.worker_"):
            category = "worker_area_cache"
        elif path.name.startswith("trials.worker_"):
            category = "worker_jsonl"
        elif path.name.startswith("worker_heartbeat_pid_"):
            category = "worker_heartbeat"
        elif re.fullmatch(r"worker_\d+\.log", path.name):
            category = "worker_log"
        else:
            category = path.name
        try:
            totals[category] += path.stat().st_size
        except OSError:
            continue
    totals["total"] = sum(path.stat().st_size for path in files)
    return dict(sorted(totals.items()))


def finalize_compact_artifacts(
    run_dir: Path,
    *,
    retained_trial_ids: set[int],
) -> dict[str, Any]:
    before = run_artifact_bytes(run_dir)
    removed: Counter[str] = Counter()
    for detail in run_dir.glob("trial_*/trial_detail.json.gz"):
        match = re.fullmatch(r"trial_(\d+)", detail.parent.name)
        if match and int(match.group(1)) in retained_trial_ids:
            continue
        removed["trial_detail"] += detail.stat().st_size
        detail.unlink(missing_ok=True)

    cleanup_patterns = (
        "trials.worker_*.jsonl",
        "worker_heartbeat_pid_*.json",
        "worker_*.log",
        "*.lock",
        "area_cache.worker_*.json",
    )
    for pattern in cleanup_patterns:
        for path in run_dir.glob(pattern):
            if not path.is_file():
                continue
            removed[pattern] += path.stat().st_size
            path.unlink(missing_ok=True)
    for lock in run_dir.glob("compiler_report_cache/*.lock"):
        removed["compiler_report_cache_locks"] += lock.stat().st_size
        lock.unlink(missing_ok=True)

    for cache_name in (
        "compiler_trace_cache",
        "compiler_v4_work_cache",
        "compiler_compute_pipeline_cache",
    ):
        cache_dir = run_dir / cache_name
        if not cache_dir.exists():
            continue
        removed[cache_name] += sum(
            path.stat().st_size
            for path in cache_dir.rglob("*")
            if path.is_file()
        )
        shutil.rmtree(cache_dir)

    after = run_artifact_bytes(run_dir)
    return {
        "schema": "compact_artifact_manifest_v1",
        "retained_detail_trial_ids": sorted(retained_trial_ids),
        "bytes_before_cleanup": before,
        "bytes_removed": dict(sorted(removed.items())),
        "bytes_after_cleanup": after,
        "compression_ratio": (
            float(after.get("total", 0)) / float(before.get("total", 1))
        ),
    }
