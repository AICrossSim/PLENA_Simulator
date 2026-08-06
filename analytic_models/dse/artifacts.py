"""Compact, resume-safe artifacts for large Optuna studies."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
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


GLOBAL_DSE_CACHE_SCHEMA = "plena_cross_study_cache_v1"


@dataclass(frozen=True)
class DSECacheDirectories:
    """Versioned content-addressed cache directories shared by DSE runs."""

    root: Path
    compiler_reports: Path
    compiler_traces: Path
    compiler_v4_work: Path
    compiler_compute_pipeline: Path
    compiler_settings: Path
    area_reports: Path

    @classmethod
    def create(cls, root: Path) -> "DSECacheDirectories":
        versioned_root = Path(root) / GLOBAL_DSE_CACHE_SCHEMA
        directories = cls(
            root=versioned_root,
            compiler_reports=versioned_root / "compiler_reports",
            compiler_traces=versioned_root / "compiler_traces",
            compiler_v4_work=versioned_root / "compiler_v4_work",
            compiler_compute_pipeline=(
                versioned_root / "compiler_compute_pipeline"
            ),
            compiler_settings=versioned_root / "compiler_settings",
            area_reports=versioned_root / "area_reports",
        )
        for directory in directories.__dict__.values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        return directories


def cache_entry_path(cache_dir: Path, key: str) -> Path:
    """Return the canonical compressed JSON path for one immutable entry."""

    return Path(cache_dir) / f"{key}.json.gz"


def load_cached_json(path: Path) -> dict[str, Any] | None:
    """Load a cache entry, returning ``None`` for a corrupt partial artifact."""

    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (
        EOFError,
        gzip.BadGzipFile,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def load_json(path: Path) -> dict[str, Any]:
    handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r")
    with handle as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    """Atomically publish a JSON artifact.

    DSE workers share compiler reports and may be terminated while writing.
    Writing the final path directly allowed another process to observe a
    truncated gzip member.  A sibling temporary preserves atomic rename
    semantics on both local filesystems used by the runner.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    handle = (
        gzip.open(temporary, "wt", compresslevel=1)
        if path.suffix == ".gz"
        else temporary.open("w")
    )
    try:
        with handle as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def json_cache_metadata_path(path: Path) -> Path:
    """Return the immutable sidecar path for a shared JSON cache entry."""

    return path.with_name(f"{path.name}.meta.json")


def build_json_cache_metadata(path: Path, data: Any) -> dict[str, Any]:
    """Build hashes once when installing a shared JSON cache entry.

    DSE trials reference the same compiler report many times. Re-reading the
    compressed file and serializing the complete report merely to reproduce
    its hashes made a warm hit scale with report size. The sidecar keeps those
    immutable properties next to the immutable report.
    """

    return {
        "schema": "shared_json_cache_metadata_v1",
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_payload_sha256": canonical_json_sha256(data),
    }


def load_or_create_json_cache_metadata(
    path: Path,
    data: Any,
) -> dict[str, Any]:
    """Load a report sidecar, creating it for pre-sidecar cache entries."""

    metadata_path = json_cache_metadata_path(path)
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if (
            metadata.get("schema") == "shared_json_cache_metadata_v1"
            and metadata.get("path") == path.name
            and int(metadata.get("size_bytes", -1)) == path.stat().st_size
        ):
            return metadata
    metadata = build_json_cache_metadata(path, data)
    write_json(metadata_path, metadata)
    return metadata


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
        "normalized_latency",
        "area_mm2",
        "system_energy_nominal_mj",
        "normalized_energy",
        "objective_normalization",
        "accuracy_score",
        "precision_profile",
        "precision_search_encoding",
        "matrix_datapath_signature",
        "matrix_weight_port_bits",
        "matrix_activation_port_bits",
        "matrix_pe_bit_product",
        "matrix_output_fp_bits",
        "MLEN",
        "BLEN",
        "VLEN",
        "chip_count",
        "physical_chip_count",
        "chip_count_search_value",
        "chips_per_a100_reference",
        "chip_count_scaling",
        "dp_degree",
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


_PHYSICAL_CANDIDATE_FIELDS = (
    "model_config",
    "precision_profile",
    "precision_search_encoding",
    "matrix_datapath_signature",
    "MLEN",
    "VLEN",
    "BLEN",
    "INT_DATA_WIDTH",
    "MATRIX_SRAM_SIZE",
    "matrix_sram_tiles",
    "softmax_row_lanes",
    "chip_count",
    "physical_chip_count",
    "nvlink_port_count",
    "vector_scalar_schedule",
    "softmax_vector_schedule",
    "softmax_state_schedule",
    "pv_accumulation_schedule",
    "packed_qk_schedule",
    "ffn_projection_schedule",
)


def physical_candidate_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return deployable fields while excluding runtime DP/TP/EP mapping."""

    return {
        key: record.get(key)
        for key in _PHYSICAL_CANDIDATE_FIELDS
        if key in record
    }


def build_physical_candidate_bank(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate completed trials by deployable physical/model design.

    Parallel mappings are runtime choices.  Keeping their source trial IDs
    alongside one physical fingerprint lets another batch study enumerate a
    new legal topology without treating the same silicon as a new design.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("state") != "complete":
            continue
        payload = physical_candidate_payload(record)
        fingerprint = canonical_json_sha256(payload)
        entry = grouped.setdefault(
            fingerprint,
            {
                "schema": "physical_candidate_bank_v1",
                "physical_design_fingerprint": fingerprint,
                "physical_design": payload,
                "source_trials": [],
                "source_runtime_topologies": [],
                "best_latency_ms": float("inf"),
                "best_system_energy_nominal_mj": float("inf"),
                "best_accuracy_score": float("-inf"),
            },
        )
        entry["source_trials"].append(int(record.get("trial", -1)))
        topology = {
            key: record.get(key)
            for key in (
                "dp_degree",
                "tp_degree",
                "ep_degree",
                "parallel_model",
            )
            if key in record
        }
        if topology not in entry["source_runtime_topologies"]:
            entry["source_runtime_topologies"].append(topology)
        entry["best_latency_ms"] = min(
            float(entry["best_latency_ms"]),
            float(record.get("latency_ms", float("inf"))),
        )
        entry["best_system_energy_nominal_mj"] = min(
            float(entry["best_system_energy_nominal_mj"]),
            float(
                record.get("system_energy_nominal_mj", float("inf"))
            ),
        )
        entry["best_accuracy_score"] = max(
            float(entry["best_accuracy_score"]),
            float(record.get("accuracy_score", float("-inf"))),
        )
    return [grouped[key] for key in sorted(grouped)]


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
