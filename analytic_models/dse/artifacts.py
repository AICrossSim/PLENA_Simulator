"""Compact, resume-safe artifacts for large Optuna studies."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
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
        "local_tile_counts_by_rank",
        "multi_chip",
        "packed_attention_metadata",
        "power_shadow",
        "vector_scalar_optimization_metadata",
    }
)

_COMPACT_NESTED_VALUE_BYTES = 4096

_COMPACT_TRIAL_KEEP_KEYS = frozenset(
    {
        # Recovery state and objectives.
        "trial",
        "state",
        "reason",
        "latency_ms",
        "normalized_latency",
        "area_mm2",
        "area_uncertainty_p10_mm2",
        "area_uncertainty_p50_mm2",
        "area_uncertainty_p90_mm2",
        "system_energy_nominal_mj",
        "normalized_energy",
        "energy_per_input_token_mj",
        "system_energy_per_input_token_mj",
        "system_average_power_w",
        "onchip_average_power_w",
        "external_hbm_average_power_w",
        "accuracy_score",
        "objective_normalization",
        # Deployable hardware and precision.
        "model_config",
        "precision_profile",
        "precision_search_encoding",
        "matrix_datapath_signature",
        "matrix_weight_port_bits",
        "matrix_activation_port_bits",
        "matrix_pe_bit_product",
        "matrix_output_fp_bits",
        "weight_precision",
        "activation_precision",
        "kv_precision",
        "internal_fp_precision",
        "MLEN",
        "VLEN",
        "BLEN",
        "HLEN",
        "INT_DATA_WIDTH",
        "BROADCAST_AMOUNT",
        "MATRIX_SRAM_SIZE",
        "MATRIX_SRAM_TILES",
        "matrix_sram_policy",
        "matrix_sram_equivalent_policies",
        "matrix_sram_tiles",
        "matrix_sram_depth",
        "matrix_sram_width_bits",
        "matrix_sram_logical_mb",
        "matrix_sram_useful_saturation_tiles",
        "requested_kv_residency_fraction",
        "realized_kv_residency_fraction",
        "resident_kv_blocks",
        "streamed_kv_blocks",
        "kv_reload_factor",
        "kv_tile_load_count",
        "attention_kv_resident",
        "softmax_row_lanes",
        # Compiler/model profile needed to reproduce the point.
        "vector_scalar_schedule",
        "softmax_vector_schedule",
        "softmax_state_schedule",
        "pv_accumulation_schedule",
        "packed_qk_schedule",
        "ffn_projection_schedule",
        "ffn_address_schedule",
        "address_generation_mode",
        "selector_schedule",
        "reduction_output_mode",
        "moe_lowering_schedule",
        "compute_timing_mode",
        "latency_model",
        "latency_source",
        "compiler_trace_granularity",
        "trial_report_materialization",
        # Multi-chip runtime mapping, budgets, and communication.
        "chip_count",
        "physical_chip_count",
        "chip_count_search_value",
        "chips_per_a100_reference",
        "chip_count_scaling",
        "reference_a100_count",
        "decode_chip_count",
        "parallel_model",
        "multi_chip_model",
        "dp_degree",
        "tp_degree",
        "cp_degree",
        "ep_degree",
        "nvlink_port_count",
        "endpoint_area_per_chip_mm2",
        "total_endpoint_area_mm2",
        "total_silicon_area_mm2",
        "per_chip_physical_area_mm2",
        "fixed_batch_requests_per_second",
        "fixed_batch_tokens_per_second",
        "tp_collective_latency_ns",
        "cp_kv_ring_latency_ns",
        "ep_dispatch_latency_ns",
        "ep_return_latency_ns",
        "communication_latency_ns",
        "interconnect_dynamic_energy_mj",
        "weight_replication_factor",
        "shared_weight_replication",
        "expert_weight_replication",
        "batch_packing_utilization",
        "slowest_rank",
        # Stage and component summaries used by result reports.
        "compiler_compute_latency_ms",
        "compiler_memory_latency_ms",
        "compiler_stage_compute_latency_ns",
        "compiler_stage_roofline_latency_ns",
        "per_chip_stage_compute_latency_ns",
        "per_chip_stage_memory_latency_ns",
        "compiler_stage_bound",
        "matrix_compute_cycles",
        "vector_compute_cycles",
        "scalar_compute_cycles",
        "control_compute_cycles",
        "ideal_compute_cycles",
        "physical_hbm_read_bytes",
        "physical_hbm_write_bytes",
        "physical_to_payload_traffic_ratio",
        "achieved_average_bandwidth_gbps",
        "bandwidth_utilization",
        "fp16_kv_handoff_bytes",
        "fp16_kv_handoff_latency_ms",
        "prefill_latency_excluding_kv_handoff_ms",
        "prefill_plus_kv_handoff_serial_shadow_ms",
        # Area/power semantics and bounded warnings.
        "selected_sram_area_mm2",
        "ideal_dual_port_sram_area_mm2",
        "replicated_single_port_sram_area_mm2",
        "dual_port_area_savings_mm2",
        "dual_port_area_savings_pct",
        "sram_port_model",
        "sram_port_energy_model",
        "clock_gating_mode",
        "clock_gating_status",
        "power_model",
        "power_scope",
        "power_excludes",
        "power_warnings",
        "power_calibration_status",
        "power_uncertainty",
        "area_model",
        "area_mode",
        "area_extrapolation_warnings",
        "vector_scalar_area_calibration_status",
        "candidate_fidelity",
        "candidate_fidelity_issues",
        "multi_chip_fidelity",
        "compute_fidelity_status",
        "calibration_in_domain",
        "broadcast_rtl_validation_status",
        # Immutable evidence and performance telemetry references.
        "compiler_cost_cache_key",
        "compiler_cost_cache_tier",
        "compiler_cost_shared_report",
        "compiler_cost_shared_report_sha256",
        "compiler_matrix_timing_artifact_hash",
        "compiler_memory_calibration_id",
        "area_cache_key",
        "area_cache_tier",
        "dse_phase_telemetry_seconds",
        "compiler_phase_telemetry_seconds",
    }
)


def _compact_summary_value(value: Any) -> bool:
    """Keep scalar metadata and bounded summaries, not replayable trace data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if not isinstance(value, (Mapping, list, tuple)):
        return False
    try:
        return len(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
        ) <= _COMPACT_NESTED_VALUE_BYTES
    except (TypeError, ValueError):
        return False


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
        if key in _COMPACT_TRIAL_KEEP_KEYS
        and key not in _COMPACT_TRIAL_DROP_KEYS
        and _compact_summary_value(value)
    }
    omitted_field_count = len(record) - len(compact)
    compact["artifact_record_schema"] = "compact_trial_v2"
    compact["detail_artifact"] = None
    compact["detail_artifact_status"] = "not_materialized_compact"
    compact["omitted_field_count"] = omitted_field_count
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
        "weight_precision",
        "activation_precision",
        "kv_precision",
        "internal_fp_precision",
        "MLEN",
        "BLEN",
        "VLEN",
        "INT_DATA_WIDTH",
        "MATRIX_SRAM_TILES",
        "matrix_sram_tiles",
        "softmax_row_lanes",
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


def materialize_sqlite_database(run_dir: Path) -> Path:
    """Restore a finalized compact study before an explicit resume."""

    database = Path(run_dir) / "study.sqlite3"
    compressed = database.with_suffix(".sqlite3.gz")
    if database.exists() or not compressed.exists():
        return database
    temporary = database.with_name(f".{database.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        os.replace(temporary, database)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return database


def _compress_file(path: Path, *, remove_source: bool = True) -> Path:
    target = path.with_name(f"{path.name}.gz")
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with path.open("rb") as source, gzip.open(
            temporary, "wb", compresslevel=1
        ) as destination:
            shutil.copyfileobj(source, destination)
        os.replace(temporary, target)
        if remove_source:
            path.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _finalize_sqlite_database(run_dir: Path) -> Path | None:
    database = Path(run_dir) / "study.sqlite3"
    if not database.exists():
        return None
    connection = sqlite3.connect(database, timeout=120)
    try:
        connection.execute("PRAGMA busy_timeout=120000")
        checkpoint = connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        if checkpoint and int(checkpoint[0]) != 0:
            raise RuntimeError(f"SQLite WAL checkpoint remained busy: {checkpoint}")
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    database.with_name(f"{database.name}-wal").unlink(missing_ok=True)
    database.with_name(f"{database.name}-shm").unlink(missing_ok=True)
    return _compress_file(database)


def persist_trial_record(
    trial_dir: Path,
    record: Mapping[str, Any],
    *,
    artifact_retention: str,
) -> None:
    if artifact_retention == "full":
        write_json(trial_dir / "trial_record.json", dict(record))
        return
    # Compact studies can contain tens of thousands of attempts.  Writing a
    # complete per-trial detail and deleting it only at finalization creates a
    # multi-GiB peak and duplicates immutable compiler reports.  Publish the
    # resume/analysis summary compressed from the outset; detailed compiler
    # evidence remains content-addressed in the shared report cache.
    write_json(
        trial_dir / "trial_record.json.gz",
        compact_trial_record(record),
    )
    (trial_dir / "trial_record.json").unlink(missing_ok=True)
    (trial_dir / "trial_detail.json.gz").unlink(missing_ok=True)


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

    for record_path in run_dir.glob("trial_*/trial_record.json"):
        before_size = record_path.stat().st_size
        _compress_file(record_path)
        removed["trial_record_json_to_gzip"] += before_size

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

    all_trials_csv = run_dir / "all_trials.csv"
    if all_trials_csv.exists():
        before_size = all_trials_csv.stat().st_size
        _compress_file(all_trials_csv)
        removed["all_trials_csv_to_gzip"] += before_size

    trials_jsonl = run_dir / "trials.jsonl"
    if trials_jsonl.exists():
        before_size = trials_jsonl.stat().st_size
        _compress_file(trials_jsonl)
        removed["trials_jsonl_to_gzip"] += before_size

    worker_resources = run_dir / "worker_resources.jsonl"
    if worker_resources.exists():
        removed["worker_resources"] += worker_resources.stat().st_size
        worker_resources.unlink(missing_ok=True)

    database = run_dir / "study.sqlite3"
    if database.exists():
        before_size = sum(
            path.stat().st_size
            for path in (
                database,
                database.with_name(f"{database.name}-wal"),
                database.with_name(f"{database.name}-shm"),
            )
            if path.exists()
        )
        _finalize_sqlite_database(run_dir)
        removed["sqlite_wal_checkpoint_and_gzip"] += before_size

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
