"""Aggregate a verified Qwen3-MoE route trace for analytic repricing.

The decode timing model accepts only a layer-invariant unique-expert count and
one scalar cycle-imbalance factor.  This module preserves the richer measured
distribution as audit data, but exposes only those two supported aggregates.
It never changes the all-expert resident-weight capacity ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    from .expert_placement import (
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
        RoutingTrace,
        trace_content_hash,
    )
except ImportError:  # script-style imports used by performance/disagg_decode.py
    from expert_placement import (  # type: ignore[no-redef]
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
        RoutingTrace,
        trace_content_hash,
    )


SUMMARY_SCHEMA = "plena-qwen3-moe-router-trace-summary/v1"
MODEL_OVERLAY_SCHEMA = "plena-qwen3-moe-route-model-overlay/v1"
AGGREGATION_POLICY = (
    "consecutive_nonoverlapping_windows_conservative_observed_max/v1"
)
IMBALANCE_DEFINITION = (
    "max_assignments_to_one_active_expert_divided_by_"
    "mean_assignments_per_active_expert"
)
SUPPORTED_OVERRIDE_FIELDS = (
    "moe_unique_experts_per_step",
    "moe_routing_imbalance_factor",
)
UNSUPPORTED_TIMING_INPUTS = (
    "per-window/per-layer expert identities",
    "per-window/per-layer expert-token histograms",
    "source-chip and placement decisions",
    "expert-cache hits, misses, and reuse",
    "BLEN-specific timing derived from the measured histogram",
    "collective topology, contention, packetization, and overlap",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_hash(value: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 used by route-summary artifacts."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_hash", None)
    return body | {"content_hash": canonical_hash(body)}


def _positive_batches(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(batch_sizes, (str, bytes)):
        raise ValueError("batch_sizes must be an integer sequence")
    values = tuple(batch_sizes)
    if not values:
        raise ValueError("at least one batch size is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("batch sizes must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("batch sizes must not contain duplicates")
    return tuple(sorted(values))


def _audit_collector_trace(trace: RoutingTrace) -> tuple[tuple[Any, ...], ...]:
    """Require the exact one-token/all-layer trace emitted by the collector."""

    if trace.source_kind != "measured" or trace.receipt is None:
        raise ValueError("route repricing requires a measured trace with a receipt")
    if trace.model_id != MODEL_ID or trace.model_revision != MODEL_REVISION:
        raise ValueError("routing trace differs from the sealed model identity")
    if tuple(step.step_index for step in trace.steps) != tuple(range(len(trace.steps))):
        raise ValueError("routing steps must be contiguous and zero-based")

    ordered: list[tuple[Any, ...]] = []
    expected_layers = set(range(NUM_LAYERS))
    for step in trace.steps:
        if len(step.records) != NUM_LAYERS:
            raise ValueError("each routing step must contain exactly 48 records")
        if {record.layer for record in step.records} != expected_layers:
            raise ValueError("each routing step must cover every layer exactly once")
        token_ids = {record.token_id for record in step.records}
        source_chips = {record.source_chip for record in step.records}
        if len(token_ids) != 1 or len(source_chips) != 1:
            raise ValueError("one routing step must describe one token and source chip")
        by_layer = tuple(sorted(step.records, key=lambda record: record.layer))
        if sum(len(record.expert_ids) for record in by_layer) != NUM_LAYERS * TOP_K:
            raise ValueError("routing assignments do not conserve at one decode step")
        ordered.append(by_layer)
    return tuple(ordered)


def _mean(values: Sequence[int | float]) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def _nearest_rank(values: Sequence[int | float], numerator: int, denominator: int):
    """Return the deterministic nearest-rank quantile without interpolation."""

    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * numerator / denominator))
    return ordered[rank - 1]


def _observation(
    *,
    records: Sequence[Any],
    batch_size: int,
    window_index: int,
    first_step_index: int,
    last_step_index: int,
    layer: int,
) -> tuple[dict[str, Any], Fraction]:
    counts: Counter[int] = Counter()
    for record in records:
        if record.layer != layer:
            raise AssertionError("layer grouping changed during route aggregation")
        counts.update(record.expert_ids)
    assignments = batch_size * TOP_K
    if sum(counts.values()) != assignments:
        raise AssertionError("route aggregation dropped or duplicated assignments")
    unique = len(counts)
    if not TOP_K <= unique <= min(NUM_EXPERTS, assignments):
        raise AssertionError("aggregated unique-expert count is outside its bounds")
    maximum = max(counts.values())
    minimum = min(counts.values())
    imbalance = Fraction(maximum * unique, assignments)
    if imbalance < 1:
        raise AssertionError("max-to-mean active-expert load must be at least one")
    histogram = Counter(counts.values())
    row = {
        "window_index": window_index,
        "first_step_index": first_step_index,
        "last_step_index": last_step_index,
        "layer": layer,
        "route_assignments": assignments,
        "unique_experts": unique,
        "max_assignments_to_one_expert": maximum,
        "min_assignments_to_one_active_expert": minimum,
        "active_expert_token_count_histogram": {
            str(token_count): expert_count
            for token_count, expert_count in sorted(histogram.items())
        },
        "assignment_imbalance": {
            "definition": IMBALANCE_DEFINITION,
            "numerator": imbalance.numerator,
            "denominator": imbalance.denominator,
            "factor": float(imbalance),
        },
        "assignments_conserved": True,
        "timing_input": False,
    }
    return row, imbalance


def _layer_aggregate(
    layer: int,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique = [int(row["unique_experts"]) for row in observations]
    imbalance = [float(row["assignment_imbalance"]["factor"]) for row in observations]
    return {
        "layer": layer,
        "observation_count": len(observations),
        "unique_experts": {
            "minimum": min(unique),
            "mean": _mean(unique),
            "nearest_rank_p95": int(_nearest_rank(unique, 95, 100)),
            "maximum": max(unique),
        },
        "assignment_imbalance": {
            "definition": IMBALANCE_DEFINITION,
            "minimum": min(imbalance),
            "mean": _mean(imbalance),
            "nearest_rank_p95": float(_nearest_rank(imbalance, 95, 100)),
            "maximum": max(imbalance),
        },
    }


def _batch_summary(
    ordered_steps: Sequence[Sequence[Any]],
    batch_size: int,
) -> dict[str, Any]:
    step_count = len(ordered_steps)
    complete_windows, tail = divmod(step_count, batch_size)
    if complete_windows == 0:
        return {
            "batch_size": batch_size,
            "status": "unsupported",
            "complete_window_count": 0,
            "used_step_count": 0,
            "dropped_tail_step_count": step_count,
            "observation_count": 0,
            "observations": [],
            "layer_aggregates": [],
            "supported_override": None,
            "blockers": ["insufficient_trace_steps_for_one_complete_batch"],
        }

    rows: list[dict[str, Any]] = []
    ratios: list[Fraction] = []
    for window_index in range(complete_windows):
        begin = window_index * batch_size
        end = begin + batch_size
        window = ordered_steps[begin:end]
        for layer in range(NUM_LAYERS):
            row, ratio = _observation(
                records=[step[layer] for step in window],
                batch_size=batch_size,
                window_index=window_index,
                first_step_index=begin,
                last_step_index=end - 1,
                layer=layer,
            )
            rows.append(row)
            ratios.append(ratio)

    expected_observations = complete_windows * NUM_LAYERS
    if len(rows) != expected_observations:
        raise AssertionError("route summary observation count does not conserve")
    unique_values = [int(row["unique_experts"]) for row in rows]
    max_ratio = max(ratios)
    layer_aggregates = [
        _layer_aggregate(
            layer,
            [row for row in rows if int(row["layer"]) == layer],
        )
        for layer in range(NUM_LAYERS)
    ]
    return {
        "batch_size": batch_size,
        "status": "materializable",
        "complete_window_count": complete_windows,
        "used_step_count": complete_windows * batch_size,
        "dropped_tail_step_count": tail,
        "observation_count": expected_observations,
        "observations": rows,
        "layer_aggregates": layer_aggregates,
        "aggregate_statistics": {
            "unique_experts": {
                "minimum": min(unique_values),
                "mean": _mean(unique_values),
                "nearest_rank_p95": int(_nearest_rank(unique_values, 95, 100)),
                "maximum": max(unique_values),
            },
            "assignment_imbalance": {
                "definition": IMBALANCE_DEFINITION,
                "minimum": float(min(ratios)),
                "mean": _mean([float(value) for value in ratios]),
                "nearest_rank_p95": float(
                    _nearest_rank([float(value) for value in ratios], 95, 100)
                ),
                "maximum": float(max_ratio),
                "maximum_exact": {
                    "numerator": max_ratio.numerator,
                    "denominator": max_ratio.denominator,
                },
            },
        },
        "supported_override": {
            "moe_unique_experts_per_step": max(unique_values),
            "moe_routing_imbalance_factor": float(max_ratio),
            "aggregation_policy": AGGREGATION_POLICY,
            "selection_rule": (
                "independent maxima over all complete windows and 48 layers"
            ),
        },
        "blockers": [],
    }


def summarize_trace(
    trace: RoutingTrace,
    batch_sizes: Sequence[int],
    *,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a content-hashed aggregate summary from a verified route trace."""

    batches = _positive_batches(batch_sizes)
    if not isinstance(source_binding, Mapping) or not source_binding:
        raise ValueError("source_binding must identify the verified trace artifacts")
    if source_binding.get("collector_verified") is not True:
        raise ValueError("source_binding must record successful collector verification")
    for field in (
        "router_index_path",
        "placement_input_path",
        "router_trace_evidence_path",
    ):
        if not isinstance(source_binding.get(field), str) or not source_binding[field]:
            raise ValueError(f"source_binding.{field} must be a non-empty path")
    for field in (
        "router_index_sha256",
        "router_index_content_hash",
        "placement_input_sha256",
        "router_trace_evidence_sha256",
    ):
        if not _SHA256.fullmatch(str(source_binding.get(field, ""))):
            raise ValueError(f"source_binding.{field} must be a SHA-256")

    ordered = _audit_collector_trace(trace)
    trace_hash = trace_content_hash(trace)
    if source_binding.get("trace_content_hash") != trace_hash:
        raise ValueError("source binding and parsed trace content hashes differ")
    batch_rows = [_batch_summary(ordered, batch) for batch in batches]
    materializable = [
        int(row["batch_size"])
        for row in batch_rows
        if row["status"] == "materializable"
    ]
    unsupported = [
        int(row["batch_size"])
        for row in batch_rows
        if row["status"] != "materializable"
    ]
    body = {
        "schema": SUMMARY_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": trace_hash,
        "trace_step_count": len(ordered),
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "source_binding": dict(source_binding),
        "windowing": {
            "policy": "consecutive_nonoverlapping_global_trace_steps",
            "step_order": "collector_global_cached_decode_step_ordinal",
            "tail_policy": "drop_and_report_incomplete_final_window",
            "larger_batch_interpretation": (
                "post_hoc_grouping_proxy_not_a_measured_batched_forward"
            ),
        },
        "imbalance_definition": IMBALANCE_DEFINITION,
        "aggregation_policy": AGGREGATION_POLICY,
        "supported_override_fields": list(SUPPORTED_OVERRIDE_FIELDS),
        "unsupported_timing_inputs": list(UNSUPPORTED_TIMING_INPUTS),
        "batches": batch_rows,
        "materializable_batches": materializable,
        "unsupported_batches": unsupported,
        "conservation": {
            "assignments_per_trace_step": NUM_LAYERS * TOP_K,
            "total_trace_assignments": len(ordered) * NUM_LAYERS * TOP_K,
            "all_observations_conserved": True,
        },
        "classification": {
            "evidence": "measured_trace_aggregate_analytic_sensitivity",
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "blockers": [
                "batch_gt_1_uses_post_hoc_consecutive_trace_grouping",
                "balanced_anonymous_timing_does_not_ingest_measured_histograms",
                "routing_imbalance_is_a_scalar_cycle_sensitivity",
                "route_summary_is_not_compiler_or_emulator_timing_evidence",
                "moe_area_and_dynamic_power_require_independent_calibration",
            ],
        },
    }
    return _hashed_body(body)


def audit_trace_summary(
    summary: Mapping[str, Any],
    trace: RoutingTrace,
) -> dict[str, Any]:
    """Recompute a summary from its bound trace and reject any difference."""

    if not isinstance(summary, Mapping):
        raise ValueError("route summary must be an object")
    body = dict(summary)
    observed_hash = body.pop("content_hash", None)
    if observed_hash != canonical_hash(body):
        raise ValueError("route summary content hash mismatch")
    if body.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("unsupported route summary schema")
    batch_rows = body.get("batches")
    if not isinstance(batch_rows, list) or not batch_rows:
        raise ValueError("route summary has no batch rows")
    source = body.get("source_binding")
    if not isinstance(source, Mapping):
        raise ValueError("route summary source binding is missing")
    expected = summarize_trace(
        trace,
        [int(row["batch_size"]) for row in batch_rows],
        source_binding=source,
    )
    if dict(summary) != expected:
        raise ValueError("route summary differs from recomputation over its trace")
    return expected


def validate_model_overlay(
    model: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Validate the optional two-field route overlay in a model JSON."""

    overlay = model.get("moe_route_repricing")
    if overlay is None:
        return None
    if not isinstance(overlay, Mapping):
        raise ValueError("moe_route_repricing must be an object")
    model_body = dict(model)
    model_content_hash = model_body.pop("content_hash", None)
    if model_content_hash is not None and model_content_hash != canonical_hash(
        model_body
    ):
        raise ValueError("MoE route model content hash mismatch")
    if overlay.get("schema") != MODEL_OVERLAY_SCHEMA:
        raise ValueError("unsupported MoE route model-overlay schema")
    for field in (
        "trace_content_hash",
        "summary_content_hash",
        "summary_sha256",
    ):
        if not _SHA256.fullmatch(str(overlay.get(field, ""))):
            raise ValueError(f"MoE route model overlay has an invalid {field}")
    batch = overlay.get("batch_size")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("MoE route model overlay batch_size must be positive")
    injected = overlay.get("injected_fields")
    if not isinstance(injected, Mapping) or set(injected) != set(
        SUPPORTED_OVERRIDE_FIELDS
    ):
        raise ValueError("MoE route model overlay injected fields differ")
    if model.get("moe_unique_experts_per_step") != injected.get(
        "moe_unique_experts_per_step"
    ):
        raise ValueError("MoE route unique-expert override is not bound")
    if model.get("moe_routing_imbalance_factor") != injected.get(
        "moe_routing_imbalance_factor"
    ):
        raise ValueError("MoE route imbalance override is not bound")
    classification = overlay.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
        )
    ):
        raise ValueError("MoE route model overlay must remain fail-closed")
    if overlay.get("aggregation_policy") != AGGREGATION_POLICY:
        raise ValueError("MoE route model overlay aggregation policy differs")
    unsupported = overlay.get("unsupported_timing_inputs")
    if unsupported != list(UNSUPPORTED_TIMING_INPUTS):
        raise ValueError("MoE route model overlay scope limits differ")
    resident = overlay.get("resident_expert_storage")
    if resident != {
        "num_experts": NUM_EXPERTS,
        "policy": "all_experts_remain_resident",
        "changed_by_overlay": False,
    }:
        raise ValueError("MoE route model overlay resident-storage contract differs")

    summary_path = Path(str(overlay.get("summary_path", "")))
    if not summary_path.is_absolute() or not summary_path.is_file():
        raise ValueError("MoE route model overlay summary is missing")
    summary_payload = summary_path.read_bytes()
    if hashlib.sha256(summary_payload).hexdigest() != overlay.get("summary_sha256"):
        raise ValueError("MoE route model overlay summary file hash mismatch")
    summary = json.loads(summary_payload)
    if not isinstance(summary, dict):
        raise ValueError("MoE route model overlay summary must be an object")
    summary_body = dict(summary)
    summary_content_hash = summary_body.pop("content_hash", None)
    if (
        summary_content_hash != canonical_hash(summary_body)
        or summary_content_hash != overlay.get("summary_content_hash")
        or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("trace_content_hash") != overlay.get("trace_content_hash")
    ):
        raise ValueError("MoE route model overlay summary content binding differs")
    source = summary.get("source_binding")
    if not isinstance(source, Mapping) or source.get("collector_verified") is not True:
        raise ValueError("MoE route model overlay summary lacks verified provenance")
    for path_field, hash_field in (
        ("router_index_path", "router_index_sha256"),
        ("placement_input_path", "placement_input_sha256"),
        ("router_trace_evidence_path", "router_trace_evidence_sha256"),
    ):
        source_path = Path(str(source.get(path_field, "")))
        if not source_path.is_absolute() or not source_path.is_file():
            raise ValueError(f"MoE route model overlay source {path_field} is missing")
        if _file_hash(source_path) != source.get(hash_field):
            raise ValueError(f"MoE route model overlay source {path_field} changed")
    matching = [
        row
        for row in summary.get("batches", [])
        if isinstance(row, Mapping) and row.get("batch_size") == batch
    ]
    if (
        len(matching) != 1
        or matching[0].get("status") != "materializable"
        or not isinstance(matching[0].get("supported_override"), Mapping)
        or {
            field: matching[0]["supported_override"].get(field)
            for field in SUPPORTED_OVERRIDE_FIELDS
        }
        != dict(injected)
    ):
        raise ValueError("MoE route model overlay batch binding differs")
    return overlay


__all__ = [
    "AGGREGATION_POLICY",
    "IMBALANCE_DEFINITION",
    "MODEL_OVERLAY_SCHEMA",
    "SUMMARY_SCHEMA",
    "SUPPORTED_OVERRIDE_FIELDS",
    "UNSUPPORTED_TIMING_INPUTS",
    "audit_trace_summary",
    "canonical_hash",
    "summarize_trace",
    "validate_model_overlay",
]
