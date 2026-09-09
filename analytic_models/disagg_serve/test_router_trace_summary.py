"""Tests for measured-route aggregate materialization."""

from __future__ import annotations

from dataclasses import replace

import pytest

from .expert_placement import (
    EvidenceReceipt,
    RouteRecord,
    RoutingStep,
    RoutingTrace,
    trace_content_hash,
)
from .router_trace_summary import audit_trace_summary, summarize_trace


def _receipt() -> EvidenceReceipt:
    return EvidenceReceipt(
        artifact_path="/trace/evidence.json",
        artifact_sha256="1" * 64,
        subject_sha256="2" * 64,
        command=("collect",),
        tool_revision="3" * 64,
        recorded_at_utc="2026-08-20T00:00:00Z",
        sample_count=5,
    )


def _trace(step_count: int = 5) -> RoutingTrace:
    steps = []
    for step_index in range(step_count):
        records = []
        for layer in range(48):
            if layer == 1 and step_index % 2:
                experts = tuple(range(8, 16))
            elif layer == 2 and step_index % 2:
                experts = (*range(7), 8)
            else:
                experts = tuple(range(8))
            records.append(
                RouteRecord(
                    token_id=f"token-{step_index}",
                    layer=layer,
                    source_chip=step_index % 4,
                    expert_ids=experts,
                )
            )
        steps.append(RoutingStep(step_index=step_index, records=tuple(records)))
    return RoutingTrace(
        source_kind="measured",
        steps=tuple(steps),
        receipt=_receipt(),
    )


def _source(trace: RoutingTrace) -> dict:
    return {
        "collector_verified": True,
        "router_index_path": "/trace/index.json",
        "router_index_sha256": "4" * 64,
        "router_index_content_hash": "5" * 64,
        "placement_input_path": "/trace/input.json",
        "placement_input_sha256": "6" * 64,
        "router_trace_evidence_path": "/trace/evidence.json",
        "router_trace_evidence_sha256": "7" * 64,
        "trace_content_hash": trace_content_hash(trace),
    }


def test_summary_conserves_routes_and_exposes_only_supported_maxima():
    trace = _trace()
    summary = summarize_trace(trace, [1, 2, 8], source_binding=_source(trace))

    assert summary["materializable_batches"] == [1, 2]
    assert summary["unsupported_batches"] == [8]
    batch_two = next(row for row in summary["batches"] if row["batch_size"] == 2)
    assert batch_two["complete_window_count"] == 2
    assert batch_two["dropped_tail_step_count"] == 1
    assert batch_two["observation_count"] == 96
    assert all(row["assignments_conserved"] for row in batch_two["observations"])
    assert all(row["route_assignments"] == 16 for row in batch_two["observations"])
    assert batch_two["supported_override"] == {
        "moe_unique_experts_per_step": 16,
        "moe_routing_imbalance_factor": 1.125,
        "aggregation_policy": (
            "consecutive_nonoverlapping_windows_conservative_observed_max/v1"
        ),
        "selection_rule": (
            "independent maxima over all complete windows and 48 layers"
        ),
    }
    assert summary["classification"]["publication_rankable"] is False
    assert summary["classification"]["hardware_rankable"] is False
    assert summary["classification"]["selection_eligible"] is False
    assert audit_trace_summary(summary, trace) == summary


def test_summary_detects_tampering_and_requires_measured_receipt():
    trace = _trace(2)
    summary = summarize_trace(trace, [2], source_binding=_source(trace))
    tampered = dict(summary)
    tampered["trace_step_count"] = 3
    with pytest.raises(ValueError, match="content hash mismatch"):
        audit_trace_summary(tampered, trace)

    with pytest.raises(ValueError, match="measured trace with a receipt"):
        summarize_trace(
            replace(trace, source_kind="synthetic", receipt=None),
            [2],
            source_binding=_source(trace),
        )


def test_summary_rejects_incomplete_layer_coverage():
    trace = _trace(2)
    broken_step = replace(trace.steps[0], records=trace.steps[0].records[:-1])
    broken = replace(trace, steps=(broken_step, *trace.steps[1:]))
    source = _source(broken)
    with pytest.raises(ValueError, match="exactly 48 records"):
        summarize_trace(broken, [1], source_binding=source)
