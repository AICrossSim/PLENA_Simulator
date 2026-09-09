from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from .expert_placement import (
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    EvidenceReceipt,
    PlacementConfig,
    RouteRecord,
    RoutingStep,
    RoutingTrace,
    analyze_expert_placement,
    hot_placement_subject,
    link_calibration_subject,
    main,
    model_bytes_subject,
    trace_content_hash,
)


def record(
    token_id: str,
    *,
    layer: int = 0,
    source_chip: int = 0,
    experts: tuple[int, ...] = tuple(range(8)),
) -> RouteRecord:
    return RouteRecord(
        token_id=token_id,
        layer=layer,
        source_chip=source_chip,
        expert_ids=experts,
    )


def trace(*steps: RoutingStep, source_kind: str = "synthetic") -> RoutingTrace:
    return RoutingTrace(source_kind=source_kind, steps=steps)


def config(policy: str, **overrides: object) -> PlacementConfig:
    values: dict[str, object] = {
        "policy": policy,
        "chip_count": 4,
        "expert_weight_bytes": 100,
        "activation_bytes_per_assignment": 10,
        "link_bandwidth_bytes_per_s_per_chip": 100.0,
        "link_energy_j_per_byte": 0.5,
    }
    values.update(overrides)
    return PlacementConfig(**values)


def receipt(
    path: Path, subject_sha256: str, *, sample_count: int = 3
) -> EvidenceReceipt:
    return EvidenceReceipt(
        artifact_path=path.name,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        subject_sha256=subject_sha256,
        command=("measure", "--sealed"),
        tool_revision="a" * 40,
        recorded_at_utc="2026-08-20T10:00:00+00:00",
        sample_count=sample_count,
    )


def test_qwen_contract_rejects_wrong_top_k_duplicates_and_ranges() -> None:
    with pytest.raises(ValueError, match="top-8"):
        record("t0", experts=tuple(range(7)))
    with pytest.raises(ValueError, match="unique"):
        record("t0", experts=(0, 1, 2, 3, 4, 5, 6, 6))
    with pytest.raises(ValueError, match="expert_id"):
        record("t0", experts=(0, 1, 2, 3, 4, 5, 6, 128))
    with pytest.raises(ValueError, match="layer"):
        record("t0", layer=48)
    with pytest.raises(ValueError, match="layer"):
        record("t0", layer=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expert_id"):
        record("t0", experts=(0, 1, 2, 3, 4, 5, 6, 7.0))  # type: ignore[arg-type]


def test_sharded_dispatch_conserves_assignments_and_accounts_unicast() -> None:
    routing = trace(RoutingStep(0, (record("t0"),)))
    result = analyze_expert_placement(routing, config("sharded_dispatch"))

    assert result["routing"]["assignment_conservation"] == {
        "expected": 8,
        "observed": 8,
        "per_chip_sum": 8,
        "conserved": True,
    }
    assert [row["assignments"] for row in result["load"]["per_chip"]] == [
        2,
        2,
        2,
        2,
    ]
    assert result["routing"]["remote_assignments"] == 6
    assert result["traffic"]["dispatch_activation_bytes"] == 60
    assert result["traffic"]["combine_activation_bytes"] == 60
    assert result["traffic"]["logical_unicast_bytes"] == 120
    assert result["link"]["time_s"] == pytest.approx(0.6)
    assert result["link"]["energy_j"] == pytest.approx(60.0)
    assert result["link"]["all_to_all_collective_modelled"] is False
    assert result["resident_weights"]["final_resident_bytes"] == 48 * 128 * 100
    assert result["provenance"]["publication_rankable"] is False
    assert result["provenance"]["evidence_grade"] == "projected"


def test_ordered_lru_reuses_remote_experts_without_losing_assignments() -> None:
    experts = (0, 2, 4, 6, 8, 10, 12, 14)
    routing = trace(
        RoutingStep(0, (record("t0", source_chip=1, experts=experts),)),
        RoutingStep(1, (record("t1", source_chip=1, experts=experts),)),
    )
    result = analyze_expert_placement(
        routing,
        config(
            "local_lru_weight_cache",
            chip_count=2,
            cache_capacity_bytes_per_chip=800,
        ),
    )

    assert result["routing"]["assignment_conservation"]["expected"] == 16
    assert result["routing"]["assignment_conservation"]["conserved"] is True
    assert result["cache"]["lookups"] == 16
    assert result["cache"]["misses"] == 8
    assert result["cache"]["hits"] == 8
    assert result["cache"]["hit_fraction"] == pytest.approx(0.5)
    assert result["cache"]["evictions"] == 0
    assert result["traffic"]["weight_fetch_bytes"] == 800
    assert result["routing"]["remote_route_fraction"] == pytest.approx(0.5)
    assert result["load"]["per_chip"][1]["assignments"] == 16


def test_lru_capacity_reports_deterministic_eviction_and_misses() -> None:
    experts = (0, 2, 4, 6, 8, 10, 12, 14)
    routing = trace(
        RoutingStep(0, (record("t0", source_chip=1, experts=experts),)),
        RoutingStep(1, (record("t1", source_chip=1, experts=experts),)),
    )
    result = analyze_expert_placement(
        routing,
        config(
            "local_lru_weight_cache",
            chip_count=2,
            cache_capacity_bytes_per_chip=200,
        ),
    )

    assert result["cache"]["hits"] == 0
    assert result["cache"]["misses"] == 16
    assert result["cache"]["evictions"] == 14
    assert result["cache"]["per_chip"][1]["final_cached_expert_instances"] == 2
    assert result["traffic"]["weight_fetch_bytes"] == 1600


def test_hot_replication_reduces_remote_routes_but_derived_list_is_unrankable() -> None:
    experts = (0, 2, 4, 6, 8, 10, 12, 14)
    routing = trace(
        RoutingStep(0, (record("t0", source_chip=1, experts=experts),))
    )
    sharded = analyze_expert_placement(
        routing, config("sharded_dispatch", chip_count=2)
    )
    replicated = analyze_expert_placement(
        routing,
        config(
            "hot_expert_replication",
            chip_count=2,
            hot_expert_count=8,
            hot_replica_count=2,
        ),
    )

    assert sharded["routing"]["remote_assignments"] == 8
    assert replicated["routing"]["remote_assignments"] == 0
    assert replicated["routing"]["assignment_conservation"]["conserved"] is True
    assert replicated["resident_weights"]["final_resident_bytes"] == (
        48 * 128 + 8
    ) * 100
    assert replicated["provenance"]["publication_rankable"] is False
    assert (
        "hot_experts_derived_from_evaluation_trace"
        in replicated["provenance"]["unrankable_reasons"]
    )


def test_complete_measured_trace_remains_projected_without_design_bound_receipts(
    tmp_path: Path,
) -> None:
    routing = RoutingTrace(
        source_kind="measured",
        steps=tuple(
            [
                RoutingStep(
                    0,
                    tuple(record("t0", layer=layer) for layer in range(48)),
                )
            ]
        ),
    )
    base_config = config("sharded_dispatch")
    trace_artifact = tmp_path / "routes.bin"
    model_artifact = tmp_path / "model-bytes.json"
    link_artifact = tmp_path / "link-calibration.csv"
    trace_artifact.write_bytes(b"measured routes")
    model_artifact.write_bytes(b"measured byte ledger")
    link_artifact.write_bytes(b"measured link sweep")

    routing = replace(
        routing,
        receipt=receipt(
            trace_artifact, trace_content_hash(routing), sample_count=1
        ),
    )
    sealed_config = replace(
        base_config,
        model_bytes_receipt=receipt(
            model_artifact,
            hashlib.sha256(
                json.dumps(
                    model_bytes_subject(base_config),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        ),
        link_calibration_receipt=receipt(
            link_artifact,
            hashlib.sha256(
                json.dumps(
                    link_calibration_subject(base_config),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        ),
    )
    result = analyze_expert_placement(
        routing, sealed_config, receipt_base_dir=tmp_path
    )

    assert result["routing"]["coverage"]["publication_complete"] is True
    assert result["provenance"]["publication_rankable"] is False
    assert result["provenance"]["all_to_all_publication_rankable"] is False
    assert result["provenance"]["evidence_grade"] == "projected"
    assert "selected_hardware_row_binding_missing" in result["provenance"][
        "unrankable_reasons"
    ]
    assert (
        "producer_specific_model_link_artifact_schema_unvalidated"
        in result["provenance"]["unrankable_reasons"]
    )
    assert result["link"]["all_to_all_collective_modelled"] is False

    zero_energy_config = replace(
        sealed_config,
        link_energy_j_per_byte=0.0,
        link_calibration_receipt=None,
    )
    zero_energy_subject = hashlib.sha256(
        json.dumps(
            link_calibration_subject(zero_energy_config),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    zero_energy_config = replace(
        zero_energy_config,
        link_calibration_receipt=receipt(link_artifact, zero_energy_subject),
    )
    zero_energy = analyze_expert_placement(
        routing, zero_energy_config, receipt_base_dir=tmp_path
    )
    assert zero_energy["link"]["energy_j"] == 0
    assert zero_energy["provenance"]["publication_rankable"] is False
    assert "link_energy_j_per_byte_not_positive" in zero_energy["provenance"][
        "unrankable_reasons"
    ]

    mismatched_trace = replace(
        routing,
        receipt=replace(routing.receipt, sample_count=2),
    )
    mismatched = analyze_expert_placement(
        mismatched_trace, sealed_config, receipt_base_dir=tmp_path
    )
    assert mismatched["provenance"]["publication_rankable"] is False
    assert (
        "routing_trace_receipt:sample_count_mismatch"
        in mismatched["provenance"]["unrankable_reasons"]
    )

    link_artifact.write_bytes(b"tampered link sweep")
    tampered = analyze_expert_placement(
        routing, sealed_config, receipt_base_dir=tmp_path
    )
    assert tampered["provenance"]["publication_rankable"] is False
    assert (
        "link_calibration_receipt:artifact_sha256_mismatch"
        in tampered["provenance"]["unrankable_reasons"]
    )


def test_explicit_hot_placement_also_requires_its_receipt(tmp_path: Path) -> None:
    routing = RoutingTrace(
        source_kind="measured",
        steps=(
            RoutingStep(
                0,
                tuple(record("t0", layer=layer) for layer in range(48)),
            ),
        ),
    )
    cfg = config(
        "hot_expert_replication",
        hot_expert_instances=((0, 0),),
        hot_replica_count=2,
    )
    assert hot_placement_subject(cfg, ((0, 0),))["hot_expert_instances"] == [
        [0, 0]
    ]
    result = analyze_expert_placement(routing, cfg, receipt_base_dir=tmp_path)
    assert result["provenance"]["publication_rankable"] is False
    assert (
        "hot_placement_receipt:receipt_missing"
        in result["provenance"]["unrankable_reasons"]
    )


def test_cli_and_content_hash_are_deterministic(tmp_path: Path) -> None:
    payload = {
        "schema": INPUT_SCHEMA,
        "trace": {
            "source_kind": "synthetic",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "steps": [
                {
                    "step_index": 0,
                    "records": [
                        {
                            "token_id": "t0",
                            "layer": 0,
                            "source_chip": 0,
                            "expert_ids": list(range(8)),
                        }
                    ],
                }
            ],
        },
        "config": {
            "policy": "sharded_dispatch",
            "chip_count": 4,
            "expert_weight_bytes": 100,
            "activation_bytes_per_assignment": 10,
            "link_bandwidth_bytes_per_s_per_chip": 100.0,
            "link_energy_j_per_byte": 0.5,
        },
    }
    input_path = tmp_path / "input.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(input_path), "--output", str(first_path)]) == 0
    assert main([str(input_path), "--output", str(second_path)]) == 0
    assert first_path.read_bytes() == second_path.read_bytes()
    parsed = json.loads(first_path.read_text(encoding="utf-8"))
    assert len(parsed["content_hash"]) == 64
    assert parsed["provenance"]["publication_rankable"] is False
