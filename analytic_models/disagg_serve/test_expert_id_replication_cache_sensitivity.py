"""Contracts for held-out hot replicas and ordered expert-weight caches."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ..performance.perf_model import PerfModel, load_hardware_config_from_toml
from .expert_id_full_decode_projection import (
    CONTROL_CYCLIC,
    CONTROL_FREQUENCY,
    FullDecodeProjectionConfig,
)
from .expert_id_replication_cache_sensitivity import (
    ReplicationCacheSensitivityConfig,
    _one_expert_plane,
    audit_expert_id_replication_cache_sensitivity,
    build_expert_id_replication_cache_sensitivity,
    materialize_expert_id_replication_cache_sensitivity,
    validate_expert_id_replication_cache_sensitivity,
)
from .expert_id_trace_balancing import build_expert_id_window_overlay, canonical_hash
from .test_expert_id_trace_balancing import _build, _full_decode_inputs


_SIMULATOR = Path(__file__).resolve().parents[2]
_CONFIG = _SIMULATOR / "plena_settings.toml"
_ISA = _SIMULATOR / "analytic_models" / "performance" / "customISA_lib.json"


@pytest.fixture
def replica_perf() -> PerfModel:
    hardware = load_hardware_config_from_toml(str(_CONFIG)).model_copy(update={"MLEN": 1024, "BLEN": 4, "VLEN": 128})
    return PerfModel(hardware, str(_ISA))


@pytest.fixture
def replica_config():
    from .expert_id_trace_balancing import ExpertIdBalanceConfig

    return ExpertIdBalanceConfig(
        tensor_parallel_degree=2,
        kv_parallel_degree=1,
        batch_size=2,
        train_step_count=4,
        mlen=1024,
        blen=4,
        vlen=128,
        ffn_element_bits=4,
        ffn_effective_bits=5.0,
        mx_block_size=8,
        precision_label="test-mxint4",
        link_generation="nvlink4",
        tp_link_bandwidth_bytes_per_s=450e9,
        tp_link_ports=1,
    )


def _inputs(tmp_path, replica_perf, replica_config, *, kvp=1):
    config = replace(replica_config, kv_parallel_degree=kvp)
    report, trace, _, _ = _build(tmp_path, replica_perf, config)
    balanced = build_expert_id_window_overlay(report, 0, policy=CONTROL_FREQUENCY)
    cyclic = build_expert_id_window_overlay(report, 0, policy=CONTROL_CYCLIC)
    dims, precision = _full_decode_inputs(replica_perf)
    serving = FullDecodeProjectionConfig(
        input_sequence_tokens=16,
        output_sequence_tokens=1,
        stride=1,
    )
    return report, balanced, cyclic, trace, dims, precision, serving, config


def test_zero_budget_parities_full_frequency_control_and_is_deterministic(tmp_path, replica_perf, replica_config):
    values = _inputs(tmp_path, replica_perf, replica_config)
    report, balanced, cyclic, trace, dims, precision, serving, _ = values
    sensitivity = ReplicationCacheSensitivityConfig()
    artifact = build_expert_id_replication_cache_sensitivity(
        report,
        balanced,
        cyclic,
        trace,
        replica_perf,
        dims,
        precision,
        serving,
        sensitivity,
    )
    repeated = build_expert_id_replication_cache_sensitivity(
        report,
        balanced,
        cyclic,
        trace,
        replica_perf,
        dims,
        precision,
        serving,
        sensitivity,
    )

    assert repeated == artifact
    assert artifact["no_replication_no_cache_parity"]["all_checks_passed"] is True
    assert artifact["split"]["train_eval_overlap_count"] == 0
    candidate = artifact["candidates"][0]
    assert candidate["status"] == "evaluated"
    assert candidate["placement_plan"]["held_out_steps_used"] is False
    assert candidate["route_projection"]["totals"]["logical_route_assignments"] == 48 * 2 * 8
    assert candidate["route_projection"]["totals"]["source_hidden_dispatch_bytes"] == 0
    assert candidate["cache_ledger"]["fills"] == 0
    validate_expert_id_replication_cache_sensitivity(artifact)
    assert (
        audit_expert_id_replication_cache_sensitivity(
            artifact,
            report,
            balanced,
            cyclic,
            trace,
            replica_perf,
            dims,
            precision,
            serving,
            sensitivity,
        )
        == artifact
    )


def test_replica_cache_grid_conserves_kvp_residence_fills_and_pareto(tmp_path, replica_perf, replica_config):
    values = _inputs(tmp_path, replica_perf, replica_config, kvp=2)
    report, balanced, cyclic, trace, dims, precision, serving, config = values
    expert_bytes = _one_expert_plane(config, precision).total_aligned
    sensitivity = ReplicationCacheSensitivityConfig(
        replica_budget_bytes_per_rank=(0, expert_bytes),
        cache_budget_bytes_per_rank=(0, expert_bytes),
    )
    artifact = build_expert_id_replication_cache_sensitivity(
        report,
        balanced,
        cyclic,
        trace,
        replica_perf,
        dims,
        precision,
        serving,
        sensitivity,
    )

    assert len(artifact["candidates"]) == 4
    replica_point = next(
        value
        for value in artifact["candidates"]
        if value["budget"]["replica_budget_bytes_per_rank"] == expert_bytes
        and value["budget"]["cache_budget_bytes_per_rank"] == 0
    )
    assert replica_point["placement_plan"]["replica_site_count_by_rank"] == [1, 1]
    assert replica_point["placement_plan"]["all_layers_training_nonregression"] is True
    cache_point = next(
        value
        for value in artifact["candidates"]
        if value["budget"]["replica_budget_bytes_per_rank"] == 0
        and value["budget"]["cache_budget_bytes_per_rank"] == expert_bytes
    )
    assert cache_point["cache_ledger"]["fills"] > 0
    assert cache_point["cache_ledger"]["evictions"] > 0
    assert all(
        value["fills"] == value["evictions"] + value["final_entries"]
        for value in cache_point["cache_ledger"]["per_rank"]
    )
    assert cache_point["full_decode_control"]["loop"]["avg_expert_weight_fetch_endpoint_seconds"] > 0
    cache_bounds = cache_point["cold_vs_ideal_warm_bounds"]
    assert cache_bounds["persistent_across_decode_steps_lru_modelled"] is False
    assert cache_bounds["included_in_pareto_or_best_selection"] is False
    assert (
        cache_bounds["ideal_zero_fill_warm_lower_bound"]["tpot_s"] <= (cache_bounds["cold_reset_projection"]["tpot_s"])
    )
    assert cache_bounds["ideal_zero_fill_warm_lower_bound"]["capacity_enforceable_for_observed_schedule"] is False
    for candidate in artifact["candidates"]:
        assert candidate["status"] == "evaluated"
        route = candidate["route_projection"]
        assert route["totals"]["logical_route_assignments"] == 48 * 2 * 8
        assert route["totals"]["physical_whole_expert_assignments_across_kvp"] == 2 * 48 * 2 * 8
        assert route["totals"]["resident_capacity_uses_rank_aggregate_not_layer_max_sum"] is True
        cache = candidate["cache_ledger"]
        assert cache["fills"] == cache["misses"]
        assert cache["physical_fetch_plane_across_kvp"]["total_aligned"] == (
            2 * cache["logical_fetch_plane_per_tp_group"]["total_aligned"]
        )
        assert (
            candidate["full_decode_control"]["loop"]["collective_breakdown_per_batch_step"]["expert_routing_bytes"] == 0
        )
    assert artifact["pareto_and_best"]["selection_applied"] is False
    assert artifact["pareto_and_best"]["headline_candidate_id"] is None
    assert set(artifact["pareto_and_best"]["pareto_candidate_ids"]).issubset(
        {value["candidate_id"] for value in artifact["candidates"]}
    )


def test_tamper_fails_closed_and_materialization_is_content_addressed(tmp_path, replica_perf, replica_config):
    report, balanced, cyclic, trace, dims, precision, serving, _ = _inputs(tmp_path, replica_perf, replica_config)
    artifact = build_expert_id_replication_cache_sensitivity(
        report,
        balanced,
        cyclic,
        trace,
        replica_perf,
        dims,
        precision,
        serving,
        ReplicationCacheSensitivityConfig(),
    )
    receipt = materialize_expert_id_replication_cache_sensitivity(artifact, tmp_path / "replica_cache")
    assert Path(receipt["path"]).is_file()
    assert receipt["selection_eligible"] is False

    tampered = json.loads(json.dumps(artifact))
    tampered["candidates"][0]["route_projection"]["layers"][0]["source_hidden_dispatch_bytes"] = 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_expert_id_replication_cache_sensitivity(tampered)

    tampered["content_hash"] = canonical_hash({key: value for key, value in tampered.items() if key != "content_hash"})
    with pytest.raises(ValueError, match="route_projection changed"):
        validate_expert_id_replication_cache_sensitivity(tampered)


def test_cache_reservation_beyond_exact_hbm_headroom_is_rejected_without_metrics(
    tmp_path, replica_perf, replica_config
):
    report, balanced, cyclic, trace, dims, precision, serving, _ = _inputs(tmp_path, replica_perf, replica_config)
    artifact = build_expert_id_replication_cache_sensitivity(
        report,
        balanced,
        cyclic,
        trace,
        replica_perf,
        dims,
        precision,
        serving,
        ReplicationCacheSensitivityConfig(cache_budget_bytes_per_rank=(0, 40_000_000_000)),
    )

    rejected = artifact["candidates"][1]
    assert rejected["status"] == ("rejected_cache_reservation_exceeds_hbm_headroom")
    assert "metrics" not in rejected
    assert rejected["classification"]["headline_eligible"] is False
    validate_expert_id_replication_cache_sensitivity(artifact)
