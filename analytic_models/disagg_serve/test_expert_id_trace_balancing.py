"""Contracts for train-only expert-ID placement and held-out repricing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ..performance.perf_model import (
    PerfModel,
    load_hardware_config_from_toml,
    routed_expert_histogram_ledger,
)
from .expert_id_trace_balancing import (
    ExpertIdBalanceConfig,
    audit_expert_id_window_body_reprice,
    audit_expert_id_balancing_report,
    build_expert_id_balancing_report,
    build_expert_id_window_overlay,
    canonical_hash,
    expert_id_body_inputs_for_layer,
    file_hash,
    hardware_binding_for_perf_model,
    materialize_expert_id_balancing_report,
    materialize_expert_id_window_body_reprice,
    materialize_expert_id_window_overlay,
    validate_expert_id_balancing_report,
    validate_expert_id_window_overlay,
    reprice_expert_id_window_body,
)
from .expert_id_full_decode_projection import (
    CONTROL_CYCLIC,
    CONTROL_FREQUENCY,
    CONTROL_TENSOR_EXACT,
    CONTROL_TENSOR_NATIVE,
    FullDecodeProjectionConfig,
    _decode_link_ports,
    _tensor_full_body_layout,
    audit_expert_id_full_decode_projection,
    build_expert_id_full_decode_projection,
    materialize_expert_id_full_decode_projection,
    validate_expert_id_full_decode_projection,
)
from ..performance import disagg_decode as decode
from .expert_placement import (
    EvidenceReceipt,
    RouteRecord,
    RoutingStep,
    RoutingTrace,
    trace_content_hash,
)


_SIMULATOR = Path(__file__).resolve().parents[2]
_CONFIG = _SIMULATOR / "plena_settings.toml"
_ISA = _SIMULATOR / "analytic_models" / "performance" / "customISA_lib.json"


def _write_hashed(path: Path, body: dict) -> dict:
    value = dict(body) | {"content_hash": canonical_hash(body)}
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return value


def _unsealed_trace(
    step_count: int = 6, *, shared_sample: bool = False
) -> RoutingTrace:
    train_experts = (0, 2, 4, 6, 8, 10, 12, 14)
    adversarial_held_out = (0, 4, 8, 12, 1, 5, 9, 13)
    steps = []
    for step_index in range(step_count):
        experts = train_experts if step_index < 4 else adversarial_held_out
        records = tuple(
            RouteRecord(
                token_id=(
                    f"sample-0:cached-decode:{step_index:06d}"
                    if shared_sample
                    else f"token-{step_index}"
                ),
                layer=layer,
                source_chip=step_index % 2,
                expert_ids=experts,
            )
            for layer in range(48)
        )
        steps.append(RoutingStep(step_index=step_index, records=records))
    provisional = EvidenceReceipt(
        artifact_path="/not-yet-bound",
        artifact_sha256="1" * 64,
        subject_sha256="2" * 64,
        command=("collect",),
        tool_revision="3" * 64,
        recorded_at_utc="2026-08-20T00:00:00Z",
        sample_count=step_count,
    )
    return RoutingTrace(
        source_kind="measured",
        steps=tuple(steps),
        receipt=provisional,
    )


def _trace_and_source(
    tmp_path: Path, step_count: int = 6, *, shared_sample: bool = False
) -> tuple[RoutingTrace, dict]:
    trace = _unsealed_trace(step_count, shared_sample=shared_sample)
    trace_hash = trace_content_hash(trace)
    evidence_path = (tmp_path / "evidence.json").resolve()
    evidence = _write_hashed(
        evidence_path,
        {
            "schema": "plena-qwen3-moe-router-trace-evidence/v1",
            "trace_content_hash": trace_hash,
        },
    )
    receipt = EvidenceReceipt(
        artifact_path=str(evidence_path),
        artifact_sha256=file_hash(evidence_path),
        subject_sha256=trace_hash,
        command=("collect",),
        tool_revision="3" * 64,
        recorded_at_utc="2026-08-20T00:00:00Z",
        sample_count=len(trace.steps),
    )
    trace = replace(trace, receipt=receipt)
    placement_path = (tmp_path / "placement.json").resolve()
    placement = _write_hashed(
        placement_path,
        {"schema": "plena-qwen3-moe-expert-placement-input/v1"},
    )
    index_path = (tmp_path / "index.json").resolve()
    index = _write_hashed(
        index_path,
        {
            "schema": "plena-qwen3-moe-router-trace-index/v1",
            "trace_content_hash": trace_hash,
            "token_step_count": len(trace.steps),
            "artifact_path": evidence_path.name,
            "artifact_sha256": file_hash(evidence_path),
            "input_path": placement_path.name,
            "input_sha256": file_hash(placement_path),
        },
    )
    source = {
        "collector_verified": True,
        "router_index_path": str(index_path),
        "router_index_sha256": file_hash(index_path),
        "router_index_content_hash": index["content_hash"],
        "placement_input_path": str(placement_path),
        "placement_input_sha256": file_hash(placement_path),
        "placement_input_content_hash": placement["content_hash"],
        "router_trace_evidence_path": str(evidence_path),
        "router_trace_evidence_sha256": file_hash(evidence_path),
        "router_trace_evidence_content_hash": evidence["content_hash"],
        "trace_content_hash": trace_hash,
    }
    return trace, source


@pytest.fixture
def perf() -> PerfModel:
    hardware = load_hardware_config_from_toml(str(_CONFIG))
    hardware = hardware.model_copy(
        update={"MLEN": 1024, "BLEN": 4, "VLEN": 128}
    )
    return PerfModel(hardware, str(_ISA))


@pytest.fixture
def config() -> ExpertIdBalanceConfig:
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


def _build(tmp_path: Path, perf: PerfModel, config: ExpertIdBalanceConfig):
    trace, source = _trace_and_source(tmp_path)
    hardware = hardware_binding_for_perf_model(
        perf,
        hardware_config_path=_CONFIG,
        custom_isa_path=_ISA,
    )
    report = build_expert_id_balancing_report(
        trace,
        perf,
        config,
        source_binding=source,
        hardware_binding=hardware,
    )
    return report, trace, source, hardware


def test_exact_histogram_perf_path_conserves_and_accepts_empty_rank(perf):
    ledger = routed_expert_histogram_ledger(
        expert_token_count_histogram={"1": 2, "3": 1},
        owned_experts=64,
        blen=4,
        expected_route_assignments=5,
    )
    timing = perf.moe_decode_expert_timing_from_histogram(
        2048,
        768,
        {"1": 2, "3": 1},
        owned_experts=64,
        expected_route_assignments=5,
    )
    empty = perf.moe_decode_expert_timing_from_histogram(
        2048,
        768,
        {},
        owned_experts=64,
        expected_route_assignments=0,
    )
    anonymous = perf.moe_decode_expert_timing(
        2048,
        2,
        128,
        8,
        768,
        active_experts=16,
    )
    exact_balanced = perf.moe_decode_expert_timing_from_histogram(
        2048,
        768,
        {"1": 16},
        owned_experts=64,
        expected_route_assignments=16,
    )

    assert ledger["route_assignments"] == 5
    assert ledger["active_experts"] == 3
    assert timing["expert_stage_cycles"] == (
        timing["matrix_cycles"]
        + timing["activation_cycles"]
        + timing["auxiliary_cycles"]
    )
    assert empty["expert_stage_cycles"] == 0
    assert empty["active_experts"] == 0
    assert exact_balanced["expert_stage_cycles"] == anonymous[
        "expert_stage_cycles"
    ]
    assert exact_balanced["matrix_instruction_histogram"] == anonymous[
        "matrix_instruction_histogram"
    ]
    with pytest.raises(ValueError, match="do not conserve"):
        routed_expert_histogram_ledger(
            expert_token_count_histogram={"2": 2},
            owned_experts=64,
            blen=4,
            expected_route_assignments=5,
        )


def test_report_is_deterministic_disjoint_capacity_exact_and_conserved(
    tmp_path, perf, config
):
    report, trace, source, hardware = _build(tmp_path, perf, config)
    repeated = build_expert_id_balancing_report(
        trace,
        perf,
        config,
        source_binding=source,
        hardware_binding=hardware,
    )

    assert repeated == report
    assert report["split"]["train_eval_overlap_count"] == 0
    assert report["split"]["training"]["last_step_index"] == 3
    assert report["split"]["held_out"]["first_step_index"] == 4
    assert report["placement"]["all_layers_training_nonregression"] is True
    for layer in report["placement"]["layers"]:
        assert layer["owned_expert_count_by_rank"] == [64, 64]
        assert max(layer["selected_assignment_frequency_by_rank"]) <= max(
            layer["baseline_assignment_frequency_by_rank"]
        )
        assert layer["held_out_steps_used_for_placement"] is False
    observations = report["held_out_evaluation"]["observations"]
    assert len(observations) == 48
    for row in observations:
        for policy in ("cyclic_expert_id", "frequency_aware_expert_id"):
            metrics = row["policies"][policy]
            assert sum(metrics["assignment_count_by_rank"]) == 16
            assert sum(
                metrics["physical_assignment_count_by_rank_across_kvp"]
            ) == 16
            assert sum(metrics["active_expert_count_by_rank"]) == len(
                set(
                    expert
                    for values in metrics["active_expert_ids_by_rank"]
                    for expert in values
                )
            )
            assert metrics["source_hidden_dispatch_bytes"] == 0
            assert metrics["expert_output_collective"]["count_per_layer"] == 1
        assert "tensor_parallel_control" in row["policies"]
    assert (
        audit_expert_id_balancing_report(
            report,
            trace,
            perf,
            config,
            source_binding=source,
            hardware_binding=hardware,
        )
        == report
    )


def test_held_out_result_is_reported_without_promising_a_win(tmp_path, perf, config):
    report, *_ = _build(tmp_path, perf, config)
    comparison = report["held_out_evaluation"]["frequency_aware_vs_cyclic"]

    assert comparison["headline_win_claimed"] is False
    assert comparison["strict_cycle_improvement_observed"] is False
    assert comparison["held_out_total_cycle_nonregression"] is False
    assert (
        report["held_out_evaluation"]["frequency_aware_vs_tensor_parallel"][
            "headline_win_claimed"
        ]
        is False
    )
    assert report["classification"]["selection_eligible"] is False
    assert report["reprice_receipt"]["full_tpot_repriced"] is False


def test_kvp_replicas_charge_physical_execution_without_new_training_samples(
    tmp_path, perf, config
):
    report, *_ = _build(
        tmp_path,
        perf,
        replace(config, kv_parallel_degree=2),
    )
    conservation = report["held_out_evaluation"]["assignment_conservation"]
    first = report["held_out_evaluation"]["observations"][0]["policies"][
        "frequency_aware_expert_id"
    ]

    assert conservation["logical_assignments"] == 48 * 16
    assert conservation["physical_whole_expert_assignments_across_kvp"] == 2 * 48 * 16
    assert sum(first["assignment_count_by_rank"]) == 16
    assert sum(first["physical_assignment_count_by_rank_across_kvp"]) == 32
    assert "not_resampled" in report["placement"]["training_trace_semantics"]


def test_split_must_be_disjoint_at_collector_sample_boundary(tmp_path, perf, config):
    trace, source = _trace_and_source(tmp_path, shared_sample=True)
    hardware = hardware_binding_for_perf_model(
        perf,
        hardware_config_path=_CONFIG,
        custom_isa_path=_ISA,
    )

    with pytest.raises(ValueError, match="sample IDs"):
        build_expert_id_balancing_report(
            trace,
            perf,
            config,
            source_binding=source,
            hardware_binding=hardware,
        )


def test_tampering_and_changed_source_fail_closed(tmp_path, perf, config):
    report, trace, source, hardware = _build(tmp_path, perf, config)
    tampered = json.loads(json.dumps(report))
    tampered["split"]["train_eval_overlap_count"] = 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_expert_id_balancing_report(tampered)

    tampered["content_hash"] = canonical_hash(
        {key: value for key, value in tampered.items() if key != "content_hash"}
    )
    with pytest.raises(ValueError, match="splits overlap"):
        validate_expert_id_balancing_report(tampered)

    source_path = Path(source["router_trace_evidence_path"])
    source_path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        audit_expert_id_balancing_report(
            report,
            trace,
            perf,
            config,
            source_binding=source,
            hardware_binding=hardware,
        )


def test_layer_exact_overlay_drives_body_inputs_without_global_collapse(
    tmp_path, perf, config
):
    report, *_ = _build(tmp_path, perf, config)
    installed = materialize_expert_id_balancing_report(
        report, tmp_path / "artifacts"
    )
    report_path = Path(installed["report_path"])
    receipt = {
        "path": str(report_path),
        "sha256": file_hash(report_path),
        "content_hash": report["content_hash"],
    }
    overlay = build_expert_id_window_overlay(
        report,
        0,
        report_receipt=receipt,
    )
    validate_expert_id_window_overlay(overlay, report=report)
    inputs = expert_id_body_inputs_for_layer(overlay, 0)
    overlay_receipt = materialize_expert_id_window_overlay(
        overlay, tmp_path / "artifacts"
    )

    assert len(overlay["layers"]) == 48
    assert overlay["adapter_contract"]["layer_specific_inputs_required"] is True
    assert (
        overlay["adapter_contract"]["global_active_count_collapse_allowed"]
        is False
    )
    assert sum(inputs["active_experts_per_rank"]) == inputs["unique_experts"]
    assert sum(inputs["physical_assignment_count_by_rank_across_kvp"]) == 16
    assert inputs["source_hidden_dispatch_bytes"] == 0
    assert inputs["expert_output_collective"]["count_per_layer"] == 1
    assert Path(overlay_receipt["path"]).is_file()

    tampered = json.loads(json.dumps(overlay))
    tampered["layers"][0]["active_experts_per_rank"][0] += 1
    tampered["content_hash"] = canonical_hash(
        {key: value for key, value in tampered.items() if key != "content_hash"}
    )
    with pytest.raises(ValueError, match="ownership or routes"):
        validate_expert_id_window_overlay(tampered)


def test_isolated_body_reprice_exposes_tp4_kvp_capacity_crossover(
    tmp_path, perf, config
):
    target = replace(
        config,
        tensor_parallel_degree=4,
        batch_size=4,
        train_step_count=4,
    )
    trace, source = _trace_and_source(tmp_path, step_count=8)
    hardware = hardware_binding_for_perf_model(
        perf,
        hardware_config_path=_CONFIG,
        custom_isa_path=_ISA,
    )
    report = build_expert_id_balancing_report(
        trace,
        perf,
        target,
        source_binding=source,
        hardware_binding=hardware,
    )
    overlay = build_expert_id_window_overlay(report, 0)
    dims = {
        "hidden": 2048,
        "inter": 768,
        "layers": 48,
        "heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "vocab": 151936,
        "num_experts": 128,
        "experts_per_token": 8,
        "qk_norm": True,
    }
    precision = {
        "attn_elem": 4,
        "attn_bits": 5.0,
        "ffn_elem": 4,
        "ffn_bits": 5.0,
        "kv_elem": 4,
        "kv_bits": 5.0,
        "key_elem": 4,
        "key_bits": 5.0,
        "value_elem": 4,
        "value_bits": 5.0,
        "head_elem": 4,
        "head_bits": 5.0,
        "block_size": 8,
        "lm_head_quantized": True,
    }
    reprice = reprice_expert_id_window_body(overlay, perf, dims, precision)
    crossover = reprice["capacity_and_chip_count_crossover"]

    assert reprice["classification"]["full_tpot_repriced"] is False
    assert reprice["totals"]["logical_route_assignments"] == 48 * 4 * 8
    assert (
        reprice["totals"]["physical_whole_expert_assignments_across_kvp"]
        == 48 * 4 * 8
    )
    assert crossover["expert_id_parallel"]["rows"][0]["feasible"] is True
    assert crossover["tensor_parallel_control"]["rows"][0]["feasible"] is False
    assert (
        crossover["expert_id_parallel"]["minimum_feasible_kv_parallel_degree"]
        == 1
    )
    assert (
        crossover["tensor_parallel_control"][
            "minimum_feasible_kv_parallel_degree"
        ]
        == 2
    )
    assert crossover["expert_id_reduces_minimum_kvp"] is True
    assert (
        audit_expert_id_window_body_reprice(
            reprice,
            overlay,
            perf,
            dims,
            precision,
        )
        == reprice
    )
    receipt = materialize_expert_id_window_body_reprice(
        reprice, tmp_path / "body"
    )
    assert receipt["full_tpot_repriced"] is False
    assert Path(receipt["path"]).is_file()


def _full_decode_inputs(perf: PerfModel) -> tuple[dict, dict]:
    dims = {
        "hidden": 2048,
        "inter": 768,
        "layers": 48,
        "heads": 32,
        "kv_heads": 4,
        "head_dim": 64,
        "vocab": 151936,
        "num_experts": 128,
        "experts_per_token": 8,
        "qk_norm": True,
        "n_full": 48,
        "n_sliding": 0,
        "sliding_window": 0,
    }
    precision = {
        "attn_elem": 4,
        "attn_bits": 5.0,
        "ffn_elem": 4,
        "ffn_bits": 5.0,
        "kv_elem": 4,
        "kv_bits": 5.0,
        "key_elem": 4,
        "key_bits": 5.0,
        "value_elem": 4,
        "value_bits": 5.0,
        "head_elem": 4,
        "head_bits": 5.0,
        "block_size": 8,
        "m_bits": 4,
        "density_exp": 0.0,
        "lm_head_quantized": True,
        "profile_id": "test-local-mx-head-v3",
        "head_activation_bits": 4,
        "head_activation_elem": 4,
        "head_activation_label": "MXINT4",
        "head_vector_format": "MXINT4",
        "head_matrix_storage_format": "MXINT4",
        "head_logit_container_format": "BF16",
        "head_bf16_container_precision_recovery": False,
        "head_operand_family_supported": True,
        "head_operand_family_binding": "test",
        "head_numerical_oracle_rule": "test",
        "head_partial_conversion_rule": "test",
        "head_hardware_bit_parity_verified": False,
        "head_accumulation_chain": "int32_then_bf16",
        "head_numerical_matrix_mlen": int(perf.mlen),
    }
    return dims, precision


def test_full_decode_projection_replaces_only_layer_exact_expert_work_and_parities_native_tensor(
    tmp_path, perf, config
):
    report, *_ = _build(tmp_path, perf, config)
    balanced = build_expert_id_window_overlay(
        report, 0, policy=CONTROL_FREQUENCY
    )
    cyclic = build_expert_id_window_overlay(report, 0, policy=CONTROL_CYCLIC)
    dims, precision = _full_decode_inputs(perf)
    serving = FullDecodeProjectionConfig(
        input_sequence_tokens=32,
        output_sequence_tokens=2,
        stride=1,
        kv_head_reuse=False,
    )
    projection = build_expert_id_full_decode_projection(
        report, balanced, cyclic, perf, dims, precision, serving
    )
    validate_expert_id_full_decode_projection(
        projection,
        report=report,
        balanced_overlay=balanced,
        cyclic_overlay=cyclic,
    )

    assert projection["classification"]["full_tpot_repriced"] is True
    assert projection["classification"]["selection_eligible"] is False
    assert projection["comparisons"]["headline_win_claimed"] is False
    assert projection["conservation"]["source_hidden_dispatch_bytes"] == 0
    assert (
        projection["conservation"][
            "physical_whole_expert_assignments_across_kvp_per_projected_batch_step"
        ]
        == 48 * 2 * 8
    )
    for name in (CONTROL_FREQUENCY, CONTROL_CYCLIC, CONTROL_TENSOR_EXACT):
        route = projection["route_projections"][name]
        assert len(route["layers"]) == 48
        assert route["global_layer_collapse_allowed"] is False
        assert route["totals"]["source_hidden_dispatch_bytes"] == 0
        assert route["totals"][
            "resident_bytes_use_all_layer_specific_ownership_records"
        ] is True
        assert projection["controls"][name]["route_hook_applied"] is True
        assert route["totals"][
            "sum_of_per_layer_slowest_rank_expert_resident_bytes"
        ] <= projection["controls"][name]["capacity"][
            "slowest_rank_resident_weight_bytes"
        ]
        for sample in projection["controls"][name][
            "component_and_hbm_step_proof"
        ]["samples"]:
            assert sample["component_cycle_sum"] == pytest.approx(
                sum(sample["component_cycles"].values())
            )

    native_control = projection["controls"][CONTROL_TENSOR_NATIVE]
    assert native_control["route_hook_applied"] is False
    balance = ExpertIdBalanceConfig(**report["study_config"])
    native_layout = _tensor_full_body_layout(
        None, dims, precision, balance
    )
    direct = decode.run_decode_loop(
        perf,
        None,
        dims,
        precision,
        serving.input_sequence_tokens,
        serving.output_sequence_tokens,
        balance.batch_size,
        decode.peak_hbm_bw_bytes(perf.config),
        serving.stride,
        decode.matrix_overfetch_factor(perf.config),
        batch_packed_attention=serving.batch_packed_attention,
        n_chips=balance.tensor_parallel_degree * balance.kv_parallel_degree,
        kv_layout=serving.kv_layout,
        ideal_perf=perf,
        include_lm_head=True,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        link_ports=_decode_link_ports(balance),
        link_generation=balance.link_generation,
        sram_policy=serving.sram_policy,
        legacy_ideal_parallelism=False,
        kv_head_reuse=serving.kv_head_reuse,
        body_weight_layout=native_layout,
        expert_parallel_mode="tensor_parallel",
        layer_exact_moe_route_projection=None,
        execution_mode=serving.execution_mode,
    )
    for field in (
        "total_time",
        "tpot",
        "tps",
        "avg_bytes_per_batch_step",
        "avg_realized_compute_seconds",
        "avg_memory_seconds",
        "avg_collective_seconds",
    ):
        assert native_control["loop"][field] == pytest.approx(
            direct[field], rel=1e-12, abs=1e-15
        )
    assert (
        audit_expert_id_full_decode_projection(
            projection,
            report,
            balanced,
            cyclic,
            perf,
            dims,
            precision,
            serving,
        )
        == projection
    )
    receipt = materialize_expert_id_full_decode_projection(
        projection, tmp_path / "full_decode"
    )
    assert receipt["full_tpot_repriced"] is True
    assert receipt["selection_eligible"] is False
    assert Path(receipt["path"]).is_file()


def test_full_decode_projection_tamper_and_topology_mismatch_fail_closed(
    tmp_path, perf, config
):
    report, *_ = _build(tmp_path, perf, config)
    balanced = build_expert_id_window_overlay(report, 0)
    cyclic = build_expert_id_window_overlay(
        report, 0, policy=CONTROL_CYCLIC
    )
    dims, precision = _full_decode_inputs(perf)
    serving = FullDecodeProjectionConfig(
        input_sequence_tokens=16, output_sequence_tokens=1
    )
    projection = build_expert_id_full_decode_projection(
        report, balanced, cyclic, perf, dims, precision, serving
    )
    tampered = json.loads(json.dumps(projection))
    tampered["route_projections"][CONTROL_FREQUENCY]["layers"][0][
        "source_hidden_dispatch_bytes"
    ] = 1
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_expert_id_full_decode_projection(tampered)

    route = tampered["route_projections"][CONTROL_FREQUENCY]
    route["content_hash"] = canonical_hash(
        {key: value for key, value in route.items() if key != "content_hash"}
    )
    tampered["controls"][CONTROL_FREQUENCY][
        "route_projection_content_hash"
    ] = route["content_hash"]
    tampered["controls"][CONTROL_FREQUENCY]["loop"][
        "layer_exact_moe_route_projection"
    ]["content_hash"] = route["content_hash"]
    tampered["content_hash"] = canonical_hash(
        {key: value for key, value in tampered.items() if key != "content_hash"}
    )
    with pytest.raises(ValueError, match="dispatch or collective"):
        validate_expert_id_full_decode_projection(tampered)

    wrong_precision = dict(precision) | {"ffn_bits": 6.0}
    with pytest.raises(ValueError, match="FFN precision"):
        build_expert_id_full_decode_projection(
            report,
            balanced,
            cyclic,
            perf,
            dims,
            wrong_precision,
            serving,
        )


def test_full_decode_projection_keeps_logical_routes_separate_from_kvp_replicas(
    tmp_path, perf, config
):
    kvp_config = replace(config, kv_parallel_degree=2)
    report, *_ = _build(tmp_path, perf, kvp_config)
    balanced = build_expert_id_window_overlay(report, 0)
    cyclic = build_expert_id_window_overlay(
        report, 0, policy=CONTROL_CYCLIC
    )
    dims, precision = _full_decode_inputs(perf)
    serving = FullDecodeProjectionConfig(
        input_sequence_tokens=16, output_sequence_tokens=1
    )
    projection = build_expert_id_full_decode_projection(
        report, balanced, cyclic, perf, dims, precision, serving
    )

    assert (
        projection["conservation"][
            "logical_route_assignments_per_projected_batch_step"
        ]
        == 48 * 2 * 8
    )
    assert (
        projection["conservation"][
            "physical_whole_expert_assignments_across_kvp_per_projected_batch_step"
        ]
        == 2 * 48 * 2 * 8
    )
    route = projection["route_projections"][CONTROL_FREQUENCY]
    for row in route["layers"]:
        assert sum(row["assignment_count_by_rank"]) == 2 * 8
        assert sum(row["physical_assignment_count_by_rank_across_kvp"]) == 2 * 2 * 8
        assert row["physical_whole_expert_assignments_across_kvp"] == 2 * 2 * 8
        assert row["expert_output_collective_count"] == 1
        assert row["system_expert_streamed"]["total_aligned"] == 2 * sum(
            count
            * report["held_out_evaluation"]["observations"][row["layer"]][
                "policies"
            ][CONTROL_FREQUENCY]["expert_weight_bytes_per_active_expert"]
            for count in row["active_experts_per_rank"]
        )
    collective = projection["controls"][CONTROL_FREQUENCY]["loop"][
        "collective_breakdown_per_batch_step"
    ]
    assert collective["expert_routing_bytes"] == 0
    assert collective["expert_output_collective_system_bytes"] == (
        collective["expert_output_collective_slowest_rank_bytes"]
        * kvp_config.tensor_parallel_degree
        * kvp_config.kv_parallel_degree
    )
