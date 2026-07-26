from __future__ import annotations

import math
from copy import deepcopy

import pytest

from analytic_models.performance.multi_chip_model import (
    aggregate_area,
    build_parallel_work_census,
    estimate_multi_chip_latency,
    fp16_kv_handoff,
    matrix_sram_requirements,
    matrix_sram_search_values,
    valid_ep_degrees,
    valid_tp_degrees,
    zigzag_context_partition,
)


MODEL = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
MOE_MODEL = {
    "hidden_size": 4096,
    "intermediate_size": 1536,
    "moe_intermediate_size": 1536,
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}


def _report() -> dict:
    by_stage = {
        "layer/attention": {
            "physical_read_bytes": 1_000,
            "physical_write_bytes": 0,
            "read_requests": 10,
        },
        "layer/ffn": {
            "physical_read_bytes": 1_000,
            "physical_write_bytes": 0,
            "read_requests": 10,
        },
    }
    return {
        "stage_compute_latency_ns": {
            "layer/attention": 100.0,
            "layer/ffn": 100.0,
        },
        "hbm_stage_latency_ns": {
            "layer/attention": 80.0,
            "layer/ffn": 80.0,
        },
        "stage_roofline_latency_ns": {
            "layer/attention": 100.0,
            "layer/ffn": 100.0,
        },
        "category_latency_ns": {
            "matrix_compute": 100.0,
            "vector_compute": 60.0,
            "scalar_compute": 30.0,
            "control": 10.0,
        },
        "compatibility": {
            "theoretical_floor_ns": 100.0,
            "stage_theoretical_floor_ns": {
                "layer/attention": 40.0,
                "layer/ffn": 60.0,
            },
        },
        "hbm_traffic_breakdown": {
            "by_stage": by_stage,
            "by_role": {
                "weight": {
                    "physical_read_bytes": 1_000,
                    "physical_write_bytes": 0,
                    "read_requests": 10,
                },
                "activation": {
                    "physical_read_bytes": 1_000,
                    "physical_write_bytes": 0,
                    "read_requests": 10,
                },
            },
            "by_stage_role": {
                "layer/attention::activation": by_stage["layer/attention"],
                "layer/ffn::weight": by_stage["layer/ffn"],
            },
        },
    }


def _factorized_report() -> dict:
    report = _report()
    report["compute_timing_mode"] = "ideal-ii1"
    report["stage_compute_opcode_work_cycles"] = {
        "layer/attention": {"M_MM": 40, "V_RED_SUM_SEG": 60},
        "layer/ffn": {"M_MM": 100},
    }
    breakdown = report["hbm_traffic_breakdown"]
    breakdown["by_opcode_role"] = {
        "H_PREFETCH_V::activation": breakdown["by_stage"][
            "layer/attention"
        ],
        "H_PREFETCH_M::weight": breakdown["by_stage"]["layer/ffn"],
    }
    breakdown["by_stage_opcode_role"] = {
        "layer/attention::H_PREFETCH_V::activation": breakdown["by_stage"][
            "layer/attention"
        ],
        "layer/ffn::H_PREFETCH_M::weight": breakdown["by_stage"][
            "layer/ffn"
        ],
    }
    report["compatibility"]["clock_period_ps"] = 1_000
    report["trace"] = {
        "schema_version": 7,
        "hardware": {"mlen": 8192, "blen": 1024, "hlen": 128},
        "native_layout": {
            "physical_rows": 8192,
        },
        "workload": {
            "model_type": "qwen3",
            "hidden_size": MODEL["hidden_size"],
            "inter_dim": MODEL["intermediate_size"],
            "num_experts": 0,
            "experts_per_token": 0,
            "num_heads": MODEL["num_attention_heads"],
            "num_kv_heads": MODEL["num_key_value_heads"],
            "head_dim": MODEL["head_dim"],
            "batch_size": 16,
            "seq_len": 482,
        },
        "compiler_metadata": {},
        "parallel_kernel_census_schema": (
            "parallel_kernel_census_v2_schedule_lineage"
        ),
        "parallel_kernel_census_coverage": 1.0,
        "parallel_kernel_census": [
            {
                "stage": "layer/attention",
                "kernel": "attention_projection",
                "opcode": "M_MM",
                "count": 40,
                "tp_semantics": "attention_projection_tiled",
                "cp_semantics": "token_partitioned",
                "ep_semantics": "none",
            },
            {
                "stage": "layer/attention",
                "kernel": "attention_core",
                "opcode": "V_RED_SUM_SEG",
                "count": 60,
                "tp_semantics": "attention_head_pair_sharded",
                "cp_semantics": "causal_block_partitioned",
                "ep_semantics": "none",
            },
            {
                "stage": "layer/ffn",
                "kernel": "dense_ffn_projection",
                "opcode": "M_MM",
                "count": 100,
                "tp_semantics": "ffn_projection_tiled",
                "cp_semantics": "token_partitioned",
                "ep_semantics": "none",
            },
        ],
    }
    return report


def _moe_report() -> dict:
    report = _factorized_report()
    stages = (
        "layer/moe/router",
        "layer/moe/dispatch",
        "layer/moe/experts",
        "layer/moe/combine",
    )
    report["stage_compute_latency_ns"] = {
        stage: value
        for stage, value in zip(stages, (10.0, 20.0, 100.0, 20.0))
    }
    report["stage_compute_opcode_work_cycles"] = {
        "layer/moe/router": {"M_MM": 10},
        "layer/moe/dispatch": {"V_ADD_VV": 20},
        "layer/moe/experts": {"M_MM": 100},
        "layer/moe/combine": {"V_ADD_VV": 20},
    }
    report["hbm_stage_latency_ns"] = {stage: 8.0 for stage in stages}
    report["compatibility"]["stage_theoretical_floor_ns"] = {
        stage: 4.0 for stage in stages
    }
    report["compatibility"]["theoretical_floor_ns"] = 16.0
    roles = {
        "layer/moe/router": "weight",
        "layer/moe/dispatch": "activation",
        "layer/moe/experts": "weight",
        "layer/moe/combine": "activation",
    }
    by_stage = {
        stage: {
            "physical_read_bytes": 1_000,
            "physical_write_bytes": 0,
            "payload_read_bytes": 1_000,
            "payload_write_bytes": 0,
            "read_requests": 10,
            "write_requests": 0,
        }
        for stage in stages
    }
    report["hbm_traffic_breakdown"] = {
        "by_stage": by_stage,
        "by_stage_role": {
            f"{stage}::{roles[stage]}": bucket
            for stage, bucket in by_stage.items()
        },
        "by_stage_opcode_role": {
            f"{stage}::H_PREFETCH_M::{roles[stage]}": bucket
            for stage, bucket in by_stage.items()
        },
        "by_role": {
            role: {
                field: sum(
                    bucket.get(field, 0)
                    for stage, bucket in by_stage.items()
                    if roles[stage] == role
                )
                for field in (
                    "physical_read_bytes",
                    "physical_write_bytes",
                    "payload_read_bytes",
                    "payload_write_bytes",
                    "read_requests",
                    "write_requests",
                )
            }
            for role in set(roles.values())
        },
    }
    report["trace"] = {
        "schema_version": 7,
        "hardware": {"mlen": 512, "blen": 64, "hlen": 128},
        "native_layout": {"physical_rows": 512},
        "workload": {
            "model_type": "qwen3_moe",
            "hidden_size": MOE_MODEL["hidden_size"],
            "inter_dim": MOE_MODEL["moe_intermediate_size"],
            "num_experts": MOE_MODEL["num_experts"],
            "experts_per_token": MOE_MODEL["num_experts_per_tok"],
            "num_heads": MOE_MODEL["num_attention_heads"],
            "num_kv_heads": MOE_MODEL["num_key_value_heads"],
            "head_dim": MOE_MODEL["head_dim"],
            "batch_size": 1,
            "seq_len": 32,
        },
        "compiler_metadata": {"moe_routing_mode": "fixed-balanced"},
        "parallel_kernel_census_schema": (
            "parallel_kernel_census_v2_schedule_lineage"
        ),
        "parallel_kernel_census_coverage": 1.0,
        "parallel_kernel_census": [
            {
                "stage": stage,
                "kernel": kernel,
                "opcode": opcode,
                "count": count,
                "tp_semantics": semantic,
                "cp_semantics": "token_partitioned",
                "ep_semantics": ep_semantic,
            }
            for stage, kernel, opcode, count, semantic, ep_semantic in (
                (
                    "layer/moe/router",
                    "moe_router",
                    "M_MM",
                    10,
                    "row_parallel_projection",
                    "router_replicated",
                ),
                (
                    "layer/moe/dispatch",
                    "moe_dispatch",
                    "V_ADD_VV",
                    20,
                    "token_replicated_hidden",
                    "expert_dispatch",
                ),
                (
                    "layer/moe/experts",
                    "moe_expert_ffn",
                    "M_MM",
                    100,
                    "expert_tensor_sharded",
                    "expert_partitioned",
                ),
                (
                    "layer/moe/combine",
                    "moe_combine",
                    "V_ADD_VV",
                    20,
                    "token_replicated_hidden",
                    "expert_combine",
                ),
            )
        ],
    }
    return report


def test_tp_domain_and_zigzag_partition_are_exact() -> None:
    assert valid_tp_degrees(MODEL, 16) == (1, 2, 4, 8)
    partition = zigzag_context_partition(17, 4)
    assert sum(rank["tokens"] for rank in partition["ranks"]) == 17
    assert sum(rank["causal_pairs"] for rank in partition["ranks"]) == 153
    assert partition["max_token_fraction"] < 0.30
    assert partition["max_causal_pair_fraction"] < 0.30


def test_factorized_single_chip_is_exact_and_census_is_complete() -> None:
    report = _factorized_report()
    census = build_parallel_work_census(report)
    assert census["coverage"] == 1.0
    result = estimate_multi_chip_latency(
        report,
        MODEL,
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="factorized-tp-cp-v2",
        tp_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=1,
    )
    assert result["latency_ns"] == 200.0
    assert result["tp_degree"] == result["cp_degree"] == 1
    assert result["parallel_work_census_coverage"] == 1.0
    assert result["interconnect_bytes"] == 0.0


def test_tile_aware_single_chip_is_an_exact_identity() -> None:
    result = estimate_multi_chip_latency(
        _factorized_report(),
        MODEL,
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        tp_degree=1,
        ep_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=1,
    )
    assert result["latency_ns"] == 200.0
    assert result["fractional_v2_latency"] == result["tile_aware_v3_latency"]
    assert result["parallel_kernel_census_coverage"] == 1.0
    expected = _factorized_report()["hbm_traffic_breakdown"]["by_stage"]
    actual = result["aggregate_hbm_traffic_breakdown"]["by_stage"]
    for stage, bucket in expected.items():
        assert actual[stage]["physical_read_bytes"] == bucket[
            "physical_read_bytes"
        ]
        assert actual[stage]["physical_write_bytes"] == bucket[
            "physical_write_bytes"
        ]
        assert actual[stage]["read_requests"] == bucket["read_requests"]


def test_tile_aware_rejects_non_additive_compute_timing() -> None:
    report = _factorized_report()
    report["compute_timing_mode"] = "rtl-v1"
    with pytest.raises(ValueError, match="additive ideal-II1 work"):
        estimate_multi_chip_latency(
            report,
            MODEL,
            chip_count=2,
            reference_a100_count=1,
            parallel_model="tp-sp",
            multi_chip_model="tile-aware-tp-cp-ep-v3",
            tp_degree=1,
            ep_degree=1,
            aggregate_hbm_bandwidth_gbps=2039.0,
            aggregate_hbm_capacity_bytes=80_000_000_000,
            seq_len=482,
            batch_size=16,
            fp_width_bits=12,
            kv_width_bits=8.125,
            nvlink_port_count=1,
        )


def test_tile_aware_large_mlen_preserves_projection_tile_floor() -> None:
    report = _factorized_report()
    result = estimate_multi_chip_latency(
        report,
        MODEL,
        chip_count=4,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        tp_degree=4,
        ep_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=4,
    )
    # hidden=5120 and every local packed-Q projection remain one 8192 tile.
    assert all(
        rank["q_storage_blocks"] == 1
        for rank in result["local_tile_counts_by_rank"]
    )
    assert result["tile_aware_v3_latency"] > result["fractional_v2_latency"]
    assert result["tp_rounding_overhead"] > 0
    assert result["padding_cycles"] >= 0
    assert result["parallel_action_scales_by_kernel_opcode"]
    assert result["parallel_action_scales_by_kernel"]
    assert result["v4_local_geometry_reconstruction"] is False
    assert result["v4_rank_latency_exact"] is False


def test_tile_aware_reports_average_rank_traffic_for_power() -> None:
    result = estimate_multi_chip_latency(
        _factorized_report(),
        MODEL,
        chip_count=4,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        tp_degree=1,
        ep_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=4097,
        batch_size=1,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=1,
    )
    aggregate = result["aggregate_hbm_traffic_breakdown"]["by_stage"]
    average = result["average_per_chip_hbm_traffic_breakdown"][
        "by_stage"
    ]
    for stage, bucket in aggregate.items():
        for field, value in bucket.items():
            assert average[stage][field] * 4 == value


def test_tile_aware_cp_census_conserves_tokens_and_keeps_tail_work() -> None:
    report = _factorized_report()
    report["trace"]["workload"]["batch_size"] = 1
    report["trace"]["workload"]["seq_len"] = 4097
    report["trace"]["native_layout"]["physical_rows"] = 8192
    result = estimate_multi_chip_latency(
        report,
        MODEL,
        chip_count=4,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        tp_degree=1,
        ep_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=4097,
        batch_size=1,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=4,
    )
    assert sum(
        rank["active_rows"] for rank in result["local_tile_counts_by_rank"]
    ) == 4097
    assert result["cp_tail_overhead"] > 0
    assert result["tail_isa_limitation"] == "active_row_bmm_unavailable"


def test_tile_aware_moe_ep_conserves_routes_and_replication() -> None:
    assert valid_ep_degrees(
        MOE_MODEL, 4, routing_mode="fixed-balanced"
    ) == (1, 2, 4)
    result = estimate_multi_chip_latency(
        _moe_report(),
        MOE_MODEL,
        chip_count=8,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        tp_degree=2,
        ep_degree=4,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=32,
        batch_size=1,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=4,
    )
    owned_routes = sum(
        rank["expert"]["owned_route_count"]
        for rank in result["local_tile_counts_by_rank"]
    )
    assert owned_routes == 32 * 8 * 2
    assert result["expert_weight_replication"] == 1.0
    assert result["experts_per_rank"] == 32
    assert result["ep_dispatch_bytes"] > 0
    assert result["ep_return_bytes"] > 0
    assert 0 < result["expert_bucket_utilization"] <= 1
    tp_bytes = result["tp_collective_bytes_by_stage"]
    tp_latency = result["tp_collective_latency_ns_by_stage"]
    assert "layer/ffn" not in tp_bytes
    assert "layer/ffn" not in tp_latency
    assert tp_bytes["layer/moe/router"] > 0
    assert tp_bytes["layer/moe/experts"] > 0
    assert result["tp_collective_bytes"] == sum(tp_bytes.values())
    assert result["tp_collective_latency_ns"] == sum(tp_latency.values())
    assert result["interconnect_bytes_by_stage"]["layer/moe/dispatch"] >= (
        result["ep_dispatch_bytes"]
    )
    assert result["interconnect_bytes_by_stage"]["layer/moe/combine"] >= (
        result["ep_return_bytes"]
    )


def test_factorized_ports_use_full_peak_and_bounds_are_ordered() -> None:
    common = dict(
        report=_factorized_report(),
        model=MODEL,
        chip_count=4,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="factorized-tp-cp-v2",
        tp_degree=2,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
    )
    one_port = estimate_multi_chip_latency(nvlink_port_count=1, **common)
    four_ports = estimate_multi_chip_latency(nvlink_port_count=4, **common)
    assert one_port["nvlink_peak_oneway_bandwidth_gbps"] == 450.0
    assert four_ports["nvlink_peak_oneway_bandwidth_gbps"] == 1800.0
    assert four_ports["bandwidth_efficiency"] == 1.0
    expected_local_activation_bytes = (
        one_port["context_partition"]["max_local_tokens"]
        * 16
        * MODEL["hidden_size"]
        * 12
        / 8
    )
    expected_ring_bytes = (
        2 * (one_port["tp_degree"] - 1)
        / one_port["tp_degree"]
        * expected_local_activation_bytes
        * MODEL["num_hidden_layers"]
    )
    assert one_port["tp_collective_bytes_by_stage"]["layer/attention"] == (
        expected_ring_bytes
    )
    assert four_ports["latency_ns"] <= one_port["latency_ns"]
    assert (
        four_ports["full_overlap_lower_bound_ns"]
        <= four_ports["nominal_stage_model_ns"]
        <= four_ports["no_overlap_upper_bound_ns"]
    )
    sensitivity = four_ports["interconnect_startup_sensitivity"]
    assert (
        sensitivity["1.0"]["latency_ns"]
        <= sensitivity["2.5"]["latency_ns"]
        <= sensitivity["4.0"]["latency_ns"]
    )


def test_factorized_compute_preserves_cycle_units_at_non_1ghz() -> None:
    report = _factorized_report()
    report["compatibility"]["clock_period_ps"] = 2_000
    report["stage_compute_latency_ns"] = {
        stage: value * 2.0
        for stage, value in report["stage_compute_latency_ns"].items()
    }
    result = estimate_multi_chip_latency(
        report,
        MODEL,
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="factorized-tp-cp-v2",
        tp_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=1,
    )
    assert result["per_chip_stage_compute_latency_ns"] == {
        "layer/attention": 200.0,
        "layer/ffn": 200.0,
    }


def test_factorized_cp_replicates_weight_traffic() -> None:
    result = estimate_multi_chip_latency(
        _factorized_report(),
        MODEL,
        chip_count=4,
        reference_a100_count=1,
        parallel_model="tp-sp",
        multi_chip_model="factorized-tp-cp-v2",
        tp_degree=1,
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        kv_width_bits=8.125,
        nvlink_port_count=4,
    )
    assert result["weight_replication_factor"] == 4
    assert (
        result["aggregate_hbm_traffic_breakdown"]["by_role"]["weight"][
            "physical_read_bytes"
        ]
        == 4_000
    )


def test_matrix_sram_thresholds_and_non_power_search_points() -> None:
    requirements = matrix_sram_requirements(
        MODEL,
        mlen=512,
        seq_len=482,
        chip_count=1,
        parallel_model="tp-sp",
    )
    assert requirements["projection_threshold_tiles"] == 50
    assert requirements["attention_threshold_tiles"] == 2
    assert requirements["matrix_sram_useful_saturation_tiles"] == 50

    long_context = matrix_sram_requirements(
        MODEL,
        mlen=512,
        seq_len=4097,
        chip_count=1,
        parallel_model="tp-sp",
    )
    assert long_context["attention_threshold_tiles"] == 18
    assert long_context["matrix_sram_useful_saturation_tiles"] == 50

    values = matrix_sram_search_values(
        MODEL,
        mlens=(512, 1024, 2048),
        seq_len=482,
        chip_counts=(1,),
        parallel_models=("tp-sp",),
    )
    assert {13, 25, 50} <= set(values)


def test_single_chip_exactly_reproduces_stage_roofline() -> None:
    result = estimate_multi_chip_latency(
        _report(),
        MODEL,
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
    )
    assert result["latency_ns"] == 200.0
    assert result["interconnect_latency_ns"] == 0.0
    assert result["per_chip_stage_memory_latency_ns"] == {
        "layer/attention": 80.0,
        "layer/ffn": 80.0,
    }
    assert result["per_chip_stage_v4_floor_ns"] == {
        "layer/attention": 40.0,
        "layer/ffn": 60.0,
    }


def test_fixed_aggregate_hbm_does_not_create_false_floor_speedup() -> None:
    single = estimate_multi_chip_latency(
        _report(),
        MODEL,
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        one_way_link_bandwidth_gbps=1e30,
    )
    dual = estimate_multi_chip_latency(
        _report(),
        MODEL,
        chip_count=2,
        reference_a100_count=1,
        parallel_model="tp-sp",
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=482,
        batch_size=16,
        fp_width_bits=12,
        one_way_link_bandwidth_gbps=1e30,
    )
    assert math.isclose(
        sum(single["per_chip_stage_v4_floor_ns"].values()),
        sum(dual["per_chip_stage_v4_floor_ns"].values()),
    )
    assert dual["per_chip_hbm_physical_bytes"] * 2 == 2_000
    assert (
        dual["aggregate_hbm_traffic_breakdown"]["by_stage"][
            "layer/attention"
        ]["physical_read_bytes"]
        + dual["aggregate_hbm_traffic_breakdown"]["by_stage"]["layer/ffn"][
            "physical_read_bytes"
        ]
        == 2_000
    )
    assert dual["per_chip_hbm_bandwidth_gbps"] == 2039.0 / 2
    assert dual["per_chip_equivalent_hbm_channels"] == 64
    assert (
        dual["hbm_channel_calibration_status"]
        == "between_channel_anchors_residual_scaled"
    )


def test_tp_only_is_not_faster_than_optimistic_tp_sp() -> None:
    common = {
        "report": _report(),
        "model": MODEL,
        "chip_count": 4,
        "reference_a100_count": 1,
        "aggregate_hbm_bandwidth_gbps": 2039.0,
        "aggregate_hbm_capacity_bytes": 80_000_000_000,
        "seq_len": 482,
        "batch_size": 16,
        "fp_width_bits": 12,
        "one_way_link_bandwidth_gbps": 1e30,
    }
    tp_sp = estimate_multi_chip_latency(parallel_model="tp-sp", **common)
    tp_only = estimate_multi_chip_latency(parallel_model="tp-only", **common)
    assert tp_only["latency_ns"] >= tp_sp["latency_ns"]
    assert (
        tp_only["per_chip_hbm_physical_bytes"]
        >= tp_sp["per_chip_hbm_physical_bytes"]
    )


def test_tp_sp_kv_overlay_uses_local_cache_occurrence_ratio() -> None:
    report = deepcopy(_report())
    attention = report["hbm_traffic_breakdown"]["by_stage_role"].pop(
        "layer/attention::activation"
    )
    report["hbm_traffic_breakdown"]["by_stage_role"][
        "layer/attention::kv"
    ] = attention
    report["hbm_traffic_breakdown"]["by_role"].pop("activation")
    report["hbm_traffic_breakdown"]["by_role"]["kv"] = attention

    result = estimate_multi_chip_latency(
        report,
        MODEL,
        chip_count=2,
        reference_a100_count=1,
        parallel_model="tp-sp",
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=4096,
        batch_size=1,
        fp_width_bits=12,
        one_way_link_bandwidth_gbps=1e30,
        kv_cache_overlay={
            "global_tile_loads": 10,
            "local_tile_loads": 2,
        },
    )
    assert (
        result["per_chip_hbm_traffic"]["layer/attention"][
            "physical_read_bytes"
        ]
        == 200
    )
    assert (
        result["kv_cache_fidelity"]
        == "exact_local_cache_occurrences_under_optimistic_tp_sp"
    )


def test_aggregate_area_and_fp16_handoff_units() -> None:
    area = aggregate_area(
        core_area_mm2=100.0,
        core_area_p10_mm2=90.0,
        core_area_p50_mm2=100.0,
        core_area_p90_mm2=120.0,
        chip_count=4,
        endpoint_overhead_fraction=0.10,
    )
    assert area["endpoint_area_mm2"] == 10.0
    assert math.isclose(area["physical_chip_area_mm2"], 110.0)
    assert math.isclose(area["total_silicon_area_mm2"], 440.0)
    assert math.isclose(area["total_silicon_area_p90_mm2"], 528.0)

    port_area = aggregate_area(
        core_area_mm2=100.0,
        core_area_p10_mm2=90.0,
        core_area_p50_mm2=100.0,
        core_area_p90_mm2=120.0,
        chip_count=4,
        nvlink_port_count=2,
        endpoint_area_mm2_per_port=24.7,
    )
    assert math.isclose(port_area["endpoint_area_mm2"], 49.4)
    assert math.isclose(
        port_area["total_silicon_area_mm2"],
        4 * (100.0 + 49.4),
    )

    handoff = fp16_kv_handoff(
        MODEL,
        seq_len=1,
        batch_size=1,
        one_way_link_bandwidth_gbps=1_800.0,
    )
    expected_bytes = 2 * 64 * 8 * 128 * 2
    assert handoff["fp16_kv_handoff_bytes"] == expected_bytes
    assert math.isclose(
        handoff["fp16_kv_handoff_latency_ns"],
        expected_bytes / 1_800.0,
    )
