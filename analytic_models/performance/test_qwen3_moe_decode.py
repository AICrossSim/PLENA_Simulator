"""Contracts for Qwen3 routed-MoE decode accounting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ANALYTIC = Path(__file__).resolve().parents[1]
for _name in ("performance", "memory", "disagg_serve"):
    _path = str(_ANALYTIC / _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import disagg_decode as decode  # noqa: E402
from memory_model import (  # noqa: E402
    conservative_unique_experts,
    expected_unique_experts,
)
from perf_model import (  # noqa: E402
    PerfModel,
    load_hardware_config_from_toml,
    routed_expert_decode_ledger,
)
from physical_ledger import matrix_planes, weight_ledger  # noqa: E402


def _target_config() -> dict:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }


def _precision() -> dict:
    return {
        "attn_elem": 4,
        "attn_bits": 5.0,
        "ffn_elem": 4,
        "ffn_bits": 5.0,
        "kv_elem": 4,
        "kv_bits": 5.0,
        "block_size": 8,
    }


def test_official_qwen3_moe_fields_are_loaded_without_dense_fallback(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_target_config()), encoding="utf-8")

    dims = decode.load_model_dims(str(path))

    assert dims["model_type"] == "qwen3_moe"
    assert dims["num_experts"] == 128
    assert dims["experts_per_token"] == 8
    assert dims["inter"] == 768
    assert dims["dense_inter"] == 6144
    assert dims["qk_norm"] is True
    assert dims["router_weight_bits"] == 16
    assert dims["norm_topk_prob"] is True
    assert dims["moe_route_repricing"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_shared_experts", 1),
        ("shared_expert_intermediate_size", 768),
    ],
)
def test_shared_expert_fields_fail_closed(tmp_path, field, value):
    config = _target_config() | {field: value}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="shared experts"):
        decode.load_model_dims(str(path))


def test_unique_expert_streaming_grows_with_decode_batch():
    assert expected_unique_experts(128, 8, 1) == pytest.approx(8.0)
    assert conservative_unique_experts(128, 8, 1) == 8
    assert conservative_unique_experts(128, 8, 16) > 8
    assert conservative_unique_experts(128, 8, 1024) == 128

    dims = {
        "hidden": 2048,
        "heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "layers": 2,
        "inter": 768,
        "vocab": 1024,
        "num_experts": 128,
        "experts_per_token": 8,
        "qk_norm": True,
    }
    batch_one = weight_ledger(dims, _precision(), batch=1, mlen=4)
    batch_sixteen = weight_ledger(dims, _precision(), batch=16, mlen=4)

    assert batch_one.resident == batch_sixteen.resident
    assert (
        batch_sixteen.ffn_streamed.total_aligned
        > batch_one.ffn_streamed.total_aligned
    )
    assert batch_one.bf16_router_streamed.total_aligned > 0
    assert (
        batch_one.bf16_router_streamed
        == batch_sixteen.bf16_router_streamed
    )
    partitioned = decode._partition_weight_ledger(
        batch_sixteen,
        tp=2,
        kvp=2,
        sram_policy="streaming",
    )
    assert partitioned.bf16_router_resident.total_aligned > 0
    assert partitioned.bf16_router_streamed.total_aligned > 0


def test_dense_streamed_weights_remain_batch_invariant():
    dims = {
        "hidden": 128,
        "heads": 4,
        "kv_heads": 2,
        "head_dim": 32,
        "layers": 2,
        "inter": 256,
        "vocab": 1024,
        "num_experts": 1,
        "experts_per_token": 1,
    }
    one = weight_ledger(dims, _precision(), batch=1, mlen=4)
    many = weight_ledger(dims, _precision(), batch=64, mlen=4)
    assert one == many
    assert one.bf16_router_resident.total_aligned == 0


def test_local_head_hbm_planes_use_candidate_mlen_padded_shape():
    dims = {
        "hidden": 2048,
        "heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "layers": 2,
        "inter": 768,
        "vocab": 151_936,
        "num_experts": 128,
        "experts_per_token": 8,
    }
    precision = {
        **_precision(),
        "head_elem": 4,
        "head_bits": 5.0,
        "lm_head_quantized": True,
    }
    ledger_1024 = weight_ledger(
        dims,
        precision,
        batch=1,
        mlen=1024,
    )
    ledger_4096 = weight_ledger(
        dims,
        precision,
        batch=1,
        mlen=4096,
    )
    expected_1024 = matrix_planes(
        152_576,
        2_048,
        1,
        element_bits=4,
        effective_bits=5.0,
        block_size=8,
    )
    expected_4096 = matrix_planes(
        155_648,
        4_096,
        1,
        element_bits=4,
        effective_bits=5.0,
        block_size=8,
    )

    assert ledger_1024.lm_head_resident == expected_1024
    assert ledger_1024.lm_head_streamed == expected_1024
    assert ledger_4096.lm_head_resident == expected_4096
    assert ledger_4096.lm_head_resident.total_aligned > (
        ledger_1024.lm_head_resident.total_aligned
    )


def test_local_head_padding_preparation_is_charged_in_decode_cycles(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_target_config()), encoding="utf-8")
    dims = decode.load_model_dims(str(path))
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    components = decode.decode_token_components(
        perf,
        dims,
        kv=128,
        batch=3,
        include_lm_head=True,
    )
    padding = perf.lm_head_padding_preparation(
        dims["hidden"],
        dims["vocab"],
        3,
    )

    assert components["LM head activation padding zero-fill"] == (
        padding["zero_fill_cycles_per_rank"]
    )
    assert components["LM head padded-vocab mask"] == (
        padding["padded_vocab_mask_cycles_aggregate"]
    )
    assert decode.decode_token_cycles(
        perf,
        dims,
        kv=128,
        batch=3,
        include_lm_head=True,
    ) == sum(components.values())

    partitioned_components = decode._partitioned_component_cycles(
        perf,
        dims,
        kv=128,
        batch=3,
        tp=2,
        kvp=1,
        include_lm_head=True,
        kv_layout=decode.DENSE_SELECTOR,
        packed_q1_timing_contract=None,
        batch_packed_attention=False,
        kv_head_reuse=None,
    )
    physical_hidden = (
        (dims["hidden"] + perf.mlen - 1) // perf.mlen * perf.mlen
    )
    physical_batch = (3 + perf.blen - 1) // perf.blen * perf.blen
    local_vocab = dims["vocab"] // 2
    local_vocab_tail = (
        (local_vocab + perf.mlen - 1) // perf.mlen * perf.mlen
        - local_vocab
    )
    activation_tail = (
        3 * (physical_hidden - dims["hidden"])
        + (physical_batch - 3) * physical_hidden
    )
    expected_local_padding = (
        (activation_tail + perf.vlen - 1) // perf.vlen
        + (3 * local_vocab_tail + perf.vlen - 1) // perf.vlen
    ) * perf.instr["V_BASIC"]

    assert partitioned_components["replicated_rmsnorm"] == (
        perf.rms_layer(dims["hidden"], 1, 3, "decode")
        * (2 * dims["layers"] + 1)
    )
    assert partitioned_components["replicated_residual"] == (
        perf.residual(dims["hidden"], 1, 3, "decode")
        * 2
        * dims["layers"]
    )
    assert partitioned_components["rank_local_head_padding"] == (
        expected_local_padding
    )
    assert decode._partitioned_components(
        perf,
        dims,
        kv=128,
        batch=3,
        tp=2,
        kvp=1,
        include_lm_head=True,
        kv_layout=decode.DENSE_SELECTOR,
        packed_q1_timing_contract=None,
        batch_packed_attention=False,
        kv_head_reuse=None,
    ) == sum(partitioned_components.values())


def test_operation_shaped_tp_keeps_replicated_moe_work_and_narrow_tiles(
    tmp_path,
) -> None:
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_target_config()), encoding="utf-8")
    dims = decode.load_model_dims(str(path))
    common = {
        "perf": perf,
        "d": dims,
        "kv": 128,
        "batch": 8,
        "kvp": 1,
        "include_lm_head": True,
        "kv_layout": decode.DENSE_SELECTOR,
        "packed_q1_timing_contract": None,
        "batch_packed_attention": False,
        "kv_head_reuse": None,
    }
    one = decode._partitioned_component_cycles(tp=1, **common)
    four = decode._partitioned_component_cycles(tp=4, **common)
    expert_id = decode._partitioned_component_cycles(
        tp=4,
        expert_parallel_mode=decode.EXPERT_ID_PARALLEL,
        **common,
    )

    for name in (
        "replicated_rmsnorm",
        "replicated_residual",
        "replicated_embedding_lookup",
        "replicated_router_topk_softmax_combine",
    ):
        assert four[name] == one[name]
        assert expert_id[name] == one[name]
    assert one["rank_local_qk_rmsnorm"] > 0
    assert 0 < four["rank_local_qk_rmsnorm"] < one["rank_local_qk_rmsnorm"]
    assert expert_id["rank_local_qk_rmsnorm"] == four[
        "rank_local_qk_rmsnorm"
    ]
    assert four["rank_local_routed_experts"] > (
        one["rank_local_routed_experts"] / 4
    )
    assert expert_id["rank_local_routed_experts"] == (
        one["rank_local_routed_experts"]
    )


def test_expert_id_uses_replicated_route_filter_and_charged_output_allreduce(
    tmp_path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_target_config()), encoding="utf-8")
    dims = decode.load_model_dims(str(path))
    tensor = decode.collective_cost_per_step(
        dims,
        batch=8,
        tp=4,
        kvp=2,
        link_ports=2,
        expert_parallel_mode=decode.EXPERT_TENSOR_PARALLEL,
    )
    expert_id = decode.collective_cost_per_step(
        dims,
        batch=8,
        tp=4,
        kvp=2,
        link_ports=2,
        expert_parallel_mode=decode.EXPERT_ID_PARALLEL,
    )

    assert expert_id["expert_routing_bytes"] == 0.0
    assert expert_id["expert_routing_time_s"] == 0.0
    assert expert_id["expert_output_collective_slowest_rank_bytes"] == (
        expert_id["tp_bytes"] / 2.0
    )
    assert expert_id["expert_output_collective_system_bytes"] == (
        expert_id["tp_bytes"] / 2.0 * 8
    )
    assert expert_id["expert_output_collective_time_s"] > 0.0
    assert expert_id["total_bytes"] == tensor["total_bytes"]
    assert expert_id["time_s"] == tensor["time_s"]


def test_moe_timing_includes_routing_events_and_imbalance():
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    dims = {
        "hidden": 2048,
        "inter": 768,
        "num_experts": 128,
        "experts_per_token": 8,
        "moe_routing_imbalance_factor": 1.0,
    }
    label, balanced = decode._ffn_label_cycles(perf, dims, 8)
    dims["moe_routing_imbalance_factor"] = 1.25
    _, imbalanced = decode._ffn_label_cycles(perf, dims, 8)
    route_accounting = decode._moe_route_accounting(dims, 8)

    assert "router+top-k" in label
    assert imbalanced > balanced
    assert route_accounting["physical_route_assignments_per_step"] == 8 * 8
    assert route_accounting["routes_per_step"] == 8 * 8
    assert route_accounting["route_assignment_accounting"] == (
        "conserved_batch_times_topk"
    )
    assert route_accounting["routing_imbalance_application"].endswith(
        "cycle_penalty_only"
    )


def test_ragged_expert_ledger_conserves_target_routes_and_batch_monotonicity():
    ledgers = [
        routed_expert_decode_ledger(
            batch_size=batch,
            expert_per_token=8,
            num_experts=128,
            blen=4,
        )
        for batch in (1, 2, 4, 8, 16, 32, 64, 128)
    ]

    for batch, ledger in zip((1, 2, 4, 8, 16, 32, 64, 128), ledgers):
        assert ledger["route_assignments"] == batch * 8
        assert ledger["active_experts"] == conservative_unique_experts(
            128,
            8,
            batch,
        )
        assert ledger["assignments_conserved"] is True
        assert ledger["expert_padded_rows"] == ledger["expert_row_tiles"] * 4
        assert ledger["expert_padding_rows"] == (
            ledger["expert_padded_rows"] - batch * 8
        )
        assert sum(
            int(token_count) * expert_count
            for token_count, expert_count in ledger[
                "expert_token_count_histogram"
            ].items()
        ) == batch * 8

    assert ledgers[0]["active_experts"] == 8
    assert ledgers[0]["expert_row_tiles"] == 8
    assert ledgers[0]["expert_padded_rows"] == 32
    assert [ledger["expert_row_tiles"] for ledger in ledgers] == sorted(
        ledger["expert_row_tiles"] for ledger in ledgers
    )


def test_ragged_expert_ledger_blen_and_safe_override_contracts():
    by_blen = [
        routed_expert_decode_ledger(
            batch_size=64,
            expert_per_token=8,
            num_experts=128,
            blen=blen,
        )
        for blen in (2, 4, 8)
    ]
    assert [ledger["expert_row_tiles"] for ledger in by_blen] == sorted(
        (ledger["expert_row_tiles"] for ledger in by_blen),
        reverse=True,
    )
    assert [ledger["expert_padded_rows"] for ledger in by_blen] == sorted(
        ledger["expert_padded_rows"] for ledger in by_blen
    )

    traced = routed_expert_decode_ledger(
        batch_size=8,
        expert_per_token=8,
        num_experts=128,
        blen=4,
        active_experts=16,
        active_expert_source="audited_route_trace",
    )
    assert traced["active_experts"] == 16
    assert traced["active_expert_source"] == "audited_route_trace"
    assert traced["route_assignments"] == 64
    assert traced["expert_row_tiles"] == 16

    with pytest.raises(ValueError, match="active_experts must be between"):
        routed_expert_decode_ledger(
            batch_size=1,
            expert_per_token=8,
            num_experts=128,
            blen=4,
            active_experts=9,
        )


def test_ragged_expert_timing_charges_mm_drains_auxiliary_and_cycle_only_skew():
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    balanced = perf.moe_decode_expert_timing(
        2048,
        8,
        128,
        8,
        768,
    )
    imbalanced = perf.moe_decode_expert_timing(
        2048,
        8,
        128,
        8,
        768,
        routing_imbalance_factor=1.25,
    )

    assert balanced["matrix_instruction_histogram"]["M_MM"] > 0
    assert balanced["matrix_instruction_histogram"]["M_MM_WO"] > 0
    assert balanced["auxiliary_instruction_histogram"]["H_PREFETCH_M"] > 0
    assert balanced["auxiliary_instruction_histogram"]["S_ADDI_INT"] > 0
    assert balanced["expert_stage_base_cycles"] == (
        balanced["matrix_cycles"]
        + balanced["activation_cycles"]
        + balanced["auxiliary_cycles"]
    )
    for field in (
        "route_assignments",
        "active_experts",
        "expert_row_tiles",
        "expert_padded_rows",
        "expert_padding_rows",
        "matrix_instruction_histogram",
        "auxiliary_instruction_histogram",
    ):
        assert imbalanced[field] == balanced[field]
    assert imbalanced["expert_stage_cycles"] > balanced["expert_stage_cycles"]
    assert imbalanced["routing_imbalance_penalty_cycles"] > 0


def test_moe_workload_exposes_ragged_ledger_and_remains_unrankable():
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    dims = {
        "hidden": 2048,
        "inter": 768,
        "num_experts": 128,
        "experts_per_token": 8,
        "router_weight_bits": 16,
        "moe_routing_imbalance_factor": 1.0,
        "moe_unique_experts_per_step": None,
    }
    workload = decode._moe_workload_accounting(perf, dims, 8)

    assert workload["physical_route_assignments_per_step"] == 64
    assert workload["expert_row_tiles_per_layer"] == 52
    assert workload["expert_padded_rows_per_layer"] == 208
    assert workload["expert_padding_rows_per_layer"] == 144
    assert workload["expert_batch_ledger"]["assignments_conserved"] is True
    assert workload["expert_batch_ledger"]["matrix_instruction_histogram"][
        "M_MM_WO"
    ] > 0
    assert workload["provenance"]["expert_matrix_drain_cycles_included"] is True
    assert workload["provenance"]["expert_auxiliary_issue_cycles_included"] is True
    assert workload["provenance"]["publication_rankable"] is False


def test_dense_ffn_label_remains_the_native_dense_schedule():
    root = _ANALYTIC.parent
    hardware = load_hardware_config_from_toml(str(root / "plena_settings.toml"))
    perf = PerfModel(
        hardware,
        str(_ANALYTIC / "performance" / "customISA_lib.json"),
    )
    dims = {
        "hidden": 2048,
        "inter": 6144,
        "num_experts": 1,
        "experts_per_token": 1,
    }
    label, cycles = decode._ffn_label_cycles(perf, dims, 8)
    assert label == "FFN (gate/up/down)"
    assert cycles == perf.feed_forward(2048, 6144, 1, 8, "decode")
