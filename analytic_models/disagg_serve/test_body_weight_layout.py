from __future__ import annotations

import pytest

from .body_weight_layout import (
    EXPERT_ID_PARALLEL,
    EXPERT_TENSOR_PARALLEL,
    build_body_weight_physical_layout,
)
from .physical_ledger import KVLedger, PhysicalDecodeLedger, SRAMLedger


TARGET_DIMS = {
    "hidden": 2048,
    "heads": 32,
    "kv_heads": 4,
    "head_dim": 128,
    "inter": 768,
    "vocab": 151936,
    "layers": 48,
    "num_experts": 128,
    "experts_per_token": 8,
    "qk_norm": True,
}

I4 = {
    "block_size": 8,
    "attn_elem": 4,
    "attn_bits": 5.0,
    "ffn_elem": 4,
    "ffn_bits": 5.0,
    "head_elem": 4,
    "head_bits": 5.0,
    "lm_head_quantized": True,
}


def test_target_tp1_padding_counts_narrow_experts_and_kv_projections() -> None:
    layout = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=1,
        kvp=1,
        batch=8,
        unique_experts=32,
    )
    provenance = layout.provenance
    logical = provenance["logical_elements"]
    physical = provenance["physical_elements"]

    assert logical["attention_per_tp_group"] == 905_969_664
    assert physical["attention_per_tp_group"] == 1_006_632_960
    assert logical["experts_resident_per_tp_group"] == 28_991_029_248
    assert physical["experts_resident_per_tp_group"] == 38_654_705_664
    assert physical["local_head_per_tp_group"] == 152_576 * 2_048
    assert provenance["padding_order"] == (
        "partition_rank_local_then_pad_each_matrix_to_mlen"
    )
    assert provenance["publication_rankable"] is False
    assert provenance["route_assignments_per_layer"] == 8 * 8


def test_tensor_parallel_padding_is_not_global_padding_divided_by_tp() -> None:
    one = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=1,
        kvp=1,
        batch=8,
        unique_experts=32,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
    )
    four = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=1,
        batch=8,
        unique_experts=32,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
    )

    one_physical = one.provenance["physical_elements"]
    four_physical = four.provenance["physical_elements"]
    assert one_physical["attention_per_tp_group"] == 1_006_632_960
    assert four_physical["attention_per_tp_group"] == 1_610_612_736
    assert four_physical["experts_resident_per_tp_group"] == (
        4 * one_physical["experts_resident_per_tp_group"]
    )
    assert four_physical["local_head_per_tp_group"] == 4 * 38_912 * 2_048
    assert (
        four.tensor_parallel_group.ffn_resident.total_aligned
        > one.tensor_parallel_group.ffn_resident.total_aligned
    )


def test_tensor_rank_planes_and_system_replicas_conserve_physical_bytes() -> None:
    layout = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=2,
        batch=8,
        unique_experts=32,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
    )

    for name in (
        "attention",
        "ffn_resident",
        "ffn_streamed",
        "lm_head_resident",
        "lm_head_streamed",
        "bf16_embedding",
        "bf16_norms",
        "bf16_router_resident",
        "bf16_router_streamed",
    ):
        rank = getattr(layout.slowest_rank, name).total_aligned
        group = getattr(layout.tensor_parallel_group, name).total_aligned
        system = getattr(layout.system, name).total_aligned
        assert group == 4 * rank
        assert system == 2 * group
    assert len(layout.provenance["rank_shapes"]["attention"]) == 4
    assert len(layout.provenance["rank_shapes"]["experts"]) == 4
    assert len(layout.provenance["rank_shapes"]["local_head"]) == 4


def test_expert_id_parallel_avoids_replicating_narrow_tile_padding() -> None:
    tensor = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=1,
        batch=16,
        unique_experts=16,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
    )
    expert = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=1,
        batch=16,
        unique_experts=16,
        expert_parallel_mode=EXPERT_ID_PARALLEL,
        active_experts_per_rank=(4, 4, 4, 4),
    )

    assert expert.provenance["physical_elements"][
        "experts_resident_per_tp_group"
    ] == tensor.provenance["physical_elements"][
        "experts_resident_per_tp_group"
    ] // 4
    assert expert.provenance["expert_route_assignment_exact"] is True
    assert expert.provenance["expert_owner_policy"] == "expert_id_mod_tp_rank"
    assert expert.provenance["expert_owner_by_id"][:8] == [0, 1, 2, 3] * 2
    assert all(
        value["owned_experts"] == 32
        for value in expert.provenance["rank_shapes"]["experts"]
    )


def test_expert_id_streaming_without_per_rank_trace_stays_unrankable() -> None:
    layout = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=2,
        batch=8,
        unique_experts=12,
        expert_parallel_mode=EXPERT_ID_PARALLEL,
    )
    assert layout.provenance["expert_route_assignment_exact"] is False
    assert "measured per-rank route assignment" in " ".join(
        layout.provenance["blockers"]
    )
    assert layout.system.resident.total_aligned == (
        2 * layout.tensor_parallel_group.resident.total_aligned
    )
    assert layout.provenance["expert_routing_mapping"] == (
        "replicated_hidden_local_route_filter_then_output_allreduce"
    )
    assert layout.provenance["expert_route_filter_collective_required"] is False
    assert layout.provenance["expert_output_collective_required"] is True


def test_expert_id_trace_must_conserve_unique_experts() -> None:
    with pytest.raises(ValueError, match="violates expert ownership"):
        build_body_weight_physical_layout(
            TARGET_DIMS,
            I4,
            mlen=1024,
            tp=4,
            kvp=1,
            batch=8,
            unique_experts=12,
            expert_parallel_mode=EXPERT_ID_PARALLEL,
            active_experts_per_rank=(3, 3, 3, 2),
        )


def test_mlen4096_exposes_physical_capacity_cost_instead_of_hiding_it() -> None:
    low = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=1,
        kvp=1,
        batch=1,
        unique_experts=8,
    )
    high = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=4096,
        tp=1,
        kvp=1,
        batch=1,
        unique_experts=8,
    )
    assert (
        high.tensor_parallel_group.ffn_resident.total_aligned
        == 8 * low.tensor_parallel_group.ffn_resident.total_aligned
    )
    assert (
        high.tensor_parallel_group.attention.total_aligned
        > low.tensor_parallel_group.attention.total_aligned
    )


def test_tp_must_partition_whole_q_and_kv_heads() -> None:
    with pytest.raises(ValueError, match="query-head and KV-head"):
        build_body_weight_physical_layout(
            TARGET_DIMS,
            I4,
            mlen=1024,
            tp=8,
            kvp=1,
            batch=8,
            unique_experts=16,
        )


def test_slowest_rank_hbm_failure_overrides_aggregate_slack() -> None:
    layout = build_body_weight_physical_layout(
        TARGET_DIMS,
        I4,
        mlen=1024,
        tp=4,
        kvp=1,
        batch=1,
        unique_experts=8,
    )
    sram = SRAMLedger(
        vector_capacity_bytes=1,
        vector_bytes_per_sequence=1,
        vector_required_bytes=1,
        matrix_capacity_bytes=1,
        matrix_required_bytes=1,
        matrix_tile_capacity=1,
        matrix_required_tiles=1,
        max_vector_batch=1,
        max_synchronous_batch=1,
        output_head_logit_tile_bytes=0,
        output_head_selection_state_bytes=0,
        output_head_workspace_bytes=0,
    )
    ledger = PhysicalDecodeLedger(
        weights=layout.system,
        kv=KVLedger(0, 0, 0, 0, "test"),
        sram=sram,
        hbm_capacity_bytes=400,
        runtime_hbm_reserve_bytes=0,
        hbm_required_bytes=300,
        max_resident_batch=1,
        max_runtime_batch=1,
        kv_layout="test",
        slowest_rank_hbm_required_bytes=101,
        per_chip_hbm_capacity_bytes=100,
    )

    assert ledger.hbm_required_bytes < ledger.hbm_capacity_bytes
    assert ledger.fits_hbm is False
    assert ledger.fits_runtime is False
