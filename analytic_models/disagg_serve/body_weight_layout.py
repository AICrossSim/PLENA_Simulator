"""Physical decoder-body weight layouts for tiled PLENA matrix execution.

The matrix SRAM fetches complete ``MLEN x MLEN`` tiles.  Padding therefore
belongs to each rank-local matrix after the parallel partition is chosen; it
cannot be recovered by padding a global tensor and dividing its byte count.
This module keeps that rule independent from timing so capacity and traffic
models can share one exact layout ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    from .physical_ledger import (
        PlaneBytes,
        WeightLedger,
        bf16_matrix_planes,
        matrix_planes,
    )
except ImportError:
    from physical_ledger import (  # type: ignore[no-redef]
        PlaneBytes,
        WeightLedger,
        bf16_matrix_planes,
        matrix_planes,
    )


BODY_WEIGHT_LAYOUT_SCHEMA = "plena-body-weight-physical-layout/v1"
EXPERT_TENSOR_PARALLEL = "tensor_parallel"
EXPERT_ID_PARALLEL = "expert_id_parallel"
EXPERT_PARALLEL_MODES = frozenset({EXPERT_TENSOR_PARALLEL, EXPERT_ID_PARALLEL})


def _ceil_to(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("dimensions and tile sizes must be positive")
    return math.ceil(value / multiple) * multiple


def _balanced_shards(value: int, shards: int) -> tuple[int, ...]:
    if value <= 0 or shards <= 0:
        raise ValueError("value and shards must be positive")
    quotient, remainder = divmod(value, shards)
    if quotient == 0:
        raise ValueError("parallel degree cannot exceed the partitioned dimension")
    return tuple(quotient + int(index < remainder) for index in range(shards))


def _scale_plane(value: PlaneBytes, copies: int) -> PlaneBytes:
    if copies < 0:
        raise ValueError("copies must be non-negative")
    return PlaneBytes(
        element_raw=value.element_raw * copies,
        element_aligned=value.element_aligned * copies,
        scale_raw=value.scale_raw * copies,
        scale_aligned=value.scale_aligned * copies,
    )


def _sum_planes(values: Sequence[PlaneBytes]) -> PlaneBytes:
    result = PlaneBytes()
    for value in values:
        result += value
    return result


def _plane_dict(value: PlaneBytes) -> dict[str, int]:
    return {
        "element_raw": value.element_raw,
        "element_aligned": value.element_aligned,
        "scale_raw": value.scale_raw,
        "scale_aligned": value.scale_aligned,
        "total_aligned": value.total_aligned,
    }


def _weights_scale(value: WeightLedger, copies: int) -> WeightLedger:
    return WeightLedger(
        attention=_scale_plane(value.attention, copies),
        ffn_resident=_scale_plane(value.ffn_resident, copies),
        ffn_streamed=_scale_plane(value.ffn_streamed, copies),
        lm_head_resident=_scale_plane(value.lm_head_resident, copies),
        lm_head_streamed=_scale_plane(value.lm_head_streamed, copies),
        bf16_embedding=_scale_plane(value.bf16_embedding, copies),
        bf16_norms=_scale_plane(value.bf16_norms, copies),
        bf16_lm_head_resident=_scale_plane(value.bf16_lm_head_resident, copies),
        bf16_lm_head_streamed=_scale_plane(value.bf16_lm_head_streamed, copies),
        bf16_router_resident=_scale_plane(value.bf16_router_resident, copies),
        bf16_router_streamed=_scale_plane(value.bf16_router_streamed, copies),
    )


def _weights_dict(value: WeightLedger) -> dict[str, object]:
    return {
        "attention": _plane_dict(value.attention),
        "experts_resident": _plane_dict(value.ffn_resident),
        "experts_streamed_per_step": _plane_dict(value.ffn_streamed),
        "lm_head_resident": _plane_dict(value.lm_head_resident),
        "lm_head_streamed_per_step": _plane_dict(value.lm_head_streamed),
        "bf16_embedding": _plane_dict(value.bf16_embedding),
        "bf16_norms": _plane_dict(value.bf16_norms),
        "bf16_router_resident": _plane_dict(value.bf16_router_resident),
        "bf16_router_streamed_per_step": _plane_dict(
            value.bf16_router_streamed
        ),
        "resident_total_aligned": value.resident.total_aligned,
        "streamed_total_aligned_per_step": (
            value.streamed_per_batch_step.total_aligned
        ),
    }


def _quant_matrix(
    rows: int,
    columns: int,
    *,
    mlen: int,
    precision: Mapping[str, object],
    role: str,
    instances: int = 1,
    alignment_bytes: int = 64,
) -> tuple[PlaneBytes, dict[str, int]]:
    physical_rows = _ceil_to(rows, mlen)
    physical_columns = _ceil_to(columns, mlen)
    plane = matrix_planes(
        physical_rows,
        physical_columns,
        instances,
        element_bits=int(precision[f"{role}_elem"]),
        effective_bits=float(precision[f"{role}_bits"]),
        block_size=int(precision.get("block_size", 8)),
        alignment_bytes=alignment_bytes,
    )
    return plane, {
        "logical_rows": rows,
        "logical_columns": columns,
        "physical_rows": physical_rows,
        "physical_columns": physical_columns,
        "instances": instances,
        "logical_elements": rows * columns * instances,
        "physical_elements": physical_rows * physical_columns * instances,
    }


@dataclass(frozen=True)
class BodyWeightPhysicalLayout:
    """Rank-local, TP-group, and system weight ledgers for one topology."""

    slowest_rank: WeightLedger
    tensor_parallel_group: WeightLedger
    system: WeightLedger
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": BODY_WEIGHT_LAYOUT_SCHEMA,
            "slowest_rank": _weights_dict(self.slowest_rank),
            "tensor_parallel_group": _weights_dict(self.tensor_parallel_group),
            "system": _weights_dict(self.system),
            "provenance": dict(self.provenance),
        }


def build_body_weight_physical_layout(
    dims: Mapping[str, object],
    precision: Mapping[str, object],
    *,
    mlen: int,
    tp: int,
    kvp: int,
    batch: int,
    unique_experts: int,
    expert_parallel_mode: str = EXPERT_TENSOR_PARALLEL,
    active_experts_per_rank: Sequence[int] | None = None,
    expert_owner_by_id: Sequence[int] | None = None,
    include_lm_head: bool = True,
    alignment_bytes: int = 64,
) -> BodyWeightPhysicalLayout:
    """Build exact physical bytes for the declared parallel mapping.

    Attention uses column-parallel Q/K/V and row-parallel O.  The local head
    is vocabulary-parallel.  Norm, router, and embedding vectors are
    conservatively replicated so a decode token never depends on an omitted
    lookup or normalization collective.  Experts are either dimension-sharded
    like a conventional tensor-parallel MLP or assigned whole by expert ID.

    ``active_experts_per_rank`` is required to call expert-ID streamed traffic
    trace-exact.  Without it, the ledger uses the conservative slowest-rank
    upper bound and records that routing rankability is false. Expert-ID
    ownership defaults to ``expert_id % tp``; ``expert_owner_by_id`` is the
    explicit hook for a future trace-bound placement policy.
    """

    if expert_parallel_mode not in EXPERT_PARALLEL_MODES:
        raise ValueError("unsupported expert parallel mode")
    if min(mlen, tp, kvp, batch, unique_experts, alignment_bytes) <= 0:
        raise ValueError("layout dimensions must be positive")
    if int(precision.get("block_size", 8)) != 8:
        raise ValueError("body layout requires block-8 MX storage")

    hidden = int(dims["hidden"])
    query = int(dims["heads"]) * int(dims["head_dim"])
    kv = int(dims["kv_heads"]) * int(dims["head_dim"])
    inter = int(dims["inter"])
    vocab = int(dims["vocab"])
    layers = int(dims["layers"])
    experts = int(dims.get("num_experts", 1))
    top_k = int(dims.get("experts_per_token", 1))
    if experts <= 1:
        raise ValueError("this physical layout currently requires routed experts")
    if not top_k <= unique_experts <= experts:
        raise ValueError("unique_experts must be between top-k and num_experts")
    if (
        hidden % tp
        or int(dims["heads"]) % tp
        or int(dims["kv_heads"]) % tp
        or query % tp
        or kv % tp
    ):
        raise ValueError(
            "TP must divide hidden size plus query-head and KV-head counts"
        )

    q_widths = _balanced_shards(query, tp)
    kv_widths = _balanced_shards(kv, tp)
    inter_widths = _balanced_shards(inter, tp)
    vocab_widths = _balanced_shards(vocab, tp)

    attention_rank_planes: list[PlaneBytes] = []
    attention_shapes: list[dict[str, object]] = []
    for rank in range(tp):
        q, q_shape = _quant_matrix(
            q_widths[rank], hidden, mlen=mlen, precision=precision,
            role="attn", instances=layers, alignment_bytes=alignment_bytes,
        )
        k, k_shape = _quant_matrix(
            kv_widths[rank], hidden, mlen=mlen, precision=precision,
            role="attn", instances=layers, alignment_bytes=alignment_bytes,
        )
        v, v_shape = _quant_matrix(
            kv_widths[rank], hidden, mlen=mlen, precision=precision,
            role="attn", instances=layers, alignment_bytes=alignment_bytes,
        )
        o, o_shape = _quant_matrix(
            hidden, q_widths[rank], mlen=mlen, precision=precision,
            role="attn", instances=layers, alignment_bytes=alignment_bytes,
        )
        attention_rank_planes.append(q + k + v + o)
        attention_shapes.append(
            {"rank": rank, "q": q_shape, "k": k_shape, "v": v_shape, "o": o_shape}
        )

    expert_shapes: list[dict[str, object]] = []
    expert_resident_rank_planes: list[PlaneBytes] = []
    expert_stream_rank_planes: list[PlaneBytes] = []
    routing_exact = True
    expert_owner_policy = "all_experts_tensor_sharded"
    resolved_expert_owner_by_id: tuple[int, ...] | None = None
    if expert_parallel_mode == EXPERT_TENSOR_PARALLEL:
        if expert_owner_by_id is not None:
            raise ValueError("tensor-parallel experts reject expert-ID ownership")
        for rank in range(tp):
            gate_up, gate_up_shape = _quant_matrix(
                inter_widths[rank], hidden, mlen=mlen, precision=precision,
                role="ffn", instances=2 * layers * experts,
                alignment_bytes=alignment_bytes,
            )
            down, down_shape = _quant_matrix(
                hidden, inter_widths[rank], mlen=mlen, precision=precision,
                role="ffn", instances=layers * experts,
                alignment_bytes=alignment_bytes,
            )
            streamed_gate_up, _ = _quant_matrix(
                inter_widths[rank], hidden, mlen=mlen, precision=precision,
                role="ffn", instances=2 * layers * unique_experts,
                alignment_bytes=alignment_bytes,
            )
            streamed_down, _ = _quant_matrix(
                hidden, inter_widths[rank], mlen=mlen, precision=precision,
                role="ffn", instances=layers * unique_experts,
                alignment_bytes=alignment_bytes,
            )
            expert_resident_rank_planes.append(gate_up + down)
            expert_stream_rank_planes.append(streamed_gate_up + streamed_down)
            expert_shapes.append(
                {
                    "rank": rank,
                    "owned_experts": experts,
                    "active_experts": unique_experts,
                    "gate_up": gate_up_shape,
                    "down": down_shape,
                }
            )
    else:
        if expert_owner_by_id is None:
            resolved_expert_owner_by_id = tuple(
                expert_id % tp for expert_id in range(experts)
            )
            expert_owner_policy = "expert_id_mod_tp_rank"
        else:
            resolved_expert_owner_by_id = tuple(
                int(owner) for owner in expert_owner_by_id
            )
            if (
                len(resolved_expert_owner_by_id) != experts
                or any(
                    owner < 0 or owner >= tp
                    for owner in resolved_expert_owner_by_id
                )
            ):
                raise ValueError("expert_owner_by_id violates TP ownership")
            expert_owner_policy = "explicit_expert_owner_by_id"
        owned_counts = tuple(
            resolved_expert_owner_by_id.count(rank) for rank in range(tp)
        )
        if any(count <= 0 for count in owned_counts):
            raise ValueError("every expert-ID rank must own at least one expert")
        if active_experts_per_rank is None:
            routing_exact = False
            active_counts = tuple(min(unique_experts, count) for count in owned_counts)
        else:
            active_counts = tuple(int(value) for value in active_experts_per_rank)
            if (
                len(active_counts) != tp
                or any(value < 0 for value in active_counts)
                or sum(active_counts) != unique_experts
                or any(value > owned for value, owned in zip(active_counts, owned_counts))
            ):
                raise ValueError("active_experts_per_rank violates expert ownership")
        for rank in range(tp):
            gate_up, gate_up_shape = _quant_matrix(
                inter, hidden, mlen=mlen, precision=precision,
                role="ffn", instances=2 * layers * owned_counts[rank],
                alignment_bytes=alignment_bytes,
            )
            down, down_shape = _quant_matrix(
                hidden, inter, mlen=mlen, precision=precision,
                role="ffn", instances=layers * owned_counts[rank],
                alignment_bytes=alignment_bytes,
            )
            streamed_gate_up, _ = _quant_matrix(
                inter, hidden, mlen=mlen, precision=precision,
                role="ffn", instances=2 * layers * active_counts[rank],
                alignment_bytes=alignment_bytes,
            ) if active_counts[rank] else (PlaneBytes(), {})
            streamed_down, _ = _quant_matrix(
                hidden, inter, mlen=mlen, precision=precision,
                role="ffn", instances=layers * active_counts[rank],
                alignment_bytes=alignment_bytes,
            ) if active_counts[rank] else (PlaneBytes(), {})
            expert_resident_rank_planes.append(gate_up + down)
            expert_stream_rank_planes.append(streamed_gate_up + streamed_down)
            expert_shapes.append(
                {
                    "rank": rank,
                    "owned_experts": owned_counts[rank],
                    "active_experts": active_counts[rank],
                    "gate_up": gate_up_shape,
                    "down": down_shape,
                }
            )

    if not (
        len(attention_rank_planes)
        == len(expert_resident_rank_planes)
        == len(expert_stream_rank_planes)
        == len(expert_shapes)
        == tp
    ):
        raise AssertionError("body rank-plane construction must conserve TP ranks")

    head_rank_planes: list[PlaneBytes] = []
    head_shapes: list[dict[str, object]] = []
    if include_lm_head:
        if not bool(precision.get("lm_head_quantized", False)):
            raise ValueError("canonical local head must use the profile MX format")
        for rank in range(tp):
            plane, shape = _quant_matrix(
                vocab_widths[rank], hidden, mlen=mlen, precision=precision,
                role="head", alignment_bytes=alignment_bytes,
            )
            head_rank_planes.append(plane)
            head_shapes.append({"rank": rank, **shape})
    else:
        head_rank_planes = [PlaneBytes() for _ in range(tp)]
    if len(head_rank_planes) != tp or (
        include_lm_head and len(head_shapes) != tp
    ):
        raise AssertionError("local-head rank-plane construction must conserve TP ranks")

    embedding_one = bf16_matrix_planes(
        vocab, hidden, alignment_bytes=alignment_bytes
    )
    norms_one = bf16_matrix_planes(
        1, hidden, 2 * layers + 1, alignment_bytes=alignment_bytes
    )
    if bool(dims.get("qk_norm", False)):
        norms_one += bf16_matrix_planes(
            1, int(dims["head_dim"]), 2 * layers,
            alignment_bytes=alignment_bytes,
        )
    router_one = bf16_matrix_planes(
        experts, hidden, layers, alignment_bytes=alignment_bytes
    )

    rank_ledgers: list[WeightLedger] = []
    for rank in range(tp):
        rank_ledgers.append(
            WeightLedger(
                attention=attention_rank_planes[rank],
                ffn_resident=expert_resident_rank_planes[rank],
                ffn_streamed=expert_stream_rank_planes[rank],
                lm_head_resident=head_rank_planes[rank],
                lm_head_streamed=head_rank_planes[rank],
                bf16_embedding=embedding_one,
                bf16_norms=norms_one,
                bf16_lm_head_resident=PlaneBytes(),
                bf16_lm_head_streamed=PlaneBytes(),
                bf16_router_resident=router_one,
                bf16_router_streamed=router_one,
            )
        )

    group = WeightLedger(
        attention=_sum_planes(attention_rank_planes),
        ffn_resident=_sum_planes(expert_resident_rank_planes),
        ffn_streamed=_sum_planes(expert_stream_rank_planes),
        lm_head_resident=_sum_planes(head_rank_planes),
        lm_head_streamed=_sum_planes(head_rank_planes),
        bf16_embedding=_scale_plane(embedding_one, tp),
        bf16_norms=_scale_plane(norms_one, tp),
        bf16_lm_head_resident=PlaneBytes(),
        bf16_lm_head_streamed=PlaneBytes(),
        bf16_router_resident=_scale_plane(router_one, tp),
        bf16_router_streamed=_scale_plane(router_one, tp),
    )
    # Different components can peak on different ranks only for a trace-derived
    # expert-ID assignment.  All non-expert tensors are symmetric for the
    # target's divisible dimensions, so selecting by streamed bytes is exact.
    slowest = max(
        rank_ledgers,
        key=lambda value: value.streamed_per_batch_step.total_aligned,
    )
    system = _weights_scale(group, kvp)

    logical_attention_elements = layers * (
        query * hidden + 2 * kv * hidden + hidden * query
    )
    logical_expert_resident_elements = layers * experts * 3 * hidden * inter
    logical_expert_streamed_elements = layers * unique_experts * 3 * hidden * inter
    logical_head_elements = vocab * hidden if include_lm_head else 0
    physical_attention_elements = sum(
        sum(
            int(shape[name]["physical_elements"])
            for name in ("q", "k", "v", "o")
        )
        for shape in attention_shapes
    )
    physical_expert_resident_elements = sum(
        int(shape["gate_up"]["physical_elements"])
        + int(shape["down"]["physical_elements"])
        for shape in expert_shapes
    )
    physical_expert_streamed_elements = sum(
        int(shape["active_experts"])
        * (
            int(shape["gate_up"]["physical_rows"])
            * int(shape["gate_up"]["physical_columns"])
            * 2
            + int(shape["down"]["physical_rows"])
            * int(shape["down"]["physical_columns"])
        )
        * layers
        for shape in expert_shapes
    )
    physical_head_elements = sum(
        int(shape["physical_elements"]) for shape in head_shapes
    )
    provenance = {
        "schema_version": BODY_WEIGHT_LAYOUT_SCHEMA,
        "mlen": mlen,
        "tp": tp,
        "kvp": kvp,
        "chip_count": tp * kvp,
        "batch": batch,
        "experts": experts,
        "experts_per_token": top_k,
        "route_assignments_per_layer": batch * top_k,
        "unique_experts_per_layer": unique_experts,
        "expert_parallel_mode": expert_parallel_mode,
        "expert_owner_policy": expert_owner_policy,
        "expert_owner_by_id": (
            list(resolved_expert_owner_by_id)
            if resolved_expert_owner_by_id is not None
            else None
        ),
        "padding_order": "partition_rank_local_then_pad_each_matrix_to_mlen",
        "attention_partition": "column_qkv_row_o",
        "local_head_partition": "vocabulary_parallel",
        "embedding_partition": "replicated",
        "norm_partition": "replicated",
        "router_partition": "replicated_bf16",
        "expert_routing_mapping": (
            "replicated_hidden_local_route_filter_then_output_allreduce"
            if expert_parallel_mode == EXPERT_ID_PARALLEL
            else "replicated_router_tensor_sharded_experts_then_output_allreduce"
        ),
        "expert_route_filter_collective_required": False,
        "expert_output_collective_required": tp > 1,
        "expert_route_assignment_exact": routing_exact,
        "precision": {
            role: {
                "element_bits": int(precision[f"{role}_elem"]),
                "effective_bits": float(precision[f"{role}_bits"]),
            }
            for role in ("attn", "ffn", "head")
        },
        "compiler_layout_valid": False,
        "rtl_layout_valid": False,
        "analytic_layout_valid": True,
        "publication_rankable": False,
        "selection_eligible": False,
        "blockers": [
            "rank-local body padding is not yet wired into full-model compiler/emulator evidence",
            *(
                []
                if routing_exact
                else ["expert-ID streamed traffic lacks a measured per-rank route assignment"]
            ),
        ],
        "logical_elements": {
            "attention_per_tp_group": logical_attention_elements,
            "experts_resident_per_tp_group": logical_expert_resident_elements,
            "experts_streamed_per_step_per_tp_group": logical_expert_streamed_elements,
            "local_head_per_tp_group": logical_head_elements,
        },
        "physical_elements": {
            "attention_per_tp_group": physical_attention_elements,
            "experts_resident_per_tp_group": physical_expert_resident_elements,
            "experts_streamed_per_step_per_tp_group": (
                physical_expert_streamed_elements
            ),
            "local_head_per_tp_group": physical_head_elements,
        },
        "padding_inflation": {
            "attention_physical_over_logical": (
                physical_attention_elements / logical_attention_elements
            ),
            "experts_resident_physical_over_logical": (
                physical_expert_resident_elements
                / logical_expert_resident_elements
            ),
            "experts_streamed_physical_over_logical": (
                physical_expert_streamed_elements
                / logical_expert_streamed_elements
            ),
            "local_head_physical_over_logical": (
                physical_head_elements / logical_head_elements
                if logical_head_elements
                else None
            ),
            "tensor_parallel_group_resident_bytes_over_slowest_rank": (
                group.resident.total_aligned
                / max(slowest.resident.total_aligned, 1)
            ),
            "system_resident_bytes_over_tp_group": (
                system.resident.total_aligned
                / max(group.resident.total_aligned, 1)
            ),
        },
        "streamed_expert_group_scope": (
            "trace_exact_owned_active_experts"
            if routing_exact
            else "sum_of_conservative_per_rank_active_expert_upper_bounds"
        ),
        "rank_shapes": {
            "attention": attention_shapes,
            "experts": expert_shapes,
            "local_head": head_shapes,
        },
    }
    return BodyWeightPhysicalLayout(
        slowest_rank=slowest,
        tensor_parallel_group=group,
        system=system,
        provenance=provenance,
    )


__all__ = [
    "BODY_WEIGHT_LAYOUT_SCHEMA",
    "EXPERT_ID_PARALLEL",
    "EXPERT_PARALLEL_MODES",
    "EXPERT_TENSOR_PARALLEL",
    "BodyWeightPhysicalLayout",
    "build_body_weight_physical_layout",
]
