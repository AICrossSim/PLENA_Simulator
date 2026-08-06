"""Tile-aware request-, tensor-, and expert-parallel inference model.

The native compiler remains single-chip.  This module rebuilds the physical
work of each analytical rank from compiler-emitted kernel lineage, while
assigning whole requests to data/expert origins.  It intentionally keeps the
older context-parallel implementation in :mod:`multi_chip_model` untouched as
an A/B compatibility model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from compiler.aten.plena.native_layout import SequencePackingPlan
from compiler.aten.cost_emitter import parallel_kernel_lineage_id

from .multi_chip_model import (
    DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS,
    DEFAULT_NVLINK_STARTUP_US,
    KV_TRAFFIC_ROLES,
    TRAFFIC_FIELDS,
    _FFN_ACTIVATION_OPS,
    _SOFTMAX_ROW_GROUP_OPS,
    _aggregate_traffic_breakdown,
    _attention_block_work,
    _attention_projection_counts,
    _average_traffic_breakdown,
    _balanced_part,
    _ffn_projection_counts,
    _head_storage_width,
    _memory_latency_for_traffic,
    _parallel_kernel_entries,
    _projection_counts,
    _rank_opcode_scale,
    _scale_traffic_breakdown,
    _sum_traffic_breakdowns,
    _tile_aware_stage_role_scales,
    _traffic_total,
    _estimate_tile_aware_multi_chip_latency,
    TILE_AWARE_DP_MULTI_CHIP_MODEL,
    valid_tp_degrees,
)


DP_TP_EP_TOPOLOGY_SCHEMA = "whole_request_dp_tp_ep_topology_v1"
DP_REQUEST_PARTITION_SCHEMA = "balanced_contiguous_whole_request_v1"
DP_COMMUNICATION_SCHEMA = "dependency_serial_tp_ring_ep_port_schedule_v1"
SUPPORTED_NVLINK_PORT_COUNTS = frozenset({1, 2, 4})


def valid_dp_tp_ep_topologies(
    model: Mapping[str, Any],
    chip_count: int,
    batch_size: int,
    *,
    routing_mode: str | None = None,
) -> tuple[tuple[int, int, int], ...]:
    """Return legal ``(DP, TP, EP)`` tuples for one physical chip count.

    Dense models use EP=1.  MoE EP is an independent physical axis and may
    exceed the number of request origins; such ranks remain useful as expert
    owners even when they do not originate a request.
    """

    if chip_count <= 0 or batch_size <= 0:
        raise ValueError("chip_count and batch_size must be positive")
    num_experts = int(model.get("num_experts", 0) or 0)
    is_moe = num_experts > 1
    topologies: list[tuple[int, int, int]] = []
    for tp_degree in valid_tp_degrees(model, chip_count):
        remaining = chip_count // tp_degree
        if not is_moe:
            if remaining <= batch_size:
                topologies.append((remaining, tp_degree, 1))
            continue
        ep_values = (
            tuple(
                ep
                for ep in range(1, remaining + 1)
                if remaining % ep == 0 and num_experts % ep == 0
            )
            if routing_mode in {None, "fixed-balanced"}
            else (1,)
        )
        for ep_degree in ep_values:
            dp_degree = remaining // ep_degree
            if dp_degree <= batch_size:
                topologies.append((dp_degree, tp_degree, ep_degree))
    return tuple(sorted(set(topologies)))


def validate_dp_tp_ep_topology(
    model: Mapping[str, Any],
    *,
    chip_count: int,
    batch_size: int,
    dp_degree: int,
    tp_degree: int,
    ep_degree: int,
    routing_mode: str | None = None,
) -> None:
    legal = valid_dp_tp_ep_topologies(
        model,
        chip_count,
        batch_size,
        routing_mode=routing_mode,
    )
    topology = (int(dp_degree), int(tp_degree), int(ep_degree))
    if topology not in legal:
        raise ValueError(
            f"DP/TP/EP={topology} is illegal for N={chip_count}, "
            f"batch={batch_size}, routing={routing_mode!r}; valid={legal}"
        )


def balanced_request_partition(
    batch_size: int,
    origin_count: int,
) -> tuple[dict[str, int], ...]:
    """Assign complete, contiguous requests to analytical origins."""

    if batch_size <= 0 or origin_count <= 0:
        raise ValueError("batch_size and origin_count must be positive")
    base, remainder = divmod(batch_size, origin_count)
    result: list[dict[str, int]] = []
    cursor = 0
    for origin in range(origin_count):
        count = base + int(origin < remainder)
        result.append(
            {
                "origin": origin,
                "batch_start": cursor,
                "batch_count": count,
                "batch_end": cursor + count,
            }
        )
        cursor += count
    if cursor != batch_size:
        raise AssertionError("request partition is not conservative")
    return tuple(result)


def _fixed_balanced_route_counts(
    *,
    batch_start: int,
    batch_count: int,
    seq_len: int,
    num_experts: int,
    top_k: int,
) -> list[int]:
    """Count round-robin fixed-balanced routes without materializing tokens."""

    if batch_count <= 0:
        return [0] * num_experts
    first_route = batch_start * seq_len * top_k
    route_count = batch_count * seq_len * top_k
    quotient, remainder = divmod(route_count, num_experts)
    start = first_route % num_experts
    return [
        quotient + int((expert - start) % num_experts < remainder)
        for expert in range(num_experts)
    ]


def _zero_ffn_counts(intermediate_size: int, mlen: int) -> dict[str, int]:
    physical_intermediate = math.ceil(intermediate_size / mlen) * mlen
    return {
        "M_MM": 0,
        "M_MM_WO": 0,
        "physical_intermediate": physical_intermediate,
    }


def _zero_attention_counts() -> dict[str, int]:
    return {"M_MM": 0, "M_MM_WO": 0}


def _zero_attention_core() -> dict[str, int]:
    return {
        "bmm_occurrences": 0,
        "softmax_row_groups": 0,
        "q_tail_rows": 0,
        "tail_bmm_occurrences": 0,
        "tail_full_width_work_cycles": 0,
    }


def _build_dp_rank_plans(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    dp_degree: int,
    tp_degree: int,
    ep_degree: int,
    seq_len: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = dict(report.get("trace") or {})
    hardware = dict(trace.get("hardware") or {})
    native_layout = dict(trace.get("native_layout") or {})
    workload = dict(trace.get("workload") or {})
    if not hardware or not native_layout:
        raise ValueError(
            "tile-aware DP model requires trace hardware and native layout"
        )
    mlen = int(hardware["mlen"])
    blen = int(hardware["blen"])
    hlen = int(hardware.get("hlen", model.get("head_dim", 128)))
    max_k_tiles = max(1, int(hardware.get("mram_tile_capacity", 1)))
    hidden = int(model["hidden_size"])
    intermediate = int(
        workload.get("inter_dim")
        or model.get("moe_intermediate_size")
        or model.get("intermediate_size")
    )
    q_heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    head_dim = int(model["head_dim"])
    logical_broadcast = q_heads // kv_heads
    global_rows = int(native_layout["physical_rows"])
    global_head = _head_storage_width(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        hlen=hlen,
        mlen=mlen,
        logical_broadcast=logical_broadcast,
    )
    global_ffn = _ffn_projection_counts(
        physical_rows=global_rows,
        hidden_size=hidden,
        intermediate_size=intermediate,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    global_attention_projection = _attention_projection_counts(
        physical_rows=global_rows,
        hidden_size=hidden,
        q_storage_width=int(global_head["total_q_dim"]),
        kv_heads=kv_heads,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    full_shape = {"chunks": ({"start": 0, "length": seq_len},)}
    global_attention_core = _attention_block_work(
        rank_shape=full_shape,
        batch_size=batch_size,
        mlen=mlen,
        head_groups=int(global_head["logical_groups"]),
        broadcast_heads=int(global_head["physical_broadcast"]),
        softmax_row_lanes=int(trace.get("softmax_row_lanes", 1)),
    )
    num_experts = int(
        workload.get("num_experts") or model.get("num_experts", 0) or 0
    )
    top_k = int(
        workload.get("experts_per_token")
        or model.get("num_experts_per_tok", 0)
        or model.get("experts_per_token", 0)
        or 0
    )
    routing_mode = dict(trace.get("compiler_metadata") or {}).get(
        "moe_routing_mode"
    )
    if num_experts > 1 and routing_mode not in {None, "fixed-balanced"}:
        raise ValueError(
            "distributed DP/TP/EP MoE requires fixed-balanced routing"
        )
    global_router_counts = (
        dict(
            zip(
                ("M_MM", "M_MM_WO"),
                _projection_counts(
                    physical_rows=global_rows,
                    k_size=hidden,
                    out_size=num_experts,
                    mlen=mlen,
                    blen=blen,
                    max_k_tiles=max_k_tiles,
                ),
            )
        )
        if num_experts > 1
        else _zero_attention_counts()
    )

    origin_count = dp_degree * ep_degree
    origins = balanced_request_partition(batch_size, origin_count)
    origin_rows: list[dict[str, Any]] = []
    origin_routes: list[list[int]] = []
    for assignment in origins:
        local_batch = int(assignment["batch_count"])
        if local_batch:
            packing = SequencePackingPlan.build(
                batch_size=local_batch,
                seq_len=seq_len,
                mlen=mlen,
                mode="compact",
            )
            physical_rows = packing.compile_seq_rows
        else:
            packing = None
            physical_rows = 0
        origin_rows.append(
            {
                **assignment,
                "active_rows": local_batch * seq_len,
                "physical_rows": physical_rows,
                "chunks": ({"start": 0, "length": seq_len},),
                "token_fraction": local_batch / batch_size,
                "causal_pair_fraction": local_batch / batch_size,
                "packing": packing,
            }
        )
        origin_routes.append(
            _fixed_balanced_route_counts(
                batch_start=int(assignment["batch_start"]),
                batch_count=local_batch,
                seq_len=seq_len,
                num_experts=max(1, num_experts),
                top_k=max(1, top_k),
            )
            if num_experts > 1
            else []
        )

    global_route_counts = (
        [
            sum(origin[expert] for origin in origin_routes)
            for expert in range(num_experts)
        ]
        if num_experts > 1
        else []
    )
    global_padded_routes = [
        math.ceil(count / blen) * blen if count else 0
        for count in global_route_counts
    ]
    global_expert_mm = 0
    global_expert_wo = 0
    for rows in global_padded_routes:
        if rows:
            counts = _ffn_projection_counts(
                physical_rows=rows,
                hidden_size=hidden,
                intermediate_size=intermediate,
                mlen=mlen,
                blen=blen,
                max_k_tiles=max_k_tiles,
            )
            global_expert_mm += counts["M_MM"]
            global_expert_wo += counts["M_MM_WO"]

    plans: list[dict[str, Any]] = []
    for dp_rank in range(dp_degree):
        group_origins = [
            dp_rank * ep_degree + ep_rank for ep_rank in range(ep_degree)
        ]
        group_route_counts = (
            [
                sum(origin_routes[source][expert] for source in group_origins)
                for expert in range(num_experts)
            ]
            if num_experts > 1
            else []
        )
        for ep_rank in range(ep_degree):
            origin = dp_rank * ep_degree + ep_rank
            row_shape = origin_rows[origin]
            local_batch = int(row_shape["batch_count"])
            experts_per_rank = num_experts // ep_degree if num_experts > 1 else 0
            expert_start = ep_rank * experts_per_rank
            expert_end = expert_start + experts_per_rank
            owned_counts = group_route_counts[expert_start:expert_end]
            padded_rows = [
                math.ceil(count / blen) * blen if count else 0
                for count in owned_counts
            ]
            for tp_rank in range(tp_degree):
                local_hidden = _balanced_part(hidden, tp_degree, tp_rank)
                local_q_heads = _balanced_part(q_heads, tp_degree, tp_rank)
                local_kv_heads = _balanced_part(kv_heads, tp_degree, tp_rank)
                local_intermediate = _balanced_part(
                    intermediate, tp_degree, tp_rank
                )
                local_head = _head_storage_width(
                    q_heads=local_q_heads,
                    kv_heads=local_kv_heads,
                    head_dim=head_dim,
                    hlen=hlen,
                    mlen=mlen,
                    logical_broadcast=logical_broadcast,
                )
                if local_batch:
                    local_ffn = _ffn_projection_counts(
                        physical_rows=int(row_shape["physical_rows"]),
                        hidden_size=hidden,
                        intermediate_size=local_intermediate,
                        mlen=mlen,
                        blen=blen,
                        max_k_tiles=max_k_tiles,
                    )
                    local_attention_projection = _attention_projection_counts(
                        physical_rows=int(row_shape["physical_rows"]),
                        hidden_size=hidden,
                        q_storage_width=int(local_head["total_q_dim"]),
                        kv_heads=local_kv_heads,
                        mlen=mlen,
                        blen=blen,
                        max_k_tiles=max_k_tiles,
                    )
                    local_attention_core = _attention_block_work(
                        rank_shape=row_shape,
                        batch_size=local_batch,
                        mlen=mlen,
                        head_groups=int(local_head["logical_groups"]),
                        broadcast_heads=int(local_head["physical_broadcast"]),
                        softmax_row_lanes=int(trace.get("softmax_row_lanes", 1)),
                    )
                    local_router_counts = (
                        dict(
                            zip(
                                ("M_MM", "M_MM_WO"),
                                _projection_counts(
                                    physical_rows=int(row_shape["physical_rows"]),
                                    k_size=local_hidden,
                                    out_size=num_experts,
                                    mlen=mlen,
                                    blen=blen,
                                    max_k_tiles=max_k_tiles,
                                ),
                            )
                        )
                        if num_experts > 1
                        else _zero_attention_counts()
                    )
                else:
                    local_ffn = _zero_ffn_counts(local_intermediate, mlen)
                    local_attention_projection = _zero_attention_counts()
                    local_attention_core = _zero_attention_core()
                    local_router_counts = _zero_attention_counts()

                local_expert_mm = 0
                local_expert_wo = 0
                for rows in padded_rows:
                    if rows:
                        counts = _ffn_projection_counts(
                            physical_rows=rows,
                            hidden_size=hidden,
                            intermediate_size=local_intermediate,
                            mlen=mlen,
                            blen=blen,
                            max_k_tiles=max_k_tiles,
                        )
                        local_expert_mm += counts["M_MM"]
                        local_expert_wo += counts["M_MM_WO"]

                token_scale = int(row_shape["active_rows"]) / max(
                    1, seq_len * batch_size
                )
                physical_row_scale = int(row_shape["physical_rows"]) / max(
                    1, global_rows
                )
                segmented_norm_scale = (
                    int(row_shape["active_rows"])
                    * (local_q_heads + local_kv_heads)
                    / max(1, seq_len * batch_size * (q_heads + kv_heads))
                )
                ffn_tensor_scale = (
                    int(row_shape["physical_rows"])
                    * int(local_ffn["physical_intermediate"])
                    / max(
                        1,
                        global_rows
                        * int(global_ffn["physical_intermediate"]),
                    )
                )
                q_tensor_scale = (
                    int(row_shape["physical_rows"])
                    * int(local_head["total_q_dim"])
                    / max(
                        1,
                        global_rows * int(global_head["total_q_dim"]),
                    )
                )
                expert = None
                if num_experts > 1:
                    local_routes = sum(origin_routes[origin])
                    remote_routes = sum(
                        count
                        for expert_id, count in enumerate(origin_routes[origin])
                        if expert_id // experts_per_rank != ep_rank
                    )
                    expert = {
                        "dp_group": dp_rank,
                        "ep_rank": ep_rank,
                        "experts_per_rank": experts_per_rank,
                        "expert_start": expert_start,
                        "expert_end": expert_end,
                        "local_routes": local_routes,
                        "remote_routes": remote_routes,
                        "owned_route_count": sum(owned_counts),
                        "owned_route_counts": tuple(owned_counts),
                        "owned_padded_bucket_rows": sum(padded_rows),
                        "expert_bucket_utilization": (
                            sum(owned_counts) / sum(padded_rows)
                            if sum(padded_rows)
                            else 1.0
                        ),
                        "expert_M_MM": local_expert_mm,
                        "expert_M_MM_WO": local_expert_wo,
                        "global_expert_M_MM": global_expert_mm,
                        "global_expert_M_MM_WO": global_expert_wo,
                    }
                rank = (dp_rank * ep_degree + ep_rank) * tp_degree + tp_rank
                plans.append(
                    {
                        "rank": rank,
                        "dp_rank": dp_rank,
                        "ep_rank": ep_rank,
                        "tp_rank": tp_rank,
                        "origin": origin,
                        "batch_start": int(row_shape["batch_start"]),
                        "local_batch_size": local_batch,
                        "shared_active": bool(local_batch),
                        "active_rows": int(row_shape["active_rows"]),
                        "physical_rows": int(row_shape["physical_rows"]),
                        "chunks": row_shape["chunks"],
                        "local_q_heads": local_q_heads,
                        "local_kv_heads": local_kv_heads,
                        "local_intermediate_size": local_intermediate,
                        "local_hidden_size": local_hidden,
                        "local_head_packing": local_head,
                        "ffn_counts": local_ffn,
                        "attention_projection_counts": local_attention_projection,
                        "attention_core_counts": local_attention_core,
                        "router_counts": local_router_counts,
                        "semantic_scales": {
                            "token_replicated_hidden": token_scale,
                            "token_tensor_sharded": segmented_norm_scale,
                            "attention_projection_tiled": (
                                local_attention_projection["M_MM"]
                                / max(1, global_attention_projection["M_MM"])
                            ),
                            "attention_head_pair_sharded": (
                                local_attention_core["bmm_occurrences"]
                                / max(1, global_attention_core["bmm_occurrences"])
                            ),
                            "ffn_projection_tiled": (
                                local_ffn["M_MM"] / max(1, global_ffn["M_MM"])
                            ),
                            "row_parallel_projection": token_scale,
                            "expert_tensor_sharded": (
                                local_expert_mm / max(1, global_expert_mm)
                                if num_experts > 1
                                else token_scale / max(1, tp_degree)
                            ),
                            "replicated_setup": float(bool(local_batch)),
                        },
                        "segmented_norm_scale": segmented_norm_scale,
                        "q_tensor_scale": q_tensor_scale,
                        "physical_row_scale": physical_row_scale,
                        "ffn_tensor_scale": ffn_tensor_scale,
                        "expert": expert,
                    }
                )

    return plans, {
        "context": {"seq_len": seq_len},
        "request_partition": origins,
        "origin_route_counts": tuple(tuple(item) for item in origin_routes),
        "global_rows": global_rows,
        "global_head_packing": global_head,
        "global_ffn_counts": global_ffn,
        "global_attention_projection_counts": global_attention_projection,
        "global_attention_core_counts": global_attention_core,
        "global_router_counts": global_router_counts,
        "num_experts": num_experts,
        "top_k": top_k,
        "routing_mode": routing_mode,
        "global_route_counts": tuple(global_route_counts),
        "dp_degree": dp_degree,
        "tp_degree": tp_degree,
        "ep_degree": ep_degree,
    }


def _dp_stage_role_scales(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    global_plan: Mapping[str, Any],
    *,
    tp_degree: int,
    ep_degree: int,
    kv_cache_overlay: Mapping[str, Any] | None,
) -> dict[tuple[str, str], float]:
    scales = _tile_aware_stage_role_scales(
        report,
        plan,
        global_plan,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        kv_cache_overlay=kv_cache_overlay,
    )
    if plan["shared_active"]:
        return scales
    # An EP rank with no request can still own and execute experts.  It does
    # not issue shared-layer or router DMA merely because the weights occupy
    # capacity on that rank.
    return {
        key: (
            value
            if key[0].startswith("layer/moe/experts")
            else 0.0
        )
        for key, value in scales.items()
    }


def _ideal_rank_scale(
    plan: Mapping[str, Any],
    global_plan: Mapping[str, Any],
    *,
    semantic: str,
    tp_degree: int,
    actual_scale: float,
) -> float:
    """Return the unpadded logical share used for rounding diagnostics."""

    token_fraction = float(plan["semantic_scales"]["token_replicated_hidden"])
    if semantic in {
        "attention_projection_tiled",
        "attention_head_pair_sharded",
        "ffn_projection_tiled",
        "token_tensor_sharded",
    }:
        return token_fraction / tp_degree
    if semantic == "expert_tensor_sharded" and plan.get("expert"):
        total_routes = math.fsum(global_plan.get("global_route_counts", ()))
        return (
            float(plan["expert"]["owned_route_count"])
            / max(1.0, total_routes)
            / tp_degree
        )
    if semantic in {
        "token_replicated_hidden",
        "row_parallel_projection",
    }:
        return token_fraction
    if semantic == "replicated_setup":
        return float(bool(plan["shared_active"]))
    return actual_scale


def _ring_all_reduce(
    tensor_bytes: float,
    *,
    group_size: int,
    port_count: int,
    per_port_one_way_gbps: float,
    startup_ns: float,
) -> dict[str, float | int]:
    if group_size <= 1 or tensor_bytes <= 0:
        return {
            "latency_ns": 0.0,
            "wire_bytes_per_rank": 0.0,
            "active_rings": 0,
        }
    active_rings = min(port_count, group_size - 1)
    wire_bytes = 2.0 * (group_size - 1) / group_size * tensor_bytes
    latency = (
        2.0 * (group_size - 1) * startup_ns
        + wire_bytes / (active_rings * per_port_one_way_gbps)
    )
    return {
        "latency_ns": latency,
        "wire_bytes_per_rank": wire_bytes,
        "active_rings": active_rings,
    }


def _all_to_all_schedule(
    transfer_bytes: Sequence[Sequence[float]],
    *,
    port_count: int,
    per_port_one_way_gbps: float,
    startup_ns: float,
) -> dict[str, Any]:
    """Schedule a nonblocking pairwise exchange with a finite port count.

    Cyclic peer offsets are conflict-free: every source and destination occurs
    at most once per offset.  Up to ``port_count`` offsets are striped in one
    round, which makes endpoint concurrency explicit without an empirical link
    efficiency factor.
    """

    size = len(transfer_bytes)
    if size <= 1:
        return {
            "latency_ns": 0.0,
            "aggregate_bytes": 0.0,
            "max_rank_bytes": 0.0,
            "round_count": 0,
            "round_latency_ns": (),
        }
    if any(len(row) != size for row in transfer_bytes):
        raise ValueError("all-to-all matrix must be square")
    active_ports = min(port_count, size - 1)
    offsets = list(range(1, size))
    round_latencies: list[float] = []
    for index in range(0, len(offsets), active_ports):
        group = offsets[index : index + active_ports]
        largest = max(
            float(transfer_bytes[source][(source + offset) % size])
            for source in range(size)
            for offset in group
        )
        round_latencies.append(startup_ns + largest / per_port_one_way_gbps)
    rank_bytes = [math.fsum(float(value) for value in row) for row in transfer_bytes]
    return {
        "latency_ns": math.fsum(round_latencies),
        "aggregate_bytes": math.fsum(rank_bytes),
        "max_rank_bytes": max(rank_bytes, default=0.0),
        "round_count": len(round_latencies),
        "round_latency_ns": tuple(round_latencies),
    }


def _moe_layer_count(model: Mapping[str, Any]) -> int:
    mlp_types = tuple(model.get("mlp_types") or ())
    if mlp_types:
        return sum(str(value).lower() == "moe" for value in mlp_types)
    return int(model["num_hidden_layers"]) if int(model.get("num_experts", 0) or 0) > 1 else 0


def _dense_layer_count(model: Mapping[str, Any]) -> int:
    mlp_types = tuple(model.get("mlp_types") or ())
    if mlp_types:
        return sum(str(value).lower() != "moe" for value in mlp_types)
    return int(model["num_hidden_layers"]) if int(model.get("num_experts", 0) or 0) <= 1 else 0


def _dp_tp_ep_communication(
    plans: Sequence[Mapping[str, Any]],
    global_plan: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    dp_degree: int,
    tp_degree: int,
    ep_degree: int,
    seq_len: int,
    fp_width_bits: int,
    nvlink_port_count: int,
    nvlink_port_bidirectional_gbps: float,
    startup_ns: float,
) -> dict[str, Any]:
    per_port_one_way = nvlink_port_bidirectional_gbps / 2.0
    layers = int(model["num_hidden_layers"])
    moe_layers = _moe_layer_count(model)
    dense_layers = _dense_layer_count(model)
    hidden = int(model["hidden_size"])
    num_experts = int(global_plan["num_experts"])
    plan_by_coord = {
        (int(plan["dp_rank"]), int(plan["ep_rank"]), int(plan["tp_rank"])): plan
        for plan in plans
    }
    replica_latency_by_stage: list[dict[str, float]] = []
    per_rank_bytes: dict[int, float] = defaultdict(float)
    aggregate_by_stage: dict[str, float] = defaultdict(float)
    max_rank_by_stage: dict[str, float] = defaultdict(float)
    stage_latency_global: dict[str, float] = defaultdict(float)
    tp_bytes_by_stage: dict[str, float] = defaultdict(float)
    tp_latency_by_stage: dict[str, float] = defaultdict(float)
    tp_active_rings = min(nvlink_port_count, max(0, tp_degree - 1))

    for dp_rank in range(dp_degree):
        replica_stage: dict[str, float] = defaultdict(float)
        for ep_rank in range(ep_degree):
            representative = plan_by_coord[(dp_rank, ep_rank, 0)]
            local_tokens = int(representative["active_rows"])
            dense_tensor_bytes = local_tokens * hidden * fp_width_bits / 8.0
            for stage, stage_layers in (
                ("layer/attention", layers),
                ("layer/ffn", dense_layers),
            ):
                if stage_layers == 0:
                    continue
                collective = _ring_all_reduce(
                    dense_tensor_bytes,
                    group_size=tp_degree,
                    port_count=nvlink_port_count,
                    per_port_one_way_gbps=per_port_one_way,
                    startup_ns=startup_ns,
                )
                latency = float(collective["latency_ns"]) * stage_layers
                wire_per_rank = (
                    float(collective["wire_bytes_per_rank"]) * stage_layers
                )
                replica_stage[stage] = max(replica_stage[stage], latency)
                tp_latency_by_stage[stage] = max(tp_latency_by_stage[stage], latency)
                tp_bytes_by_stage[stage] = max(tp_bytes_by_stage[stage], wire_per_rank)
                for tp_rank in range(tp_degree):
                    rank = int(plan_by_coord[(dp_rank, ep_rank, tp_rank)]["rank"])
                    per_rank_bytes[rank] += wire_per_rank
                aggregate_by_stage[stage] += wire_per_rank * tp_degree

            if num_experts > 1:
                router_tensor_bytes = (
                    local_tokens * num_experts * fp_width_bits / 8.0
                )
                router = _ring_all_reduce(
                    router_tensor_bytes,
                    group_size=tp_degree,
                    port_count=nvlink_port_count,
                    per_port_one_way_gbps=per_port_one_way,
                    startup_ns=startup_ns,
                )
                router_latency = float(router["latency_ns"]) * moe_layers
                router_wire = float(router["wire_bytes_per_rank"]) * moe_layers
                replica_stage["layer/moe/router"] = max(
                    replica_stage["layer/moe/router"], router_latency
                )
                tp_latency_by_stage["layer/moe/router"] = max(
                    tp_latency_by_stage["layer/moe/router"], router_latency
                )
                tp_bytes_by_stage["layer/moe/router"] = max(
                    tp_bytes_by_stage["layer/moe/router"], router_wire
                )
                for tp_rank in range(tp_degree):
                    rank = int(plan_by_coord[(dp_rank, ep_rank, tp_rank)]["rank"])
                    per_rank_bytes[rank] += router_wire
                aggregate_by_stage["layer/moe/router"] += router_wire * tp_degree

                owned_routes = int((representative.get("expert") or {}).get("owned_route_count", 0))
                expert_tensor_bytes = owned_routes * hidden * fp_width_bits / 8.0
                expert = _ring_all_reduce(
                    expert_tensor_bytes,
                    group_size=tp_degree,
                    port_count=nvlink_port_count,
                    per_port_one_way_gbps=per_port_one_way,
                    startup_ns=startup_ns,
                )
                expert_latency = float(expert["latency_ns"]) * moe_layers
                expert_wire = float(expert["wire_bytes_per_rank"]) * moe_layers
                replica_stage["layer/moe/experts"] = max(
                    replica_stage["layer/moe/experts"], expert_latency
                )
                tp_latency_by_stage["layer/moe/experts"] = max(
                    tp_latency_by_stage["layer/moe/experts"], expert_latency
                )
                tp_bytes_by_stage["layer/moe/experts"] = max(
                    tp_bytes_by_stage["layer/moe/experts"], expert_wire
                )
                for tp_rank in range(tp_degree):
                    rank = int(plan_by_coord[(dp_rank, ep_rank, tp_rank)]["rank"])
                    per_rank_bytes[rank] += expert_wire
                aggregate_by_stage["layer/moe/experts"] += expert_wire * tp_degree

        replica_latency_by_stage.append(dict(replica_stage))

    ep_dispatch_latency = 0.0
    ep_return_latency = 0.0
    ep_dispatch_bytes = 0.0
    ep_return_bytes = 0.0
    ep_rounds = 0
    if ep_degree > 1 and num_experts > 1:
        route_counts = tuple(global_plan["origin_route_counts"])
        experts_per_rank = num_experts // ep_degree
        for dp_rank in range(dp_degree):
            dispatch_routes = [[0] * ep_degree for _ in range(ep_degree)]
            for source_ep in range(ep_degree):
                origin = dp_rank * ep_degree + source_ep
                for expert_id, count in enumerate(route_counts[origin]):
                    dispatch_routes[source_ep][expert_id // experts_per_rank] += int(count)
            for tp_rank in range(tp_degree):
                dispatch_matrix = [
                    [
                        routes
                        * (hidden * fp_width_bits / (8.0 * tp_degree) + 8.0)
                        if source != target
                        else 0.0
                        for target, routes in enumerate(row)
                    ]
                    for source, row in enumerate(dispatch_routes)
                ]
                return_matrix = [
                    [
                        dispatch_routes[target][source]
                        * hidden
                        * fp_width_bits
                        / (8.0 * tp_degree)
                        if source != target
                        else 0.0
                        for target in range(ep_degree)
                    ]
                    for source in range(ep_degree)
                ]
                dispatch = _all_to_all_schedule(
                    dispatch_matrix,
                    port_count=nvlink_port_count,
                    per_port_one_way_gbps=per_port_one_way,
                    startup_ns=startup_ns,
                )
                returned = _all_to_all_schedule(
                    return_matrix,
                    port_count=nvlink_port_count,
                    per_port_one_way_gbps=per_port_one_way,
                    startup_ns=startup_ns,
                )
                ep_dispatch_latency = max(
                    ep_dispatch_latency,
                    float(dispatch["latency_ns"]) * moe_layers,
                )
                ep_return_latency = max(
                    ep_return_latency,
                    float(returned["latency_ns"]) * moe_layers,
                )
                ep_rounds = max(ep_rounds, int(dispatch["round_count"]))
                ep_dispatch_bytes += float(dispatch["aggregate_bytes"]) * moe_layers
                ep_return_bytes += float(returned["aggregate_bytes"]) * moe_layers
                for ep_rank in range(ep_degree):
                    rank = int(plan_by_coord[(dp_rank, ep_rank, tp_rank)]["rank"])
                    per_rank_bytes[rank] += (
                        math.fsum(dispatch_matrix[ep_rank])
                        + math.fsum(return_matrix[ep_rank])
                    ) * moe_layers
        for replica in replica_latency_by_stage:
            replica["layer/moe/dispatch"] = ep_dispatch_latency
            replica["layer/moe/combine"] = ep_return_latency
        aggregate_by_stage["layer/moe/dispatch"] += ep_dispatch_bytes
        aggregate_by_stage["layer/moe/combine"] += ep_return_bytes

    for stage in set().union(*(item.keys() for item in replica_latency_by_stage)):
        stage_latency_global[stage] = max(
            item.get(stage, 0.0) for item in replica_latency_by_stage
        )
    for rank, value in per_rank_bytes.items():
        max_rank_by_stage["all"] = max(max_rank_by_stage["all"], value)
    aggregate_bytes = math.fsum(aggregate_by_stage.values())
    return {
        "tp_collective_bytes_by_stage": dict(tp_bytes_by_stage),
        "tp_collective_latency_ns_by_stage": dict(tp_latency_by_stage),
        "tp_collective_bytes": math.fsum(tp_bytes_by_stage.values()),
        "tp_collective_latency_ns": math.fsum(tp_latency_by_stage.values()),
        "tp_active_rings": tp_active_rings,
        "ep_dispatch_bytes": ep_dispatch_bytes,
        "ep_return_bytes": ep_return_bytes,
        "ep_dispatch_latency_ns": ep_dispatch_latency,
        "ep_return_latency_ns": ep_return_latency,
        "ep_all_to_all_rounds_per_layer": ep_rounds,
        "interconnect_bytes_by_stage": dict(aggregate_by_stage),
        "interconnect_latency_ns_by_stage": dict(stage_latency_global),
        "interconnect_bytes": max(per_rank_bytes.values(), default=0.0),
        "interconnect_latency_ns": math.fsum(stage_latency_global.values()),
        "aggregate_interconnect_bytes": aggregate_bytes,
        "replica_communication_latency_ns_by_stage": replica_latency_by_stage,
        "dp_internal_communication_bytes": 0.0,
        "dp_internal_communication_latency_ns": 0.0,
    }


def _single_chip_identity(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    reference_a100_count: int,
    aggregate_hbm_bandwidth_gbps: float,
    aggregate_hbm_capacity_bytes: int,
    seq_len: int,
    batch_size: int,
    fp_width_bits: int,
    kv_width_bits: float,
    nvlink_port_count: int,
    nvlink_port_bidirectional_gbps: float,
    interconnect_startup_ns: float,
    kv_cache_overlay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    census_entries, census_totals = _parallel_kernel_entries(report)
    stage_opcode_cycles = {
        str(stage): {
            str(opcode): float(cycles)
            for opcode, cycles in dict(values).items()
        }
        for stage, values in (
            report.get("stage_compute_opcode_work_cycles") or {}
        ).items()
    }
    replicated_compute_cycles = math.fsum(
        stage_opcode_cycles[str(entry["stage"])][str(entry["opcode"])]
        * int(entry["count"])
        / census_totals[(str(entry["stage"]), str(entry["opcode"]))]
        for entry in census_entries
        if str(entry["tp_semantics"])
        in {
            "token_replicated_hidden",
            "row_parallel_projection",
            "replicated_setup",
        }
    )
    baseline = _estimate_tile_aware_multi_chip_latency(
        report,
        model,
        chip_count=1,
        tp_degree=1,
        ep_degree=1,
        reference_a100_count=reference_a100_count,
        aggregate_hbm_bandwidth_gbps=aggregate_hbm_bandwidth_gbps,
        aggregate_hbm_capacity_bytes=aggregate_hbm_capacity_bytes,
        seq_len=seq_len,
        batch_size=batch_size,
        fp_width_bits=fp_width_bits,
        kv_width_bits=kv_width_bits,
        nvlink_port_count=nvlink_port_count,
        nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
        interconnect_startup_ns=interconnect_startup_ns,
        kv_cache_overlay=kv_cache_overlay,
    )
    baseline.update(
        {
            "parallel_model": "dp-tp-ep",
            "multi_chip_model": TILE_AWARE_DP_MULTI_CHIP_MODEL,
            "dp_degree": 1,
            "tp_degree": 1,
            "ep_degree": 1,
            "dp_tp_ep_legality": "exact_single_chip_identity",
            "request_partition_schema": DP_REQUEST_PARTITION_SCHEMA,
            "request_origin_count": 1,
            "active_request_origin_count": 1,
            "idle_request_origin_count": 0,
            "local_batch_by_origin": [batch_size],
            "max_local_batch_size": batch_size,
            "batch_packing_utilization": float(
                (report.get("trace") or {})
                .get("native_layout", {})
                .get("row_utilization", 1.0)
            ),
            "shared_weight_replication": 1,
            "expert_weight_replication": 1,
            "weight_replication_factor": 1,
            "physical_weight_traffic_replication": 1.0,
            "padding_cycles": 0.0,
            "padding_cycles_by_rank": [0.0],
            "replicated_compute_cycles": replicated_compute_cycles,
            "replicated_compute_cycles_by_rank": [
                replicated_compute_cycles
            ],
            "tp_rounding_overhead": 0.0,
            "dependency_serial_nominal_ns": float(baseline["latency_ns"]),
            "fixed_batch_makespan_ns": float(baseline["latency_ns"]),
            "fixed_batch_requests_per_second": (
                batch_size * 1e9 / float(baseline["latency_ns"])
            ),
            "fixed_batch_tokens_per_second": (
                batch_size * seq_len * 1e9 / float(baseline["latency_ns"])
            ),
            "dp_internal_communication_bytes": 0.0,
            "dp_internal_communication_latency_ns": 0.0,
            "communication_overlap_bound": (
                "local_compute_hbm_roofline;tp_ep_dependency_serial"
            ),
            "communication_model_schema": DP_COMMUNICATION_SCHEMA,
            "multi_chip_fidelity": {
                "compute": "exact_single_chip_costemitter_identity",
                "memory": "exact_single_chip_v4_identity",
                "communication": "none",
                "compiler_isa": "native_single_chip",
            },
            "v4_local_geometry_reconstruction": True,
            "v4_rank_latency_exact": True,
        }
    )
    baseline.pop("cp_degree", None)
    baseline.pop("tp_cp_ep_legality", None)
    baseline.pop("context_partition", None)
    return baseline


def estimate_tile_aware_dp_tp_ep_latency(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    chip_count: int,
    dp_degree: int,
    tp_degree: int,
    ep_degree: int,
    reference_a100_count: int,
    aggregate_hbm_bandwidth_gbps: float,
    aggregate_hbm_capacity_bytes: int,
    seq_len: int,
    batch_size: int,
    fp_width_bits: int,
    kv_width_bits: float,
    nvlink_port_count: int,
    nvlink_port_bidirectional_gbps: float = DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS,
    interconnect_startup_ns: float = DEFAULT_NVLINK_STARTUP_US * 1_000.0,
    kv_cache_overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate fixed-batch makespan for a DP x TP x EP topology."""

    trace = dict(report.get("trace") or {})
    if report.get("compute_timing_mode") != "ideal-ii1":
        raise ValueError(
            "tile-aware-dp-tp-ep-v4 requires additive ideal-II1 work"
        )
    if int(trace.get("schema_version", 0)) != 7:
        raise ValueError(
            "tile-aware-dp-tp-ep-v4 requires CostTrace schema 7 lineage"
        )
    if nvlink_port_count not in SUPPORTED_NVLINK_PORT_COUNTS:
        raise ValueError("nvlink_port_count must be one of 1, 2, or 4")
    routing_mode = dict(trace.get("compiler_metadata") or {}).get(
        "moe_routing_mode"
    )
    validate_dp_tp_ep_topology(
        model,
        chip_count=chip_count,
        batch_size=batch_size,
        dp_degree=dp_degree,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        routing_mode=routing_mode,
    )
    if chip_count == 1:
        return _single_chip_identity(
            report,
            model,
            reference_a100_count=reference_a100_count,
            aggregate_hbm_bandwidth_gbps=aggregate_hbm_bandwidth_gbps,
            aggregate_hbm_capacity_bytes=aggregate_hbm_capacity_bytes,
            seq_len=seq_len,
            batch_size=batch_size,
            fp_width_bits=fp_width_bits,
            kv_width_bits=kv_width_bits,
            nvlink_port_count=nvlink_port_count,
            nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
            interconnect_startup_ns=interconnect_startup_ns,
            kv_cache_overlay=kv_cache_overlay,
        )

    census_entries, census_totals = _parallel_kernel_entries(report)
    plans, global_plan = _build_dp_rank_plans(
        report,
        model,
        dp_degree=dp_degree,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    if len(plans) != chip_count:
        raise AssertionError(
            f"rank planner emitted {len(plans)} plans for N={chip_count}"
        )
    stage_opcode_cycles = {
        str(stage): {
            str(opcode): float(cycles)
            for opcode, cycles in dict(values).items()
        }
        for stage, values in (
            report.get("stage_compute_opcode_work_cycles") or {}
        ).items()
    }
    if not stage_opcode_cycles:
        raise ValueError(
            "tile-aware DP model requires stage opcode work cycles"
        )
    clock_period_ps = float(
        (report.get("compatibility") or {}).get("clock_period_ps", 1_000.0)
    )
    cycle_to_ns = clock_period_ps / 1_000.0

    rank_stage_cycles: list[dict[str, float]] = []
    rank_opcode_scales: list[dict[str, float]] = []
    rank_kernel_opcode_scales: list[dict[str, float]] = []
    rank_kernel_scales: list[dict[str, float]] = []
    rank_padding_cycles: list[float] = []
    rank_replicated_compute_cycles: list[float] = []
    for plan in plans:
        stage_cycles: dict[str, float] = defaultdict(float)
        local_opcode_cycles: dict[str, float] = defaultdict(float)
        local_kernel_counts: dict[str, float] = defaultdict(float)
        baseline_kernel_counts: dict[str, float] = defaultdict(float)
        kernel_opcode_scales: dict[str, float] = {}
        padding_cycles = 0.0
        replicated_cycles = 0.0
        for entry in census_entries:
            stage = str(entry["stage"])
            opcode = str(entry["opcode"])
            lineage = parallel_kernel_lineage_id(entry)
            try:
                cycles = stage_opcode_cycles[stage][opcode]
            except KeyError as exc:
                raise ValueError(
                    f"missing timing work for lineage {stage}/{opcode}"
                ) from exc
            base_cycles = (
                cycles
                * int(entry["count"])
                / census_totals[(stage, opcode)]
            )
            semantic = str(entry["tp_semantics"])
            scale = _rank_opcode_scale(
                plan,
                global_plan,
                stage=stage,
                opcode=opcode,
                kernel=str(entry["kernel"]),
                semantic=semantic,
            )
            local_cycles = base_cycles * scale
            ideal_scale = _ideal_rank_scale(
                plan,
                global_plan,
                semantic=semantic,
                tp_degree=tp_degree,
                actual_scale=scale,
            )
            padding_cycles += max(0.0, local_cycles - base_cycles * ideal_scale)
            if semantic in {
                "token_replicated_hidden",
                "row_parallel_projection",
                "replicated_setup",
            }:
                replicated_cycles += local_cycles
            stage_cycles[stage] += local_cycles
            local_opcode_cycles[f"{stage}::{opcode}"] += local_cycles
            lineage_key = f"{stage}::{lineage}"
            kernel_opcode_key = f"{lineage_key}::{opcode}"
            previous = kernel_opcode_scales.setdefault(kernel_opcode_key, scale)
            if not math.isclose(previous, scale, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"inconsistent rank scale for {kernel_opcode_key}"
                )
            local_kernel_counts[lineage_key] += int(entry["count"]) * scale
            baseline_kernel_counts[lineage_key] += int(entry["count"])
        rank_stage_cycles.append(dict(stage_cycles))
        rank_opcode_scales.append(
            {
                key: value
                / max(
                    1e-30,
                    stage_opcode_cycles[key.split("::", 1)[0]][
                        key.split("::", 1)[1]
                    ],
                )
                for key, value in local_opcode_cycles.items()
            }
        )
        rank_kernel_opcode_scales.append(kernel_opcode_scales)
        rank_kernel_scales.append(
            {
                key: local_kernel_counts[key] / count
                for key, count in baseline_kernel_counts.items()
            }
        )
        rank_padding_cycles.append(padding_cycles)
        rank_replicated_compute_cycles.append(replicated_cycles)

    original_traffic = dict(report.get("hbm_traffic_breakdown") or {})
    rank_traffic: list[dict[str, Any]] = []
    rank_stage_memory: list[dict[str, float]] = []
    rank_stage_floor: list[dict[str, float]] = []
    per_chip_bandwidth = aggregate_hbm_bandwidth_gbps / chip_count
    baseline_bandwidth = aggregate_hbm_bandwidth_gbps / reference_a100_count
    for plan in plans:
        role_scales = _dp_stage_role_scales(
            report,
            plan,
            global_plan,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            kv_cache_overlay=kv_cache_overlay,
        )
        traffic = _scale_traffic_breakdown(
            original_traffic,
            stage_role_scales=role_scales,
        )
        memory, floor = _memory_latency_for_traffic(
            report,
            traffic,
            per_chip_bandwidth_gbps=per_chip_bandwidth,
            baseline_bandwidth_gbps=baseline_bandwidth,
        )
        rank_traffic.append(traffic)
        rank_stage_memory.append(memory)
        rank_stage_floor.append(floor)

    communication = _dp_tp_ep_communication(
        plans,
        global_plan,
        model,
        dp_degree=dp_degree,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        seq_len=seq_len,
        fp_width_bits=fp_width_bits,
        nvlink_port_count=nvlink_port_count,
        nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
        startup_ns=interconnect_startup_ns,
    )
    stages = sorted(
        set(stage_opcode_cycles)
        | set(report.get("hbm_stage_latency_ns") or {})
        | set(communication["interconnect_latency_ns_by_stage"])
    )
    rank_stage_compute_ns = [
        {stage: cycles * cycle_to_ns for stage, cycles in rank.items()}
        for rank in rank_stage_cycles
    ]
    replica_reports: list[dict[str, Any]] = []
    for dp_rank in range(dp_degree):
        rank_indices = [
            int(plan["rank"])
            for plan in plans
            if int(plan["dp_rank"]) == dp_rank
        ]
        replica_comm = dict(
            communication["replica_communication_latency_ns_by_stage"][dp_rank]
        )
        nominal: dict[str, float] = {}
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        local_roofline: dict[str, float] = {}
        compute_max: dict[str, float] = {}
        memory_max: dict[str, float] = {}
        for stage in stages:
            compute_values = [
                rank_stage_compute_ns[rank].get(stage, 0.0)
                for rank in rank_indices
            ]
            memory_values = [
                rank_stage_memory[rank].get(stage, 0.0)
                for rank in rank_indices
            ]
            compute_max[stage] = max(compute_values, default=0.0)
            memory_max[stage] = max(memory_values, default=0.0)
            roofline = max(
                (
                    max(
                        rank_stage_compute_ns[rank].get(stage, 0.0),
                        rank_stage_memory[rank].get(stage, 0.0),
                    )
                    for rank in rank_indices
                ),
                default=0.0,
            )
            comm = float(replica_comm.get(stage, 0.0))
            local_roofline[stage] = roofline
            nominal[stage] = roofline + comm
            lower[stage] = max(roofline, comm)
            upper[stage] = max(
                (
                    rank_stage_compute_ns[rank].get(stage, 0.0)
                    + rank_stage_memory[rank].get(stage, 0.0)
                    for rank in rank_indices
                ),
                default=0.0,
            ) + comm
        replica_reports.append(
            {
                "dp_rank": dp_rank,
                "rank_indices": rank_indices,
                "compute_ns_by_stage": compute_max,
                "memory_ns_by_stage": memory_max,
                "local_roofline_ns_by_stage": local_roofline,
                "communication_ns_by_stage": replica_comm,
                "nominal_ns_by_stage": nominal,
                "lower_bound_ns_by_stage": lower,
                "upper_bound_ns_by_stage": upper,
                "nominal_ns": math.fsum(nominal.values()),
                "lower_bound_ns": math.fsum(lower.values()),
                "upper_bound_ns": math.fsum(upper.values()),
            }
        )
    slowest_replica = max(
        range(dp_degree),
        key=lambda index: replica_reports[index]["nominal_ns"],
    )
    selected_replica = replica_reports[slowest_replica]
    total_latency = float(selected_replica["nominal_ns"])
    slowest_rank = max(
        range(chip_count),
        key=lambda rank: math.fsum(
            max(
                rank_stage_compute_ns[rank].get(stage, 0.0),
                rank_stage_memory[rank].get(stage, 0.0),
            )
            for stage in stages
        ),
    )

    aggregate_traffic = _sum_traffic_breakdowns(rank_traffic)
    average_traffic = _average_traffic_breakdown(aggregate_traffic, chip_count)
    representative_traffic = rank_traffic[slowest_rank]
    per_chip_bytes = math.fsum(
        _traffic_total(bucket)
        for bucket in representative_traffic.get("by_stage", {}).values()
    )
    aggregate_bytes = math.fsum(
        _traffic_total(bucket)
        for bucket in aggregate_traffic.get("by_stage", {}).values()
    )
    all_opcode_keys = set().union(*(item.keys() for item in rank_opcode_scales))
    aggregate_opcode_scale = {
        key: math.fsum(item.get(key, 0.0) for item in rank_opcode_scales)
        / chip_count
        for key in all_opcode_keys
    }
    all_kernel_opcode_keys = set().union(
        *(item.keys() for item in rank_kernel_opcode_scales)
    )
    aggregate_kernel_opcode_scale = {
        key: math.fsum(
            item.get(key, 0.0) for item in rank_kernel_opcode_scales
        )
        / chip_count
        for key in all_kernel_opcode_keys
    }
    all_kernel_keys = set().union(*(item.keys() for item in rank_kernel_scales))
    aggregate_kernel_scale = {
        key: math.fsum(item.get(key, 0.0) for item in rank_kernel_scales)
        / chip_count
        for key in all_kernel_keys
    }
    aggregate_stage_scale = {
        stage: math.fsum(
            rank.get(stage, 0.0) for rank in rank_stage_cycles
        )
        / chip_count
        / max(1e-30, math.fsum(stage_opcode_cycles.get(stage, {}).values()))
        for stage in stage_opcode_cycles
    }
    active_plans = [plan for plan in plans if plan["shared_active"]]
    local_tiles = [
        {
            "rank": int(plan["rank"]),
            "dp_rank": int(plan["dp_rank"]),
            "tp_rank": int(plan["tp_rank"]),
            "ep_rank": int(plan["ep_rank"]),
            "batch_start": int(plan["batch_start"]),
            "local_batch_size": int(plan["local_batch_size"]),
            "active_rows": int(plan["active_rows"]),
            "physical_rows": int(plan["physical_rows"]),
            "local_q_heads": int(plan["local_q_heads"]),
            "local_kv_heads": int(plan["local_kv_heads"]),
            "local_intermediate_size": int(plan["local_intermediate_size"]),
            "q_storage_blocks": int(plan["local_head_packing"]["storage_blocks"]),
            "attention_projection": dict(plan["attention_projection_counts"]),
            "attention_core": dict(plan["attention_core_counts"]),
            "ffn": dict(plan["ffn_counts"]),
            "expert": dict(plan["expert"] or {}),
        }
        for plan in plans
    ]
    request_partition = list(global_plan["request_partition"])
    local_batches = [int(item["batch_count"]) for item in request_partition]
    max_local_batch = max(local_batches)
    row_utilization = min(
        (
            float(plan["active_rows"]) / max(1, int(plan["physical_rows"]))
            for plan in active_plans
        ),
        default=1.0,
    )
    equivalent_channels = 128.0 * reference_a100_count / chip_count
    average_stage_memory = {
        stage: math.fsum(rank.get(stage, 0.0) for rank in rank_stage_memory)
        / chip_count
        for stage in stages
    }
    stage_floor = {
        stage: max(
            (rank.get(stage, 0.0) for rank in rank_stage_floor),
            default=0.0,
        )
        for stage in stages
    }
    num_experts = int(global_plan["num_experts"])
    baseline_weight_traffic = _traffic_total(
        dict(original_traffic.get("by_role", {}).get("weight") or {})
    )
    aggregate_weight_traffic = _traffic_total(
        dict(aggregate_traffic.get("by_role", {}).get("weight") or {})
    )
    physical_weight_traffic_replication = (
        aggregate_weight_traffic / baseline_weight_traffic
        if baseline_weight_traffic > 0
        else 0.0
    )
    result = {
        "chip_count": chip_count,
        "reference_a100_count": reference_a100_count,
        "parallel_model": "dp-tp-ep",
        "multi_chip_model": TILE_AWARE_DP_MULTI_CHIP_MODEL,
        "dp_degree": dp_degree,
        "tp_degree": tp_degree,
        "ep_degree": ep_degree,
        "dp_tp_ep_legality": "valid_tile_reconstructed_partition",
        "topology_schema": DP_TP_EP_TOPOLOGY_SCHEMA,
        "request_partition_schema": DP_REQUEST_PARTITION_SCHEMA,
        "request_origin_count": dp_degree * ep_degree,
        "active_request_origin_count": sum(value > 0 for value in local_batches),
        "idle_request_origin_count": sum(value == 0 for value in local_batches),
        "local_batch_by_origin": local_batches,
        "max_local_batch_size": max_local_batch,
        "batch_packing_utilization": row_utilization,
        "request_partition": request_partition,
        "slowest_replica": slowest_replica,
        "slowest_rank": slowest_rank,
        "replica_latency": replica_reports,
        "local_tile_counts_by_rank": local_tiles,
        "parallel_kernel_census_coverage": float(
            trace["parallel_kernel_census_coverage"]
        ),
        "parallel_work_census_coverage": float(
            trace["parallel_kernel_census_coverage"]
        ),
        "matrix_utilization_by_stage": {
            "attention": min(
                (
                    float(plan["active_rows"])
                    / max(1, int(plan["physical_rows"]))
                    * float(plan["local_head_packing"]["head_lane_utilization"])
                    for plan in active_plans
                ),
                default=1.0,
            ),
            "ffn": min(
                (
                    float(plan["active_rows"])
                    / max(1, int(plan["physical_rows"]))
                    * float(plan["local_intermediate_size"])
                    / max(1, int(plan["ffn_counts"]["physical_intermediate"]))
                    for plan in active_plans
                ),
                default=1.0,
            ),
        },
        "vector_utilization_by_stage": {
            "attention": row_utilization,
            "ffn": row_utilization,
        },
        "padding_cycles": rank_padding_cycles[slowest_rank],
        "padding_cycles_by_rank": rank_padding_cycles,
        "replicated_compute_cycles": (
            rank_replicated_compute_cycles[slowest_rank]
        ),
        "replicated_compute_cycles_by_rank": rank_replicated_compute_cycles,
        "tp_rounding_overhead": (
            rank_padding_cycles[slowest_rank]
            / max(
                1e-30,
                math.fsum(rank_stage_cycles[slowest_rank].values()),
            )
        ),
        "dp_batch_imbalance": max_local_batch - min(local_batches),
        "tail_isa_limitation": "active_row_bmm_unavailable",
        "per_chip_stage_compute_latency_ns": dict(
            selected_replica["compute_ns_by_stage"]
        ),
        "rank_stage_compute_latency_ns": rank_stage_compute_ns,
        "per_chip_stage_memory_latency_ns": dict(
            selected_replica["memory_ns_by_stage"]
        ),
        "rank_stage_memory_latency_ns": rank_stage_memory,
        "per_chip_stage_v4_floor_ns": stage_floor,
        "per_chip_stage_roofline_latency_ns": dict(
            selected_replica["nominal_ns_by_stage"]
        ),
        "per_chip_stage_full_overlap_lower_bound_ns": dict(
            selected_replica["lower_bound_ns_by_stage"]
        ),
        "per_chip_stage_no_overlap_upper_bound_ns": dict(
            selected_replica["upper_bound_ns_by_stage"]
        ),
        "full_overlap_lower_bound_ns": max(
            float(item["lower_bound_ns"]) for item in replica_reports
        ),
        "dependency_serial_nominal_ns": total_latency,
        "nominal_stage_model_ns": total_latency,
        "no_overlap_upper_bound_ns": max(
            float(item["upper_bound_ns"]) for item in replica_reports
        ),
        "communication_overlap_bound": (
            "local_compute_hbm_roofline;tp_ep_dependency_serial"
        ),
        "communication_model_schema": DP_COMMUNICATION_SCHEMA,
        "aggregate_hbm_capacity_bytes": aggregate_hbm_capacity_bytes,
        "aggregate_hbm_bandwidth_gbps": aggregate_hbm_bandwidth_gbps,
        "per_chip_hbm_capacity_bytes": aggregate_hbm_capacity_bytes / chip_count,
        "per_chip_hbm_bandwidth_gbps": per_chip_bandwidth,
        "per_chip_equivalent_hbm_channels": equivalent_channels,
        "hbm_channel_calibration_status": (
            "calibrated_channel_anchor"
            if equivalent_channels in {8.0, 32.0, 128.0}
            else "channel_extrapolation"
        ),
        "hbm_channel_extrapolation_ratio": max(
            1.0, equivalent_channels / 128.0, 8.0 / equivalent_channels
        ),
        "per_chip_hbm_traffic": dict(representative_traffic.get("by_stage") or {}),
        "per_chip_hbm_traffic_breakdown": representative_traffic,
        "rank_hbm_traffic_breakdown": rank_traffic,
        "aggregate_hbm_traffic_breakdown": aggregate_traffic,
        "average_per_chip_hbm_traffic_breakdown": average_traffic,
        "average_per_chip_stage_memory_latency_ns": average_stage_memory,
        "hbm_traffic_partition_fidelity": (
            "rank_local_tile_role_physical_traffic_v4"
        ),
        "v4_rank_latency_fidelity": (
            "rank_local_physical_traffic_with_scaled_v4_residual_v4"
        ),
        "v4_local_geometry_reconstruction": False,
        "v4_rank_latency_exact": False,
        "per_chip_hbm_physical_bytes": per_chip_bytes,
        "aggregate_hbm_physical_bytes": aggregate_bytes,
        "per_chip_achieved_bandwidth_gbps": (
            per_chip_bytes / total_latency if total_latency > 0 else 0.0
        ),
        "per_chip_bandwidth_utilization": (
            per_chip_bytes / total_latency / per_chip_bandwidth
            if total_latency > 0 and per_chip_bandwidth > 0
            else 0.0
        ),
        "shared_weight_replication": dp_degree * ep_degree,
        "expert_weight_replication": dp_degree if num_experts > 1 else 1,
        "weight_replication_factor": physical_weight_traffic_replication,
        "physical_weight_traffic_replication": (
            physical_weight_traffic_replication
        ),
        "experts_per_rank": num_experts // ep_degree if num_experts > 1 else 0,
        "expert_bucket_utilization": min(
            (
                float(plan["expert"]["expert_bucket_utilization"])
                for plan in plans
                if plan.get("expert")
            ),
            default=1.0,
        ),
        "parallel_action_scales_by_stage_opcode": aggregate_opcode_scale,
        "parallel_action_scales_by_kernel_opcode": aggregate_kernel_opcode_scale,
        "parallel_action_scales_by_kernel": aggregate_kernel_scale,
        "rank_parallel_action_scales_by_kernel_opcode": rank_kernel_opcode_scales,
        "rank_parallel_action_scales_by_kernel": rank_kernel_scales,
        "parallel_action_scales_by_stage": aggregate_stage_scale,
        "per_chip_compute_scale": math.fsum(
            selected_replica["compute_ns_by_stage"].values()
        ) / max(1e-30, math.fsum(report["stage_compute_latency_ns"].values())),
        "nvlink_port_count": nvlink_port_count,
        "nvlink_peak_bidirectional_bandwidth_gbps": (
            nvlink_port_count * nvlink_port_bidirectional_gbps
        ),
        "nvlink_peak_oneway_bandwidth_gbps": (
            nvlink_port_count * nvlink_port_bidirectional_gbps / 2.0
        ),
        "interconnect_bandwidth_semantics": "architectural_peak_assumption",
        "bandwidth_efficiency": 1.0,
        "interconnect_startup_ns": interconnect_startup_ns,
        **communication,
        "fixed_batch_makespan_ns": total_latency,
        "fixed_batch_requests_per_second": batch_size * 1e9 / total_latency,
        "fixed_batch_tokens_per_second": (
            batch_size * seq_len * 1e9 / total_latency
        ),
        "latency_ns": total_latency,
        "latency_ms": total_latency / 1e6,
        "tile_aware_v4_latency": total_latency,
        "multi_chip_fidelity": {
            "compute": "rank_local_tile_reconstructed_from_compiler_lineage",
            "memory": (
                "rank_local_role_physical_census_and_v4_residual_rescaling;"
                "distributed_address_geometry_not_replayed"
            ),
            "communication": (
                "dependency_serial_tp_striped_ring_ep_port_aware_peak_link"
            ),
            "compiler_isa": "single_chip_planners_reused_without_distributed_isa",
        },
        "kv_cache_fidelity": (
            "whole_request_local_sequence_rank_census"
            if kv_cache_overlay
            else "rank_local_owned_kv_only"
        ),
        "kv_cache_overlay": dict(kv_cache_overlay or {}),
        "warnings": [
            "network fabric is assumed nonblocking and its area/static power are excluded",
            "rank-local HBM V4 residuals are reweighted from compiler physical traffic; distributed address geometry is not replayed",
        ],
    }
    return result
