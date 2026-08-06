#!/usr/bin/env python3
"""Stage-level multi-chip and Matrix-SRAM helpers for the Qwen3 DSE.

The native compiler remains a single-chip compiler.  The v3 analytical path
reconstructs rank-local padded rows, packed heads, projection tiles, attention
blocks, and expert buckets from a compiler-emitted semantic census.  Older
fractional scaling models remain available as explicit comparison modes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from compiler.asm_templates.ffn_projection_plan import (
    FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    build_ffn_projection_plan,
)
from compiler.aten.plena.native_layout import (
    SequencePackingPlan,
    build_attention_head_packing,
)
from compiler.aten.cost_emitter import parallel_kernel_lineage_id


BASE_MATRIX_SRAM_TILES = (2, 4, 8, 16, 32, 64)
PARALLEL_MODELS = ("tp-sp", "tp-only")
MULTI_CHIP_MODELS = (
    "tile-aware-dp-tp-ep-v4",
    "tile-aware-tp-cp-ep-v3",
    "factorized-tp-cp-v2",
    "ideal-linear-lower-bound-v1",
)
TILE_AWARE_MULTI_CHIP_MODEL = "tile-aware-tp-cp-ep-v3"
TILE_AWARE_DP_MULTI_CHIP_MODEL = "tile-aware-dp-tp-ep-v4"
PARALLEL_WORK_AXES = (
    "token_hidden_sharded",
    "attention_pair_head_sharded",
    "tensor_only",
    "replicated_setup",
)
KV_TRAFFIC_ROLES = frozenset({"matrix_kv", "vector_kv", "kv"})
_FFN_ACTIVATION_OPS = frozenset(
    {
        "V_EXP_V",
        "V_MUL_VV",
        "V_MUL_VF",
        "V_ADD_VF",
        "V_SUB_VF",
        "V_RECI_V",
    }
)
_SOFTMAX_ROW_GROUP_OPS = frozenset(
    {
        "V_RED_SUM_ROWS",
        "V_RED_MAX_ROWS",
        "V_SUB_ROWS",
        "V_EXP_ROWS",
        "V_MUL_ROWS_STATS",
        "V_MUL_ROWS_F",
        "V_SFM_MAX_ROWS",
        "V_SFM_SUM_ROWS",
        "V_SFM_FINAL_ROWS",
    }
)
DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS = 900.0
DEFAULT_NVLINK_STARTUP_US = 2.5
ENDPOINT_AREA_MM2_PER_PORT = {
    "optimistic": 15.7,
    "nominal": 24.7,
    "conservative": 38.4,
}
TRAFFIC_FIELDS = (
    "physical_read_bytes",
    "physical_write_bytes",
    "payload_read_bytes",
    "payload_write_bytes",
    "read_requests",
    "write_requests",
)


def parse_positive_int_csv(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse a comma-separated positive integer set with stable ordering."""

    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        values = [int(item) for item in items]
    else:
        values = [int(item) for item in value]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"expected one or more positive integers, got {value!r}")
    return tuple(dict.fromkeys(values))


def valid_tp_degrees(
    model: Mapping[str, Any],
    chip_count: int,
) -> tuple[int, ...]:
    """Return natural head-sharding TP factors for a chip count."""

    if chip_count <= 0:
        raise ValueError("chip_count must be positive")
    q_heads = int(model["num_attention_heads"])
    kv_heads = int(model["num_key_value_heads"])
    return tuple(
        tp
        for tp in range(1, min(chip_count, kv_heads) + 1)
        if chip_count % tp == 0
        and q_heads % tp == 0
        and kv_heads % tp == 0
    )


def validate_tp_cp(
    model: Mapping[str, Any],
    *,
    chip_count: int,
    tp_degree: int,
) -> int:
    """Validate TP and return the derived CP degree."""

    legal = valid_tp_degrees(model, chip_count)
    if tp_degree not in legal:
        raise ValueError(
            f"TP={tp_degree} is illegal for chip_count={chip_count}; "
            f"valid degrees are {legal}"
        )
    return chip_count // tp_degree


def valid_ep_degrees(
    model: Mapping[str, Any],
    cp_degree: int,
    *,
    routing_mode: str | None = None,
) -> tuple[int, ...]:
    """Return EP factors that reuse, rather than multiply, the CP ranks."""

    if cp_degree <= 0:
        raise ValueError("cp_degree must be positive")
    num_experts = int(model.get("num_experts", 0) or 0)
    if num_experts <= 1:
        return (1,)
    if routing_mode not in {None, "fixed-balanced"}:
        return (1,)
    return tuple(
        ep
        for ep in range(1, cp_degree + 1)
        if cp_degree % ep == 0 and num_experts % ep == 0
    )


def validate_tp_cp_ep(
    model: Mapping[str, Any],
    *,
    chip_count: int,
    tp_degree: int,
    ep_degree: int,
    routing_mode: str | None = None,
) -> int:
    cp_degree = validate_tp_cp(
        model,
        chip_count=chip_count,
        tp_degree=tp_degree,
    )
    legal = valid_ep_degrees(
        model,
        cp_degree,
        routing_mode=routing_mode,
    )
    if ep_degree not in legal:
        raise ValueError(
            f"EP={ep_degree} is illegal for CP={cp_degree}, "
            f"num_experts={int(model.get('num_experts', 0) or 0)}, "
            f"routing_mode={routing_mode!r}; valid degrees are {legal}"
        )
    return cp_degree


def zigzag_context_partition(seq_len: int, cp_degree: int) -> dict[str, Any]:
    """Compute exact token and causal-pair load for a two-chunk zigzag CP map."""

    if seq_len <= 0 or cp_degree <= 0:
        raise ValueError("seq_len and cp_degree must be positive")
    chunk_count = 2 * cp_degree
    base, remainder = divmod(seq_len, chunk_count)
    lengths = [base + (1 if index < remainder else 0) for index in range(chunk_count)]
    starts: list[int] = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length

    ranks = []
    total_pairs = seq_len * (seq_len + 1) // 2
    for rank in range(cp_degree):
        indices = tuple(dict.fromkeys((rank, chunk_count - 1 - rank)))
        tokens = sum(lengths[index] for index in indices)
        pairs = 0
        chunks = []
        for index in indices:
            start = starts[index]
            length = lengths[index]
            end = start + length
            # Sum of (query_position + 1) over this contiguous query chunk.
            chunk_pairs = length * (start + 1 + end) // 2
            pairs += chunk_pairs
            chunks.append(
                {
                    "chunk_index": index,
                    "start": start,
                    "length": length,
                    "causal_pairs": chunk_pairs,
                }
            )
        ranks.append(
            {
                "rank": rank,
                "chunks": chunks,
                "tokens": tokens,
                "causal_pairs": pairs,
                "token_fraction": tokens / seq_len,
                "causal_pair_fraction": pairs / total_pairs,
            }
        )

    if sum(item["tokens"] for item in ranks) != seq_len:
        raise AssertionError("zigzag CP token partition is not conservative")
    if sum(item["causal_pairs"] for item in ranks) != total_pairs:
        raise AssertionError("zigzag CP causal-pair partition is not conservative")
    return {
        "scheme": "two_chunk_zigzag_causal_v1",
        "seq_len": seq_len,
        "cp_degree": cp_degree,
        "chunk_lengths": lengths,
        "ranks": ranks,
        "max_local_tokens": max(item["tokens"] for item in ranks),
        "max_token_fraction": max(item["token_fraction"] for item in ranks),
        "max_local_causal_pairs": max(item["causal_pairs"] for item in ranks),
        "max_causal_pair_fraction": max(
            item["causal_pair_fraction"] for item in ranks
        ),
    }


def _compute_axis(stage: str, opcode: str) -> str:
    """Classify compiler compute work by the dimension a TP/CP model shards."""

    if stage in {
        "global/input_load",
        "global/rope_load",
        "global/final_norm",
    }:
        return "token_hidden_sharded"
    if stage == "global/mask_load":
        return "attention_pair_head_sharded"
    if stage.startswith("global/"):
        return "replicated_setup"
    if stage in {"layer/ffn", "layer/moe"} or stage.startswith("layer/moe/"):
        return "token_hidden_sharded"
    if stage == "layer/attention":
        # Dense projection and output-projection MM instructions scale with
        # token rows and hidden shards. Broadcast/transpose MM instructions
        # are QK/PV attention-pair work. Vector/scalar work in this coarse
        # compiler stage is the packed softmax and per-head update path.
        if opcode in {"M_MM", "M_MM_WO", "M_MV", "M_MV_WO"}:
            return "token_hidden_sharded"
        return "attention_pair_head_sharded"
    raise ValueError(
        f"factorized TP/CP has no work-axis classification for {stage}/{opcode}"
    )


def classify_parallel_work_axis(stage: str, opcode: str) -> str:
    """Public shared classifier for compute and power action partitioning."""

    return _compute_axis(stage, opcode)


def build_parallel_work_census(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a complete stage/opcode work census from CostEmitter timing."""

    clock_period_ps = int(
        (report.get("compatibility") or {}).get("clock_period_ps", 1000)
    )
    if clock_period_ps <= 0:
        raise ValueError("CostEmitter clock_period_ps must be positive")
    raw = report.get("stage_compute_opcode_work_cycles")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(
            "factorized-tp-cp-v2 requires "
            "stage_compute_opcode_work_cycles from the current CostEmitter"
        )
    axes: dict[str, dict[str, float]] = {
        axis: defaultdict(float) for axis in PARALLEL_WORK_AXES
    }
    stage_totals: dict[str, float] = defaultdict(float)
    total = 0.0
    classified = 0.0
    for stage, opcodes in raw.items():
        if not isinstance(opcodes, Mapping):
            raise ValueError(f"invalid stage opcode census for {stage!r}")
        for opcode, value in opcodes.items():
            cycles = float(value)
            if cycles < 0:
                raise ValueError(f"negative compute work for {stage}/{opcode}")
            axis = _compute_axis(str(stage), str(opcode))
            axes[axis][str(stage)] += cycles
            stage_totals[str(stage)] += cycles
            total += cycles
            classified += cycles
    expected_ns = sum(
        float(value)
        for value in (report.get("stage_compute_latency_ns") or {}).values()
    )
    expected = expected_ns * 1000.0 / clock_period_ps
    if not math.isclose(total, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "parallel work census does not match CostEmitter stage work: "
            f"census={total}, stage_compute={expected}"
        )
    return {
        "schema": "parallel_work_census_v1",
        "axes": {
            axis: dict(sorted(stages.items()))
            for axis, stages in axes.items()
        },
        "stage_total_cycles": dict(sorted(stage_totals.items())),
        "total_cycles": total,
        "classified_cycles": classified,
        "coverage": 1.0 if total == 0 else classified / total,
        "clock_period_ps": clock_period_ps,
    }


def local_attention_sequence_length(
    seq_len: int,
    chip_count: int,
    parallel_model: str,
    *,
    cp_degree: int | None = None,
    multi_chip_model: str = "ideal-linear-lower-bound-v1",
) -> int:
    """Return the sequence extent retained by one chip's attention schedule."""

    if multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
        if seq_len <= 0 or chip_count <= 0:
            raise ValueError("seq_len and chip_count must be positive")
        # Request parallelism never cuts a sequence.  Every active origin
        # therefore plans the same K/V extent as the single-chip compiler.
        return seq_len
    if multi_chip_model in {
        "factorized-tp-cp-v2",
        TILE_AWARE_MULTI_CHIP_MODEL,
    }:
        if cp_degree is None or cp_degree <= 0:
            raise ValueError("factorized TP/CP requires a positive cp_degree")
        return int(zigzag_context_partition(seq_len, cp_degree)["max_local_tokens"])
    if parallel_model not in PARALLEL_MODELS:
        raise ValueError(f"unsupported parallel model {parallel_model!r}")
    if seq_len <= 0 or chip_count <= 0:
        raise ValueError("seq_len and chip_count must be positive")
    if parallel_model == "tp-sp":
        return math.ceil(seq_len / chip_count)
    return seq_len


def matrix_sram_requirements(
    model: Mapping[str, Any],
    *,
    mlen: int,
    seq_len: int,
    chip_count: int,
    parallel_model: str,
    cp_degree: int | None = None,
    tp_degree: int = 1,
    multi_chip_model: str = "ideal-linear-lower-bound-v1",
) -> dict[str, Any]:
    """Compute compiler-visible Matrix SRAM capacity thresholds.

    A tile is one ``MLEN x MLEN`` Matrix SRAM allocation unit.  The projection
    threshold follows the largest dense K dimension.  The attention threshold
    reserves one K and one V tile per local sequence tile.
    """

    hidden = int(model["hidden_size"])
    intermediate = int(
        model.get("moe_intermediate_size")
        if int(model.get("num_experts", 0) or 0) > 0
        and model.get("moe_intermediate_size") is not None
        else model["intermediate_size"]
    )
    local_seq = local_attention_sequence_length(
        seq_len,
        chip_count,
        parallel_model,
        cp_degree=cp_degree,
        multi_chip_model=multi_chip_model,
    )
    if tp_degree <= 0:
        raise ValueError("tp_degree must be positive")
    # Column-parallel QKV/up/gate retain the full hidden K dimension, while
    # row-parallel O/down only retain the local K shard.  This is the largest
    # live projection footprint visible to one TP rank.
    local_row_parallel_k = math.ceil(intermediate / tp_degree)
    projection_threshold = max(
        math.ceil(hidden / mlen),
        math.ceil(local_row_parallel_k / mlen),
    )
    attention_threshold = 2 * math.ceil(local_seq / mlen)
    useful_saturation = max(
        2,
        projection_threshold,
        attention_threshold,
    )
    return {
        "local_attention_seq_len": local_seq,
        "projection_threshold_tiles": projection_threshold,
        "attention_threshold_tiles": attention_threshold,
        "matrix_sram_useful_saturation_tiles": useful_saturation,
    }


def matrix_sram_search_values(
    model: Mapping[str, Any],
    *,
    mlens: Sequence[int],
    seq_len: int,
    chip_counts: Sequence[int],
    parallel_models: Sequence[str],
    base_values: Sequence[int] = BASE_MATRIX_SRAM_TILES,
) -> tuple[int, ...]:
    """Build a static Optuna domain including useful non-power-of-two points."""

    values = set(parse_positive_int_csv(base_values))
    for mlen in mlens:
        for chip_count in chip_counts:
            for parallel_model in parallel_models:
                requirements = matrix_sram_requirements(
                    model,
                    mlen=int(mlen),
                    seq_len=seq_len,
                    chip_count=int(chip_count),
                    parallel_model=parallel_model,
                )
                values.add(
                    int(requirements["matrix_sram_useful_saturation_tiles"])
                )
    return tuple(sorted(values))


def projection_chunk_metadata(
    model: Mapping[str, Any],
    *,
    mlen: int,
    matrix_sram_tiles: int,
) -> dict[str, Any]:
    """Describe the real K-tile chunking used by dense projection lowering."""

    hidden_tiles = math.ceil(int(model["hidden_size"]) / mlen)
    active_intermediate = int(
        model.get("moe_intermediate_size")
        if int(model.get("num_experts", 0) or 0) > 0
        and model.get("moe_intermediate_size") is not None
        else model["intermediate_size"]
    )
    intermediate_tiles = math.ceil(active_intermediate / mlen)

    def chunks(tiles: int) -> int:
        return math.ceil(tiles / matrix_sram_tiles)

    return {
        "projection_k_tiles": {
            "qkv": hidden_tiles,
            "attention_output": hidden_tiles,
            "ffn_gate_up": hidden_tiles,
            "ffn_down": intermediate_tiles,
        },
        "projection_k_chunks": {
            "qkv": chunks(hidden_tiles),
            "attention_output": chunks(hidden_tiles),
            "ffn_gate_up": chunks(hidden_tiles),
            "ffn_down": chunks(intermediate_tiles),
        },
    }


def _traffic_total(bucket: Mapping[str, Any]) -> float:
    return float(bucket.get("physical_read_bytes", 0)) + float(
        bucket.get("physical_write_bytes", 0)
    )


def _merge_traffic(
    target: dict[str, float],
    source: Mapping[str, Any],
    scale: float,
) -> None:
    for field in TRAFFIC_FIELDS:
        target[field] += float(source.get(field, 0)) * scale


def _stage_role_traffic(
    breakdown: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """Normalize V4 traffic into an exact stage/role cross-tab when available."""

    cross = breakdown.get("by_stage_role") or {}
    if cross:
        result: dict[str, dict[str, dict[str, float]]] = {}
        for key, bucket in cross.items():
            if isinstance(key, str) and "::" in key:
                stage, role = key.split("::", 1)
            else:
                continue
            result.setdefault(stage, {})[role] = {
                field: float(bucket.get(field, 0)) for field in TRAFFIC_FIELDS
            }
        if result:
            return result

    # Older V4 reports only contain independent stage and role marginals.  Use
    # their global role fractions as a transparent fallback; new reports emit
    # the exact cross-tab.
    by_stage = breakdown.get("by_stage") or {}
    by_role = breakdown.get("by_role") or {}
    role_totals = {role: _traffic_total(bucket) for role, bucket in by_role.items()}
    total = sum(role_totals.values())
    fractions = (
        {role: value / total for role, value in role_totals.items()}
        if total > 0
        else {"unknown": 1.0}
    )
    result = {}
    for stage, stage_bucket in by_stage.items():
        result[stage] = {}
        for role, fraction in fractions.items():
            result[stage][role] = {
                field: float(stage_bucket.get(field, 0)) * fraction
                for field in TRAFFIC_FIELDS
            }
    return result


def _traffic_scale_for_role(
    role: str,
    *,
    chip_count: int,
    parallel_model: str,
) -> float:
    if parallel_model == "tp-sp":
        return 1.0 / chip_count
    # Tensor-parallel dense weights are sharded.  Activation, KV, integer and
    # control traffic are conservatively replicated in TP-only mode.
    return 1.0 / chip_count if role == "weight" else 1.0


def _scaled_stage_traffic(
    report: Mapping[str, Any],
    *,
    chip_count: int,
    parallel_model: str,
) -> tuple[dict[str, dict[str, float]], str]:
    breakdown = report.get("hbm_traffic_breakdown") or {}
    stage_roles = _stage_role_traffic(breakdown)
    fidelity = (
        "exact_stage_role_v4_manifest"
        if breakdown.get("by_stage_role")
        else "stage_role_marginal_fallback"
    )
    scaled: dict[str, dict[str, float]] = {}
    for stage, roles in stage_roles.items():
        bucket: dict[str, float] = defaultdict(float)
        for role, role_bucket in roles.items():
            _merge_traffic(
                bucket,
                role_bucket,
                _traffic_scale_for_role(
                    role,
                    chip_count=chip_count,
                    parallel_model=parallel_model,
                ),
            )
        scaled[stage] = dict(bucket)
    return scaled, fidelity


def _scaled_traffic_breakdown(
    report: Mapping[str, Any],
    *,
    chip_count: int,
    parallel_model: str,
) -> tuple[dict[str, dict[str, dict[str, float]]], str]:
    """Scale V4 traffic per precision role and rebuild all useful marginals.

    TP-only shards weight traffic while conservatively replicating activation,
    KV, and integer traffic.  Scaling independent stage/opcode marginals would
    lose that distinction, so new V4 reports retain role cross-tabs.  Older
    reports remain readable through a documented proportional fallback.
    """

    source = dict(report.get("hbm_traffic_breakdown") or {})
    if not source:
        return {}, "traffic_breakdown_unavailable"

    result: dict[str, dict[str, dict[str, float]]] = {
        group: {} for group in source
    }

    def scaled_bucket(bucket: Mapping[str, Any], role: str) -> dict[str, float]:
        scale = _traffic_scale_for_role(
            role,
            chip_count=chip_count,
            parallel_model=parallel_model,
        )
        return {
            field: float(bucket.get(field, 0)) * scale
            for field in TRAFFIC_FIELDS
        }

    for role, bucket in dict(source.get("by_role") or {}).items():
        result.setdefault("by_role", {})[str(role)] = scaled_bucket(bucket, str(role))

    exact_groups = {
        "by_stage_role": 2,
        "by_opcode_role": 2,
        "by_stage_opcode_role": 3,
    }
    exact = True
    for group, parts in exact_groups.items():
        entries = dict(source.get(group) or {})
        if not entries:
            exact = False
            continue
        for key, bucket in entries.items():
            fields = str(key).split("::")
            if len(fields) != parts:
                continue
            role = fields[-1]
            result.setdefault(group, {})[str(key)] = scaled_bucket(bucket, role)

    def aggregate_cross(
        cross_group: str,
        target_group: str,
        key_index: int,
    ) -> None:
        buckets: dict[str, dict[str, float]] = {}
        for key, source_bucket in result.get(cross_group, {}).items():
            parts = str(key).split("::")
            if key_index >= len(parts):
                continue
            target = buckets.setdefault(
                parts[key_index],
                {field: 0.0 for field in TRAFFIC_FIELDS},
            )
            _merge_traffic(target, source_bucket, 1.0)
        if buckets:
            result[target_group] = buckets

    aggregate_cross("by_stage_role", "by_stage", 0)
    aggregate_cross("by_opcode_role", "by_opcode", 0)

    # Backward-compatible reports have only independent marginals.  Preserve
    # their totals using the aggregate role-weighted traffic ratio.
    original_total = sum(
        _traffic_total(bucket)
        for bucket in dict(source.get("by_role") or {}).values()
    )
    scaled_total = sum(
        _traffic_total(bucket)
        for bucket in result.get("by_role", {}).values()
    )
    fallback_scale = scaled_total / original_total if original_total > 0 else 0.0
    for group in ("by_stage", "by_opcode"):
        if result.get(group):
            continue
        result[group] = {
            str(key): {
                field: float(bucket.get(field, 0)) * fallback_scale
                for field in TRAFFIC_FIELDS
            }
            for key, bucket in dict(source.get(group) or {}).items()
        }

    return (
        result,
        "exact_role_cross_tab_v4" if exact else "role_weighted_marginal_fallback",
    )


def _aggregate_traffic_breakdown(
    per_chip: Mapping[str, Mapping[str, Mapping[str, float]]],
    chip_count: int,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(group): {
            str(key): {
                field: float(bucket.get(field, 0)) * chip_count
                for field in TRAFFIC_FIELDS
            }
            for key, bucket in entries.items()
        }
        for group, entries in per_chip.items()
    }


def _apply_attention_kv_overlay(
    breakdown: dict[str, dict[str, dict[str, float]]],
    *,
    relative_scale: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Replace attention-KV traffic with exact local-cache occurrence scaling."""

    if relative_scale < 0:
        raise ValueError("attention K/V traffic scale must be nonnegative")
    if relative_scale == 1.0:
        return breakdown

    for group in ("by_stage_role", "by_stage_opcode_role"):
        for key, bucket in breakdown.get(group, {}).items():
            parts = str(key).split("::")
            if len(parts) < 2:
                continue
            stage = parts[0]
            role = parts[-1]
            if stage == "layer/attention" and role in KV_TRAFFIC_ROLES:
                for field in TRAFFIC_FIELDS:
                    bucket[field] = float(bucket.get(field, 0.0)) * relative_scale

    def rebuild(
        source_group: str,
        target_group: str,
        key_index: int,
    ) -> None:
        rebuilt: dict[str, dict[str, float]] = {}
        for key, bucket in breakdown.get(source_group, {}).items():
            parts = str(key).split("::")
            if key_index >= len(parts):
                continue
            target = rebuilt.setdefault(
                parts[key_index],
                {field: 0.0 for field in TRAFFIC_FIELDS},
            )
            _merge_traffic(target, bucket, 1.0)
        if rebuilt:
            breakdown[target_group] = rebuilt

    rebuild("by_stage_role", "by_stage", 0)
    rebuild("by_stage_role", "by_role", 1)
    rebuild("by_stage_opcode_role", "by_opcode", 1)
    return breakdown


def _compute_stage_scale(
    report: Mapping[str, Any],
    *,
    chip_count: int,
    parallel_model: str,
) -> tuple[float, dict[str, float], str]:
    stage_compute = {
        str(stage): float(value)
        for stage, value in (report.get("stage_compute_latency_ns") or {}).items()
    }
    if parallel_model == "tp-sp":
        return (
            1.0 / chip_count,
            {stage: value / chip_count for stage, value in stage_compute.items()},
            "optimistic_all_compute_partitioned",
        )

    category = report.get("category_latency_ns") or {}
    matrix = float(category.get("matrix_compute", 0.0))
    non_matrix = sum(
        float(category.get(name, 0.0))
        for name in ("vector_compute", "scalar_compute", "control")
    )
    total = matrix + non_matrix
    scale = ((matrix / chip_count) + non_matrix) / total if total > 0 else 1.0
    return (
        scale,
        {stage: value * scale for stage, value in stage_compute.items()},
        "matrix_sharded_nonmatrix_replicated_stage_proportional",
    )


def _communication_by_stage(
    model: Mapping[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    chip_count: int,
    fp_width_bits: int,
    one_way_link_bandwidth_gbps: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if chip_count == 1:
        return {}, {}
    if one_way_link_bandwidth_gbps <= 0:
        raise ValueError("one-way link bandwidth must be positive")
    layers = int(model["num_hidden_layers"])
    activation_bytes_per_layer = (
        seq_len * batch_size * int(model["hidden_size"]) * fp_width_bits / 8.0
    )
    # One ring collective follows attention and one follows FFN.  The
    # 2*(N-1)/N factor is the per-chip send volume of a ring all-reduce.  This
    # is intentionally an optimistic peak-link lower bound.
    bytes_per_collective = (
        2.0 * (chip_count - 1) / chip_count * activation_bytes_per_layer
    )
    stage_bytes = {
        "layer/attention": bytes_per_collective * layers,
        "layer/ffn": bytes_per_collective * layers,
    }
    # Decimal GB/s is numerically bytes/ns.
    stage_latency = {
        stage: value / one_way_link_bandwidth_gbps
        for stage, value in stage_bytes.items()
    }
    return stage_bytes, stage_latency


def _factorized_traffic_scale(
    role: str,
    *,
    tp_degree: int,
    max_token_fraction: float,
) -> float:
    if role == "weight":
        return 1.0 / tp_degree
    if role in {"activation", "matrix_kv", "vector_kv", "kv"}:
        return max_token_fraction / tp_degree
    if role == "integer":
        return max_token_fraction
    raise ValueError(
        f"factorized TP/CP has no HBM partition rule for precision role {role!r}"
    )


def _factorized_traffic_breakdown(
    report: Mapping[str, Any],
    *,
    tp_degree: int,
    max_token_fraction: float,
) -> tuple[dict[str, dict[str, dict[str, float]]], str]:
    source = dict(report.get("hbm_traffic_breakdown") or {})
    required = ("by_role", "by_stage_role", "by_stage_opcode_role")
    missing = [name for name in required if not source.get(name)]
    if missing:
        raise ValueError(
            "factorized TP/CP requires exact V4 role cross-tabs; missing "
            + ", ".join(missing)
        )
    result: dict[str, dict[str, dict[str, float]]] = {}

    def scale_bucket(bucket: Mapping[str, Any], role: str) -> dict[str, float]:
        scale = _factorized_traffic_scale(
            role,
            tp_degree=tp_degree,
            max_token_fraction=max_token_fraction,
        )
        return {
            field: float(bucket.get(field, 0.0)) * scale
            for field in TRAFFIC_FIELDS
        }

    for group, role_parts in (
        ("by_role", 1),
        ("by_stage_role", 2),
        ("by_opcode_role", 2),
        ("by_stage_opcode_role", 3),
    ):
        entries = dict(source.get(group) or {})
        if not entries and group == "by_opcode_role":
            continue
        for key, bucket in entries.items():
            parts = str(key).split("::")
            if len(parts) != role_parts:
                raise ValueError(f"invalid {group} key {key!r}")
            role = parts[-1] if role_parts > 1 else str(key)
            result.setdefault(group, {})[str(key)] = scale_bucket(bucket, role)

    def rebuild(source_group: str, target_group: str, key_index: int) -> None:
        rebuilt: dict[str, dict[str, float]] = {}
        for key, bucket in result.get(source_group, {}).items():
            parts = str(key).split("::")
            target = rebuilt.setdefault(
                parts[key_index],
                {field: 0.0 for field in TRAFFIC_FIELDS},
            )
            _merge_traffic(target, bucket, 1.0)
        result[target_group] = rebuilt

    rebuild("by_stage_role", "by_stage", 0)
    rebuild("by_stage_opcode_role", "by_opcode", 1)
    return result, "exact_role_cross_tab_factorized_tp_cp_v2"


def _scale_factorized_compute(
    census: Mapping[str, Any],
    *,
    tp_degree: int,
    max_token_fraction: float,
    max_causal_pair_fraction: float,
) -> tuple[dict[str, float], dict[str, float]]:
    scales = {
        "token_hidden_sharded": max_token_fraction / tp_degree,
        "attention_pair_head_sharded": max_causal_pair_fraction / tp_degree,
        "tensor_only": 1.0 / tp_degree,
        "replicated_setup": 1.0,
    }
    by_stage: dict[str, float] = defaultdict(float)
    for axis, stages in dict(census["axes"]).items():
        scale = scales[axis]
        for stage, cycles in dict(stages).items():
            by_stage[str(stage)] += float(cycles) * scale
    cycle_to_ns = float(census["clock_period_ps"]) / 1000.0
    return (
        {
            stage: cycles * cycle_to_ns
            for stage, cycles in sorted(by_stage.items())
        },
        scales,
    )


def _factorized_communication(
    model: Mapping[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    tp_degree: int,
    cp_degree: int,
    local_tokens: int,
    fp_width_bits: int,
    kv_width_bits: float,
    one_way_link_bandwidth_gbps: float,
    startup_latency_ns: float,
) -> dict[str, Any]:
    if one_way_link_bandwidth_gbps <= 0:
        raise ValueError("one-way link bandwidth must be positive")
    if startup_latency_ns < 0:
        raise ValueError("interconnect startup latency must be nonnegative")
    layers = int(model["num_hidden_layers"])
    activation_bytes = (
        local_tokens
        * batch_size
        * int(model["hidden_size"])
        * fp_width_bits
        / 8.0
    )
    tp_bytes_per_collective = (
        2.0 * (tp_degree - 1) / tp_degree * activation_bytes
        if tp_degree > 1
        else 0.0
    )
    tp_latency_per_collective = (
        2.0 * (tp_degree - 1) * startup_latency_ns
        + tp_bytes_per_collective / one_way_link_bandwidth_gbps
        if tp_degree > 1
        else 0.0
    )
    local_kv_bytes = (
        2.0
        * batch_size
        * local_tokens
        * (int(model["num_key_value_heads"]) / tp_degree)
        * int(model["head_dim"])
        * kv_width_bits
        / 8.0
    )
    cp_bytes_per_layer = (
        (cp_degree - 1) * local_kv_bytes if cp_degree > 1 else 0.0
    )
    cp_latency_per_layer = (
        (cp_degree - 1) * startup_latency_ns
        + cp_bytes_per_layer / one_way_link_bandwidth_gbps
        if cp_degree > 1
        else 0.0
    )
    tp_stage_bytes = tp_bytes_per_collective * layers
    tp_stage_latency = tp_latency_per_collective * layers
    cp_stage_bytes = cp_bytes_per_layer * layers
    cp_stage_latency = cp_latency_per_layer * layers
    chip_count = tp_degree * cp_degree
    global_kv_bytes = (
        2.0
        * batch_size
        * seq_len
        * int(model["num_key_value_heads"])
        * int(model["head_dim"])
        * kv_width_bits
        / 8.0
    )
    aggregate_tp_bytes = 2.0 * tp_stage_bytes * chip_count
    aggregate_cp_bytes = (cp_degree - 1) * global_kv_bytes * layers
    return {
        "tp_collective_bytes_by_stage": {
            "layer/attention": tp_stage_bytes,
            "layer/ffn": tp_stage_bytes,
        },
        "tp_collective_latency_ns_by_stage": {
            "layer/attention": tp_stage_latency,
            "layer/ffn": tp_stage_latency,
        },
        "cp_kv_ring_bytes_by_stage": {
            "layer/attention": cp_stage_bytes,
        },
        "cp_kv_ring_latency_ns_by_stage": {
            "layer/attention": cp_stage_latency,
        },
        "tp_collective_bytes": 2.0 * tp_stage_bytes,
        "tp_collective_latency_ns": 2.0 * tp_stage_latency,
        "cp_kv_ring_bytes": cp_stage_bytes,
        "cp_kv_ring_latency_ns": cp_stage_latency,
        "aggregate_tp_collective_bytes": aggregate_tp_bytes,
        "aggregate_cp_kv_ring_bytes": aggregate_cp_bytes,
        "aggregate_interconnect_bytes": (
            aggregate_tp_bytes + aggregate_cp_bytes
        ),
        "local_kv_bytes_per_layer": local_kv_bytes,
    }


def _ceil_to(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError(f"value and multiple must be positive, got {value}, {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _balanced_part(total: int, parts: int, rank: int) -> int:
    if total < 0 or parts <= 0 or not 0 <= rank < parts:
        raise ValueError(
            f"invalid balanced partition total={total}, parts={parts}, rank={rank}"
        )
    base, remainder = divmod(total, parts)
    return base + int(rank < remainder)


def _packed_rows_for_slab(*, batch_size: int, seq_len: int, mlen: int) -> int:
    """Return compiler-planned physical rows for one local CP slab."""

    return SequencePackingPlan.build(
        batch_size=batch_size,
        seq_len=seq_len,
        mlen=mlen,
        mode="compact",
    ).compile_seq_rows


def _rank_row_shape(
    context: Mapping[str, Any],
    *,
    cp_rank: int,
    batch_size: int,
    mlen: int,
    global_physical_rows: int,
) -> dict[str, Any]:
    cp_degree = int(context["cp_degree"])
    rank = dict(context["ranks"][cp_rank])
    if cp_degree == 1:
        physical_rows = global_physical_rows
    else:
        physical_rows = sum(
            _packed_rows_for_slab(
                batch_size=batch_size,
                seq_len=int(chunk["length"]),
                mlen=mlen,
            )
            for chunk in rank["chunks"]
            if int(chunk["length"]) > 0
        )
    return {
        "cp_rank": cp_rank,
        "active_rows": int(rank["tokens"]) * batch_size,
        "physical_rows": physical_rows,
        "chunks": tuple(dict(chunk) for chunk in rank["chunks"]),
        "token_fraction": float(rank["token_fraction"]),
        "causal_pair_fraction": float(rank["causal_pair_fraction"]),
    }


def _head_storage_width(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    hlen: int,
    mlen: int,
    logical_broadcast: int,
) -> dict[str, int | float]:
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError(
            f"invalid local GQA heads q={q_heads}, kv={kv_heads}"
        )
    ratio = q_heads // kv_heads
    packing = build_attention_head_packing(
        mlen=mlen,
        hlen=hlen,
        head_dim=head_dim,
        logical_broadcast_amount=logical_broadcast,
        gqa_ratio=ratio,
        num_kv_heads=kv_heads,
        mode="compact",
    )
    return {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_ratio": ratio,
        "physical_broadcast": packing.broadcast_amount,
        "chunks_per_kv": packing.chunks_per_kv,
        "logical_groups": packing.logical_group_count,
        "groups_per_storage_block": packing.groups_per_storage_block,
        "storage_blocks": packing.storage_block_count,
        "total_q_dim": packing.total_q_dim,
        "head_lane_utilization": packing.head_lane_utilization,
        "head_dim": head_dim,
    }


def _projection_counts(
    *,
    physical_rows: int,
    k_size: int,
    out_size: int,
    mlen: int,
    blen: int,
    max_k_tiles: int,
) -> tuple[int, int]:
    plan = build_ffn_projection_plan(
        schedule=FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
        mlen=mlen,
        blen=blen,
        batch_rows=_ceil_to(physical_rows, blen),
        k_size=_ceil_to(k_size, mlen),
        out_size=_ceil_to(out_size, mlen),
        max_k_tiles=max_k_tiles,
    )
    return plan.matrix_compute_count, plan.matrix_writeout_count


def _ffn_projection_counts(
    *,
    physical_rows: int,
    hidden_size: int,
    intermediate_size: int,
    mlen: int,
    blen: int,
    max_k_tiles: int,
) -> dict[str, int]:
    hidden = _ceil_to(hidden_size, mlen)
    intermediate = _ceil_to(intermediate_size, mlen)
    up_mm, up_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=hidden,
        out_size=intermediate,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    down_mm, down_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=intermediate,
        out_size=hidden,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    return {
        "M_MM": 2 * up_mm + down_mm,
        "M_MM_WO": 2 * up_wo + down_wo,
        "physical_intermediate": intermediate,
    }


def _attention_projection_counts(
    *,
    physical_rows: int,
    hidden_size: int,
    q_storage_width: int,
    kv_heads: int,
    mlen: int,
    blen: int,
    max_k_tiles: int,
) -> dict[str, int]:
    hidden = _ceil_to(hidden_size, mlen)
    q_mm, q_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=hidden,
        out_size=q_storage_width,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    o_mm, o_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=q_storage_width,
        out_size=hidden,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    kv_mm, kv_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=hidden,
        out_size=mlen,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    # K and V each have one projection per local KV head.  RoPE projections
    # retain the same TP head partition and are represented by one MLEN square
    # projection per KV head plus one per packed Q storage block.
    rope_mm, rope_wo = _projection_counts(
        physical_rows=physical_rows,
        k_size=mlen,
        out_size=mlen,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    q_blocks = q_storage_width // mlen
    return {
        "M_MM": (
            q_mm
            + o_mm
            + 2 * kv_heads * kv_mm
            + (kv_heads + q_blocks) * rope_mm
        ),
        "M_MM_WO": (
            q_wo
            + o_wo
            + 2 * kv_heads * kv_wo
            + (kv_heads + q_blocks) * rope_wo
        ),
    }


def _attention_block_work(
    *,
    rank_shape: Mapping[str, Any],
    batch_size: int,
    mlen: int,
    head_groups: int,
    broadcast_heads: int = 1,
    softmax_row_lanes: int = 1,
) -> dict[str, int]:
    if broadcast_heads <= 0 or softmax_row_lanes <= 0:
        raise ValueError("broadcast_heads and softmax_row_lanes must be positive")
    occurrences = 0
    softmax_row_groups = 0
    q_tail_rows = 0
    tail_occurrences = 0
    for chunk in rank_shape["chunks"]:
        start = int(chunk["start"])
        length = int(chunk["length"])
        if length <= 0:
            continue
        if length <= mlen:
            slot_rows = 1 << (length - 1).bit_length()
            pack_factor = max(1, min(batch_size, mlen // slot_rows))
        else:
            pack_factor = 1
        groups = math.ceil(batch_size / pack_factor)
        q_blocks = math.ceil(length / mlen)
        for q_block in range(q_blocks):
            local_start = q_block * mlen
            valid_rows = min(mlen, length - local_start)
            global_last = start + local_start + valid_rows
            visible_k_blocks = math.ceil(global_last / mlen)
            block_occurrences = groups * visible_k_blocks * head_groups
            occurrences += block_occurrences
            # One broadcast BMM feeds every packed Q head.  Packed batch
            # segments remain separate causal domains, so a row group cannot
            # cross a segment boundary even when two tails would fit together.
            groups_per_bmm = (
                broadcast_heads
                * pack_factor
                * math.ceil(valid_rows / softmax_row_lanes)
            )
            softmax_row_groups += block_occurrences * groups_per_bmm
            if valid_rows < mlen:
                q_tail_rows += valid_rows * groups
                tail_occurrences += block_occurrences
    return {
        "bmm_occurrences": occurrences,
        "softmax_row_groups": softmax_row_groups,
        "q_tail_rows": q_tail_rows,
        "tail_bmm_occurrences": tail_occurrences,
        "tail_full_width_work_cycles": tail_occurrences,
    }


def _parallel_kernel_entries(
    report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    trace = dict(report.get("trace") or {})
    census_schema = str(trace.get("parallel_kernel_census_schema", ""))
    if census_schema != "parallel_kernel_census_v2_schedule_lineage":
        raise ValueError(
            "tile-aware-tp-cp-ep-v3 requires schedule-lineage census schema; "
            f"got {census_schema!r}"
        )
    entries = [
        dict(entry) for entry in trace.get("parallel_kernel_census") or ()
    ]
    if not entries:
        raise ValueError(
            "tile-aware-tp-cp-ep-v3 requires CostTrace schema v7 "
            "parallel_kernel_census"
        )
    totals: dict[tuple[str, str], int] = defaultdict(int)
    covered = 0
    for entry in entries:
        key = (str(entry["stage"]), str(entry["opcode"]))
        count = int(entry.get("count", 0))
        if count <= 0:
            raise ValueError(f"invalid parallel kernel census count: {entry}")
        totals[key] += count
        covered += count
    if float(trace.get("parallel_kernel_census_coverage", 0.0)) != 1.0:
        raise ValueError("parallel kernel census coverage must be 100%")
    if covered <= 0:
        raise ValueError("parallel kernel census is empty")
    return entries, dict(totals)


def _scale_traffic_breakdown(
    source: Mapping[str, Any],
    *,
    stage_role_scales: Mapping[tuple[str, str], float],
) -> dict[str, dict[str, dict[str, float]]]:
    required = ("by_role", "by_stage_role", "by_stage_opcode_role")
    missing = [name for name in required if not source.get(name)]
    if missing:
        raise ValueError(
            "tile-aware model requires exact V4 role cross-tabs; missing "
            + ", ".join(missing)
        )
    result: dict[str, dict[str, dict[str, float]]] = {}
    for group in ("by_stage_role", "by_stage_opcode_role"):
        for key, raw_bucket in dict(source.get(group) or {}).items():
            parts = str(key).split("::")
            stage = parts[0]
            role = parts[-1]
            scale = float(stage_role_scales.get((stage, role), 0.0))
            result.setdefault(group, {})[str(key)] = {
                field: float(dict(raw_bucket).get(field, 0.0)) * scale
                for field in TRAFFIC_FIELDS
            }

    def rebuild(group: str, target: str, index: int) -> None:
        buckets: dict[str, dict[str, float]] = {}
        for key, bucket in result[group].items():
            selected = str(key).split("::")[index]
            output = buckets.setdefault(
                selected, {field: 0.0 for field in TRAFFIC_FIELDS}
            )
            _merge_traffic(output, bucket, 1.0)
        result[target] = buckets

    rebuild("by_stage_role", "by_stage", 0)
    rebuild("by_stage_role", "by_role", 1)
    rebuild("by_stage_opcode_role", "by_opcode", 1)
    return result


def _sum_traffic_breakdowns(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for value in values:
        for group, entries in value.items():
            for key, bucket in dict(entries).items():
                target = result.setdefault(str(group), {}).setdefault(
                    str(key), {field: 0.0 for field in TRAFFIC_FIELDS}
                )
                _merge_traffic(target, dict(bucket), 1.0)
    return result


def _average_traffic_breakdown(
    aggregate: Mapping[str, Any],
    chip_count: int,
) -> dict[str, dict[str, dict[str, float]]]:
    if chip_count <= 0:
        raise ValueError("chip_count must be positive")
    return {
        str(group): {
            str(key): {
                field: float(dict(bucket).get(field, 0.0)) / chip_count
                for field in TRAFFIC_FIELDS
            }
            for key, bucket in dict(entries).items()
        }
        for group, entries in aggregate.items()
    }


def _memory_latency_for_traffic(
    report: Mapping[str, Any],
    traffic: Mapping[str, Any],
    *,
    per_chip_bandwidth_gbps: float,
    baseline_bandwidth_gbps: float,
) -> tuple[dict[str, float], dict[str, float]]:
    original = dict((report.get("hbm_traffic_breakdown") or {}).get("by_stage") or {})
    current = dict(traffic.get("by_stage") or {})
    baseline_memory = {
        str(stage): float(value)
        for stage, value in (report.get("hbm_stage_latency_ns") or {}).items()
    }
    exact_floor = {
        str(stage): float(value)
        for stage, value in (
            (report.get("compatibility") or {}).get(
                "stage_theoretical_floor_ns", {}
            )
            or {}
        ).items()
    }
    baseline_stage_opcode = {
        str(key): float(value)
        for key, value in (
            report.get("hbm_stage_opcode_latency_ns") or {}
        ).items()
    }
    original_stage_opcode = dict(
        (report.get("hbm_traffic_breakdown") or {}).get(
            "by_stage_opcode_role", {}
        )
        or {}
    )
    current_stage_opcode = dict(
        traffic.get("by_stage_opcode_role") or {}
    )

    def opcode_buckets(
        source: Mapping[str, Any],
    ) -> dict[tuple[str, str], dict[str, float]]:
        result: dict[tuple[str, str], dict[str, float]] = {}
        for key, raw in source.items():
            stage, opcode, _role = str(key).split("::", 2)
            target = result.setdefault(
                (stage, opcode),
                {field: 0.0 for field in TRAFFIC_FIELDS},
            )
            _merge_traffic(target, dict(raw), 1.0)
        return result

    old_opcode_traffic = opcode_buckets(original_stage_opcode)
    new_opcode_traffic = opcode_buckets(current_stage_opcode)
    stage_memory: dict[str, float] = {}
    stage_floor: dict[str, float] = {}
    for stage in sorted(set(original) | set(current)):
        old_bucket = dict(original.get(stage) or {})
        new_bucket = dict(current.get(stage) or {})
        old_bytes = _traffic_total(old_bucket)
        new_bytes = _traffic_total(new_bucket)
        old_requests = float(old_bucket.get("read_requests", 0.0)) + float(
            old_bucket.get("write_requests", 0.0)
        )
        new_requests = float(new_bucket.get("read_requests", 0.0)) + float(
            new_bucket.get("write_requests", 0.0)
        )
        old_floor = exact_floor.get(stage, 0.0)
        new_floor = (
            old_floor
            * new_bytes
            / old_bytes
            * baseline_bandwidth_gbps
            / per_chip_bandwidth_gbps
            if old_bytes > 0
            else 0.0
        )
        residual = max(baseline_memory.get(stage, 0.0) - old_floor, 0.0)
        stage_floor[stage] = new_floor
        if not baseline_stage_opcode:
            stage_memory[stage] = (
                new_floor
                + residual * new_requests / old_requests
                if old_requests > 0
                else new_floor
            )
            continue
        stage_residual = 0.0
        for (opcode_stage, opcode), old_opcode_bucket in (
            old_opcode_traffic.items()
        ):
            if opcode_stage != stage:
                continue
            old_opcode_bytes = _traffic_total(old_opcode_bucket)
            old_opcode_requests = float(
                old_opcode_bucket.get("read_requests", 0.0)
            ) + float(old_opcode_bucket.get("write_requests", 0.0))
            new_opcode_bucket = new_opcode_traffic.get(
                (stage, opcode), {}
            )
            new_opcode_requests = float(
                new_opcode_bucket.get("read_requests", 0.0)
            ) + float(new_opcode_bucket.get("write_requests", 0.0))
            floor_share = (
                old_floor * old_opcode_bytes / old_bytes
                if old_bytes > 0
                else 0.0
            )
            base = baseline_stage_opcode.get(
                f"{stage}::{opcode}", floor_share
            )
            opcode_residual = max(base - floor_share, 0.0)
            if old_opcode_requests > 0:
                stage_residual += (
                    opcode_residual
                    * new_opcode_requests
                    / old_opcode_requests
                )
        stage_memory[stage] = new_floor + stage_residual
    return stage_memory, stage_floor


def _fixed_balanced_route_counts_for_rank(
    *,
    rank_shape: Mapping[str, Any],
    batch_size: int,
    seq_len: int,
    num_experts: int,
    top_k: int,
) -> list[int]:
    counts = [0] * num_experts
    for batch in range(batch_size):
        batch_base = batch * seq_len
        for chunk in rank_shape["chunks"]:
            start = int(chunk["start"])
            end = start + int(chunk["length"])
            for token in range(start, end):
                route_base = (batch_base + token) * top_k
                for route_rank in range(top_k):
                    counts[(route_base + route_rank) % num_experts] += 1
    return counts


def _tile_aware_rank_plans(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    tp_degree: int,
    cp_degree: int,
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
            "tile-aware model requires trace hardware and native_layout metadata"
        )
    mlen = int(hardware["mlen"])
    blen = int(hardware["blen"])
    max_k_tiles = max(1, int(hardware.get("mram_tile_capacity", 1)))
    hlen = int(hardware.get("hlen", model.get("head_dim", 128)))
    global_rows = int(native_layout["physical_rows"])
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
    context = zigzag_context_partition(seq_len, cp_degree)
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
    global_attn_projection = _attention_projection_counts(
        physical_rows=global_rows,
        hidden_size=hidden,
        q_storage_width=int(global_head["total_q_dim"]),
        kv_heads=kv_heads,
        mlen=mlen,
        blen=blen,
        max_k_tiles=max_k_tiles,
    )
    global_rank_shape = {
        "chunks": ({"start": 0, "length": seq_len},),
    }
    global_attn_core = _attention_block_work(
        rank_shape=global_rank_shape,
        batch_size=batch_size,
        mlen=mlen,
        head_groups=int(global_head["logical_groups"]),
        broadcast_heads=int(global_head["physical_broadcast"]),
        softmax_row_lanes=int(trace.get("softmax_row_lanes", 1)),
    )
    num_experts = int(workload.get("num_experts") or model.get("num_experts", 0) or 0)
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
        else {"M_MM": 0, "M_MM_WO": 0}
    )
    top_k = int(
        workload.get("experts_per_token")
        or model.get("num_experts_per_tok", 0)
        or model.get("experts_per_token", 0)
        or 0
    )
    routing_mode = trace.get("compiler_metadata", {}).get("moe_routing_mode")
    if num_experts > 1 and routing_mode not in {None, "fixed-balanced"}:
        raise ValueError(
            "tile-aware MoE multi-chip supports only fixed-balanced routing"
        )

    rank_rows = [
        _rank_row_shape(
            context,
            cp_rank=rank,
            batch_size=batch_size,
            mlen=mlen,
            global_physical_rows=global_rows,
        )
        for rank in range(cp_degree)
    ]
    route_counts_by_cp = (
        [
            _fixed_balanced_route_counts_for_rank(
                rank_shape=shape,
                batch_size=batch_size,
                seq_len=seq_len,
                num_experts=num_experts,
                top_k=top_k,
            )
            for shape in rank_rows
        ]
        if num_experts > 1 and top_k > 0
        else []
    )
    global_route_counts = (
        [
            (seq_len * batch_size * top_k) // num_experts
            + int(
                expert
                < (seq_len * batch_size * top_k) % num_experts
            )
            for expert in range(num_experts)
        ]
        if route_counts_by_cp
        else []
    )

    plans: list[dict[str, Any]] = []
    for tp_rank in range(tp_degree):
        local_hidden = _balanced_part(hidden, tp_degree, tp_rank)
        local_q_heads = _balanced_part(q_heads, tp_degree, tp_rank)
        local_kv_heads = _balanced_part(kv_heads, tp_degree, tp_rank)
        local_intermediate = _balanced_part(intermediate, tp_degree, tp_rank)
        local_head = _head_storage_width(
            q_heads=local_q_heads,
            kv_heads=local_kv_heads,
            head_dim=head_dim,
            hlen=hlen,
            mlen=mlen,
            logical_broadcast=logical_broadcast,
        )
        for cp_rank, row_shape in enumerate(rank_rows):
            local_ffn = _ffn_projection_counts(
                physical_rows=int(row_shape["physical_rows"]),
                hidden_size=hidden,
                intermediate_size=local_intermediate,
                mlen=mlen,
                blen=blen,
                max_k_tiles=max_k_tiles,
            )
            local_attn_projection = _attention_projection_counts(
                physical_rows=int(row_shape["physical_rows"]),
                hidden_size=hidden,
                q_storage_width=int(local_head["total_q_dim"]),
                kv_heads=local_kv_heads,
                mlen=mlen,
                blen=blen,
                max_k_tiles=max_k_tiles,
            )
            local_attn_core = _attention_block_work(
                rank_shape=row_shape,
                batch_size=batch_size,
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
                else {"M_MM": 0, "M_MM_WO": 0}
            )
            token_scale = (
                int(row_shape["active_rows"]) / (seq_len * batch_size)
            )
            physical_row_scale = int(row_shape["physical_rows"]) / global_rows
            ffn_tensor_scale = (
                int(row_shape["physical_rows"])
                * int(local_ffn["physical_intermediate"])
                / (
                    global_rows
                    * int(global_ffn["physical_intermediate"])
                )
            )
            segmented_norm_scale = (
                int(row_shape["active_rows"])
                * (local_q_heads + local_kv_heads)
                / (
                    seq_len * batch_size * (q_heads + kv_heads)
                )
            )
            scales = {
                "token_replicated_hidden": token_scale,
                "token_tensor_sharded": segmented_norm_scale,
                "attention_projection_tiled": (
                    local_attn_projection["M_MM"]
                    / max(1, global_attn_projection["M_MM"])
                ),
                "attention_head_pair_sharded": (
                    local_attn_core["bmm_occurrences"]
                    / max(1, global_attn_core["bmm_occurrences"])
                ),
                "ffn_projection_tiled": (
                    local_ffn["M_MM"] / max(1, global_ffn["M_MM"])
                ),
                "row_parallel_projection": token_scale,
                "expert_tensor_sharded": token_scale / max(1, tp_degree),
                "replicated_setup": 1.0,
            }
            q_tensor_scale = (
                int(row_shape["physical_rows"])
                * int(local_head["total_q_dim"])
                / (
                    global_rows
                    * int(global_head["total_q_dim"])
                )
            )
            expert = None
            if route_counts_by_cp:
                group_start = (cp_rank // ep_degree) * ep_degree
                group_counts = [
                    sum(
                        route_counts_by_cp[source][expert_id]
                        for source in range(group_start, group_start + ep_degree)
                    )
                    for expert_id in range(num_experts)
                ]
                owner = cp_rank - group_start
                experts_per_rank = num_experts // ep_degree
                expert_start = owner * experts_per_rank
                expert_end = expert_start + experts_per_rank
                owned_counts = group_counts[expert_start:expert_end]
                padded_rows = [
                    math.ceil(count / blen) * blen if count else 0
                    for count in owned_counts
                ]
                global_padded = [
                    math.ceil(count / blen) * blen if count else 0
                    for count in global_route_counts
                ]
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
                global_expert_mm = 0
                global_expert_wo = 0
                for rows in global_padded:
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
                local_routes = sum(route_counts_by_cp[cp_rank])
                scales.update(
                    {
                        "expert_tensor_sharded": (
                            local_expert_mm / max(1, global_expert_mm)
                        ),
                        "row_parallel_projection": (
                            token_scale / max(1, tp_degree)
                        ),
                    }
                )
                remote_routes = sum(
                    count
                    for expert_id, count in enumerate(
                        route_counts_by_cp[cp_rank]
                    )
                    if expert_id // experts_per_rank != owner
                )
                expert = {
                    "ep_group": cp_rank // ep_degree,
                    "ep_rank": owner,
                    "experts_per_rank": experts_per_rank,
                    "expert_start": expert_start,
                    "expert_end": expert_end,
                    "local_routes": local_routes,
                    "remote_routes": remote_routes,
                    "owned_route_count": sum(owned_counts),
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
            plans.append(
                {
                    "rank": tp_rank * cp_degree + cp_rank,
                    "tp_rank": tp_rank,
                    **row_shape,
                    "local_q_heads": local_q_heads,
                    "local_kv_heads": local_kv_heads,
                    "local_intermediate_size": local_intermediate,
                    "local_hidden_size": local_hidden,
                    "local_head_packing": local_head,
                    "ffn_counts": local_ffn,
                    "attention_projection_counts": local_attn_projection,
                    "attention_core_counts": local_attn_core,
                    "router_counts": local_router_counts,
                    "semantic_scales": scales,
                    "segmented_norm_scale": segmented_norm_scale,
                    "q_tensor_scale": q_tensor_scale,
                    "physical_row_scale": physical_row_scale,
                    "ffn_tensor_scale": ffn_tensor_scale,
                    "expert": expert,
                }
            )
    return plans, {
        "context": context,
        "global_rows": global_rows,
        "global_head_packing": global_head,
        "global_ffn_counts": global_ffn,
        "global_attention_projection_counts": global_attn_projection,
        "global_attention_core_counts": global_attn_core,
        "global_router_counts": global_router_counts,
        "num_experts": num_experts,
        "top_k": top_k,
        "routing_mode": routing_mode,
        "global_route_counts": tuple(global_route_counts),
    }


def _rank_opcode_scale(
    plan: Mapping[str, Any],
    global_plan: Mapping[str, Any],
    *,
    stage: str,
    opcode: str,
    kernel: str,
    semantic: str,
) -> float:
    """Return the exact tile-count ratio for one rank/opcode class."""

    if stage == "layer/attention" and opcode in _SOFTMAX_ROW_GROUP_OPS:
        return float(plan["attention_core_counts"]["softmax_row_groups"]) / max(
            1,
            int(
                global_plan["global_attention_core_counts"][
                    "softmax_row_groups"
                ]
            ),
        )

    if stage == "layer/attention":
        if opcode in {"M_MM", "M_MV"}:
            return float(plan["attention_projection_counts"]["M_MM"]) / max(
                1, int(global_plan["global_attention_projection_counts"]["M_MM"])
            )
        if opcode in {"M_MM_WO", "M_MV_WO"}:
            return float(plan["attention_projection_counts"]["M_MM_WO"]) / max(
                1,
                int(global_plan["global_attention_projection_counts"]["M_MM_WO"]),
            )
        if opcode in {"M_BTMM", "M_BMM_WO", "M_BTMV", "M_BMV_WO"}:
            return float(plan["attention_core_counts"]["bmm_occurrences"]) / max(
                1, int(global_plan["global_attention_core_counts"]["bmm_occurrences"])
            )
    if stage == "layer/ffn":
        if opcode == "M_MM":
            return float(plan["ffn_counts"]["M_MM"]) / max(
                1, int(global_plan["global_ffn_counts"]["M_MM"])
            )
        if opcode == "M_MM_WO":
            return float(plan["ffn_counts"]["M_MM_WO"]) / max(
                1, int(global_plan["global_ffn_counts"]["M_MM_WO"])
            )
    if stage.startswith("layer/moe/router"):
        if opcode in {"M_MM", "M_MV"}:
            return float(plan["router_counts"]["M_MM"]) / max(
                1, int(global_plan["global_router_counts"]["M_MM"])
            )
        if opcode in {"M_MM_WO", "M_MV_WO"}:
            return float(plan["router_counts"]["M_MM_WO"]) / max(
                1, int(global_plan["global_router_counts"]["M_MM_WO"])
            )
    if stage.startswith("layer/moe/experts") and plan.get("expert"):
        expert = dict(plan["expert"])
        if opcode == "M_MM":
            return float(expert["expert_M_MM"]) / max(
                1, int(expert["global_expert_M_MM"])
            )
        if opcode == "M_MM_WO":
            return float(expert["expert_M_MM_WO"]) / max(
                1, int(expert["global_expert_M_MM_WO"])
            )
    if semantic == "token_tensor_sharded":
        if kernel in {
            "attention_q_segmented_norm",
            "attention_k_segmented_norm",
            "attention_segmented_norm",
        }:
            return float(plan["segmented_norm_scale"])
        if stage == "layer/attention":
            return float(plan["q_tensor_scale"])
        if stage == "layer/ffn":
            return float(plan["ffn_tensor_scale"])
    if (
        stage == "layer/ffn"
        and kernel == "dense_ffn_projection"
        and opcode in _FFN_ACTIVATION_OPS
    ):
        return float(plan["ffn_tensor_scale"])
    return float(plan["semantic_scales"][semantic])


def _tile_aware_stage_role_scales(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    global_plan: Mapping[str, Any],
    *,
    tp_degree: int,
    ep_degree: int,
    kv_cache_overlay: Mapping[str, Any] | None,
) -> dict[tuple[str, str], float]:
    """Build rank-local HBM traffic scales from physical local shapes."""

    source = dict(
        (report.get("hbm_traffic_breakdown") or {}).get("by_stage_role") or {}
    )
    result: dict[tuple[str, str], float] = {}
    token_scale = float(plan["active_rows"]) / max(
        1, int(global_plan["context"]["seq_len"])
    )
    # active_rows contains batch while context.seq_len does not.
    workload = dict((report.get("trace") or {}).get("workload") or {})
    batch_size = int(workload.get("batch_size", 1))
    token_scale /= max(1, batch_size)
    physical_row_scale = float(plan["physical_row_scale"])

    global_kv_heads = int(
        workload.get("num_key_value_heads")
        or workload.get("num_kv_heads")
    )
    hardware = dict((report.get("trace") or {})["hardware"])
    max_k_tiles = max(1, int(hardware.get("mram_tile_capacity", 1)))
    local_attention_weight = _attention_projection_counts(
        physical_rows=int(hardware["blen"]),
        hidden_size=int(workload["hidden_size"]),
        q_storage_width=int(plan["local_head_packing"]["total_q_dim"]),
        kv_heads=int(plan["local_kv_heads"]),
        mlen=int(hardware["mlen"]),
        blen=int(hardware["blen"]),
        max_k_tiles=max_k_tiles,
    )
    global_attention_weight = _attention_projection_counts(
        physical_rows=int(hardware["blen"]),
        hidden_size=int(workload["hidden_size"]),
        q_storage_width=int(global_plan["global_head_packing"]["total_q_dim"]),
        kv_heads=global_kv_heads,
        mlen=int(hardware["mlen"]),
        blen=int(hardware["blen"]),
        max_k_tiles=max_k_tiles,
    )
    attention_weight_scale = float(local_attention_weight["M_MM"]) / max(
        1, int(global_attention_weight["M_MM"])
    )
    ffn_weight_scale = float(plan["ffn_counts"]["physical_intermediate"]) / max(
        1, int(global_plan["global_ffn_counts"]["physical_intermediate"])
    )

    for key in source:
        stage, role = str(key).split("::", 1)
        if role == "weight":
            if stage == "layer/attention":
                scale = attention_weight_scale
            elif stage == "layer/ffn":
                scale = ffn_weight_scale
            elif stage.startswith("layer/moe/experts"):
                scale = 1.0 / (tp_degree * ep_degree)
            elif stage.startswith("layer/moe/router"):
                scale = 1.0 / tp_degree
            else:
                scale = 1.0
        elif role in KV_TRAFFIC_ROLES:
            scale = token_scale * (
                float(plan["local_kv_heads"])
                / max(1, global_kv_heads)
            )
            if kv_cache_overlay and stage == "layer/attention":
                global_loads = float(kv_cache_overlay.get("global_tile_loads", 0.0))
                local_loads = float(kv_cache_overlay.get("local_tile_loads", 0.0))
                if global_loads > 0:
                    scale = (
                        local_loads
                        / global_loads
                        * float(plan["local_kv_heads"])
                        / max(1, global_kv_heads)
                    )
        elif role == "activation":
            if stage.startswith("layer/moe/experts") and plan.get("expert"):
                scale = (
                    float(plan["expert"]["owned_padded_bucket_rows"])
                    / max(
                        1,
                        sum(
                            math.ceil(count / int((report.get("trace") or {})["hardware"]["blen"]))
                            * int((report.get("trace") or {})["hardware"]["blen"])
                            for count in global_plan.get("global_route_counts", ())
                            if count
                        ),
                    )
                )
            else:
                scale = physical_row_scale
        elif role == "integer":
            scale = token_scale
        else:
            raise ValueError(
                f"tile-aware HBM census has no rule for precision role {role!r}"
            )
        result[(stage, role)] = max(0.0, scale)
    return result


def _tile_aware_ep_communication(
    plans: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    ep_degree: int,
    fp_width_bits: int,
    one_way_link_bandwidth_gbps: float,
    startup_latency_ns: float,
) -> dict[str, Any]:
    layers = int(model["num_hidden_layers"])
    if ep_degree == 1:
        return {
            "ep_dispatch_bytes": 0.0,
            "ep_return_bytes": 0.0,
            "ep_dispatch_latency_ns": 0.0,
            "ep_return_latency_ns": 0.0,
            "aggregate_ep_interconnect_bytes": 0.0,
        }
    hidden = int(model["hidden_size"])
    max_remote_routes = max(
        int((plan.get("expert") or {}).get("remote_routes", 0))
        for plan in plans
    )
    metadata_bytes = 8
    dispatch_per_layer = max_remote_routes * (
        hidden * fp_width_bits / 8.0 + metadata_bytes
    )
    return_per_layer = max_remote_routes * hidden * fp_width_bits / 8.0
    dispatch_latency = (
        (ep_degree - 1) * startup_latency_ns
        + dispatch_per_layer / one_way_link_bandwidth_gbps
    ) * layers
    return_latency = (
        (ep_degree - 1) * startup_latency_ns
        + return_per_layer / one_way_link_bandwidth_gbps
    ) * layers
    aggregate_remote_routes = sum(
        int((plan.get("expert") or {}).get("remote_routes", 0))
        for plan in plans
    )
    aggregate_dispatch = aggregate_remote_routes * (
        hidden * fp_width_bits / 8.0 + metadata_bytes
    ) * layers
    aggregate_return = (
        aggregate_remote_routes * hidden * fp_width_bits / 8.0 * layers
    )
    return {
        "ep_dispatch_bytes": dispatch_per_layer * layers,
        "ep_return_bytes": return_per_layer * layers,
        "ep_dispatch_latency_ns": dispatch_latency,
        "ep_return_latency_ns": return_latency,
        "aggregate_ep_interconnect_bytes": aggregate_dispatch + aggregate_return,
    }


def _retarget_moe_tp_communication(
    communication: dict[str, Any],
    *,
    report: Mapping[str, Any],
    chip_count: int,
    tp_degree: int,
    router_bytes: float,
    router_latency_ns: float,
    expert_bytes: float,
    expert_latency_ns: float,
) -> None:
    """Replace the dense FFN collective with MoE-specific TP collectives.

    The shared communication helper starts from a dense decoder's two
    collectives per layer.  A pure MoE trace has no dense ``layer/ffn`` stage:
    its dependency-bound reductions occur after the row-parallel router and
    expert down projection instead.  Keep a dense FFN collective only when the
    emitted trace actually contains that stage, which also covers hybrid
    traces without guessing their layer mix.
    """

    bytes_by_stage = dict(communication["tp_collective_bytes_by_stage"])
    latency_by_stage = dict(communication["tp_collective_latency_ns_by_stage"])
    emitted_stages = set(report.get("stage_compute_latency_ns") or {})
    if "layer/ffn" not in emitted_stages:
        bytes_by_stage.pop("layer/ffn", None)
        latency_by_stage.pop("layer/ffn", None)

    if tp_degree > 1:
        bytes_by_stage["layer/moe/router"] = router_bytes
        bytes_by_stage["layer/moe/experts"] = expert_bytes
        latency_by_stage["layer/moe/router"] = router_latency_ns
        latency_by_stage["layer/moe/experts"] = expert_latency_ns

    per_rank_bytes = math.fsum(bytes_by_stage.values())
    per_rank_latency = math.fsum(latency_by_stage.values())
    aggregate_tp_bytes = per_rank_bytes * chip_count
    aggregate_cp_bytes = float(communication["aggregate_cp_kv_ring_bytes"])
    communication.update(
        {
            "tp_collective_bytes_by_stage": bytes_by_stage,
            "tp_collective_latency_ns_by_stage": latency_by_stage,
            "tp_collective_bytes": per_rank_bytes,
            "tp_collective_latency_ns": per_rank_latency,
            "aggregate_tp_collective_bytes": aggregate_tp_bytes,
            "aggregate_interconnect_bytes": (
                aggregate_tp_bytes + aggregate_cp_bytes
            ),
        }
    )


def _estimate_tile_aware_multi_chip_latency(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    chip_count: int,
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
    nvlink_port_bidirectional_gbps: float,
    interconnect_startup_ns: float,
    kv_cache_overlay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace = dict(report.get("trace") or {})
    if report.get("compute_timing_mode") != "ideal-ii1":
        raise ValueError(
            "tile-aware-tp-cp-ep-v3 reconstructs additive ideal-II1 work; "
            "hazard-aware or legacy timing must use a dedicated distributed "
            "scheduler"
        )
    if int(trace.get("schema_version", 0)) != 7:
        raise ValueError(
            "tile-aware-tp-cp-ep-v3 requires CostTrace schema 7 with final "
            "schedule lineage"
        )
    compiler_metadata = dict(trace.get("compiler_metadata") or {})
    routing_mode = compiler_metadata.get("moe_routing_mode")
    cp_degree = validate_tp_cp_ep(
        model,
        chip_count=chip_count,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        routing_mode=routing_mode,
    )
    if nvlink_port_count not in {1, 2, 4}:
        raise ValueError("nvlink_port_count must be one of 1, 2, or 4")
    census_entries, census_totals = _parallel_kernel_entries(report)
    plans, global_plan = _tile_aware_rank_plans(
        report,
        model,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        ep_degree=ep_degree,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    # The context helper is batch-agnostic; retain the batch count explicitly
    # for traffic scaling and downstream diagnostics.
    global_plan["context"]["seq_len"] = seq_len
    global_plan["global_route_counts"] = (
        [
            (seq_len * batch_size * int(global_plan["top_k"]))
            // int(global_plan["num_experts"])
            + int(
                expert
                < (seq_len * batch_size * int(global_plan["top_k"]))
                % int(global_plan["num_experts"])
            )
            for expert in range(int(global_plan["num_experts"]))
        ]
        if int(global_plan["num_experts"]) > 1
        else []
    )
    if chip_count == 1:
        baseline = _estimate_factorized_multi_chip_latency(
            report,
            model,
            chip_count=1,
            tp_degree=1,
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
        plan = plans[0]
        baseline.update(
            {
                "parallel_model": "tp-cp-ep",
                "multi_chip_model": TILE_AWARE_MULTI_CHIP_MODEL,
                "ep_degree": 1,
                "tp_cp_ep_legality": "exact_single_chip_identity",
                "parallel_kernel_census_coverage": float(
                    trace["parallel_kernel_census_coverage"]
                ),
                "local_tile_counts_by_rank": [
                    {
                        "rank": 0,
                        "tp_rank": 0,
                        "cp_rank": 0,
                        "active_rows": int(plan["active_rows"]),
                        "physical_rows": int(plan["physical_rows"]),
                        "local_q_heads": int(plan["local_q_heads"]),
                        "local_kv_heads": int(plan["local_kv_heads"]),
                        "local_intermediate_size": int(
                            plan["local_intermediate_size"]
                        ),
                        "q_storage_blocks": int(
                            plan["local_head_packing"]["storage_blocks"]
                        ),
                        "attention_projection": dict(
                            plan["attention_projection_counts"]
                        ),
                        "attention_core": dict(
                            plan["attention_core_counts"]
                        ),
                        "ffn": dict(plan["ffn_counts"]),
                        "expert": dict(plan["expert"] or {}),
                    }
                ],
                "slowest_rank": 0,
                "slowest_rank_by_stage": {
                    stage: {"compute": 0, "memory": 0}
                    for stage in baseline["per_chip_stage_compute_latency_ns"]
                },
                "matrix_utilization_by_stage": {
                    "attention": 1.0,
                    "ffn": 1.0,
                },
                "vector_utilization_by_stage": {
                    "attention": 1.0,
                    "ffn": 1.0,
                },
                "padding_cycles": 0.0,
                "replicated_compute_cycles": 0.0,
                "tp_rounding_overhead": 0.0,
                "cp_tail_overhead": float(
                    plan["attention_core_counts"]["tail_bmm_occurrences"]
                ),
                "tail_isa_limitation": "active_row_bmm_unavailable",
                "rank_stage_compute_latency_ns": [
                    dict(baseline["per_chip_stage_compute_latency_ns"])
                ],
                "rank_stage_memory_latency_ns": [
                    dict(baseline["per_chip_stage_memory_latency_ns"])
                ],
                "rank_hbm_traffic_breakdown": [
                    dict(baseline["per_chip_hbm_traffic_breakdown"])
                ],
                "average_per_chip_hbm_traffic_breakdown": dict(
                    baseline["per_chip_hbm_traffic_breakdown"]
                ),
                "average_per_chip_stage_memory_latency_ns": dict(
                    baseline["per_chip_stage_memory_latency_ns"]
                ),
                "expert_weight_replication": 1.0,
                "experts_per_rank": int(global_plan["num_experts"]),
                "expert_bucket_utilization": (
                    float(plan["expert"]["expert_bucket_utilization"])
                    if plan.get("expert")
                    else 1.0
                ),
                "parallel_action_scales_by_stage_opcode": {
                    f"{stage}::{opcode}": 1.0
                    for stage, opcodes in (
                        report.get("stage_compute_opcode_work_cycles") or {}
                    ).items()
                    for opcode in opcodes
                },
                "parallel_action_scales_by_kernel_opcode": {
                    (
                        f"{entry['stage']}::"
                        f"{parallel_kernel_lineage_id(entry)}::"
                        f"{entry['opcode']}"
                    ): 1.0
                    for entry in census_entries
                },
                "parallel_action_scales_by_kernel": {
                    (
                        f"{entry['stage']}::"
                        f"{parallel_kernel_lineage_id(entry)}"
                    ): 1.0
                    for entry in census_entries
                },
                "fractional_v2_latency": float(baseline["latency_ns"]),
                "tile_aware_v3_latency": float(baseline["latency_ns"]),
                "multi_chip_fidelity": {
                    "compute": "exact_single_chip_costemitter_identity",
                    "memory": "exact_single_chip_v4_identity",
                    "communication": "none",
                    "compiler_isa": "native_single_chip",
                },
            }
        )
        return baseline

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
            "tile-aware model requires stage_compute_opcode_work_cycles"
        )
    clock_period_ps = float((report.get("compatibility") or {}).get(
        "clock_period_ps", 1_000.0
    ))
    cycle_to_ns = clock_period_ps / 1_000.0

    rank_stage_cycles: list[dict[str, float]] = []
    rank_opcode_scales: list[dict[str, float]] = []
    rank_kernel_opcode_scales: list[dict[str, float]] = []
    rank_kernel_scales: list[dict[str, float]] = []
    for plan in plans:
        stage_cycles: dict[str, float] = defaultdict(float)
        local_opcode_cycles: dict[str, float] = defaultdict(float)
        local_kernel_counts: dict[str, float] = defaultdict(float)
        baseline_kernel_counts: dict[str, float] = defaultdict(float)
        kernel_opcode_scales: dict[str, float] = {}
        for entry in census_entries:
            stage = str(entry["stage"])
            opcode = str(entry["opcode"])
            lineage = parallel_kernel_lineage_id(entry)
            try:
                cycles = stage_opcode_cycles[stage][opcode]
            except KeyError as exc:
                raise ValueError(
                    "parallel kernel census references missing timing work "
                    f"{stage}/{opcode}"
                ) from exc
            base_count = census_totals[(stage, opcode)]
            base_cycles = cycles * int(entry["count"]) / base_count
            scale = _rank_opcode_scale(
                plan,
                global_plan,
                stage=stage,
                opcode=opcode,
                kernel=str(entry["kernel"]),
                semantic=str(entry["tp_semantics"]),
            )
            local_cycles = base_cycles * scale
            stage_cycles[stage] += local_cycles
            local_opcode_cycles[f"{stage}::{opcode}"] += local_cycles
            lineage_key = f"{stage}::{lineage}"
            kernel_opcode_key = f"{lineage_key}::{opcode}"
            previous = kernel_opcode_scales.setdefault(
                kernel_opcode_key, scale
            )
            if not math.isclose(previous, scale, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    "one kernel-lineage/opcode received inconsistent rank "
                    f"scales: {kernel_opcode_key} -> {previous} vs {scale}"
                )
            local_kernel_counts[lineage_key] += int(entry["count"]) * scale
            baseline_kernel_counts[lineage_key] += int(entry["count"])
        opcode_scales = {
            key: value
            / max(
                1e-30,
                stage_opcode_cycles[key.split("::", 1)[0]][
                    key.split("::", 1)[1]
                ],
            )
            for key, value in local_opcode_cycles.items()
        }
        rank_stage_cycles.append(dict(stage_cycles))
        rank_opcode_scales.append(opcode_scales)
        rank_kernel_opcode_scales.append(kernel_opcode_scales)
        rank_kernel_scales.append(
            {
                key: local_kernel_counts[key] / baseline_count
                for key, baseline_count in baseline_kernel_counts.items()
            }
        )

    stages = sorted(
        set(stage_opcode_cycles)
        | set(report.get("hbm_stage_latency_ns") or {})
    )
    stage_compute: dict[str, float] = {}
    stage_compute_slowest_rank: dict[str, int] = {}
    for stage in stages:
        values = [
            cycles.get(stage, 0.0) * cycle_to_ns
            for cycles in rank_stage_cycles
        ]
        slowest = max(range(len(values)), key=values.__getitem__)
        stage_compute[stage] = values[slowest]
        stage_compute_slowest_rank[stage] = slowest

    original_traffic = dict(report.get("hbm_traffic_breakdown") or {})
    rank_traffic: list[dict[str, Any]] = []
    rank_stage_memory: list[dict[str, float]] = []
    rank_stage_floor: list[dict[str, float]] = []
    per_chip_bandwidth = aggregate_hbm_bandwidth_gbps / chip_count
    baseline_bandwidth = aggregate_hbm_bandwidth_gbps / reference_a100_count
    for plan in plans:
        scales = _tile_aware_stage_role_scales(
            report,
            plan,
            global_plan,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            kv_cache_overlay=kv_cache_overlay,
        )
        traffic = _scale_traffic_breakdown(
            original_traffic,
            stage_role_scales=scales,
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

    stage_memory: dict[str, float] = {}
    stage_floor: dict[str, float] = {}
    stage_memory_slowest_rank: dict[str, int] = {}
    for stage in stages:
        values = [value.get(stage, 0.0) for value in rank_stage_memory]
        slowest = max(range(len(values)), key=values.__getitem__)
        stage_memory[stage] = values[slowest]
        stage_floor[stage] = rank_stage_floor[slowest].get(stage, 0.0)
        stage_memory_slowest_rank[stage] = slowest

    one_way_bandwidth = (
        nvlink_port_count * nvlink_port_bidirectional_gbps / 2.0
    )
    context = global_plan["context"]
    communication = _factorized_communication(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        local_tokens=int(context["max_local_tokens"]),
        fp_width_bits=fp_width_bits,
        kv_width_bits=kv_width_bits,
        one_way_link_bandwidth_gbps=one_way_bandwidth,
        startup_latency_ns=interconnect_startup_ns,
    )
    ep_communication = _tile_aware_ep_communication(
        plans,
        model,
        ep_degree=ep_degree,
        fp_width_bits=fp_width_bits,
        one_way_link_bandwidth_gbps=one_way_bandwidth,
        startup_latency_ns=interconnect_startup_ns,
    )
    communication.update(ep_communication)

    if int(global_plan["num_experts"]) > 1:
        layers = int(model["num_hidden_layers"])
        max_local_tokens = int(context["max_local_tokens"]) * batch_size
        router_bytes_per_layer = (
            2.0
            * (tp_degree - 1)
            / tp_degree
            * max_local_tokens
            * int(global_plan["num_experts"])
            * fp_width_bits
            / 8.0
        )
        expert_bytes_per_layer = (
            2.0
            * (tp_degree - 1)
            / tp_degree
            * max(
                int((plan.get("expert") or {}).get("owned_route_count", 0))
                for plan in plans
            )
            * int(model["hidden_size"])
            * fp_width_bits
            / 8.0
        )
        router_bytes = router_bytes_per_layer * layers
        expert_bytes = expert_bytes_per_layer * layers
        router_latency = (
            (
                2.0 * (tp_degree - 1) * interconnect_startup_ns
                + router_bytes_per_layer / one_way_bandwidth
            )
            * layers
            if tp_degree > 1
            else 0.0
        )
        expert_latency = (
            (
                2.0 * (tp_degree - 1) * interconnect_startup_ns
                + expert_bytes_per_layer / one_way_bandwidth
            )
            * layers
            if tp_degree > 1
            else 0.0
        )
        _retarget_moe_tp_communication(
            communication,
            report=report,
            chip_count=chip_count,
            tp_degree=tp_degree,
            router_bytes=router_bytes,
            router_latency_ns=router_latency,
            expert_bytes=expert_bytes,
            expert_latency_ns=expert_latency,
        )

    tp_latency = dict(communication["tp_collective_latency_ns_by_stage"])
    cp_latency = dict(communication["cp_kv_ring_latency_ns_by_stage"])

    all_stages = set(stages) | set(tp_latency) | set(cp_latency)
    nominal: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    bounds: dict[str, str] = {}
    for stage in sorted(all_stages):
        compute = stage_compute.get(stage, 0.0)
        memory = stage_memory.get(stage, 0.0)
        tp_comm = tp_latency.get(stage, 0.0)
        cp_comm = cp_latency.get(stage, 0.0)
        serial_extra = 0.0
        if stage == "layer/moe/dispatch":
            serial_extra = float(ep_communication["ep_dispatch_latency_ns"])
        elif stage == "layer/moe/combine":
            serial_extra = float(ep_communication["ep_return_latency_ns"])
        nominal[stage] = max(compute, memory, cp_comm) + tp_comm + serial_extra
        lower[stage] = max(compute, memory, cp_comm, tp_comm, serial_extra)
        upper[stage] = compute + memory + cp_comm + tp_comm + serial_extra
        bounds[stage] = max(
            {
                "compute": compute,
                "memory": memory,
                "cp_communication": cp_comm,
            },
            key={
                "compute": compute,
                "memory": memory,
                "cp_communication": cp_comm,
            }.get,
        )

    aggregate_traffic = _sum_traffic_breakdowns(rank_traffic)
    average_traffic = _average_traffic_breakdown(
        aggregate_traffic, chip_count
    )
    average_stage_memory = {
        stage: math.fsum(
            rank.get(stage, 0.0) for rank in rank_stage_memory
        )
        / chip_count
        for stage in stages
    }
    slowest_rank = max(
        range(len(plans)),
        key=lambda rank: sum(
            max(
                rank_stage_cycles[rank].get(stage, 0.0) * cycle_to_ns,
                rank_stage_memory[rank].get(stage, 0.0),
            )
            for stage in stages
        ),
    )
    representative_traffic = rank_traffic[slowest_rank]
    per_chip_bytes = sum(
        _traffic_total(bucket)
        for bucket in representative_traffic.get("by_stage", {}).values()
    )
    aggregate_bytes = sum(
        _traffic_total(bucket)
        for bucket in aggregate_traffic.get("by_stage", {}).values()
    )
    total_latency = math.fsum(nominal.values())
    aggregate_opcode_scale: dict[str, float] = {}
    for key in rank_opcode_scales[0]:
        aggregate_opcode_scale[key] = math.fsum(
            values[key] for values in rank_opcode_scales
        ) / chip_count
    aggregate_kernel_opcode_scale = {
        key: math.fsum(
            values[key] for values in rank_kernel_opcode_scales
        )
        / chip_count
        for key in rank_kernel_opcode_scales[0]
    }
    aggregate_kernel_scale = {
        key: math.fsum(values[key] for values in rank_kernel_scales)
        / chip_count
        for key in rank_kernel_scales[0]
    }
    aggregate_stage_scale = {
        stage: (
            math.fsum(
                rank.get(stage, 0.0) for rank in rank_stage_cycles
            )
            / chip_count
            / max(1e-30, math.fsum(stage_opcode_cycles[stage].values()))
        )
        for stage in stage_opcode_cycles
    }

    fractional = _estimate_factorized_multi_chip_latency(
        report,
        model,
        chip_count=chip_count,
        tp_degree=tp_degree,
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
    local_tiles = [
        {
            "rank": int(plan["rank"]),
            "tp_rank": int(plan["tp_rank"]),
            "cp_rank": int(plan["cp_rank"]),
            "active_rows": int(plan["active_rows"]),
            "physical_rows": int(plan["physical_rows"]),
            "local_q_heads": int(plan["local_q_heads"]),
            "local_kv_heads": int(plan["local_kv_heads"]),
            "local_intermediate_size": int(plan["local_intermediate_size"]),
            "q_storage_blocks": int(
                plan["local_head_packing"]["storage_blocks"]
            ),
            "attention_projection": dict(plan["attention_projection_counts"]),
            "attention_core": dict(plan["attention_core_counts"]),
            "ffn": dict(plan["ffn_counts"]),
            "expert": dict(plan["expert"] or {}),
        }
        for plan in plans
    ]
    matrix_utilization = {
        "attention": min(
            (
                float(plan["active_rows"])
                / max(1, int(plan["physical_rows"]))
                * float(plan["local_head_packing"]["head_lane_utilization"])
                for plan in plans
            ),
            default=1.0,
        ),
        "ffn": min(
            (
                float(plan["active_rows"])
                / max(1, int(plan["physical_rows"]))
                * float(plan["local_intermediate_size"])
                / max(1, int(plan["ffn_counts"]["physical_intermediate"]))
                for plan in plans
            ),
            default=1.0,
        ),
    }
    padding_cycles = max(
        0.0,
        math.fsum(stage_compute.values())
        - math.fsum(
            float(value)
            for value in fractional["per_chip_stage_compute_latency_ns"].values()
        ),
    )
    equivalent_channels = 128.0 * reference_a100_count / chip_count
    interconnect_bytes = (
        float(communication["tp_collective_bytes"])
        + float(communication["cp_kv_ring_bytes"])
        + float(ep_communication["ep_dispatch_bytes"])
        + float(ep_communication["ep_return_bytes"])
    )
    interconnect_latency = (
        float(communication["tp_collective_latency_ns"])
        + float(communication["cp_kv_ring_latency_ns"])
        + float(ep_communication["ep_dispatch_latency_ns"])
        + float(ep_communication["ep_return_latency_ns"])
    )
    aggregate_interconnect_bytes = (
        float(communication["aggregate_interconnect_bytes"])
        + float(ep_communication["aggregate_ep_interconnect_bytes"])
    )
    return {
        "chip_count": chip_count,
        "reference_a100_count": reference_a100_count,
        "parallel_model": "tp-cp-ep",
        "multi_chip_model": TILE_AWARE_MULTI_CHIP_MODEL,
        "tp_degree": tp_degree,
        "cp_degree": cp_degree,
        "ep_degree": ep_degree,
        "tp_cp_ep_legality": "valid_balanced_tile_partition",
        "context_partition": context,
        "max_token_fraction": float(context["max_token_fraction"]),
        "max_causal_pair_fraction": float(context["max_causal_pair_fraction"]),
        "parallel_kernel_census_coverage": float(
            trace["parallel_kernel_census_coverage"]
        ),
        "parallel_work_census_coverage": float(
            trace["parallel_kernel_census_coverage"]
        ),
        "local_tile_counts_by_rank": local_tiles,
        "slowest_rank": slowest_rank,
        "slowest_rank_by_stage": {
            stage: {
                "compute": stage_compute_slowest_rank.get(stage),
                "memory": stage_memory_slowest_rank.get(stage),
            }
            for stage in stages
        },
        "matrix_utilization_by_stage": matrix_utilization,
        "vector_utilization_by_stage": {
            "attention": min(
                float(plan["active_rows"]) / max(1, int(plan["physical_rows"]))
                for plan in plans
            ),
            "ffn": min(
                float(plan["active_rows"]) / max(1, int(plan["physical_rows"]))
                for plan in plans
            ),
        },
        "padding_cycles": padding_cycles / cycle_to_ns,
        "replicated_compute_cycles": math.fsum(
            stage_opcode_cycles[str(entry["stage"])][str(entry["opcode"])]
            * int(entry["count"])
            / census_totals[(str(entry["stage"]), str(entry["opcode"]))]
            for entry in census_entries
            if str(entry["tp_semantics"]) == "replicated_setup"
        ),
        "tp_rounding_overhead": (
            total_latency / max(float(fractional["latency_ns"]), 1e-30) - 1.0
        ),
        "cp_tail_overhead": math.fsum(
            float(plan["attention_core_counts"]["tail_bmm_occurrences"])
            for plan in plans
        ),
        "tail_isa_limitation": "active_row_bmm_unavailable",
        "per_chip_stage_compute_latency_ns": stage_compute,
        "rank_stage_compute_latency_ns": [
            {
                stage: value * cycle_to_ns
                for stage, value in rank.items()
            }
            for rank in rank_stage_cycles
        ],
        "per_chip_stage_memory_latency_ns": stage_memory,
        "rank_stage_memory_latency_ns": rank_stage_memory,
        "per_chip_stage_v4_floor_ns": stage_floor,
        "per_chip_stage_roofline_latency_ns": nominal,
        "per_chip_stage_full_overlap_lower_bound_ns": lower,
        "per_chip_stage_no_overlap_upper_bound_ns": upper,
        "full_overlap_lower_bound_ns": math.fsum(lower.values()),
        "nominal_stage_model_ns": total_latency,
        "no_overlap_upper_bound_ns": math.fsum(upper.values()),
        "communication_overlap_bound": (
            "cp_ring_overlaps_compute_memory;tp_and_ep_dependency_bound"
        ),
        "per_chip_stage_bound": bounds,
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
        "per_chip_hbm_traffic": dict(
            representative_traffic.get("by_stage") or {}
        ),
        "per_chip_hbm_traffic_breakdown": representative_traffic,
        "rank_hbm_traffic_breakdown": rank_traffic,
        "aggregate_hbm_traffic_breakdown": aggregate_traffic,
        "average_per_chip_hbm_traffic_breakdown": average_traffic,
        "average_per_chip_stage_memory_latency_ns": average_stage_memory,
        "hbm_traffic_partition_fidelity": (
            "rank_local_role_scaled_physical_traffic_v3"
        ),
        "v4_rank_latency_fidelity": (
            "bandwidth_floor_plus_opcode_request_residual_rescaling_v3"
        ),
        "v4_local_geometry_reconstruction": False,
        "v4_rank_latency_exact": False,
        "per_chip_hbm_physical_bytes": per_chip_bytes,
        "aggregate_hbm_physical_bytes": aggregate_bytes,
        "per_chip_achieved_bandwidth_gbps": (
            per_chip_bytes / total_latency if total_latency > 0 else 0.0
        ),
        "per_chip_bandwidth_utilization": (
            per_chip_bytes
            / total_latency
            / per_chip_bandwidth
            if total_latency > 0 and per_chip_bandwidth > 0
            else 0.0
        ),
        "weight_replication_factor": cp_degree,
        "expert_weight_replication": (
            cp_degree / ep_degree
            if int(global_plan["num_experts"]) > 1
            else 1.0
        ),
        "experts_per_rank": (
            int(global_plan["num_experts"]) // ep_degree
            if int(global_plan["num_experts"]) > 1
            else 0
        ),
        "expert_bucket_utilization": min(
            (
                float(plan["expert"]["expert_bucket_utilization"])
                for plan in plans
                if plan.get("expert")
            ),
            default=1.0,
        ),
        "parallel_action_scales_by_stage_opcode": aggregate_opcode_scale,
        "parallel_action_scales_by_kernel_opcode": (
            aggregate_kernel_opcode_scale
        ),
        "parallel_action_scales_by_kernel": aggregate_kernel_scale,
        "rank_parallel_action_scales_by_kernel_opcode": (
            rank_kernel_opcode_scales
        ),
        "rank_parallel_action_scales_by_kernel": rank_kernel_scales,
        "parallel_action_scales_by_stage": aggregate_stage_scale,
        "per_chip_compute_scale": math.fsum(stage_compute.values()) / max(
            math.fsum(
                float(value)
                for value in report["stage_compute_latency_ns"].values()
            ),
            1e-30,
        ),
        "nvlink_port_count": nvlink_port_count,
        "nvlink_peak_bidirectional_bandwidth_gbps": (
            nvlink_port_count * nvlink_port_bidirectional_gbps
        ),
        "nvlink_peak_oneway_bandwidth_gbps": one_way_bandwidth,
        "interconnect_bandwidth_semantics": "architectural_peak_assumption",
        "bandwidth_efficiency": 1.0,
        "interconnect_startup_ns": interconnect_startup_ns,
        **communication,
        "aggregate_interconnect_bytes": aggregate_interconnect_bytes,
        "interconnect_bytes": interconnect_bytes,
        "interconnect_latency_ns": interconnect_latency,
        "interconnect_bytes_by_stage": {
            stage: float(
                communication["tp_collective_bytes_by_stage"].get(stage, 0.0)
            )
            + float(
                communication["cp_kv_ring_bytes_by_stage"].get(stage, 0.0)
            )
            + (
                float(ep_communication["ep_dispatch_bytes"])
                if stage == "layer/moe/dispatch"
                else 0.0
            )
            + (
                float(ep_communication["ep_return_bytes"])
                if stage == "layer/moe/combine"
                else 0.0
            )
            for stage in all_stages
        },
        "interconnect_latency_ns_by_stage": {
            stage: (
                tp_latency.get(stage, 0.0)
                + cp_latency.get(stage, 0.0)
                + (
                    float(ep_communication["ep_dispatch_latency_ns"])
                    if stage == "layer/moe/dispatch"
                    else 0.0
                )
                + (
                    float(ep_communication["ep_return_latency_ns"])
                    if stage == "layer/moe/combine"
                    else 0.0
                )
            )
            for stage in all_stages
        },
        "latency_ns": total_latency,
        "latency_ms": total_latency / 1e6,
        "fractional_v2_latency": float(fractional["latency_ns"]),
        "tile_aware_v3_latency": total_latency,
        "multi_chip_fidelity": {
            "compute": "tile_reconstructed_from_compiler_parallel_kernel_census",
            "memory": (
                "rank_local_role_scaled_v4_floor_and_residual;"
                "not_distributed_manifest_replay"
            ),
            "communication": "analytical_tp_cp_ep_peak_link",
            "compiler_isa": "single_chip_planners_reused_without_distributed_isa",
        },
        "kv_cache_fidelity": (
            "exact_local_cache_occurrences_under_tile_aware_cp"
            if kv_cache_overlay
            else "rank_local_owned_kv_only"
        ),
        "kv_cache_overlay": dict(kv_cache_overlay or {}),
    }


def _estimate_factorized_multi_chip_latency(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    chip_count: int,
    tp_degree: int,
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
    cp_degree = validate_tp_cp(
        model,
        chip_count=chip_count,
        tp_degree=tp_degree,
    )
    if nvlink_port_count not in {1, 2, 4}:
        raise ValueError("nvlink_port_count must be one of 1, 2, or 4")
    context = zigzag_context_partition(seq_len, cp_degree)
    max_token_fraction = float(context["max_token_fraction"])
    max_pair_fraction = float(context["max_causal_pair_fraction"])
    census = build_parallel_work_census(report)
    stage_compute, compute_scales = _scale_factorized_compute(
        census,
        tp_degree=tp_degree,
        max_token_fraction=max_token_fraction,
        max_causal_pair_fraction=max_pair_fraction,
    )

    traffic, traffic_fidelity = _factorized_traffic_breakdown(
        report,
        tp_degree=tp_degree,
        max_token_fraction=max_token_fraction,
    )
    if kv_cache_overlay:
        global_loads = float(kv_cache_overlay.get("global_tile_loads", 0.0))
        local_loads = float(kv_cache_overlay.get("local_tile_loads", 0.0))
        if global_loads <= 0:
            raise ValueError("global K/V tile loads must be positive")
        relative_scale = (
            local_loads / global_loads / max_token_fraction
            if max_token_fraction > 0
            else 0.0
        )
        traffic = _apply_attention_kv_overlay(
            traffic,
            relative_scale=relative_scale,
        )
        traffic_fidelity += ":exact_local_cache_occurrence_overlay"

    per_chip_by_stage = dict(traffic.get("by_stage") or {})
    original_breakdown = dict(report.get("hbm_traffic_breakdown") or {})
    original_by_stage = dict(original_breakdown.get("by_stage") or {})
    baseline_memory = {
        str(stage): float(value)
        for stage, value in (report.get("hbm_stage_latency_ns") or {}).items()
    }
    exact_floor = {
        str(stage): float(value)
        for stage, value in (
            (report.get("compatibility") or {}).get(
                "stage_theoretical_floor_ns", {}
            )
            or {}
        ).items()
    }
    baseline_total_floor = float(
        (report.get("compatibility") or {}).get("theoretical_floor_ns", 0.0)
    )
    original_total_bytes = sum(
        _traffic_total(bucket) for bucket in original_by_stage.values()
    )
    per_chip_bandwidth = aggregate_hbm_bandwidth_gbps / chip_count
    baseline_bandwidth = aggregate_hbm_bandwidth_gbps / reference_a100_count
    stage_memory: dict[str, float] = {}
    stage_floor: dict[str, float] = {}
    for stage in sorted(set(original_by_stage) | set(per_chip_by_stage)):
        old_bucket = dict(original_by_stage.get(stage) or {})
        new_bucket = dict(per_chip_by_stage.get(stage) or {})
        old_bytes = _traffic_total(old_bucket)
        new_bytes = _traffic_total(new_bucket)
        old_requests = float(old_bucket.get("read_requests", 0.0)) + float(
            old_bucket.get("write_requests", 0.0)
        )
        new_requests = float(new_bucket.get("read_requests", 0.0)) + float(
            new_bucket.get("write_requests", 0.0)
        )
        old_floor = exact_floor.get(
            stage,
            (
                baseline_total_floor * old_bytes / original_total_bytes
                if original_total_bytes > 0
                else 0.0
            ),
        )
        new_floor = (
            old_floor
            * new_bytes
            / old_bytes
            * baseline_bandwidth
            / per_chip_bandwidth
            if old_bytes > 0
            else 0.0
        )
        # V4 residual represents startup/channel/row/drain service. Preserve
        # the calibrated per-request service rather than scaling it by bytes.
        residual = max(baseline_memory.get(stage, 0.0) - old_floor, 0.0)
        residual_scale = new_requests / old_requests if old_requests > 0 else 0.0
        stage_floor[stage] = new_floor
        stage_memory[stage] = new_floor + residual * residual_scale

    one_way_bandwidth = (
        nvlink_port_count * nvlink_port_bidirectional_gbps / 2.0
    )
    communication = _factorized_communication(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        local_tokens=int(context["max_local_tokens"]),
        fp_width_bits=fp_width_bits,
        kv_width_bits=kv_width_bits,
        one_way_link_bandwidth_gbps=one_way_bandwidth,
        startup_latency_ns=interconnect_startup_ns,
    )

    def total_for_communication(
        candidate: Mapping[str, Any],
    ) -> float:
        candidate_tp = candidate["tp_collective_latency_ns_by_stage"]
        candidate_cp = candidate["cp_kv_ring_latency_ns_by_stage"]
        candidate_stages = (
            set(stage_compute)
            | set(stage_memory)
            | set(candidate_tp)
            | set(candidate_cp)
        )
        return sum(
            max(
                stage_compute.get(stage, 0.0),
                stage_memory.get(stage, 0.0),
                candidate_cp.get(stage, 0.0),
            )
            + candidate_tp.get(stage, 0.0)
            for stage in candidate_stages
        )

    startup_sensitivity = {}
    for startup_us in (1.0, 2.5, 4.0):
        candidate = (
            communication
            if math.isclose(
                interconnect_startup_ns,
                startup_us * 1_000.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else _factorized_communication(
                model,
                seq_len=seq_len,
                batch_size=batch_size,
                tp_degree=tp_degree,
                cp_degree=cp_degree,
                local_tokens=int(context["max_local_tokens"]),
                fp_width_bits=fp_width_bits,
                kv_width_bits=kv_width_bits,
                one_way_link_bandwidth_gbps=one_way_bandwidth,
                startup_latency_ns=startup_us * 1_000.0,
            )
        )
        startup_sensitivity[str(startup_us)] = {
            "startup_us": startup_us,
            "latency_ns": total_for_communication(candidate),
            "tp_collective_latency_ns": candidate[
                "tp_collective_latency_ns"
            ],
            "cp_kv_ring_latency_ns": candidate["cp_kv_ring_latency_ns"],
        }
    tp_latency = communication["tp_collective_latency_ns_by_stage"]
    cp_latency = communication["cp_kv_ring_latency_ns_by_stage"]
    stages = set(stage_compute) | set(stage_memory) | set(tp_latency) | set(cp_latency)
    nominal: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    bounds: dict[str, str] = {}
    for stage in sorted(stages):
        compute = stage_compute.get(stage, 0.0)
        memory = stage_memory.get(stage, 0.0)
        tp_comm = tp_latency.get(stage, 0.0)
        cp_comm = cp_latency.get(stage, 0.0)
        nominal[stage] = max(compute, memory, cp_comm) + tp_comm
        lower[stage] = max(compute, memory, cp_comm, tp_comm)
        upper[stage] = compute + memory + cp_comm + tp_comm
        candidates = {
            "compute": compute,
            "memory": memory,
            "cp_communication": cp_comm,
        }
        bounds[stage] = max(candidates, key=candidates.get)

    per_chip_bytes = sum(_traffic_total(bucket) for bucket in per_chip_by_stage.values())
    aggregate_traffic = _aggregate_traffic_breakdown(traffic, chip_count)
    total_latency = sum(nominal.values())
    interconnect_bytes = (
        communication["tp_collective_bytes"]
        + communication["cp_kv_ring_bytes"]
    )
    interconnect_latency = (
        communication["tp_collective_latency_ns"]
        + communication["cp_kv_ring_latency_ns"]
    )
    equivalent_channels = 128.0 * reference_a100_count / chip_count
    return {
        "chip_count": chip_count,
        "reference_a100_count": reference_a100_count,
        "parallel_model": "tp-cp",
        "multi_chip_model": "factorized-tp-cp-v2",
        "tp_degree": tp_degree,
        "cp_degree": cp_degree,
        "tp_cp_legality": "valid_natural_head_sharding",
        "context_partition": context,
        "max_token_fraction": max_token_fraction,
        "max_causal_pair_fraction": max_pair_fraction,
        "parallel_work_census": census,
        "parallel_work_census_coverage": census["coverage"],
        "parallel_work_axis_scales": compute_scales,
        "per_chip_compute_scale": (
            sum(stage_compute.values())
            / (
                census["total_cycles"]
                * float(census["clock_period_ps"])
                / 1000.0
            )
            if census["total_cycles"] > 0
            else 1.0
        ),
        "per_chip_stage_compute_latency_ns": stage_compute,
        "per_chip_stage_memory_latency_ns": stage_memory,
        "per_chip_stage_v4_floor_ns": stage_floor,
        "per_chip_stage_roofline_latency_ns": nominal,
        "per_chip_stage_full_overlap_lower_bound_ns": lower,
        "per_chip_stage_no_overlap_upper_bound_ns": upper,
        "full_overlap_lower_bound_ns": sum(lower.values()),
        "nominal_stage_model_ns": total_latency,
        "no_overlap_upper_bound_ns": sum(upper.values()),
        "communication_overlap_bound": (
            "cp_overlaps_compute_memory;tp_collective_serial_after_stage"
        ),
        "per_chip_stage_bound": bounds,
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
        "per_chip_hbm_traffic": per_chip_by_stage,
        "per_chip_hbm_traffic_breakdown": traffic,
        "aggregate_hbm_traffic_breakdown": aggregate_traffic,
        "hbm_traffic_partition_fidelity": traffic_fidelity,
        "per_chip_hbm_physical_bytes": per_chip_bytes,
        "aggregate_hbm_physical_bytes": per_chip_bytes * chip_count,
        "weight_replication_factor": cp_degree,
        "per_chip_achieved_bandwidth_gbps": (
            per_chip_bytes / total_latency if total_latency > 0 else 0.0
        ),
        "per_chip_bandwidth_utilization": (
            per_chip_bytes / total_latency / per_chip_bandwidth
            if total_latency > 0 and per_chip_bandwidth > 0
            else 0.0
        ),
        "nvlink_port_count": nvlink_port_count,
        "nvlink_peak_bidirectional_bandwidth_gbps": (
            nvlink_port_count * nvlink_port_bidirectional_gbps
        ),
        "nvlink_peak_oneway_bandwidth_gbps": one_way_bandwidth,
        "interconnect_bandwidth_semantics": "architectural_peak_assumption",
        "bandwidth_efficiency": 1.0,
        "interconnect_startup_ns": interconnect_startup_ns,
        "interconnect_startup_sensitivity": startup_sensitivity,
        **communication,
        "interconnect_bytes_by_stage": {
            stage: communication["tp_collective_bytes_by_stage"].get(stage, 0.0)
            + communication["cp_kv_ring_bytes_by_stage"].get(stage, 0.0)
            for stage in stages
        },
        "interconnect_latency_ns_by_stage": {
            stage: tp_latency.get(stage, 0.0) + cp_latency.get(stage, 0.0)
            for stage in stages
        },
        "interconnect_bytes": interconnect_bytes,
        "interconnect_latency_ns": interconnect_latency,
        "latency_ns": total_latency,
        "latency_ms": total_latency / 1e6,
        "multi_chip_fidelity": {
            "compute": "exact_costemitter_opcode_census_analytical_partition",
            "memory": (
                "exact_role_traffic_partition_v4_floor_and_per_request_residual"
            ),
            "communication": "analytical_tp_collective_cp_kv_ring_peak_link",
            "compiler_isa": "single_chip_work_postprocessed_no_distributed_isa",
        },
        "kv_cache_fidelity": (
            "exact_local_cache_occurrences_under_factorized_cp"
            if kv_cache_overlay
            else "aggregate_role_scaled"
        ),
        "kv_cache_overlay": dict(kv_cache_overlay or {}),
    }


def estimate_multi_chip_latency(
    report: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    chip_count: int,
    reference_a100_count: int,
    parallel_model: str,
    aggregate_hbm_bandwidth_gbps: float,
    aggregate_hbm_capacity_bytes: int,
    seq_len: int,
    batch_size: int,
    fp_width_bits: int,
    one_way_link_bandwidth_gbps: float = 1_800.0,
    kv_cache_overlay: Mapping[str, Any] | None = None,
    multi_chip_model: str = "ideal-linear-lower-bound-v1",
    dp_degree: int | None = None,
    tp_degree: int | None = None,
    ep_degree: int = 1,
    kv_width_bits: float | None = None,
    nvlink_port_count: int = 4,
    nvlink_port_bidirectional_gbps: float = (
        DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS
    ),
    interconnect_startup_ns: float = DEFAULT_NVLINK_STARTUP_US * 1_000.0,
) -> dict[str, Any]:
    """Partition a single-chip CostEmitter report and recompute stage roofline."""

    if multi_chip_model not in MULTI_CHIP_MODELS:
        raise ValueError(f"unsupported multi-chip model {multi_chip_model!r}")
    if multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
        if dp_degree is None or tp_degree is None:
            raise ValueError(
                "tile-aware-dp-tp-ep-v4 requires dp_degree and tp_degree"
            )
        # Lazy import avoids a module cycle while allowing the v4 model to
        # reuse the mature v3 tile-count and CostTrace lineage helpers.
        from .multi_chip_dp_model import estimate_tile_aware_dp_tp_ep_latency

        return estimate_tile_aware_dp_tp_ep_latency(
            report,
            model,
            chip_count=chip_count,
            dp_degree=dp_degree,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            reference_a100_count=reference_a100_count,
            aggregate_hbm_bandwidth_gbps=aggregate_hbm_bandwidth_gbps,
            aggregate_hbm_capacity_bytes=aggregate_hbm_capacity_bytes,
            seq_len=seq_len,
            batch_size=batch_size,
            fp_width_bits=fp_width_bits,
            kv_width_bits=(
                float(kv_width_bits)
                if kv_width_bits is not None
                else float(fp_width_bits)
            ),
            nvlink_port_count=nvlink_port_count,
            nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
            interconnect_startup_ns=interconnect_startup_ns,
            kv_cache_overlay=kv_cache_overlay,
        )
    if multi_chip_model == TILE_AWARE_MULTI_CHIP_MODEL:
        if tp_degree is None:
            raise ValueError(
                "tile-aware-tp-cp-ep-v3 requires tp_degree"
            )
        return _estimate_tile_aware_multi_chip_latency(
            report,
            model,
            chip_count=chip_count,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            reference_a100_count=reference_a100_count,
            aggregate_hbm_bandwidth_gbps=aggregate_hbm_bandwidth_gbps,
            aggregate_hbm_capacity_bytes=aggregate_hbm_capacity_bytes,
            seq_len=seq_len,
            batch_size=batch_size,
            fp_width_bits=fp_width_bits,
            kv_width_bits=(
                float(kv_width_bits)
                if kv_width_bits is not None
                else float(fp_width_bits)
            ),
            nvlink_port_count=nvlink_port_count,
            nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
            interconnect_startup_ns=interconnect_startup_ns,
            kv_cache_overlay=kv_cache_overlay,
        )
    if multi_chip_model == "factorized-tp-cp-v2":
        if tp_degree is None:
            raise ValueError("factorized-tp-cp-v2 requires tp_degree")
        return _estimate_factorized_multi_chip_latency(
            report,
            model,
            chip_count=chip_count,
            tp_degree=tp_degree,
            reference_a100_count=reference_a100_count,
            aggregate_hbm_bandwidth_gbps=aggregate_hbm_bandwidth_gbps,
            aggregate_hbm_capacity_bytes=aggregate_hbm_capacity_bytes,
            seq_len=seq_len,
            batch_size=batch_size,
            fp_width_bits=fp_width_bits,
            kv_width_bits=(
                float(kv_width_bits)
                if kv_width_bits is not None
                else float(fp_width_bits)
            ),
            nvlink_port_count=nvlink_port_count,
            nvlink_port_bidirectional_gbps=nvlink_port_bidirectional_gbps,
            interconnect_startup_ns=interconnect_startup_ns,
            kv_cache_overlay=kv_cache_overlay,
        )

    if chip_count <= 0 or reference_a100_count <= 0:
        raise ValueError("chip and A100 reference counts must be positive")
    if parallel_model not in PARALLEL_MODELS:
        raise ValueError(f"unsupported parallel model {parallel_model!r}")
    if aggregate_hbm_bandwidth_gbps <= 0 or aggregate_hbm_capacity_bytes <= 0:
        raise ValueError("aggregate HBM resources must be positive")

    per_chip_bandwidth = aggregate_hbm_bandwidth_gbps / chip_count
    per_chip_capacity = aggregate_hbm_capacity_bytes / chip_count
    baseline_v4_bandwidth = (
        aggregate_hbm_bandwidth_gbps / reference_a100_count
    )
    equivalent_channels = 128.0 * reference_a100_count / chip_count
    calibrated_channels = (8.0, 32.0, 128.0)
    if equivalent_channels in calibrated_channels:
        channel_status = "calibrated_channel_anchor"
    elif calibrated_channels[0] <= equivalent_channels <= calibrated_channels[-1]:
        channel_status = "between_channel_anchors_residual_scaled"
    else:
        channel_status = "channel_extrapolation"
    channel_extrapolation_ratio = max(
        1.0,
        equivalent_channels / calibrated_channels[-1],
        calibrated_channels[0] / equivalent_channels,
    )
    compute_scale, stage_compute, compute_fidelity = _compute_stage_scale(
        report,
        chip_count=chip_count,
        parallel_model=parallel_model,
    )
    baseline_stage_memory = {
        str(stage): float(value)
        for stage, value in (report.get("hbm_stage_latency_ns") or {}).items()
    }
    exact_stage_floor = {
        str(stage): float(value)
        for stage, value in (
            (report.get("compatibility") or {}).get(
                "stage_theoretical_floor_ns", {}
            )
            or {}
        ).items()
    }

    if chip_count == 1 and reference_a100_count == 1:
        stage_memory = dict(baseline_stage_memory)
        memory_fidelity = "exact_single_chip_v4_baseline"
        per_chip_traffic_breakdown = {
            str(group): {
                str(key): {
                    field: float(bucket.get(field, 0))
                    for field in TRAFFIC_FIELDS
                }
                for key, bucket in dict(entries).items()
            }
            for group, entries in dict(
                report.get("hbm_traffic_breakdown") or {}
            ).items()
        }
        traffic_breakdown_fidelity = "exact_single_chip_v4_manifest"
        scaled_traffic = {
            stage: {
                field: float(bucket.get(field, 0))
                for field in TRAFFIC_FIELDS
            }
            for stage, bucket in (
                (report.get("hbm_traffic_breakdown") or {}).get("by_stage") or {}
            ).items()
        }
        baseline_floor = float(
            (report.get("compatibility") or {}).get("theoretical_floor_ns", 0.0)
        )
        stage_floor = {}
        total_bytes = sum(_traffic_total(bucket) for bucket in scaled_traffic.values())
        for stage, bucket in scaled_traffic.items():
            stage_floor[stage] = (
                exact_stage_floor[stage]
                if stage in exact_stage_floor
                else baseline_floor * _traffic_total(bucket) / total_bytes
                if total_bytes > 0
                else 0.0
            )
    else:
        per_chip_traffic_breakdown, traffic_breakdown_fidelity = (
            _scaled_traffic_breakdown(
                report,
                chip_count=chip_count,
                parallel_model=parallel_model,
            )
        )
        if kv_cache_overlay and parallel_model == "tp-sp":
            global_loads = float(kv_cache_overlay["global_tile_loads"])
            local_loads = float(kv_cache_overlay["local_tile_loads"])
            if global_loads <= 0:
                raise ValueError("global K/V tile loads must be positive")
            # Role scaling already divided aggregate traffic by N. Restore the
            # exact local/global occurrence ratio for attention K/V only.
            relative_scale = chip_count * local_loads / global_loads
            per_chip_traffic_breakdown = _apply_attention_kv_overlay(
                per_chip_traffic_breakdown,
                relative_scale=relative_scale,
            )
            traffic_breakdown_fidelity += (
                ":exact_local_cache_occurrence_overlay"
            )
        scaled_traffic = {
            str(stage): {
                field: float(bucket.get(field, 0.0))
                for field in TRAFFIC_FIELDS
            }
            for stage, bucket in per_chip_traffic_breakdown.get(
                "by_stage", {}
            ).items()
        }
        traffic_fidelity = traffic_breakdown_fidelity
        baseline_breakdown = report.get("hbm_traffic_breakdown") or {}
        baseline_by_stage = baseline_breakdown.get("by_stage") or {}
        baseline_floor = float(
            (report.get("compatibility") or {}).get("theoretical_floor_ns", 0.0)
        )
        baseline_total_bytes = sum(
            _traffic_total(bucket) for bucket in baseline_by_stage.values()
        )
        stage_floor = {}
        stage_memory = {}
        for stage in set(baseline_stage_memory) | set(scaled_traffic):
            original_bucket = baseline_by_stage.get(stage, {})
            original_bytes = _traffic_total(original_bucket)
            new_bytes = _traffic_total(scaled_traffic.get(stage, {}))
            original_floor = exact_stage_floor.get(
                stage,
                (
                    baseline_floor * original_bytes / baseline_total_bytes
                    if baseline_total_bytes > 0
                    else 0.0
                ),
            )
            # V4 is calibrated against one reference HBM subsystem. Recompute
            # only its theoretical bandwidth floor for R/N resources.
            new_floor = (
                original_floor
                * (new_bytes / original_bytes)
                * (baseline_v4_bandwidth / per_chip_bandwidth)
                if original_bytes > 0
                else 0.0
            )
            residual = max(
                float(baseline_stage_memory.get(stage, 0.0)) - original_floor,
                0.0,
            )
            residual_scale = new_bytes / original_bytes if original_bytes > 0 else 0.0
            stage_floor[stage] = new_floor
            stage_memory[stage] = new_floor + residual * residual_scale
        memory_fidelity = (
            "v4_floor_recomputed_residual_traffic_scaled:" + traffic_fidelity
        )

    aggregate_traffic_breakdown = _aggregate_traffic_breakdown(
        per_chip_traffic_breakdown,
        chip_count,
    )

    communication_bytes, communication_latency = _communication_by_stage(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        chip_count=chip_count,
        fp_width_bits=fp_width_bits,
        one_way_link_bandwidth_gbps=one_way_link_bandwidth_gbps,
    )
    stages = (
        set(stage_compute)
        | set(stage_memory)
        | set(communication_latency)
    )
    stage_latency = {}
    stage_bound = {}
    for stage in sorted(stages):
        compute = stage_compute.get(stage, 0.0)
        memory = stage_memory.get(stage, 0.0)
        communication = communication_latency.get(stage, 0.0)
        stage_latency[stage] = max(compute, memory) + communication
        stage_bound[stage] = "compute" if compute >= memory else "memory"

    per_chip_physical_bytes = sum(
        _traffic_total(bucket) for bucket in scaled_traffic.values()
    )
    total_latency_ns = sum(stage_latency.values())
    achieved_bandwidth = (
        per_chip_physical_bytes / total_latency_ns
        if total_latency_ns > 0
        else 0.0
    )
    memory_fidelity = f"{memory_fidelity};channel={channel_status}"
    return {
        "chip_count": chip_count,
        "reference_a100_count": reference_a100_count,
        "parallel_model": parallel_model,
        "multi_chip_model": "ideal-linear-lower-bound-v1",
        "multi_chip_fidelity": {
            "compute": compute_fidelity,
            "memory": memory_fidelity,
            "communication": "nvlink6_peak_one_way_lower_bound",
            "compiler_isa": "single_chip_aggregate_work_postprocessed",
        },
        "kv_cache_fidelity": (
            "exact_compiler_schedule_single_chip"
            if chip_count == 1
            else (
                "exact_local_cache_occurrences_under_optimistic_tp_sp"
                if kv_cache_overlay and parallel_model == "tp-sp"
                else "aggregate_role_scaled"
            )
        ),
        "kv_cache_overlay": dict(kv_cache_overlay or {}),
        "aggregate_hbm_capacity_bytes": aggregate_hbm_capacity_bytes,
        "aggregate_hbm_bandwidth_gbps": aggregate_hbm_bandwidth_gbps,
        "per_chip_hbm_capacity_bytes": per_chip_capacity,
        "per_chip_hbm_bandwidth_gbps": per_chip_bandwidth,
        "per_chip_equivalent_hbm_channels": equivalent_channels,
        "hbm_channel_calibration_status": channel_status,
        "hbm_channel_extrapolation_ratio": channel_extrapolation_ratio,
        "per_chip_compute_scale": compute_scale,
        "per_chip_stage_compute_latency_ns": stage_compute,
        "per_chip_stage_memory_latency_ns": stage_memory,
        "per_chip_stage_v4_floor_ns": stage_floor,
        "per_chip_stage_roofline_latency_ns": stage_latency,
        "per_chip_stage_bound": stage_bound,
        "per_chip_hbm_traffic": scaled_traffic,
        "per_chip_hbm_traffic_breakdown": per_chip_traffic_breakdown,
        "aggregate_hbm_traffic_breakdown": aggregate_traffic_breakdown,
        "hbm_traffic_partition_fidelity": traffic_breakdown_fidelity,
        "per_chip_hbm_physical_bytes": per_chip_physical_bytes,
        "aggregate_hbm_physical_bytes": per_chip_physical_bytes * chip_count,
        "per_chip_achieved_bandwidth_gbps": achieved_bandwidth,
        "per_chip_bandwidth_utilization": (
            achieved_bandwidth / per_chip_bandwidth
            if per_chip_bandwidth > 0
            else 0.0
        ),
        "interconnect_bytes_by_stage": communication_bytes,
        "interconnect_latency_ns_by_stage": communication_latency,
        "interconnect_bytes": sum(communication_bytes.values()),
        "interconnect_latency_ns": sum(communication_latency.values()),
        "latency_ns": total_latency_ns,
        "latency_ms": total_latency_ns / 1e6,
    }


def aggregate_area(
    *,
    core_area_mm2: float,
    core_area_p10_mm2: float,
    core_area_p50_mm2: float,
    core_area_p90_mm2: float,
    chip_count: int,
    endpoint_overhead_fraction: float | None = None,
    nvlink_port_count: int | None = None,
    endpoint_area_mm2_per_port: float = ENDPOINT_AREA_MM2_PER_PORT["nominal"],
) -> dict[str, float]:
    """Add per-chip endpoint overhead and aggregate all PLENA dice."""

    if core_area_mm2 < 0 or chip_count <= 0:
        raise ValueError("core area must be nonnegative and chip_count positive")
    if nvlink_port_count is not None:
        if nvlink_port_count not in {1, 2, 4}:
            raise ValueError("nvlink_port_count must be one of 1, 2, or 4")
        if endpoint_area_mm2_per_port < 0:
            raise ValueError("endpoint area per port must be nonnegative")
        endpoint = nvlink_port_count * endpoint_area_mm2_per_port
        endpoint_p10 = (
            nvlink_port_count * ENDPOINT_AREA_MM2_PER_PORT["optimistic"]
        )
        endpoint_p90 = (
            nvlink_port_count * ENDPOINT_AREA_MM2_PER_PORT["conservative"]
        )
        physical = core_area_mm2 + endpoint
        return {
            "core_area_mm2": core_area_mm2,
            "endpoint_area_mm2": endpoint,
            "endpoint_area_p10_mm2": endpoint_p10,
            "endpoint_area_p50_mm2": endpoint,
            "endpoint_area_p90_mm2": endpoint_p90,
            "endpoint_area_model": "fixed_nvlink_c2c_port_proxy_v1",
            "endpoint_area_mm2_per_port": endpoint_area_mm2_per_port,
            "nvlink_port_count": nvlink_port_count,
            "physical_chip_area_mm2": physical,
            "total_silicon_area_mm2": physical * chip_count,
            "total_silicon_area_p10_mm2": (
                core_area_p10_mm2 + endpoint_p10
            )
            * chip_count,
            "total_silicon_area_p50_mm2": (
                core_area_p50_mm2 + endpoint
            )
            * chip_count,
            "total_silicon_area_p90_mm2": (
                core_area_p90_mm2 + endpoint_p90
            )
            * chip_count,
        }
    if endpoint_overhead_fraction is None or endpoint_overhead_fraction < 0:
        raise ValueError("endpoint overhead must be nonnegative")
    physical_scale = 1.0 + endpoint_overhead_fraction
    return {
        "core_area_mm2": core_area_mm2,
        "endpoint_area_mm2": core_area_mm2 * endpoint_overhead_fraction,
        "physical_chip_area_mm2": core_area_mm2 * physical_scale,
        "total_silicon_area_mm2": core_area_mm2 * physical_scale * chip_count,
        "total_silicon_area_p10_mm2": core_area_p10_mm2
        * physical_scale
        * chip_count,
        "total_silicon_area_p50_mm2": core_area_p50_mm2
        * physical_scale
        * chip_count,
        "total_silicon_area_p90_mm2": core_area_p90_mm2
        * physical_scale
        * chip_count,
    }


def fp16_kv_handoff(
    model: Mapping[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    one_way_link_bandwidth_gbps: float = 1_800.0,
) -> dict[str, float]:
    """Estimate a one-time FP16 K+V transfer to a disaggregated decode chip."""

    values = (
        2
        * int(model["num_hidden_layers"])
        * batch_size
        * seq_len
        * int(model["num_key_value_heads"])
        * int(model["head_dim"])
    )
    byte_count = values * 2
    return {
        "fp16_kv_handoff_bytes": float(byte_count),
        "fp16_kv_handoff_latency_ns": byte_count / one_way_link_bandwidth_gbps,
        "fp16_kv_handoff_latency_ms": byte_count
        / one_way_link_bandwidth_gbps
        / 1e6,
    }


def estimate_decode_kv_handoff(
    model: Mapping[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    source_chip_count: int,
    decode_chip_count: int,
    source_port_count: int,
    decode_port_count: int,
    per_port_one_way_bandwidth_gbps: float,
    startup_ns: float = 0.0,
) -> dict[str, Any]:
    """Estimate a balanced FP16 KV transfer into a decode system."""

    integer_inputs = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "source_chip_count": source_chip_count,
        "decode_chip_count": decode_chip_count,
        "source_port_count": source_port_count,
        "decode_port_count": decode_port_count,
    }
    if any(int(value) <= 0 for value in integer_inputs.values()):
        raise ValueError(
            "decode KV handoff dimensions and endpoint counts must be positive"
        )
    if per_port_one_way_bandwidth_gbps <= 0:
        raise ValueError("per-port one-way bandwidth must be positive")
    if startup_ns < 0:
        raise ValueError("handoff startup must be nonnegative")

    legacy = fp16_kv_handoff(
        model,
        seq_len=seq_len,
        batch_size=batch_size,
        one_way_link_bandwidth_gbps=per_port_one_way_bandwidth_gbps,
    )
    byte_count = float(legacy["fp16_kv_handoff_bytes"])
    source_bandwidth = (
        source_chip_count
        * source_port_count
        * per_port_one_way_bandwidth_gbps
    )
    sink_bandwidth = (
        decode_chip_count
        * decode_port_count
        * per_port_one_way_bandwidth_gbps
    )
    effective_bandwidth = min(source_bandwidth, sink_bandwidth)
    connection_waves = math.ceil(
        source_chip_count / (decode_chip_count * decode_port_count)
    )
    startup_total_ns = connection_waves * startup_ns
    payload_latency_ns = byte_count / effective_bandwidth
    service_latency_ns = startup_total_ns + payload_latency_ns

    return {
        "fp16_kv_handoff_bytes": byte_count,
        "fp16_kv_handoff_max_source_bytes": math.ceil(
            byte_count / source_chip_count
        ),
        "fp16_kv_handoff_source_chip_count": source_chip_count,
        "fp16_kv_handoff_decode_chip_count": decode_chip_count,
        "fp16_kv_handoff_source_port_count": source_port_count,
        "fp16_kv_handoff_decode_port_count": decode_port_count,
        "fp16_kv_handoff_per_port_oneway_bandwidth_gbps": (
            per_port_one_way_bandwidth_gbps
        ),
        "fp16_kv_handoff_source_aggregate_bandwidth_gbps": source_bandwidth,
        "fp16_kv_handoff_sink_aggregate_bandwidth_gbps": sink_bandwidth,
        "fp16_kv_handoff_effective_bandwidth_gbps": effective_bandwidth,
        "fp16_kv_handoff_bottleneck": (
            "source"
            if source_bandwidth < sink_bandwidth
            else "decode_sink"
            if sink_bandwidth < source_bandwidth
            else "balanced_endpoints"
        ),
        "fp16_kv_handoff_connection_waves": connection_waves,
        "fp16_kv_handoff_startup_latency_ns": startup_total_ns,
        "fp16_kv_handoff_payload_latency_ns": payload_latency_ns,
        "fp16_kv_handoff_latency_ns": service_latency_ns,
        "fp16_kv_handoff_latency_ms": service_latency_ns / 1e6,
        "fp16_kv_handoff_precision": "FP16",
        "fp16_kv_handoff_model": "dual_endpoint_peak_bandwidth_v1",
        "fp16_kv_handoff_bandwidth_semantics": "architectural_peak",
        "fp16_kv_handoff_balanced_sharding_assumed": True,
    }
