"""Full decode projection driven by layer-exact held-out MoE routes.

Only the routed-expert stage and its streamed weight planes are replaced.  The
normal decode loop continues to price attention, the replicated router and
route-control operations, local MX output head, growing packed KV, and the TP
and KVP collectives.  Every result remains a non-selection analytic receipt.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .body_weight_layout import (
        EXPERT_ID_PARALLEL,
        EXPERT_TENSOR_PARALLEL,
        BodyWeightPhysicalLayout,
        build_body_weight_physical_layout,
    )
    from .expert_id_trace_balancing import (
        EXPERT_ID_MAPPING,
        TENSOR_MAPPING,
        ExpertIdBalanceConfig,
        canonical_hash,
        expert_id_body_inputs_for_layer,
        file_hash,
        validate_expert_id_balancing_report,
        validate_hardware_binding_for_perf_model,
        validate_expert_id_window_overlay,
    )
    from .expert_placement import (
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
    )
    from .handoff import LINK_GENS
    from .packed_kv import DENSE_SELECTOR
    from .physical_ledger import (
        PlaneBytes,
        build_physical_decode_ledger,
    )
    from ..memory.memory_model import conservative_unique_experts
    from ..performance import disagg_decode as decode
    from ..performance.perf_model import PerfModel
except ImportError:  # script-style imports
    from body_weight_layout import (  # type: ignore[no-redef]
        EXPERT_ID_PARALLEL,
        EXPERT_TENSOR_PARALLEL,
        BodyWeightPhysicalLayout,
        build_body_weight_physical_layout,
    )
    from expert_id_trace_balancing import (  # type: ignore[no-redef]
        EXPERT_ID_MAPPING,
        TENSOR_MAPPING,
        ExpertIdBalanceConfig,
        canonical_hash,
        expert_id_body_inputs_for_layer,
        file_hash,
        validate_expert_id_balancing_report,
        validate_hardware_binding_for_perf_model,
        validate_expert_id_window_overlay,
    )
    from expert_placement import (  # type: ignore[no-redef]
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
    )
    from handoff import LINK_GENS  # type: ignore[no-redef]
    from packed_kv import DENSE_SELECTOR  # type: ignore[no-redef]
    from physical_ledger import (  # type: ignore[no-redef]
        PlaneBytes,
        build_physical_decode_ledger,
    )
    from memory_model import conservative_unique_experts  # type: ignore[no-redef]
    import disagg_decode as decode  # type: ignore[no-redef]
    from perf_model import PerfModel  # type: ignore[no-redef]


FULL_DECODE_SCHEMA = "plena-qwen3-moe-expert-id-full-decode-projection/v1"
FULL_DECODE_RECEIPT_SCHEMA = (
    "plena-qwen3-moe-expert-id-full-decode-projection-receipt/v1"
)
ROUTE_PROJECTION_SCHEMA = decode.LAYER_EXACT_MOE_ROUTE_PROJECTION_SCHEMA
WINDOW_APPLICATION = (
    "held_out_batch_window_stationary_route_proxy_for_each_projected_decode_step/v1"
)
CONTROL_FREQUENCY = "frequency_aware_expert_id"
CONTROL_CYCLIC = "cyclic_expert_id"
CONTROL_TENSOR_EXACT = "tensor_parallel_held_out_exact"
CONTROL_TENSOR_NATIVE = "tensor_parallel_native_base"
CONTROL_NAMES = (
    CONTROL_FREQUENCY,
    CONTROL_CYCLIC,
    CONTROL_TENSOR_EXACT,
    CONTROL_TENSOR_NATIVE,
)


@dataclass(frozen=True)
class FullDecodeProjectionConfig:
    """Serving-loop settings not already sealed by the route artifact."""

    input_sequence_tokens: int
    output_sequence_tokens: int
    stride: int = 1
    sram_policy: str = "streaming"
    kv_head_reuse: bool = False
    batch_packed_attention: bool = False
    kv_layout: str = DENSE_SELECTOR
    output_head_location: str = decode.DECODE_MX_HEAD
    execution_mode: str = decode.LEGACY_AGGREGATE_BANDWIDTH
    window_application: str = WINDOW_APPLICATION

    def __post_init__(self) -> None:
        for field in (
            "input_sequence_tokens",
            "output_sequence_tokens",
            "stride",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.stride > self.output_sequence_tokens:
            raise ValueError("stride cannot exceed output sequence length")
        if self.sram_policy not in decode.SRAM_POLICIES:
            raise ValueError("unsupported SRAM policy")
        if not isinstance(self.kv_head_reuse, bool):
            raise ValueError("kv_head_reuse must be boolean")
        if not isinstance(self.batch_packed_attention, bool):
            raise ValueError("batch_packed_attention must be boolean")
        if self.kv_layout != DENSE_SELECTOR:
            raise ValueError("full held-out projection requires dense-selector KV")
        if self.output_head_location != decode.DECODE_MX_HEAD:
            raise ValueError("full held-out projection requires the local MX head")
        if self.execution_mode != decode.LEGACY_AGGREGATE_BANDWIDTH:
            raise ValueError("route projection is an analytic sensitivity only")
        if self.window_application != WINDOW_APPLICATION:
            raise ValueError("unsupported held-out window application policy")


def _hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_hash", None)
    return body | {"content_hash": canonical_hash(body)}


def _plane(value: PlaneBytes) -> dict[str, int]:
    return {
        "element_raw": int(value.element_raw),
        "element_aligned": int(value.element_aligned),
        "scale_raw": int(value.scale_raw),
        "scale_aligned": int(value.scale_aligned),
        "total_aligned": int(value.total_aligned),
    }


def _sum_planes(values: Sequence[PlaneBytes]) -> PlaneBytes:
    result = PlaneBytes()
    for value in values:
        result += value
    return result


def _validate_target(
    report: Mapping[str, Any],
    balanced_overlay: Mapping[str, Any],
    cyclic_overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
) -> ExpertIdBalanceConfig:
    validate_expert_id_balancing_report(report)
    validate_expert_id_window_overlay(balanced_overlay, report=report)
    validate_expert_id_window_overlay(cyclic_overlay, report=report)
    if (
        balanced_overlay.get("policy") != CONTROL_FREQUENCY
        or cyclic_overlay.get("policy") != CONTROL_CYCLIC
        or balanced_overlay.get("window_index") != cyclic_overlay.get("window_index")
        or balanced_overlay.get("study_config_content_hash")
        != cyclic_overlay.get("study_config_content_hash")
    ):
        raise ValueError("full decode controls must bind one held-out window")
    balance = ExpertIdBalanceConfig(**report["study_config"])
    validate_hardware_binding_for_perf_model(
        report["hardware_binding"], perf, balance
    )
    expected_dims = {
        "hidden": 2048,
        "inter": 768,
        "layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "experts_per_token": TOP_K,
    }
    for field, expected in expected_dims.items():
        if int(dims.get(field, -1)) != expected:
            raise ValueError(f"decoder dimension {field} differs from target")
    for field in ("heads", "kv_heads", "head_dim", "vocab"):
        if int(dims.get(field, 0)) <= 0:
            raise ValueError(f"decoder dimension {field} is missing")
    if int(dims["heads"]) % balance.tensor_parallel_degree or int(
        dims["kv_heads"]
    ) % balance.tensor_parallel_degree:
        raise ValueError("TP must own complete query and KV heads")
    if dims.get("moe_route_repricing") is not None:
        raise ValueError("full route projection rejects a nested model overlay")
    required_precision = {
        "attn_elem",
        "attn_bits",
        "ffn_elem",
        "ffn_bits",
        "kv_elem",
        "kv_bits",
        "head_elem",
        "head_bits",
        "block_size",
        "m_bits",
        "density_exp",
        "lm_head_quantized",
        "profile_id",
        "head_activation_bits",
        "head_activation_elem",
        "head_activation_label",
        "head_vector_format",
        "head_matrix_storage_format",
        "head_logit_container_format",
        "head_bf16_container_precision_recovery",
        "head_operand_family_supported",
        "head_operand_family_binding",
        "head_numerical_oracle_rule",
        "head_partial_conversion_rule",
        "head_hardware_bit_parity_verified",
        "head_accumulation_chain",
        "head_numerical_matrix_mlen",
    }
    missing = sorted(required_precision - set(precision))
    if missing:
        raise ValueError("local MX head v3 precision binding missing: " + ",".join(missing))
    if (
        int(precision["ffn_elem"]) != balance.ffn_element_bits
        or not math.isclose(
            float(precision["ffn_bits"]),
            balance.ffn_effective_bits,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(precision["block_size"]) != balance.mx_block_size
    ):
        raise ValueError("FFN precision differs from held-out route evidence")
    if (
        precision["lm_head_quantized"] is not True
        or int(precision["head_elem"]) != int(precision["attn_elem"])
        or not math.isclose(
            float(precision["head_bits"]),
            float(precision["attn_bits"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or precision["head_matrix_storage_format"]
        != precision["head_vector_format"]
        or precision["head_logit_container_format"] != "BF16"
        or precision["head_bf16_container_precision_recovery"] is not False
        or int(precision["head_numerical_matrix_mlen"]) != int(perf.mlen)
    ):
        raise ValueError("precision does not satisfy the local MX head v3 contract")
    if balance.link_generation not in LINK_GENS or not math.isclose(
        balance.tp_link_bandwidth_bytes_per_s,
        LINK_GENS[balance.link_generation],
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("route evidence link rate differs from decode topology")
    return balance


def _observations_for_window(
    report: Mapping[str, Any], window_index: int
) -> list[Mapping[str, Any]]:
    values = [
        row
        for row in report["held_out_evaluation"]["observations"]
        if int(row["window_index"]) == int(window_index)
    ]
    values.sort(key=lambda row: int(row["layer"]))
    if len(values) != NUM_LAYERS or [int(row["layer"]) for row in values] != list(
        range(NUM_LAYERS)
    ):
        raise ValueError("held-out tensor control lacks exact layer coverage")
    return values


def _native_collective(
    dims: Mapping[str, Any], balance: ExpertIdBalanceConfig, mode: str
) -> dict[str, float]:
    return decode.collective_cost_per_step(
        dict(dims),
        batch=balance.batch_size,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        link_ports=_decode_link_ports(balance),
        link_generation=balance.link_generation,
        include_local_output_selection=True,
        expert_parallel_mode=mode,
    )


def _decode_link_ports(balance: ExpertIdBalanceConfig) -> int:
    """Preserve the sealed TP ports and add one independent KVP port."""

    return (
        balance.tp_link_ports * int(balance.tensor_parallel_degree > 1)
        + int(balance.kv_parallel_degree > 1)
    )


def _expert_id_route_projection(
    overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
) -> dict[str, Any]:
    one_layer_dims = dict(dims) | {"layers": 1}
    native_collective = _native_collective(dims, balance, EXPERT_ID_PARALLEL)
    expected_collective_count = int(balance.tensor_parallel_degree > 1)
    layer_rows: list[dict[str, Any]] = []
    for layer in range(NUM_LAYERS):
        inputs = expert_id_body_inputs_for_layer(overlay, layer)
        layout = build_body_weight_physical_layout(
            one_layer_dims,
            precision,
            mlen=balance.mlen,
            tp=balance.tensor_parallel_degree,
            kvp=balance.kv_parallel_degree,
            batch=balance.batch_size,
            unique_experts=int(inputs["unique_experts"]),
            expert_parallel_mode=EXPERT_ID_PARALLEL,
            active_experts_per_rank=inputs["active_experts_per_rank"],
            expert_owner_by_id=inputs["expert_owner_by_id"],
            include_lm_head=False,
            alignment_bytes=balance.weight_alignment_bytes,
        )
        cycles: list[int] = []
        assignments = overlay["layers"][layer]["assignment_count_by_rank"]
        for rank, histogram in enumerate(
            inputs["expert_token_count_histogram_by_rank"]
        ):
            timing = perf.moe_decode_expert_timing_from_histogram(
                balance.hidden_size,
                balance.expert_intermediate_size,
                histogram,
                owned_experts=NUM_EXPERTS // balance.tensor_parallel_degree,
                expected_route_assignments=int(assignments[rank]),
                source="full_decode_held_out_expert_id_rank_histogram",
            )
            cycles.append(int(timing["expert_stage_cycles"]))
        if tuple(cycles) != inputs["expert_stage_cycles_by_rank"]:
            raise ValueError("expert-ID timing changed since overlay creation")
        if (
            layout.slowest_rank.ffn_streamed.total_aligned
            != max(inputs["expert_weight_hbm_bytes_by_rank"])
            or layout.system.ffn_streamed.total_aligned
            != sum(inputs["expert_weight_hbm_bytes_by_rank"])
            * balance.kv_parallel_degree
        ):
            raise ValueError("expert-ID streamed body planes changed")
        collective = inputs["expert_output_collective"]
        native_per_layer_bytes = (
            native_collective["expert_output_collective_slowest_rank_bytes"]
            / NUM_LAYERS
        )
        native_per_layer_system_bytes = (
            native_collective["expert_output_collective_system_bytes"]
            / NUM_LAYERS
        )
        native_per_layer_time = (
            native_collective["expert_output_collective_time_s"] / NUM_LAYERS
        )
        if (
            int(collective["count_per_layer"]) != expected_collective_count
            or not math.isclose(
                float(collective["slowest_rank_bytes"]),
                native_per_layer_bytes,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(collective["system_bytes"]),
                native_per_layer_system_bytes,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(collective["time_s"]),
                native_per_layer_time,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or int(collective["source_hidden_dispatch_bytes"]) != 0
        ):
            raise ValueError("expert-ID output collective differs from decode loop")
        layer_rows.append(
            {
                "layer": layer,
                "expert_owner_by_id": list(inputs["expert_owner_by_id"]),
                "active_experts_per_rank": list(inputs["active_experts_per_rank"]),
                "assignment_count_by_rank": list(assignments),
                "physical_assignment_count_by_rank_across_kvp": list(
                    inputs["physical_assignment_count_by_rank_across_kvp"]
                ),
                "expert_token_count_histogram_by_rank": [
                    dict(value)
                    for value in inputs["expert_token_count_histogram_by_rank"]
                ],
                "expert_stage_cycles_by_rank": cycles,
                "slowest_rank_expert_stage_cycles": max(cycles),
                "slowest_rank_expert_streamed": _plane(
                    layout.slowest_rank.ffn_streamed
                ),
                "system_expert_streamed": _plane(layout.system.ffn_streamed),
                "slowest_rank_expert_resident": _plane(
                    layout.slowest_rank.ffn_resident
                ),
                "system_expert_resident": _plane(layout.system.ffn_resident),
                "logical_route_assignments": balance.batch_size * TOP_K,
                "physical_whole_expert_assignments_across_kvp": (
                    balance.batch_size * TOP_K * balance.kv_parallel_degree
                ),
                "source_hidden_dispatch_bytes": 0,
                "expert_output_collective_count": expected_collective_count,
            }
        )
    return _route_projection(
        policy=str(overlay["policy"]),
        mapping=EXPERT_ID_MAPPING,
        expert_parallel_mode=EXPERT_ID_PARALLEL,
        layers=layer_rows,
        report_content_hash=str(overlay["report_content_hash"]),
        source_content_hash=str(overlay["content_hash"]),
        window_index=int(overlay["window_index"]),
        first_step_index=int(overlay["first_step_index"]),
        last_step_index=int(overlay["last_step_index"]),
        balance=balance,
    )


def _tensor_route_projection(
    report: Mapping[str, Any],
    window_index: int,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
) -> dict[str, Any]:
    one_layer_dims = dict(dims) | {"layers": 1}
    observations = _observations_for_window(report, window_index)
    native_collective = _native_collective(dims, balance, EXPERT_TENSOR_PARALLEL)
    native_output_per_layer = (
        float(native_collective["tp_bytes"]) / 2.0 / NUM_LAYERS
        if balance.tensor_parallel_degree > 1
        else 0.0
    )
    native_output_system_per_layer = (
        native_output_per_layer
        * balance.tensor_parallel_degree
        * balance.kv_parallel_degree
    )
    native_output_time_per_layer = (
        native_output_per_layer
        / (
            LINK_GENS[balance.link_generation]
            * balance.tp_link_ports
        )
        if balance.tensor_parallel_degree > 1
        else 0.0
    )
    layer_rows: list[dict[str, Any]] = []
    for layer, observation in enumerate(observations):
        metrics = observation["policies"]["tensor_parallel_control"]
        unique = int(metrics["active_expert_count_by_rank"][0])
        layout = build_body_weight_physical_layout(
            one_layer_dims,
            precision,
            mlen=balance.mlen,
            tp=balance.tensor_parallel_degree,
            kvp=balance.kv_parallel_degree,
            batch=balance.batch_size,
            unique_experts=unique,
            expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
            include_lm_head=False,
            alignment_bytes=balance.weight_alignment_bytes,
        )
        cycles: list[int] = []
        assignments = balance.batch_size * TOP_K
        for rank, histogram in enumerate(
            metrics["expert_token_count_histogram_by_rank"]
        ):
            timing = perf.moe_decode_expert_timing_from_histogram(
                balance.hidden_size,
                int(metrics["local_intermediate_width_by_rank"][rank]),
                dict(histogram),
                owned_experts=NUM_EXPERTS,
                expected_route_assignments=assignments,
                source="full_decode_held_out_tensor_shard_global_histogram",
            )
            cycles.append(int(timing["expert_stage_cycles"]))
        if cycles != list(metrics["expert_stage_cycles_by_rank"]):
            raise ValueError("tensor timing changed since report creation")
        if (
            layout.slowest_rank.ffn_streamed.total_aligned
            != int(metrics["slowest_rank_expert_weight_hbm_bytes"])
            or layout.system.ffn_streamed.total_aligned
            != int(metrics["system_expert_weight_hbm_bytes"])
        ):
            raise ValueError("tensor streamed body planes changed")
        collective = metrics["expert_output_collective"]
        if not math.isclose(
            float(collective["slowest_rank_bytes"]),
            native_output_per_layer,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(collective["system_bytes"]),
            native_output_system_per_layer,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(collective["time_s"]),
            native_output_time_per_layer,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or int(collective["source_hidden_dispatch_bytes"]) != 0:
            raise ValueError("tensor output collective differs from decode loop")
        layer_rows.append(
            {
                "layer": layer,
                "active_experts_per_rank": list(
                    metrics["active_expert_count_by_rank"]
                ),
                "assignment_count_by_rank": list(metrics["assignment_count_by_rank"]),
                "expert_token_count_histogram_by_rank": [
                    dict(value)
                    for value in metrics["expert_token_count_histogram_by_rank"]
                ],
                "expert_stage_cycles_by_rank": cycles,
                "slowest_rank_expert_stage_cycles": max(cycles),
                "slowest_rank_expert_streamed": _plane(
                    layout.slowest_rank.ffn_streamed
                ),
                "system_expert_streamed": _plane(layout.system.ffn_streamed),
                "slowest_rank_expert_resident": _plane(
                    layout.slowest_rank.ffn_resident
                ),
                "system_expert_resident": _plane(layout.system.ffn_resident),
                "logical_route_assignments": assignments,
                "physical_whole_expert_assignments_across_kvp": (
                    assignments * balance.kv_parallel_degree
                ),
                "physical_tensor_shard_executions_across_tp_and_kvp": (
                    assignments
                    * balance.tensor_parallel_degree
                    * balance.kv_parallel_degree
                ),
                "source_hidden_dispatch_bytes": 0,
                "expert_output_collective_count": int(
                    balance.tensor_parallel_degree > 1
                ),
            }
        )
    return _route_projection(
        policy=CONTROL_TENSOR_EXACT,
        mapping=TENSOR_MAPPING,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
        layers=layer_rows,
        report_content_hash=str(report["content_hash"]),
        source_content_hash=str(report["content_hash"]),
        window_index=window_index,
        first_step_index=int(observations[0]["first_step_index"]),
        last_step_index=int(observations[0]["last_step_index"]),
        balance=balance,
    )


def _route_projection(
    *,
    policy: str,
    mapping: str,
    expert_parallel_mode: str,
    layers: Sequence[Mapping[str, Any]],
    report_content_hash: str,
    source_content_hash: str,
    window_index: int,
    first_step_index: int,
    last_step_index: int,
    balance: ExpertIdBalanceConfig,
) -> dict[str, Any]:
    if len(layers) != NUM_LAYERS:
        raise AssertionError("route projection must retain every layer")
    logical = sum(int(row["logical_route_assignments"]) for row in layers)
    physical = sum(
        int(row["physical_whole_expert_assignments_across_kvp"])
        for row in layers
    )
    rank_element = sum(
        int(row["slowest_rank_expert_streamed"]["element_aligned"])
        for row in layers
    )
    rank_scale = sum(
        int(row["slowest_rank_expert_streamed"]["scale_aligned"])
        for row in layers
    )
    system_bytes = sum(
        int(row["system_expert_streamed"]["total_aligned"])
        for row in layers
    )
    rank_resident = sum(
        int(row["slowest_rank_expert_resident"]["total_aligned"])
        for row in layers
    )
    system_resident = sum(
        int(row["system_expert_resident"]["total_aligned"])
        for row in layers
    )
    body = {
        "schema": ROUTE_PROJECTION_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "report_content_hash": report_content_hash,
        "source_content_hash": source_content_hash,
        "window_index": window_index,
        "first_step_index": first_step_index,
        "last_step_index": last_step_index,
        "study_config_content_hash": canonical_hash(asdict(balance)),
        "policy": policy,
        "mapping": mapping,
        "expert_parallel_mode": expert_parallel_mode,
        "batch_size": balance.batch_size,
        "tensor_parallel_degree": balance.tensor_parallel_degree,
        "kv_parallel_degree": balance.kv_parallel_degree,
        "layer_count": NUM_LAYERS,
        "global_layer_collapse_allowed": False,
        "layers": [dict(value) for value in layers],
        "totals": {
            "sum_of_per_layer_slowest_rank_expert_stage_cycles": sum(
                int(row["slowest_rank_expert_stage_cycles"]) for row in layers
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_element_bytes": (
                rank_element
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_scale_bytes": rank_scale,
            "sum_of_per_layer_slowest_rank_expert_streamed_bytes": (
                rank_element + rank_scale
            ),
            "system_expert_streamed_bytes": system_bytes,
            "sum_of_per_layer_slowest_rank_expert_resident_bytes": rank_resident,
            "system_expert_resident_bytes": system_resident,
            "logical_route_assignments": logical,
            "expected_logical_route_assignments": (
                NUM_LAYERS * balance.batch_size * TOP_K
            ),
            "physical_whole_expert_assignments_across_kvp": physical,
            "expected_physical_whole_expert_assignments_across_kvp": (
                NUM_LAYERS
                * balance.batch_size
                * TOP_K
                * balance.kv_parallel_degree
            ),
            "source_hidden_dispatch_bytes": 0,
            "expert_output_collective_count": (
                NUM_LAYERS * int(balance.tensor_parallel_degree > 1)
            ),
            "resident_bytes_use_all_layer_specific_ownership_records": True,
        },
        "classification": {
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
        },
    }
    return _hashed(body)


def _expert_id_full_body_layout(
    overlay: Mapping[str, Any],
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
) -> BodyWeightPhysicalLayout:
    inputs = expert_id_body_inputs_for_layer(overlay, 0)
    return build_body_weight_physical_layout(
        dims,
        precision,
        mlen=balance.mlen,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        batch=balance.batch_size,
        unique_experts=int(inputs["unique_experts"]),
        expert_parallel_mode=EXPERT_ID_PARALLEL,
        active_experts_per_rank=inputs["active_experts_per_rank"],
        expert_owner_by_id=inputs["expert_owner_by_id"],
        include_lm_head=True,
        alignment_bytes=balance.weight_alignment_bytes,
    )


def _tensor_full_body_layout(
    route_projection: Mapping[str, Any] | None,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
) -> BodyWeightPhysicalLayout:
    unique = (
        sum(
            int(value)
            for value in route_projection["layers"][0]["active_experts_per_rank"]
        )
        // balance.tensor_parallel_degree
        if route_projection is not None
        else int(
            dims.get("moe_unique_experts_per_step")
            or conservative_unique_experts(NUM_EXPERTS, TOP_K, balance.batch_size)
        )
    )
    return build_body_weight_physical_layout(
        dims,
        precision,
        mlen=balance.mlen,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        batch=balance.batch_size,
        unique_experts=unique,
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
        include_lm_head=True,
        alignment_bytes=balance.weight_alignment_bytes,
    )


def _validate_full_layout_residence(
    layout: BodyWeightPhysicalLayout,
    route_projection: Mapping[str, Any],
) -> None:
    totals = route_projection["totals"]
    if (
        layout.slowest_rank.ffn_resident.total_aligned
        != int(totals["sum_of_per_layer_slowest_rank_expert_resident_bytes"])
        or layout.system.ffn_resident.total_aligned
        != int(totals["system_expert_resident_bytes"])
    ):
        raise ValueError("full body expert residence differs from all layer records")


def _capacity(
    body_layout: BodyWeightPhysicalLayout,
    *,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
    config: FullDecodeProjectionConfig,
) -> dict[str, Any]:
    context = config.input_sequence_tokens + config.output_sequence_tokens
    base = build_physical_decode_ledger(
        dict(dims),
        dict(precision),
        perf.config,
        context=context,
        batch=balance.batch_size,
        hbm_capacity_bytes=balance.hbm_capacity_bytes_per_chip,
        runtime_hbm_reserve_bytes=(
            balance.runtime_hbm_reserve_bytes_per_chip
        ),
        kv_layout=config.kv_layout,
        include_lm_head=True,
    )
    slowest_kv, system_kv, kv_provenance = decode._partitioned_kv_ledgers(
        dims,
        precision,
        context=context,
        batch=balance.batch_size,
        mlen=balance.mlen,
        kv_layout=config.kv_layout,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
    )
    ledger = decode._partition_physical_ledger(
        base,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        hbm_per_chip=balance.hbm_capacity_bytes_per_chip,
        sram_policy=config.sram_policy,
        batch=balance.batch_size,
        body_weight_layout=body_layout,
        slowest_rank_kv=slowest_kv,
        system_kv=system_kv,
    )
    slowest_required = int(ledger.slowest_rank_hbm_required_bytes or 0)
    per_chip = int(ledger.per_chip_hbm_capacity_bytes or 0)
    if slowest_required <= 0 or per_chip != balance.hbm_capacity_bytes_per_chip:
        raise AssertionError("partitioned capacity omitted its slowest rank")
    return {
        "context_tokens_at_capacity_check": context,
        "chip_count": balance.tensor_parallel_degree
        * balance.kv_parallel_degree,
        "slowest_rank_resident_weight_bytes": int(
            body_layout.slowest_rank.resident.total_aligned
        ),
        "system_resident_weight_bytes": int(
            body_layout.system.resident.total_aligned
        ),
        "slowest_rank_kv_bytes": int(slowest_kv.total_bytes),
        "system_kv_bytes": int(system_kv.total_bytes),
        "runtime_hbm_reserve_bytes_per_chip": (
            balance.runtime_hbm_reserve_bytes_per_chip
        ),
        "slowest_rank_hbm_required_bytes": slowest_required,
        "per_chip_hbm_capacity_bytes": per_chip,
        "slowest_rank_capacity_margin_bytes": per_chip - slowest_required,
        "system_hbm_required_bytes": int(ledger.hbm_required_bytes),
        "system_hbm_capacity_bytes": int(ledger.hbm_capacity_bytes),
        "fits_hbm": bool(ledger.fits_hbm),
        "fits_runtime": bool(ledger.fits_runtime),
        "max_resident_batch": int(ledger.max_resident_batch),
        "max_runtime_batch": int(ledger.max_runtime_batch),
        "sram_fits": bool(ledger.sram.fits),
        "kv_partition_provenance": dict(kv_provenance),
    }


def _replace_expert_traffic(
    target: dict[str, float],
    original: PlaneBytes,
    replacement: PlaneBytes,
) -> None:
    target["weight_element_read_bytes"] += float(
        replacement.element_aligned - original.element_aligned
    )
    target["weight_scale_read_bytes"] += float(
        replacement.scale_aligned - original.scale_aligned
    )
    if (
        target["weight_element_read_bytes"] < 0
        or target["weight_scale_read_bytes"] < 0
    ):
        raise AssertionError("full decode route replacement made traffic negative")


def _step_proofs(
    *,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
    config: FullDecodeProjectionConfig,
    body_layout: BodyWeightPhysicalLayout,
    expert_parallel_mode: str,
    route_projection: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    tp = balance.tensor_parallel_degree
    kvp = balance.kv_parallel_degree
    peak_bw = decode.peak_hbm_bw_bytes(perf.config)
    overfetch = decode.matrix_overfetch_factor(perf.config)
    density = decode.compute_density(precision)
    collective = _native_collective(dims, balance, expert_parallel_mode)
    route_cycles = None
    rank_expert = None
    system_expert = None
    if route_projection is not None:
        route_cycles = tuple(
            float(row["slowest_rank_expert_stage_cycles"])
            for row in route_projection["layers"]
        )
        rank_expert = _sum_planes(
            tuple(
                PlaneBytes(
                    element_raw=int(row["slowest_rank_expert_streamed"]["element_raw"]),
                    element_aligned=int(
                        row["slowest_rank_expert_streamed"]["element_aligned"]
                    ),
                    scale_raw=int(row["slowest_rank_expert_streamed"]["scale_raw"]),
                    scale_aligned=int(
                        row["slowest_rank_expert_streamed"]["scale_aligned"]
                    ),
                )
                for row in route_projection["layers"]
            )
        )
        system_expert = _sum_planes(
            tuple(
                PlaneBytes(
                    element_raw=int(row["system_expert_streamed"]["element_raw"]),
                    element_aligned=int(
                        row["system_expert_streamed"]["element_aligned"]
                    ),
                    scale_raw=int(row["system_expert_streamed"]["scale_raw"]),
                    scale_aligned=int(
                        row["system_expert_streamed"]["scale_aligned"]
                    ),
                )
                for row in route_projection["layers"]
            )
        )
    samples: list[dict[str, Any]] = []
    totals = {
        "time_s": 0.0,
        "compute_s": 0.0,
        "memory_s": 0.0,
        "collective_s": 0.0,
        "system_bytes": 0.0,
    }
    for offset in range(0, config.output_sequence_tokens, config.stride):
        context = config.input_sequence_tokens + offset
        span = min(config.stride, config.output_sequence_tokens - offset)
        components = decode._partitioned_component_cycles(
            perf,
            dict(dims),
            context,
            balance.batch_size,
            tp=tp,
            kvp=kvp,
            include_lm_head=True,
            kv_layout=config.kv_layout,
            packed_q1_timing_contract=None,
            batch_packed_attention=config.batch_packed_attention,
            kv_head_reuse=config.kv_head_reuse,
            body_layout=body_layout,
            expert_parallel_mode=expert_parallel_mode,
        )
        native_expert_cycles = float(components["rank_local_routed_experts"])
        if route_cycles is not None:
            components["rank_local_routed_experts"] = math.fsum(route_cycles)
        component_sum = math.fsum(components.values())
        compute_s = component_sum / (decode.FREQ_HZ * density)
        rank_step, system_step = decode._partitioned_step_traffic_pair(
            dims,
            precision,
            context=context,
            batch=balance.batch_size,
            mlen=balance.mlen,
            kv_layout=config.kv_layout,
            tp=tp,
            kvp=kvp,
            weights=body_layout,
            kv_head_reuse=config.kv_head_reuse,
        )
        rank_policy = decode._traffic_for_policy(
            rank_step, body_layout.slowest_rank, config.sram_policy
        )
        system_policy = decode._traffic_for_policy(
            system_step, body_layout.system, config.sram_policy
        )
        rank_traffic = {
            name: float(getattr(rank_policy, name))
            for name in rank_policy.__dataclass_fields__
        }
        system_traffic = {
            name: float(getattr(system_policy, name))
            for name in system_policy.__dataclass_fields__
        }
        if rank_expert is not None and system_expert is not None:
            _replace_expert_traffic(
                rank_traffic, body_layout.slowest_rank.ffn_streamed, rank_expert
            )
            _replace_expert_traffic(
                system_traffic, body_layout.system.ffn_streamed, system_expert
            )
        rank_read = sum(
            value
            for name, value in rank_traffic.items()
            if name.endswith("_read_bytes")
        )
        rank_write = sum(
            value
            for name, value in rank_traffic.items()
            if name.endswith("_write_bytes")
        )
        system_read = sum(
            value
            for name, value in system_traffic.items()
            if name.endswith("_read_bytes")
        )
        system_write = sum(
            value
            for name, value in system_traffic.items()
            if name.endswith("_write_bytes")
        )
        rank_bytes = rank_read * overfetch + rank_write
        system_bytes = system_read * overfetch + system_write
        memory_s = rank_bytes / peak_bw
        step_s = max(compute_s, memory_s) + float(collective["time_s"])
        if not math.isclose(component_sum, math.fsum(components.values())):
            raise AssertionError("component cycle proof does not sum")
        samples.append(
            {
                "context_tokens": context,
                "represented_output_steps": span,
                "component_cycles": dict(components),
                "component_cycle_sum": component_sum,
                "native_unreplaced_routed_expert_cycles": native_expert_cycles,
                "compute_time_s": compute_s,
                "slowest_rank_hbm_time_s": memory_s,
                "collective_time_s": float(collective["time_s"]),
                "projected_step_time_s": step_s,
                "slowest_rank_hbm_bytes_after_overfetch": rank_bytes,
                "system_hbm_bytes_after_overfetch": system_bytes,
            }
        )
        totals["time_s"] += step_s * span
        totals["compute_s"] += compute_s * span
        totals["memory_s"] += memory_s * span
        totals["collective_s"] += float(collective["time_s"]) * span
        totals["system_bytes"] += system_bytes * span
    return samples, totals


def _run_control(
    *,
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    balance: ExpertIdBalanceConfig,
    config: FullDecodeProjectionConfig,
    body_layout: BodyWeightPhysicalLayout,
    expert_parallel_mode: str,
    route_projection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    peak_bw = decode.peak_hbm_bw_bytes(perf.config)
    overfetch = decode.matrix_overfetch_factor(perf.config)
    loop = decode.run_decode_loop(
        perf,
        None,
        dict(dims),
        dict(precision),
        config.input_sequence_tokens,
        config.output_sequence_tokens,
        balance.batch_size,
        peak_bw,
        config.stride,
        overfetch,
        batch_packed_attention=config.batch_packed_attention,
        n_chips=balance.tensor_parallel_degree * balance.kv_parallel_degree,
        bw_model=None,
        kv_layout=config.kv_layout,
        ideal_perf=perf,
        physical_weights=None,
        include_lm_head=True,
        packed_q1_timing_contract=None,
        tp=balance.tensor_parallel_degree,
        kvp=balance.kv_parallel_degree,
        link_ports=_decode_link_ports(balance),
        link_generation=balance.link_generation,
        sram_policy=config.sram_policy,
        legacy_ideal_parallelism=False,
        kv_head_reuse=config.kv_head_reuse,
        body_weight_layout=body_layout,
        expert_parallel_mode=expert_parallel_mode,
        layer_exact_moe_route_projection=route_projection,
        execution_mode=config.execution_mode,
    )
    samples, proof = _step_proofs(
        perf=perf,
        dims=dims,
        precision=precision,
        balance=balance,
        config=config,
        body_layout=body_layout,
        expert_parallel_mode=expert_parallel_mode,
        route_projection=route_projection,
    )
    output_steps = config.output_sequence_tokens
    checks = {
        "total_time_matches_step_proof": math.isclose(
            float(loop["total_time"]),
            proof["time_s"],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "compute_time_matches_step_proof": math.isclose(
            float(loop["avg_realized_compute_seconds"]),
            proof["compute_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "memory_time_matches_step_proof": math.isclose(
            float(loop["avg_memory_seconds"]),
            proof["memory_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "collective_time_matches_step_proof": math.isclose(
            float(loop["avg_collective_seconds"]),
            proof["collective_s"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "system_hbm_bytes_match_step_proof": math.isclose(
            float(loop["avg_bytes_per_batch_step"]),
            proof["system_bytes"] / output_steps,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ),
    }
    if not all(checks.values()):
        raise AssertionError("full decode loop differs from its component proof")
    hook_receipt = loop["layer_exact_moe_route_projection"]
    if route_projection is None:
        if hook_receipt is not None:
            raise AssertionError("native tensor control unexpectedly applied a hook")
    elif (
        not isinstance(hook_receipt, Mapping)
        or hook_receipt.get("content_hash") != route_projection.get("content_hash")
        or hook_receipt.get("other_decode_components_changed") is not False
    ):
        raise AssertionError("decode loop did not bind the exact route projection")
    capacity = _capacity(
        body_layout,
        perf=perf,
        dims=dims,
        precision=precision,
        balance=balance,
        config=config,
    )
    return {
        "route_hook_applied": route_projection is not None,
        "route_projection_content_hash": (
            route_projection["content_hash"] if route_projection is not None else None
        ),
        "body_layout_content_hash": canonical_hash(body_layout.to_dict()),
        "body_layout_mapping": body_layout.provenance[
            "expert_routing_mapping"
        ],
        "resident_body_layout_semantics": (
            "all_48_layer_specific_resident_planes_sum_exactly;"
            "representative_owner_ids_are_storage_equivalent_only"
            if route_projection is not None
            else "native_tensor_body_layout"
        ),
        "loop": loop,
        "component_and_hbm_step_proof": {
            "sample_count": len(samples),
            "samples": samples,
            "weighted_totals": proof,
            "checks": checks,
            "only_routed_expert_component_replaced": route_projection is not None,
        },
        "capacity": capacity,
    }


def _metric_delta(candidate: float, reference: float, *, higher_is_better: bool) -> dict:
    candidate = float(candidate)
    reference = float(reference)
    if not math.isfinite(candidate) or not math.isfinite(reference) or reference == 0:
        raise ValueError("comparison metrics must be finite with nonzero reference")
    delta = candidate - reference
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_minus_reference": delta,
        "candidate_over_reference": candidate / reference,
        "percent_change": 100.0 * delta / reference,
        "nonregression_observed": (
            candidate >= reference if higher_is_better else candidate <= reference
        ),
    }


def _comparison(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict:
    candidate_loop = candidate["loop"]
    reference_loop = reference["loop"]
    candidate_capacity = candidate["capacity"]
    reference_capacity = reference["capacity"]
    return {
        "tpot": _metric_delta(
            candidate_loop["tpot"], reference_loop["tpot"], higher_is_better=False
        ),
        "throughput_tokens_per_s": _metric_delta(
            candidate_loop["tps"], reference_loop["tps"], higher_is_better=True
        ),
        "system_hbm_bytes_per_batch_step": _metric_delta(
            candidate_loop["avg_bytes_per_batch_step"],
            reference_loop["avg_bytes_per_batch_step"],
            higher_is_better=False,
        ),
        "slowest_rank_hbm_required_bytes": _metric_delta(
            candidate_capacity["slowest_rank_hbm_required_bytes"],
            reference_capacity["slowest_rank_hbm_required_bytes"],
            higher_is_better=False,
        ),
        "system_hbm_required_bytes": _metric_delta(
            candidate_capacity["system_hbm_required_bytes"],
            reference_capacity["system_hbm_required_bytes"],
            higher_is_better=False,
        ),
        "candidate_fits_hbm": candidate_capacity["fits_hbm"],
        "reference_fits_hbm": reference_capacity["fits_hbm"],
        "headline_win_claimed": False,
    }


def build_expert_id_full_decode_projection(
    report: Mapping[str, Any],
    balanced_overlay: Mapping[str, Any],
    cyclic_overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    config: FullDecodeProjectionConfig,
) -> dict[str, Any]:
    """Reprice a full projected decode against held-out and native controls."""

    balance = _validate_target(
        report, balanced_overlay, cyclic_overlay, perf, dims, precision
    )
    balanced_projection = _expert_id_route_projection(
        balanced_overlay, perf, dims, precision, balance
    )
    cyclic_projection = _expert_id_route_projection(
        cyclic_overlay, perf, dims, precision, balance
    )
    tensor_projection = _tensor_route_projection(
        report,
        int(balanced_overlay["window_index"]),
        perf,
        dims,
        precision,
        balance,
    )
    balanced_layout = _expert_id_full_body_layout(
        balanced_overlay, dims, precision, balance
    )
    cyclic_layout = _expert_id_full_body_layout(
        cyclic_overlay, dims, precision, balance
    )
    tensor_exact_layout = _tensor_full_body_layout(
        tensor_projection, dims, precision, balance
    )
    tensor_native_layout = _tensor_full_body_layout(
        None, dims, precision, balance
    )
    _validate_full_layout_residence(balanced_layout, balanced_projection)
    _validate_full_layout_residence(cyclic_layout, cyclic_projection)
    _validate_full_layout_residence(tensor_exact_layout, tensor_projection)
    controls = {
        CONTROL_FREQUENCY: _run_control(
            perf=perf,
            dims=dims,
            precision=precision,
            balance=balance,
            config=config,
            body_layout=balanced_layout,
            expert_parallel_mode=EXPERT_ID_PARALLEL,
            route_projection=balanced_projection,
        ),
        CONTROL_CYCLIC: _run_control(
            perf=perf,
            dims=dims,
            precision=precision,
            balance=balance,
            config=config,
            body_layout=cyclic_layout,
            expert_parallel_mode=EXPERT_ID_PARALLEL,
            route_projection=cyclic_projection,
        ),
        CONTROL_TENSOR_EXACT: _run_control(
            perf=perf,
            dims=dims,
            precision=precision,
            balance=balance,
            config=config,
            body_layout=tensor_exact_layout,
            expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
            route_projection=tensor_projection,
        ),
        CONTROL_TENSOR_NATIVE: _run_control(
            perf=perf,
            dims=dims,
            precision=precision,
            balance=balance,
            config=config,
            body_layout=tensor_native_layout,
            expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
            route_projection=None,
        ),
    }
    candidate = controls[CONTROL_FREQUENCY]
    vs_cyclic = _comparison(candidate, controls[CONTROL_CYCLIC])
    vs_tensor_exact = _comparison(candidate, controls[CONTROL_TENSOR_EXACT])
    vs_tensor_native = _comparison(candidate, controls[CONTROL_TENSOR_NATIVE])
    heldout_max_rank_nonregression = vs_cyclic["tpot"][
        "nonregression_observed"
    ]
    route_projections = {
        CONTROL_FREQUENCY: balanced_projection,
        CONTROL_CYCLIC: cyclic_projection,
        CONTROL_TENSOR_EXACT: tensor_projection,
    }
    logical_expected = NUM_LAYERS * balance.batch_size * TOP_K
    physical_expected = logical_expected * balance.kv_parallel_degree
    if any(
        projection["totals"]["logical_route_assignments"] != logical_expected
        or projection["totals"][
            "physical_whole_expert_assignments_across_kvp"
        ]
        != physical_expected
        for projection in route_projections.values()
    ):
        raise AssertionError("full decode controls do not conserve route work")
    body = {
        "schema": FULL_DECODE_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "report_content_hash": report["content_hash"],
        "balanced_overlay_content_hash": balanced_overlay["content_hash"],
        "cyclic_overlay_content_hash": cyclic_overlay["content_hash"],
        "trace_content_hash": report["trace_content_hash"],
        "study_config_content_hash": report["study_config_content_hash"],
        "projection_config": asdict(config),
        "projection_config_content_hash": canonical_hash(asdict(config)),
        "precision": dict(precision),
        "precision_content_hash": canonical_hash(dict(precision)),
        "dims": dict(dims),
        "dims_content_hash": canonical_hash(dict(dims)),
        "hardware_binding": dict(report["hardware_binding"]),
        "window_index": int(balanced_overlay["window_index"]),
        "first_step_index": int(balanced_overlay["first_step_index"]),
        "last_step_index": int(balanced_overlay["last_step_index"]),
        "window_application": WINDOW_APPLICATION,
        "route_projections": route_projections,
        "controls": controls,
        "comparisons": {
            "frequency_aware_vs_cyclic_expert_id": vs_cyclic,
            "frequency_aware_vs_tensor_held_out_exact": vs_tensor_exact,
            "frequency_aware_vs_tensor_native_base": vs_tensor_native,
            "held_out_max_rank_tpot_nonregression_vs_cyclic_observed": (
                heldout_max_rank_nonregression
            ),
            "headline_win_claimed": False,
        },
        "conservation": {
            "logical_route_assignments_per_projected_batch_step": logical_expected,
            "physical_whole_expert_assignments_across_kvp_per_projected_batch_step": (
                physical_expected
            ),
            "kvp_route_semantics": "identical_route_replica_not_new_trace_sample",
            "source_hidden_dispatch_bytes": 0,
            "tp_expert_output_allreduce_count_per_batch_step": (
                NUM_LAYERS * int(balance.tensor_parallel_degree > 1)
            ),
            "router_hidden_and_route_filter_mapping": EXPERT_ID_MAPPING,
            "all_controls_conserved": True,
        },
        "tensor_native_base_parity_contract": {
            "api": "disagg_decode.run_decode_loop",
            "layer_exact_route_hook": None,
            "route_hook_applied": False,
            "all_nonroute_components_use_native_loop": True,
            "test_reinvokes_native_loop_with_identical_inputs": True,
        },
        "classification": {
            "evidence": "held_out_trace_exact_full_decode_analytic_projection",
            "full_tpot_repriced": True,
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
            "compiler_validated": False,
            "emulator_validated": False,
            "rtl_validated": False,
            "power_calibrated": False,
            "headline_eligible": False,
            "headline_win_claimed": False,
            "blockers": [
                "held_out_window_is_a_stationary_post_hoc_batch_route_proxy",
                "full_decode_timing_is_an_unvalidated_analytic_projection",
                "matched_compiler_emulator_rtl_and_power_receipts_missing",
            ],
        },
    }
    return _hashed(body)


def validate_expert_id_full_decode_projection(
    projection: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
    balanced_overlay: Mapping[str, Any] | None = None,
    cyclic_overlay: Mapping[str, Any] | None = None,
) -> None:
    """Reject changed inputs, collapsed layers, and promoted analytic claims."""

    if not isinstance(projection, Mapping):
        raise ValueError("full decode projection must be an object")
    body = dict(projection)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("full decode projection content hash mismatch")
    if projection.get("schema") != FULL_DECODE_SCHEMA:
        raise ValueError("unsupported full decode projection schema")
    classification = projection.get("classification")
    if (
        not isinstance(classification, Mapping)
        or classification.get("full_tpot_repriced") is not True
        or any(
            classification.get(field) is not False
            for field in (
                "publication_rankable",
                "hardware_rankable",
                "selection_eligible",
                "timing_selection_allowed",
                "compiler_validated",
                "emulator_validated",
                "rtl_validated",
                "power_calibrated",
                "headline_eligible",
                "headline_win_claimed",
            )
        )
    ):
        raise ValueError("full decode projection must remain fail-closed")
    if projection.get("window_application") != WINDOW_APPLICATION:
        raise ValueError("full decode projection window application changed")
    config = FullDecodeProjectionConfig(**projection["projection_config"])
    if canonical_hash(asdict(config)) != projection.get(
        "projection_config_content_hash"
    ):
        raise ValueError("full decode projection config hash changed")
    if canonical_hash(dict(projection["precision"])) != projection.get(
        "precision_content_hash"
    ) or canonical_hash(dict(projection["dims"])) != projection.get(
        "dims_content_hash"
    ):
        raise ValueError("full decode model or precision binding changed")
    controls = projection.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != set(CONTROL_NAMES):
        raise ValueError("full decode controls are incomplete")
    routes = projection.get("route_projections")
    if not isinstance(routes, Mapping) or set(routes) != {
        CONTROL_FREQUENCY,
        CONTROL_CYCLIC,
        CONTROL_TENSOR_EXACT,
    }:
        raise ValueError("full decode route projections are incomplete")
    for name, route in routes.items():
        route_body = dict(route)
        route_hash = route_body.pop("content_hash", None)
        layers = route.get("layers")
        totals = route.get("totals")
        if (
            route_hash != canonical_hash(route_body)
            or route.get("schema") != ROUTE_PROJECTION_SCHEMA
            or route.get("global_layer_collapse_allowed") is not False
            or int(route.get("window_index", -1))
            != int(projection.get("window_index", -2))
            or int(route.get("first_step_index", -1))
            != int(projection.get("first_step_index", -2))
            or int(route.get("last_step_index", -1))
            != int(projection.get("last_step_index", -2))
            or not isinstance(layers, list)
            or len(layers) != NUM_LAYERS
            or [int(row.get("layer", -1)) for row in layers]
            != list(range(NUM_LAYERS))
            or not isinstance(totals, Mapping)
            or totals.get("source_hidden_dispatch_bytes") != 0
            or totals.get("logical_route_assignments")
            != totals.get("expected_logical_route_assignments")
            or totals.get("physical_whole_expert_assignments_across_kvp")
            != totals.get(
                "expected_physical_whole_expert_assignments_across_kvp"
            )
            or totals.get("sum_of_per_layer_slowest_rank_expert_stage_cycles")
            != sum(int(row["slowest_rank_expert_stage_cycles"]) for row in layers)
            or totals.get("sum_of_per_layer_slowest_rank_expert_streamed_bytes")
            != sum(
                int(row["slowest_rank_expert_streamed"]["total_aligned"])
                for row in layers
            )
            or totals.get("sum_of_per_layer_slowest_rank_expert_resident_bytes")
            != sum(
                int(row["slowest_rank_expert_resident"]["total_aligned"])
                for row in layers
            )
            or totals.get("system_expert_resident_bytes")
            != sum(
                int(row["system_expert_resident"]["total_aligned"])
                for row in layers
            )
            or totals.get("resident_bytes_use_all_layer_specific_ownership_records")
            is not True
        ):
            raise ValueError(f"full decode route projection {name} is malformed")
        if any(
            row.get("source_hidden_dispatch_bytes") != 0
            or row.get("expert_output_collective_count")
            != int(int(route["tensor_parallel_degree"]) > 1)
            for row in layers
        ):
            raise ValueError("route projection dispatch or collective changed")
        route_classification = route.get("classification")
        if not isinstance(route_classification, Mapping) or any(
            route_classification.get(field) is not False
            for field in (
                "publication_rankable",
                "hardware_rankable",
                "selection_eligible",
                "timing_selection_allowed",
            )
        ):
            raise ValueError("route projection must remain fail-closed")
        tp = int(route["tensor_parallel_degree"])
        kvp = int(route["kv_parallel_degree"])
        batch = int(route["batch_size"])
        if route.get("expert_parallel_mode") == EXPERT_ID_PARALLEL:
            if route.get("mapping") != EXPERT_ID_MAPPING or any(
                not isinstance(row.get("expert_owner_by_id"), list)
                or len(row["expert_owner_by_id"]) != NUM_EXPERTS
                or any(
                    row["expert_owner_by_id"].count(rank) != NUM_EXPERTS // tp
                    for rank in range(tp)
                )
                or sum(row.get("assignment_count_by_rank", [])) != batch * TOP_K
                or row.get("physical_assignment_count_by_rank_across_kvp")
                != [
                    int(value) * kvp
                    for value in row.get("assignment_count_by_rank", [])
                ]
                for row in layers
            ):
                raise ValueError("full expert-ID ownership or KVP work changed")
        elif (
            route.get("expert_parallel_mode") != EXPERT_TENSOR_PARALLEL
            or route.get("mapping") != TENSOR_MAPPING
        ):
            raise ValueError("full tensor route mapping changed")
        control = controls[name]
        if (
            control.get("route_hook_applied") is not True
            or control.get("route_projection_content_hash") != route["content_hash"]
            or control.get("loop", {})
            .get("layer_exact_moe_route_projection", {})
            .get("content_hash")
            != route["content_hash"]
            or not all(
                control.get("component_and_hbm_step_proof", {})
                .get("checks", {})
                .values()
            )
        ):
            raise ValueError("full decode loop did not apply its route projection")
    native = controls[CONTROL_TENSOR_NATIVE]
    parity = projection.get("tensor_native_base_parity_contract")
    if (
        native.get("route_hook_applied") is not False
        or native.get("route_projection_content_hash") is not None
        or native.get("loop", {}).get("layer_exact_moe_route_projection") is not None
        or not isinstance(parity, Mapping)
        or parity.get("layer_exact_route_hook") is not None
        or parity.get("all_nonroute_components_use_native_loop") is not True
    ):
        raise ValueError("native tensor parity control is not the base loop")
    for name, control in controls.items():
        capacity = control.get("capacity")
        proof = control.get("component_and_hbm_step_proof")
        if (
            not isinstance(capacity, Mapping)
            or capacity.get("fits_hbm")
            is not (
                int(capacity["slowest_rank_hbm_required_bytes"])
                <= int(capacity["per_chip_hbm_capacity_bytes"])
                and int(capacity["system_hbm_required_bytes"])
                <= int(capacity["system_hbm_capacity_bytes"])
            )
            or not isinstance(proof, Mapping)
            or proof.get("sample_count") != len(proof.get("samples", []))
            or sum(
                int(row["represented_output_steps"])
                for row in proof.get("samples", [])
            )
            != config.output_sequence_tokens
            or any(
                not math.isclose(
                    float(row["component_cycle_sum"]),
                    math.fsum(float(value) for value in row["component_cycles"].values()),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                for row in proof.get("samples", [])
            )
        ):
            raise ValueError(f"full decode control {name} does not conserve")
    comparisons = projection.get("comparisons")
    if (
        not isinstance(comparisons, Mapping)
        or comparisons.get("headline_win_claimed") is not False
        or any(
            value.get("headline_win_claimed") is not False
            for key, value in comparisons.items()
            if key.startswith("frequency_aware_vs_")
        )
    ):
        raise ValueError("full decode comparison promoted a headline")
    conservation = projection.get("conservation")
    if (
        not isinstance(conservation, Mapping)
        or conservation.get("source_hidden_dispatch_bytes") != 0
        or conservation.get("all_controls_conserved") is not True
        or conservation.get("router_hidden_and_route_filter_mapping")
        != EXPERT_ID_MAPPING
    ):
        raise ValueError("full decode physical work does not conserve")
    if report is not None:
        validate_expert_id_balancing_report(report)
        if projection.get("report_content_hash") != report.get("content_hash"):
            raise ValueError("full decode projection binds a different report")
    if balanced_overlay is not None:
        validate_expert_id_window_overlay(balanced_overlay, report=report)
        if projection.get("balanced_overlay_content_hash") != balanced_overlay.get(
            "content_hash"
        ):
            raise ValueError("full decode projection binds a different balanced overlay")
    if cyclic_overlay is not None:
        validate_expert_id_window_overlay(cyclic_overlay, report=report)
        if projection.get("cyclic_overlay_content_hash") != cyclic_overlay.get(
            "content_hash"
        ):
            raise ValueError("full decode projection binds a different cyclic overlay")


def audit_expert_id_full_decode_projection(
    projection: Mapping[str, Any],
    report: Mapping[str, Any],
    balanced_overlay: Mapping[str, Any],
    cyclic_overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
    config: FullDecodeProjectionConfig,
) -> dict[str, Any]:
    """Recompute the complete projected loop and reject any drift."""

    validate_expert_id_full_decode_projection(
        projection,
        report=report,
        balanced_overlay=balanced_overlay,
        cyclic_overlay=cyclic_overlay,
    )
    expected = build_expert_id_full_decode_projection(
        report,
        balanced_overlay,
        cyclic_overlay,
        perf,
        dims,
        precision,
        config,
    )
    if dict(projection) != expected:
        raise ValueError("full decode projection differs from exact recomputation")
    return expected


def _encode(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_install(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace concurrently installed artifact: {path}"
                )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_expert_id_full_decode_projection(
    projection: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Install one immutable, content-addressed full-decode artifact."""

    validate_expert_id_full_decode_projection(projection)
    path = output_dir.resolve() / (
        f"expert_id_full_decode_window_{int(projection['window_index']):06d}."
        f"{projection['content_hash']}.json"
    )
    payload = _encode(projection)
    _atomic_install(path, payload)
    receipt_body = {
        "schema": FULL_DECODE_RECEIPT_SCHEMA,
        "path": str(path),
        "sha256": file_hash(path),
        "content_hash": projection["content_hash"],
        "report_content_hash": projection["report_content_hash"],
        "trace_content_hash": projection["trace_content_hash"],
        "window_index": projection["window_index"],
        "full_tpot_repriced": True,
        "publication_rankable": False,
        "selection_eligible": False,
        "timing_selection_allowed": False,
    }
    receipt = _hashed(receipt_body)
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    _atomic_install(receipt_path, _encode(receipt))
    return receipt


__all__ = [
    "CONTROL_CYCLIC",
    "CONTROL_FREQUENCY",
    "CONTROL_TENSOR_EXACT",
    "CONTROL_TENSOR_NATIVE",
    "FULL_DECODE_SCHEMA",
    "FullDecodeProjectionConfig",
    "audit_expert_id_full_decode_projection",
    "build_expert_id_full_decode_projection",
    "materialize_expert_id_full_decode_projection",
    "validate_expert_id_full_decode_projection",
]
