"""Multi-chip aggregation for the PLENA system-energy model.

The compiler emits aggregate single-chip work.  This module applies the same
explicit TP partition used by :mod:`analytic_models.performance.multi_chip_model`
before evaluating on-chip power.  External HBM and interconnect energy are then
added at aggregate-system scope so provisioned memory capacity is never counted
once per PLENA die.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from analytic_models.performance.multi_chip_model import (
    classify_parallel_work_axis,
)

from .external_memory import estimate_external_hbm_power
from .power_model import estimate_onchip_power
from .system_power import estimate_system_power


POWER_DIR = Path(__file__).resolve().parent
DEFAULT_INTERCONNECT_ENERGY = (
    POWER_DIR / "calibration/interconnect_energy_nvlink_proxy_v1.json"
)
_TILE_AWARE_MULTI_CHIP_MODELS = frozenset(
    {"tile-aware-tp-cp-ep-v3", "tile-aware-dp-tp-ep-v4"}
)
_TRAFFIC_FIELDS = (
    "physical_read_bytes",
    "physical_write_bytes",
    "payload_read_bytes",
    "payload_write_bytes",
    "read_requests",
    "write_requests",
)


@lru_cache(maxsize=8)
def _read_interconnect_artifact(selected: str) -> dict[str, Any]:
    payload = json.loads(Path(selected).read_text())
    if payload.get("model") != "interconnect_energy_nvlink_proxy_v1":
        raise ValueError(
            "unsupported interconnect-energy artifact "
            f"{payload.get('model')!r}"
        )
    return payload


def _load_interconnect_artifact(path: str | Path | None) -> dict[str, Any]:
    return _read_interconnect_artifact(
        str(Path(path or DEFAULT_INTERCONNECT_ENERGY).resolve())
    )


def _traffic_total(bucket: Mapping[str, Any]) -> float:
    return float(bucket.get("physical_read_bytes", 0)) + float(
        bucket.get("physical_write_bytes", 0)
    )


def _group_totals(
    breakdown: Mapping[str, Any],
    group: str,
) -> dict[str, dict[str, float]]:
    return {
        str(key): {
            field: float(dict(bucket).get(field, 0))
            for field in _TRAFFIC_FIELDS
        }
        for key, bucket in dict(breakdown.get(group) or {}).items()
    }


def _traffic_summary(
    breakdown: Mapping[str, Any],
) -> dict[str, float]:
    by_stage = _group_totals(breakdown, "by_stage")
    return {
        field: math.fsum(
            float(bucket.get(field, 0)) for bucket in by_stage.values()
        )
        for field in _TRAFFIC_FIELDS
    }


def _stage_scale(
    original: Mapping[str, Any],
    per_chip: Mapping[str, Any],
    stage: str,
) -> float:
    original_bucket = dict((original.get("by_stage") or {}).get(stage) or {})
    per_chip_bucket = dict((per_chip.get("by_stage") or {}).get(stage) or {})
    original_total = _traffic_total(original_bucket)
    return _traffic_total(per_chip_bucket) / original_total if original_total else 0.0


def _dma_action_opcode(action: Mapping[str, Any]) -> str | None:
    controller_actions = {
        "matrix_prefetch": "H_PREFETCH_M",
        "vector_prefetch": "H_PREFETCH_V",
        "vector_writeback": "H_STORE_V",
    }
    component = str(action.get("component", ""))
    family = str(action.get("action", ""))
    if component == "hbm_controller":
        return controller_actions.get(family)
    precision = str(action.get("precision", ""))
    if (
        component in {"matrix_sram", "vector_sram"}
        and not precision.startswith(("M_", "V_", "S_", "C_", "implicit_"))
    ):
        if component == "matrix_sram":
            return "H_PREFETCH_M"
        return "H_STORE_V" if family == "read" else "H_PREFETCH_V"
    return None


def _stage_opcode_role_scale(
    original: Mapping[str, Any],
    per_chip: Mapping[str, Any],
    *,
    stage: str,
    opcode: str,
    role: str,
    require_exact: bool,
    original_cross: Mapping[str, Any] | None = None,
    per_chip_cross: Mapping[str, Any] | None = None,
) -> float:
    key = f"{stage}::{opcode}::{role}"
    if original_cross is None:
        original_cross = dict(
            original.get("by_stage_opcode_role") or {}
        )
    if per_chip_cross is None:
        per_chip_cross = dict(
            per_chip.get("by_stage_opcode_role") or {}
        )
    if key in original_cross:
        original_total = _traffic_total(dict(original_cross[key]))
        current_total = _traffic_total(dict(per_chip_cross.get(key) or {}))
        return current_total / original_total if original_total else 0.0
    if require_exact:
        raise ValueError(
            "tile-aware DMA energy requires exact stage/opcode/role traffic "
            f"for {key}"
        )
    return _stage_scale(original, per_chip, stage)


def _scaled_energy_actions(
    cost_trace: Any,
    *,
    original_traffic: Mapping[str, Any],
    per_chip_traffic: Mapping[str, Any],
    chip_count: int,
    parallel_model: str,
    multi_chip_report: Mapping[str, Any] | None = None,
    rank_index: int | None = None,
) -> dict[str, Any]:
    if isinstance(cost_trace, Mapping):
        trace = dict(cost_trace)
        raw_actions = trace.get("energy_actions") or ()
        schema = trace.get("schema_version")
        metadata = dict(trace.get("metadata") or {})
    elif hasattr(cost_trace, "energy_actions"):
        raw_actions = cost_trace.energy_actions
        schema = getattr(cost_trace, "schema_version", None)
        metadata = dict(getattr(cost_trace, "metadata", {}) or {})
    else:
        raise TypeError("cost_trace must expose compressed energy_actions")

    tile_aware = bool(
        multi_chip_report
        and multi_chip_report.get("multi_chip_model")
        in _TILE_AWARE_MULTI_CHIP_MODELS
    )
    original_cross = dict(
        original_traffic.get("by_stage_opcode_role") or {}
    )
    per_chip_cross = dict(
        per_chip_traffic.get("by_stage_opcode_role") or {}
    )
    kernel_opcode_scales: Mapping[str, Any] = {}
    kernel_scales: Mapping[str, Any] = {}
    opcode_scales: Mapping[str, Any] = {}
    stage_scales: Mapping[str, Any] = {}
    if tile_aware:
        rank_kernel_opcode = list(
            multi_chip_report.get(
                "rank_parallel_action_scales_by_kernel_opcode"
            )
            or ()
        )
        rank_kernel = list(
            multi_chip_report.get(
                "rank_parallel_action_scales_by_kernel"
            )
            or ()
        )
        kernel_opcode_scales = dict(
            rank_kernel_opcode[rank_index]
            if rank_index is not None
            else multi_chip_report.get(
                "parallel_action_scales_by_kernel_opcode"
            )
            or {}
        )
        kernel_scales = dict(
            rank_kernel[rank_index]
            if rank_index is not None
            else multi_chip_report.get("parallel_action_scales_by_kernel")
            or {}
        )
        opcode_scales = dict(
            multi_chip_report.get(
                "parallel_action_scales_by_stage_opcode"
            )
            or {}
        )
        stage_scales = dict(
            multi_chip_report.get("parallel_action_scales_by_stage") or {}
        )

    actions: list[dict[str, Any]] = []
    for raw in raw_actions:
        if isinstance(raw, Mapping):
            action = dict(raw)
        elif hasattr(raw, "to_dict"):
            action = dict(raw.to_dict())
        else:
            action = dict(vars(raw))
        component = str(action.get("component", ""))
        dma_opcode = _dma_action_opcode(action)
        if dma_opcode is not None:
            scale = _stage_opcode_role_scale(
                original_traffic,
                per_chip_traffic,
                stage=str(action.get("stage", "global")),
                opcode=dma_opcode,
                role=str(action.get("precision", "")),
                require_exact=bool(tile_aware),
                original_cross=original_cross,
                per_chip_cross=per_chip_cross,
            )
        elif tile_aware:
            stage = str(action.get("stage", "global"))
            lineage = str(
                action.get("parallel_kernel") or "__unclassified__"
            )
            source_opcode = str(
                action.get("precision") or action.get("action", "")
            )
            exact_key = f"{stage}::{lineage}::{source_opcode}"
            kernel_key = f"{stage}::{lineage}"
            if lineage != "__unclassified__":
                selected = kernel_opcode_scales.get(
                    exact_key, kernel_scales.get(kernel_key)
                )
                if selected is None:
                    raise ValueError(
                        "tile-aware power has no exact scale for tagged "
                        f"EnergyAction {exact_key}"
                    )
                scale = float(selected)
            else:
                if stage.startswith("layer/"):
                    raise ValueError(
                        "layer EnergyAction lost parallel-kernel lineage: "
                        f"{stage}/{source_opcode}"
                    )
                scale = float(
                    opcode_scales.get(
                        f"{stage}::{source_opcode}",
                        stage_scales.get(stage, 0.0),
                    )
                )
        elif (
            multi_chip_report
            and multi_chip_report.get("multi_chip_model")
            == "factorized-tp-cp-v2"
        ):
            source_opcode = str(action.get("precision") or action.get("action", ""))
            axis = classify_parallel_work_axis(
                str(action.get("stage", "global")),
                source_opcode,
            )
            scale = float(
                (multi_chip_report.get("parallel_work_axis_scales") or {})[axis]
            )
        elif parallel_model == "tp-sp":
            scale = 1.0 / chip_count
        elif component in {"matrix", "matrix_sram"}:
            scale = 1.0 / chip_count
        else:
            # TP-only conservatively replicates Vector/Scalar/control work and
            # their local SRAM accesses on every die.
            scale = 1.0
        action["count"] = float(action.get("count", 0)) * scale
        if action["count"] > 0:
            actions.append(action)
    return {
        "schema_version": schema,
        "metadata": {
            **metadata,
            "multi_chip_energy_partition": {
                "chip_count": chip_count,
                "parallel_model": parallel_model,
                "multi_chip_model": (
                    None
                    if multi_chip_report is None
                    else multi_chip_report.get("multi_chip_model")
                ),
                "rank_index": rank_index,
            },
        },
        "energy_actions": actions,
    }


def _opcode_service_windows(
    per_chip_traffic: Mapping[str, Any],
    stage_latency_ns: Mapping[str, Any],
) -> dict[str, float]:
    """Allocate each R-aware V4 stage window to opcodes by physical traffic."""

    cross = dict(per_chip_traffic.get("by_stage_opcode_role") or {})
    by_stage_opcode: dict[tuple[str, str], float] = {}
    stage_totals: dict[str, float] = {}
    for key, bucket in cross.items():
        parts = str(key).split("::")
        if len(parts) != 3:
            continue
        stage, opcode, _role = parts
        traffic = _traffic_total(dict(bucket))
        by_stage_opcode[(stage, opcode)] = (
            by_stage_opcode.get((stage, opcode), 0.0) + traffic
        )
        stage_totals[stage] = stage_totals.get(stage, 0.0) + traffic

    result: dict[str, float] = {}
    for (stage, opcode), traffic in by_stage_opcode.items():
        total = stage_totals.get(stage, 0.0)
        if total > 0:
            result[opcode] = result.get(opcode, 0.0) + (
                float(stage_latency_ns.get(stage, 0.0)) * traffic / total
            )
    if not result:
        by_opcode = _group_totals(per_chip_traffic, "by_opcode")
        total_traffic = sum(_traffic_total(bucket) for bucket in by_opcode.values())
        total_latency = sum(float(value) for value in stage_latency_ns.values())
        if total_traffic > 0:
            result = {
                opcode: total_latency * _traffic_total(bucket) / total_traffic
                for opcode, bucket in by_opcode.items()
            }
    return result


def _timing_with_traffic(
    timing_report: Mapping[str, Any],
    *,
    traffic: Mapping[str, Any],
    runtime_ns: float,
    stage_memory_latency_ns: Mapping[str, Any],
) -> dict[str, Any]:
    timing = dict(timing_report)
    totals = _traffic_summary(traffic)
    timing.update(
        {
            "roofline_latency_ns": runtime_ns,
            "hbm_read_bytes": totals["physical_read_bytes"],
            "hbm_write_bytes": totals["physical_write_bytes"],
            "hbm_payload_read_bytes": totals["payload_read_bytes"],
            "hbm_payload_write_bytes": totals["payload_write_bytes"],
            "hbm_read_requests": totals["read_requests"],
            "hbm_write_requests": totals["write_requests"],
            "hbm_traffic_breakdown": traffic,
            "hbm_opcode_latency_ns": _opcode_service_windows(
                traffic, stage_memory_latency_ns
            ),
        }
    )
    return timing


def _multiply_mapping(values: Mapping[str, Any], factor: int) -> dict[str, Any]:
    return {
        str(key): float(value) * factor
        for key, value in values.items()
    }


def _aggregate_onchip(
    per_chip: Mapping[str, Any],
    *,
    chip_count: int,
    runtime_ms: float,
) -> dict[str, Any]:
    result = dict(per_chip)
    energy_fields = (
        "logic_dynamic_energy_mj",
        "action_logic_dynamic_energy_mj",
        "clock_energy_mj",
        "ideal_clock_energy_mj",
        "ungated_clock_energy_mj",
        "sram_dynamic_energy_mj",
        "sram_background_energy_mj",
        "sram_leakage_energy_mj",
        "logic_leakage_energy_mj",
        "onchip_energy_mj",
        "ungated_onchip_energy_mj",
        "onchip_energy_p10_mj",
        "onchip_energy_p50_mj",
        "onchip_energy_p90_mj",
        "ungated_onchip_energy_p10_mj",
        "ungated_onchip_energy_p50_mj",
        "ungated_onchip_energy_p90_mj",
    )
    for field in energy_fields:
        if per_chip.get(field) is not None:
            result[field] = float(per_chip[field]) * chip_count

    for field in (
        "stage_logic_dynamic_energy_pj",
        "component_logic_dynamic_energy_pj",
        "action_logic_dynamic_energy_by_action_pj",
        "action_active_elements_by_action",
        "action_active_rows_by_action",
        "component_sram_dynamic_energy_pj",
        "sram_background_energy_by_component_pj",
        "clock_energy_by_component_pj",
        "ungated_clock_energy_by_component_pj",
        "ideal_clock_energy_by_component_pj",
        "ideal_clock_energy_by_subcomponent_pj",
    ):
        result[field] = _multiply_mapping(
            dict(per_chip.get(field) or {}), chip_count
        )

    result["logic_leakage_power_mw"] = (
        float(per_chip.get("logic_leakage_power_mw", 0.0)) * chip_count
    )
    result["sram_background_power_w"] = (
        float(per_chip.get("sram_background_power_w", 0.0)) * chip_count
    )
    result["sram_leakage_power_w"] = result["sram_background_power_w"]
    result["sram_allocated_capacity_gb"] = (
        float(per_chip.get("sram_allocated_capacity_gb", 0.0)) * chip_count
    )
    result["area_used_for_logic_leakage_um2"] = (
        float(per_chip.get("area_used_for_logic_leakage_um2", 0.0))
        * chip_count
    )
    result["onchip_average_power_w"] = result["onchip_energy_mj"] / runtime_ms
    result["ungated_onchip_average_power_w"] = (
        result["ungated_onchip_energy_mj"] / runtime_ms
    )
    for quantile in ("p10", "p50", "p90"):
        result[f"onchip_average_power_{quantile}_w"] = (
            result[f"onchip_energy_{quantile}_mj"] / runtime_ms
        )
        result[f"ungated_onchip_average_power_{quantile}_w"] = (
            result[f"ungated_onchip_energy_{quantile}_mj"] / runtime_ms
        )
    result["multi_chip_per_chip_onchip"] = dict(per_chip)
    return result


def _sum_onchip_reports(
    rank_reports: list[Mapping[str, Any]],
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    """Aggregate heterogeneous rank energy after each rank's clock cap."""

    if not rank_reports:
        raise ValueError("rank_reports must not be empty")
    result = dict(rank_reports[0])
    energy_fields = (
        "logic_dynamic_energy_mj",
        "action_logic_dynamic_energy_mj",
        "clock_energy_mj",
        "ideal_clock_energy_mj",
        "ungated_clock_energy_mj",
        "sram_dynamic_energy_mj",
        "sram_background_energy_mj",
        "sram_leakage_energy_mj",
        "logic_leakage_energy_mj",
        "onchip_energy_mj",
        "ungated_onchip_energy_mj",
        "onchip_energy_p10_mj",
        "onchip_energy_p50_mj",
        "onchip_energy_p90_mj",
        "ungated_onchip_energy_p10_mj",
        "ungated_onchip_energy_p50_mj",
        "ungated_onchip_energy_p90_mj",
    )
    for field in energy_fields:
        result[field] = math.fsum(
            float(report.get(field, 0.0)) for report in rank_reports
        )
    mapping_fields = (
        "stage_logic_dynamic_energy_pj",
        "component_logic_dynamic_energy_pj",
        "action_logic_dynamic_energy_by_action_pj",
        "action_active_elements_by_action",
        "action_active_rows_by_action",
        "component_sram_dynamic_energy_pj",
        "sram_background_energy_by_component_pj",
        "clock_energy_by_component_pj",
        "ungated_clock_energy_by_component_pj",
        "ideal_clock_energy_by_component_pj",
        "ideal_clock_energy_by_subcomponent_pj",
    )
    for field in mapping_fields:
        keys = {
            str(key)
            for report in rank_reports
            for key in dict(report.get(field) or {})
        }
        result[field] = {
            key: math.fsum(
                float(dict(report.get(field) or {}).get(key, 0.0))
                for report in rank_reports
            )
            for key in sorted(keys)
        }
    result["logic_leakage_power_mw"] = math.fsum(
        float(report.get("logic_leakage_power_mw", 0.0))
        for report in rank_reports
    )
    result["sram_background_power_w"] = math.fsum(
        float(report.get("sram_background_power_w", 0.0))
        for report in rank_reports
    )
    result["sram_leakage_power_w"] = result["sram_background_power_w"]
    result["sram_allocated_capacity_gb"] = math.fsum(
        float(report.get("sram_allocated_capacity_gb", 0.0))
        for report in rank_reports
    )
    result["area_used_for_logic_leakage_um2"] = math.fsum(
        float(report.get("area_used_for_logic_leakage_um2", 0.0))
        for report in rank_reports
    )
    result["onchip_average_power_w"] = result["onchip_energy_mj"] / runtime_ms
    result["ungated_onchip_average_power_w"] = (
        result["ungated_onchip_energy_mj"] / runtime_ms
    )
    for quantile in ("p10", "p50", "p90"):
        result[f"onchip_average_power_{quantile}_w"] = (
            result[f"onchip_energy_{quantile}_mj"] / runtime_ms
        )
        result[f"ungated_onchip_average_power_{quantile}_w"] = (
            result[f"ungated_onchip_energy_{quantile}_mj"] / runtime_ms
        )
    result["multi_chip_per_chip_onchip"] = dict(rank_reports[0])
    result["multi_chip_rank_onchip"] = [
        dict(report) for report in rank_reports
    ]
    result["multi_chip_onchip_aggregation"] = (
        "sum_rank_energy_after_per_rank_clock_cap_v2"
    )
    return result


def _interconnect_energy_mj(bits: float, pj_per_bit: float) -> float:
    return bits * pj_per_bit * 1e-9


def estimate_multi_chip_system_power(
    config: Mapping[str, Any],
    cost_trace: Any,
    timing_report: Mapping[str, Any],
    multi_chip_report: Mapping[str, Any],
    *,
    chip_count: int,
    parallel_model: str,
    external_memory_config: Mapping[str, Any] | None = None,
    logic_coefficients_path: str | Path | None = None,
    sram_energy_path: str | Path | None = None,
    sram_background_path: str | Path | None = None,
    external_memory_artifact_path: str | Path | None = None,
    interconnect_energy_artifact_path: str | Path | None = None,
    clock_gating_mode: str = "ideal_hierarchical",
) -> dict[str, Any]:
    """Estimate aggregate energy for one fixed-workload multi-chip trial."""

    if chip_count <= 0:
        raise ValueError("chip_count must be positive")
    if parallel_model not in {"tp-sp", "tp-only", "tp-cp", "dp-tp-ep"}:
        raise ValueError(f"unsupported parallel model {parallel_model!r}")
    runtime_ns = float(multi_chip_report["latency_ns"])
    if runtime_ns <= 0:
        raise ValueError("multi-chip runtime must be positive")
    runtime_ms = runtime_ns / 1e6
    artifact = _load_interconnect_artifact(interconnect_energy_artifact_path)

    memory_config = dict(config)
    if external_memory_config is not None:
        memory_config.update(external_memory_config)

    if chip_count == 1:
        result = estimate_system_power(
            config,
            cost_trace,
            timing_report,
            external_memory_config=external_memory_config,
            logic_coefficients_path=logic_coefficients_path,
            sram_energy_path=sram_energy_path,
            sram_background_path=sram_background_path,
            external_memory_artifact_path=external_memory_artifact_path,
            clock_gating_mode=clock_gating_mode,
        )
        interconnect_bits = 0.0
    else:
        original_traffic = dict(timing_report.get("hbm_traffic_breakdown") or {})
        per_chip_traffic = dict(
            multi_chip_report.get(
                "average_per_chip_hbm_traffic_breakdown"
            )
            or multi_chip_report.get("per_chip_hbm_traffic_breakdown")
            or {}
        )
        aggregate_traffic = dict(
            multi_chip_report.get("aggregate_hbm_traffic_breakdown") or {}
        )
        if not original_traffic or not per_chip_traffic or not aggregate_traffic:
            raise ValueError(
                "multi-chip energy requires original, per-chip, and aggregate "
                "HBM traffic breakdowns"
            )
        if multi_chip_report.get("multi_chip_model") in _TILE_AWARE_MULTI_CHIP_MODELS:
            rank_traffic = list(
                multi_chip_report.get("rank_hbm_traffic_breakdown") or ()
            )
            rank_stage_memory = list(
                multi_chip_report.get("rank_stage_memory_latency_ns") or ()
            )
            if (
                len(rank_traffic) != chip_count
                or len(rank_stage_memory) != chip_count
            ):
                raise ValueError(
                    "tile-aware energy requires one HBM traffic and memory "
                    "latency record per rank"
                )
            rank_onchip: list[dict[str, Any]] = []
            rank_onchip_cache: dict[str, dict[str, Any]] = {}
            rank_kernel_opcode = list(
                multi_chip_report.get(
                    "rank_parallel_action_scales_by_kernel_opcode"
                )
                or ()
            )
            rank_kernel = list(
                multi_chip_report.get(
                    "rank_parallel_action_scales_by_kernel"
                )
                or ()
            )
            for rank_index in range(chip_count):
                local_traffic = dict(rank_traffic[rank_index])
                rank_signature = json.dumps(
                    {
                        "traffic": local_traffic,
                        "stage_memory": rank_stage_memory[rank_index],
                        "kernel_opcode_scale": (
                            rank_kernel_opcode[rank_index]
                            if rank_kernel_opcode
                            else {}
                        ),
                        "kernel_scale": (
                            rank_kernel[rank_index]
                            if rank_kernel
                            else {}
                        ),
                        "opcode_scale": multi_chip_report.get(
                            "parallel_action_scales_by_stage_opcode"
                        )
                        or {},
                        "stage_scale": multi_chip_report.get(
                            "parallel_action_scales_by_stage"
                        )
                        or {},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                cached_rank = rank_onchip_cache.get(rank_signature)
                if cached_rank is not None:
                    rank_onchip.append(cached_rank)
                    continue
                rank_trace = _scaled_energy_actions(
                    cost_trace,
                    original_traffic=original_traffic,
                    per_chip_traffic=local_traffic,
                    chip_count=chip_count,
                    parallel_model=parallel_model,
                    multi_chip_report=multi_chip_report,
                    rank_index=rank_index,
                )
                rank_timing = _timing_with_traffic(
                    timing_report,
                    traffic=local_traffic,
                    runtime_ns=runtime_ns,
                    stage_memory_latency_ns=dict(
                        rank_stage_memory[rank_index]
                    ),
                )
                rank_report = estimate_onchip_power(
                    config,
                    rank_trace,
                    rank_timing,
                    logic_coefficients_path=logic_coefficients_path,
                    sram_energy_path=sram_energy_path,
                    sram_background_path=sram_background_path,
                    makespan_ns_override=runtime_ns,
                    makespan_source_override=(
                        "multi_chip_stage_roofline_rank_local_work"
                    ),
                    clock_gating_mode=clock_gating_mode,
                )
                rank_onchip_cache[rank_signature] = rank_report
                rank_onchip.append(rank_report)
            onchip = _sum_onchip_reports(
                rank_onchip,
                runtime_ms=runtime_ms,
            )
            onchip["multi_chip_unique_rank_power_workloads"] = len(
                rank_onchip_cache
            )
        else:
            per_chip_trace = _scaled_energy_actions(
                cost_trace,
                original_traffic=original_traffic,
                per_chip_traffic=per_chip_traffic,
                chip_count=chip_count,
                parallel_model=parallel_model,
                multi_chip_report=multi_chip_report,
            )
            per_chip_timing = _timing_with_traffic(
                timing_report,
                traffic=per_chip_traffic,
                runtime_ns=runtime_ns,
                stage_memory_latency_ns=multi_chip_report[
                    (
                        "average_per_chip_stage_memory_latency_ns"
                        if multi_chip_report.get(
                            "average_per_chip_stage_memory_latency_ns"
                        )
                        else "per_chip_stage_memory_latency_ns"
                    )
                ],
            )
            per_chip_onchip = estimate_onchip_power(
                config,
                per_chip_trace,
                per_chip_timing,
                logic_coefficients_path=logic_coefficients_path,
                sram_energy_path=sram_energy_path,
                sram_background_path=sram_background_path,
                makespan_ns_override=runtime_ns,
                makespan_source_override="multi_chip_stage_roofline",
                clock_gating_mode=clock_gating_mode,
            )
            onchip = _aggregate_onchip(
                per_chip_onchip,
                chip_count=chip_count,
                runtime_ms=runtime_ms,
            )
        aggregate_timing = _timing_with_traffic(
            timing_report,
            traffic=aggregate_traffic,
            runtime_ns=runtime_ns,
            stage_memory_latency_ns=multi_chip_report[
                "per_chip_stage_memory_latency_ns"
            ],
        )
        external = estimate_external_hbm_power(
            memory_config,
            aggregate_timing,
            artifact_path=external_memory_artifact_path,
            runtime_ns=runtime_ns,
        )
        result = dict(onchip)
        result.update(
            {
                "external_memory": external,
                "external_memory_model": external["external_memory_model"],
                "external_memory_calibration_status": external[
                    "external_memory_calibration_status"
                ],
                "external_memory_configuration_semantics": external[
                    "external_memory_configuration_semantics"
                ],
                "external_hbm_capacity_bytes": external[
                    "provisioned_capacity_bytes"
                ],
                "external_hbm_configured_bandwidth_gbps": external[
                    "configured_bandwidth_gbps"
                ],
                "hbm_background_energy_mj": external[
                    "hbm_background_energy_mj"
                ],
                "hbm_read_energy_mj": external["hbm_read_energy_mj"],
                "hbm_write_energy_mj": external["hbm_write_energy_mj"],
                "external_hbm_energy_mj": external[
                    "external_hbm_energy_mj"
                ],
                "external_hbm_energy_p10_mj": external[
                    "external_hbm_energy_p10_mj"
                ],
                "external_hbm_energy_p50_mj": external[
                    "external_hbm_energy_p50_mj"
                ],
                "external_hbm_energy_p90_mj": external[
                    "external_hbm_energy_p90_mj"
                ],
                "external_hbm_average_power_w": external[
                    "external_hbm_average_power_w"
                ],
                "hbm_physical_read_bytes": external["physical_read_bytes"],
                "hbm_physical_write_bytes": external["physical_write_bytes"],
                "hbm_payload_read_bytes": external["payload_read_bytes"],
                "hbm_payload_write_bytes": external["payload_write_bytes"],
                "physical_to_payload_traffic_ratio": external[
                    "physical_to_payload_traffic_ratio"
                ],
                "achieved_average_bandwidth_gbps": external[
                    "achieved_average_bandwidth_gbps"
                ],
                "bandwidth_utilization": external["bandwidth_utilization"],
                "external_hbm_energy_by_role": external[
                    "external_hbm_dynamic_energy_breakdown"
                ].get("by_role", {}),
                "external_hbm_energy_by_stage": external[
                    "external_hbm_dynamic_energy_breakdown"
                ].get("by_stage", {}),
                "external_hbm_energy_by_opcode": external[
                    "external_hbm_dynamic_energy_breakdown"
                ].get("by_opcode", {}),
            }
        )
        aggregate_interconnect_bytes = multi_chip_report.get(
            "aggregate_interconnect_bytes"
        )
        if aggregate_interconnect_bytes is None:
            aggregate_interconnect_bytes = (
                float(multi_chip_report.get("interconnect_bytes", 0.0))
                * chip_count
            )
        interconnect_bits = float(aggregate_interconnect_bytes) * 8.0

    coefficients = artifact["dynamic_energy_pj_per_bit"]
    link_energy = {
        name: _interconnect_energy_mj(interconnect_bits, float(value))
        for name, value in coefficients.items()
    }
    nominal_link = link_energy["nominal"]
    onchip_energy = {
        q: float(result[f"onchip_energy_{q}_mj"])
        for q in ("p10", "p50", "p90")
    }
    external_energy = {
        q: float(result[f"external_hbm_energy_{q}_mj"])
        for q in ("p10", "p50", "p90")
    }
    system_energy = {
        q: onchip_energy[q] + external_energy[q] + nominal_link
        for q in ("p10", "p50", "p90")
    }
    input_tokens = max(
        1,
        int(config.get("INPUT_TOKENS", 0))
        or int(config.get("SEQ_LEN", 1)) * int(config.get("BATCH_SIZE", 1)),
    )
    warnings = list(result.get("warnings", ())) + list(
        artifact.get("warnings", ())
    )
    warnings.extend(
        [
            "interconnect static/link-maintenance and switch energy are excluded",
            "FP16 KV handoff to the decode system is a shadow and is excluded from the prefill energy objective",
        ]
    )
    nominal_without_link = onchip_energy["p50"] + external_energy["p50"]
    result.update(
        {
            "power_model": "plena_multichip_system_energy_v1",
            "power_scope": (
                "onchip_logic+sram+controller+external_hbm3e_equivalent"
                "+internal_multichip_interconnect_dynamic"
            ),
            "calibration_status": (
                "mixed_rtl_activity_onchip_literature_hbm3e_and_interconnect"
            ),
            "chip_count": chip_count,
            "parallel_model": parallel_model,
            "multi_chip_model": multi_chip_report.get("multi_chip_model"),
            "dp_degree": multi_chip_report.get("dp_degree", 1),
            "tp_degree": multi_chip_report.get("tp_degree", chip_count),
            "cp_degree": multi_chip_report.get("cp_degree", 1),
            "ep_degree": multi_chip_report.get("ep_degree", 1),
            "multi_chip_energy_partition_fidelity": (
                "all_rank_action_census_with_rank_local_role_traffic"
                if multi_chip_report.get("multi_chip_model")
                in _TILE_AWARE_MULTI_CHIP_MODELS
                else "component_action_partition_with_exact_role_traffic"
                if multi_chip_report.get(
                    "hbm_traffic_partition_fidelity", ""
                ).startswith("exact")
                else "component_action_partition_with_traffic_fallback"
            ),
            "system_runtime_ns": runtime_ns,
            "system_runtime_source": "multi_chip_stage_roofline",
            "interconnect_energy_model": artifact["model"],
            "interconnect_energy_calibration_status": artifact[
                "calibration_status"
            ],
            "interconnect_energy_sources": artifact["sources"],
            "interconnect_energy_uncertainty_semantics": artifact[
                "uncertainty_semantics"
            ],
            "interconnect_per_chip_bytes": float(
                multi_chip_report.get("interconnect_bytes", 0.0)
            ),
            "interconnect_aggregate_bytes": interconnect_bits / 8.0,
            "interconnect_aggregate_bits": interconnect_bits,
            "interconnect_dynamic_energy_mj": nominal_link,
            "interconnect_dynamic_energy_optimistic_c2c_mj": link_energy[
                "optimistic_c2c"
            ],
            "interconnect_dynamic_energy_nominal_mj": nominal_link,
            "interconnect_dynamic_energy_conservative_measured_path_mj": (
                link_energy["conservative_measured_path"]
            ),
            "interconnect_nominal_pj_per_bit": float(coefficients["nominal"]),
            "system_energy_mj": system_energy["p50"],
            "system_energy_nominal_mj": system_energy["p50"],
            "system_energy_p10_mj": system_energy["p10"],
            "system_energy_p50_mj": system_energy["p50"],
            "system_energy_p90_mj": system_energy["p90"],
            "system_average_power_w": system_energy["p50"] / runtime_ms,
            "system_average_power_p10_w": system_energy["p10"] / runtime_ms,
            "system_average_power_p50_w": system_energy["p50"] / runtime_ms,
            "system_average_power_p90_w": system_energy["p90"] / runtime_ms,
            "system_energy_per_input_token_mj": (
                system_energy["p50"] / input_tokens
            ),
            "system_energy_optimistic_c2c_mj": (
                nominal_without_link + link_energy["optimistic_c2c"]
            ),
            "system_energy_conservative_measured_path_mj": (
                nominal_without_link
                + link_energy["conservative_measured_path"]
            ),
            "system_energy_sensitivity": {
                "optimistic_c2c_1p3_pj_per_bit_mj": (
                    nominal_without_link + link_energy["optimistic_c2c"]
                ),
                "nominal_proxy_8_pj_per_bit_mj": system_energy["p50"],
                "conservative_measured_path_70p9_pj_per_bit_mj": (
                    nominal_without_link
                    + link_energy["conservative_measured_path"]
                ),
            },
            "system_uncertainty_energy_mj": system_energy,
            "system_uncertainty_power_w": {
                q: value / runtime_ms for q, value in system_energy.items()
            },
            "warnings": list(dict.fromkeys(warnings)),
            "excludes": [
                "static_link_maintenance",
                "nvlink_switch",
                "package",
                "cooling",
                "board_regulator",
                "decode_kv_handoff",
                "cts",
                "asap7_macro_intrinsic_leakage_calibration",
            ],
        }
    )
    return result


__all__ = [
    "DEFAULT_INTERCONNECT_ENERGY",
    "estimate_multi_chip_system_power",
]
