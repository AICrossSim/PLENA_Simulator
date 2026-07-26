"""Compressed clocked-area occupancy for ideal hierarchical gating.

The compiler emits structural :class:`EnergyAction` records, while the shared
RTL opcode timing artifact supplies backend occupancy.  This module combines
those sources without expanding the compressed ISA schedule.  It deliberately
does not multiply by area or clock-energy density; that remains the power
model's responsibility.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from analytic_models.performance.rtl_opcode_timing import (
    ComputeFormat,
    ComputePrecisionConfig,
    DEFAULT_RTL_TIMING_CALIBRATION,
    FpFormat,
    PARAMETERIZED_TIMING_OPS,
    RtlOpcodeTimingCalibration,
    TimingHardware,
)


_HBM_ACTION_OPCODE = {
    "matrix_prefetch": "H_PREFETCH_M",
    "vector_prefetch": "H_PREFETCH_V",
    "vector_writeback": "H_STORE_V",
}


@lru_cache(maxsize=4)
def _load_timing(path: str) -> RtlOpcodeTimingCalibration:
    return RtlOpcodeTimingCalibration.load(path)


def _precision(config: Mapping[str, Any]) -> ComputePrecisionConfig:
    scale_bits = int(config.get("MX_SCALE_WIDTH", 8))
    block = int(config.get("MX_SCALE_BLOCK_SIZE", config.get("BLOCK_DIM", 64)))
    fp = FpFormat.parse(str(config.get("FP_SETTING", "FP_E5M6")))
    return ComputePrecisionConfig(
        weight=ComputeFormat.parse(
            config.get("WEIGHT_WIDTH", "MXINT4"),
            default_scale_bits=scale_bits,
            default_block=block,
        ),
        activation=ComputeFormat.parse(
            config.get("ACT_WIDTH", "MXINT4"),
            default_scale_bits=scale_bits,
            default_block=block,
        ),
        kv=ComputeFormat.parse(
            config.get("KV_WIDTH", "MXINT4"),
            default_scale_bits=scale_bits,
            default_block=block,
        ),
        matrix_internal_fp=fp,
        vector_internal_fp=fp,
        scalar_fp=fp,
        integer_bits=int(config.get("INT_DATA_WIDTH", 32)),
    )


def _operands(action: Mapping[str, Any]) -> tuple[str, ...]:
    variant = str(action.get("variant", ""))
    if not variant or variant == "aggregate":
        return ()
    return tuple(part.strip() for part in variant.split(","))


def _timing_cache_key(
    opcode: str,
    operands: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Keep only operands that can change the shared timing estimate.

    Most instruction operands are register numbers or addresses.  Caching on
    those values caused the same fixed-latency opcode to be evaluated hundreds
    of times in a large compiler trace.  The rtl-v3 timing model is
    operand-sensitive only for segment reductions, where the final operand is
    ``segment_log2``.
    """

    if opcode in PARAMETERIZED_TIMING_OPS:
        return opcode, operands[-1:] if operands else ()
    return opcode, ()


def _vector_activity(
    action: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, int, str]:
    vlen = int(config.get("VLEN", config["MLEN"]))
    hlen = int(config.get("HLEN", 1))
    fidelity = str(action.get("activity_fidelity", "unannotated"))
    family = str(action.get("action", ""))
    if fidelity == "full_width":
        return vlen, vlen, fidelity
    if fidelity == "exact_segment_mask":
        active = int(action.get("segment_count", 0)) * hlen
        if not 0 < active <= vlen:
            return 0, vlen, "clock_work_unavailable"
        return active, vlen, fidelity
    if fidelity == "exact_single_segment":
        segment_log2 = int(action.get("segment_log2", -1))
        active = 1 << segment_log2 if segment_log2 >= 0 else 0
        if not 0 < active <= vlen:
            return 0, vlen, "clock_work_unavailable"
        return active, vlen, fidelity
    if fidelity == "exact_compact_lanes":
        active = int(action.get("segment_count", 0))
        if not 0 < active <= 16:
            return 0, 16, "clock_work_unavailable"
        return active, 16, fidelity
    if family.endswith("_segments"):
        return vlen, vlen, "structural_full_width"
    return 0, vlen, "clock_work_unavailable"


def _append(
    records: dict[tuple[Any, ...], dict[str, Any]],
    *,
    stage: str,
    component: str,
    subcomponent: str,
    cycles: float,
    component_active_cycles: float,
    opcode: str,
    active_instances: int,
    total_instances: int,
    fidelity: str,
) -> None:
    if cycles <= 0 or total_instances <= 0:
        return
    key = (
        stage,
        component,
        subcomponent,
        opcode,
        int(active_instances),
        int(total_instances),
        fidelity,
    )
    target = records.get(key)
    if target is None:
        records[key] = {
            "stage": stage,
            "component": component,
            "subcomponent": subcomponent,
            "equivalent_full_area_cycles": float(cycles),
            "component_active_cycles": float(component_active_cycles),
            "source_opcode": opcode,
            "active_instances": int(active_instances),
            "total_instances": int(total_instances),
            "fidelity": fidelity,
        }
        return
    target["equivalent_full_area_cycles"] += float(cycles)
    target["component_active_cycles"] += float(component_active_cycles)


def _logic_clock_work(
    actions: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    timing_path: str,
    compute_timing_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration = _load_timing(timing_path)
    hardware = TimingHardware(
        mlen=int(config["MLEN"]),
        blen=int(config["BLEN"]),
        vlen=int(config.get("VLEN", config["MLEN"])),
        hlen=int(config.get("HLEN", 1)),
        broadcast_amount=int(config.get("BROADCAST_AMOUNT", 1)),
    )
    precision = _precision(config)
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    unavailable: list[dict[str, Any]] = []
    component_sources: dict[tuple[Any, ...], tuple[float, str]] = {}
    mlen = hardware.mlen
    blen = hardware.blen
    splits = max(1, mlen // blen)
    reduce_nodes = blen * blen * max(splits - 1, 0)
    estimate_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    for action in actions:
        component = str(action.get("component", ""))
        if component not in {"matrix", "vector", "scalar", "control", "agu"}:
            continue
        opcode = str(action.get("precision", ""))
        count = float(action.get("count", 0))
        if count <= 0:
            continue
        stage = str(action.get("stage", "global"))
        family = str(action.get("action", ""))
        variant = str(action.get("variant", ""))
        operands = _operands(action) if opcode in PARAMETERIZED_TIMING_OPS else ()
        estimate_key = _timing_cache_key(opcode, operands)
        if compute_timing_mode == "ideal-ii1" and component in {
            "vector",
            "scalar",
            "control",
        }:
            estimate = None
            occupancy = count
            timing_fidelity = "architectural_ideal_ii1"
        elif component == "agu":
            occupancy = count
            timing_fidelity = "structural_loop_agu_v1"
            if family == "agu_stream_step":
                active_instances = max(1, int(action.get("active_lanes", 0)))
                total_instances = max(
                    active_instances, int(action.get("total_lanes", 6))
                )
                fraction = active_instances / total_instances
                subcomponent = "stride_adders"
            elif family == "agu_offset_read":
                active_instances = max(1, int(action.get("active_lanes", 1)))
                total_instances = max(
                    active_instances, int(action.get("total_lanes", 2))
                )
                fraction = active_instances / total_instances
                subcomponent = "offset_read_path"
            elif family == "agu_config":
                active_instances = total_instances = 1
                fraction = 1.0
                subcomponent = "descriptor_storage"
            else:
                active_instances = total_instances = 1
                fraction = 1.0
                subcomponent = "component_control"
            _append(
                records,
                stage=stage,
                component=component,
                subcomponent=subcomponent,
                cycles=occupancy * fraction,
                component_active_cycles=occupancy,
                opcode=opcode,
                active_instances=active_instances,
                total_instances=total_instances,
                fidelity=timing_fidelity,
            )
            continue
        else:
            if estimate_key not in estimate_cache:
                estimate_cache[estimate_key] = calibration.estimate(
                    opcode, hardware, precision, operands
                )
            estimate = estimate_cache[estimate_key]
            if estimate is None:
                unavailable.append(
                    {
                        "stage": action.get("stage", "global"),
                        "component": component,
                        "source_opcode": opcode,
                        "count": count,
                        "reason": "timing_unavailable",
                    }
                )
                continue
            occupancy = count * int(estimate.resource_cycles)
            timing_fidelity = str(estimate.calibration_status)
        # Keep the raw variant in the source key so several structural actions
        # emitted for one instruction collapse together, while independent
        # instructions with different operands remain distinct.
        source_key = (stage, component, opcode, variant, count)
        component_sources[source_key] = (occupancy, timing_fidelity)

        if component == "matrix":
            if family in {"array_compute", "matrix_vector_compute"}:
                for subcomponent in ("array_stack", "io_pipeline"):
                    _append(
                        records,
                        stage=stage,
                        component=component,
                        subcomponent=subcomponent,
                        cycles=occupancy,
                        component_active_cycles=0,
                        opcode=opcode,
                        active_instances=mlen * blen,
                        total_instances=mlen * blen,
                        fidelity=timing_fidelity,
                    )
            elif family == "cross_k_reduce" and reduce_nodes:
                for subcomponent in ("reduce_tree", "output_accumulator"):
                    _append(
                        records,
                        stage=stage,
                        component=component,
                        subcomponent=subcomponent,
                        cycles=occupancy,
                        component_active_cycles=0,
                        opcode=opcode,
                        active_instances=reduce_nodes,
                        total_instances=reduce_nodes,
                        fidelity=timing_fidelity,
                    )
            elif family == "output_conversion":
                for subcomponent in ("output_conversion", "result_buffer"):
                    _append(
                        records,
                        stage=stage,
                        component=component,
                        subcomponent=subcomponent,
                        cycles=occupancy,
                        component_active_cycles=0,
                        opcode=opcode,
                        active_instances=blen * blen,
                        total_instances=blen * blen,
                        fidelity=timing_fidelity,
                    )
        elif component == "vector":
            active_lanes, total_lanes, activity_fidelity = _vector_activity(
                action, config
            )
            if activity_fidelity == "clock_work_unavailable":
                unavailable.append(
                    {
                        "stage": stage,
                        "component": component,
                        "source_opcode": opcode,
                        "count": count,
                        "variant": action.get("variant"),
                        "reason": "masked_vector_active_lanes_unavailable",
                    }
                )
                continue
            if family.startswith("compact_stats_"):
                fraction = active_lanes / total_lanes
                active_instances, total_instances = active_lanes, total_lanes
                datapath = "compact_stats_simd"
            elif family.startswith("reduction"):
                if family.endswith("_segments"):
                    segment_width = 1 << max(0, int(action.get("segment_log2", 0)))
                    active_nodes = max(1, total_lanes - total_lanes // segment_width)
                else:
                    active_segments = max(1, active_lanes // max(1, hardware.hlen))
                    active_nodes = max(1, active_lanes - active_segments)
                total_nodes = max(1, total_lanes - 1)
                fraction = min(1.0, active_nodes / total_nodes)
                active_instances, total_instances = active_nodes, total_nodes
                datapath = "reduction_tree"
            else:
                fraction = active_lanes / total_lanes
                active_instances, total_instances = active_lanes, total_lanes
                datapath = "lane_datapath"
            fidelity = f"{timing_fidelity}+{activity_fidelity}"
            for subcomponent in (datapath, "buffers"):
                subcomponent_fraction = (
                    min(1.0, active_lanes / hardware.vlen)
                    if family.startswith("compact_stats_")
                    and subcomponent == "buffers"
                    else fraction
                )
                _append(
                    records,
                    stage=stage,
                    component=component,
                    subcomponent=subcomponent,
                    cycles=occupancy * subcomponent_fraction,
                    component_active_cycles=0,
                    opcode=opcode,
                    active_instances=active_instances,
                    total_instances=total_instances,
                    fidelity=fidelity,
                )
            if (
                family.endswith("_segments")
                or "vseg" in family
                or opcode in {"S_LD_VLANE_FP", "S_ST_VLANE_FP"}
            ):
                _append(
                    records,
                    stage=stage,
                    component=component,
                    subcomponent="segment_parallel_delta",
                    cycles=occupancy * fraction,
                    component_active_cycles=0,
                    opcode=opcode,
                    active_instances=active_instances,
                    total_instances=total_instances,
                    fidelity=fidelity,
                )
        elif component == "scalar":
            if family.startswith("integer"):
                subcomponents = ("int_datapath",)
            elif family in {"vector_lane_load", "vector_lane_store", "register_or_sram_access"}:
                subcomponents = ("lane_access",)
            else:
                subcomponents = ("fp_datapath",)
            for subcomponent in subcomponents:
                _append(
                    records,
                    stage=stage,
                    component=component,
                    subcomponent=subcomponent,
                    cycles=occupancy,
                    component_active_cycles=0,
                    opcode=opcode,
                    active_instances=1,
                    total_instances=1,
                    fidelity=timing_fidelity,
                )
            _append(
                records,
                stage=stage,
                component=component,
                subcomponent="pipeline_delta",
                cycles=occupancy,
                component_active_cycles=0,
                opcode=opcode,
                active_instances=1,
                total_instances=1,
                fidelity=timing_fidelity,
            )
        else:
            _append(
                records,
                stage=stage,
                component=component,
                subcomponent="frontend",
                cycles=occupancy,
                component_active_cycles=0,
                opcode=opcode,
                active_instances=1,
                total_instances=1,
                fidelity=timing_fidelity,
            )

    # Fixed per-component clock cost is charged once per source instruction,
    # not once for every subcomponent to which the instruction maps.
    for (stage, component, opcode, _, _), (
        occupancy,
        fidelity,
    ) in component_sources.items():
        _append(
            records,
            stage=stage,
            component=component,
            subcomponent="component_control",
            cycles=occupancy,
            component_active_cycles=occupancy,
            opcode=opcode,
            active_instances=1,
            total_instances=1,
            fidelity=fidelity,
        )
    return list(records.values()), unavailable, calibration.metadata()


def _hbm_clock_work(
    actions: Iterable[Mapping[str, Any]],
    timing: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clock_period_ps = int(config.get("CLOCK_PERIOD_PS", 1000))
    opcode_latency = {
        str(opcode): float(value)
        for opcode, value in dict(timing.get("hbm_opcode_latency_ns", {})).items()
    }
    selected = [
        action
        for action in actions
        if str(action.get("component")) == "hbm_controller"
        and str(action.get("action")) in _HBM_ACTION_OPCODE
        and float(action.get("count", 0)) > 0
    ]
    count_by_opcode: Counter[str] = Counter()
    for action in selected:
        count_by_opcode[_HBM_ACTION_OPCODE[str(action["action"])]] += float(
            action["count"]
        )
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    unavailable: list[dict[str, Any]] = []
    for action in selected:
        opcode = _HBM_ACTION_OPCODE[str(action["action"])]
        total_count = count_by_opcode[opcode]
        if total_count <= 0:
            continue
        if opcode not in opcode_latency or opcode_latency[opcode] <= 0:
            unavailable.append(
                {
                    "stage": action.get("stage", "global"),
                    "component": "hbm_controller",
                    "source_opcode": opcode,
                    "count": float(action["count"]),
                    "reason": "v4_dma_service_window_unavailable",
                }
            )
            continue
        service_cycles = (
            opcode_latency[opcode]
            * 1000.0
            / clock_period_ps
            * float(action["count"])
            / total_count
        )
        family = str(action["action"])
        subcomponents = (
            ("matrix_path", "scale_path", "address_control", "prefetch_writeback")
            if family == "matrix_prefetch"
            else ("vector_path", "scale_path", "address_control", "prefetch_writeback")
        )
        for subcomponent in subcomponents:
            _append(
                records,
                stage=str(action.get("stage", "global")),
                component="hbm_controller",
                subcomponent=subcomponent,
                cycles=service_cycles,
                component_active_cycles=0,
                opcode=opcode,
                active_instances=max(1, int(action.get("active_lanes", 0))),
                total_instances=max(1, int(action.get("total_lanes", 0))),
                fidelity="production_dma_v4_service_window",
            )
        _append(
            records,
            stage=str(action.get("stage", "global")),
            component="hbm_controller",
            subcomponent="fixed_control",
            cycles=service_cycles,
            component_active_cycles=math.ceil(service_cycles),
            opcode=opcode,
            active_instances=1,
            total_instances=1,
            fidelity="production_dma_v4_service_window",
        )
    return list(records.values()), unavailable


def build_clock_work(
    actions: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    timing_report: Mapping[str, Any],
    *,
    rtl_timing_path: str | None = None,
) -> dict[str, Any]:
    """Build compact hierarchical occupancy and fail closed on mask ambiguity."""

    # Callers already materialize EnergyAction records as mappings.  Avoid a
    # second deep-ish dictionary copy on every power evaluation.
    materialized = list(actions)
    selected_timing = str(rtl_timing_path or DEFAULT_RTL_TIMING_CALIBRATION)
    compute_timing_mode = str(
        timing_report.get("compute_timing_mode", "rtl-v1")
    )
    if compute_timing_mode not in {"ideal-ii1", "rtl-v1", "legacy"}:
        raise ValueError(
            f"unsupported compute timing mode for ClockWork: {compute_timing_mode!r}"
        )
    records, unavailable, timing_metadata = _logic_clock_work(
        materialized,
        config,
        timing_path=selected_timing,
        compute_timing_mode=compute_timing_mode,
    )
    timing_metadata = dict(timing_metadata)
    timing_metadata.update(
        {
            "compute_timing_mode": compute_timing_mode,
            "compute_hazards_included": compute_timing_mode == "rtl-v1",
            "compute_timing_status": (
                "architectural_ideal_assumption"
                if compute_timing_mode == "ideal-ii1"
                else "rtl_calibrated"
                if compute_timing_mode == "rtl-v1"
                else "legacy"
            ),
        }
    )
    hbm_records, hbm_unavailable = _hbm_clock_work(
        materialized, timing_report, config
    )
    records.extend(hbm_records)
    unavailable.extend(hbm_unavailable)
    records.sort(
        key=lambda record: (
            record["stage"],
            record["component"],
            record["subcomponent"],
            record["source_opcode"],
            record["active_instances"],
            record["total_instances"],
            record["fidelity"],
        )
    )
    source_counts: Counter[str] = Counter()
    source_opcode_variants: dict[tuple[str, str, str], int] = {}
    for action in materialized:
        component = str(action.get("component", ""))
        if component.endswith("_sram"):
            continue
        source_counts[component] += float(action.get("count", 0))
        opcode = (
            _HBM_ACTION_OPCODE.get(str(action.get("action")))
            if component == "hbm_controller"
            else str(action.get("precision", ""))
        )
        if opcode:
            variant_key = (
                str(action.get("stage", "global")),
                opcode,
                str(action.get("variant", "aggregate")),
            )
            source_opcode_variants[variant_key] = max(
                source_opcode_variants.get(variant_key, 0),
                float(action.get("count", 0)),
            )
    source_opcode_counts: Counter[str] = Counter()
    for (_, opcode, _), count in source_opcode_variants.items():
        source_opcode_counts[opcode] += count
    by_subcomponent: Counter[str] = Counter()
    for record in records:
        by_subcomponent[
            f"{record['component']}.{record['subcomponent']}"
        ] += float(record["equivalent_full_area_cycles"])
    return {
        "schema": "compressed_clock_work_v1",
        "records": records,
        "status": "complete" if not unavailable else "clock_work_unavailable",
        "unavailable": unavailable,
        "source_action_counts": dict(source_counts),
        "source_opcode_counts": dict(source_opcode_counts),
        "equivalent_full_area_cycles_by_subcomponent": dict(by_subcomponent),
        "timing_artifact": timing_metadata,
        "fidelity": (
            (
                "ideal_ii1_matrix_structural_plus_exact_compiler_activity_and_v4_dma"
                if compute_timing_mode == "ideal-ii1"
                else "rtl_timing_plus_exact_compiler_activity_and_v4_dma"
            )
            if not unavailable
            else "partial"
        ),
        "compute_timing_mode": compute_timing_mode,
        "compute_timing_status": (
            "architectural_ideal_assumption"
            if compute_timing_mode == "ideal-ii1"
            else "rtl_calibrated"
            if compute_timing_mode == "rtl-v1"
            else "legacy"
        ),
        "compute_hazards_included": compute_timing_mode == "rtl-v1",
    }


__all__ = ["build_clock_work"]
