"""Calibrated dynamic-energy evaluation over structural actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from typing import Any

from compiler.aten.program_sink import CostTrace

from .actions import build_energy_actions
from .schemas import ActionEnergyReport, ActionHardwareConfig, EnergyAction


def _format_coefficient(table: Mapping[str, float], fp_format: str, width: int) -> float:
    if fp_format in table:
        return float(table[fp_format])
    if "default" in table:
        return float(table["default"])
    anchors: list[tuple[int, float]] = []
    for name, value in table.items():
        try:
            exponent, mantissa = name.removeprefix("FP_E").split("M", 1)
            anchors.append((1 + int(exponent) + int(mantissa), float(value)))
        except (AttributeError, ValueError):
            continue
    if not anchors:
        raise ValueError(f"no coefficient for {fp_format}")
    anchor_width, anchor = min(anchors, key=lambda item: abs(item[0] - width))
    return anchor * width / anchor_width


def _activity_ratio(coefficients: Mapping[str, Any], action: EnergyAction, bound: str) -> float:
    if bound == "nominal":
        return 1.0
    envelope = coefficients.get("activity_envelope", {})
    key = f"{action.component}.{action.action}"
    if action.component == "matrix" and action.action in {"array_compute", "matrix_vector_compute", "cross_k_reduce"}:
        key = f"{key}.{coefficients['_hardware'].matrix_mode}"
    return float(envelope.get(key, envelope.get(f"{action.component}.{action.action}", {})).get(bound, 1.0))


def _nominal_energy(action: EnergyAction, hardware: ActionHardwareConfig, coefficients: Mapping[str, Any]) -> float:
    dynamic = coefficients["dynamic_nominal_pj"]
    count = action.count
    if action.component == "matrix":
        table = dynamic["matrix"][hardware.matrix_mode]
        if action.action in {"array_compute", "matrix_vector_compute"}:
            leaf = table["pe_cycle"]
            pe = (
                float(leaf["base"])
                + float(leaf["bit_product"]) * hardware.matrix_t_bits * hardware.matrix_l_bits
                + float(leaf["width_sum"]) * (hardware.matrix_t_bits + hardware.matrix_l_bits)
            )
            split_count = max(1, hardware.mlen // hardware.blen)
            pe_cycles = hardware.blen**2 if action.action == "matrix_vector_compute" else hardware.blen**3
            per_slice = float(leaf.get("slice_fixed", 0.0)) + hardware.blen * float(leaf.get("feed_cycle", 0.0)) + pe_cycles * pe
            return count * split_count * per_slice
        if action.action == "cross_k_reduce":
            nodes = hardware.blen**2 * max(hardware.mlen // hardware.blen - 1, 0)
            accumulator = (
                hardware.fp_width
                if hardware.matrix_mode == "mxfp"
                else hardware.matrix_t_bits
                + hardware.matrix_l_bits
                + math.ceil(math.log2(max(1, hardware.blen)))
            )
            return count * nodes * accumulator * float(table["reduce_node_bit"])
        if action.action == "output_conversion":
            return count * hardware.blen**2 * hardware.fp_width * float(table["output_bit"])
    if action.component == "vector":
        family = dynamic["vector"][action.action]
        lanes = action.active_instances or hardware.vlen
        if action.action.startswith("reduction_"):
            if action.action.endswith("_full"):
                scale = max(1, (lanes - 1) * int(math.log2(max(2, hardware.vlen))))
            else:
                scale = hardware.vlen
        else:
            scale = lanes
        return count * scale * _format_coefficient(family, hardware.fp_format, hardware.fp_width)
    if action.component == "scalar":
        family = dynamic["scalar"][action.action]
        if action.action.startswith("integer_"):
            return count * float(family.get(str(hardware.int_width), family["default"]))
        return count * _format_coefficient(family, hardware.fp_format, hardware.fp_width)
    if action.component == "control":
        return count * float(dynamic["control"]["frontend_issue"])
    if action.component == "hbm_controller":
        per_lane = float(dynamic["hbm_controller"].get(action.action, dynamic["hbm_controller"]["default"]))
        return count * max(1, action.active_instances) * per_lane
    if action.component.endswith("_sram"):
        return 0.0
    raise ValueError(f"unsupported power component/action {action.component}.{action.action}")


def estimate_action_energy(
    trace: CostTrace,
    hardware_config: ActionHardwareConfig | Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> ActionEnergyReport:
    """Evaluate calibrated non-clock logic energy for a compiler trace."""

    hardware = ActionHardwareConfig.from_mapping(hardware_config)
    actions = build_energy_actions(trace, hardware)
    enriched = dict(coefficients)
    enriched["_hardware"] = hardware
    totals = {"low": 0.0, "nominal": 0.0, "high": 0.0}
    by_component: dict[str, float] = defaultdict(float)
    by_stage: dict[str, float] = defaultdict(float)
    logic_count = 0
    active_count = 0
    sram_count = 0
    explicit_sram_count = 0
    for action in actions:
        if action.component.endswith("_sram"):
            sram_count += action.count
            if action.fidelity in {"compiler-sram-descriptor", "compiler-dma-geometry"}:
                explicit_sram_count += action.count
            continue
        nominal = _nominal_energy(action, hardware, enriched)
        low = nominal * _activity_ratio(enriched, action, "low")
        high = nominal * _activity_ratio(enriched, action, "high")
        lo, hi = sorted((low, high))
        totals["low"] += lo
        totals["nominal"] += nominal
        totals["high"] += hi
        by_component[action.component] += nominal
        by_stage[action.stage] += nominal
        logic_count += action.count
        if action.fidelity != "physical-full-width-from-main-isa":
            active_count += action.count
    warnings: list[str] = []
    active_coverage = 1.0 if logic_count == 0 else active_count / logic_count
    if active_coverage < 1.0:
        warnings.append(
            "some main lowering instructions lack logical active-shape metadata; "
            "their physical full-width ISA activity is charged"
        )
    sram_coverage = 1.0 if sram_count == 0 else explicit_sram_count / sram_count
    if sram_coverage < 1.0:
        warnings.append(
            "SRAM access descriptors are incomplete; main ISA-implied access counts are reported separately"
        )
    return ActionEnergyReport(
        actions=actions,
        nominal_energy_pj=totals["nominal"],
        low_energy_pj=totals["low"],
        high_energy_pj=totals["high"],
        by_component_pj=dict(sorted(by_component.items())),
        by_stage_pj=dict(sorted(by_stage.items())),
        opcode_coverage=1.0,
        active_shape_coverage=active_coverage,
        sram_descriptor_coverage=sram_coverage,
        provenance={
            "model": coefficients.get("model"),
            "calibration_status": coefficients.get("calibration_status"),
            "trace_schema": trace.schema_version,
            "trace_isa_hash": trace.isa_hash,
            "compiler_hash": trace.compiler_hash,
            "activity_semantics": "qwen-like nominal with empirical low/random envelope",
        },
        warnings=tuple(warnings),
    )


__all__ = ["estimate_action_energy"]
