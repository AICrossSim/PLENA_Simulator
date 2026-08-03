"""Compose compiler actions, clock bounds, leakage, SRAM, and external HBM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from compiler.aten.program_sink import CostTrace

from analytic_models.latency.compute import resource_for_opcode
from analytic_models.latency.schemas import LatencyReport

from .energy import estimate_action_energy
from .external_memory import ExternalHbmEnergy, estimate_external_hbm_energy
from .schemas import ComponentPhysicalProperties, PowerReport


CLOCK_GATING_MODES = ("ideal-hierarchical", "ungated")


def _load_coefficients(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    with Path(value).open() as handle:
        return json.load(handle)


def _sram_dynamic_energy(action_report, properties: ComponentPhysicalProperties) -> tuple[float, list[str]]:
    total = 0.0
    missing: set[str] = set()
    for action in action_report.actions:
        if not action.component.endswith("_sram"):
            continue
        table = properties.sram_access_energy_pj.get(action.component)
        if table is None or action.action not in table:
            missing.add(f"{action.component}.{action.action}")
            continue
        total += action.count * float(table[action.action])
    warnings = []
    if missing:
        warnings.append("SRAM dynamic energy unavailable for: " + ", ".join(sorted(missing)))
    return total, warnings


def _compute_activity(
    trace: CostTrace,
    latency: LatencyReport,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return resource busy time and active-area fractions from exact entries."""

    active_fraction_by_key: dict[tuple[str, str], float] = {}
    for item in trace.instructions:
        if item.opcode.startswith("H_"):
            continue
        resource = resource_for_opcode(item.opcode)
        if resource in {"matrix", "scalar", "control"}:
            fraction = 1.0
        else:
            active = item.active or {}
            lanes = int(active.get("lanes") or 0)
            total_lanes = int(active.get("total_lanes") or 0)
            # Main Vector ISA physically clocks a full-width operation if the
            # lowering does not preserve an explicit lane mask.
            fraction = lanes / total_lanes if lanes and total_lanes else 1.0
        key = (item.stage, item.opcode)
        active_fraction_by_key[key] = max(active_fraction_by_key.get(key, 0.0), fraction)

    busy: dict[str, int] = defaultdict(int)
    weighted: dict[str, float] = defaultdict(float)
    for entry in latency.compute.entries:
        resource = entry.resource
        busy[resource] += entry.total_picos
        weighted[resource] += entry.total_picos * active_fraction_by_key.get((entry.stage, entry.opcode), 1.0)
    busy["hbm_controller"] = latency.memory.total_picos
    weighted["hbm_controller"] = float(latency.memory.total_picos)
    fractions = {
        component: (0.0 if busy_picos == 0 else min(1.0, weighted[component] / busy_picos))
        for component, busy_picos in busy.items()
    }
    return dict(busy), fractions


def _clock_energy(
    trace: CostTrace,
    latency: LatencyReport,
    properties: ComponentPhysicalProperties,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    runtime = latency.total_picos
    period = properties.hardware.clock_period_picos
    runtime_cycles = runtime / period
    busy_picos, active_area_fractions = _compute_activity(trace, latency)
    ideal: dict[str, float] = {}
    ungated: dict[str, float] = {}
    active_time_fractions: dict[str, float] = {}
    for component, physical in properties.components.items():
        ungated[component] = (
            physical.logic_area_um2
            * physical.clock_density_pj_per_cycle_um2
            * runtime_cycles
            + physical.fixed_clock_pj_per_active_cycle * runtime_cycles
        )
        component_busy = min(runtime, busy_picos.get(component, 0))
        active_cycles = component_busy / period
        active_area = active_area_fractions.get(component, 0.0)
        ideal[component] = (
            physical.logic_area_um2
            * active_area
            * physical.clock_density_pj_per_cycle_um2
            * active_cycles
            + physical.fixed_clock_pj_per_active_cycle * active_cycles
        )
        ideal[component] = min(ideal[component], ungated[component])
        active_time_fractions[component] = 0.0 if runtime == 0 else component_busy / runtime
    return ideal, ungated, active_time_fractions


def _empty_external() -> ExternalHbmEnergy:
    return ExternalHbmEnergy(0.0, 0.0, 0.0, 0.0, 0.0, {"status": "not_configured"})


def estimate_power(
    trace: CostTrace,
    latency_report: LatencyReport,
    physical_properties: ComponentPhysicalProperties | Mapping[str, Any] | str | Path,
    clock_gating_mode: str = "ideal-hierarchical",
    *,
    logic_coefficients: Mapping[str, Any] | str | Path,
) -> PowerReport:
    """Estimate single-chip PLENA energy for the latency-report time window.

    The selected average-power result requires versioned physical properties.
    Dynamic-only users can call :func:`estimate_action_energy` without those
    properties.
    """

    if clock_gating_mode not in CLOCK_GATING_MODES:
        raise ValueError(f"clock_gating_mode must be one of {CLOCK_GATING_MODES}")
    if latency_report.total_picos <= 0:
        raise ValueError("power requires a positive latency makespan")
    properties = ComponentPhysicalProperties.from_mapping(physical_properties)
    coefficients = _load_coefficients(logic_coefficients)
    expected_period = int(coefficients.get("corner", {}).get("clock_period_ps", properties.hardware.clock_period_picos))
    if properties.hardware.clock_period_picos != expected_period:
        raise ValueError(
            f"power coefficients require {expected_period} ps, got "
            f"{properties.hardware.clock_period_picos} ps"
        )
    required_components = {"matrix", "vector", "scalar", "control", "hbm_controller"}
    missing_components = required_components - properties.components.keys()
    if missing_components:
        raise ValueError("physical properties missing components: " + ", ".join(sorted(missing_components)))

    action = estimate_action_energy(trace, properties.hardware, coefficients)
    sram_pj, sram_warnings = _sram_dynamic_energy(action, properties)
    if sram_warnings:
        raise ValueError("; ".join(sram_warnings))
    ideal_by_component, ungated_by_component, active_fraction = _clock_energy(
        trace, latency_report, properties
    )
    ideal_clock = sum(ideal_by_component.values())
    ungated_clock = sum(ungated_by_component.values())
    selected_by_component = ideal_by_component if clock_gating_mode == "ideal-hierarchical" else ungated_by_component
    selected_clock = sum(selected_by_component.values())
    runtime_ns = latency_report.total_picos / 1_000
    leakage_pj = sum(item.logic_leakage_mw for item in properties.components.values()) * runtime_ns
    external = (
        _empty_external()
        if properties.external_memory is None
        else estimate_external_hbm_energy(
            latency_report.memory,
            latency_report.total_picos,
            properties.external_memory,
        )
    )
    onchip = action.nominal_energy_pj + sram_pj + selected_clock + leakage_pj
    external_nominal = external.nominal_pj
    system = onchip + external_nominal
    common_without_logic_or_background = sram_pj + selected_clock + leakage_pj + external.read_pj + external.write_pj
    system_low = action.low_energy_pj + common_without_logic_or_background + external.background_low_pj
    system_high = action.high_energy_pj + common_without_logic_or_background + external.background_high_pj
    if system_low > system or system > system_high:
        raise ValueError("power activity/background envelope is not ordered")
    watts_per_pj = 1.0 / latency_report.total_picos
    warnings = list(action.warnings)
    warnings.extend(latency_report.warnings)
    warnings.extend(
        [
            "ideal hierarchical clock gating is an architectural assumption; current RTL clock gating is not claimed",
            "SRAM leakage, CTS, routed parasitics, package, cooling, and board regulation are excluded",
        ]
    )
    if properties.external_memory is None:
        warnings.append("external HBM energy is not configured")
    return PowerReport(
        runtime_picos=latency_report.total_picos,
        logic_dynamic_energy_pj=action.nominal_energy_pj,
        sram_dynamic_energy_pj=sram_pj,
        selected_clock_energy_pj=selected_clock,
        ideal_clock_energy_pj=ideal_clock,
        ungated_clock_energy_pj=ungated_clock,
        logic_leakage_energy_pj=leakage_pj,
        external_hbm_background_energy_pj=external.background_nominal_pj,
        external_hbm_read_energy_pj=external.read_pj,
        external_hbm_write_energy_pj=external.write_pj,
        onchip_energy_pj=onchip,
        external_hbm_energy_pj=external_nominal,
        system_energy_pj=system,
        system_energy_low_pj=system_low,
        system_energy_high_pj=system_high,
        onchip_average_power_w=onchip * watts_per_pj,
        external_hbm_average_power_w=external_nominal * watts_per_pj,
        system_average_power_w=system * watts_per_pj,
        clock_gating_mode=clock_gating_mode,
        component_dynamic_energy_pj=action.by_component_pj,
        stage_dynamic_energy_pj=action.by_stage_pj,
        component_clock_energy_pj=dict(selected_by_component),
        component_clock_active_fraction=active_fraction,
        provenance={
            "model": "compiler-action-system-power-v1",
            "physical_properties": properties.calibration_id,
            "logic_calibration": coefficients.get("calibration_status"),
            "clock_gating_status": (
                "architectural_ideal_assumption"
                if clock_gating_mode == "ideal-hierarchical"
                else "ungated_upper_bound"
            ),
            "rtl_clock_gating_implemented": False,
            "gating_overhead_included": False,
            "external_hbm": external.provenance,
            "trace_isa_hash": trace.isa_hash,
            "latency_overlap_policy": latency_report.overlap_policy,
        },
        exclusions=(
            "multi_chip",
            "nvlink",
            "package",
            "cooling",
            "board_regulator",
            "cts",
            "sram_leakage",
        ),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["CLOCK_GATING_MODES", "estimate_power"]
