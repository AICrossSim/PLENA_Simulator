"""Canonical model combinations used by formal DSE and RTL validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSEModelProfile:
    """A named, internally compatible simulator/compiler model stack."""

    name: str
    compiler_schedule_profile: str
    compute_timing: str
    hbm_model: str
    cost_trace_granularity: str
    multi_chip_model: str
    clock_gating_mode: str
    sram_port_model: str
    fidelity: str


CURRENT_DSE_PROFILE = DSEModelProfile(
    name="current-dse-v1",
    compiler_schedule_profile="current-dse-v1",
    compute_timing="ideal-ii1",
    hbm_model="hbm-dma-v4",
    cost_trace_granularity="affine-block-summary-v1",
    multi_chip_model="tile-aware-tp-cp-ep-v3",
    clock_gating_mode="ideal-hierarchical",
    sram_port_model="ideal-dual-port",
    fidelity="architectural_assumptions_with_calibrated_components",
)


RTL_VALIDATION_PROFILE = DSEModelProfile(
    name="rtl-validation-v1",
    compiler_schedule_profile="rtl-validation-v1",
    compute_timing="rtl-v1",
    hbm_model="hbm-dma-v4",
    cost_trace_granularity="detailed",
    multi_chip_model="tile-aware-tp-cp-ep-v3",
    clock_gating_mode="ungated",
    sram_port_model="replicated-single-port",
    fidelity="rtl_sensitivity_and_transactional_validation",
)
