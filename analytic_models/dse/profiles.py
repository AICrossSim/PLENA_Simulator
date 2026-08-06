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
    vector_scalar_schedule: str
    softmax_vector_schedule: str
    softmax_state_schedule: str
    pv_accumulation_schedule: str
    softmax_row_lanes: tuple[int, ...]
    fidelity: str


CURRENT_DSE_PROFILE = DSEModelProfile(
    name="current-dse-v1",
    compiler_schedule_profile="current-dse-v1",
    compute_timing="ideal-ii1",
    hbm_model="hbm-dma-v4",
    cost_trace_granularity="affine-block-summary-v1",
    multi_chip_model="tile-aware-dp-tp-ep-v4",
    clock_gating_mode="ideal-hierarchical",
    sram_port_model="ideal-dual-port",
    vector_scalar_schedule="rtl-v6",
    softmax_vector_schedule="multi-row-v1",
    softmax_state_schedule="row-bank-simd-v3",
    pv_accumulation_schedule="direct-packed-rmw-v1",
    softmax_row_lanes=(2, 4, 8),
    fidelity="architectural_assumptions_with_calibrated_components",
)


RTL_VALIDATION_PROFILE = DSEModelProfile(
    name="rtl-validation-v1",
    compiler_schedule_profile="rtl-validation-v1",
    compute_timing="rtl-v1",
    hbm_model="hbm-dma-v4",
    cost_trace_granularity="detailed",
    multi_chip_model="tile-aware-dp-tp-ep-v4",
    clock_gating_mode="ungated",
    sram_port_model="replicated-single-port",
    vector_scalar_schedule="rtl-v6",
    softmax_vector_schedule="multi-row-v1",
    softmax_state_schedule="row-bank-simd-v3",
    pv_accumulation_schedule="direct-packed-rmw-v1",
    softmax_row_lanes=(1,),
    fidelity="rtl_sensitivity_and_transactional_validation",
)
