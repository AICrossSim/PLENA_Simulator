import pytest

from compiler.aten.isa_builder import DmaTransfer, Instr
from compiler.aten.program_sink import SymbolicCostSink

from analytic_models.latency import ConfiguredBandwidthMemoryProvider, MainTimingConfig, estimate_compute_latency, estimate_latency
from analytic_models.power import estimate_power

from .test_actions import _coefficients, _hardware


def _trace_and_latency():
    sink = SymbolicCostSink(default_stage="decoder/test")
    sink.emit_instruction(Instr("M_MM"))
    sink.emit_instruction(Instr("V_MUL_VV"))
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        role="activation",
        element_base_bytes=0,
        scale_base_bytes=None,
        dim=16,
        amount=4,
        stride_bytes=16,
    )
    sink.emit_instruction(Instr.with_metadata("H_PREFETCH_V", dma=transfer))
    trace = sink.finish()
    timing = MainTimingConfig(mlen=16, blen=4, vlen=16, hlen=4, broadcast_amount=4)
    latency = estimate_latency(
        trace,
        estimate_compute_latency(trace, timing),
        ConfiguredBandwidthMemoryProvider(16.0).estimate(trace),
    )
    return trace, latency


def _properties(external=True):
    components = {
        component: {
            "logic_area_um2": 100.0,
            "logic_leakage_mw": 0.01,
            "clock_density_pj_per_cycle_um2": 0.001,
            "fixed_clock_pj_per_active_cycle": 0.1,
        }
        for component in ("matrix", "vector", "scalar", "control", "hbm_controller")
    }
    value = {
        "schema_version": "test-v1",
        "calibration_id": "test-properties",
        "hardware": vars(_hardware()),
        "components": components,
        "sram_access_energy_pj": {
            memory: {"read": 1.0, "write": 1.0}
            for memory in ("matrix_sram", "vector_sram", "scalar_fp_sram", "scalar_int_sram")
        },
    }
    if external:
        value["external_memory"] = {
            "model": "test-hbm",
            "technology": "HBM3E",
            "calibration_status": "test",
            "capacity_bytes": 80_000_000_000,
            "coefficients": {
                "background_power_mw_per_gb": {"p10": 50, "p50": 75, "p90": 100},
                "read_energy_pj_per_bit": 3.0,
                "write_energy_pj_per_bit": 3.6,
            },
        }
    return value


def test_power_components_sum_and_bounds_are_ordered():
    trace, latency = _trace_and_latency()
    report = estimate_power(trace, latency, _properties(), logic_coefficients=_coefficients())
    assert report.system_energy_pj == pytest.approx(report.onchip_energy_pj + report.external_hbm_energy_pj)
    assert report.system_energy_low_pj <= report.system_energy_pj <= report.system_energy_high_pj
    assert report.ideal_clock_energy_pj <= report.ungated_clock_energy_pj
    assert report.external_hbm_read_energy_pj == latency.memory.physical_read_bytes * 8 * 3.0


def test_ungated_mode_is_an_upper_bound_without_changing_dynamic_energy():
    trace, latency = _trace_and_latency()
    ideal = estimate_power(trace, latency, _properties(), logic_coefficients=_coefficients())
    ungated = estimate_power(
        trace,
        latency,
        _properties(),
        clock_gating_mode="ungated",
        logic_coefficients=_coefficients(),
    )
    assert ungated.logic_dynamic_energy_pj == ideal.logic_dynamic_energy_pj
    assert ungated.system_energy_pj >= ideal.system_energy_pj


def test_missing_physical_component_fails_closed():
    trace, latency = _trace_and_latency()
    properties = _properties()
    del properties["components"]["matrix"]
    with pytest.raises(ValueError, match="missing components"):
        estimate_power(trace, latency, properties, logic_coefficients=_coefficients())
