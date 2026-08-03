from compiler.aten.isa_builder import ActiveDimensions, DmaTransfer, Instr
from compiler.aten.program_sink import SymbolicCostSink

from analytic_models.power import (
    DEFAULT_LOGIC_ENERGY,
    ActionHardwareConfig,
    build_energy_actions,
    estimate_action_energy,
)


def _hardware():
    return ActionHardwareConfig(mlen=16, blen=4, vlen=16)


def _coefficients():
    fp = {"FP_E6M5": 1.0, "default": 1.0}
    return {
        "model": "test-actions-v1",
        "calibration_status": "test",
        "dynamic_nominal_pj": {
            "matrix": {
                mode: {
                    "pe_cycle": {
                        "base": 1.0,
                        "bit_product": 0.0,
                        "width_sum": 0.0,
                        "feed_cycle": 0.0,
                        "slice_fixed": 0.0,
                    },
                    "reduce_node_bit": 1.0,
                    "output_bit": 1.0,
                }
                for mode in ("mxint", "mxfp")
            },
            "vector": {
                family: fp
                for family in (
                    "lane_add_sub_vv",
                    "lane_multiply_vv",
                    "reduction_sum_full",
                )
            },
            "scalar": {
                "integer_alu": {"32": 1.0, "default": 1.0},
                "register_or_sram_access": fp,
            },
            "control": {"frontend_issue": 1.0},
            "hbm_controller": {
                "default": 1.0,
                "matrix_prefetch": 1.0,
                "vector_prefetch": 1.0,
                "vector_writeback": 1.0,
            },
        },
        "activity_envelope": {},
    }


def _trace():
    sink = SymbolicCostSink(default_stage="decoder/test")
    sink.emit_instruction(Instr("M_MM"))
    sink.emit_instruction(
        Instr.with_metadata(
            "V_ADD_VV",
            active=ActiveDimensions(lanes=4, total_lanes=16),
        )
    )
    sink.emit_instruction(Instr("S_ADDI_INT"))
    sink.emit_instruction(Instr("C_SET_ADDR_REG"))
    transfer = DmaTransfer(
        opcode="H_PREFETCH_M",
        direction="read",
        role="weight",
        element_base_bytes=0,
        scale_base_bytes=None,
        dim=16,
        amount=4,
        stride_bytes=16,
    )
    sink.emit_instruction(Instr.with_metadata("H_PREFETCH_M", dma=transfer))
    return sink.finish()


def test_action_census_covers_compute_dma_and_sram():
    actions = build_energy_actions(_trace(), _hardware())
    keys = {(item.component, item.action) for item in actions}
    assert ("matrix", "array_compute") in keys
    assert ("matrix", "cross_k_reduce") in keys
    assert ("vector", "lane_add_sub_vv") in keys
    assert ("scalar", "integer_alu") in keys
    assert ("control", "frontend_issue") in keys
    assert ("hbm_controller", "matrix_prefetch") in keys
    assert ("matrix_sram", "write") in keys


def test_dynamic_energy_uses_active_vector_lanes_and_is_nonnegative():
    report = estimate_action_energy(_trace(), _hardware(), _coefficients())
    assert report.opcode_coverage == 1.0
    assert report.nominal_energy_pj > 0
    assert report.low_energy_pj <= report.nominal_energy_pj <= report.high_energy_pj
    vector = report.by_component_pj["vector"]
    assert vector == 4.0


def test_hbm_instruction_without_dma_fails_closed():
    sink = SymbolicCostSink(default_stage="decoder/test")
    sink.emit_instruction(Instr("H_PREFETCH_M"))
    try:
        build_energy_actions(sink.finish(), _hardware())
    except ValueError as error:
        assert "DMA coverage" in str(error) or "HBM instruction/DMA parity" in str(error)
    else:
        raise AssertionError("missing DMA metadata was accepted")


def test_default_main_calibration_covers_the_unmodified_isa_actions():
    report = estimate_action_energy(_trace(), _hardware(), DEFAULT_LOGIC_ENERGY)
    assert report.opcode_coverage == 1.0
    assert report.provenance["calibration_status"] == "rtl_activity_calibrated_candidate_main_compatible"
