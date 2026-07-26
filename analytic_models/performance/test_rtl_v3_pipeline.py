"""Focused checks for the RTL-v3 scalar ROB and segment-parallel schedule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from compiler.aten.cost_emitter import (
    CostTrace,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
)

from analytic_models.performance.rtl_opcode_timing import (
    ComputeFormat,
    ComputePrecisionConfig,
    FpFormat,
    RtlOpcodeTimingCalibration,
    TimingHardware,
)
from analytic_models.performance.compiler_cost_model import (
    ComputeTimingContext,
    TransactionalCycleModel,
    _evaluate_compute_pipeline,
    _scale_model_layer_pipeline,
)
from analytic_models.performance.scheduled_shadow import evaluate_scheduled_shadow


CALIBRATION = RtlOpcodeTimingCalibration.load()
HARDWARE = TimingHardware(mlen=16, blen=4, vlen=16, hlen=4, broadcast_amount=1)
PRECISION = ComputePrecisionConfig(
    weight=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    activation=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    kv=ComputeFormat("mxfp", 8, exponent=4, mantissa=3, block=8),
    matrix_internal_fp=FpFormat(8, 7),
    vector_internal_fp=FpFormat(8, 7),
    scalar_fp=FpFormat(8, 7),
    integer_bits=32,
)


def _cycle_model() -> TransactionalCycleModel:
    return TransactionalCycleModel(
        settings_path=CALIBRATION.path,
        raw_settings={},
        mlen=16,
        blen=4,
        vlen=16,
        hlen=4,
        broadcast_amount=1,
        hbm_channels=8,
        hbm_m_prefetch_amount=16,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
        matrix_sram_size=32,
        dc_en=0,
        systolic_processing_overhead=1,
        vector_add_cycles=1,
        vector_mul_cycles=1,
        vector_exp_cycles=1,
        vector_reci_cycles=1,
        vector_max_cycles=1,
        vector_sum_cycles=1,
        scalar_fp_basic_cycles=1,
        scalar_fp_exp_cycles=1,
        scalar_fp_sqrt_cycles=1,
        scalar_fp_reci_cycles=1,
        scalar_int_basic_cycles=1,
    )


def _schedule(*instructions: ScheduleInstruction):
    trace = CostTrace(
        dynamic_opcodes=Counter(item.opcode for item in instructions),
        schedule=ScheduleSequence(tuple(instructions)),
        metadata={"vector_scalar_schedule": "rtl-v3"},
    )
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=PRECISION,
        calibration=CALIBRATION,
        retain_events=True,
    )
    assert result.status == "complete", result.reason
    return result


def test_independent_scalar_operations_issue_at_one_cycle_ii() -> None:
    result = _schedule(
        *(
            ScheduleInstruction("S_MUL_FP", (f"f{destination}", "f1", "f2"))
            for destination in range(3, 11)
        )
    )
    accepted = [event.accepted_cycle for event in result.events]
    assert accepted == list(range(8))
    assert [event.scalar_rob_tag for event in result.events] == list(range(8))
    assert result.makespan_cycles < 8 * CALIBRATION.data["scalar"]["fp_mul_done_cycles"]


def test_scalar_rob_full_stalls_ninth_long_latency_operation() -> None:
    result = _schedule(
        *(
            ScheduleInstruction("S_EXP_FP", (f"f{destination}", "f1"))
            for destination in range(3, 12)
        )
    )
    ninth = result.events[-1]
    assert ninth.stall_reason == "scalar_rob_full"
    assert ninth.recovery_cycles == 1
    assert result.stall_cycles_by_reason["scalar_rob_full"] > 0


def test_scalar_raw_and_waw_use_ready_and_retire_boundaries() -> None:
    raw = _schedule(
        ScheduleInstruction("S_MUL_FP", ("f3", "f1", "f2")),
        ScheduleInstruction("S_ADD_FP", ("f4", "f3", "f1")),
    )
    assert raw.events[1].stall_reason == "scalar_fp_operand_not_ready"
    assert raw.events[1].accepted_cycle > raw.events[0].issue_cycle + 1

    waw = _schedule(
        ScheduleInstruction("S_EXP_FP", ("f3", "f1")),
        ScheduleInstruction("S_MV_FP", ("f3", "f2")),
    )
    assert waw.events[1].stall_reason == "scalar_fp_write_port_busy"
    assert waw.events[1].accepted_cycle > int(waw.events[0].scalar_retire_cycle or 0)


def test_multi_segment_reduction_and_lane_chain_observe_vector_sram() -> None:
    result = _schedule(
        ScheduleInstruction("S_ADDI_INT", ("gp1", "gp0", "1024")),
        ScheduleInstruction("S_ADDI_INT", ("gp2", "gp0", "2048")),
        ScheduleInstruction("S_ADDI_INT", ("gp3", "gp0", "0")),
        ScheduleInstruction("V_RED_SUM_SEGS", ("gp2", "gp1", "2")),
        ScheduleInstruction("S_LD_VLANE_FP", ("f3", "gp2", "gp3")),
        ScheduleInstruction("S_ST_VLANE_FP", ("f3", "gp2", "gp3")),
        ScheduleInstruction("V_MUL_VSEG", ("gp1", "gp1", "gp2", "2", "1")),
    )
    reduction, lane_load, lane_store, broadcast = result.events[-4:]
    assert lane_load.stall_reason == "vector_sram_operand_not_ready"
    assert lane_store.stall_reason == "scalar_fp_operand_not_ready"
    assert broadcast.start_cycle >= lane_store.result_ready_cycle
    assert reduction.resource == "vector_pipeline"


def test_vector_shift_uses_measured_logarithmic_pipeline_depth() -> None:
    measured = CALIBRATION.estimate("V_SHIFT_V", HARDWARE, PRECISION)
    assert measured is not None
    assert measured.rtl_supported
    assert measured.calibration_status == "full_machine_measured"
    assert measured.resource_cycles == (
        CALIBRATION.data["vector"]["shift_base_cycles"]
        + CALIBRATION.data["vector"]["shift_per_level_cycles"] * 4
    )

    production = TimingHardware(
        mlen=2048,
        blen=1024,
        vlen=2048,
        hlen=128,
        broadcast_amount=8,
    )
    extrapolated = CALIBRATION.estimate("V_SHIFT_V", production, PRECISION)
    assert extrapolated is not None
    assert extrapolated.rtl_supported
    assert extrapolated.calibration_status == "structural_extrapolation"
    assert extrapolated.resource_cycles == (
        CALIBRATION.data["vector"]["shift_base_cycles"]
        + CALIBRATION.data["vector"]["shift_per_level_cycles"] * 11
    )


def test_costemitter_compute_pipeline_uses_makespan_not_resource_sum() -> None:
    instructions = tuple(
        ScheduleInstruction("S_MUL_FP", (f"f{destination}", "f1", "f2"))
        for destination in range(3, 11)
    )
    trace = CostTrace(
        dynamic_opcodes=Counter(item.opcode for item in instructions),
        schedule=ScheduleSequence(instructions),
        metadata={
            "vector_scalar_schedule": "rtl-v3",
            "num_layers": 1,
            "one_layer_dynamic_opcodes": {"S_MUL_FP": 8},
        },
    )
    model = _cycle_model()
    pipeline = _evaluate_compute_pipeline(
        trace,
        model,
        ComputeTimingContext("rtl-v1", PRECISION, CALIBRATION),
    )
    assert pipeline is not None
    assert pipeline.total.makespan_cycles == 15
    assert pipeline.total.stage_critical_path_cycles == {"global": 15}
    assert sum(pipeline.total.stage_critical_path_cycles.values()) == (
        pipeline.total.makespan_cycles
    )
    assert pipeline.stage_latency_ns == {"global": 15.0}
    serial_work = 8 * CALIBRATION.data["scalar"]["fp_mul_done_cycles"]
    assert int(pipeline.total.makespan_cycles or 0) < serial_work


def test_compute_pipeline_persistent_cache_reuses_exact_replay(
    tmp_path: Path,
) -> None:
    instructions = tuple(
        ScheduleInstruction("S_MUL_FP", (f"f{destination}", "f1", "f2"))
        for destination in range(3, 11)
    )
    trace = CostTrace(
        dynamic_opcodes=Counter(item.opcode for item in instructions),
        schedule=ScheduleSequence(instructions),
        metadata={
            "config_hash": "rtl-v3-cache-test",
            "vector_scalar_schedule": "rtl-v3",
            "num_layers": 1,
            "one_layer_dynamic_opcodes": {"S_MUL_FP": 8},
        },
    )
    timing = ComputeTimingContext("rtl-v1", PRECISION, CALIBRATION)

    first = _evaluate_compute_pipeline(
        trace,
        _cycle_model(),
        timing,
        persistent_cache_dir=tmp_path,
    )
    second = _evaluate_compute_pipeline(
        trace,
        _cycle_model(),
        timing,
        persistent_cache_dir=tmp_path,
    )

    assert first is not None and second is not None
    assert first.total.makespan_cycles == second.total.makespan_cycles == 15
    assert first.persistent_cache_hit is False
    assert second.persistent_cache_hit is True
    assert first.persistent_cache_key == second.persistent_cache_key


def test_model_layer_pipeline_scaling_repeats_only_layer_stages() -> None:
    one_layer = _schedule(
        ScheduleInstruction("S_MUL_FP", ("f3", "f1", "f2"), stage="layer/ffn"),
        ScheduleInstruction("S_MV_FP", ("f4", "f3"), stage="global/final"),
    )
    scaled = _scale_model_layer_pipeline(one_layer, 4)

    expected = {
        "layer/ffn": one_layer.stage_critical_path_cycles["layer/ffn"] * 4,
        "global/final": one_layer.stage_critical_path_cycles["global/final"],
    }
    assert scaled.stage_critical_path_cycles == expected
    assert scaled.makespan_cycles == sum(expected.values())
    assert scaled.validation["full_model_pipeline_fidelity"] == (
        "repeated_layer_stage_scaling"
    )
    assert scaled.validation["serial_resource_work_fallback"] is False


def test_repeat_fast_forward_ignores_overwritten_gp_scratch_state() -> None:
    body = ScheduleSequence(
        (
            ScheduleInstruction("S_LUI_INT", ("gp2", "4096")),
            ScheduleInstruction("S_ADDI_INT", ("gp2", "gp2", "64")),
            ScheduleInstruction("S_ADDI_INT", ("gp1", "gp1", "16")),
            ScheduleInstruction("V_ADD_VV", ("gp3", "gp1", "gp2", "0")),
        )
    )
    trace = CostTrace(
        dynamic_opcodes=Counter(
            {
                "S_LUI_INT": 1024,
                "S_ADDI_INT": 2048,
                "V_ADD_VV": 1024,
            }
        ),
        schedule=ScheduleSequence(
            (ScheduleRepeat(1024, body, name="overwrite_scratch_rows"),)
        ),
        metadata={"vector_scalar_schedule": "rtl-v3"},
    )
    result = evaluate_scheduled_shadow(
        trace,
        hardware=HARDWARE,
        precision=PRECISION,
        calibration=CALIBRATION,
        retain_events=False,
        max_expanded_instructions=256,
        initial_gp={1: 0, 3: 4096},
    )
    assert result.status == "complete", result.reason
    assert result.validation["repeat_fast_forwards"] > 0
