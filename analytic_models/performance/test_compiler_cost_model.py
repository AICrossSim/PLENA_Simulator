from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.aten.cost_emitter import (
    CostTrace,
    MemoryEvent,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
)
from compiler.aten.isa_builder import DmaTransfer, RepeatAxis

from analytic_models.performance.compiler_cost_model import (
    TransactionalCycleModel,
    _one_layer_v4_trace,
    evaluate_compiler_cost,
)
from analytic_models.performance.hbm_service_v4 import _schedule_dma_count
from analytic_models.performance.hbm_cost import (
    HbmCalibration,
    HbmFormat,
    dma_geometry,
    dma_stream_geometry,
)
from analytic_models.performance.ramulator_calibration import generate_dma_calibration_plan


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "analytic_models/performance/calibration/hbm_surrogate_historical_v1.json"
HYBRID_CALIBRATION = ROOT / "analytic_models/performance/calibration/hbm_surrogate_hybrid_v1.json"
RAMULATOR_VALIDATION = (
    ROOT / "analytic_models/performance/calibration/hbm_surrogate_ramulator_v1_validation.json"
)
TRANSACTIONAL_V2_CALIBRATION = (
    ROOT / "analytic_models/performance/calibration/hbm_surrogate_transactional_dma_v2.json"
)
TRANSACTIONAL_V2_VALIDATION = (
    ROOT
    / "analytic_models/performance/calibration/hbm_surrogate_transactional_dma_v2_validation.json"
)
TRANSACTIONAL_V2_TARGET_VALIDATION = (
    ROOT
    / "analytic_models/performance/calibration/hbm_surrogate_transactional_dma_v2_target_validation.json"
)

TARGET_OPCODES = {
    "C_LOOP_END": 14_147_427,
    "C_LOOP_START": 319_889,
    "C_SET_ADDR_REG": 26_393,
    "C_SET_SCALE_REG": 26_392,
    "C_SET_STRIDE_REG": 26_392,
    "H_PREFETCH_M": 394_177,
    "H_PREFETCH_V": 5_249,
    "H_STORE_V": 1_024,
    "M_BMM_WO": 10_240,
    "M_BTMM": 10_240,
    "M_MM": 1_919_488,
    "M_MM_WO": 150_528,
    "S_ADDI_INT": 71_435_978,
    "S_ADD_FP": 3_612_675,
    "S_ADD_INT": 136_768,
    "S_EXP_FP": 1_187_840,
    "S_LD_FP": 5_263_368,
    "S_LUI_INT": 2_156_924,
    "S_MAP_V_FP": 128,
    "S_MAX_FP": 1_187_840,
    "S_MUL_FP": 1_212_416,
    "S_RECI_FP": 518_144,
    "S_SQRT_FP": 24_576,
    "S_ST_FP": 4_612_224,
    "S_SUB_FP": 1_187_840,
    "V_ADD_VF": 1_638_400,
    "V_ADD_VV": 16_758_784,
    "V_EXP_V": 2_826_240,
    "V_MUL_VF": 5_031_936,
    "V_MUL_VV": 5_378_048,
    "V_RECI_V": 1_638_400,
    "V_RED_MAX": 1_187_840,
    "V_RED_SUM": 2_170_880,
    "V_SUB_VF": 2_826_240,
}


def _mx_section() -> dict:
    return {
        "format": "Mx",
        "block": 8,
        "ELEM": {"type": "Fp", "sign": True, "exponent": 4, "mantissa": 3},
        "SCALE": {"type": "Fp", "sign": False, "exponent": 8, "mantissa": 0},
    }


def _target_cycle_model() -> TransactionalCycleModel:
    precision = {
        "HBM_M_WEIGHT_TYPE": _mx_section(),
        "HBM_M_KV_TYPE": _mx_section(),
        "HBM_V_ACT_TYPE": _mx_section(),
        "HBM_V_KV_TYPE": _mx_section(),
        "HBM_V_INT_TYPE": {
            "format": "Plain",
            "DATA_TYPE": {"type": "Int", "width": 32},
        },
    }
    return TransactionalCycleModel(
        settings_path=Path("target-test-settings.toml"),
        raw_settings={"PRECISION": precision},
        mlen=128,
        blen=128,
        vlen=128,
        hlen=128,
        broadcast_amount=1,
        hbm_channels=128,
        hbm_m_prefetch_amount=128,
        hbm_v_prefetch_amount=128,
        hbm_v_writeback_amount=128,
        matrix_sram_size=2048,
        dc_en=1,
        systolic_processing_overhead=0,
        vector_add_cycles=1,
        vector_mul_cycles=1,
        vector_exp_cycles=1,
        vector_reci_cycles=2,
        vector_max_cycles=4,
        vector_sum_cycles=8,
        scalar_fp_basic_cycles=1,
        scalar_fp_exp_cycles=1,
        scalar_fp_sqrt_cycles=1,
        scalar_fp_reci_cycles=1,
        scalar_int_basic_cycles=1,
    )


def _target_transfer(opcode: str) -> DmaTransfer:
    return DmaTransfer(
        opcode=opcode,
        direction="write" if opcode == "H_STORE_V" else "read",
        precision="matrix" if opcode == "H_PREFETCH_M" else "activation",
        element_base=0,
        scale_base=0,
        dim=128,
        amount=128,
        stride=128,
        write_amount=128 if opcode == "H_PREFETCH_M" else 1,
    )


def _target_trace() -> CostTrace:
    return CostTrace(
        dynamic_opcodes=Counter(TARGET_OPCODES),
        memory_events=[
            MemoryEvent("layer/decoder", _target_transfer(opcode), TARGET_OPCODES[opcode])
            for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
        ],
        metadata={
            "num_layers": 1,
            "hardware": {
                "mlen": 128,
                "blen": 128,
                "vlen": 128,
                "hlen": 128,
                "hbm_m_prefetch_amount": 128,
                "hbm_v_prefetch_amount": 128,
                "hbm_v_writeback_amount": 128,
                "hbm_channels": 128,
            },
            "attention_schedule": {"physical_broadcast": 1},
            "one_layer_dynamic_opcodes": TARGET_OPCODES,
        },
    )


def test_v4_one_layer_trace_reduces_only_model_layer_repeat() -> None:
    decoder_axis = RepeatAxis(
        name="decoder_layer",
        count=4,
        element_base_delta=4096,
        scale_base_delta=512,
    )
    global_event = MemoryEvent(
        "global/input", _target_transfer("H_PREFETCH_V"), 1, stream_index=0
    )
    layer_event = MemoryEvent(
        "layer/decoder",
        _target_transfer("H_PREFETCH_V"),
        4,
        enclosing_axes=(decoder_axis,),
        stream_index=1,
    )
    schedule = ScheduleSequence(
        (
            ScheduleInstruction(
                "H_PREFETCH_V", stage="global/input", memory_stream_index=0
            ),
            ScheduleRepeat(
                count=4,
                name="decoder_layer",
                repeat_kind="model_layer",
                body=ScheduleSequence(
                    (
                        ScheduleRepeat(
                            count=3,
                            name="inner",
                            repeat_kind="compile_time",
                            body=ScheduleSequence(()),
                        ),
                        ScheduleInstruction(
                            "H_PREFETCH_V",
                            stage="layer/decoder",
                            memory_stream_index=1,
                        ),
                    )
                ),
            ),
        )
    )
    one_layer = _one_layer_v4_trace(
        CostTrace(
            memory_events=[global_event, layer_event],
            schedule=schedule,
            metadata={"num_layers": 4},
        )
    )

    assert one_layer.metadata["num_layers"] == 1
    assert [event.multiplicity for event in one_layer.memory_events] == [1, 1]
    assert one_layer.memory_events[1].enclosing_axes == ()
    assert _schedule_dma_count(one_layer.schedule) == 2
    decoder_repeat = one_layer.schedule.children[1]
    assert isinstance(decoder_repeat, ScheduleRepeat)
    assert decoder_repeat.count == 1
    inner_repeat = decoder_repeat.body.children[0]
    assert isinstance(inner_repeat, ScheduleRepeat)
    assert inner_repeat.count == 3


def _historical_calibration_for_regression(path: Path) -> HbmCalibration:
    """Replay frozen surrogate math without weakening production compatibility checks.

    These legacy artifacts predate the precision-aware DMA semantics hash.  Their
    regression tests intentionally exercise the old model coefficients; normal
    callers still reject the stale hash through ``assert_compatible``.
    """
    calibration = HbmCalibration.load(path)
    return replace(
        calibration,
        compatibility={**calibration.compatibility, "dma_semantics_hash": "unknown"},
    )


def test_dma_geometry_matches_mxfp8_transaction_counts() -> None:
    fmt = HbmFormat("mxfp8", element_bits=8, scale_bits=8, block=8)
    prefetch = dma_geometry(_target_transfer("H_PREFETCH_M"), fmt)
    store = dma_geometry(_target_transfer("H_STORE_V"), fmt)

    assert prefetch.read_requests == 384
    assert prefetch.read_bytes == 24_576
    assert prefetch.write_bytes == 0
    assert store.read_requests == 384
    assert store.write_requests == 384
    assert store.read_bytes == 24_576
    assert store.write_bytes == 24_576


def test_target_histogram_reproduces_nonmemory_profile_and_hbm_bytes() -> None:
    trace = _target_trace()
    report = evaluate_compiler_cost(
        trace,
        _target_cycle_model(),
        _historical_calibration_for_regression(CALIBRATION),
        compute_timing_mode="legacy",
    )

    assert report.one_layer_category_latency_ns["matrix_compute"] == 247_165_952
    assert report.one_layer_category_latency_ns["vector_compute"] == 59_854_848
    assert report.one_layer_category_latency_ns["scalar_compute"] == 92_569_361
    assert report.one_layer_category_latency_ns["control"] == 14_546_493
    assert report.one_layer_hbm_read_bytes == 9_841_459_200
    assert report.one_layer_hbm_write_bytes == 25_165_824
    assert report.one_layer_latency_ns == pytest.approx(495_666_456, rel=0.01)


def test_hybrid_surrogate_meets_target_total_latency_threshold() -> None:
    report = evaluate_compiler_cost(
        _target_trace(),
        _target_cycle_model(),
        _historical_calibration_for_regression(HYBRID_CALIBRATION),
        compute_timing_mode="legacy",
    )

    assert report.one_layer_latency_ns == pytest.approx(495_666_456, rel=0.10)


def test_production_path_rejects_stale_dma_semantics_hash() -> None:
    with pytest.raises(ValueError, match="DMA semantics hash differs"):
        evaluate_compiler_cost(
            _target_trace(),
            _target_cycle_model(),
            HbmCalibration.load(CALIBRATION),
            compute_timing_mode="legacy",
        )


def test_standalone_ramulator_holdout_meets_error_thresholds() -> None:
    validation = json.loads(RAMULATOR_VALIDATION.read_text())

    assert validation["median_absolute_error_percent"] <= 15
    assert validation["max_absolute_error_percent"] <= 25


def test_transactional_dma_v2_meets_stream_and_target_thresholds() -> None:
    calibration = HbmCalibration.load(TRANSACTIONAL_V2_CALIBRATION)
    validation = json.loads(TRANSACTIONAL_V2_VALIDATION.read_text())
    target = json.loads(TRANSACTIONAL_V2_TARGET_VALIDATION.read_text())

    assert calibration.stream_models
    assert calibration.calibration_id == target["calibration_id"]
    assert validation["median_absolute_error_percent"] <= 10
    assert validation["p95_absolute_error_percent"] <= 15
    assert validation["max_absolute_error_percent"] <= 25
    assert abs(target["memory_error_percent"]) <= 15
    assert abs(target["total_error_percent"]) <= 5
    assert target["opcode_counts_match"]
    assert target["static_machine_instructions_match"]
    assert target["hbm_read_bytes_match"]
    assert target["hbm_write_bytes_match"]


def test_s_map_pays_vector_transfer_and_dispatch_cycles() -> None:
    assert _target_cycle_model().instruction_cycles("S_MAP_V_FP") == 256


def test_ramulator_plan_has_grouped_deterministic_split() -> None:
    plan = generate_dma_calibration_plan(repetitions=1, warmup=0)

    assert plan["schema_version"] == 2
    assert len(plan["patterns"]) == 7_560
    assert len({pattern["id"] for pattern in plan["patterns"]}) == 7_560
    assert {pattern["channels"] for pattern in plan["patterns"]} == {8, 32, 128}
    assert {pattern["split"] for pattern in plan["patterns"]} == {"train", "holdout"}
    assert {pattern["stream_instruction_count"] for pattern in plan["patterns"]} == {1, 8, 64}
    assert {pattern["precision"] for pattern in plan["patterns"]} == {"weight", "activation"}
    assert {pattern["group"].rsplit(":", 1)[-1] for pattern in plan["patterns"]} == {
        "reuse",
        "affine",
        "far",
    }
    grouped_splits = {}
    for pattern in plan["patterns"]:
        grouped_splits.setdefault(pattern["split_group"], set()).add(pattern["split"])
    assert all(len(splits) == 1 for splits in grouped_splits.values())


def test_ramulator_plan_can_emit_nested_projection_streams() -> None:
    plan = generate_dma_calibration_plan(
        repetitions=1,
        warmup=0,
        channels=(128,),
        dimensions=(128,),
        amount_multipliers=(1,),
        stream_lengths=(1, 8, 64),
        model_column_strides=(8192,),
        alignment_pairs=((0, 0),),
        nested_stream_shapes=((64, 8), (64, 16)),
        nested_opcodes=("H_PREFETCH_M",),
        include_base_streams=False,
        include_standard_strides=False,
    )

    assert len(plan["patterns"]) == 2
    assert {pattern["stream_instruction_count"] for pattern in plan["patterns"]} == {
        512,
        1024,
    }
    for pattern in plan["patterns"]:
        outer, inner = pattern["repeat_axes"]
        assert outer["count"] == 64
        assert outer["element_base_delta"] == 0
        assert inner["element_base_delta"] == 128 * 8192
        assert inner["scale_base_delta"] == 128 * 8192 // 8


def test_ordered_dma_stream_geometry_preserves_counts_and_reuse() -> None:
    transfer = _target_transfer("H_PREFETCH_M")
    fmt = HbmFormat("mxfp8", element_bits=8, scale_bits=8, block=8)
    reused = dma_stream_geometry(
        transfer,
        (RepeatAxis("instruction", 8),),
        fmt,
    )
    affine = dma_stream_geometry(
        transfer,
        (
            RepeatAxis(
                "instruction",
                8,
                element_base_delta=16_384,
                scale_base_delta=2_048,
            ),
        ),
        fmt,
    )

    assert reused.instruction_count == 8
    assert reused.read_bytes == 8 * 24_576
    assert reused.unique_blocks < affine.unique_blocks
    assert reused.reuse_ratio > affine.reuse_ratio

    scale_residue = dma_stream_geometry(
        replace(transfer, scale_base=16),
        (RepeatAxis("instruction", 8),),
        fmt,
    )
    assert reused.calibration_cell(128) != scale_residue.calibration_cell(128)
