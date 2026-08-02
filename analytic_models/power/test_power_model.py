from __future__ import annotations

from dataclasses import replace
import csv
import gzip
import json
from pathlib import Path
import subprocess
import threading
from typing import ClassVar

import pytest

from compiler.aten.cost_emitter import (
    CONTROL_OPS,
    MATRIX_COMPUTE_OPS,
    MEMORY_OPS,
    SCALAR_COMPUTE_OPS,
    VECTOR_COMPUTE_OPS,
    CostTrace,
    CostSink,
    EnergyAction,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
    optimize_cost_trace_loop_agu,
)
from compiler.aten.cost_frontend import _scale_trace
from compiler.aten.isa_builder import DmaTransfer, IsaBuilder

from analytic_models.power import (
    estimate_external_hbm_power,
    estimate_multi_chip_system_power,
    estimate_onchip_power,
    estimate_system_power,
)
from analytic_models.performance.multi_chip_model import (
    estimate_multi_chip_latency,
)
from analytic_models.power.power_model import (
    DEFAULT_LOGIC_ENERGY,
    _load_logic_coefficients,
    _logic_action_energy_v2,
    _read_vector_rtl_v4_energy,
    _read_vector_rtl_v5_energy,
    _trace_actions,
)
from analytic_models.power.multi_chip import (
    _scaled_energy_actions,
    _sum_onchip_reports,
)
from analytic_models.power.clock_work import build_clock_work
from analytic_models.power.sram_energy import (
    build_sram_energy_catalog,
    parse_asap7_sram_lib,
)
from analytic_models.power.scripts.run_power_calibration import (
    SCENARIOS,
    build_agu_plan,
    build_compact_v5_plan,
    build_plan,
    build_plan_v2,
    scenarios_for_compact_v5,
    scenarios_for_point_v2,
)
from analytic_models.power.scripts.cleanup_power_tmp import cleanup as cleanup_power_tmp
from analytic_models.power.scripts.fit_power_calibration import _fit
from analytic_models.power.scripts.fit_power_calibration_v2 import (
    _activity_envelope,
    _linear_slope,
    _matrix_invariants,
    _pair_idle,
)
from analytic_models.power.scripts.fit_vector_rtl_v4_power_delta import (
    _action_dynamic_energy,
)
from analytic_models.power.scripts.run_rtl_activity_power_calibration import (
    POWER_FIELDS,
    ResourceGate,
    _acquire_runner_locks,
    _export_latest_complete,
    _parse_saif_map_seq_coverage,
    _patch_power_synthesis_flow,
    _run_dc_retry,
    _release_runner_locks,
    _resume_row_is_current,
    _write_verilator_saif_name_map,
    _packed_port_activity_tcl,
    _parse_power_group_table,
    _power_group_fields,
    _saif_signal_activity,
)
from analytic_models.power.scripts.rtl_activity import (
    qwen_mix_semantic_hash,
    weighted_microkernel_schedule,
)
from analytic_models.area_new.scripts.license_utils import is_dc_license_unavailable_text
from analytic_models.performance.hbm_service_model import MemoryFormat
from analytic_models.performance.hbm_service_v4 import plan_dma_request_manifest

ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict[str, object]:
    return {
        "MLEN": 16,
        "VLEN": 16,
        "BLEN": 4,
        "HLEN": 8,
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "WEIGHT_WIDTH": "MXINT4",
        "FP_SETTING": "FP_E5M6",
        "MX_SCALE_WIDTH": 8,
        "INT_DATA_WIDTH": 32,
        "MATRIX_SRAM_DEPTH": 32,
        "VECTOR_SRAM_DEPTH": 32,
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": 64,
        "HBM_M_Prefetch_Amount": 16,
        "HBM_V_Prefetch_Amount": 4,
        "HBM_V_Writeback_Amount": 4,
        "CLOCK_PERIOD_PS": 1000,
        "SEQ_LEN": 8,
        "BATCH_SIZE": 1,
    }


def _trace(repeats: int = 1):
    body = IsaBuilder()
    body.instr("M_MM", "gp1", "gp2")
    body.instr("M_MM_WO", "gp3")
    body.instr("V_ADD_VV", "gp1", "gp2", "gp3")
    body.instr("V_RED_SUM_SEGS", "gp1", "gp2", 2)
    body.instr("S_MUL_FP", "f1", "f2", "f3")
    body.instr("C_SET_ADDR_REG", "gp1", 0)
    transfer = DmaTransfer(
        opcode="H_PREFETCH_V",
        direction="read",
        precision="activation",
        precision_role="activation",
        element_base=0,
        scale_base=4096,
        dim=16,
        amount=4,
        stride=16,
    )
    sink = CostSink()
    with sink.repeated_region(repeats, name="power_test"):
        sink.emit(body)
    dma = IsaBuilder().dma_instr("H_PREFETCH_V", "gp1", "gp2", dma=transfer)
    sink.emit(dma)
    return sink.finish()


def test_energy_actions_cover_all_compiler_compute_and_memory_opcodes() -> None:
    trace = _trace()
    emitted_components = {action.component for action in trace.energy_actions}
    assert {"matrix", "vector", "scalar", "control", "hbm_controller"} <= emitted_components
    covered = MATRIX_COMPUTE_OPS | VECTOR_COMPUTE_OPS | SCALAR_COMPUTE_OPS | CONTROL_OPS
    # The private family mapping is exercised through one CostSink per opcode,
    # which catches newly added compute opcodes without an energy action.
    for opcode in covered - {"C_LOOP_START", "C_LOOP_END"}:
        builder = IsaBuilder().instr(opcode)
        sink = CostSink()
        sink.emit(builder)
        actions = sink.finish().energy_actions
        assert any(not action.component.endswith("_sram") for action in actions), opcode
    loop = IsaBuilder().instr("C_LOOP_START", "gp1", 1).instr("C_LOOP_END")
    sink = CostSink()
    sink.emit(loop)
    assert sum(action.component == "control" for action in sink.finish().energy_actions) == 2
    assert MEMORY_OPS == {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}


def test_energy_actions_preserve_structural_variants_and_segments() -> None:
    actions = _trace().energy_actions
    matrix = {(action.action, action.precision) for action in actions if action.component == "matrix"}
    assert ("array_compute", "M_MM") in matrix
    assert ("cross_k_reduce", "M_MM") in matrix
    assert ("output_conversion", "M_MM_WO") in matrix
    reduction = next(action for action in actions if action.action == "reduction_sum_segments")
    assert reduction.segment_log2 == 2
    assert reduction.segment_count == 0  # Runtime VLEN resolves the count.
    assert reduction.variant != ""


def test_energy_actions_preserve_exact_compressed_vector_mask_activity() -> None:
    body = IsaBuilder()
    body.instr("S_ADDI_INT", "gp7", "gp0", 0b0101)
    body.instr("C_SET_V_MASK_REG", "gp7")
    body.instr("V_MUL_VF", "gp1", "gp1", "f1", 1)
    sink = CostSink()
    with sink.repeated_region(8, name="masked_rows"):
        sink.emit(body)
    action = next(
        item
        for item in sink.finish().energy_actions
        if item.component == "vector"
    )
    assert action.count == 8
    assert action.segment_count == 2
    assert action.activity_fidelity == "exact_segment_mask"


def test_vector_lane_load_and_store_use_distinct_energy_families() -> None:
    families = {}
    for opcode in ("S_LD_VLANE_FP", "S_ST_VLANE_FP"):
        sink = CostSink()
        sink.emit(IsaBuilder().instr(opcode, "f1", "gp1", "gp2"))
        families[opcode] = {
            action.action
            for action in sink.finish().energy_actions
            if action.component == "scalar"
        }
    assert families["S_LD_VLANE_FP"] == {"vector_lane_load"}
    assert families["S_ST_VLANE_FP"] == {"vector_lane_store"}


def test_energy_actions_are_coalesced_and_do_not_serialize_schedule() -> None:
    trace = _trace(8)
    keys = [
        (
            action.stage,
            action.component,
            action.action,
            action.precision,
            action.active_lanes,
            action.total_lanes,
            action.active_bits,
            action.segment_count,
        )
        for action in trace.energy_actions
    ]
    assert len(keys) == len(set(keys))

    class GuardedTrace:
        schema_version = 4
        metadata: ClassVar[dict[str, object]] = {}
        energy_actions: ClassVar[list[EnergyAction]] = [
            EnergyAction("stage", "control", "frontend_issue", 3)
        ]

        def to_dict(self):
            raise AssertionError("power estimator must not serialize the schedule")

    actions, metadata = _trace_actions(GuardedTrace())
    assert actions[0]["count"] == 3
    assert metadata["schema_version"] == 4


def test_loop_agu_implicit_work_uses_mapped_activity_coefficients() -> None:
    stage = "layer/ffn"
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp3", "gp3", "gp4"), stage),
            ScheduleInstruction("S_ADDI_INT", ("gp3", "gp3", "64"), stage),
        )
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (ScheduleRepeat(32, body, "ffn_accumulate", "compile_time"),)
        )
    )
    trace.stages[stage].dynamic_opcodes.update(
        {"V_ADD_VV": 32, "S_ADDI_INT": 32}
    )
    trace.stages[stage].static_opcodes.update(
        {"V_ADD_VV": 32, "S_ADDI_INT": 32}
    )
    trace.dynamic_opcodes.update(trace.stages[stage].dynamic_opcodes)
    trace.static_opcodes.update(trace.stages[stage].static_opcodes)

    optimized = optimize_cost_trace_loop_agu(trace)
    agu_actions = {
        action.action: action
        for action in optimized.energy_actions
        if action.component == "agu"
    }

    assert {
        "agu_config",
        "agu_loop_setup",
        "agu_loop_boundary",
        "agu_stream_step",
        "agu_offset_read",
    } <= set(agu_actions)
    assert agu_actions["agu_loop_boundary"].count == 32
    assert agu_actions["agu_stream_step"].active_lanes == 1
    report = estimate_onchip_power(
        {**_config(), "address_generation_mode": "loop-agu-v1"},
        optimized,
        {
            "compute_timing_mode": "ideal-ii1",
            "compute_pipeline_makespan_cycles": 64,
        },
        clock_gating_mode="ideal_hierarchical",
    )
    assert (
        report["agu_power_calibration_status"]
        == "rtl_activity_mapped_dc_candidate"
    )
    assert report["component_logic_dynamic_energy_pj"]["agu"] > 0.0
    assert not any("frontend-derived proxy" in warning for warning in report["warnings"])


def test_power_breakdown_is_nonnegative_and_sums() -> None:
    report = estimate_onchip_power(
        _config(),
        _trace(),
        {
            "compute_pipeline_makespan_cycles": 100,
            "hbm_read_bytes": 256,
            "hbm_write_bytes": 64,
            "hbm_read_requests": 4,
            "hbm_write_requests": 1,
        },
    )
    terms = [
        report["logic_dynamic_energy_mj"],
        report["sram_dynamic_energy_mj"],
        report["logic_leakage_energy_mj"],
    ]
    assert all(value >= 0 for value in terms)
    assert report["onchip_energy_mj"] == pytest.approx(sum(terms))
    expected_power = report["onchip_energy_mj"] * 1e6 / report["makespan_ns"]
    assert report["onchip_average_power_w"] == pytest.approx(expected_power)
    assert report["sram_leakage_status"] == "unavailable"
    assert "external_hbm" in report["excludes"]


def _external_timing(
    *,
    read_bytes: int,
    write_bytes: int,
    payload_read_bytes: int,
    payload_write_bytes: int,
    runtime_ns: float = 1_000_000.0,
) -> dict[str, object]:
    def bucket() -> dict[str, int]:
        return {
            "physical_read_bytes": read_bytes,
            "physical_write_bytes": write_bytes,
            "payload_read_bytes": payload_read_bytes,
            "payload_write_bytes": payload_write_bytes,
            "read_requests": read_bytes // 64,
            "write_requests": write_bytes // 64,
        }

    return {
        "roofline_latency_ns": runtime_ns,
        "hbm_read_bytes": read_bytes,
        "hbm_write_bytes": write_bytes,
        "hbm_payload_read_bytes": payload_read_bytes,
        "hbm_payload_write_bytes": payload_write_bytes,
        "hbm_traffic_breakdown": {
            "by_role": {"weight": bucket()},
            "by_stage": {"layer/projection": bucket()},
            "by_opcode": {"H_PREFETCH_M": bucket()},
        },
    }


def test_hbm3e_external_power_uses_actual_traffic_not_peak_bandwidth() -> None:
    report = estimate_external_hbm_power(
        {
            "HBM_CAPACITY_BYTES": 80_000_000_000,
            "HBM_BANDWIDTH_GBPS": 2039,
        },
        _external_timing(
            read_bytes=1_000_000_000,
            write_bytes=500_000_000,
            payload_read_bytes=900_000_000,
            payload_write_bytes=400_000_000,
        ),
    )

    assert report["hbm_background_power_p10_w"] == pytest.approx(4.0)
    assert report["hbm_background_power_p50_w"] == pytest.approx(6.0)
    assert report["hbm_background_power_p90_w"] == pytest.approx(8.0)
    assert report["hbm_read_energy_mj"] == pytest.approx(24.0)
    assert report["hbm_write_energy_mj"] == pytest.approx(14.4)
    assert report["hbm_background_energy_mj"] == pytest.approx(6.0)
    assert report["external_hbm_energy_mj"] == pytest.approx(44.4)
    assert report["achieved_average_bandwidth_gbps"] == pytest.approx(1500.0)
    assert report["bandwidth_utilization"] == pytest.approx(1500.0 / 2039.0)


def test_hbm3e_external_power_charges_partial_line_rmw_physical_bytes() -> None:
    manifest = plan_dma_request_manifest(
        {
            "opcode": "H_STORE_V",
            "direction": "write",
            "element_base": 0,
            "scale_base": 16,
            "dim": 64,
            "amount": 1,
            "stride_bytes": 32,
            "rstride": 1,
            "write_amount": 1,
        },
        MemoryFormat("mxint", 4, 8, 64, "MXINT4"),
    )
    assert manifest.read_bytes == 64
    assert manifest.write_bytes == 64
    report = estimate_external_hbm_power(
        {
            "HBM_CAPACITY_BYTES": 80_000_000_000,
            "HBM_BANDWIDTH_GBPS": 2039,
        },
        _external_timing(
            read_bytes=manifest.read_bytes,
            write_bytes=manifest.write_bytes,
            payload_read_bytes=manifest.payload_read_bytes,
            payload_write_bytes=manifest.payload_write_bytes,
            runtime_ns=1000.0,
        ),
    )
    assert report["hbm_read_energy_mj"] == pytest.approx(64 * 8 * 3.0e-9)
    assert report["hbm_write_energy_mj"] == pytest.approx(64 * 8 * 3.6e-9)


def test_hbm3e_external_power_tracks_precision_packed_physical_traffic() -> None:
    transfer = {
        "opcode": "H_PREFETCH_V",
        "direction": "read",
        "element_base": 0,
        "scale_base": 1 << 20,
        "dim": 512,
        "amount": 64,
        "stride_bytes": 512,
        "rstride": 1,
        "write_amount": 1,
    }
    formats = (
        MemoryFormat("mxint", 4, 8, 64, "MXINT4"),
        MemoryFormat("mxint", 8, 8, 64, "MXINT8"),
        MemoryFormat("mxfp", 8, 8, 8, "MXFP_E4M3"),
    )
    reports = []
    for fmt in formats:
        manifest = plan_dma_request_manifest(transfer, fmt)
        reports.append(
            estimate_external_hbm_power(
                {
                    "HBM_CAPACITY_BYTES": 80_000_000_000,
                    "HBM_BANDWIDTH_GBPS": 2039,
                },
                _external_timing(
                    read_bytes=manifest.read_bytes,
                    write_bytes=manifest.write_bytes,
                    payload_read_bytes=manifest.payload_read_bytes,
                    payload_write_bytes=manifest.payload_write_bytes,
                ),
            )
        )

    physical_bytes = [report["physical_read_bytes"] for report in reports]
    read_energy = [report["hbm_read_energy_mj"] for report in reports]
    assert physical_bytes == [17_408, 33_280, 36_864]
    assert read_energy[0] < read_energy[1] < read_energy[2]


def test_external_power_rejects_inconsistent_traffic_breakdown() -> None:
    timing = _external_timing(
        read_bytes=128,
        write_bytes=64,
        payload_read_bytes=100,
        payload_write_bytes=32,
    )
    timing["hbm_traffic_breakdown"]["by_role"]["weight"][
        "physical_read_bytes"
    ] = 64
    with pytest.raises(ValueError, match="does not sum"):
        estimate_external_hbm_power(
            {
                "HBM_CAPACITY_BYTES": 80_000_000_000,
                "HBM_BANDWIDTH_GBPS": 2039,
            },
            timing,
        )


def test_system_power_uses_roofline_window_and_sums_components() -> None:
    config = {
        **_config(),
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    }
    timing = {
        **_external_timing(
            read_bytes=256,
            write_bytes=64,
            payload_read_bytes=192,
            payload_write_bytes=32,
            runtime_ns=200.0,
        ),
        "compute_pipeline_makespan_cycles": 100,
        "hbm_opcode_latency_ns": {"H_PREFETCH_V": 8.0},
        "hbm_read_requests": 4,
        "hbm_write_requests": 1,
    }
    report = estimate_system_power(config, _trace(), timing)

    assert report["makespan_ns"] == pytest.approx(200.0)
    assert report["makespan_source"] == "system:roofline_latency_ns"
    assert report["system_energy_mj"] == pytest.approx(
        report["onchip_energy_mj"] + report["external_hbm_energy_mj"]
    )
    assert report["hbm_physical_read_bytes"] == 256
    assert report["hbm_payload_write_bytes"] == 32
    assert (
        report["external_memory_configuration_semantics"]
        == "abstract_80gb_a100_aligned"
    )
    assert report["external_hbm_capacity_bytes"] == 80_000_000_000
    assert report["physical_to_payload_traffic_ratio"] == pytest.approx(
        320 / 224
    )
    assert report["external_hbm_energy_by_role"]["weight"][
        "physical_read_bytes"
    ] == 256
    assert (
        report["system_energy_p10_mj"]
        <= report["system_energy_p50_mj"]
        <= report["system_energy_p90_mj"]
    )


def _multi_chip_timing() -> dict[str, object]:
    timing = _external_timing(
        read_bytes=256,
        write_bytes=64,
        payload_read_bytes=192,
        payload_write_bytes=32,
        runtime_ns=200.0,
    )
    timing.update(
        {
            "compute_pipeline_makespan_cycles": 200,
            "stage_compute_latency_ns": {"layer/projection": 200.0},
            "hbm_stage_latency_ns": {"layer/projection": 160.0},
            "stage_roofline_latency_ns": {"layer/projection": 200.0},
            "category_latency_ns": {
                "matrix_compute": 100.0,
                "vector_compute": 60.0,
                "scalar_compute": 30.0,
                "control": 10.0,
            },
            "compatibility": {
                "theoretical_floor_ns": 120.0,
                "stage_theoretical_floor_ns": {
                    "layer/projection": 120.0
                },
            },
            "hbm_opcode_latency_ns": {"H_PREFETCH_V": 160.0},
        }
    )
    return timing


def test_multi_chip_energy_n1_reproduces_system_estimator() -> None:
    config = {
        **_config(),
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    }
    timing = _multi_chip_timing()
    multi = estimate_multi_chip_latency(
        timing,
        {
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
        chip_count=1,
        reference_a100_count=1,
        parallel_model="tp-sp",
        aggregate_hbm_bandwidth_gbps=2039.0,
        aggregate_hbm_capacity_bytes=80_000_000_000,
        seq_len=8,
        batch_size=1,
        fp_width_bits=12,
    )
    expected = estimate_system_power(
        config,
        _trace(),
        timing,
        clock_gating_mode="ideal_hierarchical",
    )
    actual = estimate_multi_chip_system_power(
        config,
        _trace(),
        timing,
        multi,
        chip_count=1,
        parallel_model="tp-sp",
        clock_gating_mode="ideal_hierarchical",
    )
    assert actual["onchip_energy_mj"] == pytest.approx(
        expected["onchip_energy_mj"]
    )
    assert actual["external_hbm_energy_mj"] == pytest.approx(
        expected["external_hbm_energy_mj"]
    )
    assert actual["interconnect_dynamic_energy_mj"] == 0.0
    assert actual["system_energy_mj"] == pytest.approx(
        expected["system_energy_mj"]
    )


def test_tile_aware_energy_actions_use_kernel_lineage_not_stage_average() -> None:
    trace = {
        "schema_version": 7,
        "energy_actions": [
            EnergyAction(
                "layer/attention",
                "scalar",
                "fp_sfu_exp",
                100,
                precision="S_EXP_FP",
                parallel_kernel="softmax-lineage",
            ).to_dict(),
            EnergyAction(
                "layer/attention",
                "scalar",
                "fp_sfu_exp",
                100,
                precision="S_EXP_FP",
                parallel_kernel="norm-lineage",
            ).to_dict(),
        ],
    }
    multi = {
        "multi_chip_model": "tile-aware-tp-cp-ep-v3",
        "parallel_action_scales_by_kernel_opcode": {
            "layer/attention::softmax-lineage::S_EXP_FP": 0.25,
            "layer/attention::norm-lineage::S_EXP_FP": 1.0,
        },
        "parallel_action_scales_by_kernel": {},
        # A stage average would incorrectly make both records 0.625.
        "parallel_action_scales_by_stage_opcode": {
            "layer/attention::S_EXP_FP": 0.625,
        },
        "parallel_action_scales_by_stage": {
            "layer/attention": 0.625,
        },
    }
    scaled = _scaled_energy_actions(
        trace,
        original_traffic={},
        per_chip_traffic={},
        chip_count=4,
        parallel_model="tp-sp",
        multi_chip_report=multi,
    )
    assert [action["count"] for action in scaled["energy_actions"]] == [
        25.0,
        100.0,
    ]


def test_tile_aware_dma_energy_uses_opcode_role_not_stage_average() -> None:
    trace = {
        "schema_version": 7,
        "energy_actions": [
            EnergyAction(
                "layer/attention",
                "hbm_controller",
                "matrix_prefetch",
                100,
                precision="weight",
            ).to_dict(),
            EnergyAction(
                "layer/attention",
                "hbm_controller",
                "vector_prefetch",
                100,
                precision="activation",
            ).to_dict(),
            EnergyAction(
                "layer/attention",
                "matrix_sram",
                "write",
                100,
                precision="weight",
            ).to_dict(),
        ],
    }

    def bucket(value: float) -> dict[str, float]:
        return {
            "physical_read_bytes": value,
            "physical_write_bytes": 0.0,
        }

    original = {
        "by_stage": {"layer/attention": bucket(200.0)},
        "by_stage_opcode_role": {
            "layer/attention::H_PREFETCH_M::weight": bucket(100.0),
            "layer/attention::H_PREFETCH_V::activation": bucket(100.0),
        },
    }
    local = {
        "by_stage": {"layer/attention": bucket(125.0)},
        "by_stage_opcode_role": {
            "layer/attention::H_PREFETCH_M::weight": bucket(25.0),
            "layer/attention::H_PREFETCH_V::activation": bucket(100.0),
        },
    }
    scaled = _scaled_energy_actions(
        trace,
        original_traffic=original,
        per_chip_traffic=local,
        chip_count=4,
        parallel_model="tp-sp",
        multi_chip_report={
            "multi_chip_model": "tile-aware-tp-cp-ep-v3",
        },
    )
    assert [action["count"] for action in scaled["energy_actions"]] == [
        25.0,
        100.0,
        25.0,
    ]


def test_tile_aware_energy_rejects_unclassified_layer_action() -> None:
    trace = {
        "schema_version": 7,
        "energy_actions": [
            EnergyAction(
                "layer/attention",
                "scalar",
                "fp_sfu_exp",
                1,
                precision="S_EXP_FP",
            ).to_dict()
        ],
    }
    with pytest.raises(ValueError, match="lost parallel-kernel lineage"):
        _scaled_energy_actions(
            trace,
            original_traffic={},
            per_chip_traffic={},
            chip_count=2,
            parallel_model="tp-sp",
            multi_chip_report={
                "multi_chip_model": "tile-aware-tp-cp-ep-v3",
            },
        )


def test_tile_aware_rank_energy_is_summed_after_each_rank_clock_cap() -> None:
    rank_reports = []
    for repeats in (1, 9):
        rank_reports.append(
            estimate_onchip_power(
                _config(),
                _trace(repeats=repeats),
                {
                    "compute_timing_mode": "ideal-ii1",
                    "roofline_latency_ns": 100.0,
                    "hbm_opcode_latency_ns": {"H_PREFETCH_V": 8.0},
                },
                makespan_ns_override=100.0,
                clock_gating_mode="ideal_hierarchical",
            )
        )
    aggregate = _sum_onchip_reports(rank_reports, runtime_ms=0.0001)
    assert aggregate["multi_chip_onchip_aggregation"] == (
        "sum_rank_energy_after_per_rank_clock_cap_v2"
    )
    assert len(aggregate["multi_chip_rank_onchip"]) == 2
    assert aggregate["onchip_energy_mj"] == pytest.approx(
        sum(report["onchip_energy_mj"] for report in rank_reports)
    )
    assert aggregate["clock_energy_mj"] == pytest.approx(
        sum(report["clock_energy_mj"] for report in rank_reports)
    )


def test_multi_chip_hbm_dynamic_is_aggregate_and_link_sensitivity_is_ordered() -> None:
    config = {
        **_config(),
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    }
    timing = _multi_chip_timing()
    model = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
    }
    reports = {}
    for chips in (1, 2):
        multi = estimate_multi_chip_latency(
            timing,
            model,
            chip_count=chips,
            reference_a100_count=1,
            parallel_model="tp-sp",
            aggregate_hbm_bandwidth_gbps=2039.0,
            aggregate_hbm_capacity_bytes=80_000_000_000,
            seq_len=8,
            batch_size=1,
            fp_width_bits=12,
        )
        reports[chips] = estimate_multi_chip_system_power(
            config,
            _trace(),
            timing,
            multi,
            chip_count=chips,
            parallel_model="tp-sp",
            clock_gating_mode="ideal_hierarchical",
        )
    assert reports[2]["hbm_read_energy_mj"] == pytest.approx(
        reports[1]["hbm_read_energy_mj"]
    )
    assert reports[2]["hbm_write_energy_mj"] == pytest.approx(
        reports[1]["hbm_write_energy_mj"]
    )
    assert (
        reports[2]["system_energy_optimistic_c2c_mj"]
        <= reports[2]["system_energy_nominal_mj"]
        <= reports[2]["system_energy_conservative_measured_path_mj"]
    )


def test_tp_only_replicated_local_energy_is_not_below_tp_sp() -> None:
    config = {
        **_config(),
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039,
    }
    timing = _multi_chip_timing()
    model = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
    }
    energy = {}
    for mode in ("tp-sp", "tp-only"):
        multi = estimate_multi_chip_latency(
            timing,
            model,
            chip_count=2,
            reference_a100_count=1,
            parallel_model=mode,
            aggregate_hbm_bandwidth_gbps=2039.0,
            aggregate_hbm_capacity_bytes=80_000_000_000,
            seq_len=8,
            batch_size=1,
            fp_width_bits=12,
        )
        energy[mode] = estimate_multi_chip_system_power(
            config,
            _trace(),
            timing,
            multi,
            chip_count=2,
            parallel_model=mode,
            clock_gating_mode="ideal_hierarchical",
        )["onchip_energy_mj"]
    assert energy["tp-only"] >= energy["tp-sp"]


def test_ideal_hierarchical_clock_is_bounded_and_preserves_ungated_upper_bound() -> None:
    timing = {
        "compute_pipeline_makespan_cycles": 100,
        "hbm_opcode_latency_ns": {"H_PREFETCH_V": 8.0},
    }
    default_ungated = estimate_onchip_power(_config(), _trace(), timing)
    explicit_ungated = estimate_onchip_power(
        _config(), _trace(), timing, clock_gating_mode="ungated"
    )
    ideal = estimate_onchip_power(
        _config(), _trace(), timing, clock_gating_mode="ideal_hierarchical"
    )

    assert default_ungated["onchip_energy_mj"] == pytest.approx(
        explicit_ungated["onchip_energy_mj"]
    )
    assert ideal["clock_energy_mj"] <= ideal["ungated_clock_energy_mj"]
    assert ideal["onchip_energy_mj"] <= ideal["ungated_onchip_energy_mj"]
    assert ideal["clock_gating_status"] == "architectural_ideal_assumption"
    assert ideal["rtl_clock_gating_implemented"] is False
    assert ideal["gating_overhead_included"] is False
    assert ideal["idle_clock_fraction"] == 0.0
    assert ideal["ideal_clock_energy_by_component_pj"]["control"] == 0.0
    assert ideal["ungated_clock_energy_by_component_pj"]["control"] > 0.0
    assert ideal["unmodeled_clock_residual_area_um2"] > 0.0
    assert ideal["clock_work"]["status"] == "complete"
    assert all(
        value <= ideal["makespan_cycles"]
        for value in ideal["clock_work_by_subcomponent"].values()
    )


def test_ideal_ii1_clock_work_uses_one_cycle_for_vector_scalar_control() -> None:
    trace = _trace(repeats=3)
    timing = {
        "compute_timing_mode": "ideal-ii1",
        "hbm_opcode_latency_ns": {"H_PREFETCH_V": 8.0},
    }
    work = build_clock_work(
        [action.to_dict() for action in trace.energy_actions],
        _config(),
        timing,
    )
    records = work["records"]

    for opcode in ("V_ADD_VV", "V_RED_SUM_SEGS", "S_MUL_FP", "C_SET_ADDR_REG"):
        matching = [
            record
            for record in records
            if record["source_opcode"] == opcode
            and record["component"] != "hbm_controller"
        ]
        assert matching, opcode
        assert all(
            "architectural_ideal_ii1" in record["fidelity"]
            for record in matching
        )
        assert all(
            record["component_active_cycles"] <= 3
            for record in matching
        )
    assert work["compute_timing_status"] == "architectural_ideal_assumption"
    assert work["compute_hazards_included"] is False


@pytest.mark.parametrize(
    "configured,active,expected_cycles",
    [(16, 8, 1.5), (64, 64, 3.0)],
)
def test_compact_stats_clock_work_uses_configured_lane_tier(
    configured: int, active: int, expected_cycles: float
) -> None:
    action = EnergyAction(
        stage="layer/attention",
        component="vector",
        action="compact_stats_mul",
        count=3,
        precision="V_STAT_MUL_F",
        variant=f"gp5,gp5,f2,{active}",
        segment_count=active,
        activity_fidelity="exact_compact_lanes",
    )
    work = build_clock_work(
        [action.to_dict()],
        {**_config(), "COMPACT_STATS_LANES": configured},
        {"compute_timing_mode": "ideal-ii1"},
    )
    assert work["status"] == "complete"
    compact = [
        record
        for record in work["records"]
        if record["subcomponent"] == "compact_stats_simd"
    ]
    assert len(compact) == 1
    assert compact[0]["active_instances"] == active
    assert compact[0]["total_instances"] == configured
    assert compact[0]["equivalent_full_area_cycles"] == pytest.approx(
        expected_cycles
    )


def test_compact_stats_power_marks_32_64_lane_activity_extrapolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PLENA_POWER_VECTOR_RTL_V5_DELTA",
        str(tmp_path / "missing-v5.json"),
    )
    _read_vector_rtl_v5_energy.cache_clear()
    action = EnergyAction(
        stage="layer/attention",
        component="vector",
        action="compact_stats_mul",
        count=3,
        precision="V_STAT_MUL_F",
        variant="gp5,gp5,f2,64",
        segment_count=64,
        activity_fidelity="exact_compact_lanes",
    )
    report = estimate_onchip_power(
        {**_config(), "COMPACT_STATS_LANES": 64},
        {"schema_version": 4, "energy_actions": [action.to_dict()]},
        {"compute_pipeline_makespan_cycles": 100},
        clock_gating_mode="ideal_hierarchical",
    )
    assert (
        report["compact_stats_power_calibration_status"]
        == "rtl_activity_per_lane_extrapolation_32_64"
    )
    assert any(
        "dedicated Qwen-like RTL-activity replay" in warning
        for warning in report["warnings"]
    )


def test_rtl_v5_power_overlay_uses_measured_per_lane_energy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = tmp_path / "vector_rtl_v5.json"
    overlay.write_text(
        json.dumps(
            {
                "calibration_status": "rtl_activity_calibrated_rtl_v5_tiers",
                "dynamic_nominal_pj_per_lane_action": {
                    "compact_stats_mul": 2.0
                },
                "activity_envelope": {
                    "compact_stats_mul": {
                        "low": 0.5,
                        "nominal": 1.0,
                        "high": 2.0,
                    }
                },
            }
        )
    )
    monkeypatch.setenv("PLENA_POWER_VECTOR_RTL_V5_DELTA", str(overlay))
    _read_vector_rtl_v5_energy.cache_clear()
    config = {
        **_config(),
        "COMPACT_STATS_LANES": 64,
        "FP_SETTING": "FP_E5M6",
    }
    action = {
        "component": "vector",
        "action": "compact_stats_mul",
        "count": 3,
        "segment_count": 32,
    }
    coefficients = {
        "dynamic_nominal_pj": {"vector": {}},
        "activity_envelope": {},
    }
    widths = {"mode": "mxint", "t": 4, "l": 4, "fp": 12}

    assert _logic_action_energy_v2(
        action, config, coefficients, widths, quantile="nominal"
    ) == pytest.approx(192.0)
    assert _logic_action_energy_v2(
        action, config, coefficients, widths, quantile="high"
    ) == pytest.approx(384.0)


def test_ideal_hierarchical_idle_trace_has_zero_logic_clock_energy() -> None:
    report = estimate_onchip_power(
        _config(),
        {"schema_version": 4, "energy_actions": []},
        {"compute_pipeline_makespan_cycles": 100},
        clock_gating_mode="ideal_hierarchical",
    )
    assert report["clock_energy_mj"] == 0.0
    assert report["logic_dynamic_energy_mj"] == 0.0
    assert report["logic_leakage_energy_mj"] > 0.0
    assert report["ungated_clock_energy_mj"] > 0.0


def test_clock_baseline_does_not_receive_activity_uncertainty() -> None:
    report = estimate_onchip_power(
        _config(),
        {"schema_version": 4, "energy_actions": []},
        {"compute_pipeline_makespan_cycles": 100},
        clock_gating_mode="ungated",
    )
    assert report["onchip_energy_p10_mj"] == pytest.approx(
        report["onchip_energy_p50_mj"]
    )
    assert report["onchip_energy_p90_mj"] == pytest.approx(
        report["onchip_energy_p50_mj"]
    )


def test_ideal_hierarchical_unknown_mask_fails_closed() -> None:
    trace = {
        "schema_version": 4,
        "energy_actions": [
            EnergyAction(
                "stage",
                "vector",
                "lane_multiply_vf",
                1,
                precision="V_MUL_VF",
                variant="gp1,gp1,f1,1",
                activity_fidelity="clock_work_unavailable",
            ).to_dict()
        ],
    }
    with pytest.raises(ValueError, match="clock work is unavailable"):
        estimate_onchip_power(
            _config(),
            trace,
            {"compute_pipeline_makespan_cycles": 100},
            clock_gating_mode="ideal_hierarchical",
        )


def test_ideal_hierarchical_missing_hbm_service_window_fails_closed() -> None:
    with pytest.raises(ValueError, match="clock work is unavailable"):
        estimate_onchip_power(
            _config(),
            _trace(),
            {"compute_pipeline_makespan_cycles": 100},
            clock_gating_mode="ideal_hierarchical",
        )


def test_default_power_artifact_is_promoted_v2_shadow() -> None:
    coefficients = _load_logic_coefficients(None)
    assert coefficients["model"] == "onchip_action_energy_v2"
    assert (
        coefficients["calibration_status"]
        == "rtl_activity_calibrated_candidate_v2"
    )
    assert coefficients["gate_level_validation"] == "not_run_by_scope"


def test_more_actions_do_not_reduce_dynamic_energy() -> None:
    timing = {"compute_pipeline_makespan_cycles": 1000}
    one = estimate_onchip_power(_config(), _trace(1), timing)
    four = estimate_onchip_power(_config(), _trace(4), timing)
    assert four["logic_dynamic_energy_mj"] >= one["logic_dynamic_energy_mj"]
    assert four["sram_dynamic_energy_mj"] >= one["sram_dynamic_energy_mj"]


def test_sram_dynamic_energy_is_independent_of_port_area_assumption() -> None:
    timing = {"compute_pipeline_makespan_cycles": 1000}
    replicated = estimate_onchip_power(_config(), _trace(), timing)
    ideal = estimate_onchip_power(
        {**_config(), "SRAM_PORT_MODEL": "ideal-dual-port"},
        _trace(),
        timing,
    )
    assert (
        ideal["sram_dynamic_energy_mj"]
        == replicated["sram_dynamic_energy_mj"]
    )
    assert (
        ideal["sram_access_metadata"]["sram_port_energy_model"]
        == "ideal_independent_access"
    )
    assert (
        ideal["sram_access_metadata"]["dual_port_overhead_included"] is False
    )


def test_decoder_layer_scaling_preserves_and_scales_energy_actions() -> None:
    one_layer = _trace()
    one_layer.energy_actions = [
        replace(
            action,
            stage=("layer/test" if action.component != "control" else "global/setup"),
        )
        for action in one_layer.energy_actions
    ]
    scaled = _scale_trace(one_layer, 4)
    expected = 0
    for action in one_layer.energy_actions:
        expected += action.count * (4 if action.stage.startswith("layer/") else 1)
    assert sum(action.count for action in scaled.energy_actions) == expected
    assert sum(
        action.count for action in scaled.stages["global/setup"].energy_actions
    ) == sum(
        action.count
        for action in one_layer.energy_actions
        if action.stage == "global/setup"
    )


def test_non_1ghz_is_reported_out_of_domain_without_scaling() -> None:
    config = _config()
    config["CLOCK_PERIOD_PS"] = 500
    report = estimate_onchip_power(config, _trace(), {"compute_pipeline_makespan_cycles": 100})
    assert any("outside the fixed 1000 ps calibration" in warning for warning in report["warnings"])


def test_asap7_sram_liberty_energy_and_catalog() -> None:
    library = ROOT / "Workspace/external/asap7_sram_0p0/generated/LIB/srambank_64x4x16_6t122.lib"
    parsed = parse_asap7_sram_lib(library)
    assert parsed["depth"] == 256
    assert parsed["width"] == 16
    assert parsed["read_energy_pj"] == pytest.approx(1.815855)
    assert parsed["write_energy_pj"] == pytest.approx(0.186927)
    assert parsed["sram_leakage_status"] == "unavailable"
    catalog = build_sram_energy_catalog()
    assert catalog["macro_count"] == 36
    assert all(row["read_energy_pj"] > 0 for row in catalog["macros"])
    assert all(row["write_energy_pj"] > 0 for row in catalog["macros"])


def test_calibration_plan_maps_exactly_sixteen_netlists_once() -> None:
    points = build_plan()
    assert len(points) == 16
    assert len({point.point_key for point in points}) == 16
    assert {point.component for point in points} == {"matrix", "vector", "scalar", "hbm", "control"}
    assert len(SCENARIOS) == 9
    assert sum(point.component == "matrix" for point in points) == 7
    assert sum(point.component == "vector" for point in points) == 4
    assert sum(point.component == "scalar" for point in points) == 2
    assert sum(point.component == "hbm" for point in points) == 2
    assert sum(point.component == "control" for point in points) == 1


def test_v2_plan_adds_fifteen_configs_and_family_microkernels() -> None:
    v1 = build_plan()
    v2 = build_plan_v2()
    assert len(v2) == 31
    assert len({point.point_key for point in v2} - {point.point_key for point in v1}) == 15
    vector = next(point for point in v2 if point.point_id == "power_vector_v32_e6m5")
    vector_scenarios = scenarios_for_point_v2(vector)
    microkernels = {scenario[3] for scenario in vector_scenarios}
    assert {
        "add_vv", "mul_vseg", "reduce_sum_seg", "reduce_max_seg",
        "reduce_sum_segs", "lane_load", "lane_store",
    } <= microkernels
    assert {scenario[2] for scenario in vector_scenarios if scenario[3] == "add_vv"} == {32, 128, 512}


def test_compact_v5_power_plan_is_focused_and_tiered() -> None:
    points = build_compact_v5_plan()
    scenarios = scenarios_for_compact_v5()

    assert [point.params["COMPACT_STATS_LANES"] for point in points] == [32, 64]
    assert all(point.params["VLEN"] == 64 for point in points)
    assert len({point.point_key for point in points}) == 2
    assert {scenario[3] for scenario in scenarios} == {
        "idle",
        "compact_stats_mul",
        "compact_stats_add",
        "compact_stats_rsqrt",
    }
    assert {
        scenario[2]
        for scenario in scenarios
        if scenario[3] == "compact_stats_mul"
        and scenario[1] == "representative-qwen"
    } == {32, 128, 512}


def test_agu_power_plan_is_small_and_identifiable() -> None:
    point = build_agu_plan()[0]
    scenarios = scenarios_for_point_v2(point)
    names = {scenario[0] for scenario in scenarios}

    assert point.component == "agu"
    assert point.top_module == "loop_agu_state"
    assert len(scenarios) == 13
    assert {"idle_32", "idle_128", "idle_512"} <= names
    assert {
        "qwen_boundary_1_128",
        "qwen_boundary_3_128",
        "qwen_boundary_6_128",
        "qwen_offset_read_128",
        "qwen_setup_1_128",
        "qwen_setup_6_128",
    } <= names


def test_v2_qwen_mix_is_costtrace_derived_and_projects_exactly() -> None:
    artifact = json.loads(
        (ROOT / "analytic_models/power/calibration/qwen3_32b_action_mix_v2.json").read_text()
    )
    assert artifact["workload"]["seq_len"] == 482
    assert artifact["workload"]["batch_size"] == 16
    assert artifact["schedule"]["vector_scalar_schedule"] == "rtl-v3"
    assert artifact["schedule"]["gqa_pipeline_schedule"] == "row-interleaved-v1"
    weights = artifact["components"]["vector"]["microkernel_weights"]
    first = weighted_microkernel_schedule(weights, 128)
    second = weighted_microkernel_schedule(weights, 128)
    assert first == second
    assert len(first) == 128
    assert set(first) <= set(weights)
    assert {"reduce_sum_seg", "reduce_max_seg", "lane_load", "lane_store"} <= set(first)


def test_resume_replays_only_stale_qwen_mix_semantics() -> None:
    scenario = ("qwen_mix_128", "representative-qwen", 128, "mixed")
    stale = {
        "status": "complete",
        "features_json": json.dumps({"qwen_mix_semantic_hash": "handwritten-v1"}),
    }
    current = {
        "status": "complete",
        "features_json": json.dumps(
            {"qwen_mix_semantic_hash": qwen_mix_semantic_hash()}
        ),
    }
    assert not _resume_row_is_current(stale, scenario)
    assert _resume_row_is_current(current, scenario)
    assert _resume_row_is_current(
        {"status": "complete"},
        ("qwen_add_vv_128", "representative-qwen", 128, "add_vv"),
    )


def test_v2_slope_and_activity_envelope_use_qwen_as_nominal() -> None:
    slope, intercept, r2 = _linear_slope([(32, 69.0), (128, 261.0), (512, 1029.0)])
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(5.0)
    assert r2 == pytest.approx(1.0)
    base = {
        "point_id": "v", "point_key": "v", "component": "vector",
        "holdout": False, "microkernel": "add_vv", "params": {"VLEN": 32},
        "sample_count": 3,
    }
    envelope, _ = _activity_envelope(
        [
            {**base, "pattern": "representative-qwen", "energy_per_action_pj": 2.0},
            {**base, "pattern": "low-toggle", "energy_per_action_pj": 1.0},
            {**base, "pattern": "random", "energy_per_action_pj": 3.0},
        ]
    )
    assert envelope["vector.lane_add_sub_vv"] == pytest.approx(
        {"low": 0.5, "nominal": 1.0, "high": 1.5}
    )


def test_power_group_parser_normalizes_units_and_separates_clock() -> None:
    report = """
Power Group      Power            Power               Power              Power   (   %    )  Attrs
--------------------------------------------------------------------------------------------------
clock_network     15.2777            0.0000            0.0000           15.2777  (  99.86%)  i
register       6.3505e-03        1.4453e-03        2.6737e+06        1.0479e-02  (   0.07%)
combinational  1.4146e-03        1.7934e-03        7.3103e+06        1.0518e-02  (   0.07%)
--------------------------------------------------------------------------------------------------
Total             15.2854 mW     3.2387e-03 mW     9.9840e+06 pW        15.2986 mW
"""
    groups = _parse_power_group_table(report)
    assert groups["register"]["leakage_power_mw"] == pytest.approx(0.0026737)
    fields = _power_group_fields(report, 100.0)
    assert fields["clock_network_energy_pj"] == pytest.approx(1527.77)
    assert fields["nonclock_dynamic_power_mw"] == pytest.approx(
        0.0063505 + 0.0014453 + 0.0014146 + 0.0017934
    )


def test_v2_idle_pair_uses_nonclock_residual_not_raw_total_delta() -> None:
    idle = {
        "point_id": "v", "point_key": "v", "scenario": "idle_128",
        "pattern": "idle", "repeat_count": 128, "window_cycles": 128,
        "window_energy_pj": 62640.128, "clock_network_energy_pj": 62600.0,
        "nonclock_dynamic_energy_pj": 40.128,
    }
    active = {
        "point_id": "v", "point_key": "v", "scenario": "qwen_mix_128",
        "pattern": "representative-qwen", "repeat_count": 128,
        "window_cycles": 128, "window_energy_pj": 62469.3248,
        "clock_network_energy_pj": 60400.0,
        "nonclock_dynamic_energy_pj": 2069.3248,
    }
    _pair_idle([idle, active])
    assert active["window_energy_pj"] < idle["window_energy_pj"]
    assert active["incremental_energy_pj"] == pytest.approx(2029.1968)
    assert active["normalized_dynamic_energy_pj"] == pytest.approx(64669.3248)
    assert active["excluded_clock_delta_pj"] == pytest.approx(-2200.0)
    assert active["nonclock_residual_positive"] is True


def test_v2_matrix_reduce_is_zero_for_single_split_and_monotonic() -> None:
    coefficients = {
        "dynamic_nominal_pj": {
            "matrix": {
                "mxint": {
                    "pe_cycle": {"base": 0.1, "bit_product": 0.01, "width_sum": 0.02},
                    "reduce_node_bit": 0.03,
                    "output_bit": 0.04,
                },
                "mxfp": {
                    "pe_cycle": {"base": 0.2, "bit_product": 0.02, "width_sum": 0.03},
                    "reduce_node_bit": 0.04,
                    "output_bit": 0.05,
                },
            }
        },
        "activity_envelope": {},
    }
    config = {"MLEN": 16, "BLEN": 16, "VLEN": 16, "INT_DATA_WIDTH": 32}
    widths = {"mode": "mxint", "t": 4, "l": 4, "fp": 12}
    action = {"component": "matrix", "action": "cross_k_reduce", "count": 1}
    assert _logic_action_energy_v2(action, config, coefficients, widths, quantile="nominal") == 0.0
    assert all(_matrix_invariants({"dynamic_nominal_pj": coefficients["dynamic_nominal_pj"]}).values())


def test_single_split_cross_k_reduce_is_structurally_zero_not_unknown() -> None:
    config = {**_config(), "MLEN": 16, "VLEN": 16, "BLEN": 16}
    report = estimate_onchip_power(
        config,
        {
            "schema_version": 4,
            "energy_actions": [
                EnergyAction(
                    "layer/ffn",
                    "matrix",
                    "cross_k_reduce",
                    7,
                    precision="M_MM",
                ).to_dict()
            ],
        },
        {"compute_pipeline_makespan_cycles": 100},
    )

    coverage = report["calibration_coverage"]
    assert coverage["unknown_actions"] == {}
    assert coverage["structurally_zero_actions"] == {
        "matrix.cross_k_reduce": 7.0
    }
    assert coverage["structural_physical_instances"] == {
        "matrix.cross_k_reduce": 0
    }
    assert report["component_logic_dynamic_energy_pj"]["matrix"] == 0.0
    assert not any(
        "lack energy coefficients" in warning for warning in report["warnings"]
    )


def test_multi_split_cross_k_reduce_with_zero_coefficient_remains_unknown(
    tmp_path: Path,
) -> None:
    artifact = json.loads(DEFAULT_LOGIC_ENERGY.read_text())
    artifact["dynamic_nominal_pj"]["matrix"]["mxint"]["reduce_node_bit"] = 0.0
    calibration = tmp_path / "zero_reduce_coefficient.json"
    calibration.write_text(json.dumps(artifact))
    config = {**_config(), "MLEN": 16, "VLEN": 16, "BLEN": 4}

    report = estimate_onchip_power(
        config,
        {
            "schema_version": 4,
            "energy_actions": [
                EnergyAction(
                    "layer/ffn",
                    "matrix",
                    "cross_k_reduce",
                    7,
                    precision="M_MM",
                ).to_dict()
            ],
        },
        {"compute_pipeline_makespan_cycles": 100},
        logic_coefficients_path=calibration,
    )

    coverage = report["calibration_coverage"]
    assert coverage["structurally_zero_actions"] == {}
    assert coverage["unknown_actions"] == {"matrix.cross_k_reduce": 7.0}
    assert any(
        "lack energy coefficients" in warning for warning in report["warnings"]
    )


def test_rtl_v4_power_overlay_maps_overwrite_opcode_and_compact_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "vector_rtl_v4.json"
    overlay.write_text(
        json.dumps(
            {
                "calibration_status": "rtl_activity_calibrated_rtl_v4_delta",
                "dynamic_nominal_pj": {"compact_stats_mul": 16.0},
                "activity_envelope": {
                    "compact_stats_mul": {
                        "low": 0.5,
                        "nominal": 1.0,
                        "high": 2.0,
                    }
                },
                "reduction_overwrite_delta_pj": {
                    "reduce_sum_ovr": 7.0,
                    "reduce_sum_seg_ovr": 11.0,
                },
            }
        )
    )
    monkeypatch.setenv("PLENA_POWER_VECTOR_RTL_V4_DELTA", str(overlay))
    _read_vector_rtl_v4_energy.cache_clear()
    coefficients = {
        "dynamic_nominal_pj": {
            "vector": {
                "reduction_sum_full": {"FP_E5M6": 1.0},
                "reduction_sum_segment": {"FP_E5M6": 1.0},
            }
        },
        "activity_envelope": {},
    }
    config = {"MLEN": 16, "BLEN": 4, "VLEN": 16, "FP_SETTING": "FP_E5M6"}
    widths = {"mode": "mxint", "t": 4, "l": 4, "fp": 12}

    compact = {
        "component": "vector",
        "action": "compact_stats_mul",
        "count": 2,
        "active_lanes": 4,
    }
    assert _logic_action_energy_v2(
        compact, config, coefficients, widths, quantile="nominal"
    ) == pytest.approx(8.0)
    assert _logic_action_energy_v2(
        compact, config, coefficients, widths, quantile="high"
    ) == pytest.approx(16.0)

    full = {
        "component": "vector",
        "action": "reduction_sum_full",
        "count": 2,
        "active_lanes": 16,
        "precision": "V_RED_SUM_OVR",
    }
    # Base reduction energy is 2 * (VLEN - 1) * log2(VLEN) = 120 pJ;
    # the paired overwrite overlay contributes 2 * 7 pJ.
    assert _logic_action_energy_v2(
        full, config, coefficients, widths, quantile="nominal"
    ) == pytest.approx(134.0)

    segmented = {
        "component": "vector",
        "action": "reduction_sum_segment",
        "count": 1,
        "precision": "V_RED_SUM_SEG_OVR",
    }
    assert _logic_action_energy_v2(
        segmented, config, coefficients, widths, quantile="nominal"
    ) == pytest.approx(64.0 + 11.0)


def test_installed_rtl_v4_power_overlay_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLENA_POWER_VECTOR_RTL_V4_DELTA", raising=False)
    _read_vector_rtl_v4_energy.cache_clear()
    overlay = _read_vector_rtl_v4_energy(
        str(
            (
                Path(__file__).with_name("calibration")
                / "vector_rtl_v4_power_delta.json"
            ).resolve()
        )
    )

    assert (
        overlay["calibration_status"]
        == "rtl_activity_calibrated_rtl_v4_delta"
    )
    assert all(
        float(overlay["dynamic_nominal_pj"][family]) > 0.0
        for family in (
            "compact_stats_mul",
            "compact_stats_add",
            "compact_stats_rsqrt",
        )
    )
    _read_vector_rtl_v4_energy.cache_clear()


def test_rtl_v4_delta_fit_excludes_clock_network_drift() -> None:
    energy, basis = _action_dynamic_energy(
        {
            "window_dynamic_energy_pj": "1000.0",
            "nonclock_dynamic_energy_pj": "12.5",
        }
    )
    assert energy == pytest.approx(12.5)
    assert basis == "nonclock_dynamic_energy_pj"
    _read_vector_rtl_v4_energy.cache_clear()


def test_nonnegative_calibration_fit_recovers_action_slopes() -> None:
    rows = [
        {
            "point_id": "a",
            "scenario": "random",
            "activity_level": "rtl",
            "dynamic_features": {"a": 1.0, "b": 0.0},
            "target": 2.0,
        },
        {
            "point_id": "b",
            "scenario": "random",
            "activity_level": "rtl",
            "dynamic_features": {"a": 0.0, "b": 1.0},
            "target": 3.0,
        },
        {
            "point_id": "ab",
            "scenario": "mixed",
            "activity_level": "gate",
            "dynamic_features": {"a": 2.0, "b": 4.0},
            "target": 16.0,
        },
    ]
    names, coefficients, diagnostics = _fit(
        rows, field="dynamic_features", target="target"
    )
    errors = [
        abs(
            sum(row["dynamic_features"].get(name, 0.0) * value for name, value in zip(names, coefficients))
            - row["target"]
        )
        for row in rows
    ]
    assert dict(zip(names, coefficients)) == pytest.approx({"a": 2.0, "b": 3.0})
    assert max(errors) < 1e-5
    assert diagnostics["rank"] == 2


def test_compact_power_csv_keeps_only_latest_complete_attempt(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    rows = [
        {"point_key": "a", "scenario": "random_32", "status": "complete", "point_id": "old"},
        {"point_key": "a", "scenario": "random_32", "status": "complete", "point_id": "new"},
        {"point_key": "b", "scenario": "random_32", "status": "complete", "point_id": "done"},
        {"point_key": "b", "scenario": "random_32", "status": "failed", "point_id": "retry_failed"},
    ]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POWER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "compact.csv"
    assert _export_latest_complete(source, output) == 1
    compact = list(csv.DictReader(output.open()))
    assert [(row["point_key"], row["point_id"]) for row in compact] == [("a", "new")]


def test_power_mapping_flow_enables_saif_name_tracking_idempotently(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    synth = worker / "tools/synopsys/synth.tcl"
    synth.parent.mkdir(parents=True)
    synth.write_text(
        'puts "\\n>>> Reading RTL files..."\n'
        '#------------------------------\n# Save mapped design\n'
    )
    _patch_power_synthesis_flow(worker)
    first = synth.read_text()
    _patch_power_synthesis_flow(worker)
    second = synth.read_text()
    assert first == second
    assert first.index("saif_map -start") < first.index(">>> Reading RTL files")
    assert "saif_map -write_map ${out_dir}/${MODULE}_saif.namemap" in first


def test_activity_worker_retains_supported_packed_struct_trace(tmp_path: Path) -> None:
    from analytic_models.power.scripts.rtl_activity import _patch_worker_trace_mode

    runner = tmp_path / "tools/cfl_cocotb/runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        '*(["--trace", "--trace-structs"] if trace else []),\n'
        'plusargs=["--trace", "--trace-structs"] if trace else [],\n'
    )
    before = runner.read_text()
    _patch_worker_trace_mode(tmp_path)
    assert runner.read_text() == before


def test_saif_map_sequential_coverage_parser(tmp_path: Path) -> None:
    report = tmp_path / "saif_map.rpt"
    report.write_text(
        "Object type  Auto Set      Auto Set      User Set      User Set       Total\n"
        "Seq Cells   80(80.00%)    5(5.00%)      3(3.00%)     2(2.00%)       100\n"
    )
    mapped, total, coverage = _parse_saif_map_seq_coverage(report)
    assert (mapped, total) == (90, 100)
    assert coverage == pytest.approx(90.0)


def test_verilator_saif_name_map_translates_packed_members(tmp_path: Path) -> None:
    source = tmp_path / "dc.namemap"
    source.write_text(
        "SAIF name mapping file\n"
        "port decode_stage_op[128]\n"
        "oname Bs:decode_stage_op Bd:m_op Bd:3 L:decode_stage_op[m_op][3]\n"
        "sname - decode_stage_op[m_op][3]\n"
        "cell delayed_reg_rd_stage_op_reg_gp_rd__0_\n"
        "oname B:x Bs:delayed_reg_rd_stage_op_reg Bd:gp_rd Bd:0 "
        "L:delayed_reg_rd_stage_op_reg[gp_rd][0]\n"
        "sname - delayed_reg_rd_stage_op[gp_rd][0]\n"
        "port gp_addr_1[0]\n"
        "oname Bs:gp_addr_1 Bd:0 L:gp_addr_1[0]\n"
        "sname - gp_addr_1[0]\n"
    )
    output = tmp_path / "verilator.namemap"
    assert _write_verilator_saif_name_map(source, output) == 2
    text = output.read_text()
    assert "sname - decode_stage_op/m_op[3]" in text
    assert "sname - delayed_reg_rd_stage_op/gp_rd[0]" in text
    assert "sname - gp_addr_1[0]" in text


def test_packed_saif_members_generate_exact_flat_port_overrides(tmp_path: Path) -> None:
    saif = tmp_path / "activity.saif.gz"
    with gzip.open(saif, "wt") as handle:
        handle.write(
            "(SAIFILE\n"
            "(INSTANCE power_activity_tb\n"
            "  (INSTANCE dut\n"
            "    (INSTANCE decode_stage_op\n"
            "      (NET\n"
            "        (m_op\\[3\\]\n"
            "          (T0 750) (T1 250) (TX 0)\n"
            "          (TC 20) (IG 0)\n"
            "        )\n"
            "      )\n"
            "    )\n"
            "  )\n"
            ")\n"
            ")\n"
        )
    name_map = tmp_path / "map.namemap"
    name_map.write_text(
        "port decode_stage_op[128]\n"
        "oname L:decode_stage_op[m_op][3]\n"
        "sname - decode_stage_op/m_op[3]\n"
        "port direct[0]\n"
        "oname L:direct[0]\n"
        "sname - direct[0]\n"
    )
    values = _saif_signal_activity(saif, instance_path="power_activity_tb/dut")
    assert values["decode_stage_op/m_op[3]"] == (750, 250, 20)
    tcl, count = _packed_port_activity_tcl(
        name_map=name_map,
        saif=saif,
        instance_path="power_activity_tb/dut",
        window_ns=100.0,
    )
    assert count == 1
    assert "-static_probability 0.25" in tcl
    assert "-toggle_rate 0.2" in tcl
    assert "[get_ports {decode_stage_op[128]}]" in tcl


def test_dcsh_1_feature_checkout_failure_is_retryable() -> None:
    assert is_dc_license_unavailable_text(
        "Fatal: Design Compiler is not enabled. (DCSH-1)"
    )


def test_dc_retry_releases_local_license_slot_while_waiting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import analytic_models.power.scripts.run_rtl_activity_power_calibration as runtime

    attempts = iter(
        (
            subprocess.CompletedProcess(["dc_shell"], 1, "SEC-50", ""),
            subprocess.CompletedProcess(["dc_shell"], 0, "complete", ""),
        )
    )
    semaphore = threading.Semaphore(1)
    monkeypatch.setattr(runtime, "_run_tracked_process", lambda *args, **kwargs: next(attempts))

    def wait_without_owning_slot(_seconds: float) -> None:
        assert semaphore.acquire(blocking=False)
        semaphore.release()

    monkeypatch.setattr(runtime.time, "sleep", wait_without_owning_slot)
    result = _run_dc_retry(
        ["dc_shell"], cwd=tmp_path, log=tmp_path / "dc.log",
        wait_sec=1.0, max_retries=2, license_sem=semaphore,
    )
    assert result.returncode == 0
    assert (tmp_path / "dc.attempt_1.stdout.log").read_text() == "SEC-50"
    assert (tmp_path / "dc.attempt_2.stdout.log").read_text() == "complete"


def test_runner_rejects_concurrent_run_or_worker_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    worker_root = tmp_path / "workers"
    first = _acquire_runner_locks(run_dir, worker_root)
    try:
        with pytest.raises(RuntimeError, match="lock is already held"):
            _acquire_runner_locks(run_dir, worker_root)
        with pytest.raises(RuntimeError, match="lock is already held"):
            _acquire_runner_locks(tmp_path / "other-run", worker_root)
    finally:
        _release_runner_locks(first)

    resumed = _acquire_runner_locks(run_dir, worker_root)
    _release_runner_locks(resumed)


def test_resource_gate_preserves_live_memory_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    import analytic_models.power.scripts.run_rtl_activity_power_calibration as runtime

    available = {"gib": 40.0}
    monkeypatch.setattr(runtime, "_mem_available_gib", lambda: available["gib"])
    gate = ResourceGate(memory_reserve_gib=24.0, tmp_reserve_gib=15.0)
    available["gib"] = 26.0  # Less than the 2.5 GiB replay token above reserve.
    acquired = threading.Event()

    def acquire() -> None:
        token = gate.acquire("power")
        acquired.set()
        gate.release("power", token)

    worker = threading.Thread(target=acquire, daemon=True)
    worker.start()
    assert not acquired.wait(timeout=0.1)
    available["gib"] = 30.0
    with gate.condition:
        gate.condition.notify_all()
    assert acquired.wait(timeout=1.0)
    worker.join(timeout=1.0)


def test_resource_gate_expands_after_external_memory_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import analytic_models.power.scripts.run_rtl_activity_power_calibration as runtime

    available = {"gib": 40.0}
    monkeypatch.setattr(runtime, "_mem_available_gib", lambda: available["gib"])
    gate = ResourceGate(memory_reserve_gib=24.0, tmp_reserve_gib=15.0)
    first = gate.acquire("map_heavy")
    acquired = threading.Event()

    def acquire_second() -> None:
        token = gate.acquire("map_heavy")
        acquired.set()
        gate.release("map_heavy", token)

    worker = threading.Thread(target=acquire_second, daemon=True)
    worker.start()
    assert not acquired.wait(timeout=0.1)
    gate.release("map_heavy", first)
    available["gib"] = 72.0
    with gate.condition:
        gate.condition.notify_all()
    assert acquired.wait(timeout=1.0)
    worker.join(timeout=1.0)


def test_tmp_cleanup_manifest_records_pre_state_and_open_file_state(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "area_new_power_old"
    candidate.mkdir()
    (candidate / "payload").write_bytes(b"x" * 4096)
    unknown = tmp_path / "unknown_data"
    unknown.mkdir()
    manifest = tmp_path / "manifest.json"

    result = cleanup_power_tmp(
        tmp_root=tmp_path, manifest=manifest, apply=True, min_age_hours=0.0
    )
    assert not candidate.exists()
    assert unknown.exists()
    row = next(item for item in result["records"] if item["path"] == str(candidate))
    assert row["action"] == "deleted"
    assert row["open_file_state"] == "not_referenced"
    assert result["disk_before"]["free_bytes"] <= result["disk_after"]["free_bytes"]
