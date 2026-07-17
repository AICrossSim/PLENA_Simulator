#!/usr/bin/env python3
"""Cycle-level hazard tests for the production pipeline controller."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH


STALL_M = 0
STALL_V_ELEMENT = 0
STALL_V_REDUCTION = 0
STALL_S_FP = 0
STALL_C = 0
STALL_H = 0

MM_IC = 5
ADD_V = 1
MUL_V = 3
SUM_V = 1
RECI_FP = 6
LD_REG_FP = 8
LD_OUT_FP = 9


def clear_decode(dut):
    dut.decode_m_op.value = STALL_M
    dut.decode_v_element_op.value = STALL_V_ELEMENT
    dut.decode_v_reduction_op.value = STALL_V_REDUCTION
    dut.decode_s_fp_op.value = STALL_S_FP
    dut.decode_c_op.value = STALL_C
    dut.decode_h_op.value = STALL_H
    dut.decode_v_broadcast_en.value = 0


async def reset(dut):
    dut.rst.value = 1
    clear_decode(dut)
    for name in (
        "hbm_m_prefetch_in_progress",
        "hbm_v_prefetch_in_progress",
        "continuous_write_to_v_sram",
        "fp_stall_req",
        "fp_sram_stall_req",
        "m_load_in_process",
        "m_mcu_active",
        "s_received_v_reduct_result",
        "mem_wreq_m_sram",
        "mem_wreq_s_sram_port_a",
        "mem_wreq_s_sram_port_b",
        "mem_wreq_from_m",
    ):
        getattr(dut, name).value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)


def recovery_only(dut) -> bool:
    return bool(int(dut.pipeline_stall_req.value)) and not bool(
        int(dut.raw_pipeline_stall.value)
    )


async def hbm_vector_conflict_trace(dut):
    await reset(dut)
    dut.hbm_v_prefetch_in_progress.value = 1
    dut.decode_v_element_op.value = ADD_V
    await RisingEdge(dut.clk)
    clear_decode(dut)

    raw_cycles = 0
    recovery_cycles = 0
    execute_cycle = None
    release_cycle = None
    for cycle in range(1, 41):
        await RisingEdge(dut.clk)
        if int(dut.raw_pipeline_stall.value):
            raw_cycles += 1
            if raw_cycles == 3:
                dut.hbm_v_prefetch_in_progress.value = 0
                release_cycle = cycle
        if recovery_only(dut):
            recovery_cycles += 1
        if int(dut.execute_v_element_op.value) == ADD_V:
            execute_cycle = cycle
            break
    assert raw_cycles >= 3
    assert recovery_cycles == 1
    assert execute_cycle is not None and release_cycle is not None
    cocotb.log.info(
        "[RTL_TIMING] HAZARD_HBM_V_VECTOR measurement_boundary=pipeline_control "
        f"raw_stall_cycles={raw_cycles} recovery_cycles={recovery_cycles} "
        f"release_cycle={release_cycle} execute_cycle={execute_cycle}"
    )


async def reduction_scalar_trace(dut):
    await reset(dut)
    dut.decode_v_reduction_op.value = SUM_V
    await RisingEdge(dut.clk)
    clear_decode(dut)
    dut.decode_s_fp_op.value = LD_REG_FP
    await RisingEdge(dut.clk)
    clear_decode(dut)

    reduction_execute = None
    scalar_execute = None
    raw_cycles = 0
    recovery_cycles = 0
    completion_pulsed = False
    for cycle in range(1, 61):
        await RisingEdge(dut.clk)
        if int(dut.execute_v_reduction_op.value) == SUM_V:
            reduction_execute = cycle
        if int(dut.raw_pipeline_stall.value):
            raw_cycles += 1
            if raw_cycles == 4 and not completion_pulsed:
                dut.s_received_v_reduct_result.value = 1
                completion_pulsed = True
        elif completion_pulsed:
            dut.s_received_v_reduct_result.value = 0
        if recovery_only(dut):
            recovery_cycles += 1
        if int(dut.execute_s_fp_op.value) == LD_REG_FP:
            scalar_execute = cycle
            break
    assert reduction_execute is not None
    assert raw_cycles >= 4
    assert recovery_cycles == 1
    assert scalar_execute is not None and scalar_execute > reduction_execute
    cocotb.log.info(
        "[RTL_TIMING] HAZARD_REDUCTION_SCALAR measurement_boundary=pipeline_control "
        f"producer_execute_cycle={reduction_execute} raw_stall_cycles={raw_cycles} "
        f"recovery_cycles={recovery_cycles} consumer_execute_cycle={scalar_execute}"
    )


async def scalar_sfu_broadcast_trace(dut):
    await reset(dut)
    dut.decode_s_fp_op.value = RECI_FP
    await RisingEdge(dut.clk)
    clear_decode(dut)
    dut.decode_v_element_op.value = MUL_V
    dut.decode_s_fp_op.value = LD_OUT_FP
    dut.decode_v_broadcast_en.value = 1
    await RisingEdge(dut.clk)
    clear_decode(dut)

    producer_execute = None
    consumer_execute = None
    raw_cycles = 0
    recovery_cycles = 0
    for cycle in range(1, 61):
        await RisingEdge(dut.clk)
        if int(dut.execute_s_fp_op.value) == RECI_FP:
            producer_execute = cycle
            dut.fp_stall_req.value = 1
        if int(dut.raw_pipeline_stall.value):
            raw_cycles += 1
            if raw_cycles == 4:
                dut.fp_stall_req.value = 0
        if recovery_only(dut):
            recovery_cycles += 1
        if int(dut.execute_v_element_op.value) == MUL_V:
            consumer_execute = cycle
            break
    assert producer_execute is not None
    assert raw_cycles >= 4
    assert recovery_cycles == 1
    assert consumer_execute is not None and consumer_execute > producer_execute
    cocotb.log.info(
        "[RTL_TIMING] HAZARD_SFU_BROADCAST measurement_boundary=pipeline_control "
        f"producer_execute_cycle={producer_execute} raw_stall_cycles={raw_cycles} "
        f"recovery_cycles={recovery_cycles} consumer_execute_cycle={consumer_execute}"
    )


async def vector_port_write_trace(dut):
    await reset(dut)
    dut.mem_wreq_s_sram_port_a.value = 1
    dut.decode_v_element_op.value = ADD_V
    await RisingEdge(dut.clk)
    clear_decode(dut)

    raw_cycles = 0
    recovery_cycles = 0
    execute_cycle = None
    for cycle in range(1, 41):
        await RisingEdge(dut.clk)
        if int(dut.raw_pipeline_stall.value):
            raw_cycles += 1
            if raw_cycles == 2:
                dut.mem_wreq_s_sram_port_a.value = 0
        if recovery_only(dut):
            recovery_cycles += 1
        if int(dut.execute_v_element_op.value) == ADD_V:
            execute_cycle = cycle
            break
    assert raw_cycles >= 2
    assert recovery_cycles == 1
    assert execute_cycle is not None
    cocotb.log.info(
        "[RTL_TIMING] HAZARD_VECTOR_PORT_WRITE measurement_boundary=pipeline_control "
        f"raw_stall_cycles={raw_cycles} recovery_cycles={recovery_cycles} "
        f"execute_cycle={execute_cycle}"
    )


@cocotb.test()
async def pipeline_control_hazard_traces(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hbm_vector_conflict_trace(dut)
    await reduction_scalar_trace(dut)
    await scalar_sfu_broadcast_trace(dut)
    await vector_port_write_trace(dut)


if __name__ == "__main__":
    veri_runner(
        group="control",
        module="pipeline_control_timing_wrapper",
        additional_include_paths=[str(SRC_PATH / "basic_components/common")],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=False,
        test_module=Path(__file__).stem,
    )
