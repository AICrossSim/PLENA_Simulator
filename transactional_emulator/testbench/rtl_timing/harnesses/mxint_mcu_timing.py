#!/usr/bin/env python3
"""Cycle harness for the implemented MXINT GEMM MCU path."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[4] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

STALL, MM_IC, MM_WO = 0, 5, 7


async def reset(dut):
    dut.rst.value = 1
    dut.control.value = STALL
    dut.v1_element.value = 0
    dut.v1_scale.value = 0
    dut.v1_in_valid.value = 0
    dut.v2_element.value = 0
    dut.v2_scale.value = 0
    dut.v2_in_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


@cocotb.test()
async def mxint_mcu_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    blen, mlen = int(dut.BLEN.value), int(dut.MLEN.value)

    dut.control.value = MM_IC
    dut.v1_in_valid.value = 1
    dut.v2_in_valid.value = 1
    compute_cycles = 0
    saw_busy = False
    for _ in range(blen):
        await RisingEdge(dut.clk)
        compute_cycles += 1
        saw_busy |= bool(int(dut.mcu_active.value))
    dut.control.value = STALL
    dut.v1_in_valid.value = 0
    dut.v2_in_valid.value = 0
    for _ in range(1000):
        await RisingEdge(dut.clk)
        compute_cycles += 1
        active = bool(int(dut.mcu_active.value))
        saw_busy |= active
        if saw_busy and not active:
            break
    else:
        raise AssertionError("MXINT MM_IC did not return idle")

    dut.control.value = MM_WO
    first, last, rows = None, None, 0
    for cycle in range(1, 1001):
        await RisingEdge(dut.clk)
        if int(dut.v_result_write_req.value):
            rows += 1
            first = first or cycle
            last = cycle
            if rows == blen:
                break
    else:
        raise AssertionError("MXINT MM_WO timed out")

    cocotb.log.info(
        f"[RTL_TIMING] MXINT_MM_IC mlen={mlen} blen={blen} cycles={compute_cycles}"
    )
    cocotb.log.info(
        f"[RTL_TIMING] MXINT_MM_WO mlen={mlen} blen={blen} "
        f"first_result_cycles={first} all_results_cycles={last} busy_cycles={last}"
    )


if __name__ == "__main__":
    mode = os.environ.get("PLENA_TIMING_MODE", "smoke")
    shapes = [(16, 4)]
    if mode == "full":
        shapes = [
            (mlen, blen)
            for mlen in (16, 32, 64)
            for blen in (4, 8, 16)
            if mlen >= blen and mlen % blen == 0
        ]
    veri_runner(
        group="systolic_gemm_mxint",
        module="mxint_systolic_mcu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/conversion"),
        ],
        module_param_list=[
            {
                "FP_EXP_WIDTH": 6,
                "FP_MANT_WIDTH": 5,
                "MX_T_INT_WIDTH": 8,
                "MX_L_INT_WIDTH": 8,
                "MXINT_SCALE_WIDTH": 8,
                "KLEN": blen,
                "BLEN": blen,
                "MLEN": mlen,
                "MAX_SHIFT": 16,
            }
            for mlen, blen in shapes
        ],
        trace=False,
        test_module=Path(__file__).stem,
    )

