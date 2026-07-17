#!/usr/bin/env python3
"""Cycle harness for the current MXFP GEMM MCU load/drain phases."""

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
    dut.acc_waddr.value = 0
    dut.fetch_next_acc_waddr_valid.value = 0
    dut.wait_for_output.value = 0
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
async def mxfp_mcu_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    blen, mlen = int(dut.M.value), int(dut.K.value)
    overhead = int(dut.SYSTOLIC_PROCESSING_OVERHEAD.value)

    dut.control.value = MM_IC
    dut.v1_in_valid.value = 1
    dut.v2_in_valid.value = 1
    for _ in range(blen):
        await RisingEdge(dut.clk)

    dut.control.value = MM_WO
    dut.v1_in_valid.value = 0
    dut.v2_in_valid.value = 0
    dut.acc_waddr.value = 0
    dut.fetch_next_acc_waddr_valid.value = 1
    dut.wait_for_output.value = 1

    first, last, rows, busy_done = None, None, 0, None
    saw_busy = False
    for cycle in range(1, 2001):
        await RisingEdge(dut.clk)
        active = bool(int(dut.mcu_active.value))
        saw_busy |= active
        if saw_busy and not active and busy_done is None:
            busy_done = cycle
        if int(dut.v_result_write_req.value):
            rows += 1
            first = first or cycle
            last = cycle
        if rows >= blen and busy_done is not None:
            break
    else:
        raise AssertionError("MXFP MM_WO timed out")

    cocotb.log.info(
        f"[RTL_TIMING] MXFP_MM_IC mlen={mlen} blen={blen} cycles={blen}"
    )
    cocotb.log.info(
        f"[RTL_TIMING] MXFP_MM_WO mlen={mlen} blen={blen} overhead={overhead} "
        f"first_result_cycles={first} all_results_cycles={last} busy_cycles={busy_done}"
    )


if __name__ == "__main__":
    mode = os.environ.get("PLENA_TIMING_MODE", "smoke")
    # B=16 makes the behavioral FP array's Verilator build several GiB. Two
    # points are sufficient to validate the linear drain counter; larger
    # shapes should be run as an explicit validation, not in the default flow.
    shapes = [(4, 4)] if mode == "smoke" else [(4, 4), (8, 8)]
    veri_runner(
        group="systolic_gemm_mx",
        module="mx_systolic_mcu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/gemv"),
        ],
        definitions_path=[
            str(SRC_PATH / "definitions"),
            str(SRC_PATH / "memory/HBM/TileLink_Lib"),
        ],
        module_param_list=[
            {
                "MX_T_MANT_WIDTH": 3,
                "MX_T_EXP_WIDTH": 4,
                "MX_L_MANT_WIDTH": 3,
                "MX_L_EXP_WIDTH": 4,
                "MX_SCALE_WIDTH": 8,
                "BLOCK_DIM": min(4, blen),
                "ACC_FP_MANT_WIDTH": 23,
                "ACC_FP_EXP_WIDTH": 8,
                "FP_MANT_WIDTH": 7,
                "FP_EXP_WIDTH": 8,
                # Match the current configuration.svh production profile.
                "SYSTOLIC_PROCESSING_OVERHEAD": 8,
                "N": blen,
                "M": blen,
                "K": mlen,
            }
            for mlen, blen in shapes
        ],
        trace=False,
        test_module=Path(__file__).stem,
    )
