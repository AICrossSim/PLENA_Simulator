#!/usr/bin/env python3
"""Measure M_MM load and M_MM_WO drain at the MatrixMachine boundary."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH


MLEN = int(os.environ.get("PLENA_TEST_MLEN", "16"))
BLEN = int(os.environ.get("PLENA_TEST_BLEN", "4"))
STALL, MM_IC, MM_WO = 0, 5, 7


async def reset(dut):
    dut.rst.value = 1
    dut.matrix_op.value = STALL
    dut.result_waddr.value = 0
    dut.result_waddr_update.value = 0
    dut.matrix_element.value = 0
    dut.matrix_scale.value = 0
    dut.matrix_valid.value = 0
    dut.vector_element.value = 0
    dut.vector_scale.value = 0
    dut.vector_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


@cocotb.test()
async def matrix_machine_full_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    # Stream one BLEN-row tile through the real MatrixMachine input alignment
    # buffers. Values are irrelevant to timing; nonzero scales avoid X paths.
    dut.matrix_scale.value = (1 << (8 * MLEN)) - 1
    dut.vector_scale.value = (1 << (8 * (MLEN // BLEN))) - 1
    dut.matrix_op.value = MM_IC
    dut.matrix_valid.value = 1
    dut.vector_valid.value = 1
    load_done = None
    for cycle in range(1, BLEN + 32):
        await RisingEdge(dut.clk)
        if cycle == BLEN:
            dut.matrix_valid.value = 0
            dut.vector_valid.value = 0
            dut.matrix_op.value = STALL
        if int(dut.complete_operand_load.value):
            load_done = cycle
            break
    if load_done is None:
        raise AssertionError("M_MM operand load did not complete")
    cocotb.log.info(
        f"[RTL_TIMING] M_MM measurement_boundary=full_machine "
        f"mlen={MLEN} blen={BLEN} ready_cycles={load_done} "
        f"done_cycles={load_done} initiation_interval_cycles={load_done}"
    )

    dut.matrix_op.value = MM_WO
    dut.result_waddr.value = 128
    dut.result_waddr_update.value = 1
    first = None
    rows = []
    busy_done = None
    saw_busy = False
    for cycle in range(1, 4 * BLEN + 128):
        await RisingEdge(dut.clk)
        if cycle == 1:
            dut.result_waddr_update.value = 0
        busy = bool(int(dut.backend_busy.value))
        saw_busy |= busy
        if saw_busy and not busy and busy_done is None:
            busy_done = cycle
        if int(dut.result_ready.value):
            rows.append(cycle)
            first = first or cycle
        if len(rows) >= BLEN and busy_done is not None:
            break
    dut.matrix_op.value = STALL
    if not rows or busy_done is None:
        raise AssertionError(f"M_MM_WO timed out: rows={rows}, busy_done={busy_done}")
    cadence = 0 if len(rows) < 2 else rows[1] - rows[0]
    row_writeback_supported = len(rows) >= BLEN
    cocotb.log.info(
        f"[RTL_TIMING] M_MM_WO measurement_boundary=full_machine "
        f"mlen={MLEN} blen={BLEN} first_result_cycles={first} "
        f"row_cadence_cycles={cadence} observed_result_pulses={len(rows)} "
        f"expected_result_rows={BLEN} "
        f"row_writeback_supported={int(row_writeback_supported)} "
        f"architectural_all_results_cycles={rows[-1] if row_writeback_supported else -1} "
        f"busy_cycles={busy_done} ready_cycles={rows[-1]} done_cycles={busy_done}"
    )


if __name__ == "__main__":
    veri_runner(
        group="matrix_machine",
        module="matrix_machine_timing_wrapper",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/systolic_gemm_mx"),
            str(SRC_PATH / "basic_components/systolic_gemm_fp"),
            str(SRC_PATH / "basic_components/mx_fp_operation"),
            str(SRC_PATH / "basic_components/gemv"),
            str(SRC_PATH / "basic_components/linear_operation"),
            str(SRC_PATH / "basic_components/mx_int_operation"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        extra_build_args=["-DSIMULATION"],
        trace=False,
        test_module=Path(__file__).stem,
    )
