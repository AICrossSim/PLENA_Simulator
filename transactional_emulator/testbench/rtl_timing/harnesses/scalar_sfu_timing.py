#!/usr/bin/env python3
"""Cycle harness for scalar EXP/reciprocal/sqrt wrappers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

STALL, EXP, RECI, SQRT = 0, 5, 6, 7
FP16_ONE = 0x3C00


async def reset(dut):
    dut.rst.value = 1
    dut.operation.value = STALL
    dut.data_in.value = FP16_ONE
    dut.reg_waddr.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


async def measure(dut, operation, name):
    dut.operation.value = operation
    for cycle in range(1, 201):
        await RisingEdge(dut.clk)
        if int(dut.data_out_valid.value):
            dut.operation.value = STALL
            cocotb.log.info(f"[RTL_TIMING] {name} cycles={cycle}")
            return
    raise AssertionError(f"{name} timed out")


@cocotb.test()
async def scalar_sfu_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    for operation, name in ((EXP, "S_EXP_FP"), (RECI, "S_RECI_FP"), (SQRT, "S_SQRT_FP")):
        await reset(dut)
        await measure(dut, operation, name)
        for _ in range(3):
            await RisingEdge(dut.clk)


if __name__ == "__main__":
    veri_runner(
        group="scalar_machine",
        module="fp_sfu",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/fixed_operation"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=False,
        test_module=Path(__file__).stem,
    )

