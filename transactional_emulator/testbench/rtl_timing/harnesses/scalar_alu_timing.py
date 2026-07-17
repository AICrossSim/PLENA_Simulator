#!/usr/bin/env python3
"""Cycle harness for the scalar FP ALU wrapper."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import veri_runner
from cfl_cocotb.runner import SRC_PATH

STALL, ADD, SUB, MUL = 0, 1, 2, 4
FP16_ONE = 0x3C00


async def reset(dut):
    dut.rst.value = 1
    dut.operation.value = STALL
    dut.reg_waddr.value = 1
    dut.data_a.value = FP16_ONE
    dut.data_b.value = FP16_ONE
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


async def measure(dut, operation, name):
    dut.operation.value = operation
    for cycle in range(1, 101):
        await RisingEdge(dut.clk)
        if int(dut.data_out_valid.value):
            dut.operation.value = STALL
            cocotb.log.info(f"[RTL_TIMING] {name} cycles={cycle}")
            return cycle
    raise AssertionError(f"{name} timed out")


@cocotb.test()
async def scalar_alu_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    results = {}
    for operation, name in ((ADD, "S_ADD_FP"), (SUB, "S_SUB_FP"), (MUL, "S_MUL_FP")):
        await reset(dut)
        results[name] = await measure(dut, operation, name)
        for _ in range(3):
            await RisingEdge(dut.clk)
    assert results["S_ADD_FP"] == results["S_SUB_FP"]


if __name__ == "__main__":
    veri_runner(
        group="scalar_machine",
        module="fp_alu",
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

