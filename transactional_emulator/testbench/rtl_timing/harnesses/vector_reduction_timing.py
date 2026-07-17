#!/usr/bin/env python3
"""Cycle harness for the parameterized vector reduction tree."""

import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import FpGenerator, veri_runner
from cfl_cocotb.runner import SRC_PATH

EXP_WIDTH, MANT_WIDTH = 4, 3


@cocotb.test()
async def vector_reduction_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    vlen = int(dut.VLEN.value)
    vector_dim = vlen + 1
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, values = generator.generate_specified_value_fp_input([1.0] * vector_dim)
    width = 1 + EXP_WIDTH + MANT_WIDTH
    packed = sum(value << (width * index) for index, value in enumerate(values))

    for operation, name in ((1, "V_RED_SUM"), (2, "V_RED_MAX")):
        dut.rst.value = 1
        dut.v_in_valid.value = 0
        dut.s_out_ready.value = 1
        dut.operation.value = 0
        for _ in range(3):
            await RisingEdge(dut.clk)
        dut.rst.value = 0
        for _ in range(2):
            await RisingEdge(dut.clk)

        dut.v_in.value = packed
        dut.operation.value = operation
        dut.v_in_valid.value = 1
        for cycle in range(1, 501):
            await RisingEdge(dut.clk)
            if int(dut.s_out_valid.value):
                cocotb.log.info(
                    f"[RTL_TIMING] {name} vlen={vlen} "
                    f"tree_levels={math.ceil(math.log2(vector_dim))} cycles={cycle}"
                )
                break
        else:
            raise AssertionError(f"{name} timed out")


if __name__ == "__main__":
    vlens = [8] if os.environ.get("PLENA_TIMING_MODE", "smoke") == "smoke" else [8, 16, 32]
    veri_runner(
        group="vector_machine",
        module="fp_reduction_compute_unit",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        module_param_list=[
            {"EXP_WIDTH": EXP_WIDTH, "MANT_WIDTH": MANT_WIDTH, "VLEN": vlen}
            for vlen in vlens
        ],
        trace=False,
        test_module=Path(__file__).stem,
    )

