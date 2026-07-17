#!/usr/bin/env python3
"""Cycle harness for lane-parallel vector elementwise operations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import FpGenerator, veri_runner
from cfl_cocotb.runner import SRC_PATH

EXP_WIDTH, MANT_WIDTH, VLEN = 4, 3, 8


async def reset(dut):
    dut.rst.value = 1
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0
    dut.v_out_ready.value = 1
    dut.operation.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)


@cocotb.test()
async def vector_element_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, values = generator.generate_specified_value_fp_input([1.0] * VLEN)
    width = 1 + EXP_WIDTH + MANT_WIDTH
    packed = sum(value << (width * index) for index, value in enumerate(values))

    for operation, name in (
        (1, "V_ADD_VV"),
        (2, "V_SUB_VV"),
        (3, "V_MUL_VV"),
        (4, "V_EXP_V"),
        (5, "V_RECI_V"),
    ):
        await reset(dut)
        dut.v_in_a.value = packed
        dut.v_in_b.value = packed
        dut.operation.value = operation
        dut.v_in_a_valid.value = 1
        dut.v_in_b_valid.value = 1
        for cycle in range(1, 201):
            await RisingEdge(dut.clk)
            if int(dut.v_out_valid.value):
                cocotb.log.info(f"[RTL_TIMING] {name} vlen={VLEN} cycles={cycle}")
                break
        else:
            raise AssertionError(f"{name} timed out")

    # Measure initiation interval independently from single-op latency. Keep
    # ADD valid for exactly two adjacent cycles and observe the two result
    # pulses; this is the condition used by rtl-v1 to pipeline independent ops.
    await reset(dut)
    dut.v_in_a.value = packed
    dut.v_in_b.value = packed
    dut.operation.value = 1
    dut.v_in_a_valid.value = 1
    dut.v_in_b_valid.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.v_in_a_valid.value = 0
    dut.v_in_b_valid.value = 0

    output_cycles = []
    for cycle in range(1, 101):
        await RisingEdge(dut.clk)
        if int(dut.v_out_valid.value):
            output_cycles.append(cycle)
            if len(output_cycles) == 2:
                break
    if len(output_cycles) != 2:
        raise AssertionError(f"expected two pipelined ADD results, got {output_cycles}")
    initiation_interval = output_cycles[1] - output_cycles[0]
    cocotb.log.info(
        f"[RTL_TIMING] V_ELEMENT_II vlen={VLEN} "
        f"initiation_interval_cycles={initiation_interval}"
    )
    assert initiation_interval == 1


if __name__ == "__main__":
    veri_runner(
        group="vector_machine",
        module="fp_elementwise_compute_unit",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/hadamard_transform"),
            str(SRC_PATH / "basic_components/synopsis_ip_inst"),
            str(SRC_PATH / "basic_components/conversion"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/synopsis"),
            str(SRC_PATH / "basic_components/cast"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        module_param_list=[{"EXP_WIDTH": EXP_WIDTH, "MANT_WIDTH": MANT_WIDTH, "VLEN": VLEN}],
        trace=False,
        test_module=Path(__file__).stem,
    )
