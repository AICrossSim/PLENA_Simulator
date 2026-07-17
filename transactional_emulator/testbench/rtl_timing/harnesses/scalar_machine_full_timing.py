#!/usr/bin/env python3
"""Measure opcode latency at the production ScalarMachine boundary."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cfl_cocotb import FpGenerator, veri_runner
from cfl_cocotb.runner import SRC_PATH


VLEN = int(os.environ.get("PLENA_TEST_VLEN", "8"))
EXP_WIDTH = int(os.environ.get("PLENA_TEST_FP_EXP", "8"))
MANT_WIDTH = int(os.environ.get("PLENA_TEST_FP_MANT", "7"))
STALL = 0


async def reset(dut):
    dut.rst.value = 1
    dut.scalar_fp_op.value = STALL
    dut.fps1.value = 1
    dut.fps2.value = 2
    dut.fpd.value = 3
    dut.map_addr.value = 0
    dut.external_fp_in.value = 0
    dut.external_fp_in_valid.value = 0
    dut.external_fp_wtarget.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


async def initialize_register(dut, target: int, value: int):
    dut.external_fp_in.value = value
    dut.external_fp_wtarget.value = target
    dut.external_fp_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.external_fp_in_valid.value = 0
    await RisingEdge(dut.clk)


async def measure_compute(dut, operation: int, name: str):
    await reset(dut)
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([1.0])
    await initialize_register(dut, 1, int(encoded[0]))
    await initialize_register(dut, 2, int(encoded[0]))

    dut.scalar_fp_op.value = operation
    dut.fps1.value = 1
    dut.fps2.value = 2
    dut.fpd.value = 3
    ready = None
    done = None
    saw_busy = False
    for cycle in range(1, 401):
        await RisingEdge(dut.clk)
        if cycle == 1:
            dut.scalar_fp_op.value = STALL
        busy = bool(int(dut.backend_busy.value))
        saw_busy |= busy
        if ready is None and int(dut.compute_result_ready.value):
            ready = cycle
        if saw_busy and not busy and done is None:
            done = cycle
        if ready is not None and done is not None:
            break
    if ready is None or done is None:
        raise AssertionError(f"{name} timed out: ready={ready}, done={done}")
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"ready_cycles={ready} done_cycles={done} initiation_interval_cycles={done}"
    )


async def measure_map(dut):
    await reset(dut)
    dut.scalar_fp_op.value = 13  # MAP_V_FP
    dut.map_addr.value = 0
    ready = None
    done = None
    saw_busy = False
    for cycle in range(1, VLEN + 32):
        await RisingEdge(dut.clk)
        if cycle == 1:
            dut.scalar_fp_op.value = STALL
        busy = bool(int(dut.scalar_sram_busy.value))
        saw_busy |= busy
        if ready is None and int(dut.map_result_ready.value):
            ready = cycle
        if saw_busy and not busy and done is None:
            done = cycle
        if ready is not None and done is not None:
            break
    if ready is None or done is None:
        raise AssertionError(f"S_MAP_V_FP timed out: ready={ready}, done={done}")
    cocotb.log.info(
        f"[RTL_TIMING] S_MAP_V_FP measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"ready_cycles={ready} done_cycles={done}"
    )


@cocotb.test()
async def scalar_machine_full_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    for operation, name in (
        (1, "S_ADD_FP"),
        (2, "S_SUB_FP"),
        (4, "S_MUL_FP"),
        (5, "S_EXP_FP"),
        (6, "S_RECI_FP"),
        (7, "S_SQRT_FP"),
    ):
        await measure_compute(dut, operation, name)
    await measure_map(dut)

    # MAX is decoded in the ISA but has no ALU/SFU implementation. Keep this
    # as an explicit support observation rather than assigning a fake latency.
    await reset(dut)
    dut.scalar_fp_op.value = 3
    implemented = False
    for cycle in range(1, 17):
        await RisingEdge(dut.clk)
        if cycle == 1:
            dut.scalar_fp_op.value = STALL
        implemented |= bool(int(dut.compute_result_ready.value))
    cocotb.log.info(
        f"[RTL_TIMING] S_MAX_FP measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"implemented={int(implemented)}"
    )
    assert not implemented


if __name__ == "__main__":
    veri_runner(
        group="scalar_machine",
        module="scalar_machine_timing_wrapper",
        additional_include_paths=[
            str(SRC_PATH / "basic_components/buffer"),
            str(SRC_PATH / "basic_components/common"),
            str(SRC_PATH / "basic_components/fp_operation"),
            str(SRC_PATH / "basic_components/int_operation"),
            str(SRC_PATH / "basic_components/cast"),
            str(SRC_PATH / "basic_components/fixed_operation"),
            str(SRC_PATH / "memory/scalar_sram"),
            # scalar_sram reuses the OpenTitan primitive include/macro files
            # kept beside the vector SRAM implementation.
            str(SRC_PATH / "memory/vector_sram"),
        ],
        definitions_path=[str(SRC_PATH / "definitions")],
        trace=False,
        test_module=Path(__file__).stem,
    )
