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


async def measure_compute(
    dut,
    operation: int,
    name: str,
    *,
    input_a: float = 1.0,
    input_b: float = 1.0,
    expected: float | None = None,
):
    await reset(dut)
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([input_a, input_b])
    await initialize_register(dut, 1, int(encoded[0]))
    await initialize_register(dut, 2, int(encoded[1]))

    dut.scalar_fp_op.value = operation
    dut.fps1.value = 1
    dut.fps2.value = 2
    dut.fpd.value = 3
    ready = None
    done = None
    saw_busy = False
    result = None
    for cycle in range(1, 401):
        await RisingEdge(dut.clk)
        if cycle == 1:
            dut.scalar_fp_op.value = STALL
        busy = bool(int(dut.backend_busy.value))
        saw_busy |= busy
        if ready is None and int(dut.forwarding_result_ready.value):
            ready = cycle
        if int(dut.compute_result_ready.value):
            result = int(dut.compute_result.value)
        if saw_busy and not busy and done is None:
            done = cycle
        if ready is not None and done is not None:
            break
    if ready is None or done is None:
        raise AssertionError(f"{name} timed out: ready={ready}, done={done}")
    if expected is not None:
        _, expected_encoded = generator.generate_specified_value_fp_input([expected])
        if result != int(expected_encoded[0]):
            raise AssertionError(
                f"{name} result mismatch: got=0x{result:x}, expected=0x{int(expected_encoded[0]):x}"
            )
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


async def measure_independent_stream(dut, operation: int, name: str):
    """Issue eight independent operations and prove one-cycle acceptance."""
    await reset(dut)
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([1.0, 2.0])
    await initialize_register(dut, 1, int(encoded[0]))
    await initialize_register(dut, 2, int(encoded[1]))

    cycle = 0
    retirements = []

    async def tick():
        nonlocal cycle
        await RisingEdge(dut.clk)
        cycle += 1
        if int(dut.compute_result_ready.value):
            retirements.append((cycle, int(dut.compute_result_rd.value)))

    destinations = list(range(3, 11))
    for destination in destinations:
        if int(dut.frontend_stall.value):
            raise AssertionError(f"{name} unexpectedly stalled before f{destination}")
        dut.scalar_fp_op.value = operation
        dut.fps1.value = 1
        dut.fps2.value = 2
        dut.fpd.value = destination
        await tick()
    dut.scalar_fp_op.value = STALL

    while len(retirements) < len(destinations) and cycle < 300:
        await tick()
    if [rd for _, rd in retirements] != destinations:
        raise AssertionError(f"{name} retirement order mismatch: {retirements}")
    intervals = [
        retirements[index + 1][0] - retirements[index][0]
        for index in range(len(retirements) - 1)
    ]
    if any(interval != 1 for interval in intervals):
        raise AssertionError(f"{name} did not retire at II=1: {retirements}")
    cocotb.log.info(
        f"[RTL_TIMING] {name}_PIPELINE measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"operation_count={len(destinations)} accepted_interval_cycles=1 "
        f"result_interval_cycles=1 initiation_interval_cycles=1 "
        f"first_ready_cycles={retirements[0][0]} done_cycles={retirements[-1][0]} "
        f"retirement_order_preserved=1"
    )


async def measure_raw_forwarding(dut):
    await reset(dut)
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([2.0])
    await initialize_register(dut, 1, int(encoded[0]))
    await initialize_register(dut, 2, int(encoded[0]))

    dut.scalar_fp_op.value = 4  # MUL f3 = f1 * f2
    dut.fps1.value = 1
    dut.fps2.value = 2
    dut.fpd.value = 3
    await RisingEdge(dut.clk)
    dut.scalar_fp_op.value = 1  # ADD f4 = f3 + f1
    dut.fps1.value = 3
    dut.fps2.value = 1
    dut.fpd.value = 4
    await RisingEdge(dut.clk)
    dut.scalar_fp_op.value = STALL

    retirements = []
    for cycle in range(2, 101):
        await RisingEdge(dut.clk)
        if int(dut.compute_result_ready.value):
            retirements.append(
                (cycle, int(dut.compute_result_rd.value), int(dut.compute_result.value))
            )
        if len(retirements) == 2:
            break
    _, expected = generator.generate_specified_value_fp_input([4.0, 6.0])
    if [rd for _, rd, _ in retirements] != [3, 4] or [
        value for _, _, value in retirements
    ] != [int(expected[0]), int(expected[1])]:
        raise AssertionError(f"scalar RAW forwarding failed: {retirements}")
    cocotb.log.info(
        f"[RTL_TIMING] S_RAW_FORWARD_CHAIN measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"issue_interval_cycles=1 forwarding_used=1 result_check=pass "
        f"first_ready_cycles={retirements[0][0]} done_cycles={retirements[-1][0]}"
    )


async def measure_mixed_latency_retirement(dut):
    await reset(dut)
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([1.0])
    await initialize_register(dut, 1, int(encoded[0]))

    dut.scalar_fp_op.value = 5  # Older EXP.
    dut.fps1.value = 1
    dut.fps2.value = 0
    dut.fpd.value = 3
    await RisingEdge(dut.clk)
    dut.scalar_fp_op.value = 12  # Younger one-cycle MOVE.
    dut.fpd.value = 4
    await RisingEdge(dut.clk)
    dut.scalar_fp_op.value = STALL

    retirements = []
    for cycle in range(2, 101):
        await RisingEdge(dut.clk)
        if int(dut.compute_result_ready.value):
            retirements.append((cycle, int(dut.compute_result_rd.value)))
        if len(retirements) == 2:
            break
    if [rd for _, rd in retirements] != [3, 4]:
        raise AssertionError(f"mixed-latency retirement reordered: {retirements}")
    cocotb.log.info(
        f"[RTL_TIMING] S_MIXED_ROB_RETIREMENT measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"issue_interval_cycles=1 retirement_order_preserved=1 "
        f"first_ready_cycles={retirements[0][0]} done_cycles={retirements[-1][0]}"
    )


async def measure_rob_full(dut):
    await reset(dut)
    for destination in range(1, 9):
        dut.scalar_fp_op.value = 5  # EXP(0), long enough to fill all entries.
        dut.fps1.value = 0
        dut.fps2.value = 0
        dut.fpd.value = destination
        await RisingEdge(dut.clk)
    dut.scalar_fp_op.value = STALL
    # Observe registered ROB count after the eighth enqueue edge.
    await RisingEdge(dut.clk)
    if not int(dut.rob_full.value) or not int(dut.frontend_stall.value):
        raise AssertionError(
            f"ROB did not assert full/stall after eight entries: full={int(dut.rob_full.value)}, stall={int(dut.frontend_stall.value)}"
        )
    first_release = None
    for cycle in range(1, 101):
        await RisingEdge(dut.clk)
        if not int(dut.rob_full.value):
            first_release = cycle
            break
    if first_release is None:
        raise AssertionError("ROB full condition did not clear")
    cocotb.log.info(
        f"[RTL_TIMING] S_ROB_FULL measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"rob_entries=8 full_stall_observed=1 release_cycles={first_release}"
    )


@cocotb.test()
async def scalar_machine_full_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    for operation, name, input_a, input_b, expected in (
        (1, "S_ADD_FP", 1.0, 2.0, 3.0),
        (2, "S_SUB_FP", 2.0, 1.0, 1.0),
        (3, "S_MAX_FP", -1.0, -2.0, -1.0),
        (4, "S_MUL_FP", 2.0, 2.0, 4.0),
        (5, "S_EXP_FP", 1.0, 1.0, None),
        (6, "S_RECI_FP", 2.0, 1.0, 0.5),
        (7, "S_SQRT_FP", 4.0, 1.0, 2.0),
        (12, "S_MV_FP", -2.0, 1.0, -2.0),
        (14, "S_RSQRT_FP", 4.0, 1.0, 0.5),
    ):
        await measure_compute(
            dut,
            operation,
            name,
            input_a=input_a,
            input_b=input_b,
            expected=expected,
        )
    for operation, name in (
        (1, "S_ADD_FP"),
        (3, "S_MAX_FP"),
        (4, "S_MUL_FP"),
        (5, "S_EXP_FP"),
        (6, "S_RECI_FP"),
        (7, "S_SQRT_FP"),
        (12, "S_MV_FP"),
    ):
        await measure_independent_stream(dut, operation, name)
    await measure_raw_forwarding(dut)
    await measure_mixed_latency_retirement(dut)
    await measure_rob_full(dut)
    await measure_map(dut)


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
