#!/usr/bin/env python3
"""Measure opcode latency at the production VectorMachine boundary."""

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
STALL_ELEMENT, STALL_REDUCTION = 0, 0


async def reset(dut):
    dut.rst.value = 1
    dut.element_op.value = STALL_ELEMENT
    dut.reduction_op.value = STALL_REDUCTION
    dut.broadcast_fp2.value = 0
    dut.v_a.value = 0
    dut.v_b.value = 0
    dut.v_a_valid.value = 0
    dut.v_b_valid.value = 0
    dut.scalar_in.value = 0
    dut.scalar_in_valid.value = 0
    dut.scalar_target.value = 1
    dut.result_waddr.value = 0
    dut.result_waddr_update.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


def packed_one() -> tuple[int, int]:
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input([1.0] * VLEN)
    width = 1 + EXP_WIDTH + MANT_WIDTH
    return sum(int(value) << (width * index) for index, value in enumerate(encoded)), int(encoded[0])


async def measure_element(dut, operation: int, name: str, *, broadcast: bool) -> tuple[int, int]:
    await reset(dut)
    packed, scalar = packed_one()
    dut.result_waddr.value = 64
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    dut.element_op.value = operation
    dut.broadcast_fp2.value = int(broadcast)
    dut.v_a.value = packed
    dut.v_b.value = packed
    dut.scalar_in.value = scalar
    if broadcast:
        # The production control path records broadcast_fp2 before port B can
        # select the scalar. Accept the opcode first, then present both SRAM
        # operands on the following cycle, matching execute -> SRAM timing.
        dut.v_a_valid.value = 0
        dut.v_b_valid.value = 0
        dut.scalar_in_valid.value = 0
        await RisingEdge(dut.clk)
        first_cycle = 2
        dut.v_a_valid.value = 1
        dut.scalar_in_valid.value = 1
    else:
        first_cycle = 1
        dut.v_a_valid.value = 1
        dut.v_b_valid.value = 1
        dut.scalar_in_valid.value = 0

    leaf_cycle = None
    top_cycle = None
    for cycle in range(first_cycle, 301):
        await RisingEdge(dut.clk)
        if cycle == first_cycle:
            dut.element_op.value = STALL_ELEMENT
            dut.v_a_valid.value = 0
            dut.v_b_valid.value = 0
            dut.scalar_in_valid.value = 0
        if leaf_cycle is None and int(dut.leaf_element_ready.value):
            leaf_cycle = cycle
        if int(dut.vector_result_ready.value):
            top_cycle = cycle
            break
    if leaf_cycle is None or top_cycle is None:
        raise AssertionError(f"{name} timed out: leaf={leaf_cycle}, top={top_cycle}")
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"leaf_ready_cycles={leaf_cycle} ready_cycles={top_cycle} "
        f"done_cycles={top_cycle}"
    )
    return leaf_cycle, top_cycle


async def measure_reduction(dut, operation: int, name: str) -> tuple[int, int]:
    await reset(dut)
    packed, scalar = packed_one()
    dut.reduction_op.value = operation
    dut.v_a.value = packed
    dut.v_a_valid.value = 1
    dut.scalar_in.value = scalar
    dut.scalar_in_valid.value = 1
    dut.scalar_target.value = 2

    # As with broadcast element ops, reduction control selects the scalar-seed
    # buffer one stage before vector/scalar data arrive from their SRAMs.
    dut.v_a_valid.value = 0
    dut.scalar_in_valid.value = 0
    await RisingEdge(dut.clk)
    dut.v_a_valid.value = 1
    dut.scalar_in_valid.value = 1

    leaf_cycle = None
    top_cycle = None
    for cycle in range(2, 501):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.reduction_op.value = STALL_REDUCTION
            dut.v_a_valid.value = 0
            dut.scalar_in_valid.value = 0
        if leaf_cycle is None and int(dut.leaf_reduction_ready.value):
            leaf_cycle = cycle
        if int(dut.reduction_result_ready.value):
            top_cycle = cycle
            break
    if leaf_cycle is None or top_cycle is None:
        raise AssertionError(f"{name} timed out: leaf={leaf_cycle}, top={top_cycle}")
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"leaf_ready_cycles={leaf_cycle} ready_cycles={top_cycle} "
        f"done_cycles={top_cycle}"
    )
    return leaf_cycle, top_cycle


async def measure_element_ii(dut):
    """Launch two independent ADDs in consecutive full-Machine cycles.

    The leaf ALU advertises a one-cycle initiation interval, but the production
    VectorMachine also contains operand slices and a tracking FIFO.  Observing
    both ``elem_push`` and the architectural write pulses proves that the top
    can sustain that interval without dropping or reordering an operation.
    """
    await reset(dut)
    packed, _ = packed_one()

    # Prime the first destination before the first operand launch, matching the
    # normal execute -> SRAM -> VectorMachine control/data alignment.
    dut.result_waddr.value = 64
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    dut.element_op.value = 1  # ADD
    dut.v_a.value = packed
    dut.v_b.value = packed
    dut.v_a_valid.value = 1
    dut.v_b_valid.value = 1
    await RisingEdge(dut.clk)

    # Keep a second independent operand packet directly behind the first and
    # update its destination while the first packet enters the tracking FIFO.
    dut.result_waddr.value = 128
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.element_op.value = STALL_ELEMENT
    dut.result_waddr_update.value = 0
    dut.v_a_valid.value = 0
    dut.v_b_valid.value = 0

    launches = []
    results = []
    # The first launch may be visible on the edge just consumed above. Count
    # the current sampled value before advancing further.
    if int(dut.element_launch.value):
        launches.append(0)
    for cycle in range(1, 101):
        await RisingEdge(dut.clk)
        if int(dut.element_launch.value):
            launches.append(cycle)
        if int(dut.vector_result_ready.value):
            results.append((cycle, int(dut.committed_waddr.value)))
        if len(results) >= 2:
            break

    if len(launches) != 2 or len(results) != 2:
        raise AssertionError(
            f"V_ADD_VV II measurement lost work: launches={launches}, results={results}"
        )
    launch_ii = launches[1] - launches[0]
    result_ii = results[1][0] - results[0][0]
    if [address for _, address in results] != [64, 128]:
        raise AssertionError(f"V_ADD_VV results reordered: {results}")
    cocotb.log.info(
        f"[RTL_TIMING] V_ADD_VV_II measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"accepted_interval_cycles={launch_ii} "
        f"result_interval_cycles={result_ii} "
        f"initiation_interval_cycles={launch_ii} result_order_preserved=1"
    )


async def measure_mixed_element_order(dut):
    """Prove the safe issue rule for differently-latent element opcodes.

    The production element ALU has one ``recorded_operation`` selector. If a
    MUL replaces ADD before ADD's valid pulse, the earlier result is not
    independently selectable. We first observe that unsafe back-to-back case,
    then wait for the ADD result before issuing MUL and require both committed
    addresses to remain ordered.
    """
    packed, encoded_one = packed_one()
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded_two_values = generator.generate_specified_value_fp_input([2.0])
    encoded_two = int(encoded_two_values[0])

    async def run_unsafe():
        await reset(dut)
        cycle = 0
        launches = []
        results = []

        async def tick():
            nonlocal cycle
            await RisingEdge(dut.clk)
            cycle += 1
            if int(dut.element_launch.value):
                launches.append(cycle)
            if int(dut.vector_result_ready.value):
                results.append(
                    (
                        cycle,
                        int(dut.committed_waddr.value),
                        int(dut.committed_lane0.value),
                    )
                )

        dut.result_waddr.value = 64
        dut.result_waddr_update.value = 1
        await tick()
        dut.result_waddr_update.value = 0

        dut.element_op.value = 1  # ADD
        dut.v_a.value = packed
        dut.v_b.value = packed
        dut.v_a_valid.value = 1
        dut.v_b_valid.value = 1
        await tick()

        # Replace the global operation selector with MUL while ADD is still in
        # flight, exactly the sequence the scheduler must prevent.
        dut.element_op.value = 3  # MUL
        dut.result_waddr.value = 128
        dut.result_waddr_update.value = 1
        await tick()
        dut.element_op.value = STALL_ELEMENT
        dut.result_waddr_update.value = 0
        dut.v_a_valid.value = 0
        dut.v_b_valid.value = 0

        for _ in range(50):
            await tick()
        return launches, results

    async def run_safe():
        await reset(dut)
        cycle = 0
        launches = []
        results = []

        async def tick():
            nonlocal cycle
            await RisingEdge(dut.clk)
            cycle += 1
            if int(dut.element_launch.value):
                launches.append(cycle)
            if int(dut.vector_result_ready.value):
                results.append(
                    (
                        cycle,
                        int(dut.committed_waddr.value),
                        int(dut.committed_lane0.value),
                    )
                )

        dut.result_waddr.value = 64
        dut.result_waddr_update.value = 1
        await tick()
        dut.result_waddr_update.value = 0
        dut.element_op.value = 1  # ADD
        dut.v_a.value = packed
        dut.v_b.value = packed
        dut.v_a_valid.value = 1
        dut.v_b_valid.value = 1
        await tick()
        dut.element_op.value = STALL_ELEMENT
        dut.v_a_valid.value = 0
        dut.v_b_valid.value = 0

        while len(results) < 1 and cycle < 100:
            await tick()
        if len(results) != 1:
            raise AssertionError(f"safe mixed-op ADD did not complete: {results}")

        # One control cycle records the new destination/opcode, followed by
        # the normal SRAM operand cycle.
        dut.result_waddr.value = 128
        dut.result_waddr_update.value = 1
        dut.element_op.value = 3  # MUL
        await tick()
        dut.result_waddr_update.value = 0
        dut.v_a_valid.value = 1
        dut.v_b_valid.value = 1
        await tick()
        dut.element_op.value = STALL_ELEMENT
        dut.v_a_valid.value = 0
        dut.v_b_valid.value = 0
        while len(results) < 2 and cycle < 160:
            await tick()
        return launches, results

    unsafe_launches, unsafe_results = await run_unsafe()
    safe_launches, safe_results = await run_safe()
    unsafe_order_preserved = int(
        len(unsafe_results) == 2
        and [address for _, address, _ in unsafe_results] == [64, 128]
        and [value for _, _, value in unsafe_results] == [encoded_two, encoded_one]
    )
    safe_order_preserved = int(
        len(safe_results) == 2
        and [address for _, address, _ in safe_results] == [64, 128]
        and [value for _, _, value in safe_results] == [encoded_two, encoded_one]
    )
    if len(unsafe_launches) != 2:
        raise AssertionError(f"unsafe mixed-op launch setup failed: {unsafe_launches}")
    if len(safe_launches) != 2 or not safe_order_preserved:
        raise AssertionError(
            f"safe mixed-op sequence failed: launches={safe_launches}, results={safe_results}"
        )
    cocotb.log.info(
        f"[RTL_TIMING] V_MIXED_ADD_MUL_ORDER measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"unsafe_launch_interval_cycles={unsafe_launches[1] - unsafe_launches[0]} "
        f"unsafe_result_count={len(unsafe_results)} "
        f"unsafe_order_preserved={unsafe_order_preserved} "
        f"safe_launch_interval_cycles={safe_launches[1] - safe_launches[0]} "
        f"safe_result_count={len(safe_results)} "
        f"safe_order_preserved={safe_order_preserved}"
    )


@cocotb.test()
async def vector_machine_full_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    for operation, stem in (
        (1, "ADD"),
        (2, "SUB"),
        (3, "MUL"),
        (4, "EXP"),
        (5, "RECI"),
    ):
        await measure_element(dut, operation, f"V_{stem}_VV", broadcast=False)
        if operation in (1, 2, 3):
            await measure_element(dut, operation, f"V_{stem}_VF", broadcast=True)
    await measure_reduction(dut, 1, "V_RED_SUM")
    await measure_reduction(dut, 2, "V_RED_MAX")
    await measure_element_ii(dut)
    await measure_mixed_element_order(dut)


if __name__ == "__main__":
    veri_runner(
        group="vector_machine",
        module="vector_machine_timing_wrapper",
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
        trace=False,
        test_module=Path(__file__).stem,
    )
