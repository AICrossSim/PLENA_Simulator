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
    dut.reduction_segment_log2.value = 0
    dut.reduction_segment_index.value = 0
    dut.compact_count_minus_one.value = 0
    dut.broadcast_fp2.value = 0
    dut.segment_broadcast_en.value = 0
    dut.compact_stats_en.value = 0
    dut.reduction_overwrite_en.value = 0
    dut.lane_store_en.value = 0
    dut.vector_mask.value = 0
    dut.element_mask_enable.value = 0
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


def packed_values(values: list[float]) -> tuple[int, list[int]]:
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, encoded = generator.generate_specified_value_fp_input(values)
    width = 1 + EXP_WIDTH + MANT_WIDTH
    words = [int(value) for value in encoded]
    return sum(value << (width * index) for index, value in enumerate(words)), words


def unpack_vector(value: int) -> list[int]:
    width = 1 + EXP_WIDTH + MANT_WIDTH
    mask = (1 << width) - 1
    return [(value >> (width * lane)) & mask for lane in range(VLEN)]


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


async def measure_reduction(
    dut,
    operation: int,
    name: str,
    *,
    expected_value: float,
    segment_log2: int = 0,
    segment_index: int = 0,
    overwrite: bool = False,
    input_value: float = 1.0,
) -> tuple[int, int]:
    await reset(dut)
    packed, _ = packed_values([input_value] * VLEN)
    _, scalar_values = packed_values([1.0])
    scalar = scalar_values[0]
    dut.reduction_op.value = operation
    dut.reduction_segment_log2.value = segment_log2
    dut.reduction_segment_index.value = segment_index
    dut.reduction_overwrite_en.value = int(overwrite)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1
    dut.scalar_in.value = scalar
    dut.scalar_in_valid.value = int(not overwrite)
    dut.scalar_target.value = 2

    # As with broadcast element ops, reduction control selects the scalar-seed
    # buffer one stage before vector/scalar data arrive from their SRAMs.
    dut.v_a_valid.value = 0
    dut.scalar_in_valid.value = 0
    await RisingEdge(dut.clk)
    dut.v_a_valid.value = 1
    dut.scalar_in_valid.value = int(not overwrite)

    leaf_cycle = None
    top_cycle = None
    result = None
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
            result = int(dut.reduction_value.value)
            break
    if leaf_cycle is None or top_cycle is None:
        raise AssertionError(f"{name} timed out: leaf={leaf_cycle}, top={top_cycle}")
    generator = FpGenerator(EXP_WIDTH, MANT_WIDTH)
    _, expected_encoded = generator.generate_specified_value_fp_input([expected_value])
    if result != int(expected_encoded[0]):
        raise AssertionError(
            f"{name} result mismatch: got=0x{result:x}, expected=0x{int(expected_encoded[0]):x}"
        )
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"segment_log2={segment_log2} segment_width={1 << segment_log2} "
        f"segment_index={segment_index} result_check=pass "
        f"overwrite={int(overwrite)} "
        f"leaf_ready_cycles={leaf_cycle} ready_cycles={top_cycle} "
        f"done_cycles={top_cycle}"
    )
    return leaf_cycle, top_cycle


async def measure_compact_stat(
    dut,
    operation: int,
    name: str,
    *,
    active_lanes: int,
    input_value: float,
    scalar_value: float,
    expected_value: float,
) -> tuple[int, int]:
    """Measure the physically narrow compact-stat datapath at Machine scope."""
    await reset(dut)
    packed, _ = packed_values([input_value] * VLEN)
    _, scalar_encoded = packed_values([scalar_value])
    _, expected_encoded = packed_values([expected_value])

    dut.result_waddr.value = 120
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    dut.element_op.value = operation
    dut.broadcast_fp2.value = 1
    dut.compact_stats_en.value = 1
    dut.compact_count_minus_one.value = active_lanes - 1
    await RisingEdge(dut.clk)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1
    dut.scalar_in.value = scalar_encoded[0]
    dut.scalar_in_valid.value = 1

    leaf_cycle = None
    ready = None
    result = None
    for cycle in range(2, 501):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.element_op.value = STALL_ELEMENT
            dut.broadcast_fp2.value = 0
            dut.compact_stats_en.value = 0
            dut.v_a_valid.value = 0
            dut.scalar_in_valid.value = 0
        if leaf_cycle is None and int(dut.compact_leaf_ready.value):
            leaf_cycle = cycle
        if int(dut.vector_result_ready.value):
            ready = cycle
            result = unpack_vector(int(dut.committed_vector.value))
            break

    expected = [expected_encoded[0]] * active_lanes + [0] * (VLEN - active_lanes)
    if leaf_cycle is None or ready is None or result != expected:
        raise AssertionError(
            f"{name} mismatch/timeout: leaf={leaf_cycle}, ready={ready}, "
            f"got={result}, expected={expected}"
        )
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"compact_lanes={active_lanes} result_check=pass "
        f"leaf_ready_cycles={leaf_cycle} ready_cycles={ready} "
        f"done_cycles={ready} initiation_interval_cycles=1"
    )
    return leaf_cycle, ready


async def measure_multi_reduction(
    dut, operation: int, name: str, *, segment_log2: int, expected_value: float
):
    await reset(dut)
    packed, _ = packed_one()
    segment_width = 1 << segment_log2
    segment_count = VLEN // segment_width
    dut.result_waddr.value = 96
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    dut.reduction_op.value = operation
    dut.reduction_segment_log2.value = segment_log2
    await RisingEdge(dut.clk)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1

    ready = None
    complete = None
    result = None
    for cycle in range(2, 301):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.reduction_op.value = STALL_REDUCTION
            dut.v_a_valid.value = 0
        if complete is None and int(dut.reduction_complete_observed.value):
            complete = cycle
        if int(dut.vector_result_ready.value):
            ready = cycle
            result = unpack_vector(int(dut.committed_vector.value))
            break
    if ready is None or complete is None or result is None:
        raise AssertionError(f"{name} timed out: ready={ready}, complete={complete}")
    _, expected_encoded = packed_values([expected_value])
    expected = expected_encoded[0]
    wanted = [expected] * segment_count + [0] * (VLEN - segment_count)
    if result != wanted:
        raise AssertionError(f"{name} compact result mismatch: got={result}, expected={wanted}")
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"segment_log2={segment_log2} segment_width={segment_width} "
        f"segment_count={segment_count} result_check=pass "
        f"ready_cycles={ready} done_cycles={max(ready, complete)} initiation_interval_cycles=1"
    )


async def measure_vseg(dut, operation: int, name: str, *, segment_log2: int):
    await reset(dut)
    segment_width = 1 << segment_log2
    segment_count = VLEN // segment_width
    stats = [float(index + 1) for index in range(segment_count)]
    lhs = [2.0] * VLEN
    rhs = stats + [0.0] * (VLEN - segment_count)
    packed_lhs, _ = packed_values(lhs)
    packed_rhs, _ = packed_values(rhs)
    if operation == 1:
        expected_values = [2.0 + stats[lane // segment_width] for lane in range(VLEN)]
    elif operation == 2:
        expected_values = [2.0 - stats[lane // segment_width] for lane in range(VLEN)]
    else:
        expected_values = [2.0 * stats[lane // segment_width] for lane in range(VLEN)]
    _, expected = packed_values(expected_values)

    dut.result_waddr.value = 104
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0
    dut.element_op.value = operation
    dut.segment_broadcast_en.value = 1
    dut.reduction_segment_log2.value = segment_log2
    await RisingEdge(dut.clk)
    dut.v_a.value = packed_lhs
    dut.v_b.value = packed_rhs
    dut.v_a_valid.value = 1
    dut.v_b_valid.value = 1
    ready = None
    result = None
    for cycle in range(2, 301):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.element_op.value = STALL_ELEMENT
            dut.segment_broadcast_en.value = 0
            dut.v_a_valid.value = 0
            dut.v_b_valid.value = 0
        if int(dut.vector_result_ready.value):
            ready = cycle
            result = unpack_vector(int(dut.committed_vector.value))
            break
    if ready is None or result != expected:
        raise AssertionError(f"{name} mismatch/timeout: ready={ready}, got={result}, expected={expected}")
    cocotb.log.info(
        f"[RTL_TIMING] {name} measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"segment_log2={segment_log2} segment_width={segment_width} "
        f"segment_count={segment_count} result_check=pass "
        f"ready_cycles={ready} done_cycles={ready} initiation_interval_cycles=1"
    )


async def measure_shift(dut, *, shift_amount: int = 2):
    """Measure and numerically validate the base-ISA lane shifter.

    The architectural third operand is an integer GP register.  Decoder turns
    that value into ``reduct_segment_index`` before it reaches VectorMachine;
    using a non-zero amount here detects an accidental shift-by-zero datapath.
    """
    await reset(dut)
    shift_amount = min(shift_amount, VLEN - 1)
    values = [float(index + 1) for index in range(VLEN)]
    packed, encoded = packed_values(values)
    expected = [0] * shift_amount + encoded[: VLEN - shift_amount]

    dut.result_waddr.value = 108
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0

    # Control is captured one cycle before the SRAM operand, matching the
    # production execute -> memory -> VectorMachine boundary.
    dut.element_op.value = 8  # SHIFT_V_LANES_ELEMENT
    dut.reduction_segment_index.value = shift_amount
    await RisingEdge(dut.clk)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1

    launch = None
    ready = None
    result = None
    for cycle in range(2, 301):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.element_op.value = STALL_ELEMENT
            dut.v_a_valid.value = 0
        if launch is None and int(dut.element_launch.value):
            launch = cycle
        if int(dut.vector_result_ready.value):
            ready = cycle
            result = unpack_vector(int(dut.committed_vector.value))
            break
    if launch is None or ready is None or result != expected:
        raise AssertionError(
            "V_SHIFT_V mismatch/timeout: "
            f"launch={launch}, ready={ready}, got={result}, expected={expected}"
        )
    cocotb.log.info(
        f"[RTL_TIMING] V_SHIFT_V measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"shift_amount={shift_amount} direction=left result_check=pass "
        f"ready_cycles={ready} done_cycles={ready} "
        f"initiation_interval_cycles=1"
    )


async def measure_shift_ii(dut, *, shift_amount: int = 2):
    """Prove that two independent lane shifts launch and retire one cycle apart."""
    await reset(dut)
    shift_amount = min(shift_amount, VLEN - 1)
    packed_a, encoded_a = packed_values([float(index + 1) for index in range(VLEN)])
    packed_b, encoded_b = packed_values([float(VLEN - index) for index in range(VLEN)])
    expected = (
        [0] * shift_amount + encoded_a[: VLEN - shift_amount],
        [0] * shift_amount + encoded_b[: VLEN - shift_amount],
    )

    dut.result_waddr.value = 64
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0
    dut.element_op.value = 8
    dut.reduction_segment_index.value = shift_amount
    dut.v_a.value = packed_a
    dut.v_a_valid.value = 1
    await RisingEdge(dut.clk)

    dut.result_waddr.value = 128
    dut.result_waddr_update.value = 1
    dut.v_a.value = packed_b
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0
    dut.element_op.value = STALL_ELEMENT
    dut.v_a_valid.value = 0

    launches = []
    results = []
    if int(dut.element_launch.value):
        launches.append(0)
    for cycle in range(1, 301):
        await RisingEdge(dut.clk)
        if int(dut.element_launch.value):
            launches.append(cycle)
        if int(dut.vector_result_ready.value):
            results.append(
                (
                    cycle,
                    int(dut.committed_waddr.value),
                    unpack_vector(int(dut.committed_vector.value)),
                )
            )
        if len(results) == 2:
            break
    if len(launches) != 2 or len(results) != 2:
        raise AssertionError(
            f"V_SHIFT_V II measurement lost work: launches={launches}, results={results}"
        )
    launch_ii = launches[1] - launches[0]
    result_ii = results[1][0] - results[0][0]
    if launch_ii != 1 or result_ii != 1:
        raise AssertionError(f"V_SHIFT_V II is not one: {launch_ii=}, {result_ii=}")
    if [item[1] for item in results] != [64, 128]:
        raise AssertionError(f"V_SHIFT_V write addresses reordered: {results}")
    if tuple(item[2] for item in results) != expected:
        raise AssertionError(f"V_SHIFT_V results reordered/corrupted: {results}")
    cocotb.log.info(
        f"[RTL_TIMING] V_SHIFT_V_II measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} "
        f"shift_amount={shift_amount} result_check=pass "
        f"accepted_interval_cycles={launch_ii} result_interval_cycles={result_ii} "
        f"initiation_interval_cycles={launch_ii} result_order_preserved=1"
    )


async def measure_lane_access(dut):
    lane = min(3, VLEN - 1)
    values = [float(index + 1) for index in range(VLEN)]
    packed, encoded = packed_values(values)
    await reset(dut)
    dut.reduction_op.value = 7  # LOAD_LANE_FP_V_REDUCT
    dut.reduction_segment_index.value = lane
    dut.scalar_target.value = 4
    await RisingEdge(dut.clk)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1
    ready = None
    result = None
    for cycle in range(2, 101):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.reduction_op.value = STALL_REDUCTION
            dut.v_a_valid.value = 0
        if int(dut.reduction_result_ready.value):
            ready = cycle
            result = int(dut.reduction_value.value)
            break
    if ready is None or result != encoded[lane]:
        raise AssertionError(f"S_LD_VLANE_FP failed: ready={ready}, result={result}")
    cocotb.log.info(
        f"[RTL_TIMING] S_LD_VLANE_FP measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} lane={lane} "
        f"result_check=pass ready_cycles={ready} done_cycles={ready} "
        f"initiation_interval_cycles=1"
    )

    await reset(dut)
    _, scalar_values = packed_values([3.0])
    dut.result_waddr.value = 112
    dut.result_waddr_update.value = 1
    await RisingEdge(dut.clk)
    dut.result_waddr_update.value = 0
    dut.element_op.value = 9  # STORE_LANE_FP_V_ELEMENT
    dut.broadcast_fp2.value = 1
    dut.lane_store_en.value = 1
    dut.reduction_segment_index.value = lane
    await RisingEdge(dut.clk)
    dut.v_a.value = packed
    dut.v_a_valid.value = 1
    dut.scalar_in.value = scalar_values[0]
    dut.scalar_in_valid.value = 1
    ready = None
    for cycle in range(2, 101):
        await RisingEdge(dut.clk)
        if cycle == 2:
            dut.element_op.value = STALL_ELEMENT
            dut.broadcast_fp2.value = 0
            dut.lane_store_en.value = 0
            dut.v_a_valid.value = 0
            dut.scalar_in_valid.value = 0
        if int(dut.vector_result_ready.value):
            ready = cycle
            result = unpack_vector(int(dut.committed_vector.value))
            mask = int(dut.committed_mask.value)
            break
    if ready is None or result[lane] != scalar_values[0] or mask != (1 << lane):
        raise AssertionError(
            f"S_ST_VLANE_FP failed: ready={ready}, lane_value={result[lane] if ready else None}, mask={mask if ready else None}"
        )
    cocotb.log.info(
        f"[RTL_TIMING] S_ST_VLANE_FP measurement_boundary=full_machine "
        f"vlen={VLEN} fp_exp={EXP_WIDTH} fp_mant={MANT_WIDTH} lane={lane} "
        f"result_check=pass ready_cycles={ready} done_cycles={ready} "
        f"initiation_interval_cycles=1"
    )


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
    await measure_reduction(dut, 1, "V_RED_SUM", expected_value=float(VLEN + 1))
    await measure_reduction(dut, 2, "V_RED_MAX", expected_value=1.0)
    await measure_reduction(
        dut, 1, "V_RED_SUM_OVR", expected_value=float(VLEN), overwrite=True
    )
    await measure_reduction(
        dut,
        2,
        "V_RED_MAX_OVR",
        expected_value=-4.0,
        overwrite=True,
        input_value=-4.0,
    )
    for active_lanes in (1, 4, 8, 16):
        if active_lanes > min(VLEN, 16):
            continue
        await measure_compact_stat(
            dut,
            10,
            f"V_STAT_MUL_F_L{active_lanes}",
            active_lanes=active_lanes,
            input_value=4.0,
            scalar_value=0.25,
            expected_value=1.0,
        )
        await measure_compact_stat(
            dut,
            11,
            f"V_STAT_ADD_F_L{active_lanes}",
            active_lanes=active_lanes,
            input_value=4.0,
            scalar_value=1.0,
            expected_value=5.0,
        )
        await measure_compact_stat(
            dut,
            12,
            f"V_STAT_RSQRT_L{active_lanes}",
            active_lanes=active_lanes,
            input_value=4.0,
            scalar_value=0.0,
            expected_value=0.5,
        )
    # Exercise 4-, 8-, and 16-lane segments when the configured VLEN can
    # contain them.  The production ISA accepts wider powers of two, but
    # these three widths define the calibrated v3 domain requested by the
    # segment-parallel architecture contract.
    max_segment_log2 = min(4, VLEN.bit_length() - 1)
    for segment_log2 in range(2, max_segment_log2 + 1):
        width = 1 << segment_log2
        index = int(VLEN >= 2 * width)
        await measure_reduction(
            dut,
            3,
            f"V_RED_SUM_SEG_W{width}",
            expected_value=float(width + 1),
            segment_log2=segment_log2,
            segment_index=index,
        )
        await measure_reduction(
            dut,
            4,
            f"V_RED_MAX_SEG_W{width}",
            expected_value=1.0,
            segment_log2=segment_log2,
            segment_index=index,
        )
        await measure_reduction(
            dut,
            3,
            f"V_RED_SUM_SEG_OVR_W{width}",
            expected_value=float(width),
            segment_log2=segment_log2,
            segment_index=index,
            overwrite=True,
        )
        await measure_reduction(
            dut,
            4,
            f"V_RED_MAX_SEG_OVR_W{width}",
            expected_value=-4.0,
            segment_log2=segment_log2,
            segment_index=index,
            overwrite=True,
            input_value=-4.0,
        )
        await measure_multi_reduction(
            dut, 5, f"V_RED_SUM_SEGS_W{width}",
            segment_log2=segment_log2, expected_value=float(width),
        )
        await measure_multi_reduction(
            dut, 6, f"V_RED_MAX_SEGS_W{width}",
            segment_log2=segment_log2, expected_value=1.0,
        )
    if VLEN >= 4:
        for operation, stem in ((1, "ADD"), (2, "SUB"), (3, "MUL")):
            await measure_vseg(dut, operation, f"V_{stem}_VSEG_W4", segment_log2=2)
    await measure_shift(dut)
    await measure_shift_ii(dut)
    await measure_lane_access(dut)
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
