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

    async def reset() -> None:
        dut.rst.value = 1
        dut.v_in_valid.value = 0
        dut.operation.value = 0
        dut.overwrite.value = 1
        dut.segment_log2.value = int(math.log2(vlen))
        dut.segment_index.value = 0
        for _ in range(3):
            await RisingEdge(dut.clk)
        dut.rst.value = 0
        for _ in range(2):
            await RisingEdge(dut.clk)

    for operation, name in ((1, "V_RED_SUM"), (2, "V_RED_MAX")):
        await reset()
        dut.v_in.value = packed
        dut.operation.value = operation
        dut.v_in_valid.value = 1
        for cycle in range(1, 501):
            await RisingEdge(dut.clk)
            dut.v_in_valid.value = 0
            if int(dut.s_out_valid.value):
                cocotb.log.info(
                    f"[RTL_TIMING] {name} vlen={vlen} "
                    f"tree_levels={math.ceil(math.log2(vector_dim))} cycles={cycle}"
                )
                single_latency = cycle
                break
        else:
            raise AssertionError(f"{name} timed out")

        await reset()
        dut.v_in.value = packed
        dut.operation.value = operation
        output_cycles = []
        accepted_cycles = []
        for cycle in range(1, 501):
            issue = cycle <= 4
            dut.v_in_valid.value = int(issue)
            if issue:
                accepted_cycles.append(cycle)
            await RisingEdge(dut.clk)
            if int(dut.s_out_valid.value):
                output_cycles.append(cycle)
                if len(output_cycles) == 4:
                    break
        if len(output_cycles) != 4:
            raise AssertionError(
                f"{name} independent-row pipeline produced {len(output_cycles)} outputs"
            )
        independent_iis = [
            right - left for left, right in zip(output_cycles, output_cycles[1:])
        ]
        if any(ii != 1 for ii in independent_iis):
            raise AssertionError(f"{name} independent row II is not one: {output_cycles}")

        await reset()
        dut.v_in.value = packed
        dut.operation.value = operation
        first_accept = None
        first_result = None
        second_accept = None
        second_result = None
        issue_next = True
        for cycle in range(1, 1001):
            dut.v_in_valid.value = int(issue_next)
            if issue_next:
                if first_accept is None:
                    first_accept = cycle
                elif second_accept is None:
                    second_accept = cycle
                issue_next = False
            await RisingEdge(dut.clk)
            if int(dut.s_out_valid.value):
                if first_result is None:
                    first_result = cycle
                    issue_next = True
                elif second_result is None:
                    second_result = cycle
                    break
        if None in (first_accept, first_result, second_accept, second_result):
            raise AssertionError(f"{name} dependent-row audit timed out")
        dependent_issue_interval = int(second_accept) - int(first_accept)
        dependent_result_interval = int(second_result) - int(first_result)
        cocotb.log.info(
            f"[RTL_PIPELINE_AUDIT] {name} vlen={vlen} "
            f"single_latency_cycles={single_latency} "
            f"independent_ii_cycles={max(independent_iis)} "
            f"dependent_issue_interval_cycles={dependent_issue_interval} "
            f"dependent_result_interval_cycles={dependent_result_interval} "
            "accepted_rows_per_cycle=1"
        )


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
