"""Independent parity checks for compiler-trace latency reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from compiler.aten.isa_builder import final_sequence
from compiler.aten.program_sink import CostTrace, SymbolicCostSink

from .schemas import ComputeLatencyReport, MemoryLatencyReport


@dataclass(frozen=True)
class AssemblyParityReport:
    exact: bool
    trace_counts: dict[str, int]
    assembly_counts: dict[str, int]
    mismatches: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class EmulatorParityReport:
    exact: bool
    expected_instruction_count: int
    observed_instruction_count: int
    expected_resource_picos: dict[str, int]
    observed_resource_picos: dict[str, int]
    mismatches: dict[str, tuple[int, int]]


def dynamic_assembly_opcode_counts(assembly: str) -> Counter[str]:
    """Count emulator-visible dynamic instructions in rendered ASM.

    This reparses the text emitted for the emulator. It is independent of the
    CostTrace attached to the original typed schedule and therefore catches
    renderer, loop-bound, and opcode-selection drift.
    """

    sink = SymbolicCostSink(default_stage="assembly-oracle")
    sink.consume(final_sequence(assembly))
    return sink.dynamic_opcode_counts()


def validate_detailed_trace_against_assembly(
    trace: CostTrace,
    assembly: str,
    *,
    raise_on_mismatch: bool = True,
) -> AssemblyParityReport:
    if trace.metadata.get("ordered_schedule_available") is not True:
        raise ValueError("assembly parity requires a detailed ordered CostTrace")
    trace_counts = trace.dynamic_opcode_counts
    assembly_counts = dynamic_assembly_opcode_counts(assembly)
    mismatches = {
        opcode: (trace_counts.get(opcode, 0), assembly_counts.get(opcode, 0))
        for opcode in sorted(set(trace_counts) | set(assembly_counts))
        if trace_counts.get(opcode, 0) != assembly_counts.get(opcode, 0)
    }
    if mismatches and raise_on_mismatch:
        detail = ", ".join(
            f"{opcode}: trace={counts[0]}, asm={counts[1]}"
            for opcode, counts in mismatches.items()
        )
        raise ValueError(f"CostTrace/ASM dynamic opcode mismatch: {detail}")
    return AssemblyParityReport(
        exact=not mismatches,
        trace_counts=dict(sorted(trace_counts.items())),
        assembly_counts=dict(sorted(assembly_counts.items())),
        mismatches=mismatches,
    )


def validate_compute_against_emulator_profile(
    trace: CostTrace,
    compute: ComputeLatencyReport,
    profile: Mapping[str, Any],
    *,
    memory: MemoryLatencyReport | None = None,
    raise_on_mismatch: bool = True,
) -> EmulatorParityReport:
    """Compare a report with main's runtime stage-profiler JSON.

    The Rust profiler classifies control instructions in its scalar bucket,
    whereas the public analytical report keeps control separate. The two are
    combined here before comparison. DMA timing remains owned by the memory
    backend; when a memory report is supplied, physical byte totals are also
    required to match.
    """

    expected_instruction_count = sum(
        instruction.multiplicity for instruction in trace.instructions
    )
    observed_instruction_count = int(profile["total_instructions_executed"])
    raw_resources = profile["total_resource_proxy_picos"]
    observed_resources = {
        name: int(raw_resources.get(name, 0))
        for name in ("matrix", "vector", "scalar", "dma", "other")
    }
    expected_resources = {
        "matrix": int(compute.by_resource_picos.get("matrix", 0)),
        "vector": int(compute.by_resource_picos.get("vector", 0)),
        "scalar": int(compute.by_resource_picos.get("scalar", 0))
        + int(compute.by_resource_picos.get("control", 0)),
        "other": 0,
    }
    mismatches: dict[str, tuple[int, int]] = {}
    if expected_instruction_count != observed_instruction_count:
        mismatches["dynamic_instruction_count"] = (
            expected_instruction_count,
            observed_instruction_count,
        )
    for name, expected in expected_resources.items():
        observed = observed_resources[name]
        if expected != observed:
            mismatches[f"{name}_picos"] = (expected, observed)
    if memory is not None:
        expected_read = memory.physical_read_bytes
        expected_write = memory.physical_write_bytes
        observed_read = int(profile["total_hbm_bytes_read"])
        observed_write = int(profile["total_hbm_bytes_written"])
        if expected_read != observed_read:
            mismatches["physical_hbm_read_bytes"] = (expected_read, observed_read)
        if expected_write != observed_write:
            mismatches["physical_hbm_write_bytes"] = (expected_write, observed_write)
    if mismatches and raise_on_mismatch:
        detail = ", ".join(
            f"{name}: expected={values[0]}, observed={values[1]}"
            for name, values in mismatches.items()
        )
        raise ValueError(f"CostEmitter/emulator profile mismatch: {detail}")
    return EmulatorParityReport(
        exact=not mismatches,
        expected_instruction_count=expected_instruction_count,
        observed_instruction_count=observed_instruction_count,
        expected_resource_picos=expected_resources,
        observed_resource_picos=observed_resources,
        mismatches=mismatches,
    )


__all__ = [
    "AssemblyParityReport",
    "EmulatorParityReport",
    "dynamic_assembly_opcode_counts",
    "validate_compute_against_emulator_profile",
    "validate_detailed_trace_against_assembly",
]
