"""Compiler-derived, stage-aware cost model for one PLENA decode program.

The input is the structured trace emitted with compiler assembly.  Dynamic
instruction counts therefore include compiler padding, address generation,
reductions, state movement, and compact hardware loops.  HBM transfers may
overlap compute inside a stage, while stage boundaries remain sequential:

    total_cycles = sum(max(compute_cycles, memory_cycles) for each stage)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from compiler.aten.execution_trace import (
    HBM_READ,
    HBM_WRITE,
    ExecutionTrace,
    ExecutionTraceEntry,
)

try:
    from .decode_timing import STEP_COMPOSITION
except ImportError:
    from decode_timing import STEP_COMPOSITION


@dataclass(frozen=True)
class StageCost:
    """Compute, memory, and overlapped cycle cost for one sequential stage."""

    stage: str
    compute_cycles: int
    memory_cycles: int
    cycles: int
    dynamic_instructions: int
    hbm_read_bytes: int
    hbm_write_bytes: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage cost requires a stage name")
        values = (
            self.compute_cycles,
            self.memory_cycles,
            self.cycles,
            self.dynamic_instructions,
            self.hbm_read_bytes,
            self.hbm_write_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("stage costs and traffic must be non-negative")
        if self.cycles != max(self.compute_cycles, self.memory_cycles):
            raise ValueError("stage cycles must use max(compute, memory)")

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def bottleneck(self) -> str:
        return "memory" if self.memory_cycles >= self.compute_cycles else "compute"

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "cycles": self.cycles,
            "dynamic_instructions": self.dynamic_instructions,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "bottleneck": self.bottleneck,
        }


@dataclass(frozen=True)
class DecodeCost:
    """Cost-model result bound to one structured compiler trace."""

    stages: tuple[StageCost, ...]
    trace_assembly_sha256: str
    step_composition: str = STEP_COMPOSITION

    def __post_init__(self) -> None:
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError(f"decode cost requires {STEP_COMPOSITION!r} composition")
        if len({stage.stage for stage in self.stages}) != len(self.stages):
            raise ValueError("decode cost contains duplicate stage names")
        if len(self.trace_assembly_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.trace_assembly_sha256
        ):
            raise ValueError("decode cost requires a trace assembly SHA-256")

    @property
    def total_cycles(self) -> int:
        return sum(stage.cycles for stage in self.stages)

    @property
    def compute_cycles(self) -> int:
        return sum(stage.compute_cycles for stage in self.stages)

    @property
    def memory_cycles(self) -> int:
        return sum(stage.memory_cycles for stage in self.stages)

    @property
    def hbm_read_bytes(self) -> int:
        return sum(stage.hbm_read_bytes for stage in self.stages)

    @property
    def hbm_write_bytes(self) -> int:
        return sum(stage.hbm_write_bytes for stage in self.stages)

    @property
    def stage_cycles(self) -> dict[str, int]:
        return {stage.stage: stage.cycles for stage in self.stages}

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_assembly_sha256": self.trace_assembly_sha256,
            "step_composition": self.step_composition,
            "total_cycles": self.total_cycles,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "stages": [stage.to_dict() for stage in self.stages],
        }


class DecodeCostModel:
    """Price a compiler execution trace with ISA latency and HBM contracts.

    ``memory_bandwidth_bytes_per_cycle`` may be one positive number or a map.
    A map is resolved in this order: ``direction:precision``, precision,
    direction, then ``default``.  This permits the existing calibrated traffic
    classes to be connected without changing the trace or composition rule.
    """

    def __init__(
        self,
        instruction_latencies: Mapping[str, int],
        *,
        memory_bandwidth_bytes_per_cycle: float | Mapping[str, float],
        dma_latency_cycles: int = 0,
        expected_geometry: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._instruction_latencies = {
            str(opcode): int(cycles)
            for opcode, cycles in instruction_latencies.items()
        }
        if not self._instruction_latencies or any(
            cycles <= 0 for cycles in self._instruction_latencies.values()
        ):
            raise ValueError("instruction latencies must be a non-empty positive map")
        if isinstance(memory_bandwidth_bytes_per_cycle, Mapping):
            bandwidth = {
                str(name): float(value)
                for name, value in memory_bandwidth_bytes_per_cycle.items()
            }
            if "default" not in bandwidth:
                raise ValueError("bandwidth maps require a default entry")
        else:
            bandwidth = {"default": float(memory_bandwidth_bytes_per_cycle)}
        if any(not math.isfinite(value) or value <= 0 for value in bandwidth.values()):
            raise ValueError("memory bandwidth must be finite and positive")
        if isinstance(dma_latency_cycles, bool) or dma_latency_cycles < 0:
            raise ValueError("DMA latency must be a non-negative integer")
        if expected_geometry is not None and (
            len(expected_geometry) != 4 or any(value <= 0 for value in expected_geometry)
        ):
            raise ValueError("expected geometry must be (MLEN, BLEN, VLEN, HLEN)")
        self._bandwidth = bandwidth
        self._dma_latency_cycles = int(dma_latency_cycles)
        self._expected_geometry = expected_geometry

    @classmethod
    def from_perf_model(
        cls,
        perf_model,
        *,
        memory_bandwidth_bytes_per_cycle: float | Mapping[str, float] | None = None,
        dma_latency_cycles: int = 0,
    ) -> "DecodeCostModel":
        """Build from the existing ``PerfModel`` latency and geometry contract."""

        if memory_bandwidth_bytes_per_cycle is None:
            hbm_width = getattr(perf_model.config, "HBM_WIDTH", None)
            if hbm_width is None or float(hbm_width) <= 0:
                raise ValueError(
                    "PerfModel configuration lacks a positive HBM_WIDTH; "
                    "provide memory bandwidth explicitly"
                )
            memory_bandwidth_bytes_per_cycle = float(hbm_width) / 8.0
        latencies = getattr(perf_model.instr, "latencies", None)
        if not isinstance(latencies, Mapping):
            latencies = dict(perf_model.instr.items())
        return cls(
            latencies,
            memory_bandwidth_bytes_per_cycle=memory_bandwidth_bytes_per_cycle,
            dma_latency_cycles=dma_latency_cycles,
            expected_geometry=(
                int(perf_model.mlen),
                int(perf_model.blen),
                int(perf_model.vlen),
                int(perf_model.hlen),
            ),
        )

    def _bandwidth_for(self, entry: ExecutionTraceEntry) -> float:
        for key in (
            f"{entry.dma_direction}:{entry.precision_mode}",
            entry.precision_mode,
            entry.dma_direction,
            "default",
        ):
            if key in self._bandwidth:
                return self._bandwidth[key]
        raise RuntimeError("bandwidth map resolution failed")

    def _entry_memory_cycles(self, entry: ExecutionTraceEntry) -> float:
        if not entry.dma_bytes:
            return 0.0
        transfer = entry.total_dma_bytes / self._bandwidth_for(entry)
        command_latency = entry.dynamic_count * self._dma_latency_cycles
        return transfer + command_latency

    def evaluate(self, trace: ExecutionTrace) -> DecodeCost:
        """Evaluate all stages, overlapping compute and HBM within each stage."""

        geometry = (trace.mlen, trace.blen, trace.vlen, trace.hlen)
        if self._expected_geometry is not None and geometry != self._expected_geometry:
            raise ValueError(
                "compiler trace geometry differs from the instruction-latency model: "
                f"trace={geometry}, model={self._expected_geometry}"
            )
        missing = sorted(
            {
                entry.opcode
                for entry in trace.entries
                if entry.opcode not in self._instruction_latencies
            }
        )
        if missing:
            raise ValueError(f"compiler trace contains unpriced opcodes {missing}")

        stage_entries = {
            stage: trace.entries_for_stage(stage)
            for stage in trace.stage_order
        }
        stages: list[StageCost] = []
        for stage, entries in stage_entries.items():
            compute_cycles = sum(
                entry.dynamic_count * self._instruction_latencies[entry.opcode]
                for entry in entries
            )
            memory_cycles = math.ceil(
                sum(self._entry_memory_cycles(entry) for entry in entries)
            )
            read_bytes = sum(
                entry.total_dma_bytes
                for entry in entries
                if entry.dma_direction == HBM_READ
            )
            write_bytes = sum(
                entry.total_dma_bytes
                for entry in entries
                if entry.dma_direction == HBM_WRITE
            )
            stages.append(
                StageCost(
                    stage=stage,
                    compute_cycles=compute_cycles,
                    memory_cycles=memory_cycles,
                    cycles=max(compute_cycles, memory_cycles),
                    dynamic_instructions=sum(
                        entry.dynamic_count for entry in entries
                    ),
                    hbm_read_bytes=read_bytes,
                    hbm_write_bytes=write_bytes,
                )
            )
        return DecodeCost(
            stages=tuple(stages),
            trace_assembly_sha256=trace.assembly_sha256,
        )


def validate_packed_q1_execution_trace(
    trace: ExecutionTrace,
    contract,
    *,
    cache_tokens: int,
) -> tuple[bool, str]:
    """Check a packed-q1 routine trace against its sealed timing point."""

    if contract is None:
        return False, "missing_packed_q1_timing_contract"
    try:
        point = contract.point(cache_tokens)
    except KeyError:
        return False, "packed_q1_cache_point_missing"
    observed = trace.opcode_histogram
    expected = dict(point.opcode_histogram)
    if observed != expected:
        return False, "packed_q1_execution_trace_histogram_mismatch"
    if trace.assembly_sha256 != point.assembly_sha256:
        return False, "packed_q1_execution_trace_assembly_mismatch"
    return True, "packed_q1_execution_trace_validated"


__all__ = [
    "DecodeCost",
    "DecodeCostModel",
    "StageCost",
    "validate_packed_q1_execution_trace",
]
