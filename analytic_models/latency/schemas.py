"""Stable report schemas shared by compute, memory, and power backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpcodeLatencyEntry:
    stage: str
    resource: str
    opcode: str
    variant: tuple[tuple[str, str], ...]
    multiplicity: int
    latency_per_instruction_picos: int
    total_picos: int


@dataclass(frozen=True)
class ComputeLatencyReport:
    total_picos: int
    by_stage_picos: dict[str, int]
    by_resource_picos: dict[str, int]
    by_opcode_picos: dict[str, int]
    entries: tuple[OpcodeLatencyEntry, ...]
    timing_provider: str
    timing_provenance: dict[str, Any]
    instruction_coverage: float
    warnings: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        return self.total_picos / 1_000_000_000_000

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seconds"] = self.seconds
        return result


@dataclass(frozen=True)
class MemoryLatencyReport:
    total_picos: int
    by_stage_picos: dict[str, int]
    physical_read_bytes: int
    physical_write_bytes: int
    provider: str
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        return self.total_picos / 1_000_000_000_000

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seconds"] = self.seconds
        return result


@dataclass(frozen=True)
class StageLatency:
    stage: str
    compute_picos: int
    memory_picos: int
    roofline_picos: int


@dataclass(frozen=True)
class LatencyReport:
    total_picos: int
    serial_total_picos: int
    stages: tuple[StageLatency, ...]
    compute: ComputeLatencyReport
    memory: MemoryLatencyReport
    overlap_policy: str
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        return self.total_picos / 1_000_000_000_000

    @property
    def serial_seconds(self) -> float:
        return self.serial_total_picos / 1_000_000_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_picos": self.total_picos,
            "seconds": self.seconds,
            "serial_total_picos": self.serial_total_picos,
            "serial_seconds": self.serial_seconds,
            "stages": [asdict(stage) for stage in self.stages],
            "compute": self.compute.to_dict(),
            "memory": self.memory.to_dict(),
            "overlap_policy": self.overlap_policy,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
        }
