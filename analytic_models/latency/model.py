"""Composition of compute and memory latency providers."""

from __future__ import annotations

from typing import Any

from compiler.aten.program_sink import CostTrace

from .compute import estimate_compute_latency
from .memory import MemoryProvider
from .schemas import ComputeLatencyReport, LatencyReport, MemoryLatencyReport, StageLatency
from .timing import TimingProvider


def estimate_latency(
    trace: CostTrace,
    compute_provider: ComputeLatencyReport | str | TimingProvider,
    memory_provider: MemoryLatencyReport | MemoryProvider,
    overlap_policy: str = "stage-roofline",
    *,
    hardware_config: Any = None,
) -> LatencyReport:
    """Compose provider reports without changing either provider's semantics."""

    compute = (
        compute_provider
        if isinstance(compute_provider, ComputeLatencyReport)
        else estimate_compute_latency(trace, hardware_config, compute_provider)
    )
    memory = (
        memory_provider
        if isinstance(memory_provider, MemoryLatencyReport)
        else memory_provider.estimate(trace)
    )
    if overlap_policy not in {"stage-roofline", "serial"}:
        raise ValueError(f"unsupported overlap policy {overlap_policy!r}")

    stages = []
    for stage in sorted(set(compute.by_stage_picos) | set(memory.by_stage_picos)):
        compute_picos = compute.by_stage_picos.get(stage, 0)
        memory_picos = memory.by_stage_picos.get(stage, 0)
        roofline_picos = (
            max(compute_picos, memory_picos)
            if overlap_policy == "stage-roofline"
            else compute_picos + memory_picos
        )
        stages.append(
            StageLatency(
                stage=stage,
                compute_picos=compute_picos,
                memory_picos=memory_picos,
                roofline_picos=roofline_picos,
            )
        )
    serial_total = compute.total_picos + memory.total_picos
    total = sum(stage.roofline_picos for stage in stages)
    return LatencyReport(
        total_picos=total,
        serial_total_picos=serial_total,
        stages=tuple(stages),
        compute=compute,
        memory=memory,
        overlap_policy=overlap_policy,
        provenance={
            "trace_schema": trace.schema_version,
            "trace_isa_hash": trace.isa_hash,
            "compiler_hash": trace.compiler_hash,
        },
        warnings=tuple(compute.warnings) + tuple(memory.warnings),
    )


__all__ = ["estimate_latency"]
