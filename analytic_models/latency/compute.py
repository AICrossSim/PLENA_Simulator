"""Fast compute-work evaluation over compiler symbolic schedules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from compiler.aten.program_sink import CostTrace, TraceInstruction

from .schemas import ComputeLatencyReport, OpcodeLatencyEntry
from .timing import (
    IdealII1TimingProvider,
    MainTimingConfig,
    MainTimingProvider,
    TimingProvider,
)


def _field(source: Any, lower: str, upper: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        if lower in source:
            return source[lower]
        if upper in source:
            return source[upper]
    if hasattr(source, lower):
        return getattr(source, lower)
    if hasattr(source, upper):
        return getattr(source, upper)
    return default


def _coerce_config(hardware_config: Any) -> MainTimingConfig:
    if isinstance(hardware_config, MainTimingConfig):
        return hardware_config
    if isinstance(hardware_config, (str, Path)):
        return MainTimingConfig.from_toml(hardware_config)

    required = {
        "mlen": _field(hardware_config, "mlen", "MLEN"),
        "blen": _field(hardware_config, "blen", "BLEN"),
        "vlen": _field(hardware_config, "vlen", "VLEN"),
        "hlen": _field(hardware_config, "hlen", "HLEN"),
        "broadcast_amount": _field(
            hardware_config, "broadcast_amount", "BROADCAST_AMOUNT"
        ),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise TypeError(
            "hardware_config must be MainTimingConfig, a settings TOML path, "
            f"or expose {', '.join(missing)}"
        )
    optional = {
        name: value
        for name in MainTimingConfig.__dataclass_fields__
        if name not in required
        and (value := _field(hardware_config, name, name.upper())) is not None
    }
    return MainTimingConfig(**required, **optional)


def _resolve_provider(
    timing_provider: str | TimingProvider,
    hardware_config: Any,
) -> TimingProvider:
    if not isinstance(timing_provider, str):
        return timing_provider
    config = _coerce_config(hardware_config)
    if timing_provider == "main":
        return MainTimingProvider(config)
    if timing_provider == "ideal-ii1":
        return IdealII1TimingProvider(config)
    raise ValueError(f"unsupported timing provider {timing_provider!r}")


def resource_for_opcode(opcode: str) -> str:
    if opcode.startswith("M_"):
        return "matrix"
    if opcode.startswith("V_"):
        return "vector"
    if opcode.startswith("S_"):
        return "scalar"
    if opcode.startswith("C_"):
        return "control"
    if opcode.startswith("H_"):
        return "memory"
    raise ValueError(f"unknown opcode resource for {opcode!r}")


def _validate_instruction(item: TraceInstruction) -> None:
    if not item.stage:
        raise ValueError(f"instruction {item.opcode} has no stage ownership")
    if item.multiplicity < 0:
        raise ValueError(f"instruction {item.opcode} has negative multiplicity")


def estimate_compute_latency(
    trace: CostTrace,
    hardware_config: Any,
    timing_provider: str | TimingProvider = "main",
) -> ComputeLatencyReport:
    """Evaluate exact serial compute work from a symbolic final schedule.

    Dynamic loops are already represented by ``multiplicity``. This function
    therefore scales with unique instruction variants, not dynamic op count.
    HBM opcodes are intentionally delegated to the selected memory backend.
    """

    provider = _resolve_provider(timing_provider, hardware_config)
    by_stage: Counter[str] = Counter()
    by_resource: Counter[str] = Counter()
    by_opcode: Counter[str] = Counter()
    entries: list[OpcodeLatencyEntry] = []
    compute_instructions = 0
    covered_instructions = 0

    for item in trace.instructions:
        _validate_instruction(item)
        resource = resource_for_opcode(item.opcode)
        if resource == "memory":
            continue
        compute_instructions += item.multiplicity
        per_instruction = provider.latency_picos(item, trace.metadata)
        if per_instruction < 0:
            raise ValueError(f"negative latency for {item.opcode}")
        total = per_instruction * item.multiplicity
        covered_instructions += item.multiplicity
        by_stage[item.stage] += total
        by_resource[resource] += total
        by_opcode[item.opcode] += total
        entries.append(
            OpcodeLatencyEntry(
                stage=item.stage,
                resource=resource,
                opcode=item.opcode,
                variant=item.variant,
                multiplicity=item.multiplicity,
                latency_per_instruction_picos=per_instruction,
                total_picos=total,
            )
        )

    total_picos = sum(by_stage.values())
    coverage = 1.0 if compute_instructions == 0 else covered_instructions / compute_instructions
    if coverage != 1.0:
        raise ValueError(f"incomplete compute timing coverage: {coverage:.6f}")
    return ComputeLatencyReport(
        total_picos=total_picos,
        by_stage_picos=dict(sorted(by_stage.items())),
        by_resource_picos=dict(sorted(by_resource.items())),
        by_opcode_picos=dict(sorted(by_opcode.items())),
        entries=tuple(entries),
        timing_provider=provider.name,
        timing_provenance={
            **provider.provenance(),
            "trace_schema": trace.schema_version,
            "trace_isa_hash": trace.isa_hash,
            "compiler_hash": trace.compiler_hash,
            "evaluation_semantics": "symbolic-multiplicity-resource-work",
        },
        instruction_coverage=coverage,
    )


__all__ = ["estimate_compute_latency", "resource_for_opcode"]
