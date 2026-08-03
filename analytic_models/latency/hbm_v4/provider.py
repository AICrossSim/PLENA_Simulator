"""CostTrace adapter for the production-DMA HBM V4 model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from compiler.aten.isa_builder import DmaTransfer, RepeatAxis
from compiler.aten.program_sink import CostTrace, TraceDma

from ..schemas import MemoryLatencyReport
from .model import HbmServiceModelV4
from .schema import HbmPrecisionConfig, HbmV4Config, plan_dma_request_manifest


def _axis_delta(axis: RepeatAxis, field: str) -> int:
    return dict(axis.deltas).get(field, 0)


def _validate_axes(event: TraceDma) -> None:
    represented = math.prod(axis.count for axis in event.repeat_axes) if event.repeat_axes else 1
    if represented != event.multiplicity:
        raise ValueError(
            f"DMA {event.transfer.opcode} at {event.stage} has multiplicity={event.multiplicity} "
            f"but repeat axes represent {represented} occurrences"
        )


def _iter_occurrence_transfers(event: TraceDma) -> Iterator[DmaTransfer]:
    _validate_axes(event)
    if not event.repeat_axes:
        yield event.transfer
        return
    for linear_index in range(event.multiplicity):
        remainder = linear_index
        element_delta = 0
        scale_delta = 0
        for axis in reversed(event.repeat_axes):
            index = remainder % axis.count
            remainder //= axis.count
            element_delta += index * _axis_delta(axis, "element_base_bytes")
            scale_delta += index * _axis_delta(axis, "scale_base_bytes")
        if remainder:
            raise ValueError("DMA repeat axes did not consume the occurrence index")
        yield replace(
            event.transfer,
            element_base_bytes=event.transfer.element_base_bytes + element_delta,
            scale_base_bytes=(
                None
                if event.transfer.scale_base_bytes is None
                else event.transfer.scale_base_bytes + scale_delta
            ),
            axes=(),
        )


class HbmV4MemoryProvider:
    name = "hbm-v4-production-dma"

    def __init__(
        self,
        model: HbmServiceModelV4,
        precision: HbmPrecisionConfig,
        config: HbmV4Config,
        *,
        aggregation: str = "scalar",
        fail_on_extrapolation: bool = False,
    ) -> None:
        if aggregation not in {"scalar", "stateful"}:
            raise ValueError("scalar HBM V4 backend supports aggregation='scalar' or 'stateful'")
        self.model = model
        self.precision = precision
        self.config = config
        self.aggregation = aggregation
        self.fail_on_extrapolation = fail_on_extrapolation

    def estimate(self, trace: CostTrace) -> MemoryLatencyReport:
        by_stage_ns: defaultdict[str, list[float]] = defaultdict(list)
        by_stage_floor_ns: defaultdict[str, list[float]] = defaultdict(list)
        by_opcode_ns: defaultdict[str, list[float]] = defaultdict(list)
        traffic: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        read_bytes = write_bytes = payload_read = payload_write = 0
        read_requests = write_requests = 0
        issues: Counter[str] = Counter()
        regimes: Counter[str] = Counter()
        occurrence_count = 0
        open_rows = (
            np.full(self.config.channels * 32, -1, dtype=np.int64)
            if self.aggregation == "stateful"
            else None
        )
        for event in trace.dma_events:
            if not event.stage:
                raise ValueError("DMA event has no stage ownership")
            fmt = self.precision.for_transfer(event.transfer)
            for transfer in _iter_occurrence_transfers(event):
                manifest = plan_dma_request_manifest(transfer, fmt)
                prediction = self.model.predict(
                    transfer.opcode,
                    transfer,
                    fmt,
                    self.config,
                    manifest,
                    open_rows=open_rows,
                )
                if self.fail_on_extrapolation and not prediction.calibration_in_domain:
                    raise ValueError(
                        f"HBM V4 occurrence is outside calibration domain: {prediction.domain_issues}"
                    )
                by_stage_ns[event.stage].append(prediction.latency_ns)
                by_stage_floor_ns[event.stage].append(prediction.theoretical_phase_floor_ns)
                by_opcode_ns[transfer.opcode].append(prediction.latency_ns)
                read_bytes += manifest.read_bytes
                write_bytes += manifest.write_bytes
                payload_read += manifest.payload_read_bytes
                payload_write += manifest.payload_write_bytes
                read_requests += len(manifest.read_lines)
                write_requests += len(manifest.write_lines)
                role = transfer.role
                traffic[event.stage][role].update(
                    physical_read_bytes=manifest.read_bytes,
                    physical_write_bytes=manifest.write_bytes,
                    payload_read_bytes=manifest.payload_read_bytes,
                    payload_write_bytes=manifest.payload_write_bytes,
                    read_requests=len(manifest.read_lines),
                    write_requests=len(manifest.write_lines),
                )
                issues.update(prediction.domain_issues)
                regimes[prediction.row_state_regime] += 1
                occurrence_count += 1

        by_stage_picos = {
            stage: round(math.fsum(values) * 1_000) for stage, values in sorted(by_stage_ns.items())
        }
        floor_picos = {
            stage: round(math.fsum(values) * 1_000)
            for stage, values in sorted(by_stage_floor_ns.items())
        }
        warnings = tuple(
            [f"HBM V4 extrapolated {sum(issues.values())} feature observations across {len(issues)} issues"]
            if issues
            else []
        )
        return MemoryLatencyReport(
            total_picos=sum(by_stage_picos.values()),
            by_stage_picos=by_stage_picos,
            physical_read_bytes=read_bytes,
            physical_write_bytes=write_bytes,
            payload_read_bytes=payload_read,
            payload_write_bytes=payload_write,
            read_requests=read_requests,
            write_requests=write_requests,
            by_opcode_picos={
                opcode: round(math.fsum(values) * 1_000)
                for opcode, values in sorted(by_opcode_ns.items())
            },
            by_stage_floor_picos=floor_picos,
            traffic_breakdown={
                stage: {role: dict(values) for role, values in sorted(roles.items())}
                for stage, roles in sorted(traffic.items())
            },
            provider=self.name,
            provenance={
                "provider": self.name,
                "calibration_id": self.model.calibration_id,
                "aggregation": self.aggregation,
                "memory_config": {
                    "channels": self.config.channels,
                    "request_bytes": self.config.request_bytes,
                    "physical_burst_bytes": self.config.physical_burst_bytes,
                    "channel_bandwidth_bytes_per_ns": self.config.channel_bandwidth_bytes_per_ns,
                    "mapper": self.config.mapper,
                    "preset": self.config.preset,
                },
                "precision": self.precision.to_dict(),
                "occurrence_count": occurrence_count,
                "row_state_regime_counts": dict(regimes),
                "calibration_in_domain": not issues,
                "domain_issues": dict(issues),
                "latency_semantics": "calibrated-production-dma-occurrence-service",
                "queue_overlap_modeled": False,
            },
            warnings=warnings,
        )


def estimate_hbm_v4(
    trace: CostTrace,
    memory_config: HbmV4Config,
    precision: HbmPrecisionConfig,
    calibration: str | Path | HbmServiceModelV4,
    *,
    aggregation: str = "scalar",
    fail_on_extrapolation: bool = False,
) -> MemoryLatencyReport:
    model = calibration if isinstance(calibration, HbmServiceModelV4) else HbmServiceModelV4.load(calibration)
    return HbmV4MemoryProvider(
        model,
        precision,
        memory_config,
        aggregation=aggregation,
        fail_on_extrapolation=fail_on_extrapolation,
    ).estimate(trace)


__all__ = ["HbmV4MemoryProvider", "estimate_hbm_v4"]
