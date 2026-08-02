"""Memory-provider interfaces used by the compiler-derived latency model."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import math
from typing import Protocol

from compiler.aten.program_sink import CostTrace, TraceDma

from .schemas import MemoryLatencyReport


class MemoryProvider(Protocol):
    name: str

    def estimate(self, trace: CostTrace) -> MemoryLatencyReport: ...


def _line_rounded_bytes(event: TraceDma, line_bytes: int) -> int:
    transfer = event.transfer
    rows = transfer.write_amount if transfer.direction == "write" else transfer.amount
    element_payload = transfer.dim * transfer.element_bytes
    element_bytes = rows * math.ceil(element_payload / line_bytes) * line_bytes
    scale_bytes = 0
    if transfer.scale_base_bytes is not None:
        # Main's current MX formats use one scale byte per eight elements.
        scale_payload = math.ceil(transfer.dim / 8) * transfer.element_bytes
        scale_bytes = rows * math.ceil(scale_payload / line_bytes) * line_bytes
    return (element_bytes + scale_bytes) * event.multiplicity


class ConfiguredBandwidthMemoryProvider:
    """Simple line-rounded bandwidth floor retained until HBM V4 is selected."""

    name = "configured-bandwidth-v1"

    def __init__(self, bandwidth_gbps: float, *, line_bytes: int = 64):
        if bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if line_bytes <= 0:
            raise ValueError("line_bytes must be positive")
        self.bandwidth_gbps = Fraction(str(bandwidth_gbps))
        self.line_bytes = line_bytes

    def estimate(self, trace: CostTrace) -> MemoryLatencyReport:
        by_stage_bytes: Counter[str] = Counter()
        read_bytes = 0
        write_bytes = 0
        for event in trace.dma_events:
            if not event.stage:
                raise ValueError("DMA event has no stage ownership")
            physical_bytes = _line_rounded_bytes(event, self.line_bytes)
            by_stage_bytes[event.stage] += physical_bytes
            if event.transfer.direction == "read":
                read_bytes += physical_bytes
            elif event.transfer.direction == "write":
                write_bytes += physical_bytes
            else:
                raise ValueError(f"unknown DMA direction {event.transfer.direction!r}")

        by_stage_picos = {
            stage: math.ceil(Fraction(byte_count * 1_000, 1) / self.bandwidth_gbps)
            for stage, byte_count in sorted(by_stage_bytes.items())
        }
        return MemoryLatencyReport(
            total_picos=sum(by_stage_picos.values()),
            by_stage_picos=by_stage_picos,
            physical_read_bytes=read_bytes,
            physical_write_bytes=write_bytes,
            provider=self.name,
            provenance={
                "provider": self.name,
                "bandwidth_gbps_decimal": float(self.bandwidth_gbps),
                "line_bytes": self.line_bytes,
                "traffic_semantics": "per-transfer-line-rounded",
                "latency_semantics": "configured-bandwidth-floor",
                "channel_startup_row_state_modeled": False,
            },
            warnings=(
                "Configured-bandwidth memory is a compatibility floor; select HBM V4 for channel and row-state effects.",
            ),
        )


__all__ = ["ConfiguredBandwidthMemoryProvider", "MemoryProvider"]
