"""Usable bandwidth, capacity and transfer granularity of a configured DRAM stack.

Ported from DeepStack (Mo et al., arXiv:2604.04750), tile-ai/DeepStack@8509061:

* ``dram_connectivity_efficiency`` and the bandwidth/capacity derivation of
  ``ConfigurableStackedGpu.update_ddr`` in
  ``src/deepstack/mosaic/arch/custom_profile.py``;
* the Little's-law cap of ``apply_littles_law`` and the thermal frequency scale of
  ``compute_thermal_freq_scale`` (caller-policy branch) in
  ``src/deepstack/mosaic/dse_space/case_study_dram_layer/dram_layer_config.py``;
* DRAM wave quantisation as DeepStack applies it to DRAM read and write traffic
  (``src/tilesight/tilesight/fused_op_dtype_wave/matmul_fused_op_new_api_wave.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar

from ._validate import fraction, positive_int
from .config import BufferingPolicy, DramEnergyConfig, StackedDramConfig, ThermalPolicy


def dram_connectivity_efficiency(
    total_layers: int,
    connected_layers: int,
    *,
    fully_connected_efficiency: float,
) -> float:
    """Fraction of peak bandwidth a stack sustains for a given vertical connectivity.

    While at most half of the layers are connected, row recharge in one layer is
    hidden behind streaming from another and the stack runs at full efficiency.
    From there the efficiency falls linearly to ``fully_connected_efficiency``
    when every layer is connected.
    """

    total = positive_int(total_layers, "total_layers")
    connected = positive_int(connected_layers, "connected_layers")
    if connected > total:
        raise ValueError("connected_layers must not exceed total_layers")
    endpoint = fraction(fully_connected_efficiency, "fully_connected_efficiency")
    half = total / 2.0
    if connected <= half:
        return 1.0
    if connected >= total:
        return endpoint
    position = (connected - half) / (total - half)
    return 1.0 + position * (endpoint - 1.0)


@dataclass(frozen=True)
class LittlesLawResult:
    limited: bool
    round_trip_latency_seconds: float
    required_buffer_bytes_per_requester: float
    buffer_bytes_per_requester: int
    bandwidth_bytes_per_s: float


@dataclass(frozen=True)
class StackedDramModel:
    """A ``StackedDramConfig`` plus the optional policies that bound its use."""

    name: str
    config: StackedDramConfig
    buffering: BufferingPolicy | None = None
    thermal: ThermalPolicy | None = None
    energy: DramEnergyConfig | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "stacked_dram"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        expected = (
            ("config", StackedDramConfig),
            ("buffering", BufferingPolicy),
            ("thermal", ThermalPolicy),
            ("energy", DramEnergyConfig),
        )
        for attribute, cls in expected:
            value = getattr(self, attribute)
            if value is None and attribute != "config":
                continue
            if not isinstance(value, cls):
                raise TypeError(f"{attribute} must be a {cls.__name__}")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def peak_bandwidth_bytes_per_s(self) -> float:
        return self.config.peak_bandwidth_bytes_per_s()

    @property
    def connectivity_efficiency(self) -> float:
        return dram_connectivity_efficiency(
            self.config.total_layers,
            self.config.connected_layers,
            fully_connected_efficiency=self.config.resolved_fully_connected_efficiency,
        )

    @property
    def effective_bandwidth_bytes_per_s(self) -> float:
        """Peak bandwidth after the connectivity efficiency (DeepStack's ``ddr_bandwidth``)."""

        return self.peak_bandwidth_bytes_per_s * self.connectivity_efficiency

    def littles_law(self) -> LittlesLawResult | None:
        if self.buffering is None:
            return None
        policy = self.buffering
        latency_s = policy.round_trip_latency_seconds(self.config.bank_timing)
        bandwidth = self.effective_bandwidth_bytes_per_s
        required = bandwidth / policy.requesters * latency_s * policy.buffering_factor
        available = policy.buffer_bytes_per_requester
        limited = available < required
        if limited:
            bandwidth *= available / required
        return LittlesLawResult(
            limited=limited,
            round_trip_latency_seconds=latency_s,
            required_buffer_bytes_per_requester=required,
            buffer_bytes_per_requester=available,
            bandwidth_bytes_per_s=bandwidth,
        )

    @property
    def usable_bandwidth_bytes_per_s(self) -> float:
        bound = self.littles_law()
        return self.effective_bandwidth_bytes_per_s if bound is None else bound.bandwidth_bytes_per_s

    @property
    def capacity_bytes(self) -> int:
        return self.config.capacity_bytes

    @property
    def compute_frequency_scale(self) -> float:
        if self.thermal is None:
            return 1.0
        return self.thermal.frequency_scale(self.config.total_layers)

    def quantize_bytes(self, num_bytes: float) -> float:
        """Round one transfer up to whole DRAM waves when wave quantisation is on."""

        if not self.config.apply_wave_quantization or num_bytes <= 0:
            return float(num_bytes)
        wave = self.config.wave_bytes
        return math.ceil(num_bytes / wave) * float(wave)

    def energy_pj(self, read_bytes: float, write_bytes: float) -> float | None:
        if self.energy is None:
            return None
        return self.energy.energy_pj(read_bytes, write_bytes)

    def with_layers(self, total_layers: int, connected_layers: int | None = None) -> StackedDramModel:
        return replace(self, config=self.config.with_layers(total_layers, connected_layers))

    def describe(self) -> dict[str, Any]:
        bound = self.littles_law()
        return {
            "name": self.name,
            "kind": self.kind,
            "total_layers": self.config.total_layers,
            "connected_layers": self.config.connected_layers,
            "peak_bandwidth_bytes_per_s": self.peak_bandwidth_bytes_per_s,
            "connectivity_efficiency": self.connectivity_efficiency,
            "effective_bandwidth_bytes_per_s": self.effective_bandwidth_bytes_per_s,
            "usable_bandwidth_bytes_per_s": self.usable_bandwidth_bytes_per_s,
            "littles_law": None
            if bound is None
            else {
                "limited": bound.limited,
                "round_trip_latency_seconds": bound.round_trip_latency_seconds,
                "required_buffer_bytes_per_requester": bound.required_buffer_bytes_per_requester,
                "buffer_bytes_per_requester": bound.buffer_bytes_per_requester,
            },
            "capacity_bytes": self.capacity_bytes,
            "transaction_bytes": self.config.transaction_bytes,
            "wave_bytes": self.config.wave_bytes,
            "apply_wave_quantization": self.config.apply_wave_quantization,
            "compute_frequency_scale": self.compute_frequency_scale,
            "energy": None
            if self.energy is None
            else {"read_pj_per_bit": self.energy.read_pj_per_bit, "write_pj_per_bit": self.energy.write_pj_per_bit},
            "provenance": dict(self.provenance),
        }
