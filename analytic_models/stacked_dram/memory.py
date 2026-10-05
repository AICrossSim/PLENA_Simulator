"""The memory interface the latency estimator prices DRAM traffic with.

Any object with this shape works; this package provides ``StackedDramModel``
(``model.py``) and ``FixedBandwidthMemory`` below, a memory described only by a
sustained bandwidth, for example an HBM baseline taken from a datasheet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from ._validate import finite_number, positive_int
from .config import DramEnergyConfig


class MemorySystem(Protocol):
    name: str
    kind: str

    @property
    def usable_bandwidth_bytes_per_s(self) -> float: ...

    @property
    def capacity_bytes(self) -> int | None: ...

    @property
    def compute_frequency_scale(self) -> float: ...

    def quantize_bytes(self, num_bytes: float) -> float: ...

    def energy_pj(self, read_bytes: float, write_bytes: float) -> float | None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FixedBandwidthMemory:
    """A memory that sustains ``bandwidth_bytes_per_s`` regardless of access pattern.

    ``capacity_bytes`` may be ``None`` when capacity is not part of the study.
    With ``transaction_bytes`` every transfer is rounded up to whole
    transactions.
    """

    name: str
    bandwidth_bytes_per_s: float
    capacity_bytes: int | None
    transaction_bytes: int | None = None
    energy: DramEnergyConfig | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "fixed_bandwidth"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        finite_number(self.bandwidth_bytes_per_s, "bandwidth_bytes_per_s")
        if self.capacity_bytes is not None:
            positive_int(self.capacity_bytes, "capacity_bytes")
        if self.transaction_bytes is not None:
            positive_int(self.transaction_bytes, "transaction_bytes")
        if self.energy is not None and not isinstance(self.energy, DramEnergyConfig):
            raise TypeError("energy must be a DramEnergyConfig")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def usable_bandwidth_bytes_per_s(self) -> float:
        return float(self.bandwidth_bytes_per_s)

    @property
    def compute_frequency_scale(self) -> float:
        return 1.0

    def quantize_bytes(self, num_bytes: float) -> float:
        if self.transaction_bytes is None or num_bytes <= 0:
            return float(num_bytes)
        return math.ceil(num_bytes / self.transaction_bytes) * float(self.transaction_bytes)

    def energy_pj(self, read_bytes: float, write_bytes: float) -> float | None:
        if self.energy is None:
            return None
        return self.energy.energy_pj(read_bytes, write_bytes)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "usable_bandwidth_bytes_per_s": self.usable_bandwidth_bytes_per_s,
            "capacity_bytes": self.capacity_bytes,
            "transaction_bytes": self.transaction_bytes,
            "compute_frequency_scale": self.compute_frequency_scale,
            "energy": None
            if self.energy is None
            else {"read_pj_per_bit": self.energy.read_pj_per_bit, "write_pj_per_bit": self.energy.write_pj_per_bit},
            "provenance": dict(self.provenance),
        }
