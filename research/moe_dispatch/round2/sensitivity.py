"""Shared weight-frontend sensitivity, independent of datapath issue rate.

Tau is the service time of one 4096-byte BF16 operand block at the shared
SRAM-to-array frontend. It caps aggregate installed W bandwidth; it does
not give each core a full extra frontend or slow its arithmetic issue II.
The universal search bound ignores this extra cap, hence remains legal
but can be looser. Certificates record this module's hash for safe resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

from .model import Design, Parameters


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SensitivityParameters(Parameters):
    weight_tile_service_cycles: float = 1.0
    timing_model_sha256: str = field(default_factory=source_sha256)

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.weight_tile_service_cycles) or self.weight_tile_service_cycles <= 0:
            raise ValueError("shared frontend service time must be finite and positive")
        if self.onchip_mode != "pipelined" or self.tile_issue_cycles != 1.0:
            raise ValueError("frontend sensitivity keeps arithmetic issue II at one")
        if self.timing_model_sha256 != source_sha256():
            raise ValueError("sensitivity timing source hash changed; cannot resume old certificate")

    def w_bandwidth(self, design: Design, c: int) -> float:
        total = min(64 * self.bank_Bpc, 4096.0 / self.weight_tile_service_cycles)
        return total * design.w_banks[c] / 64.0


def parameters_from_dict(values: dict) -> Parameters:
    """Reconstruct a frozen certificate without silently discarding tau."""
    cls = SensitivityParameters if "weight_tile_service_cycles" in values else Parameters
    return cls(**values)
