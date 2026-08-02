"""Formal latency-energy objective schema for Qwen3 prefill exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OBJECTIVE_DIRECTIONS = ("minimize", "minimize")
OBJECTIVE_NORMALIZATION = "identity-no-a100-v1"


@dataclass(frozen=True)
class ObjectiveValues:
    normalized_latency: float
    normalized_energy: float

    def as_optuna_values(self) -> tuple[float, float]:
        return (
            float(self.normalized_latency),
            float(self.normalized_energy),
        )

    @classmethod
    def from_trial_record(cls, record: dict[str, Any]) -> "ObjectiveValues":
        return cls(
            normalized_latency=float(
                record.get("normalized_latency", record["latency_ms"])
            ),
            normalized_energy=float(
                record.get(
                    "normalized_energy",
                    record["system_energy_nominal_mj"],
                )
            ),
        )
