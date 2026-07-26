"""Formal four-objective schema for Qwen3 prefill exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OBJECTIVE_DIRECTIONS = ("minimize", "minimize", "minimize", "maximize")


@dataclass(frozen=True)
class ObjectiveValues:
    latency_ms: float
    total_silicon_area_mm2: float
    system_energy_mj: float
    accuracy_score: float

    def as_optuna_values(self) -> tuple[float, float, float, float]:
        return (
            float(self.latency_ms),
            float(self.total_silicon_area_mm2),
            float(self.system_energy_mj),
            float(self.accuracy_score),
        )

    @classmethod
    def from_trial_record(cls, record: dict[str, Any]) -> "ObjectiveValues":
        return cls(
            latency_ms=float(record["latency_ms"]),
            total_silicon_area_mm2=float(record["area_mm2"]),
            system_energy_mj=float(record["system_energy_nominal_mj"]),
            accuracy_score=float(record["accuracy_score"]),
        )
