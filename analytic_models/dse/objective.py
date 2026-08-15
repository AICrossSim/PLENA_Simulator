"""Formal latency-energy objective schema for Qwen3 prefill exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


OBJECTIVE_DIRECTIONS = ("minimize", "minimize")
OBJECTIVE_NORMALIZATION = "identity"
OBJECTIVE_FIELDS = (
    "prefill_latency_ms",
    "prefill_system_energy_mj_ideal",
)


def area_budget_constraints(trial: Any) -> tuple[float]:
    """Return the durable area constraint at Optuna's callback boundary."""

    constraint = trial.user_attrs.get(
        "area_budget_constraint_mm2",
        trial.user_attrs.get("a100_area_constraint_mm2"),
    )
    if constraint is None or not math.isfinite(float(constraint)):
        return (math.inf,)
    return (float(constraint),)


@dataclass(frozen=True)
class ObjectiveValues:
    prefill_latency_ms: float
    prefill_system_energy_mj_ideal: float

    @property
    def normalized_latency(self) -> float:
        """Compatibility alias retained for historical consumers."""

        return self.prefill_latency_ms

    @property
    def normalized_energy(self) -> float:
        """Compatibility alias retained for historical consumers."""

        return self.prefill_system_energy_mj_ideal

    def as_optuna_values(self) -> tuple[float, float]:
        return (
            float(self.prefill_latency_ms),
            float(self.prefill_system_energy_mj_ideal),
        )

    @classmethod
    def from_trial_record(cls, record: dict[str, Any]) -> ObjectiveValues:
        return cls(
            prefill_latency_ms=float(
                record.get(
                    "prefill_latency_ms",
                    record.get("normalized_latency", record["latency_ms"]),
                )
            ),
            prefill_system_energy_mj_ideal=float(
                record.get(
                    "prefill_system_energy_mj_ideal",
                    record.get(
                        "normalized_energy",
                        record["system_energy_nominal_mj"],
                    ),
                )
            ),
        )
