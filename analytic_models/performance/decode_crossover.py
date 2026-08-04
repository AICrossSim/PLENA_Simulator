"""Batch and context crossover records for cached decode."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

CROSSOVER_SCHEMA = "plena-decode-crossover"
STEP_COMPOSITION = "max_compute_memory"
_CONTENT_ADDRESSED_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{64}$"
)


def _finite_non_negative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _require_content_id(
    value: str,
    name: str,
    prefix: str,
) -> None:
    if (
        not isinstance(value, str)
        or not _CONTENT_ADDRESSED_ID.fullmatch(value)
        or not value.startswith(prefix)
    ):
        raise ValueError(f"{name} must be a content-addressed identity")


@dataclass(frozen=True)
class DecodeCrossoverPoint:
    """One capacity-aware roofline point for a cached q_len=1 step."""

    context: int
    batch: int
    peak_compute_seconds: float
    ideal_compute_seconds: float
    realized_compute_seconds: float
    memory_seconds: float
    physical_bytes_per_batch_step: float
    capacity_required_bytes: int
    capacity_available_bytes: int
    timing_mode: str
    timing_calibrated: bool
    timing_evidence_id: str | None = None
    packed_q1_timing_contract_id: str | None = None
    bandwidth_calibration_id: str | None = None
    step_composition: str = STEP_COMPOSITION
    schema_version: str = CROSSOVER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CROSSOVER_SCHEMA:
            raise ValueError("unsupported crossover schema")
        for name in ("context", "batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "peak_compute_seconds",
            "ideal_compute_seconds",
            "realized_compute_seconds",
            "memory_seconds",
            "physical_bytes_per_batch_step",
        ):
            object.__setattr__(
                self,
                name,
                _finite_non_negative(getattr(self, name), name),
            )
        if self.ideal_compute_seconds + 1e-18 < self.peak_compute_seconds:
            raise ValueError("ideal issue cannot beat the peak-compute ceiling")
        if self.realized_compute_seconds + 1e-18 < self.ideal_compute_seconds:
            raise ValueError("realized compute cannot be faster than the ideal oracle")
        for name in ("capacity_required_bytes", "capacity_available_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.timing_mode:
            raise ValueError("timing_mode must be non-empty")
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError("unsupported decode-step composition")
        if self.timing_calibrated != bool(
            self.timing_evidence_id
            and self.packed_q1_timing_contract_id
        ):
            raise ValueError("timing calibration identity is inconsistent")
        if self.timing_evidence_id is not None:
            _require_content_id(
                self.timing_evidence_id,
                "timing_evidence_id",
                "timing-",
            )
        if self.packed_q1_timing_contract_id is not None:
            _require_content_id(
                self.packed_q1_timing_contract_id,
                "packed_q1_timing_contract_id",
                "packed-q1-timing-",
            )
        if self.bandwidth_calibration_id is not None:
            _require_content_id(
                self.bandwidth_calibration_id,
                "bandwidth_calibration_id",
                "bandwidth-operating-point-",
            )

    @property
    def capacity_feasible(self) -> bool:
        return self.capacity_required_bytes <= self.capacity_available_bytes

    @property
    def classical_roofline_bottleneck(self) -> str:
        if self.memory_seconds >= self.peak_compute_seconds:
            return "memory"
        return "compute"

    @property
    def architecture_issue_bottleneck(self) -> str:
        if self.memory_seconds >= self.ideal_compute_seconds:
            return "memory"
        return "compute"

    @property
    def algorithmic_bottleneck(self) -> str:
        """Compatibility alias for the architecture-issue view."""

        return self.architecture_issue_bottleneck

    @property
    def realized_bottleneck(self) -> str:
        if (
            not self.timing_calibrated
            or self.bandwidth_calibration_id is None
        ):
            return "unavailable"
        if self.memory_seconds >= self.realized_compute_seconds:
            return "memory"
        if (
            self.architecture_issue_bottleneck == "memory"
            and self.realized_compute_seconds > self.ideal_compute_seconds
        ):
            return "serialization"
        return "compute"

    @property
    def rankable(self) -> bool:
        return (
            self.capacity_feasible
            and self.timing_calibrated
            and self.timing_evidence_id is not None
            and self.packed_q1_timing_contract_id is not None
            and self.bandwidth_calibration_id is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "context": self.context,
            "batch": self.batch,
            "peak_compute_seconds": self.peak_compute_seconds,
            "ideal_compute_seconds": self.ideal_compute_seconds,
            "realized_compute_seconds": self.realized_compute_seconds,
            "memory_seconds": self.memory_seconds,
            "physical_bytes_per_batch_step": self.physical_bytes_per_batch_step,
            "capacity_required_bytes": self.capacity_required_bytes,
            "capacity_available_bytes": self.capacity_available_bytes,
            "capacity_feasible": self.capacity_feasible,
            "classical_roofline_bottleneck": (
                self.classical_roofline_bottleneck
            ),
            "architecture_issue_bottleneck": (
                self.architecture_issue_bottleneck
            ),
            "algorithmic_bottleneck": self.algorithmic_bottleneck,
            "realized_bottleneck": self.realized_bottleneck,
            "timing_mode": self.timing_mode,
            "timing_calibrated": self.timing_calibrated,
            "timing_evidence_id": self.timing_evidence_id,
            "packed_q1_timing_contract_id": (
                self.packed_q1_timing_contract_id
            ),
            "step_composition": self.step_composition,
            "bandwidth_calibration_id": self.bandwidth_calibration_id,
            "rankable": self.rankable,
        }


@dataclass(frozen=True)
class CrossoverTransition:
    """A measured label change between adjacent feasible batch points."""

    context: int
    lower_batch: int
    upper_batch: int
    source_bottleneck: str
    target_bottleneck: str

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context,
            "lower_batch": self.lower_batch,
            "upper_batch": self.upper_batch,
            "source_bottleneck": self.source_bottleneck,
            "target_bottleneck": self.target_bottleneck,
        }


@dataclass(frozen=True)
class DecodeCrossoverStudy:
    """Canonical batch-by-context roofline grid with explicit transitions."""

    points: tuple[DecodeCrossoverPoint, ...]
    schema_version: str = CROSSOVER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CROSSOVER_SCHEMA:
            raise ValueError("unsupported crossover schema")
        ordered = tuple(sorted(self.points, key=lambda point: (point.context, point.batch)))
        if not ordered:
            raise ValueError("crossover study requires at least one point")
        keys = [(point.context, point.batch) for point in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("crossover context and batch pairs must be unique")
        fixed_contract = {
            (
                point.capacity_available_bytes,
                point.timing_mode,
                point.timing_calibrated,
                point.timing_evidence_id,
                point.packed_q1_timing_contract_id,
                point.bandwidth_calibration_id,
                point.step_composition,
            )
            for point in ordered
        }
        if len(fixed_contract) != 1:
            raise ValueError("crossover points must share one hardware contract")
        for axis, groups in (
            (
                "batch",
                {
                    context: [
                        point for point in ordered if point.context == context
                    ]
                    for context in {point.context for point in ordered}
                },
            ),
            (
                "context",
                {
                    batch: [
                        point for point in ordered if point.batch == batch
                    ]
                    for batch in {point.batch for point in ordered}
                },
            ),
        ):
            for group in groups.values():
                group.sort(key=lambda point: getattr(point, axis))
                byte_values = [
                    point.physical_bytes_per_batch_step for point in group
                ]
                capacity_values = [
                    point.capacity_required_bytes for point in group
                ]
                if any(
                    current + 1e-9 < previous
                    for previous, current in zip(byte_values, byte_values[1:])
                ):
                    raise ValueError(
                        f"physical traffic decreases with {axis}"
                    )
                if any(
                    current < previous
                    for previous, current in zip(
                        capacity_values,
                        capacity_values[1:],
                    )
                ):
                    raise ValueError(
                        f"required capacity decreases with {axis}"
                    )
        object.__setattr__(self, "points", ordered)

    @classmethod
    def from_points(
        cls,
        points: Iterable[DecodeCrossoverPoint],
    ) -> "DecodeCrossoverStudy":
        return cls(tuple(points))

    @property
    def transitions(self) -> tuple[CrossoverTransition, ...]:
        transitions: list[CrossoverTransition] = []
        contexts = sorted({point.context for point in self.points})
        for context in contexts:
            feasible = [
                point
                for point in self.points
                if point.context == context and point.capacity_feasible
            ]
            for lower, upper in zip(feasible, feasible[1:]):
                if lower.realized_bottleneck != upper.realized_bottleneck:
                    transitions.append(
                        CrossoverTransition(
                            context=context,
                            lower_batch=lower.batch,
                            upper_batch=upper.batch,
                            source_bottleneck=lower.realized_bottleneck,
                            target_bottleneck=upper.realized_bottleneck,
                        )
                    )
        return tuple(transitions)

    @property
    def rankable(self) -> bool:
        for context in {point.context for point in self.points}:
            feasible = [
                point
                for point in self.points
                if point.context == context and point.capacity_feasible
            ]
            if not feasible or not all(point.rankable for point in feasible):
                return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rankable": self.rankable,
            "points": [point.to_dict() for point in self.points],
            "transitions": [
                transition.to_dict() for transition in self.transitions
            ],
        }


__all__ = [
    "CROSSOVER_SCHEMA",
    "CrossoverTransition",
    "DecodeCrossoverPoint",
    "DecodeCrossoverStudy",
    "STEP_COMPOSITION",
]
