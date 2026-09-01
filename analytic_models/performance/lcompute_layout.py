"""Physical bank/port model for PLENA affine producer-consumer co-layout.

This module is an architecture-candidate layer over the transactional
emulator's logical Vector SRAM.  It does not claim that the current RTL SRAM is
banked.  Functional values and timing use the same ``place`` function, and the
model never treats banking as extra bandwidth.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum


class LayoutKind(StrEnum):
    ROW_MAJOR = "row_major"
    TRANSPOSE = "transpose"
    CONSUMER_MAJOR = "consumer_major"
    AFFINE_SKEW = "affine_skew"


@dataclass(frozen=True, order=True)
class LogicalCoord:
    group: int
    field: int
    major: int
    minor: int


@dataclass(frozen=True, order=True)
class PhysicalCoord:
    bank: int
    bank_row: int
    sublane: int


@dataclass(frozen=True)
class BankGeometry:
    banks: int
    bank_width: int
    read_ports: int = 1
    write_ports: int = 1

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def row_elements(self) -> int:
        return self.banks * self.bank_width


@dataclass(frozen=True)
class LayoutConfig:
    kind: LayoutKind
    groups: int
    fields: int
    majors: int
    minors: int
    alpha: int = 0
    beta: int = 0
    gamma: int = 0
    major_packed: bool = False
    bank_row_base: int = 0
    bank_row_pitch: int = 0

    @classmethod
    def from_contract(cls, document: dict[str, object]) -> tuple[LayoutConfig, BankGeometry]:
        if document.get("contract") != "plena.affine_layout" or document.get("version") != 1:
            raise ValueError("unsupported affine-layout contract")
        raw_layout = dict(document["layout"])  # type: ignore[arg-type]
        raw_layout["kind"] = LayoutKind(raw_layout.get("kind", LayoutKind.ROW_MAJOR))
        raw_geometry = dict(document["geometry"])  # type: ignore[arg-type]
        layout = cls(**raw_layout)
        geometry = BankGeometry(**raw_geometry)
        layout.assert_bijective(geometry)
        return layout, geometry

    def minimum_pitch(self, geometry: BankGeometry) -> int:
        if self.major_packed:
            return 1
        inner = self.majors if self.kind == LayoutKind.TRANSPOSE else self.minors
        stripes = math.ceil(inner / geometry.bank_width)
        return math.ceil(stripes / geometry.banks)

    def pitch(self, geometry: BankGeometry) -> int:
        return self.bank_row_pitch or self.minimum_pitch(geometry)

    def validate(self, geometry: BankGeometry) -> None:
        geometry.validate()
        for name in ("groups", "fields", "majors", "minors"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bank_row_base < 0 or self.bank_row_pitch < 0:
            raise ValueError("bank-row placement must not be negative")
        if self.major_packed:
            if self.kind == LayoutKind.TRANSPOSE:
                raise ValueError("major-packed placement does not support TRANSPOSE")
            if math.gcd(self.alpha % geometry.banks, geometry.banks) != 1:
                raise ValueError("major-packed alpha must permute every physical bank")
        if self.pitch(geometry) < self.minimum_pitch(geometry):
            raise ValueError("bank_row_pitch aliases logical major rows")

    def iter_coords(self) -> Iterator[LogicalCoord]:
        for group in range(self.groups):
            for field in range(self.fields):
                for major in range(self.majors):
                    for minor in range(self.minors):
                        yield LogicalCoord(group, field, major, minor)

    def place(self, coord: LogicalCoord, geometry: BankGeometry) -> PhysicalCoord:
        self.validate(geometry)
        bounds = (
            (coord.group, self.groups),
            (coord.field, self.fields),
            (coord.major, self.majors),
            (coord.minor, self.minors),
        )
        if any(value < 0 or value >= extent for value, extent in bounds):
            raise ValueError(f"logical coordinate is out of range: {coord}")
        if self.kind == LayoutKind.TRANSPOSE:
            inner = coord.major
            outer = (coord.group * self.fields + coord.field) * self.minors + coord.minor
        else:
            inner = coord.minor
            outer = (coord.group * self.fields + coord.field) * self.majors + coord.major
        stripe, sublane = divmod(inner, geometry.bank_width)
        phase = (self.alpha * coord.major + self.beta * coord.field + self.gamma * coord.group) % geometry.banks
        bank = (stripe + phase) % geometry.banks
        if self.major_packed:
            minor_steps = math.ceil(self.minors / geometry.bank_width)
            major_blocks = math.ceil(self.majors / geometry.banks)
            field_group = coord.group * self.fields + coord.field
            packed_row = (
                (field_group * major_blocks + coord.major // geometry.banks)
                * minor_steps
                + stripe
            )
            row = self.bank_row_base + packed_row * self.pitch(geometry)
        else:
            row = self.bank_row_base + outer * self.pitch(geometry) + stripe // geometry.banks
        return PhysicalCoord(bank, row, sublane)

    def assert_bijective(self, geometry: BankGeometry) -> None:
        seen: dict[PhysicalCoord, LogicalCoord] = {}
        for logical in self.iter_coords():
            physical = self.place(logical, geometry)
            previous = seen.setdefault(physical, logical)
            if previous != logical:
                raise ValueError(f"layout aliases {previous} and {logical} at {physical}")


@dataclass(frozen=True)
class ServiceStats:
    values: int
    bank_words: int
    bandwidth_floor_cycles: int
    service_cycles: int
    busiest_bank_words: int

    @property
    def conflict_stall_cycles(self) -> int:
        return self.service_cycles - self.bandwidth_floor_cycles


def service_packet(
    layout: LayoutConfig,
    geometry: BankGeometry,
    packet: Iterable[LogicalCoord],
    *,
    write: bool = False,
) -> ServiceStats:
    logical = list(packet)
    words = {(p.bank, p.bank_row) for p in (layout.place(c, geometry) for c in logical)}
    bank_counts = Counter(bank for bank, _row in words)
    ports = geometry.write_ports if write else geometry.read_ports
    busiest = max(bank_counts.values(), default=0)
    service = math.ceil(busiest / ports) if busiest else 0
    floor = math.ceil(len(words) / (geometry.banks * ports)) if words else 0
    return ServiceStats(len(logical), len(words), floor, service, busiest)


class BankedLayoutBuffer:
    """Sparse physical buffer used to prove mapping and lane restoration."""

    def __init__(self, layout: LayoutConfig, geometry: BankGeometry) -> None:
        layout.assert_bijective(geometry)
        self.layout = layout
        self.geometry = geometry
        self._cells: dict[PhysicalCoord, int | float] = {}

    def write(self, coord: LogicalCoord, value: int | float) -> None:
        physical = self.layout.place(coord, self.geometry)
        if physical in self._cells:
            raise ValueError(f"duplicate physical write at {physical}")
        self._cells[physical] = value

    def read(self, coord: LogicalCoord) -> int | float:
        physical = self.layout.place(coord, self.geometry)
        if physical not in self._cells:
            raise ValueError(f"read before write at {physical}")
        return self._cells[physical]

    def roundtrip_all(self) -> int:
        for index, coord in enumerate(self.layout.iter_coords()):
            self.write(coord, index)
        for index, coord in enumerate(self.layout.iter_coords()):
            if self.read(coord) != index:
                raise ValueError(f"lane restoration failed at {coord}")
        return len(self._cells)


@dataclass(frozen=True)
class FifoStats:
    producer_cycles: int
    completion_cycles: int
    stall_cycles: int
    high_watermark: int
    spilled_values: int


def simulate_fifo(
    *,
    total_values: int,
    producer_values_per_cycle: int,
    consumer_values_per_cycle: int,
    capacity_values: int,
    spill_values_per_cycle: int = 0,
) -> FifoStats:
    """Deterministic bounded FIFO with optional explicit spill sink."""

    for name, value in (
        ("total_values", total_values),
        ("producer_values_per_cycle", producer_values_per_cycle),
        ("consumer_values_per_cycle", consumer_values_per_cycle),
        ("capacity_values", capacity_values),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    produced = consumed = occupancy = high = stalls = spilled = cycle = 0
    queue: deque[int] = deque()
    while consumed + spilled < total_values:
        drain = min(occupancy, consumer_values_per_cycle)
        occupancy -= drain
        consumed += drain
        while queue and drain:
            take = min(queue[0], drain)
            queue[0] -= take
            drain -= take
            if queue[0] == 0:
                queue.popleft()

        incoming = min(producer_values_per_cycle, total_values - produced)
        available = capacity_values - occupancy
        accepted = min(incoming, available)
        if accepted:
            queue.append(accepted)
            occupancy += accepted
            produced += accepted
        blocked = incoming - accepted
        if blocked:
            if spill_values_per_cycle:
                amount = min(blocked, spill_values_per_cycle)
                produced += amount
                spilled += amount
                blocked -= amount
            if blocked:
                stalls += 1
        high = max(high, occupancy)
        cycle += 1
    producer_cycles = math.ceil(total_values / producer_values_per_cycle) + stalls
    return FifoStats(producer_cycles, cycle, stalls, high, spilled)


def multirow_word_packet(
    *,
    group: int,
    field: int,
    major_start: int,
    parallel_majors: int,
    minor_start: int,
    bank_width: int,
) -> list[LogicalCoord]:
    """One bank word from each of several rows, without increasing total width."""

    return [
        LogicalCoord(group, field, major, minor)
        for major in range(major_start, major_start + parallel_majors)
        for minor in range(minor_start, minor_start + bank_width)
    ]


__all__ = [
    "BankGeometry",
    "BankedLayoutBuffer",
    "FifoStats",
    "LayoutConfig",
    "LayoutKind",
    "LogicalCoord",
    "PhysicalCoord",
    "ServiceStats",
    "multirow_word_packet",
    "service_packet",
    "simulate_fifo",
]
