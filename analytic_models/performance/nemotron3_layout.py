"""Group-major/skewed layout for the proposed L-compute projection buffer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ProjectionField(StrEnum):
    X = "x"
    GATE = "gate"
    B = "b"
    C = "c"
    DT = "dt"


@dataclass(frozen=True, order=True)
class PhysicalAddress:
    row: int
    bank: int


@dataclass(frozen=True)
class ProjectionGeometry:
    num_heads: int = 64
    head_dim: int = 64
    state_dim: int = 128
    groups: int = 8
    banks: int = 16
    head_lanes: int = 8
    head_dim_lanes: int = 4
    state_dim_lanes: int = 8

    def __post_init__(self) -> None:
        for name in (
            "num_heads",
            "head_dim",
            "state_dim",
            "groups",
            "banks",
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_heads % self.groups:
            raise ValueError("num_heads must be divisible by groups")
        if self.head_dim % self.banks:
            raise ValueError("head_dim must be bank aligned for this layout")
        if self.state_dim % self.banks:
            raise ValueError("state_dim must be bank aligned for this layout")

    @property
    def heads_per_group(self) -> int:
        return self.num_heads // self.groups


class GroupMajorSkewedLayout:
    """Bijective logical-to-physical mapping with bank-aligned group records.

    One physical row contains one word in every bank. Fields are aligned to a
    row boundary; the eight-value ``dt`` tail is padded to avoid sharing a row
    with the next group. The 64 padding values are less than 0.7% of Nemotron's
    10,304-value projection packet.
    """

    def __init__(self, geometry: ProjectionGeometry = ProjectionGeometry()) -> None:
        self.geometry = geometry
        heads = geometry.heads_per_group
        field_sizes = {
            ProjectionField.X: heads * geometry.head_dim,
            ProjectionField.GATE: heads * geometry.head_dim,
            ProjectionField.B: geometry.state_dim,
            ProjectionField.C: geometry.state_dim,
            ProjectionField.DT: heads,
        }
        offset = 0
        offsets: dict[ProjectionField, int] = {}
        for field in ProjectionField:
            offset = self._align(offset)
            offsets[field] = offset
            offset += field_sizes[field]
        self.field_sizes = field_sizes
        self.field_offsets = offsets
        self.group_span = self._align(offset)

    def _align(self, value: int) -> int:
        banks = self.geometry.banks
        return math.ceil(value / banks) * banks

    @property
    def logical_values(self) -> int:
        return self.geometry.groups * sum(self.field_sizes.values())

    @property
    def physical_values(self) -> int:
        return self.geometry.groups * self.group_span

    @property
    def padding_values(self) -> int:
        return self.physical_values - self.logical_values

    def address(
        self,
        field: ProjectionField | str,
        group: int,
        local_head: int,
        lane: int,
    ) -> PhysicalAddress:
        field = ProjectionField(field)
        geometry = self.geometry
        if not 0 <= group < geometry.groups:
            raise ValueError("group is out of range")
        if field in {ProjectionField.X, ProjectionField.GATE}:
            if not 0 <= local_head < geometry.heads_per_group:
                raise ValueError("local_head is out of range")
            if not 0 <= lane < geometry.head_dim:
                raise ValueError("head-dimension lane is out of range")
            local = local_head * geometry.head_dim + lane
            skew = local_head * geometry.head_dim_lanes
        elif field in {ProjectionField.B, ProjectionField.C}:
            if not 0 <= lane < geometry.state_dim:
                raise ValueError("state-dimension lane is out of range")
            local = lane
            skew = geometry.state_dim_lanes if field == ProjectionField.C else 0
        else:
            if not 0 <= local_head < geometry.heads_per_group:
                raise ValueError("local_head is out of range")
            local = local_head
            skew = 0

        base = group * self.group_span + self.field_offsets[field]
        row = (base + local) // geometry.banks
        bank = (local % geometry.banks + skew) % geometry.banks
        return PhysicalAddress(row, bank)

    def logical_addresses(self) -> dict[tuple[ProjectionField, int, int, int], PhysicalAddress]:
        result = {}
        geometry = self.geometry
        for group in range(geometry.groups):
            for field in (ProjectionField.X, ProjectionField.GATE):
                for head in range(geometry.heads_per_group):
                    for lane in range(geometry.head_dim):
                        result[(field, group, head, lane)] = self.address(field, group, head, lane)
            for field in (ProjectionField.B, ProjectionField.C):
                for lane in range(geometry.state_dim):
                    result[(field, group, 0, lane)] = self.address(field, group, 0, lane)
            for head in range(geometry.heads_per_group):
                result[(ProjectionField.DT, group, head, 0)] = self.address(ProjectionField.DT, group, head, 0)
        return result

    def state_input_packets(self, group: int) -> list[tuple[PhysicalAddress, ...]]:
        """Packets consumed by one group of the fused state-update pipeline."""
        geometry = self.geometry
        packets: list[tuple[PhysicalAddress, ...]] = []
        for head_start in range(0, geometry.heads_per_group, geometry.head_lanes):
            heads = range(head_start, min(head_start + geometry.head_lanes, geometry.heads_per_group))
            packets.append(tuple(self.address(ProjectionField.DT, group, head, 0) for head in heads))
            for dim_start in range(0, geometry.head_dim, geometry.head_dim_lanes):
                dimensions = range(dim_start, min(dim_start + geometry.head_dim_lanes, geometry.head_dim))
                packets.append(
                    tuple(
                        self.address(ProjectionField.X, group, head, dimension)
                        for head in heads
                        for dimension in dimensions
                    )
                )
        for state_start in range(0, geometry.state_dim, geometry.state_dim_lanes):
            states = range(state_start, min(state_start + geometry.state_dim_lanes, geometry.state_dim))
            packets.append(
                tuple(self.address(ProjectionField.B, group, 0, state) for state in states)
                + tuple(self.address(ProjectionField.C, group, 0, state) for state in states)
            )
        return packets

    def gate_packets(self, group: int) -> list[tuple[PhysicalAddress, ...]]:
        geometry = self.geometry
        packets = []
        for head_start in range(0, geometry.heads_per_group, geometry.head_lanes):
            heads = range(head_start, min(head_start + geometry.head_lanes, geometry.heads_per_group))
            for dim_start in range(0, geometry.head_dim, geometry.head_dim_lanes):
                dimensions = range(dim_start, min(dim_start + geometry.head_dim_lanes, geometry.head_dim))
                packets.append(
                    tuple(
                        self.address(ProjectionField.GATE, group, head, dimension)
                        for head in heads
                        for dimension in dimensions
                    )
                )
        return packets


def packet_service_cycles(packet: tuple[PhysicalAddress, ...], *, banks: int, ports_per_bank: int = 1) -> int:
    if banks <= 0 or ports_per_bank <= 0:
        raise ValueError("banks and ports_per_bank must be positive")
    counts = [0] * banks
    for address in packet:
        counts[address.bank] += 1
    return max((math.ceil(count / ports_per_bank) for count in counts), default=0)
