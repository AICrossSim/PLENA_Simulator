"""Chip organisations: task streams (contexts) and arrays, and the array shape of every stream."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Organisation:
    name: str
    kind: str  # single, hom or het
    streams: int
    arrays: int

    @property
    def shared_array(self) -> bool:
        """Several task streams share one array."""
        return self.streams == 4 or self.kind == "single"


ORGS = {
    o.name: o
    for o in (
        Organisation("single2", "single", 2, 1),
        Organisation("hom11", "hom", 2, 2),
        Organisation("het11", "het", 2, 2),
        Organisation("single4", "single", 4, 1),
        Organisation("hom22", "hom", 4, 2),
        Organisation("het22", "het", 4, 2),
    )
}
FAMILIES = {2: ("single2", "hom11", "het11"), 4: ("single4", "hom22", "het22")}


def expand(org: Organisation, base) -> list[list[int]]:
    """Array shape of every task stream."""
    if org.kind == "single":
        return [list(base[0])] * org.streams
    if org.streams == 4:
        return [list(base[0]), list(base[0]), list(base[1]), list(base[1])]
    return [list(x) for x in base]
