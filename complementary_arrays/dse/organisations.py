"""Chip organisations at equal resources: one array that interleaves two task streams, two identical arrays, and a
shorter and a taller array (the pair). Every organisation has two task streams."""

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
        """Both task streams share one array."""
        return self.kind == "single"


ORGS = {
    o.name: o
    for o in (
        Organisation("single2", "single", 2, 1),
        Organisation("hom11", "hom", 2, 2),
        Organisation("het11", "het", 2, 2),
    )
}
FAMILY = ("single2", "hom11", "het11")


def expand(org: Organisation, base) -> list[list[int]]:
    """Array shape of every task stream."""
    if org.kind == "single":
        return [list(base[0])] * org.streams
    return [list(x) for x in base]
