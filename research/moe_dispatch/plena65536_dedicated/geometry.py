"""Exact-budget replicated PLENA geometries for hard-affinity compute DSE.

The primary domain adds the explicitly proposed BLEN=1 scalar subarray to
the preserved BLEN=2..64 domain.  A 1x1 subarray has one actual multiplier;
it is not a gated 2x2 subarray whose unused lanes are counted as free.  Its
feed interval remains one cycle, with the same charged pipeline and
cross-group reduction model.  No cross-expert packing is implied.

This is a finite *declared* architectural domain, not proof that all these
replicated BF16/FP32 frames exist in the original PLENA RTL.  All comparisons
use the same domain.  ``include_blen1=False`` retains the previous frame
domain as a sensitivity check.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

BUDGET = 65_536
BLENS = (1, 2, 4, 8, 16, 32, 64)
PMS = BLENS
MLENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
ROUTED_PMS = (1, 2)


@dataclass(frozen=True, order=True)
class Core:
    """One immutable physical core and its native reduction feed frame."""

    b: int
    r: int
    rm: int = 1
    rn: int = 1

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in (self.b, self.r, self.rm, self.rn)):
            raise ValueError("Core dimensions and replication factors must be integers")
        if self.b not in BLENS:
            raise ValueError(f"BLEN must be one of {BLENS}")
        if self.r not in MLENS or self.r < self.b or self.r % self.b:
            raise ValueError("MLEN must be a declared power of two, at least BLEN")
        if self.rm < 1 or self.rn < 1:
            raise ValueError("Replication factors must be positive")
        if self.pm not in PMS:
            raise ValueError("Physical PM must be a declared power of two from 1 to 64")
        if self.pn % 4:
            raise ValueError("Physical PN must be a positive multiple of four")
        if self.macs > BUDGET:
            raise ValueError("A core cannot exceed the 65,536 multiplier budget")

    @property
    def pm(self) -> int:
        return self.b * self.rm

    @property
    def pn(self) -> int:
        return self.b * self.rn

    @property
    def groups(self) -> int:
        return self.r // self.b

    @property
    def macs(self) -> int:
        return self.pm * self.pn * self.groups

    @property
    def native(self) -> bool:
        """True for unreplicated frames in the preserved BLEN>=2 domain."""
        return self.b >= 2 and self.rm == self.rn == 1

    @property
    def blen1_extension(self) -> bool:
        return self.b == 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.pm, self.pn, self.groups

    def __str__(self) -> str:
        return f"{self.pm}×{self.pn}×{self.groups} (BLEN={self.b}, MLEN={self.r})"

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "b": self.b, "r": self.r, "rm": self.rm, "rn": self.rn,
            "pm": self.pm, "pn": self.pn, "groups": self.groups,
            "macs": self.macs, "native": self.native,
            "blen1_extension": self.blen1_extension,
        }


def _validate_budget(budget: int) -> None:
    if type(budget) is not int or not 1 <= budget <= BUDGET:
        raise ValueError("Budget must be an integer between 1 and 65,536")


@lru_cache(maxsize=None)
def enumerate_cores(
    budget: int = BUDGET, *, include_blen1: bool = True
) -> tuple[Core, ...]:
    """Every legal declared frame under the resource bound, retaining aliases.

    PN has no workload-derived upper bound: budget alone limits column
    replication.  Frames with equal physical dimensions remain distinct
    because BLEN/MLEN determine feed and pipeline timing.
    """
    _validate_budget(budget)
    allowed_b = BLENS if include_blen1 else BLENS[1:]
    cores: list[Core] = []
    for b in allowed_b:
        for r in MLENS:
            if r < b:
                continue
            for pm in PMS:
                if pm < b:
                    continue
                rm = pm // b
                rn_max = budget // (b * r * rm)
                rn_step = max(1, 4 // b)
                for rn in range(rn_step, rn_max + 1, rn_step):
                    cores.append(Core(b, r, rm, rn))
    return tuple(sorted(cores, key=lambda c: (c.macs, c.b, c.r, c.rm, c.rn)))


@lru_cache(maxsize=None)
def cores_by_macs(
    budget: int = BUDGET, *, include_blen1: bool = True
) -> dict[int, tuple[Core, ...]]:
    mutable: dict[int, list[Core]] = {}
    for core in enumerate_cores(budget, include_blen1=include_blen1):
        mutable.setdefault(core.macs, []).append(core)
    return {macs: tuple(group) for macs, group in mutable.items()}


def single_cores(
    budget: int = BUDGET, *, include_blen1: bool = True
) -> tuple[Core, ...]:
    return cores_by_macs(budget, include_blen1=include_blen1).get(budget, ())


def homogeneous_cores(
    budget: int = BUDGET, *, include_blen1: bool = True
) -> tuple[Core, ...]:
    _validate_budget(budget)
    return (
        cores_by_macs(budget, include_blen1=include_blen1).get(budget // 2, ())
        if budget % 2 == 0 else ()
    )


def iter_dedicated_pairs(
    budget: int = BUDGET, *, include_blen1: bool = True,
    shared_not_smaller: bool = True, exclude_identical: bool = False,
    require_shared_larger: bool | None = None,
) -> Iterator[tuple[Core, Core]]:
    """Ordered (Shared-only Core 0, Routed-only Core 1) exact-budget designs.

    Core 1 always has physical PM in {1,2}.  ``shared_not_smaller`` enforces
    the requested large Shared core and small Routed core *by multiplier
    allocation*, allowing the explicitly requested 1:1 boundary.
    ``require_shared_larger`` is a compatibility alias for this constraint;
    equality remains allowed, as the task explicitly includes a 1:1 split.
    """
    if require_shared_larger is not None:
        shared_not_smaller = require_shared_larger
    index = cores_by_macs(budget, include_blen1=include_blen1)
    for routed_macs in sorted(index):
        shared_macs = budget - routed_macs
        if shared_not_smaller and shared_macs < routed_macs:
            continue
        shared_cores = index.get(shared_macs, ())
        for routed_core in index[routed_macs]:
            if routed_core.pm not in ROUTED_PMS:
                continue
            for shared_core in shared_cores:
                if exclude_identical and shared_core == routed_core:
                    continue
                yield shared_core, routed_core


def count_dedicated_pairs(
    budget: int = BUDGET, *, include_blen1: bool = True,
    shared_not_smaller: bool = True, exclude_identical: bool = False,
    require_shared_larger: bool | None = None,
) -> int:
    """Count the exact domain without materializing its Cartesian pairs."""
    if require_shared_larger is not None:
        shared_not_smaller = require_shared_larger
    index = cores_by_macs(budget, include_blen1=include_blen1)
    total = 0
    for routed_macs, group in index.items():
        shared_macs = budget - routed_macs
        if shared_not_smaller and shared_macs < routed_macs:
            continue
        other = index.get(shared_macs, ())
        routed = tuple(c for c in group if c.pm in ROUTED_PMS)
        total += len(routed) * len(other)
        if exclude_identical and shared_macs == routed_macs:
            total -= len(routed)
    return total


def geometry_counts(
    budget: int = BUDGET, *, include_blen1: bool = True
) -> dict[str, int | bool]:
    cores = enumerate_cores(budget, include_blen1=include_blen1)
    return {
        "budget": budget, "include_blen1": include_blen1,
        "individual_cores": len(cores),
        "distinct_physical_shapes": len({c.shape for c in cores}),
        "distinct_core_budgets": len(cores_by_macs(budget, include_blen1=include_blen1)),
        "single": len(single_cores(budget, include_blen1=include_blen1)),
        "homogeneous": len(homogeneous_cores(budget, include_blen1=include_blen1)),
        "dedicated_shared_not_smaller": count_dedicated_pairs(
            budget, include_blen1=include_blen1),
        "dedicated_any_budget_order": count_dedicated_pairs(
            budget, include_blen1=include_blen1, shared_not_smaller=False),
    }
