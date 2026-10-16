"""Finite native and replicated PLENA Matrix Unit geometry domains.

``b`` is BLEN and ``r`` is MLEN of one native flattened systolic unit.
One such unit has ``b*r`` multipliers, not ``b*b*r``: its ``r/b``
square subarrays stream a logical b-by-b output over b feed cycles.
Replication ``rm`` and ``rn`` widens the simultaneously processed output
rows and columns.  The physical expansion is (pm, pn, groups), where
``groups=r/b``.  Different native frames with the same expansion remain
distinct because their feed/drain timing can differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
from typing import Iterator

BUDGET = 65_536
BLENS = (2, 4, 8, 16, 32, 64)
MLENS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True, order=True)
class Core:
    """An immutable legal core in the declared replication-aware domain."""

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
        if self.pm not in BLENS:
            raise ValueError("Physical PM must be a declared BLEN power of two")
        if self.pn % 4:
            raise ValueError("Physical PN must be a positive multiple of four")
        if self.macs > BUDGET:
            raise ValueError("A core cannot exceed the total 65,536 multiplier budget")

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
        return self.rm == self.rn == 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.pm, self.pn, self.groups

    def __str__(self) -> str:
        return f"{self.pm}×{self.pn}×{self.groups} (BLEN={self.b}, MLEN={self.r})"

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "b": self.b,
            "r": self.r,
            "rm": self.rm,
            "rn": self.rn,
            "pm": self.pm,
            "pn": self.pn,
            "groups": self.groups,
            "macs": self.macs,
            "native": self.native,
        }


def _validate_budget(budget: int) -> None:
    if type(budget) is not int or not 1 <= budget <= BUDGET:
        raise ValueError("Budget must be an integer between 1 and 65,536")


@lru_cache(maxsize=None)
def enumerate_cores(budget: int = BUDGET) -> tuple[Core, ...]:
    """All legal cores at or below budget; no workload-derived PN cap.

    A resource bound alone limits RN.  Physical aliases with different
    BLEN/MLEN are intentionally retained.  Returned order is deterministic.
    """
    _validate_budget(budget)
    cores: list[Core] = []
    for b in BLENS:
        for r in MLENS:
            if r < b:
                continue
            for pm in BLENS:
                if pm < b:
                    continue
                rm = pm // b
                rn_max = budget // (b * r * rm)
                rn_step = 2 if b == 2 else 1
                for rn in range(rn_step, rn_max + 1, rn_step):
                    cores.append(Core(b, r, rm, rn))
    return tuple(sorted(cores, key=lambda c: (c.macs, c.b, c.r, c.rm, c.rn)))


@lru_cache(maxsize=None)
def cores_by_macs(budget: int = BUDGET) -> dict[int, tuple[Core, ...]]:
    """Resource index for lazy complementary-budget pair enumeration."""
    mutable: dict[int, list[Core]] = {}
    for core in enumerate_cores(budget):
        mutable.setdefault(core.macs, []).append(core)
    return {macs: tuple(group) for macs, group in mutable.items()}


def single_cores(budget: int = BUDGET) -> tuple[Core, ...]:
    return cores_by_macs(budget).get(budget, ())


def homogeneous_cores(budget: int = BUDGET) -> tuple[Core, ...]:
    _validate_budget(budget)
    return cores_by_macs(budget).get(budget // 2, ()) if budget % 2 == 0 else ()


def iter_dual_pairs(
    budget: int = BUDGET, *, include_identical: bool = True
) -> Iterator[tuple[Core, Core]]:
    """Yield each unordered exact-budget pair once, without a pair list."""
    index = cores_by_macs(budget)
    for macs in sorted(index):
        other_macs = budget - macs
        if macs > other_macs:
            break
        other = index.get(other_macs)
        if other is None:
            continue
        group = index[macs]
        if macs == other_macs:
            pair_iter = (
                combinations_with_replacement(group, 2)
                if include_identical
                else combinations(group, 2)
            )
            yield from pair_iter
        else:
            for first in group:
                for second in other:
                    yield first, second


def iter_heterogeneous_pairs(budget: int = BUDGET) -> Iterator[tuple[Core, Core]]:
    """Heterogeneity includes distinct frames and equal-budget different shapes."""
    yield from iter_dual_pairs(budget, include_identical=False)


def count_dual_pairs(budget: int = BUDGET, *, include_identical: bool = True) -> int:
    """Count exact-budget unordered pairs without constructing them."""
    index = cores_by_macs(budget)
    total = 0
    for macs, group in index.items():
        other_macs = budget - macs
        if macs > other_macs or other_macs not in index:
            continue
        if macs < other_macs:
            total += len(group) * len(index[other_macs])
        else:
            n = len(group)
            total += n * (n + 1 if include_identical else n - 1) // 2
    return total


def count_heterogeneous_pairs(budget: int = BUDGET) -> int:
    return count_dual_pairs(budget, include_identical=False)


def enumerate_native_designs(budget: int = BUDGET) -> dict[str, tuple]:
    """Small strict-native subset under the same declared PN alignment.

    ``homogeneous`` contains two-core tuples, whereas ``single`` contains
    individual cores.  Native unequal-budget splits cannot add two powers
    of two to this power-of-two budget.
    """
    singles = tuple(c for c in single_cores(budget) if c.native)
    hom = tuple((c, c) for c in homogeneous_cores(budget) if c.native)
    natives = tuple(c for c in enumerate_cores(budget) if c.native)
    hetero = tuple(
        (first, second)
        for first, second in combinations(natives, 2)
        if first.macs + second.macs == budget
    )
    return {"single": singles, "homogeneous": hom, "heterogeneous": hetero}


def geometry_counts(budget: int = BUDGET) -> dict[str, int]:
    cores = enumerate_cores(budget)
    native = enumerate_native_designs(budget)
    return {
        "budget": budget,
        "individual_cores": len(cores),
        "distinct_physical_shapes": len({c.shape for c in cores}),
        "distinct_core_budgets": len(cores_by_macs(budget)),
        "single": len(single_cores(budget)),
        "homogeneous": len(homogeneous_cores(budget)),
        "heterogeneous": count_heterogeneous_pairs(budget),
        "dual_including_homogeneous": count_dual_pairs(budget),
        "native_single": len(native["single"]),
        "native_homogeneous": len(native["homogeneous"]),
        "native_heterogeneous": len(native["heterogeneous"]),
    }
