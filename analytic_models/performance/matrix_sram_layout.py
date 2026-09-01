"""A banked Matrix SRAM that serves a row and a column at the same cost.

The Matrix SRAM stores one operand per cell and the array serves one cell per
bank per cycle. Under the natural row-major placement a *row* is spread over
every bank and reads in one pass, while a *column* is one bank deep and reads in
`rows` passes -- so a transpose costs a full re-read of the tile, and a kernel
that wants A and A^T pays for the transpose in cycles or in a second copy.

A diagonal placement removes that asymmetry outright:

    bank = (skew*row + col) mod banks
    addr = row * (cols // banks) + col // banks

At `skew = 0` this is the row-major layout. At `skew = 1` the same storage
serves both directions at the bank-bandwidth floor, because a row walks `col`
and a column walks `row`, and both terms move the bank index by one per step.
Nothing is duplicated and nothing is moved: it is the same cells, addressed
differently.

Every number here comes from values written into numbered cells and read back
out of them, and every access is checked against what was written. A placement
that is conflict-free and returns a value to the wrong lane fails here; a cycle
count alone could not tell the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd


class MatrixLayoutError(ValueError):
    """A layout that would lose a value, or one the hardware cannot express."""


@dataclass
class MatrixSram:
    """`banks` banks of `depth` addresses. One operand per (bank, addr) cell."""

    banks: int
    depth: int
    cells: dict[tuple[int, int], float] = field(default_factory=dict)

    def write(self, bank: int, addr: int, value: float) -> None:
        if not 0 <= addr < self.depth:
            raise IndexError(f"addr {addr} outside a {self.depth}-deep bank")
        if (bank, addr) in self.cells:
            raise MatrixLayoutError(f"two operands placed in cell ({bank}, {addr})")
        self.cells[(bank, addr)] = value

    def read_packet(self, coords: list[tuple[int, int]]) -> tuple[list[float], int]:
        """Serve one access. Cost is the busiest bank, as on a 1R-per-bank array."""
        counts: dict[int, int] = {}
        values = []
        for bank, addr in coords:
            if (bank, addr) not in self.cells:
                raise KeyError(f"read of unwritten cell ({bank}, {addr})")
            values.append(self.cells[(bank, addr)])
            counts[bank] = counts.get(bank, 0) + 1
        return values, max(counts.values(), default=0)


@dataclass(frozen=True)
class MatrixView:
    """The placement of one Matrix-resident tile, as four integers.

    `skew` is the only compiler-chosen term. `banks` is a machine constant and
    is never encoded; `rows`/`cols` come from the tile the compiler already
    knows the shape of.
    """

    rows: int
    cols: int
    banks: int
    skew: int = 0
    #: Phase per group of `banks` rows. Without it the bank index is a function
    #: of `row mod banks` alone, so a stride-`s` walk with `s` even visits only
    #: `banks/gcd(s, banks)` distinct banks however the skew is chosen -- the
    #: values simply are not there to spread. This term reaches the high row
    #: bits and is the only one that fixes even strides.
    grp: int = 0
    #: Phase for this tile as a whole. Nothing else in the map distinguishes one
    #: tile from another, so without it every tile places its (row, col) on the
    #: same bank -- and a packet that gathers the same window from several tiles
    #: at once collides with itself once per tile. No choice of `skew` or `grp`
    #: repairs that, because neither term mentions the tile.
    rot: int = 0
    #: Permute the bank index by XOR instead of rotating it by addition.
    xor: bool = False

    def __post_init__(self) -> None:
        if self.banks < 1 or self.banks & (self.banks - 1):
            raise MatrixLayoutError(f"banks must be a power of two, got {self.banks}")
        if self.cols % self.banks:
            raise MatrixLayoutError(
                f"cols {self.cols} must be a multiple of banks {self.banks}; a "
                f"partial final group would alias the next row's cells"
            )
        for name, value in (("skew", self.skew), ("grp", self.grp), ("rot", self.rot)):
            if not 0 <= value < self.banks:
                raise MatrixLayoutError(f"{name} {value} outside 0..{self.banks - 1}")

    @property
    def per_row(self) -> int:
        """Addresses one row occupies."""
        return self.cols // self.banks

    def place(self, row: int, col: int) -> tuple[int, int]:
        group = col // self.banks
        lane = col % self.banks
        phase = (self.skew * row + self.grp * (row // self.banks) + self.rot) % self.banks
        if self.xor:
            # A power-of-two bank count makes the additive skew useless on even
            # strides: `skew*stride mod banks` stays even whatever the skew, so
            # half the banks are never reached. XOR is a permutation of the bank
            # index rather than a rotation of it, and it is not stride-periodic
            # in the same way. Harper (IEEE TC 41(2), 1992) moved to exactly this
            # for exactly this reason, and it is cheaper than a rotation: one
            # XOR gate array instead of a modular add.
            bank = lane ^ phase
        else:
            bank = (phase + lane) % self.banks
        return bank, row * self.per_row + group

    @property
    def serves_columns(self) -> bool:
        """True when a column walk visits every bank.

        A column fixes `col` and walks `row`, so the bank index moves by `skew`
        per step; it is a permutation exactly when `skew` is coprime with the
        bank count. `skew = 0` -- the row-major layout -- is the degenerate case
        where a whole column sits on one bank.
        """
        return gcd(self.skew, self.banks) == 1

    # -- the two access shapes -------------------------------------------

    def row_coords(self, row: int) -> list[tuple[int, int]]:
        return [self.place(row, c) for c in range(self.cols)]

    def col_coords(self, col: int) -> list[tuple[int, int]]:
        return [self.place(r, col) for r in range(self.rows)]

    def ideal_cycles(self, values: int) -> int:
        return -(-values // self.banks)


def fill(view: MatrixView, value=lambda r, c: r * 1000.0 + c) -> MatrixSram:
    """Write the whole tile through the view. Raises if the map is not a bijection."""
    sram = MatrixSram(banks=view.banks, depth=view.rows * view.per_row)
    for r in range(view.rows):
        for c in range(view.cols):
            bank, addr = view.place(r, c)
            sram.write(bank, addr, value(r, c))
    return sram


def measure(view: MatrixView) -> dict[str, int]:
    """Read every row and every column back, checking values and counting cycles."""
    sram = fill(view)
    row_cycles = col_cycles = 0
    for r in range(view.rows):
        got, cost = sram.read_packet(view.row_coords(r))
        if got != [r * 1000.0 + c for c in range(view.cols)]:
            raise AssertionError(f"row {r} came back wrong or out of order")
        row_cycles += cost
    for c in range(view.cols):
        got, cost = sram.read_packet(view.col_coords(c))
        if got != [r * 1000.0 + c for r in range(view.rows)]:
            raise AssertionError(f"column {c} came back wrong or out of order")
        col_cycles += cost
    return {
        "row_cycles": row_cycles,
        "col_cycles": col_cycles,
        "row_ideal": view.rows * view.ideal_cycles(view.cols),
        "col_ideal": view.cols * view.ideal_cycles(view.rows),
        "cells": len(sram.cells),
    }


def strided_coords(view: MatrixView, *, col: int, stride: int) -> list[tuple[int, int]]:
    """Walk one column taking every `stride`-th row.

    This is the access a head-interleaved or block-strided consumer makes, and it
    is the one that separates a programmable skew from the hardwired one: the
    bank index advances by `skew * stride` per step, so conflict-freedom is a
    condition on the product, not on the skew alone.
    """
    return [view.place(r, col) for r in range(0, view.rows, stride)]


def stride_service(view: MatrixView, *, stride: int) -> dict[str, int]:
    """Cost of one strided column walk, values checked on the way out."""
    sram = fill(view)
    coords = strided_coords(view, col=0, stride=stride)
    got, cost = sram.read_packet(coords)
    expected = [r * 1000.0 for r in range(0, view.rows, stride)]
    if got != expected:
        raise AssertionError(f"stride {stride} came back wrong or out of order")
    return {"cycles": cost, "ideal": view.ideal_cycles(len(coords)), "values": len(coords)}


def best_skew(rows: int, cols: int, banks: int, stride: int) -> tuple[int, int]:
    """The cheapest skew for this stride, and what it costs.

    Returns (skew, cycles). A hardwired skew of 1 is what the current Matrix
    SRAM implements; anything this function finds below the skew-1 cost is what
    a compiler-chosen skew buys.
    """
    best = None
    for skew in range(banks):
        try:
            view = MatrixView(rows=rows, cols=cols, banks=banks, skew=skew)
        except MatrixLayoutError:
            continue
        cost = stride_service(view, stride=stride)["cycles"]
        if best is None or cost < best[1]:
            best = (skew, cost)
    return best


def co_access_service(views: list[MatrixView], *, row: int, lanes: int) -> dict[str, int]:
    """Gather the same `lanes`-wide window of one row from several tiles at once.

    This is the access that fills a wide vector operation from a recurrent state
    held per head: one row of `lanes` elements is far narrower than the datapath,
    so the operand is assembled from as many heads as it takes. Each tile is its
    own object with its own base address, and the map has no base term -- so
    whether the tiles collide with each other is decided entirely by `rot`.
    """
    srams = [fill(v) for v in views]
    counts: dict[int, int] = {}
    values, expected = [], []
    for tile, (view, sram) in enumerate(zip(views, srams)):
        for c in range(lanes):
            bank, addr = view.place(row, c)
            if (bank, addr) not in sram.cells:
                raise KeyError(f"tile {tile} read of unwritten cell ({bank}, {addr})")
            values.append(sram.cells[(bank, addr)])
            expected.append(row * 1000.0 + c)
            counts[bank] = counts.get(bank, 0) + 1
    if values != expected:
        raise AssertionError("co-access lost or moved a value")
    total = len(views) * lanes
    return {
        "cycles": max(counts.values(), default=0),
        "ideal": -(-total // views[0].banks),
        "values": total,
        "banks_touched": len(counts),
    }
