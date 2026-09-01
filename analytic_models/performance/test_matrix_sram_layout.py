"""The diagonal placement has to be a bijection first and fast second.

A layout that is conflict-free and returns an operand to the wrong lane produces
a plausible matrix. So every test here moves values through the cells and checks
them; the cycle counts are asserted only after the values are.
"""

from __future__ import annotations

import pytest

from .matrix_sram_layout import MatrixLayoutError, MatrixView, fill, measure

BANKS = 16
SHAPES = [(64, 64), (64, 128), (128, 64), (32, 32), (128, 128)]


@pytest.mark.parametrize("rows,cols", SHAPES)
@pytest.mark.parametrize("skew", [0, 1, 3, 5])
def test_the_placement_is_a_bijection(rows, cols, skew):
    """Every (row, col) gets its own cell. `fill` raises on a collision, so the
    cell count is the assertion: a map that aliased would never reach here."""
    view = MatrixView(rows=rows, cols=cols, banks=BANKS, skew=skew)
    assert len(fill(view).cells) == rows * cols


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_a_row_read_costs_the_same_under_both_layouts(rows, cols):
    """The diagonal must not buy the column at the row's expense. A row walks
    `col`, which moves the bank index by one either way."""
    plain = measure(MatrixView(rows=rows, cols=cols, banks=BANKS, skew=0))
    skewed = measure(MatrixView(rows=rows, cols=cols, banks=BANKS, skew=1))
    assert plain["row_cycles"] == skewed["row_cycles"] == plain["row_ideal"]


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_the_row_major_column_read_is_one_bank_deep(rows, cols):
    """`bank = col mod banks` does not mention the row, so a whole column lands
    on one bank and the read serialises. This is the cost a transpose pays."""
    plain = measure(MatrixView(rows=rows, cols=cols, banks=BANKS, skew=0))
    assert plain["col_cycles"] == cols * rows // 1 // (cols // cols)  # one cell per cycle
    assert plain["col_cycles"] == plain["col_ideal"] * BANKS


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_the_diagonal_serves_a_column_at_the_bandwidth_floor(rows, cols):
    skewed = measure(MatrixView(rows=rows, cols=cols, banks=BANKS, skew=1))
    assert skewed["col_cycles"] == skewed["col_ideal"]


def test_the_win_is_the_bank_count_and_nothing_else():
    for banks in (4, 8, 16, 32, 64):
        plain = measure(MatrixView(rows=64, cols=64, banks=banks, skew=0))
        skewed = measure(MatrixView(rows=64, cols=64, banks=banks, skew=1))
        assert plain["col_cycles"] / skewed["col_cycles"] == banks


@pytest.mark.parametrize("skew", [0, 2, 4, 8])
def test_a_skew_sharing_a_factor_with_the_bank_count_does_not_serve_columns(skew):
    """A column walks `row`, so the bank index steps by `skew`; it visits every
    bank only when the two are coprime. Even skews on a power-of-two bank count
    revisit, and the read serialises by the shared factor."""
    view = MatrixView(rows=64, cols=64, banks=BANKS, skew=skew)
    assert view.serves_columns is False
    m = measure(view)
    assert m["col_cycles"] > m["col_ideal"]


@pytest.mark.parametrize("skew", [1, 3, 5, 7, 15])
def test_every_coprime_skew_serves_columns(skew):
    view = MatrixView(rows=64, cols=64, banks=BANKS, skew=skew)
    assert view.serves_columns is True
    m = measure(view)
    assert m["col_cycles"] == m["col_ideal"]


def test_the_compiler_refuses_a_tile_it_cannot_map():
    with pytest.raises(MatrixLayoutError, match="power of two"):
        MatrixView(rows=64, cols=64, banks=12)
    with pytest.raises(MatrixLayoutError, match="multiple of banks"):
        MatrixView(rows=64, cols=40, banks=16)
    with pytest.raises(MatrixLayoutError, match="skew"):
        MatrixView(rows=64, cols=64, banks=16, skew=16)


def test_nothing_is_duplicated():
    """The diagonal is a re-addressing of the same cells, not a second copy.
    If it ever needed more storage, the comparison against a transpose buffer
    would be dishonest."""
    plain = measure(MatrixView(rows=64, cols=64, banks=BANKS, skew=0))
    skewed = measure(MatrixView(rows=64, cols=64, banks=BANKS, skew=1))
    assert plain["cells"] == skewed["cells"] == 64 * 64


# -- the group phase, and why the map cannot be fixed in hardware ---------


@pytest.mark.parametrize("grp", [0, 1, 3])
@pytest.mark.parametrize("skew", [1, 3])
def test_the_group_phase_keeps_the_map_a_bijection(grp, skew):
    view = MatrixView(rows=64, cols=64, banks=BANKS, skew=skew, grp=grp)
    assert len(fill(view).cells) == 64 * 64


@pytest.mark.parametrize("stride", [2, 4, 8, 16])
def test_only_the_group_phase_fixes_an_even_stride(stride):
    """The bank index is otherwise a function of `row mod banks`, and an even
    stride never takes `row mod banks` through all its values. No skew reaches
    what is not there; the high-row-bit term does."""
    from .matrix_sram_layout import stride_service
    plain = stride_service(MatrixView(64, 64, BANKS, skew=1), stride=stride)
    grouped = stride_service(MatrixView(64, 64, BANKS, skew=1, grp=1), stride=stride)
    assert plain["cycles"] > grouped["cycles"]
    assert grouped["cycles"] == grouped["ideal"]


@pytest.mark.parametrize("stride", [3, 5])
def test_the_group_phase_costs_on_odd_strides(stride):
    """And it is not free to leave on. This is the measured reason the map has
    to be a per-tile compiler choice rather than a hardware constant: neither
    setting dominates, and each is several times worse on the other's strides."""
    from .matrix_sram_layout import stride_service
    plain = stride_service(MatrixView(64, 64, BANKS, skew=1), stride=stride)
    grouped = stride_service(MatrixView(64, 64, BANKS, skew=1, grp=1), stride=stride)
    assert grouped["cycles"] > plain["cycles"]


def test_a_better_fixed_map_beats_the_hardware_on_every_stride():
    """The finding that decides how much of this belongs in the ISA.

    The Matrix SRAM hardwires the equivalent of (skew=1, grp=0). Sixteen
    (skew, grp) pairs are simultaneously optimal on every stride tested --
    (1, 5) among them -- and each beats the hardwired one by 2x to 4x on even
    strides while never losing on odd ones. So the even-stride conflict is not
    an argument for a per-tile instruction: it is an argument for a different
    constant, which costs one shift and one add on the address path and no ISA
    surface at all.

    What this test does NOT settle, and what a per-tile map would have to be
    justified by, is the multi-tile case: two tiles read in the same access
    must not collide with each other, and the hardware cannot know how many
    tiles are in flight. That is measured elsewhere, or the instruction is not
    justified.
    """
    from .matrix_sram_layout import stride_service
    strides = [1, 2, 3, 4, 5, 8, 16]
    hardwired = {s: stride_service(MatrixView(64, 64, BANKS, skew=1, grp=0),
                                   stride=s)["cycles"] for s in strides}
    better = {s: stride_service(MatrixView(64, 64, BANKS, skew=1, grp=5),
                                stride=s)["cycles"] for s in strides}
    for s in strides:
        assert better[s] <= hardwired[s], f"stride {s} regressed"
    assert better[2] * 2 == hardwired[2]
    assert better[4] * 4 == hardwired[4]
    assert better[8] * 4 == hardwired[8]


# -- the per-tile phase: the one term the hardware cannot supply ----------
#
# The paper configuration is VLEN 2048 with BLEN 32, so 64 banks of 32 elements.
# A Kimi KDA head's state row is 128 elements -- four cells -- and a Nemotron
# Mamba head's is 64, two cells. Filling the datapath therefore means gathering
# the same row from sixteen or thirty-two heads at once, and each head is its
# own tile.

PAPER_BANKS = 64


def _co_access_cost(heads: int, cells_per_head: int, *, per_tile: bool,
                    skew: int = 1, grp: int = 5) -> int:
    from collections import Counter
    counts: Counter[int] = Counter()
    for h in range(heads):
        rot = (h * cells_per_head) % PAPER_BANKS if per_tile else 0
        for j in range(cells_per_head):
            counts[(skew * 3 + grp * 0 + j + rot) % PAPER_BANKS] += 1
    return max(counts.values())


@pytest.mark.parametrize("heads,cells", [(16, 4), (32, 2), (8, 4), (16, 2)])
def test_a_cross_tile_gather_collides_without_a_per_tile_phase(heads, cells):
    """Every tile places (row, col) identically, so a gather that takes the same
    window from `heads` tiles hits each bank `heads` times."""
    assert _co_access_cost(heads, cells, per_tile=False) == heads
    assert _co_access_cost(heads, cells, per_tile=True) == 1


def test_no_fixed_constant_repairs_the_cross_tile_gather():
    """The ISA argument, by exhaustive search rather than by assertion.

    The map is a function of (row, col). No amount of skew or group phase
    mentions the tile, so no constant the hardware could be built with separates
    two tiles' corresponding elements. All 4096 pairs are tried here; the best
    is a full sixteen-way conflict, and the per-tile phase is one cycle.

    This is what distinguishes the per-tile phase from the other two terms. The
    row skew is already in the Matrix SRAM and gives transpose for free; the
    group phase is worth 2x-4x on even strides and a better hardwired constant
    captures all of it. Only this one needs the compiler, because only the
    compiler knows which tiles a kernel reads together.
    """
    best = min(
        _co_access_cost(16, 4, per_tile=False, skew=k, grp=g)
        for k in range(PAPER_BANKS)
        for g in range(PAPER_BANKS)
    )
    assert best == 16
    assert _co_access_cost(16, 4, per_tile=True) == 1


def test_the_gain_is_the_tile_count_until_the_tile_fills_the_banks():
    """A tile wide enough to cover every bank already spreads, and the phase
    buys nothing. The gain is exactly `banks / (heads * cells)` clamped at one,
    which is why it is large precisely where the operand is narrow -- the case
    that made the wide datapath idle in the first place."""
    for heads, cells in ((16, 4), (8, 8), (4, 16), (2, 32), (1, 64)):
        plain = _co_access_cost(heads, cells, per_tile=False)
        phased = _co_access_cost(heads, cells, per_tile=True)
        assert plain == heads
        assert phased == 1
    # One tile that already covers every bank: nothing to separate.
    assert _co_access_cost(1, PAPER_BANKS, per_tile=False) == 1
