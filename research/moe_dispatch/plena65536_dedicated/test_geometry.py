"""Resource accounting and exact-domain checks for the dedicated search."""

from itertools import product

import pytest

from research.moe_dispatch.plena65536_dedicated.geometry import (
    BUDGET, Core, count_dedicated_pairs, enumerate_cores, geometry_counts,
    homogeneous_cores, iter_dedicated_pairs, single_cores,
)
from research.moe_dispatch.plena65536_shared.geometry import (
    enumerate_cores as preserved_enumerate,
)


def test_paper_geometry_and_scalar_extension_count_real_multipliers():
    paper = Core(32, 2048)
    scalar = Core(1, 128, 1, 256)
    assert paper.shape == (32, 32, 64) and paper.macs == BUDGET
    assert scalar.shape == (1, 256, 128) and scalar.macs == 32_768
    assert scalar.blen1_extension and not scalar.native
    # Merely gating half the rows of a preserved PM=2 array cannot change
    # its physical multiplier count; the proposed PM=1 is a separate core.
    two_row = Core(2, 256, 1, 128)
    assert two_row.shape == (2, 256, 128) and two_row.macs == BUDGET


def test_minimum_blen_two_preserves_existing_domain_exactly():
    actual = enumerate_cores(2048, include_blen1=False)
    old = preserved_enumerate(2048)
    assert [c.to_dict() | {"blen1_extension": False} for c in actual] == [
        c.to_dict() | {"blen1_extension": False} for c in old
    ]


@pytest.mark.parametrize("include_blen1", [True, False])
@pytest.mark.parametrize("shared_not_smaller", [True, False])
@pytest.mark.parametrize("exclude_identical", [True, False])
def test_small_domain_complete_ordered_pairs_match_brute_force(
    include_blen1, shared_not_smaller, exclude_identical,
):
    budget = 64
    cores = enumerate_cores(budget, include_blen1=include_blen1)
    expected = {
        (a, b) for a, b in product(cores, repeat=2)
        if a.macs + b.macs == budget and b.pm in (1, 2)
        and (not shared_not_smaller or a.macs >= b.macs)
        and (not exclude_identical or a != b)
    }
    kwargs = dict(include_blen1=include_blen1,
                  shared_not_smaller=shared_not_smaller,
                  exclude_identical=exclude_identical)
    actual = list(iter_dedicated_pairs(budget, **kwargs))
    assert len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected
    assert count_dedicated_pairs(budget, **kwargs) == len(expected)


def test_no_workload_derived_column_cap_and_exact_comparison_budgets():
    assert max(c.pn for c in enumerate_cores()) == 65_536
    assert all(c.macs == BUDGET for c in single_cores())
    assert all(c.macs == BUDGET // 2 for c in homogeneous_cores())
    assert all(c.pm in (1, 2) and c.macs <= BUDGET // 2
               for _, c in iter_dedicated_pairs(128))


def test_larger_shared_alias_includes_requested_equal_budget_boundary():
    assert list(iter_dedicated_pairs(64, require_shared_larger=True)) == list(
        iter_dedicated_pairs(64, shared_not_smaller=True))
    assert list(iter_dedicated_pairs(64, require_shared_larger=False)) == list(
        iter_dedicated_pairs(64, shared_not_smaller=False))
    assert count_dedicated_pairs(64, require_shared_larger=False) == count_dedicated_pairs(
        64, shared_not_smaller=False)
    assert any(a.macs == b.macs for a, b in iter_dedicated_pairs(64))


def test_arbitrary_rounded_two_to_one_split_is_not_a_legal_frame():
    # PN alignment restricts even the scalar domain to multiples of four.
    cores = enumerate_cores()
    budgets = {c.macs for c in cores}
    assert 43_690 not in budgets and 21_846 not in budgets
    assert BUDGET % 3 != 0
    large, small = Core(2, 256, 1, 96), Core(1, 128, 1, 128)
    assert (large.macs, small.macs) == (49_152, 16_384)
    assert large.macs + small.macs == BUDGET


@pytest.mark.parametrize("args", [
    (1, 3, 1, 4), (1, 2, 3, 4), (1, 2, 1, 3),
    (1, 128, 1, 0), (2, 1, 1, 2), (64, 2048, 1, 1),
    (1.0, 128, 1, 4),
])
def test_invalid_geometry_rejected(args):
    with pytest.raises(ValueError):
        Core(*args)
