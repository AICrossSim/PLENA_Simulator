"""Independent finite-domain and exact hard-affinity objective references."""
from __future__ import annotations

import numpy as np
import pytest

from .geometry import enumerate_cores, iter_dedicated_pairs
from .independent_audit import (independent_core_domain, independent_pair_domain,
                                independent_pair_scores, independent_score,
                                integer_window_cycles)
from .test_model import toy_window


@pytest.mark.parametrize("include_blen1", [False, True])
def test_full_physical_domain_matches_independent_pm_pn_g_enumeration(include_blen1):
    actual = enumerate_cores(include_blen1=include_blen1)
    got = tuple((c.macs, c.b, c.r, c.rm, c.rn) for c in actual)
    expected = independent_core_domain(include_blen1=include_blen1)
    assert got == expected


@pytest.mark.parametrize("include_blen1", [False, True])
@pytest.mark.parametrize("shared_not_smaller", [False, True])
def test_every_ordered_budget_pair_matches_independent_complementary_domain(
    include_blen1, shared_not_smaller,
):
    # A reduced budget is enough to compare the complete sequence of IDs;
    # the full individual-core domain is independently exhausted above.
    budget = 512
    cores = enumerate_cores(budget, include_blen1=include_blen1)
    index = {core: i for i, core in enumerate(cores)}
    actual = [(index[a], index[b]) for a, b in iter_dedicated_pairs(
        budget, include_blen1=include_blen1, shared_not_smaller=shared_not_smaller)]
    expected = list(independent_pair_domain(
        independent_core_domain(budget, include_blen1=include_blen1),
        budget=budget, shared_not_smaller=shared_not_smaller))
    assert actual == expected


def test_scalar_hard_affinity_scores_equal_direct_loads():
    from .geometry import Core
    cores = [Core(1, 128, 1, 256), Core(2, 128, 1, 128),
             Core(4, 512, 1, 16), Core(1, 256, 1, 128)]
    pairs = np.asarray([(0, 0), (1, 1), (2, 0), (3, 1)], dtype=np.int32)
    windows = [toy_window(2), toy_window(2), toy_window(4), toy_window(8)]
    scores = independent_pair_scores(cores, pairs, windows)
    for pi, (a, b) in enumerate(pairs):
        loads = [integer_window_cycles((cores[a], cores[b]), window)[1] for window in windows]
        assert scores[pi] == independent_score([max(x) for x in loads], windows)


def test_vector_search_matches_exhaustive_scalar_pricing_and_exact_products():
    from math import prod
    from research.moe_dispatch.plena65536_shared.timing import PROFILES
    from .geometry import Core
    from .independent_audit import independent_role_load_tables, independent_log_weights
    from .search import evaluate_family, integer_objective_exponents
    cores = [Core(1, 128, 1, 256), Core(2, 128, 1, 128),
             Core(4, 512, 1, 16), Core(1, 256, 1, 128)]
    descriptors = tuple((c.macs, c.b, c.r, c.rm, c.rn) for c in cores)
    pairs = np.asarray([(a, b) for a in range(len(cores)) for b in range(len(cores))], dtype=np.int32)
    windows = [toy_window(2), toy_window(2), toy_window(4), toy_window(8)]
    weights = independent_log_weights(windows)
    exponents, denominator = integer_objective_exponents(windows)
    assert denominator == 6 and exponents.tolist() == [1, 1, 2, 2]
    for profile in PROFILES:
        shared, routed = independent_role_load_tables(descriptors, windows,
                                                      tree_stage=profile.tree_stage,
                                                      serialized=profile.serialized)
        chosen, scores = evaluate_family(cores, pairs, shared, routed, weights,
                                          exponents=exponents, chunk=3)
        scalar_cycles = [[integer_window_cycles((cores[a], cores[b]), w,
                                                tree_stage=profile.tree_stage,
                                                serialized=profile.serialized)[0]
                          for w in windows] for a, b in pairs]
        expected_scores = [independent_score(c, windows) for c in scalar_cycles]
        np.testing.assert_allclose(scores, expected_scores, rtol=5e-15)
        products = [prod(int(c)**int(e) for c, e in zip(values, exponents))
                    for values in scalar_cycles]
        assert int(chosen["objective_exact_integer_product"]) == min(products)
        assert chosen["every_geometry_exactly_evaluated"] is True
        assert chosen["geometry_count"] == len(pairs)


def test_exact_integer_product_resolves_a_float_log_tie():
    from .geometry import Core
    from .search import evaluate_family
    cores = [Core(1, 128, 1, 256), Core(2, 128, 1, 128)]
    # Float logs round these distinct positive int64 makespans together.
    # Their difference is still exact in integer arithmetic and product.
    large = 2**60
    shared = np.asarray([[large + 1], [large]], dtype=np.int64)
    routed = np.ones_like(shared)
    pairs = np.asarray([(0, 0), (1, 1)], dtype=np.int32)
    chosen, scores = evaluate_family(cores, pairs, shared, routed, np.asarray([1.0]),
                                      exponents=np.asarray([1], dtype=np.int64))
    assert scores[0] == scores[1]
    assert chosen["indices"] == [1, 1]
    assert chosen["objective_exact_integer_product"] == str(large)
