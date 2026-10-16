"""Independent checks of fractional pruning and exact split optimisation.

The exhaustive checks deliberately use small physical budgets and direct
integer cuts.  They do not merely repeat the binary-search implementation.
"""

import random

import numpy as np

from .geometry import Core, cores_by_macs, enumerate_cores, iter_heterogeneous_pairs
from .search import core_cost_arrays, evaluate, fractional_pair, lower_bounds, solve_domain
from .timing import BATCHES, F, H, PROFILES, _split_monotone, partition, phase, shared


def test_binary_split_matches_all_integer_cuts_with_plateaus():
    rng = random.Random(65536)
    for total in range(1, 40):
        for _ in range(8):
            left = [0]
            right = [0]
            for _ in range(total):
                left.append(left[-1] + rng.randrange(4))
                right.append(right[-1] + rng.randrange(4))
            result, cut = _split_monotone(total, left.__getitem__, right.__getitem__)
            expected = min(max(left[x], right[total - x]) for x in range(total + 1))
            assert result == expected
            assert result == max(left[cut], right[total - cut])


def test_partition_matches_direct_row_and_column_cuts():
    candidates = (
        (Core(2, 32, 1, 2), Core(4, 16)),
        (Core(4, 128, 1, 3), Core(2, 32, 4, 6)),
        (Core(32, 1024), Core(16, 2048)),
    )
    for profile in PROFILES:
        for cores in candidates:
            a, b = cores
            for batch in (2, 8, 96):
                rows = min(max(shared(a, cut, profile), shared(b, batch - cut, profile))
                           for cut in range(batch + 1))
                gu = min(max(phase(a, batch, cut, H, 2, profile),
                             phase(b, batch, F - cut, H, 2, profile))
                         for cut in range(F + 1))
                down = min(max(phase(a, batch, cut, F, 1, profile),
                               phase(b, batch, H - cut, F, 1, profile))
                           for cut in range(H + 1))
                assert partition(cores, batch, profile)['cycles'] == min(rows, gu + down)


def test_fractional_lower_bound_safe_per_batch_on_complementary_budgets():
    cores = enumerate_cores()
    index = {core: i for i, core in enumerate(cores)}
    by_budget = cores_by_macs()
    rng = random.Random(20261011)
    possible_budgets = sorted(n for n in by_budget if 65536 - n in by_budget)
    candidates = []
    for _ in range(180):
        n = rng.choice(possible_budgets)
        candidates.append((rng.choice(by_budget[n]), rng.choice(by_budget[65536 - n])))
    ids = np.array([[index[a], index[b]] for a, b in candidates], dtype=np.int32)
    i, j = ids[:, 0], ids[:, 1]
    for profile in PROFILES:
        arrays = core_cost_arrays(cores, profile)
        gu, down, ag, ad, ar, d = arrays
        cols = fractional_pair(F, ag[i], d[i], ag[j], d[j], gu[i], gu[j])
        cols += fractional_pair(H, ad[i], d[i], ad[j], d[j], down[i], down[j])
        rows = fractional_pair(np.array(BATCHES)[None, :], ar[i], 2*d[i],
                               ar[j], 2*d[j], gu[i]+down[i], gu[j]+down[j])
        bound = np.minimum(cols, rows)
        exact = np.array([[partition(cs, batch, profile)['cycles'] for batch in BATCHES]
                          for cs in candidates])
        assert np.all(bound <= exact + 1e-7)
        assert np.all(lower_bounds(ids, arrays) <=
                      np.exp(np.mean(np.log(exact), axis=1)) + 1e-7)


def test_vector_cost_equals_scalar_single_core_cost_including_k_padding():
    cores = enumerate_cores(256)
    for profile in PROFILES:
        gu, down, *_ = core_cost_arrays(cores, profile)
        for i, core in enumerate(cores):
            assert list(gu[i] + down[i]) == [shared(core, batch, profile) for batch in BATCHES]


def test_pruned_search_and_minimax_match_exhaustive_domain():
    # Every candidate is evaluated independently here.  This catches invalid
    # pruning even when the profile-specific winner happens to be unchanged.
    cores = enumerate_cores(128)
    index = {core: i for i, core in enumerate(cores)}
    ids = np.array([[index[a], index[b]] for a, b in iter_heterogeneous_pairs(128)],
                   dtype=np.int32)
    bounds = [lower_bounds(ids, core_cost_arrays(cores, profile)) for profile in PROFILES]
    selected, _ = solve_domain('audit_small_domain', cores, ids, bounds)
    exhaustive = np.array([[evaluate((cores[i], cores[j]), profile)[0] for i, j in ids]
                           for profile in PROFILES])
    optimum = exhaustive.min(axis=1)
    for p, expected in zip(PROFILES, optimum):
        assert abs(selected[p.name]['GM_cycles'] - expected) < 1e-7
    robust_expected = np.max(exhaustive / optimum[:, None], axis=0).min()
    assert abs(selected['timing_robust']['max_regret'] + 1 - robust_expected) < 1e-10
