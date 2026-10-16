"""Vector search versus scalar schedules and independent assignment bounds."""
from __future__ import annotations

from itertools import product
import numpy as np
import pytest

from research.moe_dispatch.plena65536_affinity import search
from research.moe_dispatch.plena65536_affinity.model import schedule
from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_moe.model import cost_table, window_histograms
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_shared.timing import PROFILES


def toy(batch, counts):
    return {"id": f"toy{batch}", "batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": 2048, "F": 2816, "is_shared": True},
        *[{"id": i, "Me": m, "H": 2048, "F": 1408, "is_shared": False}
          for i, m in enumerate(counts)],
    ]}


def fixture_cores():
    return [Core(1, 256, 1, 128), Core(1, 128, 2, 128),
            Core(2, 128, 1, 128), Core(32, 1024),
            Core(1, 64, 2, 128), Core(1, 512, 1, 96)]


def test_vector_greedy_and_threshold_agree_with_scalar_every_profile():
    captured = load_workloads("heldout")["heldout"]
    windows = [next(w for w in captured if w["batch"] == b) for b in (2, 128)]
    windows += [toy(8, (1, 2, 2, 4, 7, 8)), toy(2, (1, 1, 2))]
    cores = fixture_cores()
    ids = np.array([[0, 0], [0, 1], [1, 0], [2, 3], [4, 5], [5, 4]], dtype=np.int32)
    types, hist, _ = window_histograms(windows)
    for profile in PROFILES:
        costs = cost_table(cores, types, profile)
        dense, stream = search.threshold_populations(cores, types, hist, costs)
        hard = search.hard_vectors(ids, dense, stream)
        for ti, threshold in enumerate(search.THRESHOLDS):
            greedy = search.greedy_vectors(ids, costs, windows, types, threshold)
            portfolio = np.minimum(greedy, hard.min(axis=0))
            for pi, pair in enumerate(ids):
                cs = tuple(cores[int(i)] for i in pair)
                for wi, window in enumerate(windows):
                    assert hard[ti, pi, wi] == schedule(cs, window, profile, "hard_threshold", threshold)["cycles"]
                    assert greedy[pi, wi] == schedule(cs, window, profile, "greedy_eft", threshold)["cycles"]
                    assert portfolio[pi, wi] == schedule(cs, window, profile, "workload_aware", threshold)["cycles"]


def test_ordered_pair_builder_exact_complements_both_roles_no_duplicates():
    cores = fixture_cores()
    expected = [(i, j) for i, a in enumerate(cores) for j, b in enumerate(cores)
                if a.macs + b.macs == 65536 and i != j]
    actual = list(map(tuple, search.ordered_pair_ids(cores)))
    assert len(actual) == len(set(actual))
    assert set(actual) == set(expected)
    assert (4, 5) in actual and (5, 4) in actual


def test_random_small_window_vector_portfolio_is_actual_schedule_minimum():
    rng = np.random.default_rng(20261013)
    windows = [toy(batch, rng.integers(1, batch + 1, size=7).tolist())
               for batch in (2, 4, 8, 16, 64, 128)]
    cores = fixture_cores()
    ids = np.array([[4, 5], [5, 4]], dtype=np.int32)
    types, hist, _ = window_histograms(windows)
    costs = cost_table(cores, types)
    dense, stream = search.threshold_populations(cores, types, hist, costs)
    hard = search.hard_vectors(ids, dense, stream)
    for threshold in search.THRESHOLDS:
        expected = np.minimum(search.greedy_vectors(ids, costs, windows, types, threshold), hard.min(axis=0))
        for pi, pair in enumerate(ids):
            cs = tuple(cores[int(i)] for i in pair)
            for wi, window in enumerate(windows):
                actual = schedule(cs, window, policy="workload_aware", threshold=threshold)
                assert actual["cycles"] == expected[pi, wi]
                assert actual["cycles"] == min(actual["candidate_cycles"].values())


def test_guarded_fractional_bound_is_below_independent_exact_assignments():
    bound = search.guarded_fractional_pair_bound
    ids = np.array([[0, 1]], dtype=np.int32)
    rng = np.random.default_rng(65536)
    for _ in range(40):
        costs = rng.integers(1, 500, size=(2, 4), dtype=np.int64)
        counts = rng.integers(0, 3, size=(1, 4), dtype=np.int64)
        if not counts.any():
            counts[0, 0] = 1
        exact = min(max(sum(int(costs[0, j]) * chosen[j] for j in range(4)),
                        sum(int(costs[1, j]) * (int(counts[0, j]) - chosen[j]) for j in range(4)))
                    for chosen in product(*(range(int(n) + 1) for n in counts[0])))
        lower = float(bound(costs, ids, counts)[0, 0])
        assert 0 <= lower <= exact


def test_guarded_fractional_accepts_legal_unbalanced_costs_and_rejects_overflow():
    bound = search.guarded_fractional_pair_bound
    ids = np.array([[0, 1]], dtype=np.int32)
    costs = np.array([[2_000_000_000, 1_999_999_999], [2, 3]], dtype=np.int64)
    counts = np.array([[32, 32]], dtype=np.int64)
    # Old max(cost)^2*jobs guard rejects this; actual cross-products fit.
    lower = bound(costs, ids, counts)
    assert 0 < lower[0, 0] <= 160
    unsafe = np.array([[2_000_000_000, 2_000_000_000],
                       [2_000_000_000, 2_000_000_000]], dtype=np.int64)
    with pytest.raises(OverflowError):
        bound(unsafe, ids, counts)


def test_exact_tie_selection_uses_integer_objective_then_storage():
    cores = [Core(1, 512, 1, 128), Core(32, 2048)]
    weights = np.array([0.5, 0.5])
    exponents = np.array([1, 1])
    candidates = [(np.array([2, 8]), (0, -1), 2, "single_serial", 0),
                  (np.array([4, 4]), (1, -1), 2, "single_serial", 1)]
    winner = search.pick(candidates, cores, weights, exponents)
    expected = min(candidates, key=lambda c: (16, search.tie_key(cores, c[1], c[2])))
    assert winner["indices"] == list(expected[1])
    assert winner["objective_exact_integer_product"] == "16"
