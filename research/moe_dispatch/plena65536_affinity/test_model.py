"""Affinity conservation, legal descriptors, independent hand schedules."""
from __future__ import annotations

from copy import deepcopy
import itertools

import pytest

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_dedicated.model import schedule as old_single_schedule
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from research.moe_dispatch.plena65536_affinity.model import (
    POLICIES, THRESHOLDS, hard_owner, ordered_experts, plan_from_costs, schedule)


def toy(batch=8, routed=(2, 2)):
    return {"id": "toy", "batch": batch, "experts": [
        {"id": -1, "Me": batch, "H": 2048, "F": 2816, "is_shared": True},
        *[{"id": i, "Me": m, "H": 2048, "F": 1408, "is_shared": False}
          for i, m in enumerate(routed)],
    ]}


PAIR = (Core(1, 256, 1, 128), Core(1, 256, 1, 128))
SINGLE = (Core(1, 512, 1, 128),)


def audit(cores, window, result):
    expected = {e["id"]: e for e in window["experts"]}
    assert sorted(j["expert_id"] for j in result["jobs"]) == sorted(expected)
    assert result["cycles"] == max(result["core_cycles"])
    total = 0
    for owner, core in enumerate(cores):
        clock = 0
        for job in [j for j in result["jobs"] if j["core_id"] == owner]:
            e = expected[job["expert_id"]]
            work = 3 * e["Me"] * e["H"] * e["F"]
            assert job["useful_macs"] == work
            assert job["issued_macs"] >= work
            assert job["start"] == clock
            assert job["end"] - job["start"] == job["cycles"]
            assert job["cycles"] * core.macs >= work
            clock = job["end"]
            total += work
        assert clock == result["core_cycles"][owner]
    assert total == result["useful_MACs"]
    assert result["unique_weight_bytes"] == sum(6 * e["H"] * e["F"] for e in expected.values())
    assert 0 < result["spatial_utilization"] <= 1
    assert 0 < result["wall_peak_utilization"] <= 1
    assert result["migration_count"] == result["steal_count"] == 0


def test_hard_threshold_boundaries_shared_hot_cold_and_empty_core():
    window = toy(8, (1, 2, 4, 8))
    for threshold in THRESHOLDS:
        result = schedule(PAIR, window, policy="hard_threshold", threshold=threshold)
        audit(PAIR, window, result)
        assert all(j["core_id"] == hard_owner(next(e for e in window["experts"]
                    if e["id"] == j["expert_id"]), threshold) for j in result["jobs"])
        assert result["affinity_deviation_count"] == 0
        assert threshold in result["boundary_stats"]["realizable_thresholds"]
    empty = schedule(PAIR, toy(8, (4, 8)), policy="hard_threshold", threshold=2)
    assert empty["core_cycles"][1] == 0
    assert empty["boundary_stats"]["imbalance_ratio"] is None
    assert empty["boundary_stats"]["idle_fractions"][1] == 1


def test_independent_eft_hand_case_and_exact_tie_affinity():
    experts = ordered_experts(toy())
    result = plan_from_costs(experts, [[8, 3, 3], [5, 5, 5]], "greedy_eft", 2)
    # Shared chooses C1 (5 < 8); the two other jobs finish at C0=3 and C0=6.
    assert result["owners"] == [1, 0, 0]
    assert result["core_cycles"] == [6, 5]
    assert result["cycles"] == 6
    equal = plan_from_costs(experts, [[3, 3, 3], [3, 3, 3]], "greedy_eft", 2)
    assert equal["owners"] == [0, 1, 1]  # Third decision ties at finish=6.
    assert equal["core_cycles"] == [3, 6]


def test_portfolio_can_choose_threshold_when_greedy_blocks_fast_core():
    experts = ordered_experts(toy())
    costs = [[10, 100, 100], [5, 5, 5]]
    greedy = plan_from_costs(experts, costs, "greedy_eft", 2)
    best = plan_from_costs(experts, costs, "workload_aware", 2)
    assert greedy["cycles"] == 15
    assert best["cycles"] == 10
    assert best["owners"] == [0, 1, 1]
    assert best["selected_policy"] == "hard_threshold"
    assert best["selected_threshold"] == 2
    assert set(best["candidate_cycles"]) == {
        "greedy_eft", "hard_threshold_1", "hard_threshold_2", "hard_threshold_4", "hard_threshold_8"}


def test_portfolio_tie_prefers_greedy_then_ascending_threshold():
    experts = ordered_experts(toy(8, (1,)))
    costs = [[5, 5], [5, 5]]
    best = plan_from_costs(experts, costs, "workload_aware", 2)
    assert best["cycles"] == 5
    assert best["selected_policy"] == "greedy_eft"
    assert best["selected_threshold"] is None
    adaptive = plan_from_costs(experts, costs, "adaptive_threshold", 2)
    assert adaptive["cycles"] == 5
    assert adaptive["selected_threshold"] == 1


def test_small_exhaustive_assignment_bound_and_portfolio_set():
    experts = ordered_experts(toy(4, (1, 3)))
    for a in itertools.product((1, 5), repeat=3):
        for b in itertools.product((2, 6), repeat=3):
            # Independent enumeration gives a lower bound, not a claim that
            # the proposed inexpensive portfolio solves global assignment.
            exact = min(max(sum(a[i] for i, o in enumerate(owners) if o == 0),
                            sum(b[i] for i, o in enumerate(owners) if o == 1))
                        for owners in itertools.product((0, 1), repeat=3))
            results = [plan_from_costs(experts, [a, b], "greedy_eft", 2)]
            results += [plan_from_costs(experts, [a, b], "hard_threshold", t) for t in THRESHOLDS]
            portfolio = plan_from_costs(experts, [a, b], "workload_aware", 2)
            assert exact <= portfolio["cycles"] == min(r["cycles"] for r in results)


def test_single_core_exact_cycles_and_work_match_preserved_model():
    for window in (toy(2), toy(128, (1, 2, 32, 64, 128))):
        for profile in PROFILES:
            old = old_single_schedule(SINGLE, window, profile)
            for policy in POLICIES:
                new = schedule(SINGLE, window, profile, policy=policy)
                audit(SINGLE, window, new)
                for field in ("cycles", "core_cycles", "useful_MACs", "issued_macs",
                              "unique_weight_bytes", "spatial_utilization", "wall_peak_utilization"):
                    assert new[field] == old[field]
                assert new["selected_policy"] == "single_serial"


def test_input_order_does_not_change_schedule_or_affinity_feedback():
    original = toy(8, (1, 2, 4, 8))
    changed = deepcopy(original)
    changed["experts"].reverse()
    for policy in POLICIES:
        assert schedule(PAIR, original, policy=policy) == schedule(PAIR, changed, policy=policy)
    result = schedule(PAIR, original, policy="greedy_eft")
    independently_counted = sum(j["core_id"] != (0 if j["is_shared"] or j["M"] > 2 else 1)
                                for j in result["jobs"])
    assert result["affinity_deviation_count"] == independently_counted


def test_true_b2_and_b128_routes_preserved_under_every_policy():
    inputs = load_workloads("heldout")["heldout"]
    cases = [next(w for w in inputs if w["batch"] == batch) for batch in (2, 128)]
    asymmetric = (Core(1, 64, 2, 128), Core(1, 512, 1, 96))
    for window in cases:
        for cores in (SINGLE, PAIR, asymmetric):
            for policy in POLICIES:
                result = schedule(cores, window, policy=policy)
                audit(cores, window, result)
                assert result == schedule(cores, window, policy=policy)
        hard = schedule(PAIR, window, policy="hard_threshold", threshold=2)
        if window["batch"] == 2:
            assert hard["boundary_stats"]["core_useful_MACs"][1] == 3 * hard["boundary_stats"]["core_useful_MACs"][0]
        else:
            assert hard["boundary_stats"]["core_useful_MACs"][0] > 0.9 * hard["useful_MACs"]


@pytest.mark.parametrize("mutate", [
    lambda w: w["experts"].append(deepcopy(w["experts"][1])),
    lambda w: w["experts"][1].update(id=64),
    lambda w: w["experts"][1].update(id=-2),
    lambda w: w["experts"][1].update(Me=0),
    lambda w: w["experts"][1].update(Me=9),
    lambda w: w["experts"][1].update(H=0),
    lambda w: w["experts"][1].update(Me=2.5),
    lambda w: w["experts"][1].pop("F"),
    lambda w: w["experts"][0].update(Me=1),
    lambda w: w["experts"].pop(0),
])
def test_bad_descriptors_cannot_be_silently_scheduled(mutate):
    window = toy()
    mutate(window)
    with pytest.raises(ValueError):
        schedule(PAIR, window)


def test_invalid_budget_policy_threshold_and_cost_matrix():
    with pytest.raises(ValueError):
        schedule((PAIR[0],), toy())
    with pytest.raises(ValueError):
        schedule(PAIR, toy(), policy="unimplemented")
    with pytest.raises(ValueError):
        schedule(PAIR, toy(), threshold=3)
    experts = ordered_experts(toy())
    with pytest.raises(ValueError):
        plan_from_costs(experts, [[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        plan_from_costs(experts, [[1, 0, 2], [1, 2, 2]])
