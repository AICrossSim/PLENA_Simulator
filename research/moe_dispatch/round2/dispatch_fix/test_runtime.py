"""Focused checks for capacity-aware admission and frozen single-core behavior.

These are small physical simulations and scheduler decision examples. They do
not run the development or held-out campaigns or write frozen artifacts.
"""
from copy import deepcopy
import json

import pytest

from research.moe_dispatch.round2 import model as old
from research.moe_dispatch.round2.dispatch_fix import runtime
from research.moe_dispatch.round2.predictors import Predictor


def expert(rows=3, *, index=0, shared=False, h=512, f=128):
    return {"id": index, "Me": rows, "H": h, "F": f, "is_shared": shared}


def workload(experts, *, batch=16, topk=1, name="dispatch-fix-unit"):
    return {
        "id": name,
        "batch": batch,
        "top_k": topk,
        "hidden": experts[0]["H"],
        "experts": experts,
    }


def canonical(result):
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def dual(flow="OS", *, reverse=False):
    cores = (old.Core(4, 4, 512), old.Core(2, 4, 512))
    if reverse:
        cores = cores[::-1]
    return old.Design(cores, flows=(flow, flow))


@pytest.mark.parametrize("future_start,future_service,refetch_service", [
    (25.0, 100.0, 200.0),
    (25.0, 100.0, 125.0),
    (90.0, 10.0, 100.0),
])
def test_wait_beats_or_ties_immediate_refetch(
    future_start, future_service, refetch_service,
):
    immediate = runtime.Candidate(0, refetch_service, 0.0, 2.0)
    future = runtime.Candidate(1, future_service, future_start, 1.0, False)
    decision = runtime.choose_candidate([immediate, future], now=0.0)
    assert decision.chosen is None
    assert decision.deferred


def test_immediate_refetch_is_allowed_when_it_finishes_earlier():
    immediate = runtime.Candidate(0, 80.0, 0.0, 3.0)
    future = runtime.Candidate(1, 100.0, 25.0, 1.0, False)
    decision = runtime.choose_candidate([immediate, future], now=0.0)
    assert decision.chosen == immediate


def test_two_zero_refetch_cores_use_finish_time_and_deterministic_ties():
    slow = runtime.Candidate(0, 100.0, 0.0, 1.0)
    fast_after_wait = runtime.Candidate(1, 25.0, 30.0, 1.0)
    assert runtime.choose_candidate([slow, fast_after_wait]).chosen == fast_after_wait
    tie0 = runtime.Candidate(0, 50.0, 5.0, 1.0)
    tie1 = runtime.Candidate(1, 25.0, 30.0, 1.0)
    assert runtime.choose_candidate([tie1, tie0]).chosen == tie0


def test_all_refetch_candidates_still_use_predicted_finish_time():
    lower_traffic = runtime.Candidate(0, 200.0, 0.0, 1.5)
    faster = runtime.Candidate(1, 70.0, 20.0, 3.0)
    assert runtime.choose_candidate([lower_traffic, faster]).chosen == faster


@pytest.mark.parametrize("flow,rows", [("OS", 3), ("IS", 3), ("WS", 40)])
@pytest.mark.parametrize("reverse", [False, True])
def test_refetch_classification_uses_cost_and_private_capacity(flow, rows, reverse):
    design = dual(flow, reverse=reverse)
    if flow == "WS":
        accumulator = (95 * 1024, 1024)
        if reverse:
            accumulator = accumulator[::-1]
        design = old.Design(
            design.cores, flows=design.flows, acc_bytes=accumulator,
        )
    costs = [old.task_cost(expert(rows), design, core) for core in (0, 1)]
    no_refetch_core = int(reverse)
    refetch_core = 1 - no_refetch_core
    assert costs[no_refetch_core].hbm_bytes == costs[no_refetch_core].unique_hbm_bytes
    assert costs[refetch_core].hbm_bytes > costs[refetch_core].unique_hbm_bytes
    candidates = [
        runtime.Candidate(
            core,
            cost.isolated_cycles,
            100.0 if core == no_refetch_core else 0.0,
            runtime.refetch_factor(cost),
            core != no_refetch_core,
        )
        for core, cost in enumerate(costs)
    ]
    assert runtime.choose_candidate(candidates).chosen is None


@pytest.mark.parametrize("threshold", [2, 3, 4, 6, 8])
def test_large_order_includes_shared_uses_service_estimate_and_keeps_small_order(threshold):
    experts = [expert(1, index=i) for i in range(12)]
    # Both priority tasks occur beyond the old eight-descriptor FIFO view.
    experts[9] = expert(threshold + 1, index=9)
    experts[10] = expert(1, index=10, shared=True)
    experts[11] = expert(threshold, index=11)
    estimates = [float(i + 1) for i in range(12)]
    estimates[9], estimates[10] = 40.0, 70.0
    order = runtime.priority_order(experts, estimates, threshold, True)
    assert list(order) == [10, 9, *range(9), 11]
    assert list(runtime.priority_order(experts, estimates, threshold, False)) == list(range(12))


@pytest.mark.parametrize("threshold", [2, 3, 4, 6, 8])
def test_runtime_prioritizes_large_descriptors_beyond_old_fifo_window(threshold):
    design = dual("OS")
    experts = [expert(1, index=i) for i in range(12)]
    experts[9] = expert(threshold + 1, index=9)
    experts[10] = expert(1, index=10, shared=True, f=256)
    estimates = [
        min(old.task_cost(e, design, core).isolated_cycles for core in (0, 1))
        for e in experts
    ]
    first_large = min((9, 10), key=lambda i: (-estimates[i], i))
    window = workload(experts)
    prioritized = runtime.simulate(window, design, t_big=threshold)
    fifo = runtime.simulate(window, design, t_big=threshold, large_first=False)
    assert prioritized["bindings"][0]["expert_index"] == first_large
    assert fifo["bindings"][0]["expert_index"] == 0
    assert {task["expert_index"] for task in prioritized["tasks"]} == set(range(12))


@pytest.mark.parametrize("reverse", [False, True])
def test_waiting_large_task_leaves_other_core_available_for_small_work(reverse):
    design = dual("OS", reverse=reverse)
    experts = [expert(3, index=0), expert(3, index=1), expert(1, index=2)]
    params = old.Parameters(binding_lead_cycles=64.0, prefetch=False)
    result = runtime.simulate(workload(experts), design, params, t_big=2)
    tasks = {task["expert_index"]: task for task in result["tasks"]}
    no_refetch_core = int(reverse)
    assert tasks[0]["core"] == tasks[1]["core"] == no_refetch_core
    assert tasks[2]["core"] == 1 - no_refetch_core
    assert tasks[2]["start"] < tasks[1]["start"]
    assert tasks[2]["start"] < tasks[0]["finish"]
    assert tasks[1]["start"] >= tasks[0]["finish"]
    assert len(result["tasks"]) == len(experts)
    assert all(binding["bounded_core_queue_depth"] <= 2 for binding in result["bindings"])
    assert result["hbm_bytes"] == sum(
        old.task_cost(experts[task["expert_index"]], design, task["core"], params).hbm_bytes
        for task in result["tasks"]
    )


@pytest.mark.parametrize("mode", ["pipelined", "port_tight", "fixed_issue"])
@pytest.mark.parametrize("core,flow", [
    (old.Core(6, 4, 512), "OS"),
    (old.Core(6, 16, 128), "WS"),
])
@pytest.mark.parametrize("learned", [False, True])
def test_single_core_is_bit_identical_across_modes_and_reused_predictors(mode, core, flow, learned):
    design = old.Design((core,), flows=(flow,))
    params = old.Parameters(onchip_mode=mode)
    old_predictor = Predictor("ours") if learned else None
    new_predictor = Predictor("ours") if learned else None
    windows = [
        workload([expert((i % 4) + 1, index=i, shared=i == 3) for i in range(6)], name="first"),
        workload([expert((i % 3) + 1, index=i, shared=i == 2) for i in range(5)], name="second"),
    ]
    for window in windows:
        reference = old.simulate(window, design, params, policy="eft", predictor=old_predictor)
        result = runtime.simulate(window, design, params, t_big=4, predictor=new_predictor)
        assert canonical(result) == canonical(reference)
        if learned:
            assert new_predictor.correction == old_predictor.correction
            assert new_predictor.counts == old_predictor.counts
            assert new_predictor.means == old_predictor.means
            assert new_predictor.calls == old_predictor.calls


def test_single_core_storage_chunk_path_is_bit_identical():
    design = old.Design((old.Core(6, 16, 128),), flows=("WS",))
    window = workload(
        [expert(256, h=2048, index=0), expert(256, h=2048, index=1, shared=True)],
        batch=256,
    )
    reference = old.simulate(window, design, policy="eft")
    result = runtime.simulate(window, design, t_big=8)
    assert reference["storage_chunks"] > 1
    assert canonical(result) == canonical(reference)


def test_dual_runtime_reuses_ours_learning_and_reports_actual_progress():
    class AuditedOurs(Predictor):
        def __init__(self):
            super().__init__("ours")
            self.progress_events = []

        def on_progress(self, e, c, elapsed, quarter, remaining):
            self.progress_events.append((e["id"], c, quarter, elapsed))
            return super().on_progress(e, c, elapsed, quarter, remaining)

    predictor = AuditedOurs()
    design = dual("WS")
    first_window = workload([expert((i % 4) + 1, index=i) for i in range(6)])
    first = runtime.simulate(first_window, design, predictor=predictor, t_big=2)
    learned = deepcopy(predictor.correction)
    assert learned
    assert sum(predictor.counts) == len(first["tasks"])
    assert len(predictor.progress_events) == 3 * len(first["tasks"])
    second_window = workload([expert(3, index=100 + i) for i in range(4)], name="second")
    second = runtime.simulate(second_window, design, predictor=predictor, t_big=2)
    first_binding = second["bindings"][0]
    e = second_window["experts"][first_binding["expert_index"]]
    key = predictor.okey(e, first_binding["core"])
    assert key in learned and learned[key] != 1.0
    assert first_binding["predicted_cycles"] == max(
        1.0, first_binding["nominal_cycles"] * learned[key],
    )
    assert first_binding["predicted_cycles"] != first_binding["nominal_cycles"]
    assert sum(predictor.counts) == len(first["tasks"]) + len(second["tasks"])
    assert len(predictor.progress_events) == 3 * (len(first["tasks"]) + len(second["tasks"]))
    assert {event[2] for event in predictor.progress_events} == {0.25, 0.5, 0.75}
    assert all(event[3] > 0.0 for event in predictor.progress_events)
