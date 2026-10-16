"""Whole-expert affinity policies under the declared ideal-compute contract.

All router descriptors are known at dispatch.  Policy A fixes a cold/hot
threshold; policy B evaluates a small deterministic portfolio using the
declared shape costs, without observed completion times or future windows.
This is a zero-controller-latency analytical scheduling experiment, not a
timed implementation of a hardware predictor or a prefetch controller.
"""
from __future__ import annotations

from research.moe_dispatch.plena65536_dedicated.geometry import BUDGET
from research.moe_dispatch.plena65536_dedicated.model import (
    job_cycles, job_issued_macs,
)
from research.moe_dispatch.plena65536_moe.model import (
    balanced_score, cost_table, expert_key, unique_weight_bytes, useful_macs,
    window_histograms,
)
from research.moe_dispatch.plena65536_shared.timing import PROFILES

THRESHOLDS = (1, 2, 4, 8)
POLICIES = ("hard_threshold", "greedy_eft", "adaptive_threshold", "workload_aware")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_window(window):
    """Validate descriptor identities and geometry before pricing any work.

    Full captured Top-6 route conservation is checked by the existing frozen
    workload loader.  This lighter boundary also supports small hand examples
    with fewer experts, while refusing inactive, duplicate or unknown IDs.
    """
    try:
        batch = _integer(window["batch"], "batch")
        experts = window["experts"]
        if batch <= 0 or not experts:
            raise ValueError("A nonempty positive-batch task window is required")
        ids = []
        shared = []
        for e in experts:
            eid = _integer(e["id"], "expert ID")
            m, h, f = (_integer(e[key], key) for key in ("Me", "H", "F"))
            flag = e.get("is_shared", False)
            if not isinstance(flag, bool):
                raise ValueError("is_shared must be boolean")
            if not 0 < m <= batch or h <= 0 or f <= 0:
                raise ValueError("Experts require positive dimensions and 0 < Me <= batch")
            if flag:
                if eid != -1 or m != batch:
                    raise ValueError("Shared expert ID -1 must cover the complete batch")
                shared.append(e)
            elif not 0 <= eid < 64:
                raise ValueError("Unknown routed expert ID; expected an ID in [0,63]")
            ids.append(eid)
        if len(ids) != len(set(ids)):
            raise ValueError("Every expert ID must appear exactly once")
        if len(shared) != 1:
            raise ValueError("Exactly one Shared expert must cover the complete batch")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete or malformed expert descriptor") from exc


def ordered_experts(window):
    """Architecture-independent LPT proxy: useful work, then expert identity."""
    return sorted(window["experts"], key=lambda e: (
        -3 * int(e["Me"]) * int(e["H"]) * int(e["F"]), int(e["id"])))


def hard_owner(expert, threshold):
    return 0 if expert.get("is_shared", False) or int(expert["Me"]) > threshold else 1


def _plan(experts, costs, policy, threshold):
    owners = []
    loads = [0] * len(costs)
    for i, expert in enumerate(experts):
        preferred = hard_owner(expert, threshold)
        if len(costs) == 1:
            owner = 0
        elif policy == "hard_threshold":
            owner = preferred
        else:
            completion = [loads[c] + int(costs[c][i]) for c in (0, 1)]
            owner = (0 if completion[0] < completion[1] else
                     1 if completion[1] < completion[0] else preferred)
        owners.append(owner)
        loads[owner] += int(costs[owner][i])
    return {"owners": owners, "core_cycles": loads, "cycles": max(loads),
            "selected_policy": policy,
            "selected_threshold": threshold if policy == "hard_threshold" else None}


def plan_from_costs(experts, costs, policy="workload_aware", threshold=2):
    """Deterministic policy evaluation, also usable for vector-model checks.

    ``experts`` must be in ``ordered_experts`` order.  The cost matrix is
    core-by-expert.  EFT compares predicted queue finish plus whole-task cost;
    threshold preference is used only on an exact integer completion tie.
    Policy B chooses the minimum of EFT and four legal threshold plans.  Its
    tie ordering is EFT first, then thresholds 1, 2, 4, 8.  A plan is frozen
    before execution, so a different affinity is not task migration/stealing.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {POLICIES}")
    if type(threshold) is not int or threshold not in THRESHOLDS:
        raise ValueError(f"Threshold must be one of {THRESHOLDS}")
    if not experts or len(costs) not in (1, 2):
        raise ValueError("One/two cost rows and at least one expert are required")
    if any(len(row) != len(experts) for row in costs):
        raise ValueError("Every core cost row must include every expert")
    if any(int(c) <= 0 or int(c) != c for row in costs for c in row):
        raise ValueError("Compute costs must be positive integer cycles")
    if len(costs) == 1:
        result = _plan(experts, costs, "single_serial", threshold)
        result["candidate_cycles"] = {"single_serial": result["cycles"]}
        return result
    if policy in ("hard_threshold", "greedy_eft"):
        result = _plan(experts, costs, policy, threshold)
        key = f"hard_threshold_{threshold}" if policy == "hard_threshold" else policy
        result["candidate_cycles"] = {key: result["cycles"]}
        return result
    candidates = []
    if policy == "workload_aware":
        candidates.append(("greedy_eft", _plan(experts, costs, "greedy_eft", threshold)))
    for value in THRESHOLDS:
        candidates.append((f"hard_threshold_{value}",
                           _plan(experts, costs, "hard_threshold", value)))
    _, winner = min(candidates, key=lambda item: item[1]["cycles"])
    result = dict(winner)
    result["candidate_cycles"] = {key: candidate["cycles"] for key, candidate in candidates}
    return result


def schedule(cores, window, profile=PROFILES[0], policy="workload_aware", threshold=2):
    """Return a complete auditable serial-per-core schedule and work ledger."""
    cores = tuple(cores)
    if len(cores) not in (1, 2) or sum(c.macs for c in cores) != BUDGET:
        raise ValueError("One/two cores must total exactly 65,536 multipliers")
    validate_window(window)
    experts = ordered_experts(window)
    costs = [[job_cycles(core, e, profile) for e in experts] for core in cores]
    plan = plan_from_costs(experts, costs, policy, threshold)
    clocks = [0] * len(cores)
    core_useful = [0] * len(cores)
    core_issued = [0] * len(cores)
    cold = [0] * len(cores)
    hot = [0] * len(cores)
    jobs = []
    deviations = 0
    shared_core = None
    for i, (expert, owner) in enumerate(zip(experts, plan["owners"])):
        duration = int(costs[owner][i])
        slots = int(job_issued_macs(cores[owner], expert))
        work = 3 * int(expert["Me"]) * int(expert["H"]) * int(expert["F"])
        preferred = hard_owner(expert, threshold)
        is_shared = bool(expert.get("is_shared", False))
        deviations += int(len(cores) == 2 and owner != preferred)
        if is_shared:
            shared_core = owner
        elif int(expert["Me"]) <= threshold:
            cold[owner] += 1
        else:
            hot[owner] += 1
        jobs.append({"expert_id": int(expert["id"]), "is_shared": is_shared,
                     "core_id": owner, "M": int(expert["Me"]), "H": int(expert["H"]),
                     "F": int(expert["F"]), "start": clocks[owner],
                     "end": clocks[owner] + duration, "cycles": duration,
                     "issued_macs": slots, "useful_macs": work,
                     "dispatch_order": i, "preferred_core": preferred})
        clocks[owner] += duration
        core_useful[owner] += work
        core_issued[owner] += slots
    makespan = max(clocks)
    useful = useful_macs(window)
    issued = sum(core_issued)
    assert clocks == plan["core_cycles"] and makespan == plan["cycles"]
    assert sum(core_useful) == useful and issued >= useful
    assert useful <= BUDGET * makespan
    realizable = ([t for t in THRESHOLDS
                   if all(owner == hard_owner(e, t)
                          for e, owner in zip(experts, plan["owners"]))]
                  if len(cores) == 2 else [])
    boundary = {
        "reference_threshold": threshold, "realizable_thresholds": realizable,
        "shared_core": shared_core, "cold_jobs_by_core": cold, "hot_jobs_by_core": hot,
        "core_useful_MACs": core_useful, "core_issued_macs": core_issued,
        "idle_fractions": [1 - value / makespan for value in clocks],
        "imbalance_ratio": makespan / min(clocks) if min(clocks) else None,
        "bottleneck_core": "tie" if len(clocks) == 2 and clocks[0] == clocks[1]
            else int(max(range(len(clocks)), key=lambda c: clocks[c])),
    }
    return {"cycles": makespan, "core_cycles": clocks, "jobs": jobs,
            "shared_cycles": sum(j["cycles"] for j in jobs if j["is_shared"]),
            "routed_cycles": sum(j["cycles"] for j in jobs if not j["is_shared"]),
            "issued_macs": issued, "useful_MACs": useful,
            "unique_weight_bytes": unique_weight_bytes(window),
            "spatial_utilization": useful / issued,
            "wall_peak_utilization": useful / (BUDGET * makespan),
            "contract": "single_serial" if len(cores) == 1 else "whole_expert_preassigned",
            "policy": policy, "requested_threshold": threshold,
            "selected_policy": plan["selected_policy"],
            "selected_threshold": plan["selected_threshold"],
            "candidate_cycles": plan["candidate_cycles"],
            "affinity_deviation_count": deviations, "boundary_stats": boundary,
            "profile": profile.name, "migration_count": 0, "steal_count": 0,
            "supply_delay_cycles": 0, "controller_delay_cycles": 0,
            "cost_source": "declared_analytical_shape_model_not_observed_completion"}
