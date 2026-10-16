"""Finite-queue M-DPU under ideal *data* supply, with timed control.

This is a deterministic event model, not a claim of calibrated PLENA timing.
The arithmetic service model is inherited unchanged.  A descriptor enters
one of two depth-limited FIFOs in one cycle.  Each core has one active and one
shadow context; programming a shadow context takes four cycles by default.
This operation may overlap arithmetic but cannot remove arithmetic drain.

No sorting, future-window knowledge, memory delay, or accuracy oracle is used.
Source order is Shared followed by routed expert ID.  Ingress compares queue
drain estimates plus shape service.  Threshold affinity breaks exact ties;
an optional bypass can change the preferred owner or pull an unclaimed head.
"""
from __future__ import annotations

from collections import deque
from random import Random

from research.moe_dispatch.plena65536_affinity.model import validate_window
from research.moe_dispatch.plena65536_dedicated.geometry import BUDGET
from research.moe_dispatch.plena65536_dedicated.model import job_cycles, job_issued_macs
from research.moe_dispatch.plena65536_shared.timing import PROFILES

THRESHOLDS = (1, 2, 4, 8)
POLICIES = ("mdpu", "greedy", "greedy_eft", "hard_threshold", "random")


def resolve_profile(profile):
    if isinstance(profile, str):
        try:
            return next(p for p in PROFILES if p.name == profile)
        except StopIteration as exc:
            raise ValueError("Unknown arithmetic timing profile") from exc
    return profile


def prepare_experts(window):
    validate_window(window)
    return sorted(window["experts"], key=lambda e: (not e.get("is_shared", False), int(e["id"])))


def preferred_owner(expert, threshold):
    return 0 if expert.get("is_shared", False) or int(expert["Me"]) > threshold else 1


def choose_owner(preferred, finish_d, finish_s, space_d, space_s, bypass=True, wait_best=False):
    """Ready/valid ingress decision, matching the RTL selector contract.

    None is backpressure.  With bypass disabled only the preferred queue is
    legal; otherwise the earlier finishing non-full queue wins. The guarded
    ``wait_best`` variant compares both finishes *before* applying capacity;
    if the better queue is full it backpressures rather than forcing a spill.
    """
    spaces = (bool(space_d), bool(space_s))
    if not bypass:
        return preferred if spaces[preferred] else None
    if wait_best:
        best = 0 if finish_d < finish_s else 1 if finish_s < finish_d else preferred
        return best if spaces[best] else None
    if not any(spaces):
        return None
    if not spaces[0]:
        return 1
    if not spaces[1]:
        return 0
    return 0 if finish_d < finish_s else 1 if finish_s < finish_d else preferred


def compiler_coefficients(core, h, f, profile="pipelined_tree1"):
    """Exact constant coefficient for ceil(Me/PM)*coefficient + offset.

    PM is power-of-two in this experiment, so runtime ceil is add+shift.
    The coefficient multiply remains a constant multiplier, not a free shift.
    Serialized profiles fold their per-record drain into the coefficient.
    """
    profile = resolve_profile(profile)
    cd = lambda a, b: (a + b - 1) // b
    gate_records = 2 * cd(f, core.pn)
    down_records = cd(h, core.pn)
    coefficient = (gate_records * cd(h, core.r) * core.b
                   + down_records * cd(f, core.r) * core.b)
    if profile.serialized:
        coefficient += (gate_records + down_records) * profile.drain(core)
        offset = 0
    else:
        offset = 2 * profile.drain(core)
    return {"pm": core.pm, "coefficient": coefficient, "offset": offset}


def estimate_cycles(core, expert, profile="pipelined_tree1"):
    coeff = compiler_coefficients(core, int(expert["H"]), int(expert["F"]), profile)
    return ((int(expert["Me"]) + core.pm - 1) // core.pm) * coeff["coefficient"] + coeff["offset"]


def schedule(cores, window, profile="pipelined_tree1", policy="mdpu", threshold=2,
             queue_depth=8, prefetch=True, bypass=True, context_cycles=4,
             details=True, seed=0, wait_best=False):
    experts = prepare_experts(window)
    profile = resolve_profile(profile)
    costs = [[job_cycles(c, e, profile) for e in experts] for c in cores]
    return schedule_from_costs(cores, experts, costs, profile=profile,
                              policy=policy, threshold=threshold,
                              queue_depth=queue_depth, prefetch=prefetch,
                              bypass=bypass, context_cycles=context_cycles,
                              details=details, seed=seed, batch=int(window["batch"]),
                              wait_best=wait_best)


def schedule_from_costs(cores, experts, costs, profile="pipelined_tree1", policy="mdpu",
                        threshold=2, queue_depth=8, prefetch=True, bypass=True,
                        context_cycles=4, details=False, seed=0, batch=None, wait_best=False):
    """Fast event schedule using a cached core-by-expert service matrix.

    ``core_cycles`` is arithmetic busy time; ``core_finish_cycles`` includes
    idle/control time. ``cycles`` is the final completion clock. Queue entries
    and contexts never migrate after issue.  All controller costs also apply
    to single and homogeneous controls.
    """
    cores = tuple(cores)
    ncore, n = len(cores), len(experts)
    if ncore not in (1, 2) or sum(c.macs for c in cores) != BUDGET:
        raise ValueError("One or two cores must total exactly 65,536 MACs")
    if policy not in POLICIES:
        raise ValueError("Unknown scheduling policy")
    if type(threshold) is not int or threshold not in THRESHOLDS:
        raise ValueError("Threshold must be 1, 2, 4, or 8")
    if type(queue_depth) is not int or queue_depth < 1:
        raise ValueError("Queue depth must be a positive integer")
    if type(context_cycles) is not int or context_cycles < 0:
        raise ValueError("Context preparation must be non-negative integer cycles")
    if not n or len(costs) != ncore or any(len(row) != n for row in costs):
        raise ValueError("Cost matrix dimensions disagree with descriptors")
    costs = [[int(x) for x in row] for row in costs]
    if any(x <= 0 for row in costs for x in row):
        raise ValueError("Service cycles must be positive")
    if batch is None:
        batch = max(int(e["Me"]) for e in experts)
    if details:
        validate_window({"batch": batch, "experts": experts})

    # Every organization uses exactly the same per-core FIFO and context logic.
    queues = [deque() for _ in cores]
    shadow = [None] * ncore  # (expert index, ready clock, claim clock)
    active = [None] * ncore  # (expert index, completion clock)
    busy = [0] * ncore
    finish = [0] * ncore
    issued_by_core = [0] * ncore
    useful_by_core = [0] * ncore
    max_occupancy = [0] * ncore
    enqueued = [0] * n
    visible = [0] * n
    queue_owner = [None] * n
    owner = [None] * n
    predictions = [0] * n
    jobs = []
    issue_order = []
    steals = 0
    context_overlap = 0
    context_exposed = 0
    now, source, source_ready, completed = 0, 0, 0, 0
    last_accepted = -1
    rng = Random(seed)
    random_preferences = [rng.randrange(ncore) for _ in range(n)]
    use_bypass = bool(bypass and policy == "mdpu")

    def base_finish(c):
        value = max(now, active[c][1]) if active[c] is not None else now
        if shadow[c] is not None:
            i, ready, _ = shadow[c]
            value = max(value, ready) + costs[c][i]
        return value

    def predicted_finish(c, index, include_queue=True):
        value = base_finish(c)
        if include_queue:
            for i in queues[c]:
                value = max(value, visible[i]) + costs[c][i]
                if not prefetch:
                    value += context_cycles
        # Setup can overlap existing work, but cannot precede descriptor arrival.
        return max(value, now + 1 + context_cycles) + costs[c][index]

    def claim(c, index, stolen):
        nonlocal steals, context_overlap, context_exposed
        ready = now + context_cycles
        shadow[c] = (index, ready, now)
        owner[index] = c
        if stolen:
            steals += 1
        current_end = active[c][1] if active[c] is not None else now
        context_overlap += max(0, min(ready, current_end) - now)
        context_exposed += max(0, ready - max(now, current_end))

    # The number of control/compute events is O(experts), independent of the
    # arithmetic duration. Backpressure jumps to the next release event.
    while completed < n:
        for c in range(ncore):
            if active[c] is not None and active[c][1] <= now:
                active[c] = None
                completed += 1
        if completed == n:
            break

        # Retire current, activate a prepared shadow, and refill the newly freed
        # shadow in the same clock. context_cycles=0 reaches a fixed point.
        changed = True
        while changed:
            changed = False
            for c in range(ncore):
                if active[c] is None and shadow[c] is not None and shadow[c][1] <= now:
                    i, ready, claim_clock = shadow[c]
                    shadow[c] = None
                    end = now + costs[c][i]
                    active[c] = (i, end)
                    finish[c] = end
                    busy[c] += costs[c][i]
                    issue_order.append(i)
                    if details:
                        e = experts[i]
                        work = 3 * int(e["Me"]) * int(e["H"]) * int(e["F"])
                        slots = int(job_issued_macs(cores[c], e))
                        useful_by_core[c] += work
                        issued_by_core[c] += slots
                        jobs.append({"expert_id": int(e["id"]), "is_shared": bool(e.get("is_shared", False)),
                                     "core_id": c, "M": int(e["Me"]), "H": int(e["H"]), "F": int(e["F"]),
                                     "start": now, "end": end, "cycles": costs[c][i],
                                     "queue_entry": enqueued[i], "queue_visible": visible[i],
                                     "claim": claim_clock, "context_ready": ready,
                                     "queue_owner": queue_owner[i], "preferred_core": preferred_owner(e, threshold) if ncore == 2 else 0,
                                     "predicted_end_at_enqueue": predictions[i],
                                     "prediction_error": end - predictions[i],
                                     "issued_macs": slots, "useful_macs": work, "source_index": i})
                    changed = True

            # Both cores get first refusal on their own heads before cross-pull.
            eligible = [shadow[c] is None and (prefetch or active[c] is None) for c in range(ncore)]
            for c in range(ncore):
                if eligible[c] and queues[c] and visible[queues[c][0]] <= now:
                    claim(c, queues[c].popleft(), False)
                    eligible[c] = False
                    changed = True
            if ncore == 2 and use_bypass:
                for c in range(ncore):
                    other = 1 - c
                    if eligible[c] and not queues[c] and queues[other] and visible[queues[other][0]] <= now:
                        i = queues[other][0]
                        on_c = max(base_finish(c), now + context_cycles) + costs[c][i]
                        on_other = max(base_finish(other), now + context_cycles) + costs[other][i]
                        if on_c < on_other:
                            claim(c, queues[other].popleft(), True)
                            eligible[c] = False
                            changed = True

        accepted = False
        if source < n and source_ready <= now:
            i = source
            preferred = preferred_owner(experts[i], threshold) if ncore == 2 else 0
            predictions_by_core = [predicted_finish(c, i) for c in range(ncore)]
            spaces = [len(q) < queue_depth for q in queues]
            if ncore == 1:
                chosen = 0 if spaces[0] else None
            elif policy == "random":
                chosen = random_preferences[i] if spaces[random_preferences[i]] else None
            elif policy == "hard_threshold" or (policy == "mdpu" and not bypass):
                chosen = choose_owner(preferred, *predictions_by_core, *spaces, bypass=False)
            else:
                # Greedy EFT and M-DPU ingress have identical timing/cost models;
                # M-DPU additionally offers the optional post-enqueue head bypass.
                chosen = choose_owner(preferred, *predictions_by_core, *spaces,
                                      wait_best=bool(wait_best and policy == "mdpu"))
            if chosen is not None:
                queues[chosen].append(i)
                enqueued[i] = now
                visible[i] = now + 1
                queue_owner[i] = chosen
                predictions[i] = predictions_by_core[chosen]
                max_occupancy[chosen] = max(max_occupancy[chosen], len(queues[chosen]))
                source += 1
                source_ready = now + 1
                last_accepted = now
                accepted = True

        events = [a[1] for a in active if a is not None and a[1] > now]
        events += [s[1] for c, s in enumerate(shadow) if s is not None and active[c] is None and s[1] > now]
        events += [visible[q[0]] for q in queues if q and visible[q[0]] > now]
        if source < n and accepted:
            events.append(source_ready)
        elif source < n and source_ready > now:
            events.append(source_ready)
        if not events:
            raise AssertionError("M-DPU deadlock: descriptors remain without a future event")
        now = min(events)

    makespan = max(finish)
    result = {"cycles": makespan, "core_cycles": busy, "core_finish_cycles": finish,
              "queue_max_occupancy": max_occupancy, "queue_depth": queue_depth,
              "source_descriptor_count": n, "source_finish_cycle": last_accepted + 1,
              "frontend_backpressure_cycles": last_accepted + 1 - n,
              "context_service_cycles": n * context_cycles,
              "context_overlap_cycles": context_overlap, "context_exposed_cycles": context_exposed,
              "controller_delay_cycles": makespan - max(busy),
              "core_idle_cycles": [makespan - b for b in busy],
              "steal_count": steals, "migration_count": 0, "supply_delay_cycles": 0,
              "policy": policy, "threshold": threshold, "prefetch": bool(prefetch),
              "bypass": bool(bypass), "context_cycles": context_cycles,
              "profile": resolve_profile(profile).name,
              "contract": "finite_dual_fifo_current_shadow_context_ideal_data_supply"}
    # Preserve the exact legacy result serialization/hashes when the new
    # admission mechanism is disabled. This enables frozen-run regression.
    if wait_best:
        result["wait_best"] = True
    if details:
        useful = sum(useful_by_core)
        issued = sum(issued_by_core)
        assert useful <= issued and useful <= BUDGET * makespan
        assert len(jobs) == n and len(set(j["expert_id"] for j in jobs)) == n
        errors = [j["prediction_error"] for j in jobs]
        cold = [sum(int(not e.get("is_shared", False) and int(e["Me"]) <= threshold and owner[i] == c)
                    for i, e in enumerate(experts)) for c in range(ncore)]
        hot = [sum(int(not e.get("is_shared", False) and int(e["Me"]) > threshold and owner[i] == c)
                   for i, e in enumerate(experts)) for c in range(ncore)]
        realizable = ([t for t in THRESHOLDS if all(owner[i] == preferred_owner(e, t) for i, e in enumerate(experts))]
                      if ncore == 2 else [])
        boundary = {"reference_threshold": threshold, "realizable_thresholds": realizable,
                    "shared_core": owner[next(i for i, e in enumerate(experts) if e.get("is_shared", False))],
                    "cold_jobs_by_core": cold, "hot_jobs_by_core": hot,
                    "core_useful_MACs": useful_by_core, "core_issued_macs": issued_by_core,
                    "idle_fractions": [1 - b / makespan for b in busy],
                    "imbalance_ratio": max(busy) / min(busy) if min(busy) else None,
                    "bottleneck_core": "tie" if ncore == 2 and busy[0] == busy[1]
                    else max(range(ncore), key=lambda c: busy[c])}
        result.update({"jobs": jobs, "useful_MACs": useful, "issued_macs": issued,
                       "unique_weight_bytes": sum(6 * int(e["H"]) * int(e["F"]) for e in experts),
                       "wall_peak_utilization": useful / (BUDGET * makespan),
                       "spatial_utilization": useful / issued,
                       "padding_waste_fraction": 1 - useful / issued,
                       "boundary_stats": boundary,
                       "affinity_deviation_count": sum(int(owner[i] != preferred_owner(e, threshold)) for i, e in enumerate(experts)) if ncore == 2 else 0,
                       "prediction_mae_cycles": sum(abs(x) for x in errors) / len(errors),
                       "prediction_worst_underestimate_cycles": max(0, max(errors)),
                       "source_order": [int(e["id"]) for e in experts]})
    return result
