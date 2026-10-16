"""Independent integer references for the dedicated Shared/Routed study.

This file deliberately does not call the study's phase, job-cost, histogram,
scheduler or search helpers.  The reference replays the declared timing from
physical core descriptors, and checks affinity on actual per-expert queues.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, fsum, log

import numpy as np


def descriptor(core):
    """Accept an immutable Core or an archived core dictionary."""
    return core.to_dict() if hasattr(core, "to_dict") else core


def integer_job_cycles(core, expert, *, tree_stage=1, serialized=False):
    """Price Gate, Up and Down using independent positive-integer arithmetic."""
    c = descriptor(core)
    b, r, pm, pn, groups = (int(c[k]) for k in ("b", "r", "pm", "pn", "groups"))
    m, h, f = (int(expert[k]) for k in ("Me", "H", "F"))
    if min(b, r, pm, pn, groups, m, h, f) <= 0:
        raise ValueError("All core and active-expert dimensions must be positive")
    if r != b * groups or int(c["macs"]) != pm * pn * groups:
        raise ValueError("Physical expansion is inconsistent with its budget")
    rows = (m + pm - 1) // pm
    gate_up_tiles = 2 * rows * ((f + pn - 1) // pn)
    down_tiles = rows * ((h + pn - 1) // pn)
    gate_up_feed = ((h + r - 1) // r) * b
    down_feed = ((f + r - 1) // r) * b
    tree_levels = (groups - 1).bit_length()
    drain = b + 11 + tree_levels * tree_stage
    if serialized:
        return gate_up_tiles * (gate_up_feed + drain) + down_tiles * (down_feed + drain)
    return gate_up_tiles * gate_up_feed + down_tiles * down_feed + 2 * drain


def integer_window_cycles(cores, window, *, tree_stage=1, serialized=False):
    """Single owns every expert; two cores own Shared and Routed respectively."""
    if len(cores) not in (1, 2):
        raise ValueError("Expected one or two cores")
    if sum(int(descriptor(c)["macs"]) for c in cores) != 65_536:
        raise ValueError("Expected exactly 65,536 physical multipliers")
    loads = [0] * len(cores)
    for expert in window["experts"]:
        owner = 0 if len(cores) == 1 or expert.get("is_shared", False) else 1
        loads[owner] += integer_job_cycles(cores[owner], expert,
                                          tree_stage=tree_stage, serialized=serialized)
    return max(loads), loads


def integer_issued_macs(core, expert):
    """Count actual physical MAC slots in every feed tile, including padding."""
    c = descriptor(core)
    m, h, f = (int(expert[k]) for k in ("Me", "H", "F"))
    b, r, pm, pn = (int(c[k]) for k in ("b", "r", "pm", "pn"))
    rows = (m + pm - 1) // pm
    feed = (2 * rows * ((f + pn - 1) // pn) * ((h + r - 1) // r) * b
            + rows * ((h + pn - 1) // pn) * ((f + r - 1) // r) * b)
    return feed * int(c["macs"])


def independent_score(cycles, windows):
    """GM per Batch, then equal GM over Batches, without vector dot products."""
    if len(cycles) != len(windows):
        raise ValueError("Each window requires exactly one cycle value")
    logs = defaultdict(list)
    for value, window in zip(cycles, windows):
        if int(value) <= 0:
            raise ValueError("Compute cycles must be positive")
        logs[int(window["batch"])].append(log(int(value)))
    means = [fsum(values) / len(values) for values in logs.values()]
    return exp(fsum(means) / len(means))


def validate_conservation(window):
    """Independently recount Top-6 inputs, all GEMMs and unique weight bytes."""
    batch = int(window["batch"])
    actual_routes = Counter()
    token_indices = set()
    for token in window["tokens"]:
        token_indices.add(int(token["token_index"]))
        ids = [int(r["expert_id"]) for r in token["routes"]]
        assert len(ids) == len(set(ids)) == 6
        assert all(0 <= expert_id < 64 for expert_id in ids)
        actual_routes.update(ids)
    assert token_indices == set(range(batch))
    shared = [e for e in window["experts"] if e.get("is_shared", False)]
    routed = [e for e in window["experts"] if not e.get("is_shared", False)]
    assert len(shared) == 1 and int(shared[0]["Me"]) == batch
    assert int(shared[0]["id"]) == -1
    assert Counter({int(e["id"]): int(e["Me"]) for e in routed}) == actual_routes
    assert sum(int(e["Me"]) for e in routed) == batch * 6
    shared_macs = sum(3 * int(e["Me"]) * int(e["H"]) * int(e["F"]) for e in shared)
    routed_macs = sum(3 * int(e["Me"]) * int(e["H"]) * int(e["F"]) for e in routed)
    assert shared_macs == 3 * batch * 2048 * 2816
    assert routed_macs == 3 * shared_macs
    weight_bytes = sum(6 * int(e["H"]) * int(e["F"]) for e in window["experts"])
    assert weight_bytes == (33 + 16.5 * len(routed)) * 1024**2
    return {"shared_useful_MACs": shared_macs, "routed_useful_MACs": routed_macs,
            "total_useful_MACs": shared_macs + routed_macs,
            "unique_weight_bytes": weight_bytes}


def validate_schedule(cores, window, result, *, tree_stage=1, serialized=False):
    """Check every owner, queue boundary, duration and whole-layer completion."""
    total, loads = integer_window_cycles(cores, window,
                                          tree_stage=tree_stage, serialized=serialized)
    assert int(result["cycles"]) == total
    assert list(map(int, result["core_cycles"])) == loads
    experts = {int(e["id"]): e for e in window["experts"]}
    assert sorted(int(j["expert_id"]) for j in result["jobs"]) == sorted(experts)
    issued_macs = 0
    for core_id, core in enumerate(cores):
        now = 0
        jobs = sorted((j for j in result["jobs"] if int(j["core_id"]) == core_id),
                      key=lambda j: int(j["start"]))
        for job in jobs:
            expert = experts[int(job["expert_id"])]
            if len(cores) == 2:
                assert core_id == (0 if expert.get("is_shared", False) else 1)
            expected = integer_job_cycles(core, expert,
                                          tree_stage=tree_stage, serialized=serialized)
            assert int(job["start"]) == now
            assert int(job["end"]) == now + expected
            assert int(job["cycles"]) == expected
            expected_slots = integer_issued_macs(core, expert)
            assert int(job["issued_macs"]) == expected_slots
            issued_macs += expected_slots
            now += expected
        assert now == loads[core_id]
    assert int(result["issued_macs"]) == issued_macs
    useful = sum(3 * int(e["Me"]) * int(e["H"]) * int(e["F"]) for e in experts.values())
    assert useful <= issued_macs
    assert result["spatial_utilization"] == useful / issued_macs
    assert result["wall_peak_utilization"] == useful / (65536 * total)
    assert result["migration_count"] == result["steal_count"] == result["supply_delay_cycles"] == 0
    return {"cycles": total, "core_cycles": loads, "job_count": len(experts)}


def independent_pair_scores(cores, pairs, windows):
    """Exhaustive scalar reference for a small finite pair domain."""
    return np.asarray([independent_score(
        [integer_window_cycles((cores[int(a)], cores[int(b)]), w)[0] for w in windows],
        windows) for a, b in pairs])


def independent_core_domain(budget=65536, *, include_blen1=True):
    """Rebuild the full domain in PM/PN/G coordinates rather than replication.

    Each tuple is (MACs, BLEN, MLEN, row_replication, column_replication).
    The implementation never calls geometry.enumerate_cores or Core.
    """
    descriptors = []
    powers = [2**i for i in range(7)]
    for pm in powers:
        for b in powers:
            if b > pm or (b == 1 and not include_blen1):
                continue
            group = 1
            while b * group <= 2048:
                width_step = max(4, b)
                maximum_width = budget // (pm * group)
                for pn in range(width_step, maximum_width + 1, width_step):
                    descriptors.append((pm * pn * group, b, b * group,
                                        pm // b, pn // b))
                group *= 2
    return tuple(sorted(descriptors))


def independent_pair_domain(core_descriptors, *, budget=65536,
                            shared_not_smaller=True, exclude_identical=False):
    """Reconstruct ordered exact-budget pair IDs from complementary MAC sets."""
    by_budget = defaultdict(list)
    for index, (macs, b, r, rm, rn) in enumerate(core_descriptors):
        by_budget[macs].append((index, b * rm))
    for routed_budget in sorted(by_budget):
        shared_budget = budget - routed_budget
        if shared_not_smaller and shared_budget < routed_budget:
            continue
        for routed_id, pm in by_budget[routed_budget]:
            if pm not in (1, 2):
                continue
            for shared_id, _ in by_budget.get(shared_budget, ()):
                if exclude_identical and shared_id == routed_id:
                    continue
                yield shared_id, routed_id


def independent_role_load_tables(core_descriptors, windows, *, tree_stage=1,
                                 serialized=False):
    """Reprice every frame and window independently, retaining integer loads.

    Counts include the Shared flag so no equal-shaped expert can silently
    cross its owner.  The vector arithmetic implements the same declared
    one-cycle reduction profile as integer_job_cycles without importing any
    implementation pricing or histogram functions.
    """
    raw = np.asarray(core_descriptors, dtype=np.int64)
    b, r = raw[:, 1], raw[:, 2]
    pm, pn = b * raw[:, 3], b * raw[:, 4]
    groups = r // b
    drain = b + 11 + tree_stage * np.asarray([(int(g) - 1).bit_length() for g in groups], dtype=np.int64)
    counts = defaultdict(lambda: np.zeros(len(windows), dtype=np.int64))
    for wi, window in enumerate(windows):
        for expert in window["experts"]:
            key = (int(expert["Me"]), int(expert["H"]), int(expert["F"]),
                   bool(expert.get("is_shared", False)))
            counts[key][wi] += 1
    shared = np.zeros((len(raw), len(windows)), dtype=np.int64)
    routed = np.zeros_like(shared)
    for (m, h, f, is_shared), histogram in counts.items():
        rows = (m + pm - 1) // pm
        gu_tiles = 2 * rows * ((f + pn - 1) // pn)
        down_tiles = rows * ((h + pn - 1) // pn)
        h_feed = ((h + r - 1) // r) * b
        f_feed = ((f + r - 1) // r) * b
        cost = (gu_tiles * (h_feed + drain) + down_tiles * (f_feed + drain)
                if serialized else gu_tiles * h_feed + down_tiles * f_feed + 2 * drain)
        destination = shared if is_shared else routed
        destination += cost[:, None] * histogram[None, :]
    return shared, routed


def independent_log_weights(windows):
    batch_counts = Counter(int(w["batch"]) for w in windows)
    return np.asarray([1 / (len(batch_counts) * batch_counts[int(w["batch"])])
                       for w in windows], dtype=np.float64)


def independent_full_search(core_descriptors, windows, *, shared_not_smaller=True,
                            chunk=4096):
    """Independently exhaust every legal ordered pair; no bound/pruning."""
    shared, routed = independent_role_load_tables(core_descriptors, windows)
    weights = independent_log_weights(windows)
    budget = 65536
    total_loads = shared + routed
    single_ids = np.asarray([i for i, c in enumerate(core_descriptors) if c[0] == budget])
    homogeneous_ids = np.asarray([i for i, c in enumerate(core_descriptors) if c[0] == budget // 2])
    single_scores = np.exp(np.log(total_loads[single_ids]) @ weights)
    homogeneous_scores = np.exp(np.log(np.maximum(shared[homogeneous_ids], routed[homogeneous_ids])) @ weights)
    best_score = float("inf")
    best_pair = None
    pair_count = 0
    pending = []
    def consume(pairs):
        nonlocal best_score, best_pair, pair_count
        ids = np.asarray(pairs, dtype=np.int64)
        scores = np.exp(np.log(np.maximum(shared[ids[:, 0]], routed[ids[:, 1]])) @ weights)
        local_best = int(np.argmin(scores))
        if float(scores[local_best]) < best_score:
            best_score = float(scores[local_best])
            best_pair = tuple(map(int, ids[local_best]))
        pair_count += len(pairs)
    for pair in independent_pair_domain(core_descriptors, shared_not_smaller=shared_not_smaller):
        pending.append(pair)
        if len(pending) == chunk:
            consume(pending)
            pending = []
    if pending:
        consume(pending)
    return {
        "single_count": len(single_ids), "single_min_GM_cycles": float(single_scores.min()),
        "single_best_id": int(single_ids[int(np.argmin(single_scores))]),
        "homogeneous_count": len(homogeneous_ids),
        "homogeneous_min_GM_cycles": float(homogeneous_scores.min()),
        "homogeneous_best_id": int(homogeneous_ids[int(np.argmin(homogeneous_scores))]),
        "dedicated_count": pair_count, "dedicated_min_GM_cycles": best_score,
        "dedicated_best_ids": best_pair,
    }


def audit_saved_search(output, *, repeat_directory=None):
    """Audit all saved exact scores, all selected jobs and optional repeats."""
    import hashlib
    import json
    from math import lcm, prod
    from pathlib import Path
    from research.moe_dispatch.plena65536_moe.workloads import load_workloads
    from .geometry import Core
    from .model import schedule

    output = Path(output)
    frozen = json.loads((output / "FROZEN_SELECTION.json").read_text())
    inputs = load_workloads("development")
    development = inputs["development"]
    assert frozen["development_window_ids"] == [w["id"] for w in development]
    assert frozen["development_input_sha256"] == inputs["manifest"]["loaded_input_sha256"]
    assert frozen["clock_GHz"] == 1 and frozen["budget"] == 65536
    assert frozen["supply_delay_cycles"] == frozen["SRAM_port_wait_cycles"] == 0
    assert frozen["HBM_bandwidth"] == "infinite" and frozen["no_cross_core_substitution"] is True
    counts = Counter(int(w["batch"]) for w in development)
    denominator = lcm(*(len(counts) * n for n in counts.values()))
    exponents = [denominator // (len(counts) * counts[int(w["batch"])]) for w in development]
    assert exponents == frozen["objective_exponents"]
    assert denominator == frozen["objective_root_denominator"]
    for window in development:
        validate_conservation(window)
    all_descriptors = independent_core_domain()
    global_index = {descriptor: i for i, descriptor in enumerate(all_descriptors)}
    allocation = np.asarray([c[0] for c in all_descriptors], dtype=np.int64)
    domains = {}
    for extension, prefix in [(False, "strict"), (True, "extended")]:
        local = independent_core_domain(include_blen1=extension)
        domains[prefix + "_single"] = np.asarray(
            [(global_index[c], -1) for c in local if c[0] == 65536], dtype=np.int32)
        domains[prefix + "_homogeneous"] = np.asarray(
            [(global_index[c], global_index[c]) for c in local if c[0] == 32768], dtype=np.int32)
        for primary, suffix in [(False, "free"), (True, "largeShared")]:
            flat = (global_index[local[ix]]
                    for pair in independent_pair_domain(local, shared_not_smaller=primary,
                                                        exclude_identical=True)
                    for ix in pair)
            domains[prefix + "_dedicated_" + suffix] = np.fromiter(flat, dtype=np.int32).reshape(-1, 2)
    weights = independent_log_weights(development)
    profiles = {"pipelined_tree1": (1, False), "pipelined_tree2": (2, False),
                "serialized_output": (1, True)}
    families = []
    for profile_name, (tree_stage, serialized) in profiles.items():
        shared, routed = independent_role_load_tables(all_descriptors, development,
                                                      tree_stage=tree_stage,
                                                      serialized=serialized)
        for family, selection in frozen["profile_optima"][profile_name].items():
            chosen_cores = selection["cores"]
            selected = tuple(global_index[(int(c["macs"]), int(c["b"]), int(c["r"]),
                                            int(c["rm"]), int(c["rn"]))]
                             for c in chosen_cores)
            expected_cycles = [integer_window_cycles(chosen_cores, w,
                                                      tree_stage=tree_stage,
                                                      serialized=serialized)[0]
                               for w in development]
            assert expected_cycles == selection["development_window_cycles"]
            role0 = [integer_job_cycles(chosen_cores[0], next(
                e for e in w["experts"] if e.get("is_shared", False)),
                tree_stage=tree_stage, serialized=serialized) for w in development]
            routed_core = chosen_cores[-1]
            role1 = [sum(integer_job_cycles(routed_core, e, tree_stage=tree_stage,
                                           serialized=serialized)
                         for e in w["experts"] if not e.get("is_shared", False))
                     for w in development]
            assert role0 == selection["development_Shared_cycles"]
            assert role1 == selection["development_Routed_cycles"]
            objective_product = prod(c**p for c, p in zip(expected_cycles, exponents))
            assert objective_product == int(selection["objective_exact_integer_product"])
            max_relative_score_error = 0.0
            if family != "original":
                path = output / f"{family}_{profile_name}_exact.npz"
                with np.load(path, allow_pickle=False) as archive:
                    ids = archive["core_ids"]
                    scores = archive["exact_GM_cycles"]
                    assert ids.dtype == np.int32 and scores.dtype == np.float64
                    assert np.array_equal(ids, domains[family])
                    assert len(scores) == len(ids) == selection["geometry_count"]
                    assert np.all(np.isfinite(scores)) and np.all(scores > 0)
                    for start in range(0, len(ids), 8192):
                        group = ids[start:start + 8192]
                        cycles = (shared[group[:, 0]] + routed[group[:, 0]]
                                  if np.all(group[:, 1] == -1)
                                  else np.maximum(shared[group[:, 0]], routed[group[:, 1]]))
                        expected_scores = np.exp(np.log(cycles) @ weights)
                        observed = scores[start:start + len(group)]
                        error = float(np.max(np.abs(observed / expected_scores - 1)))
                        max_relative_score_error = max(max_relative_score_error, error)
                        assert error <= 5e-13
                    near = np.flatnonzero(scores <= scores.min() * (1 + 1e-12))
                    candidate_products = []
                    for ix in near:
                        a, b = map(int, ids[ix])
                        values = shared[a] + routed[a] if b < 0 else np.maximum(shared[a], routed[b])
                        candidate_products.append(prod(int(c)**p for c, p in zip(values, exponents)))
                    assert objective_product == min(candidate_products)
                    assert abs(selection["development_GM_cycles"] / scores.min() - 1) <= 1e-12
                    if "largeShared" in family:
                        assert allocation[selected[0]] >= allocation[selected[1]]
                    if "dedicated" in family:
                        assert int(chosen_cores[1]["pm"]) in (1, 2)
                        assert chosen_cores[0] != chosen_cores[1]
                    if repeat_directory:
                        with np.load(Path(repeat_directory) / path.name, allow_pickle=False) as repeated:
                            assert np.array_equal(ids, repeated["core_ids"])
                            assert np.array_equal(scores, repeated["exact_GM_cycles"])
            families.append({"profile": profile_name, "family": family,
                             "geometry_count": selection["geometry_count"],
                             "independent_scores_max_relative_error": max_relative_score_error,
                             "integer_objective_product_equal": True})
    # Read heldout only after all independent development selection checks.
    heldout_inputs = load_workloads("heldout")
    heldout = heldout_inputs["heldout"]
    jobs_checked = 0
    schedules_checked = 0
    for window in heldout:
        validate_conservation(window)
    from research.moe_dispatch.plena65536_shared.timing import PROFILES
    for profile in PROFILES:
        for selection in frozen["profile_optima"][profile.name].values():
            cores = tuple(Core(int(c["b"]), int(c["r"]), int(c["rm"]), int(c["rn"]))
                          for c in selection["cores"])
            for window in heldout:
                result = schedule(cores, window, profile)
                validate_schedule(cores, window, result,
                                  tree_stage=profile.tree_stage, serialized=profile.serialized)
                jobs_checked += len(result["jobs"])
                schedules_checked += 1
    audit = {"status": "PASS", "auditor": "independent PM/PN/G enumeration and integer timing",
             "all_profiles_all_candidates_repriced": True, "families": families,
             "development_window_count": len(development), "heldout_window_count": len(heldout),
             "schedules_checked": schedules_checked, "jobs_checked": jobs_checked,
             "every_job_fixed_owner": True, "zero_supply_wait_steal_migration": True,
             "true_top6_and_three_gemm_work_conserved": True,
             "repeat_NPZ_exactly_equal": bool(repeat_directory),
             "frozen_selection_sha256": hashlib.sha256((output / "FROZEN_SELECTION.json").read_bytes()).hexdigest(),
             "independent_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "development_input_sha256": inputs["manifest"]["loaded_input_sha256"],
             "heldout_input_sha256": heldout_inputs["manifest"]["loaded_input_sha256"]}
    (output / "INDEPENDENT_SEARCH_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat")
    arguments = parser.parse_args()
    audit = audit_saved_search(arguments.out, repeat_directory=arguments.repeat)
    print(json.dumps({k: v for k, v in audit.items() if k != "families"}, indent=2))
