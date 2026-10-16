"""Independent exhaustive score audit for the finite affinity search.

Rebuilds the core domain and integer task costs without calling search's
enumerator, histogram, cost, policy-vector or score functions.  Every fixed
threshold score is repriced, every lower bound is checked as a valid dual,
and every explicitly evaluated adaptive candidate is replayed by the scalar
model and an independent task-queue reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import exp, fsum, log, prod
from pathlib import Path
import time

import numpy as np

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_dedicated.independent_audit import (
    independent_core_domain, independent_log_weights, integer_job_cycles,
)
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_affinity.model import schedule

THRESHOLDS = (1, 2, 4, 8)
DEFAULT_OUTPUT = Path("/scratch/shared/mcl123/plena/outputs/plena65536_affinity_20261013")


def reference_tables(descriptors, windows):
    """Independent all-frame three-GEMM integer prices and role populations."""
    raw = np.asarray(descriptors, dtype=np.int64)
    b, r = raw[:, 1], raw[:, 2]
    pm, pn = b * raw[:, 3], b * raw[:, 4]
    groups = r // b
    drain = b + 11 + np.asarray([(int(g) - 1).bit_length() for g in groups])
    types = sorted({(int(e["Me"]), int(e["H"]), int(e["F"]))
                    for w in windows for e in w["experts"]})
    mapping = {key: i for i, key in enumerate(types)}
    hist = np.zeros((len(windows), len(types)), dtype=np.int64)
    shared_flags = {}
    for wi, window in enumerate(windows):
        for e in window["experts"]:
            key = int(e["Me"]), int(e["H"]), int(e["F"])
            flag = bool(e.get("is_shared", False))
            assert key not in shared_flags or shared_flags[key] == flag
            shared_flags[key] = flag
            hist[wi, mapping[key]] += 1
    costs = np.zeros((len(raw), len(types)), dtype=np.int64)
    for ti, (m, h, f) in enumerate(types):
        rows = (m + pm - 1) // pm
        gu = 2 * rows * ((f + pn - 1) // pn) * ((h + r - 1) // r) * b
        down = rows * ((h + pn - 1) // pn) * ((f + r - 1) // r) * b
        costs[:, ti] = gu + down + 2 * drain
    dense = []
    stream = []
    for threshold in THRESHOLDS:
        flags = np.array([shared_flags[key] or key[0] > threshold for key in types], dtype=np.int64)
        dense.append(costs @ (hist * flags).T)
        stream.append(costs @ (hist * (1 - flags)).T)
    return costs, hist, np.stack(dense), np.stack(stream)


def reference_plan(cores, window, policy, threshold):
    """Current-window, independent scalar ownership using integer timing."""
    experts = sorted(window["experts"], key=lambda e: (
        -(3 * int(e["Me"]) * int(e["H"]) * int(e["F"])), int(e["id"])))
    costs = [[integer_job_cycles(c, e) for e in experts] for c in cores]

    def run(hard_threshold=None):
        loads = [0] * len(cores)
        owners = []
        for i, e in enumerate(experts):
            cutoff = threshold if hard_threshold is None else hard_threshold
            preferred = 0 if e.get("is_shared", False) or int(e["Me"]) > cutoff else 1
            if len(cores) == 1:
                owner = 0
            elif hard_threshold is not None:
                owner = preferred
            else:
                finish = [loads[c] + costs[c][i] for c in (0, 1)]
                owner = preferred if finish[0] == finish[1] else int(finish[1] < finish[0])
            loads[owner] += costs[owner][i]
            owners.append(owner)
        return max(loads), loads, owners

    if len(cores) == 1 or policy in ("single", "single_serial", "greedy_eft"):
        return run()
    if policy == "hard_threshold":
        return run(threshold)
    if policy not in ("adaptive_threshold", "workload_aware"):
        raise ValueError(policy)
    candidates = ([run()] if policy == "workload_aware" else []) + [run(t) for t in THRESHOLDS]
    return min(candidates, key=lambda item: item[0])


def reference_dual_gm(costs, pairs, hist, weights):
    """Evaluate the saved search's selected rational dual independently.

    For ANY t in [0,1], sum(n*min(t*a,(1-t)*b)) lower-bounds an integer
    makespan.  The particular t below is chosen from a ratio ordering but
    its validity does not require an exact sort.  Integer numerator products
    are bounded per pair before multiplication.  This reconstructs the
    mathematical dual value without the search's downward rounding guards.
    """
    left, right = costs[pairs[:, 0]], costs[pairs[:, 1]]
    job_count = max(sum(map(int, h)) for h in hist)
    maximum = np.iinfo(np.int64).max
    for a, b in zip(left.max(axis=1), right.max(axis=1)):
        assert int(a) * int(b) * job_count <= maximum
        assert int(a) + int(b) <= 2**53
    permutation = np.argsort(left / right, axis=1, kind="stable")
    left = np.take_along_axis(left, permutation, axis=1)
    right = np.take_along_axis(right, permutation, axis=1)
    dual = np.empty((len(pairs), len(hist)), dtype=np.float64)
    rows = np.arange(len(pairs))
    for wi, histogram in enumerate(hist):
        counts = histogram[permutation]
        prefix = (left * counts).cumsum(axis=1)
        suffix = (right * counts).sum(axis=1)[:, None] - (right * counts).cumsum(axis=1)
        pivot = np.argmax(prefix >= suffix, axis=1)
        av, bv = left[rows, pivot], right[rows, pivot]
        numerator = (np.minimum(left * bv[:, None], right * av[:, None]) * counts).sum(axis=1)
        dual[:, wi] = numerator.astype(np.float64) / (av + bv)
    return np.exp(np.log(dual) @ weights)


def check_selection_minima(out=DEFAULT_OUTPUT):
    """Recheck objective weights and every frozen winner against full grids."""
    out = Path(out)
    frozen = json.loads((out / "FROZEN_SELECTION.json").read_text())
    windows = load_workloads("development")["development"]
    weights = independent_log_weights(windows)
    rational = np.asarray(frozen["objective_exponents"]) / frozen["objective_root_denominator"]
    assert np.array_equal(rational, weights)
    single = np.load(out / "single_scores.npz")
    hom = np.load(out / "homogeneous_scores.npz")
    heterogeneous = np.load(out / "heterogeneous_search.npz")
    grids = {
        "single": single["GM_cycles"], "homogeneous": hom["greedy_GM"],
        "homogeneous_B": hom["portfolio_GM"],
        "heterogeneous_A": heterogeneous["fixed_GM_cycles"],
        "heterogeneous_B": heterogeneous["portfolio_GM_cycles"],
        "heterogeneous_adaptive_threshold": heterogeneous["adaptive_threshold_GM_cycles"],
        "heterogeneous_greedy": heterogeneous["greedy_GM_cycles"],
    }
    selected = {**frozen["families"], **frozen["diagnostics"]}
    checked = []
    for name, grid in grids.items():
        record = selected[name]
        index = int(record["source_index"])
        location = ((index, THRESHOLDS.index(record["threshold"]))
                    if grid.ndim == 2 else index)
        best = float(np.nanmin(grid))
        assert abs(float(grid[location]) / best - 1) < 2e-13
        reconstructed = exp(fsum(log(c) * float(w) for c, w in
                               zip(record["development_window_cycles"], weights)))
        assert abs(reconstructed / float(grid[location]) - 1) < 2e-13
        assert abs(reconstructed / record["development_GM_cycles"] - 1) < 2e-13
        checked.append(name)
    return checked


def run(out=DEFAULT_OUTPUT, *, chunk=4096):
    start = time.monotonic()
    out = Path(out)
    frozen = json.loads((out / "FROZEN_SELECTION.json").read_text())
    certificate = json.loads((out / "SEARCH_CERTIFICATE.json").read_text())
    inputs = load_workloads("development")
    windows = inputs["development"]
    assert len(windows) == 18
    assert frozen["development_window_ids"] == [w["id"] for w in windows]
    assert frozen["development_input_sha256"] == inputs["manifest"]["loaded_input_sha256"]
    assert not frozen["policy_B_is_hardware_implemented"]
    assert not frozen["uses_actual_future_service_times"]
    assert certificate["heldout_read_during_selection"] is False
    assert frozen["budget"] == 65536 and frozen["supply_latency_cycles"] == 0
    for filename, digest in frozen["source_sha256"].items():
        assert hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() == digest

    descriptors = independent_core_domain()
    cores = [Core(b, r, rm, rn) for _, b, r, rm, rn in descriptors]
    budgets = np.asarray([item[0] for item in descriptors], dtype=np.int64)
    by_budget = Counter(map(int, budgets))
    expected_pairs = sum(n * by_budget.get(65536 - m, 0) for m, n in by_budget.items()) - by_budget[32768]
    assert expected_pairs == frozen["counts"]["ordered_heterogeneous"]
    assert len(cores) == frozen["counts"]["individual_cores"]
    weights = independent_log_weights(windows)
    costs, hist, dense, stream = reference_tables(descriptors, windows)
    archives = np.load(out / "heterogeneous_search.npz")
    pairs = archives["pair_ids"]
    assert pairs.shape == (expected_pairs, 2)
    assert np.all(pairs >= 0) and np.all(pairs < len(cores))
    assert np.all(budgets[pairs].sum(axis=1) == 65536)
    assert np.all(pairs[:, 0] != pairs[:, 1])
    assert len(np.unique(pairs[:, 0].astype(np.int64) * len(cores) + pairs[:, 1])) == expected_pairs
    saved_a = archives["fixed_GM_cycles"]
    saved_adaptive = archives["adaptive_threshold_GM_cycles"]
    saved_lp = archives["LP_lower_GM_cycles"]
    maximum_a_relative = 0.0
    maximum_adaptive_relative = 0.0
    maximum_lp_relative = 0.0
    reported = time.monotonic()
    for first in range(0, len(pairs), chunk):
        ids = pairs[first:first + chunk]
        hard = np.maximum(dense[:, ids[:, 0]], stream[:, ids[:, 1]])
        recomputed = np.exp(np.log(hard) @ weights).T
        adaptive = np.exp(np.log(hard.min(axis=0)) @ weights)
        aa = saved_a[first:first + len(ids)]
        ab = saved_adaptive[first:first + len(ids)]
        maximum_a_relative = max(maximum_a_relative, float(np.max(np.abs(recomputed / aa - 1))))
        maximum_adaptive_relative = max(maximum_adaptive_relative, float(np.max(np.abs(adaptive / ab - 1))))
        assert np.allclose(recomputed, aa, rtol=2e-13, atol=1e-8)
        assert np.allclose(adaptive, ab, rtol=2e-13, atol=1e-8)
        dual = reference_dual_gm(costs, ids, hist, weights)
        recorded_bound = saved_lp[first:first + len(ids)]
        maximum_lp_relative = max(maximum_lp_relative, float(np.max(np.abs(recorded_bound / dual - 1))))
        assert np.all(recorded_bound <= dual * (1 + 2e-14))
        assert np.all(recorded_bound > dual * (1 - 2e-13))
        if time.monotonic() - reported >= 20:
            print("AUDIT A+LP", first + len(ids), "/", len(pairs), flush=True)
            reported = time.monotonic()

    saved_g = archives["greedy_GM_cycles"]
    saved_b = archives["portfolio_GM_cycles"]
    evaluated = archives["portfolio_evaluated"]
    assert evaluated.dtype == bool
    assert int(evaluated.sum()) == frozen["counts"]["B_exact_geometry_count"]
    assert np.isfinite(saved_g[evaluated]).all() and np.isfinite(saved_b[evaluated]).all()
    assert np.isnan(saved_g[~evaluated]).all() and np.isnan(saved_b[~evaluated]).all()
    scalar_count = 0
    for pi in np.flatnonzero(evaluated):
        cs = tuple(cores[int(i)] for i in pairs[pi])
        for ti, threshold in enumerate(THRESHOLDS):
            for policy, saved in (("greedy_eft", saved_g), ("workload_aware", saved_b)):
                cycles = []
                for window in windows:
                    scalar = schedule(cs, window, policy=policy, threshold=threshold)
                    reference = reference_plan(cs, window, policy, threshold)
                    assert scalar["cycles"] == reference[0]
                    assert scalar["core_cycles"] == reference[1]
                    assert [j["core_id"] for j in scalar["jobs"]] == reference[2]
                    cycles.append(reference[0])
                    scalar_count += 1
                score = exp(fsum(log(c) * float(w) for c, w in zip(cycles, weights)))
                assert abs(score / saved[pi, ti] - 1) < 2e-13

    selections = {**frozen["families"], **frozen["diagnostics"]}
    winner_checks = []
    exponents = frozen["objective_exponents"]
    for name, record in selections.items():
        cs = tuple(cores[int(i)] for i in record["indices"] if i >= 0)
        assert [c.to_dict() for c in cs] == record["cores"]
        expected = [reference_plan(cs, w, record["policy"], record["threshold"])[0] for w in windows]
        assert expected == record["development_window_cycles"]
        product_key = prod(int(c) ** int(e) for c, e in zip(expected, exponents))
        assert str(product_key) == record["objective_exact_integer_product"]
        winner_checks.append(name)
    # Every skipped geometry is worse even under the admissible relaxed
    # assignment.  The same test covers all four threshold tie parameters.
    worst_incumbent = max(frozen["families"]["heterogeneous_B"]["development_GM_cycles"],
                          frozen["diagnostics"]["heterogeneous_greedy"]["development_GM_cycles"])
    assert np.all(saved_lp[~evaluated] > worst_incumbent + 1e-7)
    # Reprice all single and homogeneous scores as additional baselines.
    singles = np.load(out / "single_scores.npz")
    single_ids = singles["pair_ids"][:, 0]
    single_cycles = costs[single_ids] @ hist.T
    assert np.array_equal(single_cycles, singles["window_cycles"])
    assert np.allclose(np.exp(np.log(single_cycles) @ weights), singles["GM_cycles"], rtol=2e-13)
    hom = np.load(out / "homogeneous_scores.npz")
    homogeneous_replays = 0
    for pi, ids in enumerate(hom["pair_ids"]):
        assert ids[0] == ids[1] and cores[int(ids[0])].macs == 32768
        cs = (cores[int(ids[0])], cores[int(ids[1])])
        for ti, threshold in enumerate(THRESHOLDS):
            c = [reference_plan(cs, w, "workload_aware", threshold)[0] for w in windows]
            assert abs(exp(fsum(log(x) * float(w) for x, w in zip(c, weights))) / hom["portfolio_GM"][pi, ti] - 1) < 2e-13
            homogeneous_replays += len(windows)
        g = [reference_plan(cs, w, "greedy_eft", 2)[0] for w in windows]
        assert abs(exp(fsum(log(x) * float(w) for x, w in zip(g, weights))) / hom["greedy_GM"][pi] - 1) < 2e-13
        homogeneous_replays += len(windows)
    minima_checks = check_selection_minima(out)
    result = {"status": "PASS", "development_only_selection": True,
              "independent_core_domain_count": len(cores), "ordered_heterogeneous_count": expected_pairs,
              "fixed_threshold_scores_recomputed": int(saved_a.size),
              "adaptive_threshold_scores_recomputed": len(pairs),
              "dual_lower_bound_scores_recomputed": len(pairs),
              "maximum_fixed_threshold_relative_error": maximum_a_relative,
              "maximum_adaptive_threshold_relative_error": maximum_adaptive_relative,
              "maximum_dual_relative_roundoff": maximum_lp_relative,
              "evaluated_B_geometry_count": int(evaluated.sum()),
              "independent_and_scalar_B_schedule_replays": scalar_count,
              "homogeneous_reference_replays": homogeneous_replays,
              "winner_development_cycle_and_exact_objective_checks": winner_checks,
              "winner_full_grid_minimum_checks": minima_checks,
              "all_pruned_candidates_provably_worse_than_both_incumbents": True,
              "source_hashes_match_frozen_selection": True,
              "elapsed_seconds": time.monotonic() - start}
    (out / "INDEPENDENT_SEARCH_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()
    run(args.out, chunk=args.chunk)
