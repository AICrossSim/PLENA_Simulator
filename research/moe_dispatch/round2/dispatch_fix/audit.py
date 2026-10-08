"""Independent numerical audit of the saved dispatch-repair campaign.

The audit performs no new scheduling, simulation, optimization, or selection.
It compares saved outputs against frozen physical costs and original E4 rows,
and independently recomputes development objectives and bootstrap selection.
Only INDEPENDENT_AUDIT.json in the new evidence directory is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..common import ROOT, decode_design, inputs
from ..model import Parameters, task_cost


DESIGNS = ("B0", "B1", "B2", "best_hetero", "fixed_4+2")
MODES = ("pipelined", "port_tight")
DISPATCHES = ("eft_old", "milp", "fixed", "eft_ours_control")
CONFIGS = tuple((t, flag) for flag in (False, True) for t in (2, 3, 4, 6, 8))
CASE_ID = "v3_captured_mixed_heldout_gpqa_t128_l13"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(directory, name):
    """Accept an exact gzip archive after the large raw JSON is compressed."""
    path = directory / name
    if path.exists():
        with path.open() as stream:
            return json.load(stream), path
    path = directory / (name + ".gz")
    with gzip.open(path, "rt") as stream:
        return json.load(stream), path


def gm(values):
    values = list(values)
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def require(condition, label, **context):
    if not condition:
        raise AssertionError({"check": label, **context})


def audit(directory):
    directory = Path(directory).resolve()
    rows, numerical_path = load_json(directory, "per_window.json")
    manifest, manifest_path = load_json(directory, "frozen_designs.json")
    selection, selection_path = load_json(directory, "selection.json")
    data = inputs()
    workloads = {w["id"]: w for w in data["heldout"]}
    window_ids = [w["id"] for w in data["heldout"]]
    development_ids = [w["id"] for w in data["development"]]
    with (ROOT / "results/E4/per_window.csv").open() as stream:
        frozen_rows = list(csv.DictReader(stream))
    e4 = {(r["entry"], r["onchip_mode"], r["sched_type"], r["window_id"]): r
          for r in frozen_rows}
    frozen_checks = {name: sha(ROOT / name) == value
                     for name, value in manifest["source_hashes"].items()}
    require(len(frozen_checks) == 9 and all(frozen_checks.values()), "nine_frozen_hashes",
            checks=frozen_checks)
    decoded = {}
    design_checks = []
    for mode in MODES:
        for name in DESIGNS:
            encoded = manifest["modes"][mode][name]
            d = decode_design(encoded)
            decoded[mode, name] = d
            original_encodings = {r["design"] for r in frozen_rows
                                  if (r["entry"], r["onchip_mode"]) == (name, mode)}
            require(len(original_encodings) == 1 and
                    json.loads(next(iter(original_encodings))) == encoded,
                    "full_frozen_design", mode=mode, design=name)
            resource_ok = (
                d.total_macs == sum(c.macs for c in d.cores) == 12288 and
                d.ledger()["installed_storage_B"] == 2158592 and
                tuple(sum(getattr(d, key)) for key in
                      ("w_bytes", "x_bytes", "acc_bytes", "z_bytes")) ==
                (40960, 12288, 98304, 393216) and
                tuple(sum(getattr(d, key)) for key in
                      ("w_banks", "x_banks", "acc_banks", "vector_lanes")) ==
                (64, 24, 12, 64))
            require(resource_ok, "resource_budget", mode=mode, design=name)
            design_checks.append({"mode": mode, "design": name,
                                  "encoded_design_matches_E4": True, "resource_budget_exact": True})
    index = {(r["onchip_mode"], r["design"], r["dispatch"], r["window_id"]): r for r in rows}
    require(len(rows) == len(index) == 5400, "complete_unique_rows", count=len(rows))
    expected = {(m, n, k, wid) for m in MODES for n in DESIGNS
                for k in DISPATCHES for wid in window_ids}
    require(set(index) == expected, "complete_paired_window_sets")
    counts = Counter()
    for row in rows:
        mode, name, kind, wid = (row[key] for key in
                                ("onchip_mode", "design", "dispatch", "window_id"))
        w = workloads[wid]
        d = decoded[mode, name]
        p = Parameters(onchip_mode=mode)
        require(row["result_digest"] == row["repeat_digest"], "full_repeat_digest",
                mode=mode, design=name, dispatch=kind, window=wid)
        counts["full_repeat_digest_rows"] += 1
        require(row["batch"] == w["batch"] and
                row["latency_ms"] == row["cycles"] / 1e6, "units_and_batch", window=wid)
        tasks = row["tasks"]
        require(len(tasks) == len(w["experts"]) and
                {t["expert_index"] for t in tasks} == set(range(len(w["experts"]))),
                "exact_task_coverage", window=wid)
        chosen_costs = {t["expert_index"]: task_cost(w["experts"][t["expert_index"]], d, t["core"], p)
                        for t in tasks}
        require(sum(c.hbm_bytes for c in chosen_costs.values()) == row["hbm_bytes"],
                "physical_HBM_sum", window=wid)
        require(sum(c.unique_hbm_bytes for c in chosen_costs.values()) == row["native_unique_bytes"],
                "physical_unique_sum", window=wid)
        counts["physical_cost_conservation_rows"] += 1
        bindings = {b["expert_index"]: b for b in row["bindings"]}
        require(set(bindings) == set(chosen_costs), "binding_coverage", window=wid)
        for task in tasks:
            i = task["expert_index"]
            binding = bindings[i]
            require(task["core"] == binding["core"] and
                    task["start"] >= binding["bind_cycle"] - 1e-7 and
                    task["finish"] > task["start"], "physical_binding_grant", window=wid, expert=i)
            counts["physical_binding_grant_tasks"] += 1
            if kind == "fixed" and len(d.cores) == 2:
                require(binding["bounded_core_queue_depth"] <= 2, "finite_Current_Next", window=wid)
                counts["fixed_dual_queue_checks"] += 1
                candidates = binding["candidate_comparisons"]
                selected = next(q for q in candidates if q["core"] == task["core"])
                require(selected["admissible"], "actual_admission", window=wid)
                for candidate in candidates:
                    physical = task_cost(w["experts"][i], d, candidate["core"], p)
                    require(candidate["refetch"] == physical.hbm_bytes / physical.unique_hbm_bytes,
                            "candidate_physical_refetch", window=wid, expert=i)
                    require(candidate["predicted_finish"] ==
                            candidate["earliest_start"] + candidate["predicted_cycles"],
                            "candidate_finish_arithmetic", window=wid, expert=i)
                    if (selected["refetch"] > 1 and candidate["core"] != selected["core"] and
                            candidate["refetch"] == 1):
                        require(selected["predicted_finish"] < candidate["predicted_finish"],
                                "exact_rule1_immediate_refetch", window=wid, expert=i)
                        counts["refetch_vs_clean_rule_comparisons"] += 1
        for core in range(len(d.cores)):
            timeline = sorted((t for t in tasks if t["core"] == core), key=lambda t: t["start"])
            require(all(a["finish"] <= b["start"] + 1e-7 for a, b in zip(timeline, timeline[1:])),
                    "one_Current_per_core", window=wid, core=core)
        details = row["refetch_details"]
        require(row["refetch_tasks"] == len(details) ==
                sum(c.hbm_bytes > c.unique_hbm_bytes for c in chosen_costs.values()),
                "refetch_task_count", window=wid)
        for detail in details:
            i = detail["expert_index"]
            co = chosen_costs[i]
            options = []
            for core in range(len(d.cores)):
                try:
                    cost = task_cost(w["experts"][i], d, core, p)
                except ValueError:
                    continue
                options.append(cost)
            minimum = min(c.hbm_bytes for c in options)
            require(detail["chosen_hbm_bytes"] == co.hbm_bytes and
                    detail["minimum_hbm_bytes"] == minimum and
                    detail["excess_B"] == co.hbm_bytes - minimum and
                    detail["refetch_factor"] == co.hbm_bytes / co.unique_hbm_bytes,
                    "refetch_evidence_costs", window=wid, expert=i)
        if kind in ("eft_old", "milp"):
            sched = "runtime" if kind == "eft_old" else "milp"
            old = e4[name, mode, sched, wid]
            require(all(row[field] == float(old[field]) for field in
                        ("cycles", "latency_ms", "hbm_bytes", "native_unique_bytes")),
                    "exact_E4_reproduction", window=wid)
            counts["exact_E4_reproduction_rows"] += 1
        if name in ("B0", "B1") and kind in ("fixed", "eft_ours_control"):
            old = index[mode, name, "eft_old", wid]
            require(row["cycles"] == old["cycles"] and
                    row["result_digest"] == old["result_digest"], "single_bit_exact", window=wid)
            counts["single_fixed_regression_rows" if kind == "fixed" else
                   "single_control_regression_rows"] += 1

    with (directory / "development_per_window.csv").open() as stream:
        development = list(csv.DictReader(stream))
    groups = defaultdict(dict)
    for r in development:
        cfg = int(r["t_big"]), r["large_first"] == "True"
        key = r["onchip_mode"], r["design"], r["window_id"]
        require(key not in groups[cfg], "unique_development_row", config=cfg, key=key)
        groups[cfg][key] = r
        require(r["result_digest"] == r["repeat_digest"] and
                float(r["ratio"]) == float(r["fixed_cycles"]) / float(r["old_cycles"]),
                "development_repeat_and_ratio", config=cfg, key=key)
    expected_dev = {(m, n, wid) for m in MODES for n in DESIGNS for wid in development_ids}
    require(len(development) == 1800 and set(groups) == set(CONFIGS) and
            all(set(group) == expected_dev for group in groups.values()), "development_coverage")
    require(selection["development_windows"] == development_ids, "canonical_development_order")
    per_window = {cfg: [math.fsum(math.log(float(groups[cfg][m, n, wid]["ratio"]))
                                 for m in MODES for n in DESIGNS) / 10
                        for wid in development_ids] for cfg in CONFIGS}
    scores = {cfg: math.exp(math.fsum(values) / len(values)) for cfg, values in per_window.items()}
    winner = min(CONFIGS, key=lambda cfg: (scores[cfg], cfg[1], cfg[0]))
    chosen = selection["chosen"]
    require(winner == (chosen["t_big"], chosen["large_first"]), "development_only_winner")
    require(abs(scores[winner] - selection["selected_development_ratio"]) < 1e-14,
            "selected_objective_value")
    for score in selection["scores"]:
        require(abs(scores[score["t_big"], score["large_first"]] - score["ratio"]) < 1e-14,
                "all_development_objectives")
    require(selection["bootstrap_draws"] == 200 and selection["bootstrap_seed"] == 20261008,
            "bootstrap_protocol")
    rng = np.random.default_rng(20261008)
    bootstrap = Counter()
    for _ in range(200):
        indices = rng.integers(0, len(development_ids), size=len(development_ids))
        best = min(CONFIGS, key=lambda cfg:
                   (sum(per_window[cfg][int(i)] for i in indices) / len(indices), cfg[1], cfg[0]))
        bootstrap[best] += 1
    require(all(bootstrap[r["t_big"], r["large_first"]] == r["count"]
                for r in selection["bootstrap_selection_counts"]), "all_200_bootstrap_counts")

    traffic = []
    case = []
    for mode in MODES:
        for name in DESIGNS:
            for kind in ("eft_old", "fixed", "eft_ours_control"):
                group = [index[mode, name, kind, wid] for wid in window_ids]
                extra = [100 * (r["hbm_bytes"] / index[mode, name, "milp", r["window_id"]]["hbm_bytes"] - 1)
                         for r in group]
                traffic.append({"mode": mode, "design": name, "dispatch": kind,
                                "above_2pct_windows": sum(x > 2 + 1e-10 for x in extra),
                                "maximum_extra_pct": max(extra)})
            fixed = index[mode, name, "fixed", CASE_ID]
            milp = index[mode, name, "milp", CASE_ID]
            w = workloads[CASE_ID]
            d = decoded[mode, name]
            p = Parameters(onchip_mode=mode)
            fixed_tasks = {t["expert_index"]: t for t in fixed["tasks"]}
            milp_tasks = {t["expert_index"]: t for t in milp["tasks"]}
            delta = sum(task_cost(e, d, fixed_tasks[i]["core"], p).hbm_bytes -
                        task_cost(e, d, milp_tasks[i]["core"], p).hbm_bytes
                        for i, e in enumerate(w["experts"]))
            require(delta == fixed["hbm_bytes"] - milp["hbm_bytes"], "signed_case_task_deltas",
                    mode=mode, design=name)
            ratio = fixed["cycles"] / milp["cycles"]
            extra = 100 * (fixed["hbm_bytes"] / milp["hbm_bytes"] - 1)
            case.append({"mode": mode, "design": name, "fixed_ms": fixed["latency_ms"],
                         "milp_ms": milp["latency_ms"], "fixed_milp_ratio": ratio,
                         "fixed_hbm_MiB": fixed["hbm_bytes"] / 2**20,
                         "milp_hbm_MiB": milp["hbm_bytes"] / 2**20,
                         "extra_hbm_pct": extra, "signed_task_delta_MiB": delta / 2**20,
                         "latency_gate_ratio_limit": 1.02,
                         "hbm_gate_extra_pct_limit": 2,
                         "latency_gate_pass": ratio <= 1.02, "hbm_gate_pass": extra <= 2})
    require(all(sha(ROOT / name) == value for name, value in manifest["source_hashes"].items()),
            "frozen_hashes_unchanged_after_audit")
    return {
        "all_passed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent saved-output checks; no simulation, optimizer, or new policy; no frozen files written",
        "source_sha256": sha(Path(__file__)),
        "input_sha256": {path.name: sha(path) for path in
                         (numerical_path, manifest_path, selection_path, directory / "development_per_window.csv")},
        "heldout_unique_rows": len(index), "development_rows": len(development),
        "check_counts": dict(sorted(counts.items())), "all_nine_frozen_hashes": frozen_checks,
        "frozen_design_resource_checks": design_checks,
        "selection": {"chosen": chosen, "objective_ratio": scores[winner],
                      "bootstrap_counts_recomputed_exactly": True,
                      "winner_count": bootstrap[winner], "draws": 200,
                      "large_first_wins": sum(count for (threshold, flag), count in bootstrap.items() if flag)},
        "traffic_exception_counts": traffic, "gpqa_case_acceptance": case,
        "counterfactual_caveat": "Recorded candidate finish comparisons are predictions, not measured alternative schedules",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = audit(args.directory)
    target = args.directory.resolve() / "INDEPENDENT_AUDIT.json"
    target.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_passed": report["all_passed"],
                      "heldout_unique_rows": report["heldout_unique_rows"],
                      "development_rows": report["development_rows"],
                      "check_counts": report["check_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
