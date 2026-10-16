"""Feasible cooperative schedules on development-frozen full-MoE cores.

Shared-row slicing is exact within the contiguous Shared token-row contract;
routed experts remain whole jobs.  Sequential output-column cooperation is a
second feasible construction, not a global splitting or scheduling optimum.
Neither diagnostic changes the development-selected hardware geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from research.moe_dispatch.compute_ceiling.assignment_dp import exact_two_machine
from research.moe_dispatch.plena65536_shared.geometry import BUDGET, Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES, _split_monotone, phase
from research.moe_dispatch.plena65536_moe.model import group_window, job_cycles, schedule, useful_macs, unique_weight_bytes
from research.moe_dispatch.plena65536_moe.workloads import DEFAULT_OUTPUT, batch_balanced_metrics, load_workloads


def _legal_cores(cores):
    cores = tuple(cores)
    if len(cores) not in (1, 2) or any(not isinstance(core, Core) for core in cores):
        raise ValueError("One/two legal Core instances are required")
    if sum(core.macs for core in cores) != BUDGET:
        raise ValueError("Cores must total exactly 65,536 multipliers")
    return cores


def _shared_descriptor(window):
    shared = [e for e in window["experts"] if e.get("is_shared", False)]
    if len(shared) != 1 or int(shared[0]["Me"]) != int(window["batch"]):
        raise ValueError("One Shared descriptor must cover the whole batch")
    if sorted(int(t) for t in shared[0]["token_indices"]) != list(range(int(window["batch"]))):
        raise ValueError("Shared token indices must cover the whole batch")
    return shared[0]


def _shared_cost(core, descriptor, m, profile):
    if m == 0:
        return 0
    return job_cycles(core, dict(descriptor, Me=m), profile)


def schedule_shared_rows(cores, window, profile=PROFILES[0]):
    """Enumerate every Shared row cut and solve routed integer assignment.

    Both endpoint cuts are included, so this feasible makespan cannot exceed
    whole-expert ownership on the same fixed cores.  Each core runs its Shared
    slice first, followed by deterministic routed LPT jobs.  Ideal operands
    and the existing declared GU-to-Down expert dependency apply throughout.
    """
    cores = _legal_cores(cores)
    shared = _shared_descriptor(window)
    batch = int(window["batch"])
    routed = [e for e in window["experts"] if not e.get("is_shared", False)]
    groups = group_window({"experts": routed})
    prototypes = [experts[0] for _, experts in groups]
    counts = [len(experts) for _, experts in groups]
    costs = [[job_cycles(core, expert, profile) for expert in prototypes] for core in cores]
    if len(cores) == 1:
        cut, chosen = batch, tuple(counts)
        bases = (_shared_cost(cores[0], shared, batch, profile),)
        optimum = bases[0] + sum(cost * count for cost, count in zip(costs[0], counts))
        evaluated = 1
    else:
        cache = {}
        best = None
        for candidate_cut in range(batch + 1):
            base0 = _shared_cost(cores[0], shared, candidate_cut, profile)
            base1 = _shared_cost(cores[1], shared, batch - candidate_cut, profile)
            bases_key = (base0, base1)
            if bases_key not in cache:
                cache[bases_key] = exact_two_machine(costs[0], costs[1], counts,
                                                     base0=base0, base1=base1)
            finish, assignment = cache[bases_key]
            load0 = base0 + sum(cost * count for cost, count in zip(costs[0], assignment))
            key = (finish, candidate_cut, load0)
            if best is None or key < best[0]:
                best = (key, candidate_cut, assignment, bases_key)
        (optimum, _, _), cut, chosen, bases = best
        evaluated = batch + 1
    queues = [[] for _ in cores]
    for j, (_, experts) in enumerate(groups):
        for offset, expert in enumerate(experts):
            queues[0 if offset < chosen[j] else 1].append(expert)
    jobs, loads = [], []
    for owner, queue in enumerate(queues):
        start_row, stop_row = ((0, cut) if owner == 0 else (cut, batch))
        now = 0
        if stop_row > start_row:
            duration = _shared_cost(cores[owner], shared, stop_row - start_row, profile)
            jobs.append({"expert_id": int(shared["id"]), "is_shared": True,
                         "core_id": owner, "M": stop_row - start_row,
                         "M_start": start_row, "M_stop": stop_row,
                         "token_indices": list(range(start_row, stop_row)),
                         "H": int(shared["H"]), "F": int(shared["F"]),
                         "start": 0, "end": duration, "cycles": duration})
            now = duration
        queue.sort(key=lambda e: (-job_cycles(cores[owner], e, profile), int(e["id"])))
        for expert in queue:
            duration = job_cycles(cores[owner], expert, profile)
            jobs.append({"expert_id": int(expert["id"]), "is_shared": False,
                         "core_id": owner, "M": int(expert["Me"]),
                         "token_indices": list(expert["token_indices"]),
                         "H": int(expert["H"]), "F": int(expert["F"]),
                         "start": now, "end": now + duration, "cycles": duration})
            now += duration
        loads.append(now)
    assert max(loads) == optimum
    assert sorted(job["expert_id"] for job in jobs if not job["is_shared"]) == sorted(int(e["id"]) for e in routed)
    assert sorted(t for job in jobs if job["is_shared"] for t in job["token_indices"]) == list(range(batch))
    return {"cycles": int(optimum), "row_cut": int(cut), "core_cycles": loads,
            "kind": "shared_rows_routed_whole_exact", "contract": "shared_rows_routed_whole_exact",
            "jobs": jobs, "profile": profile.name, "row_cuts_evaluated": evaluated,
            "useful_MACs": useful_macs(window), "unique_weight_bytes": unique_weight_bytes(window),
            "scope": "Exact contiguous Shared-row cut plus exact whole-routed-expert ownership on frozen cores; no global column/split scheduling optimality claim"}


def sequential_cooperative_columns(cores, window, profile=PROFILES[0]):
    """Serialize all experts; cooperate on paired GU and then Down columns.

    Each expert completes its GU phase across both cores before Down starts.
    GU columns form a complete BF16 Z; disjoint Down output columns require no
    cross-core partial sums.  Perfect materialization and operand availability
    are assumed, consistently with the ideal-compute ceiling.
    """
    cores = _legal_cores(cores)
    if len(cores) == 1:
        result = schedule(cores, window, profile)
        result.update({"kind": "sequential_cooperative_columns", "scope": "Single-core whole-expert execution"})
        return result
    left, right = cores
    now, jobs = 0, []
    experts = sorted(window["experts"], key=lambda e: (not bool(e.get("is_shared", False)), int(e["id"])))
    for expert in experts:
        m, h, f = int(expert["Me"]), int(expert["H"]), int(expert["F"])
        gu, gu_cut = _split_monotone(f, lambda n: phase(left, m, n, h, 2, profile),
                                    lambda n: phase(right, m, n, h, 2, profile))
        down, down_cut = _split_monotone(h, lambda n: phase(left, m, n, f, 1, profile),
                                        lambda n: phase(right, m, n, f, 1, profile))
        for owner, core in enumerate(cores):
            gu_start, gu_stop = ((0, gu_cut) if owner == 0 else (gu_cut, f))
            down_start, down_stop = ((0, down_cut) if owner == 0 else (down_cut, h))
            jobs.append({"expert_id": int(expert["id"]), "is_shared": bool(expert.get("is_shared", False)),
                         "core_id": owner, "M": m, "H": h, "F": f,
                         "token_indices": list(expert["token_indices"]),
                         "GU_N_start": gu_start, "GU_N_stop": gu_stop,
                         "Down_N_start": down_start, "Down_N_stop": down_stop,
                         "start": now, "GU_end": now + gu, "end": now + gu + down,
                         "GU_active_cycles": phase(core, m, gu_stop - gu_start, h, 2, profile),
                         "Down_active_cycles": phase(core, m, down_stop - down_start, f, 1, profile)})
        now += gu + down
    return {"cycles": now, "core_cycles": [now, now], "kind": "sequential_cooperative_columns",
            "jobs": jobs, "profile": profile.name,
            "useful_MACs": useful_macs(window), "unique_weight_bytes": unique_weight_bytes(window),
            "scope": "Feasible sequential cooperative GU/Down columns for every Shared/hot/cold expert; globally optimal cooperative scheduling is not proven"}


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def write_frozen_diagnostics(selection: dict[str, Any], windows, output_directory=DEFAULT_OUTPUT, *, repeats=2):
    """Replay all frozen families and certify repeated primary-profile bytes."""
    if repeats < 1:
        raise ValueError("At least one replay is required")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    families = selection["families"]
    receipts, first_rows, summary = [], None, {}
    for repeat in range(repeats):
        rows = []
        cache = {}
        for family, selected in families.items():
            cores = tuple(Core(int(c["b"]), int(c["r"]), int(c["rm"]), int(c["rn"])) for c in selected["cores"])
            _legal_cores(cores)
            for window in windows:
                cache_key = (cores, str(window["id"]))
                if cache_key not in cache:
                    whole = schedule(cores, window, PROFILES[0])
                    shared_rows = schedule_shared_rows(cores, window, PROFILES[0])
                    columns = sequential_cooperative_columns(cores, window, PROFILES[0])
                    cache[cache_key] = whole, shared_rows, columns
                whole, shared_rows, columns = cache[cache_key]
                priority = {"whole_expert_exact": 0, "shared_rows_routed_whole_exact": 1,
                            "sequential_cooperative_columns": 2}
                best = min((dict(whole, kind="whole_expert_exact"), shared_rows, columns),
                           key=lambda p: (p["cycles"], priority[p["kind"]]))
                assert shared_rows["cycles"] <= whole["cycles"]
                rows.append({"family": family, "window_id": window["id"], "batch": int(window["batch"]),
                             "profile": PROFILES[0].name, "whole_cycles": whole["cycles"],
                             "shared_rows_cycles": shared_rows["cycles"], "shared_row_cut": shared_rows["row_cut"],
                             "sequential_columns_cycles": columns["cycles"], "best_feasible_cycles": best["cycles"],
                             "best_kind": best["kind"], "shared_rows_speedup_vs_whole": whole["cycles"] / shared_rows["cycles"],
                             "best_feasible_speedup_vs_whole": whole["cycles"] / best["cycles"]})
        data = _csv_bytes(rows)
        digest = hashlib.sha256(data).hexdigest()
        receipts.append({"repeat": repeat + 1, "rows": len(rows), "csv_sha256": digest})
        if first_rows is None:
            first_rows = rows
            (output / "split_diagnostic.csv").write_bytes(data)
        elif digest != receipts[0]["csv_sha256"]:
            raise AssertionError("Repeated frozen split-diagnostic bytes changed")
    for family in families:
        family_rows = [r for r in first_rows if r["family"] == family]
        summary[family] = {key: batch_balanced_metrics(windows, {r["window_id"]: r[key] for r in family_rows})
                           for key in ("whole_cycles", "shared_rows_cycles", "best_feasible_cycles",
                                       "shared_rows_speedup_vs_whole", "best_feasible_speedup_vs_whole")}
    metadata = {"status": "PASS", "profile": PROFILES[0].name,
                "frozen_geometry": True, "repeats": receipts, "summary": summary,
                "selection_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "scope": "Shared-row split is exact within its row-cut/whole-routed contract. Sequential all-expert cooperative columns are feasible only. Whole-expert DSE remains the sole globally searched ownership domain; arbitrary Shared/hot row/column splits and overlap are not globally proven."}
    (output / "SPLIT_DIAGNOSTIC.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    selected_path = args.selection or args.output / "FROZEN_SELECTION.json"
    selection = json.loads(selected_path.read_text())
    windows = load_workloads("heldout")["heldout"]
    result = write_frozen_diagnostics(selection, windows, args.output, repeats=args.repeats)
    print(json.dumps({"status": result["status"], "repeats": result["repeats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
