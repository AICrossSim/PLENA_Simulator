"""Exact assignment gap diagnostic on already frozen hardware.

This is an offline lower bound for complete, indivisible expert jobs on two
fixed cores.  It changes neither the selected hardware nor the evaluated
runtime policy and is not a hardware implementation or a measured predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from research.moe_dispatch.compute_ceiling.assignment_dp import exact_two_machine
from research.moe_dispatch.plena65536_dedicated.geometry import BUDGET, Core
from research.moe_dispatch.plena65536_dedicated.model import job_cycles, job_issued_macs
from research.moe_dispatch.plena65536_moe.workloads import BATCHES, load_workloads
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .model import schedule

FAMILIES = ("homogeneous", "homogeneous_B", "heterogeneous_A", "heterogeneous_B")
DEFAULT_OUTPUT = Path("/scratch/shared/mcl123/plena/outputs/plena65536_affinity_20261013")


def _cores(meta):
    return tuple(Core(*(int(c[k]) for k in ("b", "r", "rm", "rn"))) for c in meta["cores"])


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _gm(values):
    values = list(values)
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def exact_plan(cores, window, profile=PROFILES[0]):
    """Solve grouped exact DP, then independently reconstruct individual jobs."""
    cores = tuple(cores)
    if len(cores) != 2 or sum(c.macs for c in cores) != BUDGET:
        raise ValueError("oracle requires two frozen cores totalling 65,536 multipliers")
    experts = [e for e in window["experts"] if int(e["Me"]) > 0]
    if len({int(e["id"]) for e in experts}) != len(experts):
        raise ValueError("expert identities must be unique")
    groups = defaultdict(list)
    for expert in experts:
        key = tuple(int(expert[k]) for k in ("Me", "H", "F")) + (bool(expert.get("is_shared", False)),)
        groups[key].append(expert)
    keys = sorted(groups)
    representatives = [groups[key][0] for key in keys]
    costs = [[job_cycles(core, e, profile) for e in representatives] for core in cores]
    counts = [len(groups[key]) for key in keys]
    optimum, assigned_to_zero = exact_two_machine(*costs, counts)
    jobs, loads = [], [0, 0]
    useful = issued = 0
    for key, number in zip(keys, assigned_to_zero):
        assert 0 <= number <= len(groups[key])
        for index, expert in enumerate(sorted(groups[key], key=lambda e: int(e["id"]))):
            owner = 0 if index < number else 1
            duration = job_cycles(cores[owner], expert, profile)
            work = 3 * int(expert["Me"]) * int(expert["H"]) * int(expert["F"])
            useful += work
            issued += job_issued_macs(cores[owner], expert)
            jobs.append({"expert_id": int(expert["id"]), "core_id": owner,
                         "is_shared": bool(expert.get("is_shared", False)),
                         "Me": int(expert["Me"]), "cycles": duration,
                         "start": loads[owner], "end": loads[owner] + duration})
            loads[owner] += duration
    assert sorted(j["expert_id"] for j in jobs) == sorted(int(e["id"]) for e in experts)
    assert max(loads) == optimum
    assert useful <= issued and useful <= BUDGET * optimum
    return {"cycles": int(optimum), "core_cycles": loads, "jobs": jobs,
            "useful_macs": useful, "issued_macs": issued,
            "group_types": [list(key) for key in keys], "group_counts": counts,
            "group_counts_core0": list(assigned_to_zero),
            "contract": "offline exact two-machine indivisible whole-expert assignment; frozen geometry"}


def _summary(rows, batch):
    item = {"design": rows[0]["design"], "split": rows[0]["split"],
            "row_type": "7batch_equal_GM" if batch == "7-Batch GM" else "batch_GM",
            "batch": batch, "window_id": "", "window_count": len(rows)}
    balanced = batch == "7-Batch GM"
    def aggregate(key):
        if balanced:
            return _gm(_gm(r[key] for r in rows if r["batch"] == b) for b in BATCHES)
        return _gm(r[key] for r in rows)
    for key in ("policy_ms", "oracle_ms", "single_ms", "policy_vs_oracle", "oracle_vs_single",
                "policy_vs_single"):
        item[key] = aggregate(key)
    item["oracle_gap_pct"] = 100 * (item["policy_vs_oracle"] - 1)
    item["max_policy_vs_oracle"] = max(r["policy_vs_oracle"] for r in rows)
    item["windows_policy_equals_oracle"] = sum(r["policy_cycles"] == r["oracle_cycles"] for r in rows)
    item["max_loss_cycles"] = max(r["policy_cycles"] - r["oracle_cycles"] for r in rows)
    return item


def run(output=DEFAULT_OUTPUT, *, repeats=2, include_development=False):
    """Evaluate frozen default-profile points, without changing search outputs."""
    if repeats != 2:
        raise ValueError("this diagnostic requires exactly two deterministic repetitions")
    output = Path(output)
    frozen_path = output / "FROZEN_SELECTION.json"
    initial_bytes = frozen_path.read_bytes()
    frozen = json.loads(initial_bytes)
    profile = next(p for p in PROFILES if p.name == frozen["profile_default"])
    inputs = load_workloads() if include_development else load_workloads("heldout")
    splits = ("development", "heldout") if include_development else ("heldout",)
    single = frozen["families"]["single"]
    single_cores = _cores(single)
    cache, rows, assignments, summary = {}, [], [], []
    solve_count = 0
    for split in splits:
        for window in inputs[split]:
            single_plan = schedule(single_cores, window, profile, single["policy"], single["threshold"])
            for name in FAMILIES:
                meta = frozen["families"][name]
                cores = _cores(meta)
                key = cores, window["id"], profile.name
                if key not in cache:
                    result = exact_plan(cores, window, profile)
                    repeat = exact_plan(cores, window, profile)
                    assert _canonical(result) == _canonical(repeat), "non-deterministic exact assignment"
                    cache[key] = result
                    solve_count += 2
                oracle = cache[key]
                policy = schedule(cores, window, profile, meta["policy"], meta["threshold"])
                repeat = schedule(cores, window, profile, meta["policy"], meta["threshold"])
                assert _canonical(policy) == _canonical(repeat)
                assert oracle["cycles"] <= policy["cycles"], "exact assignment cannot exceed feasible policy"
                assert oracle["useful_macs"] == policy["useful_MACs"]
                counts = defaultdict(int)
                for job in oracle["jobs"]:
                    role = "shared" if job["is_shared"] else "routed"
                    kind = "shared" if job["is_shared"] else "cold" if job["Me"] <= int(meta["threshold"]) else "hot"
                    counts[f"oracle_{role}_jobs_core{job['core_id']}"] += 1
                    if kind != role:
                        counts[f"oracle_{kind}_jobs_core{job['core_id']}"] += 1
                row = {"design": name, "split": split, "row_type": "window",
                       "batch": int(window["batch"]), "window_id": window["id"], "window_count": 1,
                       "policy": meta["policy"], "reference_threshold": int(meta["threshold"]),
                       "policy_cycles": policy["cycles"], "oracle_cycles": oracle["cycles"],
                       "single_cycles": single_plan["cycles"],
                       "policy_ms": policy["cycles"] / 1e6, "oracle_ms": oracle["cycles"] / 1e6,
                       "single_ms": single_plan["cycles"] / 1e6,
                       "policy_vs_oracle": policy["cycles"] / oracle["cycles"],
                       "oracle_vs_single": oracle["cycles"] / single_plan["cycles"],
                       "policy_vs_single": policy["cycles"] / single_plan["cycles"],
                       "oracle_gap_pct": 100 * (policy["cycles"] / oracle["cycles"] - 1),
                       "oracle_core0_cycles": oracle["core_cycles"][0],
                       "oracle_core1_cycles": oracle["core_cycles"][1],
                       "oracle_shared_core": next(j["core_id"] for j in oracle["jobs"] if j["is_shared"]),
                       "repeat_identical": True}
                row.update({f"oracle_{kind}_jobs_core{core}": counts[f"oracle_{kind}_jobs_core{core}"]
                            for kind in ("shared", "routed", "cold", "hot") for core in (0, 1)})
                rows.append(row)
                assignments.append({"design": name, "split": split, "window_id": window["id"],
                                    "batch": int(window["batch"]), "oracle": oracle})
        for name in FAMILIES:
            selected = [r for r in rows if r["design"] == name and r["split"] == split]
            for batch in BATCHES:
                summary.append(_summary([r for r in selected if r["batch"] == batch], batch))
            summary.append(_summary(selected, "7-Batch GM"))
    all_rows = rows + summary
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (output / "assignment_gap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    envelope = {"contract": "Diagnostic only. Offline exact assignment on each already frozen geometry; "
                             "whole-expert jobs, ideal compute. No runtime implementation and no search changes.",
                "grouped_solver": "compute_ceiling.assignment_dp.exact_two_machine integer grouped DP",
                "aggregation": "window GM within each batch; equal seven-batch GM",
                "profile": profile.name, "summary": summary, "windows": rows,
                "oracle_assignments": assignments}
    (output / "assignment_gap.json").write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n")
    audit = {"passed": True, "point_count": len(rows), "unique_oracle_points": len(cache),
             "exact_solver_calls": solve_count, "oracle_repeats": repeats,
             "policy_repeats": repeats, "all_oracles_no_slower_than_feasible_policy": True,
             "expert_exact_cover_and_work_conservation": True,
             "frozen_selection_unchanged": frozen_path.read_bytes() == initial_bytes,
             "frozen_selection_sha256": hashlib.sha256(initial_bytes).hexdigest(),
             "result_sha256": hashlib.sha256(_canonical(envelope).encode()).hexdigest()}
    assert audit["frozen_selection_unchanged"]
    (output / "ASSIGNMENT_GAP_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    text = ["# 冻结硬件的离线最优派工差距", "",
            "仅改变同一硬件上的任务归属，精确求解完整专家任务的两核最小 makespan。"
            "这是离线下界诊断，不是运行时实现，不改变开发集选型或策略 B 参数。", "",
            "| 配置 | 当前策略 GM(ms) | 同硬件离线最优 GM(ms) | 当前／离线 | 离线／最优单核 |",
            "|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["batch"] == "7-Batch GM" and row["split"] == "heldout":
            text.append(f"| {row['design']} | {row['policy_ms']:.9f} | {row['oracle_ms']:.9f} | "
                        f"{row['policy_vs_oracle']:.6f} | {row['oracle_vs_single']:.6f} |")
    text += ["", "完整逐窗口结果、逐 Batch GM 和每个专家的离线核归属见同目录 CSV/JSON。"
             "所有精确派工和原策略各重复两遍；相同几何与窗口复用已验证的两遍 oracle。"]
    (output / "assignment_gap.md").write_text("\n".join(text) + "\n")
    return audit, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-development", action="store_true")
    args = parser.parse_args()
    audit, summary = run(args.output, include_development=args.include_development)
    print(json.dumps(audit, indent=2))
    for row in summary:
        if row["batch"] == "7-Batch GM":
            print(row["design"], row["split"], row["policy_ms"], row["oracle_ms"], row["policy_vs_oracle"])
