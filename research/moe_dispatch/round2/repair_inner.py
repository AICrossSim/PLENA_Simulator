"""Supplemental assignment verification; never rewrites frozen DSE evidence.

The frozen main search is retained verbatim. This CLI revisits only its
non-OPTIMAL (design, development-window) allocations with a larger finite
solver-effort budget. New owners are diagnostic: they do not retroactively
replace a headline latency or close an old temporal-search leaf.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT, MODES, canonical, decode_design, inputs, sha, write_csv, write_json
from .model import Parameters
from .optimizer import evaluate_design
from .search import _engine_hash, _workload_hash


def frozen_evidence_hashes(directory=ROOT / "results/E3"):
    """Guard completed main-search/headline artifacts, excluding live studies."""
    directory = Path(directory)
    names = ["FROZEN_SELECTION.json", "bnb_summary.json", "bnb_certificate.csv",
             "bnb_leaves.csv", "seed_leaves.csv", "schedule_gaps.csv",
             "schedule_gaps_protocol.json", "single_exhaustion_receipt.json"]
    names += [f"seed_points_{m}.json" for m in MODES]
    names += [f"bnb_{m}_{proof}.json" for m in MODES for proof in ("A", "B")]
    return {n: sha(directory / n) for n in names if (directory / n).exists()}


def collect_jobs(directory=ROOT / "results/E3", workloads=None):
    directory = Path(directory)
    workloads = inputs()["development"] if workloads is None else workloads
    engine = _engine_hash()
    workload_sha = _workload_hash(workloads)
    jobs = []
    counts = {}
    for mode in MODES:
        rows = json.loads((directory / f"seed_points_{mode}.json").read_text())
        unresolved = 0
        for point, row in enumerate(rows):
            if "invalid" in row:
                continue
            assert row["engine_sha256"] == engine, "frozen seed engine changed"
            assert row["workload_sha256"] == workload_sha, "frozen development inputs changed"
            assert len(row["allocation_statuses"]) == len(workloads)
            for wi, (w, status) in enumerate(zip(workloads, row["allocation_statuses"])):
                if status == "OPTIMAL":
                    continue
                unresolved += 1
                jobs.append({"case_index": len(jobs), "job_id": f"{mode}:{point}:{w['id']}",
                    "onchip_mode": mode, "seed_point_index": point,
                    "geometry": row["geometry"], "family": row["family"],
                    "design": row["design"], "parameters": row["parameters"],
                    "workload": w, "seed_status": status,
                    "seed_lb_ms": row["lb_ms"][wi],
                    "seed_replay_ms": row["latencies_ms"][wi],
                    "engine_sha256": engine, "workload_sha256": workload_sha})
        counts[mode] = unresolved
    return jobs, counts


def verify_job(job, effort_units=100.0):
    """Two complete old/new objects, immutable seed match, separate outcomes."""
    started = time.monotonic()
    d = decode_design(job["design"])
    p = Parameters(**job["parameters"])
    w = job["workload"]
    old = evaluate_design(w, d, p, max_seconds=10.0, detail=False)
    old_repeat = evaluate_design(w, d, p, max_seconds=10.0, detail=False)
    assert canonical(old) == canonical(old_repeat), "old full-object repeat changed"
    assert old["assignment"]["status"] == job["seed_status"]
    assert old["lb_cycles"] / 1e6 == job["seed_lb_ms"], "old seed LB no longer reproduces"
    assert old["milp_sched"]["latency_ms"] == job["seed_replay_ms"], "old seed latency no longer reproduces"
    new = evaluate_design(w, d, p, max_seconds=effort_units, detail=False)
    new_repeat = evaluate_design(w, d, p, max_seconds=effort_units, detail=False)
    assert canonical(new) == canonical(new_repeat), "higher-effort full-object repeat changed"
    assert canonical(old["runtime"]) == canonical(new["runtime"]), "unchanged online runtime unexpectedly changed"
    a, b = old["assignment"], new["assignment"]
    # A larger effort request is not permission to weaken coefficient precision.
    assert a["quantum"] == b["quantum"] == 1e-6
    owner_changes = sum(x != y for x, y in zip(a["owners"], b["owners"]))
    row = {"case_index": job["case_index"], "job_id": job["job_id"],
        "onchip_mode": job["onchip_mode"], "family": job["family"],
        "seed_point_index": job["seed_point_index"], "geometry": job["geometry"],
        "window_id": w["id"], "batch": w["batch"], "design": canonical(job["design"]),
        "old_status": a["status"], "new_status": b["status"], "new_allocation_optimal": b["optimal"],
        "old_lb_ms": old["lb_cycles"] / 1e6, "new_lb_ms": new["lb_cycles"] / 1e6,
        "old_resource_witness_upper_ms": a["objective_upper_cycles"] / 1e6,
        "new_resource_witness_upper_ms": b["objective_upper_cycles"] / 1e6,
        "unclosed_assignment_gap_ms": b["assignment_gap_cycles"] / 1e6,
        "old_replay_ms": old["milp_sched"]["latency_ms"],
        "new_replay_ms": new["milp_sched"]["latency_ms"],
        "new_replay_ratio_vs_old": new["milp_sched"]["cycles"] / old["milp_sched"]["cycles"],
        "old_owners": canonical(a["owners"]), "new_owners": canonical(b["owners"]),
        "owner_changes": owner_changes, "new_backend": b["assignment_backend"],
        "old_effort_units": 10.0, "new_effort_units": float(effort_units),
        "new_deterministic_work_limit": b["solver_budget"]["max_deterministic_time"],
        "quantum_cycles": b["quantum"], "repeat_identical": True,
        "frozen_seed_reproduced": True, "unchanged_runtime_reproduced": True,
        "old_full_object_sha256": hashlib.sha256(canonical(old).encode()).hexdigest(),
        "new_full_object_sha256": hashlib.sha256(canonical(new).encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "engine_sha256": job["engine_sha256"], "workload_sha256": job["workload_sha256"],
        "scope": "supplemental assignment/resource verification; frozen headline and old proof unchanged"}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--effort-units", type=float, default=100.0)
    ap.add_argument("--expected-cases", type=int, default=940)
    ap.add_argument("--limit", type=int, help="explicit preflight subset only; never full completion")
    ap.add_argument("--output-directory", type=Path, default=ROOT / "results/E3")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.jobs < 1 or args.effort_units < 10 or (args.limit is not None and args.limit < 1):
        ap.error("positive worker/subset count and effort at least the frozen10 required")
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    table = out / "inner_assignment_verification.csv"
    protocol_file = out / "inner_assignment_verification_protocol.json"
    jobs, counts = collect_jobs()
    assert len(jobs) == args.expected_cases, (len(jobs), args.expected_cases)
    before = frozen_evidence_hashes()
    selected = jobs[:args.limit] if args.limit else jobs
    engine = _engine_hash()
    source_sha = sha(Path(__file__))
    rows = []
    if table.exists():
        if not args.resume:
            ap.error("diagnostic output already exists; choose a new directory or --resume")
        prior = json.loads(protocol_file.read_text())
        assert prior["engine_sha256"] == engine and prior["effort_units"] == args.effort_units
        assert prior["frozen_evidence_sha256"] == before
        rows = list(csv.DictReader(table.open()))
        assert all(r["engine_sha256"] == engine for r in rows)
    done = {r["job_id"] for r in rows}
    assert len(done) == len(rows), "duplicate diagnostic resume rows"
    expected = {j["job_id"] for j in selected}
    assert done <= expected
    pending = [j for j in selected if j["job_id"] not in done]
    protocol = {"scope": "higher finite effort diagnostic; no retroactive headline/proof update",
        "engine_sha256": engine, "diagnostic_source_sha256": source_sha,
        "full_unresolved_seed_cases": len(jobs), "planned_cases": len(selected),
        "unresolved_seed_counts_by_mode": counts, "effort_units": args.effort_units,
        "deterministic_work_limit": args.effort_units * .01, "old_effort_units": 10.,
        "query_limit": 64, "quantum_cycles": 1e-6, "workers": args.jobs,
        "repeats": 2, "frozen_evidence_sha256": before, "completed": False,
        "preflight_subset": args.limit is not None}
    write_json(protocol_file, protocol)
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=args.jobs)
    futures = []
    try:
        futures = [pool.submit(verify_job, job, args.effort_units) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            # Per-result durability; fixed source/effort guard makes resume safe.
            if table.exists():
                with table.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
            else:
                write_csv(table, [row])
            if len(rows) % 32 == 0 or len(rows) == len(selected):
                print("inner assignment diagnostic", len(rows), "/", len(selected), flush=True)
    except BaseException:
        for f in futures:
            f.cancel()
        for worker in list((getattr(pool, "_processes", None) or {}).values()):
            worker.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        assert frozen_evidence_hashes() == before, "frozen evidence changed during diagnostic"
        raise
    else:
        pool.shutdown(wait=True)
    rows.sort(key=lambda r: int(r["case_index"]))
    write_csv(table, rows)
    assert frozen_evidence_hashes() == before, "frozen evidence changed during diagnostic"
    assert _engine_hash() == engine
    assert {r["job_id"] for r in rows} == expected and len(rows) == len(expected)
    protocol.update({"completed": True, "completed_cases": len(rows), "elapsed_seconds": time.monotonic() - started,
        "all_full_object_repeats_identical": all(str(r["repeat_identical"]) == "True" for r in rows),
        "new_status_counts": dict(Counter(r["new_status"] for r in rows)),
        "unresolved_cases": sum(r["new_status"] != "OPTIMAL" for r in rows),
        "changed_owner_cases": sum(int(r["owner_changes"]) > 0 for r in rows),
        "frozen_evidence_unchanged": True, "output_csv_sha256": sha(table)})
    write_json(protocol_file, protocol)
    print(json.dumps(protocol, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
