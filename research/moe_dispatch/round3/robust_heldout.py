"""旧5.4的留出集三目标诊断；不重新选择冻结主表硬件。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .common import ROOT, OLD, canonical, digest, sha, write_csv, write_json, inputs, gm, table
from .config import parameters, BATCHES, SEED
from .search import key, decode, FAMILY_LABELS
from ..round2.robust import objectives, bootstrap_choices

OUT = ROOT / "E4/robust_heldout"
OBJECTIVES = ("geomean", "cvar10", "minimax")


def near_witnesses(certificate, tolerance=.01):
    best = certificate["selected"]["score_ms"]
    rows = [r for r in certificate["witnesses"]
            if r.get("status", "evaluated") == "evaluated"
            and r["score_ms"] <= best * (1+tolerance)+1e-12]
    assert rows and any(key(decode(r["design"])) == key(decode(certificate["selected"]["design"])) for r in rows)
    return sorted(rows, key=lambda r: key(decode(r["design"])))


def physical_sources():
    files = [ROOT/f for f in ("model.py", "runtime.py", "optimizer.py", "config.py", "robust_heldout.py")]
    files += [OLD/f for f in ("model.py", "optimizer.py", "common.py", "robust.py")]
    files += [ROOT.parent/"geometry3d/compute.py"]
    return {str(p): sha(p) for p in files}


def make_plan():
    selection = json.loads((ROOT/"E4/selected_designs.json").read_text())
    groups, points, sources = [], {}, {}
    for path in sorted((ROOT/"E4/final_certificates").glob("*.json")):
        cert = json.loads(path.read_text())
        sources[str(path)] = sha(path)
        mode, constraint, family = cert["onchip_mode"], cert["constraint_group"], cert["family"]
        near = near_witnesses(cert)
        ids = []
        for row in near:
            physical_key = key(decode(row["design"]))
            point_id = hashlib.sha256((mode+physical_key).encode()).hexdigest()[:20]
            prior = points.get(point_id)
            assert prior is None or prior["physical_key"] == physical_key
            points[point_id] = {"point_id": point_id, "onchip_mode": mode, "physical_key": physical_key,
                "hardware": asdict(replace(decode(row["design"]), label="")), "parameters": cert["parameters"]}
            ids.append(point_id)
        frozen = selection["modes"][mode][constraint][FAMILY_LABELS[family]]
        frozen_id = hashlib.sha256((mode+key(decode(frozen))).encode()).hexdigest()[:20]
        assert frozen_id in ids
        groups.append({"onchip_mode": mode, "constraint_group": constraint, "family": family,
            "candidate_ids": ids, "frozen_id": frozen_id,
            "development_ms": {point_id: row["latencies_ms"] for point_id, row in zip(ids, near)},
            "development_best_ms": cert["selected"]["score_ms"], "proof_B_closed": cert["proof_B_closed"],
            "certificate_file": str(path.relative_to(ROOT)), "certificate_sha256": sha(path)})
    assert len(groups) == 20
    return selection, groups, points, sources


def save_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(str(path), "wb", mtime=0) as f:
        f.write(canonical(value).encode())


def load_gzip(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def metrics_from_results(window_inputs, results):
    assert len(window_inputs) == len(results) == 135
    return [{"window_id": w["id"], "batch": w["batch"], "cycles": r["cycles"],
             "latency_ms": r["cycles"]/1e6, "hbm_bytes": r["hbm_bytes"],
             "physical_digest": digest(r)} for w, r in zip(window_inputs, results)]


def reuse_existing(point, heldout, run_receipt, repeats, current_sources):
    for rec in repeats:
        if rec["dispatch"] != "milp" or rec["onchip_mode"] != point["onchip_mode"]:
            continue
        if key(decode(rec["hardware"])) != point["physical_key"]:
            continue
        assert canonical(rec["parameters"]) == canonical(point["parameters"])
        assert rec["exact_repeat"] and rec["result_digest"] == rec["repeat_digest"]
        assert run_receipt["source_unchanged"]
        for name in ("model.py", "runtime.py", "optimizer.py", "config.py"):
            path = str(ROOT/name)
            assert run_receipt["source_sha256"][path] == current_sources[path]
        assert run_receipt["input_manifest_sha256"] == sha(OLD/"results/E0/frozen_inputs.json")
        raw_path = ROOT/rec["raw_file"]
        assert sha(raw_path) == rec["raw_sha256"]
        results = load_gzip(raw_path)
        assert digest(results) == rec["result_digest"]
        assert [s["window_id"] for s in rec["solver_checks"]] == [w["id"] for w in heldout]
        rows = metrics_from_results(heldout, results)
        return {"point_id": point["point_id"], "hardware": point["hardware"], "parameters": point["parameters"],
            "rows": rows, "exact_repeat": True, "result_digest": rec["result_digest"],
            "repeat_digest": rec["repeat_digest"], "kind": "existing E5 independently repeated MILP+LPT physical replay",
            "raw_file": rec["raw_file"], "raw_sha256": rec["raw_sha256"],
            "solver_checks": rec["solver_checks"], "source_receipt": "E5/dispatch/RUN_RECEIPT.json",
            "source_receipt_sha256": sha(ROOT/"E5/dispatch/RUN_RECEIPT.json"),
            "source_repeat_receipt_sha256": sha(ROOT/"E5/dispatch/repeat_checks.json")}
    return None


def run_point(job):
    point, heldout = job
    from .model import _task_cached
    from .optimizer import evaluate_design
    d = decode(point["hardware"])
    p = parameters(point["onchip_mode"])
    assert canonical(asdict(p)) == canonical(point["parameters"])
    passes = []
    started = time.monotonic()
    for repetition in range(2):
        _task_cached.cache_clear()
        evaluations = []
        for w in heldout:
            result = evaluate_design(w, d, p, detail=False)
            assert result["legal"], (point["point_id"], w["id"], "heldout physical infeasible")
            assert result["lb_cycles"] <= result["milp_sched"]["cycles"]+1e-5
            evaluations.append({"window_id": w["id"], "batch": w["batch"], "evaluation": result})
        passes.append(evaluations)
    assert canonical(passes[0]) == canonical(passes[1]), "full solver/physical replay repeat mismatch"
    raw_path = OUT/"raw"/(point["point_id"]+".json.gz")
    save_gzip(raw_path, passes[0])
    results = [r["evaluation"]["milp_sched"] for r in passes[0]]
    receipt = {"point_id": point["point_id"], "hardware": point["hardware"], "parameters": point["parameters"],
        "rows": metrics_from_results(heldout, results), "exact_repeat": True,
        "result_digest": digest(passes[0]), "repeat_digest": digest(passes[1]),
        "kind": "new two-pass independently solved MILP+LPT physical replay",
        "raw_file": str(raw_path.relative_to(ROOT)), "raw_sha256": sha(raw_path),
        "solver_checks": [{"window_id": r["window_id"],
            "status": r["evaluation"]["assignment"]["status"],
            "allocation_optimal": r["evaluation"]["assignment"]["optimal"],
            "lb_cycles": r["evaluation"]["lb_cycles"],
            "owners": r["evaluation"]["assignment"]["owners"]} for r in passes[0]],
        "elapsed_seconds": time.monotonic()-started}
    write_json(OUT/"points"/(point["point_id"]+".json"), receipt)
    return receipt


def render(groups, points, receipts, selection, dev, heldout):
    candidates, winners, stability, per_window = [], [], [], []
    receipt_index = {r["point_id"]: r for r in receipts}
    common_id = {}
    for mode in ("pipelined", "port_tight"):
        b1 = key(decode(selection["modes"][mode]["C0"]["B1"]))
        common_id[mode] = hashlib.sha256((mode+b1).encode()).hexdigest()[:20]
    all_groups = list(groups)
    for mode in ("pipelined", "port_tight"):
        for constraint in ("C0", "C1"):
            family_groups = [g for g in groups if (g["onchip_mode"],g["constraint_group"]) == (mode,constraint)]
            ids = sorted(set(i for g in family_groups for i in g["candidate_ids"]), key=lambda i: points[i]["physical_key"])
            devrows = {i: next(g["development_ms"][i] for g in family_groups if i in g["development_ms"]) for i in ids}
            all_groups.append({"onchip_mode": mode, "constraint_group": constraint, "family": "all",
                "candidate_ids": ids, "frozen_id": None, "development_ms": devrows,
                "proof_B_closed": all(g["proof_B_closed"] for g in family_groups)})
    for g in all_groups:
        mode = g["onchip_mode"]
        ids = sorted(g["candidate_ids"], key=lambda i: points[i]["physical_key"])
        base = np.asarray([r["latency_ms"] for r in receipt_index[common_id[mode]]["rows"]])
        ratio = [np.asarray([r["latency_ms"] for r in receipt_index[i]["rows"]])/base for i in ids]
        met = [objectives(a, [w["batch"] for w in heldout]) for a in ratio]
        win = {name: min(range(len(ids)), key=lambda j:(met[j][name], points[ids[j]]["physical_key"])) for name in OBJECTIVES}
        agree = len(set(ids[j] for j in win.values())) == 1
        for j, point_id in enumerate(ids):
            vals = receipt_index[point_id]["rows"]
            common = {"onchip_mode": mode, "constraint_group": g["constraint_group"], "family": g["family"],
                "point_id": point_id, "geometry": decode(points[point_id]["hardware"]).geometry}
            candidates.append({**common, "hardware": canonical(points[point_id]["hardware"]),
                "heldout_geomean_ms": gm(r["latency_ms"] for r in vals),
                "GM_ratio": met[j]["geomean"], "CVaR10_ratio": met[j]["cvar10"],
                "worst_batch_ratio": met[j]["minimax"], "batch_ratios": canonical(met[j]["batch_geomeans"]),
                "is_primary_frozen_design": point_id == g["frozen_id"],
                **{"selected_by_"+name: j==win[name] for name in OBJECTIVES}})
            for w, r, paired in zip(heldout, vals, ratio[j]):
                per_window.append({**common, "window_id": w["id"], "batch": w["batch"],
                    "latency_ms": r["latency_ms"], "hbm_MiB": r["hbm_bytes"]/2**20, "ratio_vs_C0_B1": float(paired)})
        for name, j in win.items():
            point_id = ids[j]
            winners.append({"onchip_mode": mode, "constraint_group": g["constraint_group"], "family": g["family"],
                "objective": name, "point_id": point_id, "geometry": decode(points[point_id]["hardware"]).geometry,
                "objective_ratio": met[j][name], "three_objectives_same_physical_design": agree,
                "matches_primary_frozen_design": point_id == g["frozen_id"],
                "candidate_count": len(ids), "diagnostic_only": True})
        devbase = np.asarray(next(x["development_ms"][common_id[mode]] for x in groups
            if x["onchip_mode"]==mode and common_id[mode] in x["development_ms"]))
        matrix = [np.asarray(g["development_ms"][i])/devbase for i in ids]
        counts = bootstrap_choices(matrix, [w["batch"] for w in dev], draws=200, seed=SEED)
        for i, point_id in enumerate(ids):
            for name, vector in counts.items():
                stability.append({"onchip_mode": mode, "constraint_group": g["constraint_group"], "family": g["family"],
                    "point_id": point_id, "objective": name, "draws": 200, "selected_count": vector[i],
                    "selected_share": vector[i]/200, "most_selected": vector[i]==max(vector), "diagnostic_only": True})
    write_csv(OUT/"robust_objectives.csv", candidates)
    write_csv(OUT/"winners.csv", winners)
    write_csv(OUT/"selection_stability.csv", stability)
    write_csv(OUT/"per_window.csv", per_window)
    summary = ["# 开发集近优候选的留出集鲁棒诊断", "",
        "近优集合在开发集冻结：各模式/C0/C1/族已实际评估的候选，GM距该族已测最好值不超过1%。"
        "全域证明开放，不能称经证明的全域1%近优集合。本诊断不修改selected_designs.json或任何主表。", "",
        "三指标均使用相同模式C0冻结B1的逐窗口配对比值：GM；最差ceil(0.1×135)个比值的算术平均CVaR10；"
        "各batch配对GM的最大值。每配置在135个留出窗口上独立两遍，或引用已经核验的E5两遍物理回放。"
        "200次bootstrap只重采样18个开发窗口。留出赢家是后验诊断，不重新定主硬件，也不构成盲测。", "",
        table(["模式", "约束", "族", "近优候选", "三目标同配置", "GM赢家", "CVaR10赢家", "最坏batch赢家"],
            [[g["onchip_mode"],g["constraint_group"],g["family"],len(g["candidate_ids"]),
              next(r["three_objectives_same_physical_design"] for r in winners if all(r[k]==g[k] for k in ("onchip_mode","constraint_group","family"))),
              *[next(r["geometry"]+" ["+r["point_id"][:8]+"]" for r in winners
                     if all(r[k]==g[k] for k in ("onchip_mode","constraint_group","family")) and r["objective"]==name) for name in OBJECTIVES]] for g in all_groups]), "",
        "同形状不同容量/端口/数据流仍是不同配置，赢家相同按完整物理配置ID判断。MILP为资源分配松弛，"
        "LPT为有限资源合法回放；不称任意时序全局最优。本节均为BF16、256 GB/s上限、相位流体解析估计，非原生HBM/RTL。"]
    (OUT/"SUMMARY_ZH.md").write_text("\n".join(summary)+"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    assert 1 <= args.jobs <= 4, "finite parallelism avoids starving other campaigns"
    OUT.mkdir(parents=True, exist_ok=True)
    before = physical_sources()
    selected_sha = sha(ROOT/"E4/selected_designs.json")
    selection, groups, points, certificates = make_plan()
    ws = inputs(); dev, heldout = ws["development"], ws["heldout"]
    assert len(dev)==18 and len(heldout)==135
    count_rows = [{"onchip_mode": g["onchip_mode"], "constraint_group": g["constraint_group"], "family": g["family"],
        "near_candidates": len(g["candidate_ids"]), "naive_physical_window_passes": len(g["candidate_ids"])*270} for g in groups]
    write_csv(OUT/"candidate_counts.csv", count_rows)
    plan = {"groups": groups, "points": points, "certificate_sha256": certificates,
        "selected_designs_sha256": selected_sha, "input_manifest_sha256": sha(OLD/"results/E0/frozen_inputs.json"),
        "physical_source_sha256": before, "heldout_ids": [w["id"] for w in heldout], "development_ids": [w["id"] for w in dev]}
    plan_path = OUT/"CANDIDATE_PLAN.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == json.loads(canonical(plan)), "checkpoint plan/source mismatch"
    else:
        write_json(plan_path, plan)
    e5 = json.loads((ROOT/"E5/dispatch/RUN_RECEIPT.json").read_text())
    repeats = json.loads((ROOT/"E5/dispatch/repeat_checks.json").read_text())
    receipts, jobs = [], []
    started = time.monotonic()
    for point_id, point in sorted(points.items()):
        checkpoint = OUT/"points"/(point_id+".json")
        if checkpoint.exists():
            rec = json.loads(checkpoint.read_text())
            assert rec["exact_repeat"] and rec["result_digest"] == rec["repeat_digest"]
            assert key(decode(rec["hardware"])) == point["physical_key"]
            assert sha(ROOT/rec["raw_file"]) == rec["raw_sha256"]
            archived = load_gzip(ROOT/rec["raw_file"])
            assert digest(archived) == rec["result_digest"]
            physical = ([r["evaluation"]["milp_sched"] for r in archived]
                        if rec["kind"].startswith("new") else archived)
            assert canonical(rec["rows"]) == canonical(metrics_from_results(heldout, physical))
        else:
            rec = reuse_existing(point, heldout, e5, repeats, before)
            if rec is not None:
                write_json(checkpoint, rec)
        if rec is None:
            jobs.append((point,heldout))
        else:
            receipts.append(rec)
    group_count = sum(len(g["candidate_ids"]) for g in groups)
    print(f"近优组内候选{group_count}；唯一配置{len(points)}；已核验/复用{len(receipts)}；新增{len(jobs)}，窗口双遍{len(jobs)*270}", flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            pending = [pool.submit(run_point, job) for job in jobs]
            for future in as_completed(pending):
                receipts.append(future.result())
                write_json(OUT/"PROGRESS.json", {"completed_unique_points": len(receipts), "total_unique_points": len(points),
                    "elapsed_seconds": time.monotonic()-started, "all_completed_points_repeat_identical": True})
                print("近优留出诊断",len(receipts),"/",len(points),flush=True)
    assert physical_sources() == before
    assert sha(ROOT/"E4/selected_designs.json") == selected_sha
    assert all(sha(Path(p))==value for p,value in certificates.items())
    receipts.sort(key=lambda r:r["point_id"])
    render(groups, points, receipts, selection, dev, heldout)
    write_json(OUT/"repeat_checks.json", receipts)
    outputs = {str(p.relative_to(ROOT)): sha(p) for p in OUT.rglob("*") if p.is_file()
               and p.name not in ("RUN_RECEIPT.json", "COMPLETE.json", "PROGRESS.json")}
    write_json(OUT/"RUN_RECEIPT.json", {"command":sys.argv, "finished_utc":datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":time.monotonic()-started, "unique_points":len(points), "group_candidate_rows":sum(len(g["candidate_ids"]) for g in groups),
        "new_points":sum(r["kind"].startswith("new") for r in receipts),
        "reused_points":sum(r["kind"].startswith("existing") for r in receipts),
        "new_physical_window_passes":sum(r["kind"].startswith("new") for r in receipts)*270,
        "exact_repeat_all":True, "source_unchanged":True, "selected_hardware_unchanged":True,
        "source_sha256":before, "input_manifest_sha256":plan["input_manifest_sha256"],
        "candidate_plan_sha256":sha(plan_path), "output_sha256":outputs,
        "scope":"development-frozen near-best evaluated candidates; heldout objective diagnostics only; primary selection unchanged"})
    write_json(OUT/"COMPLETE.json", {"all_unique_points_repeated":True,"diagnostic_only":True,
        "selected_hardware_unchanged":True,"unique_points":len(points),"receipt_sha256":sha(OUT/"RUN_RECEIPT.json")})


if __name__ == "__main__":
    main()
