"""Full-bandwidth sensitivity only: frozen WS compute/private storage, 520 credits.

No hardware or predictor retuning. This changes the shared service window, not
the BF16 format or per-core SRAM/compute. Controller area is not established.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from ...common import ROOT, BATCHES, canonical, decode_design, encode_design, inputs, sha, write_json
from ...model import Parameters
from ...predictors import Predictor
from ..predictor_ablation_20261008.bounded_runtime import simulate, patch_metadata
from ..predictor_ablation_20261008.nominal_control import NominalPredictor
from ..dataflow_ablation_20261008.run import digest, gm, table

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
OLD = PARENT/"predictor_ablation_20261008"
MODES = ("pipelined", "port_tight")
NAMES = ("B1", "B2", "best_hetero")
PREDICTORS = ("nominal", "ours")
WS = MANIFEST = SELECTED = None


def init_worker():
    global WS, MANIFEST, SELECTED
    WS = inputs()
    MANIFEST = json.loads((PARENT/"frozen_designs.json").read_text())
    SELECTED = json.loads((PARENT/"selection.json").read_text())["chosen"]
    assert SELECTED == {"t_big": 3, "large_first": True}


def csv_text(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def save_csv(name, rows):
    (HERE/name).write_text(csv_text(rows))


def job(spec):
    mode, name, pname = spec
    original = decode_design(MANIFEST["modes"][mode][name])
    d = replace(original, flows=("WS",)*len(original.cores))
    old_hw, new_hw = encode_design(original), encode_design(d)
    assert {k:v for k,v in old_hw.items() if k != "flows"} == {k:v for k,v in new_hw.items() if k != "flows"}
    p = Parameters(onchip_mode=mode, credits=520, hbm_Bpc=256.0,
                   hbm_latency=64.0, request_bytes=32)
    assert p.hbm_bandwidth == 256.0
    before = asdict(Parameters(onchip_mode=mode))
    assert {k:v for k,v in before.items() if k != "credits"} == {k:v for k,v in asdict(p).items() if k != "credits"}
    sequences, warms, states = [], [], []
    for _ in range(2):
        pred = NominalPredictor() if pname == "nominal" else Predictor("ours")
        warm = [simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["development"]]
        warms.append(digest(warm))
        sequences.append([simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["heldout"]])
        states.append(repr({**vars(pred), "rng": pred.rng.getstate()}))
    assert canonical(sequences[0]) == canonical(sequences[1]), (spec, "full result repeat")
    assert warms[0] == warms[1] and states[0] == states[1], (spec, "warmup/state repeat")
    rows, tasks = [], []
    for w, a, b in zip(WS["heldout"], *sequences):
        common = {"onchip_mode": mode, "design": name, "predictor": pname,
                  "window_id": w["id"], "batch": w["batch"]}
        assert a["hbm_bytes"]/a["cycles"] <= p.hbm_bandwidth+1e-7
        rows.append({**common, "credits": p.credits, "hbm_service_cap_GB_s": p.hbm_bandwidth,
            "cycles": a["cycles"], "latency_ms": a["latency_ms"],
            "hbm_bytes": a["hbm_bytes"], "achieved_hbm_GB_s": a["hbm_bytes"]/a["cycles"],
            "w_sram_bytes": a["w_sram_bytes"], "x_sram_bytes": a["x_sram_bytes"],
            "acc_sram_bytes": a["acc_sram_bytes"], "issued_macs": a["issued_macs"],
            "core_compute_service_cycles": json.dumps(a["core_compute_busy"]),
            "result_digest": digest(a), "repeat_digest": digest(b)})
        bindings = {(x.get("chunk_index", 0), x["expert_index"]): x for x in a["bindings"]}
        assert len(bindings) == len(a["tasks"])
        for t in a["tasks"]:
            bind = bindings[t.get("chunk_index", 0), t["expert_index"]]
            original_index = t.get("original_expert_index", t["expert_index"])
            e = w["experts"][original_index]
            assert bind["core"] == t["core"] and t["actual_cycles"] > 0
            tasks.append({**common, "expert_index": t["expert_index"],
                "chunk_index": t.get("chunk_index", 0), "original_expert_index": original_index,
                "expert_id": e.get("id", original_index), "core": t["core"],
                "is_shared": bool(e.get("is_shared", False)), "Me": e["Me"],
                "predicted_cycles": t["predicted_cycles"], "actual_cycles": t["actual_cycles"],
                "bind_cycle": bind["bind_cycle"], "actual_start_cycle": t["start"],
                "predicted_finish_at_bind": bind["predicted_finish"], "actual_finish_cycle": t["finish"],
                "first_weight_ready": t.get("first_weight_ready"), "current_end": t.get("current_end")})
    receipt = {"design": name, "onchip_mode": mode, "predictor": pname,
        "hardware": new_hw, "parameters": asdict(p), "dispatch_parameters": SELECTED,
        "development_windows_per_repeat": len(WS["development"]), "heldout_windows_per_repeat": len(rows),
        "repeats": 2, "warmup_digest": warms[0], "repeat_warmup_digest": warms[1],
        "heldout_digest": digest(sequences[0]), "repeat_heldout_digest": digest(sequences[1]),
        "final_predictor_state_digest": digest(states[0]), "repeat_final_predictor_state_digest": digest(states[1]),
        "full_result_repeats_exact": True, "compute_private_storage_frozen": True,
        "controller_area_eligibility": "unassessed"}
    return rows, tasks, receipt


def summarize(rows):
    oldrows = []
    for path in (OLD/"per_window.csv", OLD/"nominal_control/per_window.csv"):
        with path.open() as f:
            oldrows += [r for r in csv.DictReader(f) if r["predictor"] in PREDICTORS]
    old = {(r["onchip_mode"], r["design"], r["predictor"], r["window_id"]): r for r in oldrows}
    current = {(r["onchip_mode"], r["design"], r["predictor"], r["window_id"]): r for r in rows}
    assert len(old) == len(current) == 1620 and set(old) == set(current)
    summaries = []
    for mode in MODES:
        for name in NAMES:
            for pname in PREDICTORS:
                for batch in (*BATCHES, "all"):
                    rs = [r for r in rows if (r["onchip_mode"], r["design"], r["predictor"]) == (mode,name,pname)
                          and (batch == "all" or r["batch"] == batch)]
                    refs = [old[mode,name,pname,r["window_id"]] for r in rs]
                    ratios = [r["latency_ms"]/float(ref["latency_ms"]) for r,ref in zip(rs,refs)]
                    by_design = {n:gm(r["latency_ms"]/current[mode,n,pname,r["window_id"]]["latency_ms"] for r in rs)
                                 for n in ("B1", "B2")}
                    summaries.append({"onchip_mode":mode,"design":name,"predictor":pname,"batch":batch,
                        "windows":len(rs),"old126_geomean_ms":gm(float(r["latency_ms"]) for r in refs),
                        "new256_geomean_ms":gm(r["latency_ms"] for r in rs),"ratio_new_vs_old":gm(ratios),
                        "old126_hbm_sum_GiB":sum(int(r["hbm_bytes"]) for r in refs)/2**30,
                        "new256_hbm_sum_GiB":sum(r["hbm_bytes"] for r in rs)/2**30,
                        "achieved_hbm_geomean_GB_s":gm(r["achieved_hbm_GB_s"] for r in rs),
                        "ratio_vs_B1_same_predictor":by_design["B1"],"ratio_vs_B2_same_predictor":by_design["B2"]})
    save_csv("compare.csv",summaries)
    text = ["# 共享 HBM 256 GB/s 上限敏感性：冻结硬件，不重新 DSE", "",
        "只把全部组织共享的在途额度从 256 增至 520；标称仍为 256 B/ns，响应 64 ns，32 B 请求，落地计 1 ns。",
        "因此服务上限 min(256,520×32/65)=256 GB/s；旧上限为 126.030769 GB/s。实际每窗口带宽单独输出，不能称为原生 HBM 实测。",
        "所有计算几何、私有 SRAM、bank、WS 数据流、容量派工和预测器保持冻结；新带宽下未重新搜索硬件，不宣称最优设计。",
        "135 留出窗口、18 开发暖机、每个点两次从相同初始状态执行。延迟单位 ms，几何平均按同一窗口配对。",
        "完整控制器面积未评估；相比旧点增加 264 个在途标签。按已有 Rust 保守账本，可另计 8448 B 返回缓冲和 528 B 标签；这些不是本次私有 SRAM 的已实现新增。不能直接声称与旧点等面积。", ""]
    text += ["## 汇总", "", table(["模式","设计","预测器","旧126 ms","新256 ms","新/旧","新异构/单核","新异构/同构"],
        [[r["onchip_mode"],r["design"],r["predictor"],f'{r["old126_geomean_ms"]:.4f}',f'{r["new256_geomean_ms"]:.4f}',
          f'{r["ratio_new_vs_old"]:.4f}',f'{r["ratio_vs_B1_same_predictor"]:.4f}',f'{r["ratio_vs_B2_same_predictor"]:.4f}']
         for r in summaries if r["batch"]=="all"]), "", "两列跨架构比值的分子都是该行设计；只有 best_hetero 行才是异构/基线。", ""]
    for mode in MODES:
        text += [f"## {mode} 分 batch", ""]
        for pname in PREDICTORS:
            selected = [r for r in summaries if r["onchip_mode"]==mode and r["predictor"]==pname]
            lookup = {(r["design"],r["batch"]):r for r in selected}
            text += [f"预测器 {pname}", "", table(["设计","服务上限",*[f"B{b}" for b in BATCHES],"全部GM ms"],
                [[name,label,*[f'{lookup[name,b][field]:.4f}' for b in (*BATCHES,"all")]]
                 for name in NAMES for label,field in (("126.03","old126_geomean_ms"),("256","new256_geomean_ms"))]), ""]
    (HERE/"TABLES_ZH.md").write_text("\n".join(text)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jobs",type=int,default=6)
    args=ap.parse_args()
    init_worker()
    start=time.monotonic(); started=datetime.now(timezone.utc).isoformat()
    paths=[ROOT/f for f in MANIFEST["source_hashes"]]
    paths += [PARENT/"runtime.py",PARENT/"selection.json",PARENT/"frozen_designs.json",
              OLD/"bounded_runtime.py",OLD/"nominal_control.py",HERE/"run.py",HERE/"verify.py",
              OLD/"per_window.csv",OLD/"nominal_control/per_window.csv"]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert all(before[k]==v for k,v in MANIFEST["source_hashes"].items())
    specs=[(m,n,p) for m in MODES for n in NAMES for p in PREDICTORS]
    rows=[]; tasks=[]; receipts=[]
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=init_worker) as pool:
        for spec,(rs,ts,receipt) in zip(specs,pool.map(job,specs)):
            rows.extend(rs); tasks.extend(ts); receipts.append(receipt)
            print({"done":spec,"windows":len(rs),"tasks":len(ts),"geomean_ms":gm(r["latency_ms"] for r in rs)},flush=True)
    assert len(rows)==1620 and all(r["result_digest"]==r["repeat_digest"] for r in rows)
    save_csv("per_window.csv",rows)
    raw=csv_text(tasks).encode()
    with (HERE/"task_predictions.csv.gz").open("wb") as f:
        with gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0) as compressed:
            compressed.write(raw)
    write_json(HERE/"repeat_checks.json",receipts)
    summarize(rows)
    unchanged={name:sha(ROOT/name)==value for name,value in before.items()}
    assert all(unchanged.values())
    metadata={"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":time.monotonic()-start,"command":sys.argv,
        "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT.parents[2],text=True).strip(),
        "source_sha256":before,"source_unchanged":unchanged,
        "input_hashes":json.loads((ROOT/"results/E0/frozen_inputs.json").read_text()),
        "configurations":len(specs),"development_windows":18,"heldout_windows":135,"repeats_per_configuration":2,
        "simulation_calls":len(specs)*2*(18+135),"per_window_rows":len(rows),"task_prediction_rows":len(tasks),
        "task_records_raw_sha256":__import__("hashlib").sha256(raw).hexdigest(),
        "task_records_compressed_sha256":sha(HERE/"task_predictions.csv.gz"),
        "dispatch_parameters":SELECTED,"bounded_runtime_patch":patch_metadata(),
        "hbm":{"nominal_B_per_ns":256,"service_ceiling_B_per_ns":256,"credits":520,"request_bytes":32,
               "response_ns":64,"landing_ns":1,"old_credits":256,"old_service_ceiling_B_per_ns":126.03076923076924},
        "clock_ns":1,"precision":"BF16","scope":"post-router phase-fluid analytical estimate; not native HBM/RTL/full-model timing",
        "compute_private_storage_frozen":True,"geometry_searched_at_new_bandwidth":False,
        "controller_area_eligibility":"unassessed","additional_outstanding_tag_states":264,
        "optional_conservative_Rust_ledger":{"additional_return_payload_bytes":8448,"additional_tag_bytes":528,
          "note":"conservative separate-buffer accounting, not inevitable payload capacity if return paths/destination reservation redesigned"}}
    write_json(HERE/"METADATA.json",metadata)
    print({"completed":len(specs),"window_rows":len(rows),"tasks":len(tasks),"elapsed_seconds":metadata["elapsed_seconds"]},flush=True)


if __name__=="__main__":
    main()
