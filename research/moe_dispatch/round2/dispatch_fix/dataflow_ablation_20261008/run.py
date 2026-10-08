"""OS/WS-only ablation on frozen E4 hardware with the repaired dispatcher.

No hardware search, predictor retuning, or frozen-result mutation. Every point
is replayed twice from fresh predictor state, including its development warmup.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from ...common import ROOT, BATCHES, canonical, decode_design, encode_design, inputs, sha, useful_macs, write_json
from ...model import Parameters
from ...predictors import Predictor
from ..runtime import simulate

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
NAMES = ("B1", "B2", "best_hetero")
MODES = ("pipelined", "port_tight")
WS = None
MANIFEST = None
SELECTED = None


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def gm(values):
    values = list(values)
    assert values and all(v >= 0 and math.isfinite(v) for v in values)
    return 0.0 if any(v == 0 for v in values) else math.exp(math.fsum(math.log(v) for v in values)/len(values))


def csv_out(name, rows):
    with (HERE/name).open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def init_worker():
    global WS, MANIFEST, SELECTED
    WS = inputs()
    MANIFEST = json.loads((PARENT/"frozen_designs.json").read_text())
    SELECTED = json.loads((PARENT/"selection.json").read_text())["chosen"]
    assert SELECTED == {"t_big": 3, "large_first": True}


def job(spec):
    mode, name, flows = spec
    original = decode_design(MANIFEST["modes"][mode][name])
    d = replace(original, flows=flows)
    # Only loop/dataflow changes: every hardware resource remains identical.
    old_hw = encode_design(original)
    new_hw = encode_design(d)
    assert {k:v for k,v in old_hw.items() if k != "flows"} == {k:v for k,v in new_hw.items() if k != "flows"}
    p = Parameters(onchip_mode=mode)
    sequences, warm_digests, final_states = [], [], []
    for _ in range(2):
        pred = Predictor("ours") if len(d.cores) > 1 else None
        warm = [simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["development"]]
        warm_digests.append(digest(warm))
        sequences.append([simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["heldout"]])
        # Predictor serialization excludes the RNG object and includes its state.
        final_states.append(None if pred is None else repr({**vars(pred), "rng": pred.rng.getstate()}))
    assert canonical(sequences[0]) == canonical(sequences[1]), (spec, "heldout repeat mismatch")
    assert warm_digests[0] == warm_digests[1] and final_states[0] == final_states[1]
    rows = []
    for w, a, b in zip(WS["heldout"], *sequences):
        assert a["hbm_bytes"] / a["cycles"] <= p.hbm_bandwidth + 1e-7
        rows.append({"window_id": w["id"], "batch": w["batch"], "design": name,
            "onchip_mode": mode, "flows_core_order": "/".join(flows),
            "cycles": a["cycles"], "latency_ms": a["latency_ms"],
            "hbm_bytes": a["hbm_bytes"], "w_sram_bytes": a["w_sram_bytes"],
            "x_sram_bytes": a["x_sram_bytes"], "acc_sram_bytes": a["acc_sram_bytes"],
            "useful_macs": useful_macs(w), "issued_macs": a["issued_macs"],
            "core_compute_service_cycles": a["core_compute_busy"],
            "result_digest": digest(a), "repeat_digest": digest(b)})
    receipt = {"design": name, "onchip_mode": mode, "flows_core_order": "/".join(flows),
        "hardware": new_hw, "parameters": asdict(p),
        "predictor": "ours" if len(d.cores) > 1 else "none_single_compatibility",
        "development_windows_per_repeat": len(WS["development"]),
        "heldout_windows_per_repeat": len(rows), "repeats": 2,
        "warmup_digest": warm_digests[0], "repeat_warmup_digest": warm_digests[1],
        "heldout_digest": digest(sequences[0]), "repeat_heldout_digest": digest(sequences[1]),
        "final_predictor_state_digest": digest(final_states[0]),
        "repeat_final_predictor_state_digest": digest(final_states[1]),
        "only_flows_changed": True, "full_result_repeats_exact": True}
    return rows, receipt


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    init_worker()
    start_time = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    source_paths = [ROOT/f for f in MANIFEST["source_hashes"]]
    source_paths += [PARENT/"runtime.py", PARENT/"run.py", PARENT/"selection.json",
                     PARENT/"frozen_designs.json", Path(__file__)]
    before = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    assert all(before[k] == v for k,v in MANIFEST["source_hashes"].items())
    specs = [(mode, name, fs) for mode in MODES for name in NAMES
             for fs in itertools.product(("OS", "WS"), repeat=len(MANIFEST["modes"][mode][name]["cores"]))]
    assert len(specs) == 20
    rows, receipts = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker) as pool:
        for spec, (rs, receipt) in zip(specs, pool.map(job, specs)):
            rows.extend(rs)
            receipts.append(receipt)
            print({"done": list(spec[:2])+[list(spec[2])], "windows": len(rs),
                   "geomean_ms": gm(r["latency_ms"] for r in rs)}, flush=True)
    assert len(rows) == 2700 and all(r["result_digest"] == r["repeat_digest"] for r in rows)
    # Frozen-flow configurations must reproduce the earlier repaired campaign.
    with gzip.open(PARENT/"per_window.json.gz", "rt") as f:
        baseline = {(r["onchip_mode"], r["design"], r["window_id"]): r
                    for r in json.load(f) if r["dispatch"] == "fixed" and r["design"] in NAMES}
    checks = []
    for r in rows:
        original = MANIFEST["modes"][r["onchip_mode"]][r["design"]]
        if r["flows_core_order"] != "/".join(original["flows"]):
            continue
        ref = baseline[r["onchip_mode"], r["design"], r["window_id"]]
        assert r["result_digest"] == ref["result_digest"], (r["window_id"], "existing fixed baseline changed")
        checks.append({**{k:r[k] for k in ("window_id", "design", "onchip_mode", "flows_core_order")},
                       "full_result_digest_exact": True})
    assert len(checks) == 810
    summaries = []
    for receipt in receipts:
        key = tuple(receipt[k] for k in ("onchip_mode", "design", "flows_core_order"))
        family = [r for r in rows if tuple(r[k] for k in ("onchip_mode", "design", "flows_core_order")) == key]
        for batch in (*BATCHES, "all"):
            rs = [r for r in family if batch == "all" or r["batch"] == batch]
            summaries.append({"onchip_mode": key[0], "design": key[1], "flows_core_order": key[2],
                "batch": batch, "windows": len(rs), "latency_geomean_ms": gm(r["latency_ms"] for r in rs),
                "HBM_geomean_MiB": gm(r["hbm_bytes"]/2**20 for r in rs),
                "W_SRAM_geomean_MiB": gm(r["w_sram_bytes"]/2**20 for r in rs),
                "X_SRAM_geomean_MiB": gm(r["x_sram_bytes"]/2**20 for r in rs),
                "acc_SRAM_geomean_MiB": gm(r["acc_sram_bytes"]/2**20 for r in rs),
                "HBM_sum_GiB": sum(r["hbm_bytes"] for r in rs)/2**30,
                "W_SRAM_sum_GiB": sum(r["w_sram_bytes"] for r in rs)/2**30,
                "acc_SRAM_sum_GiB": sum(r["acc_sram_bytes"] for r in rs)/2**30})
    csv_out("per_window.csv", [{**r, "core_compute_service_cycles": json.dumps(r["core_compute_service_cycles"])} for r in rows])
    csv_out("compare.csv", summaries)
    csv_out("baseline_reproduction.csv", checks)
    write_json(HERE/"repeat_checks.json", receipts)
    out = ["# 冻结 DSE 硬件：新版 fixed 派工的 OS/WS 完整对照", "",
        "本次是补充数据流消融，不重选硬件，不修改第二轮冻结结果。仅替换每个核的 flows；",
        "SRAM、bank、向量通道、共享HBM和派工参数均保持原样。",
        "所有点使用相同 T_big=3、large_first=True；双核沿用 ours，18开发窗口预热后按固定顺序评估135留出窗口。",
        "每点从新预测器状态完整重复两次；20点、6120次窗口仿真。810个冻结原数据流结果逐项digest复现。",
        "注意：这是路由后的 BF16 MoE phase-fluid analytical 模型，非 Rust/native HBM/RTL/full-model 实测。",
        "1 cycle=1 ns；表格延迟按窗口取几何平均，单位ms。", "",
        "## 数据流定义与范围", "",
        "OS：输出部分和跨K驻留RF，遍历下一个M组时重读权重。",
        "WS：权重跨M组驻留；输出部分和跨K在私有acc SRAM读改写。",
        "这些是冻结模型支持的循环和流量模板；本次未新增双数据流RTL或证明物理面积相同。",
        "结果包括既有的存储溢出/重读、有限预取、片上端口以及共享HBM竞争。",
        "不能把此表作为纯compute表，不能用各资源服务时间相加得到总延迟。", "",
        "## 硬件与核顺序", "",
        table(("模式", "设计", "M×N×K：核0；核1", "冻结原数据流"),
          [[m, n, "; ".join(f"{c['pm']}×{c['pn']}×{c['pk']}" for c in MANIFEST['modes'][m][n]['cores']),
            "/".join(MANIFEST['modes'][m][n]['flows'])] for m in MODES for n in NAMES]), "",
        "best_hetero为核0小、核1大；OS/WS表示小核OS、大核WS。B2两核尺寸相同，但私有资源分配仍按E4冻结。",
        "因此WS/OS与OS/WS并非可交换。这里不是全同构设计族的全局最优搜索。", ""]
    for mode in MODES:
        out += ["## " + mode + "：全部 batch 总延迟", "",
            table(("设计", "核0/核1数据流", *["B"+str(b) for b in BATCHES], "全部135窗口GM"),
              [[n, "/".join(fs), *[f"{next(r['latency_geomean_ms'] for r in summaries if r['onchip_mode']==mode and r['design']==n and r['flows_core_order']=='/'.join(fs) and r['batch']==b):.4f}" for b in (*BATCHES, 'all')]]
               for m,n,fs in specs if m==mode]), ""]
    out += ["## 总访存量（135窗口求和，GiB，不是单窗口平均）", "",
        table(("模式", "设计", "数据流", "HBM GiB", "W SRAM GiB", "acc SRAM GiB"),
          [[r['onchip_mode'], r['design'], r['flows_core_order'], f"{r['HBM_sum_GiB']:.3f}",
            f"{r['W_SRAM_sum_GiB']:.3f}", f"{r['acc_SRAM_sum_GiB']:.3f}"] for r in summaries if r['batch']=='all']), "",
        "旧E2网格是旧EFT、无预测器，不能与新版fixed混用；新版补齐了旧网格遗漏的B2。",
        "本次没有在留出集上改 T_big、容量或硬件，也不把留出集最低数据流重新称为DSE冻结设计。"]
    (HERE/"TABLES_ZH.md").write_text("\n".join(out)+"\n")
    unchanged = {name: sha(ROOT/name) == value for name,value in before.items()}
    assert all(unchanged.values())
    metadata = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic()-start_time,
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[2], text=True).strip(),
        "command": sys.argv, "source_sha256": before, "source_unchanged": unchanged,
        "input_hashes": json.loads((ROOT/"results/E0/frozen_inputs.json").read_text()),
        "dispatch_parameters": SELECTED, "configurations": len(specs), "development_windows": 18,
        "heldout_windows": 135, "repeats_per_configuration": 2, "simulation_calls": 6120,
        "per_window_rows": len(rows), "summary_rows": len(summaries), "existing_fixed_exact_checks": len(checks),
        "scope": "post-router BF16 phase-fluid analytical model; no native HBM/RTL/full-model timing",
        "protocol": "only flows changed; 18 dev warmup then all heldout in frozen order; fresh state each repeat; no tuning"}
    write_json(HERE/"METADATA.json", metadata)
    csv_out("PROVENANCE.csv", [{"file": str(f.relative_to(HERE)), "sha256": sha(f)} for f in sorted(HERE.iterdir())
                              if f.is_file() and f.name != "PROVENANCE.csv"])
    print({"completed": len(specs), "rows": len(rows), "elapsed_seconds": metadata['elapsed_seconds']}, flush=True)


if __name__ == "__main__":
    main()
