"""Read-only derivation of DSE-after tables from the completed frozen campaign.

No architecture search, dispatch decisions, cost changes or simulation reruns.
All new artifacts stay in this subdirectory; parent evidence stays byte exact.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from ...common import inputs, useful_macs, native_bytes
from ...model import Parameters


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
BATCHES = (2, 4, 8, 16, 64, 96, 128)
NAMES = ("B0", "B1", "B2", "best_hetero", "fixed_4+2")
MODES = ("pipelined", "port_tight")
DISPATCHES = ("eft_old", "fixed", "milp", "eft_ours_control")
SELECTED = ("B1", "B2", "best_hetero")
LABELS = {"B1": "A_DSE_single", "B2": "B_DSE_uniform", "best_hetero": "C_DSE_heterogeneous"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gm(values):
    values = list(values)
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    return 0.0 if any(v == 0 for v in values) else math.exp(math.fsum(math.log(v) for v in values) / len(values))


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    raw = EVIDENCE / "per_window.json.gz"
    with gzip.open(raw, "rt") as f:
        records = json.load(f)
    manifest = json.loads((EVIDENCE / "frozen_designs.json").read_text())
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    windows = inputs()["heldout"]
    by_id = {w["id"]: w for w in windows}
    assert len(windows) == 135 and len(records) == 5400
    capacity = Parameters().hbm_bandwidth
    workload_rows = []
    for batch in BATCHES:
        ws = [w for w in windows if w["batch"] == batch]
        routed = [[e for e in w["experts"] if not e.get("is_shared", False)] for w in ws]
        workload_rows.append({"batch": batch, "windows": len(ws),
            "mean_routed_M_gt2": mean(sum(e["Me"] > 2 for e in es) for es in routed),
            "mean_routed_M_1_or_2": mean(sum(0 < e["Me"] <= 2 for e in es) for es in routed),
            "mean_shared": mean(sum(bool(e.get("is_shared", False)) for e in w["experts"]) for w in ws),
            "shared_M": batch,
            "mean_unique_HBM_floor_ms": mean(native_bytes(w)/capacity/1e6 for w in ws),
            "geomean_unique_HBM_floor_ms": gm(native_bytes(w)/capacity/1e6 for w in ws),
            "geomean_useful_GFLOPs": gm(2*useful_macs(w)/1e9 for w in ws)})
    hardware = []
    for mode in MODES:
        for name in NAMES:
            d = manifest["modes"][mode][name]
            hardware.append({"onchip_mode": mode, "design": name,
                "label": LABELS.get(name, "reference"),
                "M_N_K": "+".join(f"{c['pm']}x{c['pn']}x{c['pk']}" for c in d["cores"]),
                "flows": "/".join(d["flows"]), "total_multipliers": d["total_macs"],
                "private_W_KiB": json.dumps([x/1024 for x in d["w_bytes"]]),
                "private_X_KiB": json.dumps([x/1024 for x in d["x_bytes"]]),
                "private_acc_KiB": json.dumps([x/1024 for x in d["acc_bytes"]]),
                "private_Z_KiB": json.dumps([x/1024 for x in d["z_bytes"]]),
                "HBM_cap_GB_s": capacity})
    per_window = []
    for r in records:
        w = by_id[r["window_id"]]
        cycles = r["cycles"]
        busy = r["core_compute_busy"]
        row = {k: r[k] for k in ("window_id", "batch", "design", "onchip_mode", "dispatch")}
        row.update({"latency_ms": cycles/1e6, "hbm_MiB": r["hbm_bytes"]/2**20,
            "unique_hbm_MiB": native_bytes(w)/2**20,
            "useful_GFLOPs": 2*useful_macs(w)/1e9,
            "equivalent_HBM_GB_s": r["hbm_bytes"]/cycles,
            "HBM_transfer_lower_bound_ms": r["hbm_bytes"]/capacity/1e6,
            "unique_HBM_transfer_lower_bound_ms": native_bytes(w)/capacity/1e6,
            "useful_GFLOP_s_end_to_end": 2*useful_macs(w)/cycles,
            "core0_compute_service_ms": busy[0]/1e6,
            "core1_compute_service_ms": busy[1]/1e6 if len(busy)>1 else None,
            "hbm_bandwidth_occupancy_pct": 100*r["hbm_busy_frac"]})
        assert abs(row["latency_ms"]-r["latency_ms"]) < 1e-10
        assert row["equivalent_HBM_GB_s"] <= capacity+1e-7
        assert abs(row["hbm_bandwidth_occupancy_pct"]-100*row["equivalent_HBM_GB_s"]/capacity) < 1e-5
        assert row["HBM_transfer_lower_bound_ms"] <= row["latency_ms"]+1e-7
        assert row["HBM_transfer_lower_bound_ms"]+1e-7 >= row["unique_HBM_transfer_lower_bound_ms"]
        assert all(0 <= value <= row["latency_ms"]+1e-7 for value in
                   (row["core0_compute_service_ms"], row["core1_compute_service_ms"] or 0))
        per_window.append(row)
    by_batch = []
    numeric = list(per_window[0])[5:]
    for mode in MODES:
        for name in NAMES:
            for dispatch in DISPATCHES:
                family = [r for r in per_window if (r["onchip_mode"],r["design"],r["dispatch"]) == (mode,name,dispatch)]
                assert len(family) == 135
                for batch in (*BATCHES,"all"):
                    rows = [r for r in family if batch == "all" or r["batch"] == batch]
                    item = {"batch": batch, "design": name, "onchip_mode": mode,
                            "dispatch": dispatch, "windows": len(rows), "aggregation": "geometric_mean_per_window"}
                    for key in numeric:
                        vals = [r[key] for r in rows]
                        item[key] = None if all(x is None for x in vals) else gm(vals)
                    # Same-window GM preserves these identities exactly.
                    assert abs(item["equivalent_HBM_GB_s"] - item["hbm_MiB"]*2**20/(item["latency_ms"]*1e6)) < 1e-7
                    assert abs(item["useful_GFLOP_s_end_to_end"] - item["useful_GFLOPs"]*1000/item["latency_ms"]) < 1e-7
                    by_batch.append(item)
    primary = [r for r in by_batch if r["onchip_mode"] == "pipelined" and r["dispatch"] == "fixed" and r["design"] in SELECTED and r["batch"] != "all"]
    primary.sort(key=lambda r:(BATCHES.index(r["batch"]),SELECTED.index(r["design"])))
    lookup = {(r["batch"],r["design"],r["onchip_mode"],r["dispatch"]):r for r in by_batch}
    with (EVIDENCE / "compare.csv").open() as f:
        old_summary = list(csv.DictReader(f))
    for old in old_summary:
        for batch in (*BATCHES,"all"):
            new = lookup[batch,old["design"],old["onchip_mode"],old["dispatch"]]
            old_value = old["all_geomean_ms"] if batch == "all" else old["B"+str(batch)]
            assert abs(new["latency_ms"]-float(old_value)) < 1e-9
    write_csv("hardware.csv",hardware)
    write_csv("workload.csv",workload_rows)
    write_csv("metrics_per_window.csv",per_window)
    write_csv("metrics_by_batch.csv",by_batch)
    write_csv("main_table.csv",primary)
    text = ["# DSE 后：冻结硬件与当前在线派工的完整指标表", "",
        "本文从已完成并重复验证的 dispatch_fix 记录派生，没有重跑仿真、改成本模型或重选硬件。",
        "主表：pipelined + fixed；BF16，1 cycle=1 ns；主三设计A=B1、B=B2、C=best_hetero。",
        "所有延迟、带宽、吞吐、服务量按同一组窗口取几何平均；专家个数取算术平均。",
        "场景：捕获的 DeepSeek-V2-Lite Top-6 路由，H=2048，routed F=1408，Shared F=2816；",
        "计时覆盖路由后 Gate/Up、SiLU、Down 和 combine，不包含路由或整个模型，非原生 HBM/RTL 实测。",
        "异构/同构全族精确最优证明未闭合；这里是开发集选定、留出集冻结的已评估候选。", "",
        "## 指标定义", "",
        "- 等效HBM带宽 = 实际HBM字节 / 整层总时间；当前上限126.030769 GB/s。",
        "- HBM传输下限 = 实际HBM字节 / 当前共享供数上限，不是暴露的等待时间或包含返回延迟的取数历时。",
        "- 有效计算吞吐 = 2×有效MAC / 整层总时间，不是峰值算力。",
        "- 核0/核1计算服务量 = core_compute_busy / 1e6，是模型的计算服务周期需求，含流水线/K依赖。",
        "  它不是核忙碌的墙钟区间，也不是关闭HBM后的整层实测延迟。不能以max或sum冒充纯计算整层时间。",
        "- HBM、计算和片上端口会重叠，不能把上述列相加成总延迟。", "",
        "## 冻结硬件（M×N×K）", "",
        table(("标签","设计","物理尺寸","数据流","总乘法器"),
              [[r["label"],r["design"],r["M_N_K"],r["flows"],r["total_multipliers"]] for r in hardware if r["onchip_mode"]=="pipelined"]),
        "", "port_tight 的B2按原E4采用3×16×128两核、WS/WS，其余主形状不变。", "",
        "B2的同构指计算阵列尺寸相同；私有资源和数据流仍按E4的DSE结果冻结，pipelined为OS/WS。", "",
        "## 路由分布", "",
        table(("Batch","窗口","Me>2 routed平均个数","Me=1/2 routed平均个数","Shared个数","唯一权重HBM下限GM ms"),
              [[r["batch"],r["windows"],fmt(r["mean_routed_M_gt2"],2),fmt(r["mean_routed_M_1_or_2"],2),1,fmt(r["geomean_unique_HBM_floor_ms"])] for r in workload_rows]),
        "", "低token专家仍然被激活，不能叫不活跃专家；Shared单独统计，Me=batch。",
        "截图1.7338/2.9185/3.7269/5.0489为算术平均下限；workload.csv同时保留算术和几何平均，不能跨口径相除。", "",
        "## 当前主表：全部Batch", "",
        table(("Batch","设计","HBM MiB","等效HBM GB/s","HBM传输下限ms","有效GFLOP/s","核0服务ms","核1服务ms","总延迟GM ms"),
              [["B"+str(r["batch"]),LABELS[r["design"]],fmt(r["hbm_MiB"],2),fmt(r["equivalent_HBM_GB_s"],2),fmt(r["HBM_transfer_lower_bound_ms"]),fmt(r["useful_GFLOP_s_end_to_end"],2),fmt(r["core0_compute_service_ms"]),fmt(r["core1_compute_service_ms"]),fmt(r["latency_ms"])] for r in primary]),
        "", "## DSE旧在线、修复在线、离线分别列出", "",
        table(("Batch","A旧/新/离线ms","B旧/新/离线ms","C旧/新/离线ms"),
              [["B"+str(b)]+[" / ".join(fmt(lookup[b,n,"pipelined",d]["latency_ms"]) for d in ("eft_old","fixed","milp")) for n in SELECTED] for b in BATCHES]),
        "", "milp表示离线分配加LPT回放参照，不能称全局最优执行时序。",
        "main_table.csv有21条主表；metrics_by_batch.csv覆盖5设计×2模式×4派工×8batch分组；",
        "metrics_per_window.csv保留所有5400条逐窗口结果；输入SHA与定义见METADATA.json。", ""]
    (HERE/"TABLES_ZH.md").write_text("\n".join(text))
    receipt = json.loads((EVIDENCE/"executions/campaign_20261008T045223Z.json").read_text())
    metadata = {"created_utc":datetime.now(timezone.utc).isoformat(),"source_evidence_commit":"9b57bac10ca4c99e854fb0a98a8b02c5cb595332",
        "numerical_execution_base_commit":receipt["execution_commit"],
        "numerical_execution_source_sha256":receipt["source_sha256"],
        "inputs":{name:sha(EVIDENCE/name) for name in ("per_window.json.gz","frozen_designs.json","selection.json","compare.csv")},
        "generator_sha256":sha(Path(__file__)),"selected_runtime":selection["chosen"],
        "scope":"post-router phase-fluid analytical estimates; no new simulations",
        "precision":"BF16", "hypothetical_cycle_ns":1,"MAC_flops":2,
        "HBM_cap_GB_s":capacity,"MiB_bytes":2**20,
        "aggregation":"performance=per-window geometric mean; expert counts=arithmetic mean",
        "row_counts":{"hardware.csv":len(hardware),"workload.csv":len(workload_rows),"metrics_per_window.csv":len(per_window),"metrics_by_batch.csv":len(by_batch),"main_table.csv":len(primary)},
        "checks":{"all_bandwidths_within_cap":True,"all_transfer_lower_bounds_valid":True,"all_GM_ratio_identities_valid":True,"all_main_latency_values_match_campaign":True},
        "compute_only_layer_latency":"not measured for new dispatch; service counters are not a substitute"}
    (HERE/"METADATA.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    write_csv("PROVENANCE.csv",[{"file":p.name,"bytes":p.stat().st_size,"sha256":sha(p)}
                                for p in sorted(HERE.iterdir()) if p.is_file() and p.name!="PROVENANCE.csv"])
    print(json.dumps(metadata["row_counts"],sort_keys=True))


if __name__ == "__main__":
    main()
