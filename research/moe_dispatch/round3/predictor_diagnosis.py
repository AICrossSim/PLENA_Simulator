"""对已完成预测器结果做归属、队序与服务积分诊断；不新增模拟。"""
from __future__ import annotations

import json
from pathlib import Path

from research.moe_dispatch.round2.common import inputs, sha, write_csv, write_json, gmean
from research.moe_dispatch.round3.evaluations import HERE, load_gzip, digest


def task_key(task):
    return task.get("chunk_index", 0), task.get("original_expert_index", task["expert_index"])


def paired_diagnosis(a, b):
    amap = {task_key(t): t for t in a["tasks"]}
    bmap = {task_key(t): t for t in b["tasks"]}
    assert set(amap) == set(bmap)
    changed = sum(amap[k]["core"] != bmap[k]["core"] for k in amap)
    changed_order = False
    for c in range(len(a["core_finish_cycles"])):
        qa = [task_key(t) for t in sorted(a["tasks"], key=lambda t: (t["start"], task_key(t))) if t["core"] == c]
        qb = [task_key(t) for t in sorted(b["tasks"], key=lambda t: (t["start"], task_key(t))) if t["core"] == c]
        changed_order |= qa != qb
    return {"owner_changed_tasks": changed, "queue_or_owner_changed": changed_order,
            "tasks": len(amap), "HBM_delta_MiB": (b["hbm_bytes"]-a["hbm_bytes"])/2**20,
            "W_SRAM_delta_MiB": (b["w_sram_bytes"]-a["w_sram_bytes"])/2**20,
            "X_SRAM_delta_MiB": (b["x_sram_bytes"]-a["x_sram_bytes"])/2**20,
            "finish_gap_delta_ms": (b["core_finish_gap"]-a["core_finish_gap"])/1e6,
            **{f"core{c}_compute_service_delta_ms": (b["core_compute_busy"][c]-a["core_compute_busy"][c])/1e6
               for c in range(len(a["core_finish_cycles"]))}}


def main():
    out = HERE / "E5/predictor"
    receipts = json.loads((out / "repeat_checks.json").read_text())
    ids = sorted({(r["bw_GBps"], r["onchip_mode"], r["design"]) for r in receipts})
    records = {(r["bw_GBps"], r["onchip_mode"], r["design"], r["method"]): r for r in receipts}
    summaries, windows, hashes = [], [], {}
    for bw, mode, design in ids:
        results = {}
        for method in ("nominal", "ours"):
            rec = records[bw, mode, design, method]
            path = HERE / rec["raw_file"]
            assert sha(path) == rec["raw_sha256"]
            hashes[str(path.relative_to(HERE))] = sha(path)
            results[method] = load_gzip(path)
        rows = []
        for w, a, b in zip(inputs()["heldout"], results["nominal"], results["ours"]):
            rows.append({"bw_GBps": bw, "onchip_mode": mode, "design": design,
                         "window_id": w["id"], "batch": w["batch"], "nominal_ms": a["latency_ms"],
                         "ours_ms": b["latency_ms"], "ours_nominal_ratio": b["cycles"]/a["cycles"],
                         **paired_diagnosis(a, b)})
        windows.extend(rows)
        summaries.append({"bw_GBps": bw, "onchip_mode": mode, "design": design,
            "ours_nominal_ratio": gmean(r["ours_nominal_ratio"] for r in rows),
            "owner_changed_windows": sum(r["owner_changed_tasks"] > 0 for r in rows),
            "owner_changed_tasks": sum(r["owner_changed_tasks"] for r in rows),
            "queue_or_owner_changed_windows": sum(r["queue_or_owner_changed"] for r in rows),
            "HBM_delta_MiB_total": sum(r["HBM_delta_MiB"] for r in rows),
            "W_SRAM_delta_MiB_total": sum(r["W_SRAM_delta_MiB"] for r in rows),
            "X_SRAM_delta_MiB_total": sum(r["X_SRAM_delta_MiB"] for r in rows),
            "finish_gap_delta_ms_mean": sum(r["finish_gap_delta_ms"] for r in rows)/len(rows),
            **{f"core{c}_compute_service_delta_ms_mean": sum(r.get(f"core{c}_compute_service_delta_ms", 0) for r in rows)/len(rows) for c in range(2)}})
    write_csv(out / "diagnosis_by_window.csv", windows)
    write_csv(out / "diagnosis_summary.csv", summaries)
    md = ["# 预测器变慢的运行记录诊断\n",
          "只从已保存的nominal/ours原始结果派生，不新增仿真，也不修改主表。135窗口；延迟比取配对几何平均，"
          "服务量差为总计或算术平均（列名标明），均为phase-fluid模型计数。\n",
          "| GB/s | 模式 | 设计 | ours/nominal | 归属变化窗口/任务 | 队序或归属变化窗口 | ΔHBM总MiB | ΔW总MiB | ΔX总MiB | Δ核完成差均值ms |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        md.append(f"| {r['bw_GBps']:.5f} | {r['onchip_mode']} | {r['design']} | {r['ours_nominal_ratio']:.6f} | "
                  f"{r['owner_changed_windows']}/{r['owner_changed_tasks']} | {r['queue_or_owner_changed_windows']} | "
                  f"{r['HBM_delta_MiB_total']:.3f} | {r['W_SRAM_delta_MiB_total']:.3f} | {r['X_SRAM_delta_MiB_total']:.3f} | {r['finish_gap_delta_ms_mean']:.6f} |")
    md += ["\n预测器在本模型里只影响绑定、队序/归属与预取时刻。固定硬件和输入下，这些行动变化会改变各私有端口的负载和核结束顺序，"
           "即使总HBM字节相同也会改变整层时间。低MAE只说明实际时长更容易预测，不说明这些行动更适合全层makespan。\n",
           "126旧异构和256/port_tight H42的额外HBM与重读任务提供了容量/归属变化的证据。"
           "256/pipelined H51的HBM与重读不变，因而不能解释为额外HBM流量；要结合这里的归属、队序、W/X服务量及核尾部变化。"
           "Next等待占比的微小变化也不能独自解释其约0.9%延迟差。\n",
           "以上是观测到的控制行动与服务变化，不能把重叠服务积分相加得到因果百分比。"
           "精确区分绑定、归属、队序和预取各自贡献，还需分别固定这些行动做受控反事实回放；本轮没有这样的因果分离结果。\n"]
    (out / "DIAGNOSIS.md").write_text("\n".join(md)+"\n")
    write_json(out / "DIAGNOSIS_DERIVATION.json", {"no_simulator_rerun": True, "raw_sha256": hashes,
        "source_sha256": sha(Path(__file__)), "paired_windows": len(windows),
        "summary_digest": digest(summaries)})
    current = (out / "PREDICTOR.md").read_text()
    note = "\n控制行动与端口负载的逐窗口诊断见[DIAGNOSIS.md](DIAGNOSIS.md)。它区分额外流量和不增加流量的归属/队序变化，但不声称已完成互斥因果分解。\n"
    if note not in current:
        (out / "PREDICTOR.md").write_text(current+note)


if __name__ == "__main__":
    main()
