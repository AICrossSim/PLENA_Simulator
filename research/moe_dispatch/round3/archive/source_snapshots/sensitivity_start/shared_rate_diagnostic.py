"""只从E3已保存的segment派生Shared速率；不重新运行仿真。"""
from __future__ import annotations

import json
from pathlib import Path

from research.moe_dispatch.round2.common import inputs, sha, write_csv, write_json
from research.moe_dispatch.round3.evaluations import HERE, BATCHES, decode, load_gzip


def shared_service(result, workload, design):
    big = max(range(len(design.cores)), key=lambda c: (design.cores[c].macs, design.cores[c].pm, -c))
    shared_ids = {e.get("id", i) for i, e in enumerate(workload["experts"]) if e.get("is_shared", False)}
    time = total_bytes = own_bytes = 0.0
    for s in result["segments"]:
        if s["active_run"][big] and s["run_expert"][big] in shared_ids:
            dt = s["end"] - s["start"]
            time += dt
            total_bytes += dt * s["hbm_rate_Bpc"]
            own_bytes += dt * s["hbm_rate_Bpc_core"][big]
    return {"shared_big_core": big, "shared_interval_cycles": time,
            "global_hbm_service_bytes": total_bytes, "big_core_hbm_service_bytes": own_bytes,
            "global_hbm_GBps": total_bytes / time if time else 0.0,
            "big_core_hbm_GBps": own_bytes / time if time else 0.0}


def main():
    out = HERE / "E3"
    checks = json.loads((out / "repeat_checks.json").read_text())
    paths = {"repeat_checks": out / "repeat_checks.json", "inputs": HERE.parent / "round2/results/E0/frozen_inputs.json"}
    rows = []
    for check in checks:
        path = HERE / check["raw_file"]
        assert sha(path) == check["raw_sha256"], "E3 raw记录摘要不匹配"
        paths[check["config"]] = path
        d = decode(check["hardware"])
        for w, r in zip(inputs()["heldout"], load_gzip(path)):
            rows.append({"config": check["config"], "window_id": w["id"], "batch": w["batch"],
                         **shared_service(r, w, d)})
    table = []
    for config in sorted({r["config"] for r in rows}):
        for batch in (*BATCHES, "all"):
            br = [r for r in rows if r["config"] == config and (batch == "all" or r["batch"] == batch)]
            duration = sum(r["shared_interval_cycles"] for r in br)
            table.append({"config": config, "batch": batch, "n_windows": len(br),
                "shared_interval_ms": duration / 1e6,
                "global_hbm_GBps": sum(r["global_hbm_service_bytes"] for r in br) / duration if duration else 0,
                "big_core_hbm_GBps": sum(r["big_core_hbm_service_bytes"] for r in br) / duration if duration else 0})
    write_csv(out / "shared_rate_by_window.csv", rows)
    write_csv(out / "shared_rate_diagnostic.csv", table)
    md = ["# E3 Shared时段：全HBM与大核自身供数\n",
          "仅从已保存的E3 raw segments做积分，没有重新运行仿真，没有改变ablation.csv原列。"
          "1GHz下B/周期等于十进制GB/s；以下取全部135窗口在大核执行Shared期间的墙钟加权平均。\n",
          "全HBM速率=该时段两核供数总字节/该时段时间；大核自身速率=该时段分给大核的供数字节/同一时间。"
          "全HBM速率包含另一核正在执行或预取的路由专家，不能称Shared独占带宽；大核自身速率也包含它的Next预取，"
          "不等于对Shared权重单独打标签的速率。两者均是phase-fluid服务积分，不是原生请求带宽实测。\n",
          "| 消融 | Shared时段全HBM GB/s | 同一时段大核自身 GB/s |",
          "|---|---:|---:|"]
    for row in table:
        if row["batch"] == "all":
            md.append(f"| {row['config']} | {row['global_hbm_GBps']:.6f} | {row['big_core_hbm_GBps']:.6f} |")
    (out / "SHARED_RATE_DIAGNOSTIC.md").write_text("\n".join(md) + "\n")
    write_json(out / "SHARED_RATE_DERIVATION.json", {"no_simulator_rerun": True,
        "source": "原E3已保存raw segments，未修改原指标", "rows": len(rows),
        "inputs_sha256": {str(p.relative_to(HERE.parent)): sha(p) for p in paths.values()},
        "derivation_source_sha256": sha(Path(__file__)),
        "output_sha256": {name: sha(out / name) for name in
            ("shared_rate_diagnostic.csv", "shared_rate_by_window.csv", "SHARED_RATE_DIAGNOSTIC.md")}})


if __name__ == "__main__":
    main()
