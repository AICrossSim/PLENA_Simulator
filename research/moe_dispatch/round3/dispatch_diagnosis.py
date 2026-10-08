"""关闭仿真后，用保存的归属与phase需求审计E5未追平的原因。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from research.moe_dispatch.round2.common import sha, write_csv, write_json, gmean
from research.moe_dispatch.round3.evaluations import HERE, load_gzip, decode


def profile(result, cores):
    ans = []
    for c in range(cores):
        tasks = [t for t in result["tasks"] if t["core"] == c]
        phases = [t for t in result["phases"] if t["core"] == c]
        ans.append({"tasks": len(tasks),
            "assigned_hbm_MiB": sum(t["hbm_bytes"] for t in phases)/2**20,
            "assigned_W_MiB": sum(t["w_sram_bytes"] for t in phases)/2**20,
            "assigned_X_MiB": sum(t["x_sram_bytes"] for t in phases)/2**20,
            "compute_service_ms": result["core_compute_busy"][c]/1e6,
            "task_wall_ms": sum(t["actual_cycles"] for t in tasks)/1e6,
            "phase_wall_ms": sum(t["finish"]-t["start"] for t in phases)/1e6,
            "core_finish_ms": result["core_finish_cycles"][c]/1e6})
    return ans


def main():
    out = HERE / "E5/dispatch"
    all_rows = json.loads((out / "per_window.json").read_text())
    rows = [r for r in all_rows if r["constraint_group"] == "C0"]
    index = {(r["onchip_mode"], r["design"], r["dispatch"], r["window_id"]): r for r in rows}
    with (out / "compare.csv").open() as f:
        compares = list(csv.DictReader(f))
    repeats = json.loads((out / "repeat_checks.json").read_text())
    selected = json.loads((out / "selection.json").read_text())
    chosen = selected["chosen"]
    checks = {(r["onchip_mode"], r["design"], r["dispatch"]): r for r in repeats
              if r["constraint_group"] == "C0" and r["t_big"] == chosen["t_big"] and r["large_first"] == chosen["large_first"]}
    profiles, details, raw_hashes, eft_summary, eft_exceptions = [], [], {}, [], []
    families = sorted({(r["onchip_mode"], r["design"]) for r in rows})
    for mode, design in families:
        fixed = [r for r in rows if (r["onchip_mode"], r["design"], r["dispatch"]) == (mode, design, "fixed")]
        rat = []
        for r in fixed:
            old = index[mode, design, "eft_old", r["window_id"]]
            ratio = r["cycles"]/old["cycles"]
            rat.append(ratio)
            if ratio > 1+1e-12:
                eft_exceptions.append({"onchip_mode": mode, "design": design, "window_id": r["window_id"],
                    "batch": r["batch"], "fixed_eft_ratio": ratio, "eft_old_ms": old["latency_ms"],
                    "fixed_ms": r["latency_ms"], "hbm_fixed_old_ratio": r["hbm_bytes"]/old["hbm_bytes"],
                    "fixed_refetch_tasks": r["refetch_tasks"], "eft_old_refetch_tasks": old["refetch_tasks"],
                    "reason_scope": "额外HBM/重读" if r["hbm_bytes"] > old["hbm_bytes"] else
                                    "总HBM不增；需检查归属、队序、阶段/端口时序，不可仅由完成差归因"})
        eft_summary.append({"onchip_mode": mode, "design": design, "fixed_eft_geomean_ratio": gmean(rat),
            "eft_faster_windows": sum(x > 1+1e-12 for x in rat),
            "eft_faster_over_1pct": sum(x > 1.01 for x in rat),
            "eft_faster_over_5pct": sum(x > 1.05 for x in rat),
            "max_fixed_eft_ratio": max(rat), "n_windows": len(rat)})
        # All main points, including failures, retain per-core demand evidence.
        for policy in ("fixed", "milp"):
            check = checks[mode, design, policy]
            path = HERE / check["raw_file"]
            assert sha(path) == check["raw_sha256"]
            raw_hashes[str(path.relative_to(HERE))] = sha(path)
            d = decode(check["hardware"])
            sums = [dict(tasks=0, assigned_hbm_MiB=0., assigned_W_MiB=0., assigned_X_MiB=0.,
                         compute_service_ms=0., task_wall_ms=0., phase_wall_ms=0., core_finish_ms=0.) for _ in d.cores]
            for r, raw in zip(fixed, load_gzip(path)):
                pr = profile(raw, len(d.cores))
                for c, values in enumerate(pr):
                    details.append({"onchip_mode": mode, "design": design, "dispatch": policy,
                                    "window_id": r["window_id"], "batch": r["batch"], "core": c, **values})
                    for key, value in values.items():
                        sums[c][key] += value
            for c, values in enumerate(sums):
                item = {"onchip_mode": mode, "design": design, "dispatch": policy, "core": c,
                        "geometry": f"{d.cores[c].pm}x{d.cores[c].pn}x{d.cores[c].pk}", "W_banks": d.w_banks[c], "X_banks": d.x_banks[c],
                        "n_windows": len(fixed), **{key+"_mean": value/len(fixed) for key, value in values.items()}}
                profiles.append(item)
    write_csv(out / "core_policy_profiles.csv", profiles)
    write_csv(out / "core_policy_profiles_by_window.csv", details)
    write_csv(out / "fixed_vs_eft_summary.csv", eft_summary)
    write_csv(out / "eft_faster_windows.csv", eft_exceptions)
    md = ["# 256 GB/s 派工验收\n",
          "C0冻结硬件、135个留出窗口；BF16；单位ms；比值取逐窗口配对几何平均。"
          "milp是离线分配加LPT物理回放参照，不能称全局时序最优或严格上界。\n",
          "E5.1两条规则在本轮提交b2e8189c的round3/runtime.py实现：额外重读字节/共享有效带宽计入代价；"
          "全部核都重读时先选最小重读倍数。后续性能补丁不改变这两条。\n",
          f"统一参数为T_big={chosen['t_big']}、大任务优先={chosen['large_first']}，200次开发集bootstrap中该参数"
          f"出现{next(x['bootstrap_count'] for x in selected['scores'] if x['t_big']==chosen['t_big'] and x['large_first']==chosen['large_first'])}次。"
          "旧T_big=3/true的完整同硬件同fixed规则对照见old_runtime_compare.csv。\n",
          "| 模式 | 设计 | fixed GM ms | 在线/离线 | ≤1.01验收 | 对B1 fixed | 对B2 fixed |",
          "|---|---|---:|---:|---|---:|---:|"]
    failures = []
    for mode, design in families:
        comp = next(r for r in compares if (r["onchip_mode"], r["design"], r["dispatch"]) == (mode, design, "fixed"))
        fixed = [r for r in rows if (r["onchip_mode"], r["design"], r["dispatch"]) == (mode, design, "fixed")]
        b2ratio = gmean(r["cycles"]/index[mode,"B2","fixed",r["window_id"]]["cycles"] for r in fixed)
        ratio = float(comp["ratio_vs_milp"])
        md.append(f"| {mode} | {design} | {float(comp['all_geomean_ms']):.6f} | {ratio:.6f} | "
                  f"{'通过' if ratio<=1.01 else '未通过'} | {float(comp['ratio_vs_B1_fixed']):.6f} | {b2ratio:.6f} |")
        if ratio > 1.01:
            milp = [index[mode, design, "milp", r["window_id"]] for r in fixed]
            failures.append({"onchip_mode": mode, "design": design, "ratio": ratio,
                "hbm_ratio": gmean(a["hbm_bytes"]/b["hbm_bytes"] for a,b in zip(fixed,milp)),
                "fixed_finish_gap_ms": sum(r["finish_gap_ms"] for r in fixed)/len(fixed),
                "milp_finish_gap_ms": sum(r["finish_gap_ms"] for r in milp)/len(milp)})
    md += ["\n未通过点的记录诊断（不打断上面的结果表）：\n",
           "| 模式 | 设计 | 在线/离线 | HBM字节在线/离线 | 在线完成差ms | 离线完成差ms | 定位范围 |",
           "|---|---|---:|---:|---:|---:|---|"]
    for r in failures:
        md.append(f"| {r['onchip_mode']} | {r['design']} | {r['ratio']:.6f} | {r['hbm_ratio']:.6f} | "
                  f"{r['fixed_finish_gap_ms']:.6f} | {r['milp_finish_gap_ms']:.6f} | 在线归属、排队与阶段/端口时序；未互斥因果分离 |")
    md += ["\nH51/pipelined的HBM字节配对比恰为1，在线完成差还小于离线；因此其慢12.54%不能归为额外HBM或单纯尾部失衡。"
           "下表直接显示相同硬件下在线策略把阶段服务分到哪一核。phase_wall是该核所有阶段的执行跨度和，包含供数限制，"
           "不是有效MAC活跃时间；需求与各服务积分可重叠，不能相加当墙钟。\n",
           "| H51/pipelined | 核 | W bank | 每窗口分配HBM MiB | 每窗口W需求MiB | 每窗口X需求MiB | 每窗口计算服务ms | 每窗口阶段跨度ms |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in profiles:
        if (r["onchip_mode"],r["design"]) == ("pipelined","H51"):
            md.append(f"| {r['dispatch']} | {r['core']} | {r['W_banks']} | {r['assigned_hbm_MiB_mean']:.3f} | "
                      f"{r['assigned_W_MiB_mean']:.3f} | {r['assigned_X_MiB_mean']:.3f} | "
                      f"{r['compute_service_ms_mean']:.6f} | {r['phase_wall_ms_mean']:.6f} |")
    md += ["\n这些记录支持在线策略的归属/阶段服务分布改变，不能证明任何一项独自占全部延迟差。"
           "私有W/X端口无法借用另一核的空闲份额；窗口内预测和绑定未优化整体共享资源时序，是当前在线策略的限制。"
           "精确分出归属、次序和预取的贡献仍需固定行动的单变量物理回放，本轮没有该因果分离结果。\n",
           "对旧EFT的完整对照（固定硬件）：\n",
           "| 模式 | 设计 | fixed/eft_old GM | EFT严格更快窗口数/135 | fixed慢>1%的窗口数 | fixed最坏/eft_old |",
           "|---|---|---:|---:|---:|---:|"]
    for r in eft_summary:
        md.append(f"| {r['onchip_mode']} | {r['design']} | {r['fixed_eft_geomean_ratio']:.6f} | "
                  f"{r['eft_faster_windows']} | {r['eft_faster_over_1pct']} | {r['max_fixed_eft_ratio']:.6f} |")
    md += ["\n全部留出窗口几何平均上没有C0设计的eft_old快于fixed（单核相同）；单个窗口仍可能更快，"
           "全部实例及HBM是否增加见eft_faster_windows.csv，不能用整体均值掩盖退化。\n",
           "单核540项完整结果回归均一致。在线HBM比离线多2%以上的全部48项已列excess_hbm_causes.csv，"
           "重读倍数并不只由Z容量决定：若Z分块=1，仍可能来自OS等数据流的权重重复加载。"
           "95%置信区间及对B1/B2的比值见acceptance.csv；本轮没有同时对两基线改善5%的点，不进入校准，也不宣称异构胜出。\n"]
    assert all(r["fixed_eft_geomean_ratio"] <= 1+1e-12 for r in eft_summary)
    (out / "SUMMARY.md").write_text("\n".join(md)+"\n")
    write_json(out / "DIAGNOSIS_DERIVATION.json", {"no_simulator_rerun": True, "raw_sha256": raw_hashes,
        "source_sha256": sha(Path(__file__)), "source_summary_sha256": sha(out/"per_window.json"),
        "consumed_sha256": {name: sha(out/name) for name in
                            ("per_window.json", "compare.csv", "repeat_checks.json", "selection.json")},
        "core_profiles": len(profiles), "eft_regression_windows": len(eft_exceptions)})


if __name__ == "__main__":
    main()
