"""Read-only reporting for the frozen-hardware, WS predictor ablation.

This module does not choose a design, predictor, threshold, or test-set policy.
Its statistics use paired windows and distinguish task duration from bind ETA.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

from ...common import BATCHES, ROOT, sha, write_json
from ...predictors import Predictor

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MODES = ("pipelined", "port_tight")
DESIGNS = ("B1", "B2", "best_hetero")
PREDICTORS = ("none", "nominal", "random", "static", "btb", "ema", "ours")


def read_csv(name, directory=HERE):
    path = directory/name
    opener = path.open if path.exists() else lambda **kw:gzip.open(path.with_suffix(path.suffix+".gz"),"rt",**kw)
    with opener(newline="") as f:
        return list(csv.DictReader(f))


def raw_sha(path):
    """Hash original source bytes, independent of exact gzip storage."""
    path = Path(path)
    if path.exists():
        return sha(path)
    with gzip.open(path.with_suffix(path.suffix+".gz"),"rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_csv(name, rows):
    rows = list(rows)
    assert rows
    with (HERE/name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or value in ("", "None", "null"):
        return None
    value = float(value)
    assert math.isfinite(value)
    return value


def gm(values):
    values = tuple(values)
    assert values and all(v > 0 and math.isfinite(v) for v in values)
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def paired_ci(ratios, key, draws=2000):
    logs = tuple(math.log(r) for r in ratios)
    seed = int(hashlib.sha256(repr(key).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    samples = sorted(100*(1-math.exp(math.fsum(rng.choice(logs) for _ in logs)/len(logs)))
                     for _ in range(draws))
    def quantile(q):
        i = (len(samples)-1)*q
        lo = int(i)
        hi = math.ceil(i)
        return samples[lo] + (samples[hi]-samples[lo])*(i-lo)
    return quantile(.025), quantile(.975)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def task_summary(tasks):
    duration_errors, duration_under = [], []
    finish_errors, finish_under = [], []
    prefetch_waits = []
    for t in tasks:
        pred, actual = number(t["predicted_cycles"]), number(t["actual_cycles"])
        if pred is not None and actual is not None and actual > 0:
            duration_errors.append(abs(pred-actual)/actual)
            duration_under.append(max(0, actual-pred)/actual)
        eta, finish = number(t["predicted_finish_at_bind"]), number(t["actual_finish_cycle"])
        if eta is not None and finish is not None:
            finish_errors.append(abs(eta-finish))
            finish_under.append(max(0, finish-eta))
        ready, end = number(t["first_weight_ready"]), number(t["current_end"])
        if ready is not None and end is not None:
            prefetch_waits.append(max(0, ready-end))
    assert len(duration_errors) == len(tasks)
    assert len(finish_errors) == len(tasks)
    mean = lambda xs: math.fsum(xs)/len(xs) if xs else None
    return {
        "tasks": len(tasks),
        "task_duration_mean_abs_pct": 100*mean(duration_errors),
        "worstunderestimate_pct": 100*max(duration_under, default=0),
        "binding_mean_absolute_error_us": mean(finish_errors)/1000,
        "binding_worst_underestimate_us": max(finish_under, default=0)/1000,
        "prefetch_late_pct": 100*mean([w > 0 for w in prefetch_waits]) if prefetch_waits else None,
        "prefetch_exposed_wait_sum_ms": math.fsum(prefetch_waits)/1e6,
        "next_samples": len(prefetch_waits),
    }


def main():
    control = HERE/"nominal_control"
    rows = read_csv("per_window.csv") + read_csv("per_window.csv", control)
    tasks = read_csv("task_predictions.csv") + read_csv("task_predictions.csv", control)
    assert len(rows) == 5670
    metadata = json.loads((HERE/"METADATA.json").read_text())
    receipts = json.loads((HERE/"repeat_checks.json").read_text()) + json.loads((control/"repeat_checks.json").read_text())
    assert len(receipts) == 42
    keys = ("onchip_mode", "design", "predictor")
    groups, task_groups, baseline = defaultdict(list), defaultdict(list), {}
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in keys)
        assert key[0] in MODES and key[1] in DESIGNS and key[2] in PREDICTORS
        assert row["result_digest"] == row["repeat_digest"]
        assert (key, row["window_id"]) not in seen
        seen.add((key, row["window_id"]))
        row["batch"] = int(row["batch"])
        for field in ("cycles", "latency_ms", "hbm_bytes"):
            row[field] = number(row[field])
        assert math.isclose(row["latency_ms"], row["cycles"]/1e6, rel_tol=1e-12, abs_tol=1e-12)
        groups[key].append(row)
        if key[2] in ("none", "nominal"):
            baseline[key[:2]+(row["window_id"],key[2])] = row
    for task in tasks:
        task["batch"] = int(task["batch"])
        key = tuple(task[k] for k in keys)
        assert key in groups
        task_groups[key].append(task)
    assert len(groups) == 42 and all(len(rs) == 135 for rs in groups.values())
    assert set(groups) == {(m,d,p) for m in MODES for d in DESIGNS for p in PREDICTORS}
    expected_batches = {2:27,4:27,8:27,16:27,64:9,96:9,128:9}
    assert all({b:sum(r["batch"] == b for r in rs) for b in BATCHES} == expected_batches
               for rs in groups.values())
    summaries = []
    for mode in MODES:
        for design in DESIGNS:
            for predictor in PREDICTORS:
                key = mode, design, predictor
                for batch in (*BATCHES, "all"):
                    rs = [r for r in groups[key] if batch == "all" or r["batch"] == batch]
                    ts = [t for t in task_groups[key] if batch == "all" or t["batch"] == batch]
                    ratios = [r["cycles"]/baseline[mode,design,r["window_id"],"none"]["cycles"] for r in rs]
                    nominal_ratios = [r["cycles"]/baseline[mode,design,r["window_id"],"nominal"]["cycles"] for r in rs]
                    lo, hi = paired_ci(ratios, key+(batch,))
                    nlo, nhi = paired_ci(nominal_ratios, key+(batch,"nominal"))
                    summary = {
                        "onchip_mode": mode, "design": design, "predictor": predictor,
                        "batch": batch, "windows": len(rs),
                        "latency_geomean_ms": gm(r["latency_ms"] for r in rs),
                        "HBM_geomean_MiB": gm(r["hbm_bytes"]/2**20 for r in rs),
                        "HBM_sum_GiB": math.fsum(r["hbm_bytes"] for r in rs)/2**30,
                        "ratio_vs_none": gm(ratios),
                        "speedup_vs_none_pct": 100*(1-gm(ratios)),
                        "speedup_CI95_lower_pct": lo, "speedup_CI95_upper_pct": hi,
                        "ratio_vs_nominal":gm(nominal_ratios),
                        "speedup_vs_nominal_pct":100*(1-gm(nominal_ratios)),
                        "nominal_speedup_CI95_lower_pct":nlo,"nominal_speedup_CI95_upper_pct":nhi,
                        **task_summary(ts),
                    }
                    assert math.isclose(summary["ratio_vs_none"], summary["latency_geomean_ms"]/gm(
                        baseline[mode,design,r["window_id"],"none"]["latency_ms"] for r in rs), rel_tol=1e-12)
                    assert math.isclose(summary["ratio_vs_nominal"], summary["latency_geomean_ms"]/gm(
                        baseline[mode,design,r["window_id"],"nominal"]["latency_ms"] for r in rs), rel_tol=1e-12)
                    summaries.append(summary)
    assert len(summaries) == 336
    write_csv("compare.csv", summaries)
    lookup = {(r["onchip_mode"],r["design"],r["predictor"],r["batch"]):r for r in summaries}
    frozen = json.loads((PARENT/"frozen_designs.json").read_text())
    old_anchor_checks = read_csv("baseline_reproduction.csv")
    assert len(old_anchor_checks) == 810
    anchor_max_delta = max(abs(number(r["cycles_delta"])) for r in old_anchor_checks)
    anchor_hbm_exact = all(r["hbm_bytes_exact"] == "True" for r in old_anchor_checks)
    anchor_digest_count = sum(r["full_result_digest_exact"] == "True" for r in old_anchor_checks)
    counts = table(("Batch", "留出窗口数"), [["B"+str(b),expected_batches[b]] for b in BATCHES])
    out = ["# 固定硬件、统一 WS：六种估计器与进度触发控制对照", "",
        "本次仅替换运行时预测/估计器，既有容量准入、等待比较和大任务优先均保持启用。",
        "单核、同构、异构均使用 T_big=3、large_first=True；不按测试 workload 换硬件或参数。",
        "各模式沿用原 E4 几何/私有资源，并给所有核统一 WS，避免把数据流差异写成预测器收益。",
        "36个预测器配置＋6个nominal进度触发控制配置，每个配置从全新状态预热18开发窗口，再按固定顺序回放135留出窗口；整条序列重复两次。",
        "这里预测任务时间与预计完成时间，用于比较归属和触发预取；Router 已给出 Expert ID 与 Me，不预测 Expert ID。",
        "范围为路由后的 BF16 MoE FFN phase-fluid analytical 模型；非完整模型、RTL或原生HBM实测。",
        "1 cycle = 1 ns。延迟表按窗口取几何平均，单位ms；改善为1−候选/基线，正数表示更快。", "",
        "## 本轮共同修复的准入检查事件", "",
        "首轮发现：核的Current+Next队列已满、物理上不能再绑定时，旧逻辑仍按预计空闲时间反复安排检查，",
        "随机估计接近绑定阈值时形成过密事件。该尝试已停止，失败执行记录和源码保留，未把半跑结果填入表。",
        "本轮所有组织和估计器共同使用bounded_runtime私有副本：物理不可准入时不安排仅由ETA触发的重试，",
        "等待实际完成、阶段切换或进度事件再次检查；资源容量和成本公式保持原样。旧runtime与第二轮冻结结果没有改写。",
        f"对上一轮WS的810个anchor核验：HBM字节全部一致={anchor_hbm_exact}，最大绝对周期差={anchor_max_delta:.12g} ns，完整结果digest一致{anchor_digest_count}/810。",
        "完整digest还包括绑定/候选检查轨迹，不能仅凭它不一致断言物理延迟有变化；具体逐窗口差见baseline_reproduction.csv。", "",
        "## 估计器究竟有什么区别", "",
        table(("名称", "使用的时间估计", "反馈更新", "抽象状态预算"), [
            ["none", "形状、存储、端口/HBM资源成本模型给出的 nominal 时间", "无历史学习", "0 B新增预测状态"],
            ["nominal", "与none相同的资源成本估计", "不学习；保留1/4进度触发重新检查", "0 B新增学习状态"],
            ["random", "0到2×每核历史均值之间的随机时长", "更新每核均值", f"{Predictor('random').state_bits()/8:.2f} B"],
            ["static", "每核已完成任务的累计均值；无样本时用nominal", "完成后更新累计均值", f"{Predictor('static').state_bits()/8:.2f} B"],
            ["btb", "按核、Shared标记、Me桶记录的上次实际时长", "直接替换该桶记录", f"{Predictor('btb').state_bits()/8:.2f} B"],
            ["ema", "同btb的桶内平滑时长", "旧值＋1/4×(实测−旧值)", f"{Predictor('ema').state_bits()/8:.2f} B"],
            ["ours", "资源成本nominal×按M桶校准系数，并修正执行中的剩余时间", "1/4更新、Q16.16系数；1/4进度点反馈", f"{Predictor('ours').state_bits()/8:.2f} B"],
        ]), "",
        "random是随机时间估计，不是随机选核；static这个名称保留自原代码，实际上是在线累计平均。",
        "random使用原代码固定随机种子20261007；两遍检查可重复性，本轮未做多种子随机策略评估。",
        "btb/ema的Me桶为min(Me,9)，不以专家ID索引；ours使用按log2(Me)截断的8个桶。",
        "状态预算只统计代码指定的安装状态位，不含整个dispatcher、描述符、队列和操作数缓冲，也不是综合后的面积或时序。", "",
        "该状态预算按代码的两核安装状态统计；本次未综合单核裁剪版本，不能将它直接当作每个设计的物理面积。", "",
        "**因果控制：none传入None，跳过25/50/75%进度触发；nominal以及其余五种对象都保留这些重新绑定/预取检查。**",
        "因此nominal→ours才隔离出有共同进度检查时的估计/反馈机制收益；none→ours还包含进度检查本身，不能全归因于预测准确性。", "",
        "## 硬件和样本", "",
        table(("模式", "组织", "核0；核1（M×N×K）", "本次数据流"),
            [[m,d,"；".join(f"{c['pm']}×{c['pn']}×{c['pk']}" for c in frozen['modes'][m][d]['cores']),
              "/".join("WS" for _ in frozen['modes'][m][d]['cores'])] for m in MODES for d in DESIGNS]), "",
        "B1为形状选定单核；B2为各模式选定同构；best_hetero的核0小、核1大。三个组织均为12288乘法器。",
        counts, "",
        "## 汇总：同一硬件上，相对共同进度触发nominal到底快多少", "",
        table(("模式", "组织", "估计器", "全部窗口延迟ms", "相对nominal改善%", "改善95%CI", "HBM总量GiB"),
            [[r['onchip_mode'],r['design'],r['predictor'],f"{r['latency_geomean_ms']:.4f}",
              f"{r['speedup_vs_nominal_pct']:+.3f}",f"[{r['nominal_speedup_CI95_lower_pct']:+.3f}, {r['nominal_speedup_CI95_upper_pct']:+.3f}]",
              f"{r['HBM_sum_GiB']:.3f}"] for r in summaries if r['batch']=='all']), "",
        "95%CI按窗口配对进行2000次bootstrap。它描述固定留出样本的统计波动，不校准解析模型的物理误差；",
        "bootstrap重采样已记录的窗口比值，不重新训练或重放预测器，因此不等于在线历史顺序变化的置信区间。",
        "各预测器使用同一份窗口且对结果逐窗口配对，不把不同硬件的差异写成预测收益。", ""]
    for mode in MODES:
        for design in DESIGNS:
            out += [f"## {mode}／{design}：完整分batch总延迟（ms）", "",
                table(("估计器", *["B"+str(b) for b in BATCHES], "全部窗口GM"),
                    [[p,*[f"{lookup[mode,design,p,b]['latency_geomean_ms']:.4f}" for b in (*BATCHES,'all')]]
                     for p in PREDICTORS]), ""]
    out += ["## 预测误差和预取到达：按所有任务统计", "",
        "任务时长误差比较绑定时保存的预测时长与任务start→finish实际时长；该实际时长含执行期间共享资源等待。",
        "绑定ETA误差比较绑定时预计完成时刻与最终完成时刻，另外包含排队及供数变化。两种误差不能互换。",
        "预取晚到仅统计first_weight_ready与current_end都存在的Next任务。正等待量不等于总内存stall，",
        "会与其他核工作或其他等待重叠，不能加到总延迟上。这里first_weight_ready是共享fluid前缀到达观测，不是原生HBM逐请求时间。", "",
        table(("模式", "组织", "估计器", "时长MAPE%", "最差时长低估%", "平均绑定ETA误差µs", "最差绑定ETA低估µs", "Next晚到%", "Next样本数", "正等待总和ms"),
            [[r['onchip_mode'],r['design'],r['predictor'],f"{r['task_duration_mean_abs_pct']:.2f}",
              f"{r['worstunderestimate_pct']:.2f}",f"{r['binding_mean_absolute_error_us']:.2f}",
              f"{r['binding_worst_underestimate_us']:.2f}",
              "—" if r['prefetch_late_pct'] is None else f"{r['prefetch_late_pct']:.2f}",r['next_samples'],
              f"{r['prefetch_exposed_wait_sum_ms']:.4f}"] for r in summaries if r['batch']=='all']), "",
        "误差按任务等权平均；延迟按窗口几何平均。高时长误差不必然意味着整层很慢：HBM下限、容量准入和真实就绪依赖仍限制可改变的空间。", "",
        "## 如何读这组数据", "",
        "先在同一组织内比较nominal与各估计器，判断预测是否提供额外收益；用none→nominal检查进度触发本身；再比较不同组织，判断硬件是否有优势。",
        "这次数据不能宣称某个估计器是全局最优、首次提出，或把随机时长基线等同于随机dispatch。",
        "旧第二轮E5使用旧EFT且只覆盖两类异构，与这里的新fixed、三种DSE组织、统一WS不混用。",
        "本轮没有补跑oracle。旧E5的conditional oracle是固定ours实际归属/绑定/预取计划的物理重放；",
        "另存的profile-guided oracle是两遍profile引导，两者都不是全局最优派工上界，不能替代离线MILP/LPT参照。", "",
        "完整逐窗口数据见per_window.csv；每次任务预测见task_predictions.csv；全部分组指标见compare.csv。",
    ]
    h0 = lookup['pipelined','best_hetero','nominal','all']
    ho = lookup['pipelined','best_hetero','ours','all']
    po = lookup['port_tight','best_hetero','ours','all']
    out += ["", "## 本轮实际结论", "",
        f"1. 在异构硬件上，ours相对同进度检查的nominal，主模式慢{-ho['speedup_vs_nominal_pct']:.3f}%，端口受限慢{-po['speedup_vs_nominal_pct']:.3f}%；本轮不能写预测器带来了系统提速。",
        f"2. 主模式异构任务时长MAPE从nominal的{h0['task_duration_mean_abs_pct']:.2f}%降到ours的{ho['task_duration_mean_abs_pct']:.2f}%，但总HBM流量从{h0['HBM_sum_GiB']:.3f}增至{ho['HBM_sum_GiB']:.3f} GiB。预测时长更准，并不保证全层派工更好。",
        "3. none/nominal本来已经按真实Me、形状、容量和有限资源计算成本；它们不是盲目派工。ours优于随机估计或全核均值，主要说明避免粗略时长估计的重要性，不能据此宣称超越已有形状/资源感知派工。",
    ]
    (HERE/"TABLES_ZH.md").write_text("\n".join(out)+"\n")
    write_json(HERE/"REPORT_CHECKS.json", {
        "per_window_rows":len(rows),"task_prediction_rows":len(tasks),"compare_rows":len(summaries),
        "configurations":len(groups),"heldout_windows_each":135,
        "exact_result_repeats":all(r['result_digest']==r['repeat_digest'] for r in rows),
        "paired_geomean_identities":True,"mode_design_predictor_grid_complete":True,
        "duration_error_denominator":"actual task cycles",
        "prefetch_scope":"conditional observed first-weight ingress after previous-task completion; not wall-time stall",
        "input_metadata_sha256":sha(HERE/"METADATA.json"),
        "source_report_sha256":sha(Path(__file__)),
        "source_grid_sha256":raw_sha(HERE/"per_window.csv"),
        "source_tasks_sha256":raw_sha(HERE/"task_predictions.csv"),
        "source_nominal_grid_sha256":raw_sha(control/"per_window.csv"),
        "source_nominal_tasks_sha256":raw_sha(control/"task_predictions.csv"),
        "raw_source_sha256":{str(p.relative_to(HERE)):raw_sha(p)
            for directory in (HERE,control)
            for p in [directory/n for n in ("per_window.csv","task_predictions.csv","repeat_checks.json","METADATA.json")]},
        "archive_file_sha256":{str(p.relative_to(HERE)):sha(p)
            for directory in (HERE,control) for p in sorted(directory.glob('*.gz'))},
        "metadata_available":bool(metadata),
        "baseline_anchor_rows":len(old_anchor_checks),
        "baseline_hbm_bytes_exact":anchor_hbm_exact,
        "baseline_max_absolute_cycle_delta":anchor_max_delta,
        "baseline_full_digest_exact_count":anchor_digest_count,
    })
    provenance_files = list(HERE.iterdir()) + list(control.iterdir())
    write_csv("REPORT_PROVENANCE.csv", [
        {"file":str(p.relative_to(HERE)),"sha256":sha(p)} for p in sorted(provenance_files)
        if p.is_file() and p.name not in ("REPORT_PROVENANCE.csv", "INDEPENDENT_AUDIT.json", "PROVENANCE.csv")
        and not (p.suffix=='.csv' and p.with_suffix('.csv.gz').exists())])
    print(json.dumps({"rows":len(summaries),"tasks":len(tasks),"output":"TABLES_ZH.md"}))


if __name__ == "__main__":
    main()
