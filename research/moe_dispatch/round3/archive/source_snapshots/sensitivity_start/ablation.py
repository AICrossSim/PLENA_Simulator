"""E3：256 GB/s 下冻结 126 硬件的 W 容量／落地池定点消融。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import itertools
import json
import math
from pathlib import Path
import sys
import time

from ..round2.common import canonical, inputs, gmean, write_csv, write_json
from ..round2.predictors import Predictor
from .evaluations import (HERE, BATCHES, old_designs, configured_params, digest,
                          save_gzip, summarize_result, freeze_sources, receipt, state_digest)

OUT = HERE / "E3"
GPQA = "v3_captured_mixed_heldout_gpqa_t128_l13"


def all_costs_compatible(before, after, p):
    from .model import task_cost
    for w in (*inputs()["development"], *inputs()["heldout"]):
        for e in w["experts"]:
            for c in range(len(before.cores)):
                try:
                    old = task_cost(e, before, c, p)
                except ValueError:
                    continue
                try:
                    new = task_cost(e, after, c, p)
                except ValueError:
                    return False
                if new.hbm_bytes > old.hbm_bytes:
                    return False
    return True


def h2_candidate(old, p):
    """优先从单一部分和/Z分区挪16KiB；都失败再枚举两分区拆分。

    只在全部开发＋留出任务的已有合法核成本不增加重读时准入。借助
    冻结 task_cost 验证，不把容量数字变化假定为无代价。
    """
    small = min(range(2), key=lambda c: old.cores[c].macs)
    big = 1 - small
    w = list(old.w_bytes)
    w[big] = 32 * 1024
    w[small] = 24 * 1024
    extra = sum(w) - sum(old.w_bytes)
    assert extra == 16 * 1024
    donors = [(name, c) for name in ("acc_bytes", "z_bytes", "x_bytes") for c in (big, small)]
    candidates = []
    for name, c in donors:
        values = list(getattr(old, name))
        if values[c] > extra:
            values[c] -= extra
            candidates.append(({name: tuple(values)}, f"{name}[{c}] −16 KiB"))
    for (n1, c1), (n2, c2) in itertools.combinations(donors, 2):
        for a in range(1, 16):
            cut = (a * 1024, (16 - a) * 1024)
            replacements = {name: list(getattr(old, name)) for name in (n1, n2)}
            replacements[n1][c1] -= cut[0]
            replacements[n2][c2] -= cut[1]
            if any(min(v) <= 0 for v in replacements.values()):
                continue
            candidates.append(({k: tuple(v) for k, v in replacements.items()}, f"{n1}[{c1}] −{a} KiB；{n2}[{c2}] −{16-a} KiB"))
    for replacements, explanation in candidates:
        try:
            d = replace(old, w_bytes=tuple(w), **replacements)
        except ValueError:
            continue
        if all_costs_compatible(old, d, p):
            return d, explanation
    # Not claiming global infeasibility across arbitrary multi-donor allocation.
    return None, "在完整单分区及双分区 1KiB 挪动枚举中未找到不增加重读的可行点；不强行放宽容量"


def configurations():
    p = configured_params("pipelined", 520)
    ds = old_designs("pipelined")
    # 7eb58061 compared all frozen geometries with common WS; the original
    # E4 uniform flow was OS/WS. Preserve every physical quota but align
    # this causal W experiment to that prior common-WS protocol.
    ds = {name: replace(d, flows=("WS",) * len(d.cores)) for name, d in ds.items()}
    s, h, m = ds["B1"], ds["best_hetero"], ds["B2"]
    small = min(range(2), key=lambda c: h.cores[c].macs)
    big = 1 - small
    assert h.w_bytes[big] == 16 * 1024 and h.w_bytes[small] == 24 * 1024
    w = list(h.w_bytes)
    w[big], w[small] = 24 * 1024, 16 * 1024
    h1 = replace(h, w_bytes=tuple(w))
    h2, h2why = h2_candidate(h, p)
    specs = {
        "S0": ("single", s, "冻结单核", True),
        "Sinf": ("single", replace(s, diagnostic_unbounded_w=True), "W前瞻无限（诊断；非等资源）", False),
        "H0": ("hetero", h, "冻结异构：大核16／小核24 KiB", True),
        "H1": ("hetero", h1, "大核24／小核16 KiB", True),
        "H3": ("hetero", replace(h, landing_mode="shared", landing_pool_bytes=sum(h.w_bytes), w_bytes=(0, 0)), "共享40 KiB落地池；私有W为0", True),
        "Hinf": ("hetero", replace(h, diagnostic_unbounded_w=True), "两核W前瞻无限（诊断；非等资源）", False),
        "M0": ("uniform", m, "冻结同构", True),
        "M3": ("uniform", replace(m, landing_mode="shared", landing_pool_bytes=sum(m.w_bytes), w_bytes=(0, 0)), "共享40 KiB落地池；私有W为0", True),
        "Minf": ("uniform", replace(m, diagnostic_unbounded_w=True), "两核W前瞻无限（诊断；非等资源）", False),
    }
    if h2 is not None:
        specs["H2"] = ("hetero", h2, "大核32／小核24 KiB；" + h2why, True)
    return specs, {"H2_feasible": h2 is not None, "H2_explanation": h2why,
                   "H2_cost_validation": "全部18开发＋135留出窗口，每个原可行任务/核的HBM字节不增加；原可行核保持可行"}


def job(spec):
    from .runtime import simulate
    label, (design, d, change, iso) = spec
    p = configured_params("pipelined", 520)
    ws = inputs()
    seq, warm_hashes, states = [], [], []
    for _ in range(2):
        pred = Predictor("ours")
        warm = [simulate(w, d, p, dispatch="fixed_legacy", t_big=3, large_first=True, predictor=pred) for w in ws["development"]]
        warm_hashes.append(digest(warm))
        seq.append([simulate(w, d, p, dispatch="fixed_legacy", t_big=3, large_first=True, predictor=pred) for w in ws["heldout"]])
        states.append(state_digest(pred))
    assert canonical(seq[0]) == canonical(seq[1]), (label, "两次完整结果不一致")
    assert warm_hashes[0] == warm_hashes[1] and states[0] == states[1]
    common = {"config": label, "design": design, "change": change, "iso": iso}
    rows = [{**common, **summarize_result(w, d, p, r), "result_digest": digest(r), "repeat_digest": digest(rr)}
            for w, r, rr in zip(ws["heldout"], *seq)]
    gpqa = next(r for w, r in zip(ws["heldout"], seq[0]) if w["id"] == GPQA)
    rec = {**common, "hardware": asdict(d), "parameters": asdict(p), "dispatch": "fixed_legacy",
           "runtime_settings": {"t_big": 3, "large_first": True}, "predictor": "ours",
           "warmup_digest": warm_hashes[0], "repeat_warmup_digest": warm_hashes[1],
           "heldout_digest": digest(seq[0]), "repeat_heldout_digest": digest(seq[1]),
           "state_digest": states[0], "repeat_state_digest": states[1], "exact_repeat": True}
    rawpath = OUT / "raw" / (label + ".json.gz")
    save_gzip(rawpath, seq[0])
    from ..round2.common import sha
    rec["raw_file"] = str(rawpath.relative_to(HERE))
    rec["raw_sha256"] = sha(rawpath)
    return rows, rec, gpqa


def report(rows, feasibility, gpqa):
    baseline = {r["window_id"]: r for r in rows if r["config"] == "S0"}
    table = []
    for config in ("S0", "Sinf", "H0", "H1", "H2", "H3", "Hinf", "M0", "M3", "Minf"):
        cfg = [r for r in rows if r["config"] == config]
        if not cfg:
            continue
        for batch in (*BATCHES, "all"):
            rs = [r for r in cfg if batch == "all" or r["batch"] == batch]
            wall = sum(r["cycles"] for r in rs)
            single_time = sum(r["single_fetcher_cycles"] for r in rs)
            shared_time = sum(r["shared_cycles"] for r in rs)
            table.append({"config": config, "design": rs[0]["design"], "change": rs[0]["change"], "batch": batch,
                "geomean_ms": gmean(r["latency_ms"] for r in rs),
                "ratio_vs_S0": gmean(r["cycles"] / baseline[r["window_id"]]["cycles"] for r in rs),
                "hbm_busy_pct": 100 * sum(r["hbm_busy_frac"] * r["cycles"] for r in rs) / wall,
                "single_fetcher_time_pct": 100 * single_time / wall,
                "single_fetcher_GBps": sum(r["single_fetcher_bytes"] for r in rs) / single_time if single_time else 0,
                "avg_inflight_KiB_core0": sum(r["inflight_area_Bcycles"][0] for r in rs) / wall / 1024,
                "avg_inflight_KiB_core1": sum(r["inflight_area_Bcycles"][1] if len(r["inflight_area_Bcycles"]) > 1 else 0 for r in rs) / wall / 1024,
                "shared_fetch_GBps": sum(r["shared_bytes"] for r in rs) / shared_time if shared_time else 0,
                "hbm_MiB": sum(r["hbm_bytes"] for r in rs) / len(rs) / 2**20,
                "refetch_tasks": sum(r["refetch_tasks"] for r in rs) / len(rs),
                "finish_gap_ms": sum(r["finish_gap_ms"] for r in rs) / len(rs), "iso": rs[0]["iso"]})
    write_csv(OUT / "ablation.csv", table)
    write_csv(OUT / "per_window.csv", rows)
    allrows = {r["config"]: r for r in table if r["batch"] == "all"}
    # Work on paired GM ratios to a common S0. This matches the stated metric
    # without mixing arithmetic means or summing ablation improvements.
    h0, hinf = allrows["H0"]["ratio_vs_S0"], allrows["Hinf"]["ratio_vs_S0"]
    recovery = (h0 - hinf) / (h0 - 1) if abs(h0 - 1) > 1e-12 else None
    conclusion = "W槽不足是主因" if recovery is not None and recovery >= .8 else "W槽不足未达到主因标准"
    facts = {"recovery_fraction": recovery, "recovery_formula": "(GM(H0/S0)−GM(Hinf/S0))/(GM(H0/S0)−1)",
             "conclusion": conclusion, "iso_configs": [k for k, v in allrows.items() if v["iso"]], **feasibility}
    residual = [r for r in rows if r["config"] == "Hinf"]
    traffic = [{"window_id": r["window_id"], "batch": r["batch"],
                "Hinf_ms": r["latency_ms"], "S0_ms": baseline[r["window_id"]]["latency_ms"],
                "Hinf_hbm_MiB": r["hbm_bytes"] / 2**20,
                "S0_hbm_MiB": baseline[r["window_id"]]["hbm_bytes"] / 2**20,
                "hbm_byte_ratio": r["hbm_bytes"] / baseline[r["window_id"]]["hbm_bytes"],
                "Hinf_hbm_floor_ms": r["hbm_bytes"] / 256 / 1e6,
                "Hinf_floor_over_S0": (r["hbm_bytes"] / 256) / baseline[r["window_id"]]["cycles"],
                "Hinf_refetch_tasks": r["refetch_tasks"],
                "S0_refetch_tasks": baseline[r["window_id"]]["refetch_tasks"],
                "refetch_details": r["refetch_details"]} for r in residual]
    facts.update(Hinf_over_S0=gmean(r["cycles"] / baseline[r["window_id"]]["cycles"] for r in residual),
                 Hinf_hbm_bytes_over_S0=gmean(r["hbm_byte_ratio"] for r in traffic),
                 Hinf_actual_hbm_floor_over_S0=gmean(r["Hinf_floor_over_S0"] for r in traffic),
                 Hinf_extra_hbm_windows=sum(r["hbm_byte_ratio"] > 1 for r in traffic),
                 Hinf_refetch_tasks=sum(r["Hinf_refetch_tasks"] for r in traffic),
                 S0_refetch_tasks=sum(r["S0_refetch_tasks"] for r in traffic))
    write_csv(OUT / "remaining_traffic.csv", traffic)
    write_json(OUT / "attribution.json", facts)
    md = ["# E3 定点消融\n", "126下选出、在520个在途请求／256GB/s下评估。135个留出窗口，所有核心WS，与7eb58061一致；原始E4同构数据流为OS/WS，本主表没有沿用该组合，物理几何／容量／bank均保持冻结。派工和预测器保持7eb58061的fixed_legacy＋ours，T_big=3、启用大任务优先。单位ms；延迟和比值为配对几何平均。\n",
          "| 配置 | 改动 | 等资源 | GM ms | /S0 | HBM忙碌% | 单核取数时间% | 单核取数GB/s | Shared时段GB/s | HBM平均MiB/窗口 | 完成差平均ms |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in (v for v in table if v["batch"] == "all"):
        md.append(f"| {r['config']} | {r['change']} | {r['iso']} | {r['geomean_ms']:.6f} | {r['ratio_vs_S0']:.6f} | {r['hbm_busy_pct']:.3f} | {r['single_fetcher_time_pct']:.3f} | {r['single_fetcher_GBps']:.3f} | {r['shared_fetch_GBps']:.3f} | {r['hbm_MiB']:.3f} | {r['finish_gap_ms']:.6f} |")
    md += [f"\n可回收比例为 {recovery:.6f}，结论：**{conclusion}**。" if recovery is not None else "\nH0与S0差为零，可回收比例不定义。",
           f"\nW无限诊断后，Hinf/S0的配对延迟比仍为{facts['Hinf_over_S0']:.6f}，HBM字节比为{facts['Hinf_hbm_bytes_over_S0']:.6f}；{facts['Hinf_extra_hbm_windows']}/135个窗口的字节更多。Hinf重读任务共{facts['Hinf_refetch_tasks']}个，S0为{facts['S0_refetch_tasks']}个，逐任务记录显示小核Z分区导致的行分块仍存在。Hinf实际字节/BW相对S0延迟的配对几何平均为{facts['Hinf_actual_hbm_floor_over_S0']:.6f}，因此额外流量是必要服务成本，但它仍不能单独证明Hinf必须更慢。详见remaining_traffic.csv。\n",
           "W无限仅放宽现有数据流的权重容量／前瞻准入，不自动引入跨Z行块保留整专家全部权重的新循环变换；该诊断不能代表任何使用无限缓存的最优数据流。\n",
           "\n取数指标以segment积分：表和曲线的inflight为**服务等价代理量**（当下流体供数率×65周期），不是实际在途请求字节。冷响应等待时该代理量为0；短预取突发时代理量可能超过其瞬时私有槽容量，不能用它验证槽位或credit准入。实际有限空间由另行记录的pool_ledger检查。single_fetcher也是仅一个核获得正流体供数的时段，不是真实请求仍在途的时段。HBM忙碌、代理量和速率按墙钟加权。HBM MiB与重读任务数、完成差为逐窗口算术平均。Shared速率为大核运行Shared期间全HBM的供数速率，包含另一核，不能称Shared独享带宽。\n",
           "Sinf/Hinf/Minf放宽W前瞻的非等资源诊断，只用于归因；不改变被计费的片上端口带宽。不同消融收益不得相加。\n",
           feasibility["H2_explanation"] + "\n",
           "剩余时间差只能用数值诊断区分，不能把重叠资源占用加成互斥latency breakdown。下面给出相同运行的重叠诊断：核完成差对应尾部失衡，绑定时刻记录派工，计算服务积分对应阵列工作。它们不能直接宣称为互斥的差距来源；本轮没有另外关闭派工或计算来因果分离剩余差距。\n",
           "| 配置 | GM ms | HBM忙碌% | 平均完成差ms | 核0计算服务平均ms | 核1计算服务平均ms | W端口服务平均ms | X端口服务平均ms |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("S0", "H0", "H2", "H3", "Hinf"):
        rs = [r for r in rows if r["config"] == name]
        if not rs:
            continue
        avg = lambda key: sum(r[key] for r in rs) / len(rs)
        core0 = sum(r["core_compute_busy"][0] for r in rs) / len(rs) / 1e6
        core1 = sum(r["core_compute_busy"][1] if len(r["core_compute_busy"]) > 1 else 0 for r in rs) / len(rs) / 1e6
        v = allrows[name]
        md.append(f"| {name} | {v['geomean_ms']:.6f} | {v['hbm_busy_pct']:.4f} | {avg('finish_gap_ms'):.6f} | {core0:.6f} | {core1:.6f} | {avg('w_port_busy')/1e6:.6f} | {avg('x_port_busy')/1e6:.6f} |")
    (OUT / "ABLATION.md").write_text("\n".join(md) + "\n")
    draw(table, gpqa)
    return facts


def draw(table, gpqa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    rs = [r for r in table if r["batch"] == "all"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    bars = ax.bar([r["config"] for r in rs], [r["geomean_ms"] for r in rs],
                  color=["#377eb8" if r["iso"] else "#aaaaaa" for r in rs])
    for bar, r in zip(bars, rs):
        if not r["iso"]:
            bar.set_hatch("//")
    ax.set(ylabel="Heldout geomean latency (ms)", xlabel="Frozen-hardware ablation")
    fig.savefig(HERE / "figures/fig_ablation.pdf")
    plt.close(fig)
    labels = [k for k in ("H0", "H3", "H2") if k in gpqa]
    fig, axs = plt.subplots(len(labels), 1, figsize=(12, 3.2 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axs = [axs]
    for label, ax in zip(labels, axs):
        seg = gpqa[label]["segments"]
        for c in (0, 1):
            ax.step([s["start"] / 1e6 for s in seg], [s["inflight_bytes"][c] / 1024 for s in seg],
                    where="post", label=f"core{c} inflight proxy")
        other = ax.twinx()
        other.step([s["start"] / 1e6 for s in seg], [s["hbm_rate_Bpc"] for s in seg],
                   color="#d95f02", alpha=.45, where="post", label="HBM rate")
        ax.set(title=label + " — phase-fluid estimate", xlabel="Time (ms)", ylabel="Inflight proxy (KiB)")
        other.set_ylabel("HBM supply (GB/s)")
        ax.legend(loc="upper left")
    fig.savefig(HERE / "figures/fig_gpqa_inflight.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = freeze_sources()
    start = time.monotonic()
    specs, feasibility = configurations()
    rows, checks, gpqa = [], [], {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for spec, (rs, rec, case) in zip(specs.items(), pool.map(job, specs.items())):
            rows.extend(rs); checks.append(rec); gpqa[spec[0]] = case
            print({"E3": spec[0], "GM_ms": gmean(r["latency_ms"] for r in rs)}, flush=True)
    save_gzip(OUT / "gpqa_full_results.json.gz", gpqa)
    save_gzip(OUT / "window_results.json.gz", rows)
    write_json(OUT / "repeat_checks.json", checks)
    facts = report(rows, feasibility, gpqa)
    receipt(OUT, sys.argv, len(specs), before, time.monotonic() - start, checks, facts)


if __name__ == "__main__":
    main()
