"""Compute-only, batch-balanced full-space DSE with development-only selection.

No memory feasibility filters, memory timing, vector service, or predictor.
K segments of the same output wait for the previous commit. Independent
Gate/Up outputs may overlap. Primary experiments impose no output-store bound,
because this is a compute-only ceiling. Expanded shapes are prospective; no
synthesis/clock claim. Finite-output scenarios are separate diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .compute import Core, ContextLimits, TimingProfile, projection
from .study import verify_capture

BUDGET = 12288
KS = (16, 32, 64, 128, 256, 512, 1024, 2048)
RESIDENT_RECORDS = (0, 8, 16, 32)  # 0 means all outputs; other values are per core.
LATENCY_NAMES = ("flat20", "log_stage1", "log_stage2")
NOMINAL = "flat20_ideal"
FAMILIES = ("single", "homogeneous", "heterogeneous")


@dataclass(frozen=True, order=True)
class Shape(Core):
    def __post_init__(self):
        if min(self.pm, self.pn) < 1 or self.pk not in KS or self.macs > BUDGET:
            raise ValueError("positive integer PM/PN, supported PK, MACs <= budget required")


@dataclass(frozen=True)
class ComputeTiming(TimingProfile):
    def __post_init__(self):
        if tuple(sorted(k for k, _ in self.latency_by_pk)) != KS:
            raise ValueError("one latency per declared PK required")
        if min(v for _, v in self.latency_by_pk) < 1 or self.initiation_interval < 1 or self.commit_cycles < 1:
            raise ValueError("positive arithmetic timing required")


def timing(name):
    slope = {"flat20": 0, "log_stage1": 1, "log_stage2": 2}[name]
    return ComputeTiming(name, tuple((k, 20 + slope * (k.bit_length() - 10)) for k in KS),
                         hypothesis="unvalidated flat spatial reduction tree; PK512 dot20 anchor")


def gid(cores):
    return "+".join(f"{c.pm}x{c.pn}x{c.pk}" for c in sorted(cores))


def parse_gid(value):
    return tuple(Shape(*map(int, s.split("x"))) for s in value.split("+"))


@lru_cache(maxsize=1)
def domain():
    shapes = tuple(sorted(Shape(m, n, k) for k in KS
                          for m in range(1, BUDGET // k + 1)
                          for n in range(1, BUDGET // (k * m) + 1)))
    by_budget = defaultdict(list)
    for i, c in enumerate(shapes):
        by_budget[c.macs].append(i)
    geometries = [(i, -1) for i in by_budget[BUDGET]]
    geometries += [(i, j) for i, c in enumerate(shapes)
                   for j in by_budget[BUDGET - c.macs] if i <= j]
    indices = np.asarray(geometries, dtype=np.int64)
    fam = np.where(indices[:, 1] < 0, 0, np.where(indices[:, 0] == indices[:, 1], 1, 2))
    return shapes, indices, fam


def grouped_cycles(records, segments, q, latency):
    """Vectorized exact group formula; latency includes FP32 commit, II=1."""
    full, tail = records // q, records % q
    replacement = (segments - 1) * np.maximum(q, latency) + q - 1 + latency
    last_q = np.where(tail > 0, tail, q)
    last = (segments - 1) * np.maximum(last_q, latency) + last_q - 1 + latency
    return np.maximum(0, full - (tail == 0)) * replacement + last


def costs(shapes, signatures, latency_name, records_per_core):
    """Return expert compute cycles and padding counts, excluding vector ops."""
    pm = np.array([c.pm for c in shapes], dtype=np.int64)[:, None]
    pn = np.array([c.pn for c in shapes], dtype=np.int64)[:, None]
    pk = np.array([c.pk for c in shapes], dtype=np.int64)[:, None]
    m, h, f = (np.array([s[i] for s in signatures], dtype=np.int64)[None, :] for i in range(3))
    nm = (m + pm - 1) // pm
    nf, nh = (f + pn - 1) // pn, (h + pn - 1) // pn
    kh, kf = (h + pk - 1) // pk, (f + pk - 1) // pk
    t = timing(latency_name)
    latency = np.array([t.completion_latency(c) for c in shapes], dtype=np.int64)[:, None]
    # Separate Gate/Up tails, but both independent output sets can issue in
    # one K-major stream. No artificial Gate -> Up dependency is inserted.
    gate_records, down_records = 2 * nm * nf, nm * nh
    gate_q = gate_records if records_per_core == 0 else records_per_core
    down_q = down_records if records_per_core == 0 else records_per_core
    gate = grouped_cycles(gate_records, kh, gate_q, latency)
    down = grouped_cycles(down_records, kf, down_q, latency)
    cycles = gate + down
    issues = 2 * nm * nf * kh + nm * nh * kf
    issued = issues * (pm * pn * pk)
    useful = np.broadcast_to(3 * m * h * f, cycles.shape)
    return cycles, issued, useful, issues


def ordered_experts(w):
    # Same logical-work LPT order for every geometry. No learned affinity,
    # per-family exception, cross-expert GEMM packing, or shared pinning.
    return sorted(w["experts"], key=lambda e: (-3 * e["Me"] * e["H"] * e["F"], e["id"]))


def vector_schedule(workloads, signatures, costs_single, costs_dual, indices):
    signature_ids = {s: i for i, s in enumerate(signatures)}
    a, b = indices[:, 0], indices[:, 1]
    single = b < 0
    b_safe = np.maximum(b, 0)
    results = np.empty((len(indices), len(workloads)), dtype=np.int64)
    for wi, w in enumerate(workloads):
        load0 = np.zeros(len(indices), dtype=np.int64)
        load1 = np.zeros(len(indices), dtype=np.int64)
        for e in ordered_experts(w):
            sid = signature_ids[(e["Me"], e["H"], e["F"])]
            c0 = np.where(single, costs_single[a, sid], costs_dual[a, sid])
            c1 = costs_dual[b_safe, sid]
            owner0 = single | (load0 + c0 <= load1 + c1)
            load0 += np.where(owner0, c0, 0)
            load1 += np.where(owner0, 0, c1)
        results[:, wi] = np.maximum(load0, load1)
    return results


def balanced_score(cycles, anchor, workloads):
    """Equal batch weights, equal windows within each batch; paired ratios."""
    batch = np.array([w["batch"] for w in workloads])
    logs = np.log(cycles / np.asarray(anchor))
    return np.exp(np.stack([logs[:, batch == b].mean(axis=1) for b in sorted(set(batch))]).mean(axis=0))


def minimax_selection(scores, fam):
    selected, regrets = {}, np.ones_like(scores)
    for fi, name in enumerate(FAMILIES):
        ids = np.flatnonzero(fam == fi)
        regrets[:, ids] = scores[:, ids] / scores[:, ids].min(axis=1)[:, None]
        # No held-out results enter selection. Equal worst regret breaks ties
        # by geometric regret, then the deterministic canonical domain index.
        worst = regrets[:, ids].max(axis=0)
        average = np.log(regrets[:, ids]).mean(axis=0)
        order = np.lexsort((ids, average, worst))
        selected[name] = int(ids[order[0]])
    return selected, regrets


def scalar_expert(e, c, latency_name, records):
    # Capacity is deliberately nonbinding: this campaign has no SRAM model.
    gate_records = 2 * ((e['Me']+c.pm-1)//c.pm) * ((e['F']+c.pn-1)//c.pn)
    down_records = ((e['Me']+c.pm-1)//c.pm) * ((e['H']+c.pn-1)//c.pn)
    limits = ContextLimits(down_records if records == 0 else records, 1 << 40)
    t = timing(latency_name)
    # Count and dependency boundaries retain Gate/Up as different outputs;
    # one combined physical-record stream does not pad a concatenated 2F.
    gu_limits = ContextLimits(gate_records if records == 0 else records, 1 << 40)
    padded_pair_n = 2 * ((e['F']+c.pn-1)//c.pn) * c.pn
    gu = projection(e["Me"], padded_pair_n, e["H"], c, t, gu_limits)
    down = projection(e["Me"], e["H"], e["F"], c, t, limits)
    return {"cycles": gu.cycles + down.cycles, "gate_up_cycles": gu.cycles,
            "down_cycles": down.cycles, "useful_macs": 3 * e['Me'] * e['H'] * e['F'],
            "issued_macs": gu.issued_macs + down.issued_macs,
            "issues": gu.issues + down.issues,
            "required_live_FP32_output_bytes": max(gate_records,down_records) * c.record_bytes
            if records == 0 else records * c.record_bytes}


def scalar_layer(w, cores, latency_name="flat20", records=0):
    loads = [0] * len(cores)
    useful = issued = issues = 0
    allocations = []
    for e in ordered_experts(w):
        choices = [scalar_expert(e, c, latency_name, records) for c in cores]
        owner = min(range(len(cores)), key=lambda i: (loads[i] + choices[i]["cycles"], i))
        cost = choices[owner]
        start = loads[owner]
        loads[owner] += cost["cycles"]
        useful += cost["useful_macs"]
        issued += cost["issued_macs"]
        issues += cost["issues"]
        allocations.append({"expert_id": e["id"], "Me": e["Me"], "is_shared": e["is_shared"],
                            "owner": owner, "start_cycle": start, "finish_cycle": loads[owner], **cost})
    cycles = max(loads)
    return {"window": w["id"], "batch": w["batch"], "geometry": gid(cores),
            "cycles": cycles, "latency_ms_at_assumed_1GHz": cycles / 1e6,
            "useful_compute_lower_bound_cycles": (useful + BUDGET - 1) // BUDGET,
            "core_cycles": loads, "core_finish_gap_cycles": max(loads) - min(loads),
            "useful_macs": useful, "issued_macs": issued, "padding_macs": issued - useful,
            "issues": issues, "spatial_utilization": useful / issued,
            "wall_mac_utilization": useful / (BUDGET * cycles), "assignments": allocations}


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(canonical(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(first), lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def bootstrap_pair(candidate, baseline, workloads, draws=4000):
    # Pair windows and stratify by batch; small dev set is not resampled for selection.
    rng = np.random.default_rng(20261006)
    logs = np.log(np.asarray(baseline) / np.asarray(candidate))
    batches = np.array([w["batch"] for w in workloads])
    samples = np.zeros(draws)
    for b in sorted(set(batches)):
        ids = np.flatnonzero(batches == b)
        picks = rng.choice(ids, size=(draws, len(ids)), replace=True)
        samples += logs[picks].mean(axis=1) / len(set(batches))
    lo, hi = np.quantile(np.exp(samples), [.025, .975])
    return [float(lo), float(hi)]


def summary_rows(named_results, workloads):
    rows = []
    batches = sorted({w["batch"] for w in workloads})
    for label, results in named_results.items():
        for b in batches:
            points = [r for r in results if r["batch"] == b]
            cycles = np.array([p["cycles"] for p in points])
            useful, issued = sum(p["useful_macs"] for p in points), sum(p["issued_macs"] for p in points)
            rows.append({"label": label, "geometry": points[0]["geometry"], "batch": b,
                         "windows": len(points), "mean_cycles": float(cycles.mean()),
                         "compute_lower_bound_ms": float(np.mean([p['useful_compute_lower_bound_cycles'] for p in points]) / 1e6),
                         "mean_ms_assumed_1GHz": float(cycles.mean() / 1e6),
                         "p95_ms_assumed_1GHz": float(np.quantile(cycles, .95, method="higher") / 1e6),
                         "useful_MACs": useful, "issued_MACs": issued,
                         "padding_MACs": issued - useful, "spatial_utilization": useful / issued,
                         "wall_mac_utilization": useful / (BUDGET * int(cycles.sum())),
                         "mean_core_finish_gap_cycles": float(np.mean([p["core_finish_gap_cycles"] for p in points]))})
    return rows


def render_report(out, selection, rows, comparisons, sensitivity):
    text = ["# 纯计算三维形状搜索：已完成\n",
            "本次没有 HBM、DMA、SRAM 端口、Router、SiLU、路由汇合或模型全流程计时。只计三个专家 GEMM，保留 Gate/Up→Down 依赖、K 累加依赖和尾块。主实验不限制输出存储。\n",
            "统一尺寸顺序 PM×PN×PK；逻辑矩阵 X[Me,K]×W[K,N]。BF16 输入/权重，FP32 部分和。总乘法器 12,288，不宣称等面积。\n",
            f"枚举 {selection['counts']['total']:,} 组：单核 {selection['counts']['single']}，同构 {selection['counts']['homogeneous']}，异构 {selection['counts']['heterogeneous']:,}。M/N 不设额外上限，PK={KS}。\n",
            "开发集 18 个窗口，留出集 135 个窗口；开发集每个 batch 等权，窗口内等权。留出集历史上已在其他实验使用，并非新的盲测。\n",
            "三类均用相同逻辑工作量降序＋预计最早完成派工。整专家归一个核，不跨专家拼成一个 GEMM，不拆专家给两核。派工零成本。\n",
            "主实验不限制输出存储，Gate 与 Up 独立输出可同时排入发射流；每个输出的后续 K 段仍等待先前提交。点积延迟取固定 20 拍或每改变一层 K 归约树改变 1/2 拍，三种主假设均全空间重搜，按相对本族最优的最坏开发集退化选一个稳健配置。所有配置发射间隔 1 拍、提交 1 拍。\n",
            "每核仅保留 8/16/32 个输出记录的九组结果为另外的有限缓冲诊断，不参与主实验选型。主表使用固定 dot20、理想输出存储。ms 仅为假设 1GHz 时 cycles/10^6 的换算，未校准为真实芯片时钟。\n",
            "| 类别 | 冻结的稳健形状 | 总乘法器 | 最坏开发集退化 | 主时序下单独最优形状 |\n|---|---|---:|---:|---|"]
    for name in FAMILIES:
        s = selection['families'][name]
        text.append(f"| {name} | `{s['robust_geometry']}` | 12,288 | {(s['worst_regret']-1)*100:.2f}% | `{s['nominal_best_geometry']}` |")
    text += ["\n| Batch | 理论有效 MAC 下界 ms | 单核纯计算 ms | 同构纯计算 ms | 异构纯计算 ms | 同构/单核加速 | 异构/单核加速 |\n|---|---:|---:|---:|---:|---:|---:|"]
    index = {(r['label'],r['batch']):r for r in rows}
    for b in sorted({r['batch'] for r in rows}):
        a,h,g = [index[("robust_"+n,b)]['mean_ms_assumed_1GHz'] for n in FAMILIES]
        lower = index[("robust_single",b)]['compute_lower_bound_ms']
        text.append(f"| B{b} | {lower:.6f} | {a:.6f} | {h:.6f} | {g:.6f} | {a/h:.3f}× | {a/g:.3f}× |")
    text += ["\n同一计算口径下，原始形状和搜索结果的对照：\n",
             "| Batch | 原始单核 6×4×512 | 原始同构 3×4×512 两核 | 原始异构 4/2×4×512 |\n|---|---:|---:|---:|"]
    for b in sorted({r['batch'] for r in rows}):
        vals = [index[("fixed_"+n,b)]['mean_ms_assumed_1GHz'] for n in FAMILIES]
        text.append(f"| B{b} | {vals[0]:.6f} ms | {vals[1]:.6f} ms | {vals[2]:.6f} ms |")
    text += ["\n| 留出集比较（batch 等权几何平均） | 加速 | 95% bootstrap 区间 |\n|---|---:|---|"]
    for c in comparisons:
        text.append(f"| {c['candidate']} / {c['baseline']} | {c['speedup']:.4f}× | {c['ci95'][0]:.4f}–{c['ci95'][1]:.4f}× |")
    text.append("\n区间仅描述按 batch 分层的窗口重采样变化；捕获窗口可能相关，不覆盖模型误差、时序假设、其他模型或芯片实现的不确定性。不能凭 0.3% 的差距宣布硬件异构胜出。\n")
    vs_single = next(c for c in comparisons if c['candidate']=='robust_heterogeneous' and c['baseline']=='robust_single')
    vs_homo = next(c for c in comparisons if c['candidate']=='robust_heterogeneous' and c['baseline']=='robust_homogeneous')
    text.append(f"\n结论：在这批路由和共同计算契约下，稳健异构配置的 batch 等权几何平均延迟比最优单核低 {(1-1/vs_single['speedup'])*100:.2f}%，比最优同构低 {(1-1/vs_homo['speedup'])*100:.2f}%。B2/B4 同构更快，其他 batch 异构更快。三个最优配置的 PM 都是 1；异构的主要差别是 PN/PK，算力接近均分。该结果支持重新选择计算几何，尚不支持明显的大小核性能优势。\n")
    text += ["\n时序与有限缓冲诊断的冻结配置留出集结果：\n",
             "| 场景 | 同构相对单核 | 异构相对单核 | 异构相对同构 |\n|---|---:|---:|---:|"]
    for s in sensitivity:
        text.append(f"| {s['scenario']} | {s['homogeneous_vs_single']:.3f}× | {s['heterogeneous_vs_single']:.3f}× | {s['heterogeneous_vs_homogeneous']:.3f}× |")
    text += ["\n限制：这是声明了执行规则的分析模型最优，不是所有硬件/所有调度方法的全局最优。独立输出可以流水重叠，但同一输出的后续 K 段必须等提交。主实验对整个投影按 K 段遍历独立输出，不额外引入小组退休边界；有限记录诊断才按组退休。不以 SRAM 容量筛选形状，逐任务输出报告所需 FP32 空间。未建模时钟下降、物理布线、面积、端口或供数。PK 改变浮点归约顺序；本次计数/调度正确性不代表预训练模型精度验证。\n"]
    (out / "REPORT_ZH.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dev_paths = [args.inputs / "development.json", args.inputs / "mixed_development.json"]
    dev = [w for p in dev_paths for w in json.loads(p.read_text())["workloads"]]
    verify_capture(dev)
    shapes, indices, fam = domain()
    counts = {name: int(np.count_nonzero(fam == i)) for i, name in enumerate(FAMILIES)}
    counts["total"] = len(indices)
    scenarios = [(f"{name}_ideal", name, 0) for name in LATENCY_NAMES]
    diagnostic_scenarios = [(f"{name}_percore_r{q}", name, q)
                            for name in LATENCY_NAMES for q in RESIDENT_RECORDS if q]
    contract = {"scope": "analytical pure GEMM compute; not native HBM, RTL, or full model inference",
                "counts": counts, "distinct_core_shapes": len(shapes), "PK": KS,
                "M_N_bounds": "all positive integers satisfying exact total multiplier budget",
                "main_multiplier_budget": BUDGET, "nominal": NOMINAL,
                "output_residency": "no bound in primary; finite 8/16/32 records per core in separate diagnostics",
                "timings": {n: dict(timing(n).latency_by_pk) for n in LATENCY_NAMES},
                "II": 1, "commit_cycles": 1, "clock": "assumed 1GHz, unvalidated",
                "selection": "development batch-balanced paired geometric latency ratios; minimax within-family regret across three compute timing profiles, no finite-buffer filter",
                "scheduler": "whole-expert logical-work LPT followed by EFT; common zero-cost rule, no N splitting or cross-expert packing",
                "excluded": ["HBM", "DMA", "SRAM timing/capacity/banks", "control costs", "Router", "SiLU", "combine", "learned predictor"],
                "development_hashes": {p.name: sha(p) for p in dev_paths},
                "source_hashes": {name: sha(Path(__file__).with_name(name)) for name in
                                  ('robust_compute.py','compute.py','test_compute.py','test_robust_compute.py','study.py')},
                "numpy_version": np.__version__,
                "development_counts": dict(sorted(Counter(w['batch'] for w in dev).items())),
                "heldout_warning": "frozen split, historically exposed captures; not pristine blind generalization",
                "numerical_warning": "PK changes reduction grouping; no pretrained accuracy verification"}
    write_json(out / "PREREGISTERED.json", contract)
    signatures = sorted({(e['Me'],e['H'],e['F']) for w in dev for e in w['experts']})
    scores, nominal_cycles = [], None
    anchor = (Shape(6,4,512),)
    began = time.monotonic()
    for scenario, name, q in scenarios:
        c_single = costs(shapes, signatures, name, q)[0]
        c_dual = costs(shapes, signatures, name, q)[0]
        cycles = vector_schedule(dev, signatures, c_single, c_dual, indices)
        # Repeat the complete search scheduling, not merely the selected points.
        repeated = vector_schedule(dev, signatures, c_single, c_dual, indices)
        assert np.array_equal(cycles, repeated), "non-deterministic complete sweep"
        useful_per_window = np.array([sum(3*e['Me']*e['H']*e['F'] for e in w['experts']) for w in dev])
        assert np.all(cycles * BUDGET >= useful_per_window), "schedule exceeds installed peak arithmetic rate"
        anchor_cycles = np.array([scalar_layer(w, anchor, name, q)['cycles'] for w in dev])
        scores.append(balanced_score(cycles, anchor_cycles, dev))
        if scenario == NOMINAL:
            nominal_cycles = cycles.copy()
        print(f"{scenario}: {len(indices):,} geometries x {len(dev)} windows x 2; {time.monotonic()-began:.1f}s", flush=True)
    scores = np.stack(scores)
    selected, regrets = minimax_selection(scores, fam)
    nominal_idx = next(i for i,s in enumerate(scenarios) if s[0] == NOMINAL)
    def cores_at(i):
        return tuple(shapes[j] for j in indices[i] if j >= 0)
    selection = {"counts": counts, "families": {}, "frozen_before_heldout_loaded": True}
    nominal_selected = {}
    for fi, name in enumerate(FAMILIES):
        ids = np.flatnonzero(fam == fi)
        nominal_best = int(ids[np.argmin(scores[nominal_idx, ids])])
        nominal_selected[name] = nominal_best
        best = selected[name]
        selection["families"][name] = {"robust_geometry": gid(cores_at(best)),
                                       "worst_regret": float(regrets[:, best].max()),
                                       "nominal_best_geometry": gid(cores_at(nominal_best)),
                                       "nominal_ratio_to_anchor": float(scores[nominal_idx, best]),
                                       "scenario_regrets": {s[0]: float(regrets[i,best]) for i,s in enumerate(scenarios)}}
    write_json(out / "FROZEN_SELECTION.json", selection)
    for i in set(selected.values()) | set(nominal_selected.values()):
        scalar = [scalar_layer(w, cores_at(i))['cycles'] for w in dev]
        assert np.array_equal(nominal_cycles[i], scalar), "accelerated search differs from scalar reference"
    np.savez_compressed(out / "complete_development_search.npz", indices=indices, family=fam,
                        shapes=np.array([(c.pm,c.pn,c.pk) for c in shapes]), scores=scores, regrets=regrets,
                        nominal_cycles=nominal_cycles, scenarios=np.array([s[0] for s in scenarios]),
                        development_ids=np.array([w['id'] for w in dev]))
    write_csv(out / "all_geometries_development.csv", (
        {"geometry": gid(cores_at(i)), "family": FAMILIES[int(fam[i])],
         "nominal_ratio_to_anchor": float(scores[nominal_idx,i]),
         "worst_within_family_regret": float(regrets[:,i].max()),
         "geometric_within_family_regret": float(np.exp(np.log(regrets[:,i]).mean()))}
        for i in range(len(indices))))
    write_json(out / "NEAR_OPTIMAL.json", {
        name: [{"geometry": gid(cores_at(int(i))), "worst_regret": float(regrets[:,i].max()),
                "nominal_ratio": float(scores[nominal_idx,i])} for i in
               sorted(np.flatnonzero(fam == fi), key=lambda i: (regrets[:,i].max(),i))
               if regrets[:,i].max() <= regrets[:,selected[name]].max() * 1.01]
        for fi,name in enumerate(FAMILIES)})
    # This is the first held-out load; neither selection nor near-optimal tier changes below.
    held_paths = [args.inputs / "heldout.json", args.inputs / "mixed_heldout.json"]
    held = [w for p in held_paths for w in json.loads(p.read_text())["workloads"]]
    verify_capture(held)
    assert not ({w['id'] for w in held} & {w['id'] for w in dev})
    configurations = {"robust_"+name: cores_at(i) for name,i in selected.items()}
    configurations.update({"nominal_best_"+name: cores_at(i) for name,i in nominal_selected.items()})
    configurations.update({"fixed_single": anchor,
                           "fixed_homogeneous": (Shape(3,4,512),Shape(3,4,512)),
                           "fixed_heterogeneous": (Shape(2,4,512),Shape(4,4,512))})
    named_results = {}
    for label, cores in configurations.items():
        results = [scalar_layer(w, cores) for w in held]
        assert canonical(results) == canonical([scalar_layer(w, cores) for w in held])
        named_results[label] = results
        write_json(out / f"heldout_{label}.json", results)
    rows = summary_rows(named_results, held)
    write_csv(out / "heldout_by_batch.csv", rows)
    comparisons = []
    for candidate, baseline in [("robust_homogeneous","robust_single"),
                                ("robust_heterogeneous","robust_single"),
                                ("robust_heterogeneous","robust_homogeneous")]:
        c = [r['cycles'] for r in named_results[candidate]]
        b = [r['cycles'] for r in named_results[baseline]]
        speedup = float(1 / balanced_score(np.array([c]), np.array(b), held)[0])
        comparisons.append({"candidate":candidate,"baseline":baseline,"speedup":speedup,
                            "ci95":bootstrap_pair(c,b,held)})
    write_json(out / "heldout_comparisons.json", comparisons)
    sensitivity = []
    for scenario, name, q in scenarios + diagnostic_scenarios:
        values = {f: [scalar_layer(w, cores_at(i), name, q)['cycles'] for w in held] for f,i in selected.items()}
        repeated = {f: [scalar_layer(w, cores_at(i), name, q)['cycles'] for w in held] for f,i in selected.items()}
        assert values == repeated, "non-deterministic frozen timing diagnostic"
        sensitivity.append({"scenario": scenario,
            "homogeneous_vs_single": float(1 / balanced_score(np.array([values['homogeneous']]), values['single'], held)[0]),
            "heterogeneous_vs_single": float(1 / balanced_score(np.array([values['heterogeneous']]), values['single'], held)[0]),
            "heterogeneous_vs_homogeneous": float(1 / balanced_score(np.array([values['heterogeneous']]), values['homogeneous'], held)[0])})
    write_json(out / "heldout_sensitivity.json", sensitivity)
    render_report(out, selection, rows, comparisons, sensitivity)
    write_json(out / "COMPLETION.json", {"full_search_repeats":2,"heldout_repeats":2,
        "development_windows":len(dev),"heldout_windows":len(held),"scenarios":len(scenarios),
        "frozen_sensitivity_repeats":2,"frozen_sensitivity_scenarios":len(scenarios+diagnostic_scenarios),
        "search_window_simulations":2*len(indices)*len(dev)*len(scenarios),
        "heldout_hashes":{p.name:sha(p) for p in held_paths},"elapsed_seconds":time.monotonic()-began})
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
