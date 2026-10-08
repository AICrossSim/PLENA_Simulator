"""第三轮冻结后的派工、预测器及按事件交集计量的独立评测。

结果是 phase-fluid 解析模型；所有资源占用均可重叠，不能相加为墙钟。
本模块只写 round3；第二轮代码、输入和冻结结果始终只读。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..round2.common import canonical, inputs, sha, write_csv, write_json, gmean, paired_ci
from ..round2.predictors import Predictor

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "round2"
MODES = ("pipelined", "port_tight")
BATCHES = (2, 4, 8, 16, 64, 96, 128)
NAMES = ("B0", "B1", "B2", "H51", "H42", "H33", "fixed_4+2")
METHODS = ("nominal", "random", "static", "btb", "ema", "ours")
TCONFIGS = tuple((t, flag) for flag in (False, True) for t in (2, 3, 4, 6, 8))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def save_gzip(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
            f.write((canonical(x) + "\n").encode())


def load_gzip(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def interval_union(intervals):
    """先合并同核等待区间，避免多个描述符重复计空转。"""
    out = []
    for lo, hi in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
        else:
            out.append((lo, hi))
    return out


def intersect_length(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def flow_metrics(result, workload, design):
    """时间积分；在途量为 rate×信用回收延迟的流体估计，非请求轨迹。"""
    wall = result["cycles"]
    n = len(result["core_finish_cycles"])
    big = max(range(n), key=lambda c: (design.cores[c].macs, design.cores[c].pm, -c))
    single_t = single_b = shared_t = shared_b = 0.0
    inflight_area = [0.0] * n
    # Segment annotations are router Expert IDs (-1 for Shared), not array indices.
    shared_indices = {e.get("id", i) for i, e in enumerate(workload["experts"]) if e.get("is_shared", False)}
    for s in result.get("segments", []):
        dt = s["end"] - s["start"]
        q = s["inflight_bytes"]
        for c in range(n):
            inflight_area[c] += q[c] * dt
        if sum(v > 1e-9 for v in q) == 1:
            single_t += dt
            single_b += s["hbm_rate_Bpc"] * dt
        if (s.get("run_expert", [None] * n)[big] in shared_indices
                and s.get("active_run", [False] * n)[big]):
            shared_t += dt
            shared_b += s["hbm_rate_Bpc"] * dt
    return {"single_fetcher_time_pct": 100 * single_t / wall if wall else 0.0,
            "single_fetcher_GBps": single_b / single_t if single_t else 0.0,
            "avg_inflight_KiB_core0": inflight_area[0] / wall / 1024 if wall else 0.0,
            "avg_inflight_KiB_core1": inflight_area[1] / wall / 1024 if n == 2 and wall else 0.0,
            "shared_fetch_GBps": shared_b / shared_t if shared_t else 0.0,
            "single_fetcher_cycles": single_t, "single_fetcher_bytes": single_b,
            "shared_cycles": shared_t, "shared_bytes": shared_b,
            "inflight_area_Bcycles": inflight_area}


def prediction_samples(result, workload, design, params):
    """时长误差与前瞻到达；stall 是等待∩该核无执行 actor 的精确区间交集。"""
    tasks = result["tasks"]
    samples = []
    waits = [[] for _ in design.cores]
    for t in tasks:
        actual = t["actual_cycles"]
        if actual <= 0:
            raise AssertionError("任务实际耗时必须为正")
        sample = {"expert_index": t["expert_index"], "core": t["core"],
                  "predicted_cycles": t["predicted_cycles"], "actual_cycles": actual,
                  "error_fraction": abs(t["predicted_cycles"] - actual) / actual,
                  "first_weight_ready": t.get("first_weight_ready"),
                  "current_end": t.get("current_end"), "eligible_next": False}
        ready, end = t.get("first_weight_ready"), t.get("current_end")
        if ready is not None and end is not None:
            sample["eligible_next"] = True
            late = ready - end
            from .model import task_cost
            index = t.get("original_expert_index", t["expert_index"])
            phase = task_cost(workload["experts"][index], design, t["core"], params).phases[0]
            # 同任务第一投影的纯计算量摊到各权重加载块；包含该任务的
            # M复用/尾块/点积树排空，不包含HBM、SRAM和控制时间。
            block_compute = phase.compute_cycles / phase.weight_loads
            sample.update(block_compute_cycles=block_compute, late_cycles=max(0.0, late),
                          late=late > 0, late_gt_64=late > 64, late_gt_256=late > 256)
            for w in (2, 4, 8):
                sample[f"success_at_{w}"] = -w * block_compute <= late <= 0
            if late > 0:
                waits[t["core"]].append((end, ready))
        samples.append(sample)
    idle_wait = 0.0
    for c, ws in enumerate(waits):
        union = interval_union(ws)
        # Segments include idle response and actor execution boundaries.
        for segment in result.get("segments", []):
            if not segment["active_run"][c]:
                idle_wait += sum(intersect_length(iv, (segment["start"], segment["end"])) for iv in union)
    return samples, idle_wait


def summarize_result(w, d, p, r):
    from .model import task_cost
    refetch = []
    shared = []
    for t in r["tasks"]:
        i = t.get("original_expert_index", t["expert_index"])
        e = w["experts"][i]
        co = task_cost(e, d, t["core"], p)
        factor = co.hbm_bytes / co.unique_hbm_bytes
        if factor > 1:
            refetch.append({"expert_index": i, "expert_id": e.get("id", i), "Me": e["Me"],
                            "shared": bool(e.get("is_shared", False)), "core": t["core"],
                            "refetch_factor": factor, "z_chunks": co.z_chunks, "spill_bytes": co.spill_bytes})
        if e.get("is_shared", False):
            shared.append({"expert_index": i, "core": t["core"], "start_ms": t["start"] / 1e6,
                           "finish_ms": t["finish"] / 1e6, "refetch_factor": factor})
    return {"window_id": w["id"], "batch": w["batch"], "cycles": r["cycles"],
            "latency_ms": r["latency_ms"], "hbm_bytes": r["hbm_bytes"],
            "refetch_tasks": len(refetch), "refetch_details": refetch, "shared_tasks": shared,
            "hbm_busy_frac": r["hbm_busy_frac"], "core_finish_cycles": r["core_finish_cycles"],
            "finish_gap_ms": r["core_finish_gap"] / 1e6,
            "core_compute_busy": r["core_compute_busy"], "w_port_busy": r["w_port_busy"],
            "x_port_busy": r["x_port_busy"], "acc_port_busy": r["acc_port_busy"],
            "idle_frac": r["idle_frac"], **flow_metrics(r, w, d)}


def encode(d):
    return asdict(d)


def decode(x):
    from .model import Design, Core
    x = dict(x)
    x["cores"] = tuple(Core(**c) if isinstance(c, dict) else Core(*c) for c in x["cores"])
    return Design(**x)


def old_designs(mode):
    manifest = json.loads((OLD / "dispatch_fix/frozen_designs.json").read_text())
    return {name: decode(x) for name, x in manifest["modes"][mode].items()}


def frozen_designs(mode, constraints="C0"):
    selection = json.loads((HERE / "E4/selected_designs.json").read_text())
    ds = {k: decode(v) for k, v in selection["modes"][mode][constraints].items()}
    old = old_designs(mode)
    ds.update(B0=old["B0"], **{"fixed_4+2": old["fixed_4+2"]})
    return ds


def configured_params(mode, credits=520):
    from .model import Parameters
    return Parameters(onchip_mode=mode, credits=credits)


class NominalPredictor(Predictor):
    """和第二轮 fair nominal 一致：有进度回调，无历史和 ETA 学习。"""
    def __init__(self):
        super().__init__("static")
        self.name = "nominal"

    def predict(self, e, c, nominal, **kw):
        self.calls += 1
        return nominal

    def on_complete(self, e, c, predicted, actual, nominal=None):
        pass

    def on_progress(self, e, c, elapsed, quarter, remaining):
        return None

    def state_bits(self):
        return 0


def make_predictor(method):
    return NominalPredictor() if method == "nominal" else Predictor(method)


def state_digest(pred):
    # Estimator tables contain tuple keys; preserve their exact deterministic
    # representation rather than lossy string-key conversion.
    return digest(repr({**{k: v for k, v in vars(pred).items() if k != "rng"},
                        "rng": pred.rng.getstate()})) if pred is not None else digest(None)


def freeze_sources():
    files = [HERE / name for name in ("model.py", "runtime.py", "config.py", "optimizer.py", "evaluations.py", "ablation.py")]
    files += [OLD / "results/E0/frozen_inputs.json", OLD / "dispatch_fix/frozen_designs.json"]
    files += [OLD / "predictors.py", OLD / "common.py"]
    return {str(p): sha(p) for p in files}


def receipt(out, command, specs, before, elapsed, repeats, extra=None):
    after = {name: sha(name) for name in before}
    if before != after:
        raise AssertionError("运行期间源文件被修改，请重新运行")
    write_json(Path(out) / "RUN_RECEIPT.json", {
        "command": command, "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
        "configurations": specs, "source_sha256": before, "source_unchanged": True,
        "input_manifest_sha256": sha(OLD / "results/E0/frozen_inputs.json"),
        "repeat_checks": repeats, "scope": "BF16 路由后 MoE phase-fluid 解析估计；非原生 HBM、RTL 或全模型计时",
        **(extra or {})})


def dispatch_job(spec):
    from .runtime import simulate
    from .optimizer import solve_assignment
    mode, name, dispatch, constraints, t_big, large_first, credits = spec
    d = frozen_designs(mode, constraints)
    d = d[name]
    p = configured_params(mode, credits)
    ws = inputs()
    seq, warm, states, solver_receipts = [], [], [], []
    for repeat in range(2):
        pred = Predictor("ours") if len(d.cores) == 2 else None
        warm_results = []
        for w in ws["development"]:
            if dispatch == "milp":
                continue
            warm_results.append(simulate(w, d, p, dispatch=dispatch, t_big=t_big, large_first=large_first, predictor=pred))
        warm.append(digest(warm_results))
        rs, sols = [], []
        for w in ws["heldout"]:
            sol = solve_assignment(w, d, p) if dispatch == "milp" else None
            r = simulate(w, d, p, dispatch=dispatch, owners=None if sol is None else sol["owners"],
                         t_big=t_big, large_first=large_first, predictor=None if sol is not None else pred)
            rs.append(r)
            if sol is not None:
                sols.append({"window_id": w["id"], "status": sol["status"], "digest": digest(sol),
                             "owners": sol["owners"]})
        seq.append(rs)
        states.append(state_digest(pred))
        solver_receipts.append(sols)
    assert canonical(seq[0]) == canonical(seq[1]), (spec, "完整输出重复不一致")
    assert warm[0] == warm[1] and states[0] == states[1]
    assert canonical(solver_receipts[0]) == canonical(solver_receipts[1])
    common = {"design": name, "onchip_mode": mode, "dispatch": dispatch, "constraint_group": constraints,
              "t_big": t_big, "large_first": large_first, "credits": credits}
    rows = [{**common, **summarize_result(w, d, p, r), "result_digest": digest(r), "repeat_digest": digest(s)}
            for w, r, s in zip(ws["heldout"], *seq)]
    rec = {**common, "hardware": encode(d), "parameters": asdict(p), "warmup_digest": warm[0],
           "repeat_warmup_digest": warm[1], "result_digest": digest(seq[0]), "repeat_digest": digest(seq[1]),
           "predictor_state_digest": states[0], "repeat_predictor_state_digest": states[1],
           "solver_checks": solver_receipts[0], "exact_repeat": True}
    rawpath = HERE / "E5/dispatch/raw" / f"{constraints}_{mode}_{name}_{dispatch}_t{t_big}_{int(large_first)}_{credits}.json.gz"
    save_gzip(rawpath, seq[0])
    rec["raw_file"] = str(rawpath.relative_to(HERE))
    rec["raw_sha256"] = sha(rawpath)
    return rows, rec


def select_job(spec):
    from .runtime import simulate
    mode, name, t_big, flag = spec
    d = frozen_designs(mode)[name]
    p = configured_params(mode)
    ws = inputs()["development"]
    repeats = []
    for _ in range(2):
        pred = Predictor("ours") if len(d.cores) == 2 else None
        repeats.append([simulate(w, d, p, dispatch="fixed", t_big=t_big, large_first=flag, predictor=pred) for w in ws])
    assert canonical(repeats[0]) == canonical(repeats[1])
    # The reference does not learn from heldout or from another threshold candidate.
    oldpred = Predictor("ours") if len(d.cores) == 2 else None
    refs = [simulate(w, d, p, dispatch="eft_old", predictor=oldpred) for w in ws]
    return [{"design": name, "onchip_mode": mode, "t_big": t_big, "large_first": flag,
             "window_id": w["id"], "ratio": a["cycles"] / b["cycles"],
             "result_digest": digest(a), "repeat_digest": digest(c)}
            for w, a, b, c in zip(ws, repeats[0], refs, repeats[1])]


def choose_runtime(rows):
    ids = [w["id"] for w in inputs()["development"]]
    scores, logs = {}, {}
    for cfg in TCONFIGS:
        per = {wid: [] for wid in ids}
        for r in rows:
            if (r["t_big"], r["large_first"]) == cfg:
                per[r["window_id"]].append(math.log(r["ratio"]))
        if not all(per.values()):
            raise AssertionError("缺少开发集配对")
        logs[cfg] = [sum(per[wid]) / len(per[wid]) for wid in ids]
        scores[cfg] = math.exp(sum(logs[cfg]) / len(ids))
    chosen = min(TCONFIGS, key=lambda x: (scores[x], x[1], x[0]))
    rng = np.random.default_rng(20261008)
    counts = {cfg: 0 for cfg in TCONFIGS}
    for _ in range(200):
        js = rng.integers(0, len(ids), len(ids))
        pick = min(TCONFIGS, key=lambda x: (sum(logs[x][j] for j in js) / len(js), x[1], x[0]))
        counts[pick] += 1
    return {"chosen": {"t_big": chosen[0], "large_first": chosen[1]},
            "old126": {"t_big": 3, "large_first": True}, "bootstrap_draws": 200,
            "objective": "每个开发窗口跨设计及模式平均 log(fixed/eft_old)，再取几何平均",
            "scores": [{"t_big": x[0], "large_first": x[1], "ratio": scores[x], "bootstrap_count": counts[x]} for x in TCONFIGS],
            "development_window_ids": ids}


def dispatch_report(rows):
    out = HERE / "E5/dispatch"
    write_json(out / "per_window.json", rows)
    table = {(r["onchip_mode"], r["design"], r["dispatch"], r["window_id"]): r for r in rows if r["constraint_group"] == "C0"}
    compares, traffic, regression = [], [], []
    for mode in MODES:
        for name in NAMES:
            for dispatch in ("eft_old", "fixed", "milp"):
                rs = [table[mode, name, dispatch, w["id"]] for w in inputs()["heldout"]]
                row = {"design": name, "onchip_mode": mode, "dispatch": dispatch}
                for batch in BATCHES:
                    row[f"B{batch}"] = gmean(r["latency_ms"] for r in rs if r["batch"] == batch)
                row["all_geomean_ms"] = gmean(r["latency_ms"] for r in rs)
                row["ratio_vs_milp"] = gmean(r["cycles"] / table[mode, name, "milp", r["window_id"]]["cycles"] for r in rs)
                row["ratio_vs_B1_fixed"] = gmean(r["cycles"] / table[mode, "B1", "fixed", r["window_id"]]["cycles"] for r in rs)
                compares.append(row)
                for r in rs:
                    ref = table[mode, name, "milp", r["window_id"]]
                    traffic.append({"window_id": r["window_id"], "design": name, "onchip_mode": mode,
                                    "dispatch": dispatch, "hbm_MiB": r["hbm_bytes"] / 2**20,
                                    "hbm_MiB_milp": ref["hbm_bytes"] / 2**20,
                                    "extra_pct": 100 * (r["hbm_bytes"] / ref["hbm_bytes"] - 1), "refetch_tasks": r["refetch_tasks"]})
            if name in ("B0", "B1"):
                for w in inputs()["heldout"]:
                    a, b = table[mode, name, "fixed", w["id"]], table[mode, name, "eft_old", w["id"]]
                    regression.append({"window_id": w["id"], "design": name, "onchip_mode": mode,
                                       "cycles_exact": a["cycles"] == b["cycles"], "hbm_bytes_exact": a["hbm_bytes"] == b["hbm_bytes"],
                                       "result_digest_exact": a["result_digest"] == b["result_digest"]})
    assert all(r["cycles_exact"] and r["hbm_bytes_exact"] and r["result_digest_exact"] for r in regression), "单核派工回归不一致"
    write_csv(out / "compare.csv", compares)
    write_csv(out / "hbm_bytes.csv", traffic)
    write_csv(out / "regression.csv", regression)
    excess = [r for r in traffic if r["dispatch"] == "fixed" and r["extra_pct"] > 2 + 1e-9]
    write_csv(out / "excess_hbm_windows.csv", excess, list(traffic[0]))
    gpqa = "v3_captured_mixed_heldout_gpqa_t128_l13"
    md = ["# GPQA B128 派工案例\n", "520 个在途请求；解析供数上限 256 GB/s。时间为路由后 FFN，单位 ms。\n",
          "| 模式 | 设计 | 派工 | Shared 核/开始 ms | HBM MiB | 延迟 ms |", "|---|---|---|---|---:|---:|"]
    for r in rows:
        if r["constraint_group"] == "C0" and r["window_id"] == gpqa:
            sh = "; ".join(f"{t['core']}/{t['start_ms']:.6f}" for t in r["shared_tasks"])
            md.append(f"| {r['onchip_mode']} | {r['design']} | {r['dispatch']} | {sh} | {r['hbm_bytes']/2**20:.3f} | {r['latency_ms']:.6f} |")
    (out / "gpqa_t128_case.md").write_text("\n".join(md) + "\n")
    return compares, regression


def run_dispatch(jobs=4, constraints=("C0", "C1")):
    start = time.monotonic()
    before = freeze_sources()
    out = HERE / "E5/dispatch"
    out.mkdir(parents=True, exist_ok=True)
    selection_specs = [(m, n, t, flag) for m in MODES for n in NAMES for t, flag in TCONFIGS]
    dev_rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for spec, rs in zip(selection_specs, pool.map(select_job, selection_specs)):
            dev_rows.extend(rs)
            print({"E5_selection": spec, "windows": len(rs)}, flush=True)
    selected = choose_runtime(dev_rows)
    write_csv(out / "development_per_window.csv", dev_rows)
    write_json(out / "selection.json", selected)
    cfg = selected["chosen"]
    specs = [(m, n, dispatch, cg, cfg["t_big"], cfg["large_first"], 520)
             for cg in constraints for m in MODES for n in NAMES for dispatch in ("eft_old", "fixed", "milp")]
    rows, checks = [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for spec, (rs, rec) in zip(specs, pool.map(dispatch_job, specs)):
            rows.extend(rs)
            checks.append(rec)
            print({"E5_dispatch": spec, "GM_ms": gmean(r["latency_ms"] for r in rs)}, flush=True)
    # Preserve old threshold policy on the same new hardware when selection changed.
    if cfg != selected["old126"]:
        oldspecs = [(m, n, "fixed", "C0", 3, True, 520) for m in MODES for n in NAMES]
        legacy = []
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for spec, (rs, rec) in zip(oldspecs, pool.map(dispatch_job, oldspecs)):
                legacy.extend(rs); checks.append(rec)
        write_json(out / "old126_runtime_settings.json", legacy)
    save_gzip(out / "full_results_summary.json.gz", rows)
    dispatch_report(rows)
    e4_report(rows)
    write_json(out / "repeat_checks.json", checks)
    receipt(out, sys.argv, len(specs), before, time.monotonic() - start, checks,
            {"development_candidate_configs": len(selection_specs), "runtime_selection": selected["chosen"]})
    return rows


def e4_report(rows):
    """同一冻结运行同时给E4主表／资源积分／留出门槛，避免重复仿真。"""
    out = HERE / "E4"
    tables, breakdown, gates = [], [], []
    indices = {(r["constraint_group"], r["onchip_mode"], r["design"], r["dispatch"], r["window_id"]): r for r in rows}
    for cg in ("C0", "C1"):
        for mode in MODES:
            for name in NAMES:
                for dispatch in ("fixed", "milp"):
                    rs = [indices[cg, mode, name, dispatch, w["id"]] for w in inputs()["heldout"]]
                    row = {"constraint_group": cg, "onchip_mode": mode, "design": name, "dispatch": dispatch}
                    for batch in BATCHES:
                        row[f"B{batch}"] = gmean(r["latency_ms"] for r in rs if r["batch"] == batch)
                    row["all_geomean_ms"] = gmean(r["latency_ms"] for r in rs)
                    tables.append(row)
                    for batch in (*BATCHES, "all"):
                        br = [r for r in rs if batch == "all" or r["batch"] == batch]
                        wall = sum(r["cycles"] for r in br)
                        d = frozen_designs(mode, cg)[name]
                        item = {**{k: row[k] for k in ("constraint_group", "onchip_mode", "design", "dispatch")},
                                "batch": batch, "hbm_busy_pct": 100 * sum(r["hbm_busy_frac"] * r["cycles"] for r in br) / wall,
                                "W_port_busy_pct": 100 * sum(r["w_port_busy"] for r in br) / wall,
                                "X_port_busy_pct": 100 * sum(r["x_port_busy"] for r in br) / wall,
                                "acc_port_busy_pct": 100 * sum(r["acc_port_busy"] for r in br) / wall,
                                "idle_pct": 100 * sum(r["idle_frac"] * r["cycles"] for r in br) / wall,
                                "finish_gap_ms": sum(r["finish_gap_ms"] for r in br) / len(br)}
                        for c in range(2):
                            item[f"core{c}_compute_busy_pct"] = 100 * sum(r["core_compute_busy"][c] if c < len(d.cores) else 0 for r in br) / wall
                        breakdown.append(item)
            for name in ("H51", "H42", "H33"):
                for dispatch in ("fixed", "milp"):
                    rs = [indices[cg, mode, name, dispatch, w["id"]] for w in inputs()["heldout"]]
                    gr = {"constraint_group": cg, "onchip_mode": mode, "design": name, "dispatch": dispatch}
                    passed = True
                    for base in ("B1", "B2"):
                        ratios = [r["cycles"] / indices[cg, mode, base, dispatch, r["window_id"]]["cycles"] for r in rs]
                        lo, hi = paired_ci(ratios)
                        gr[f"ratio_vs_{base}"] = gmean(ratios)
                        gr[f"speedup_vs_{base}_95ci_lower_pct"] = 100 * lo
                        gr[f"speedup_vs_{base}_95ci_upper_pct"] = 100 * hi
                        passed = passed and gmean(ratios) <= .95
                    gr["enter_calibration"] = passed
                    gr["calibrated_architecture_win"] = False
                    gates.append(gr)
    write_csv(out / "heldout_main_table.csv", tables)
    for mode in MODES:
        write_csv(out / f"heldout_main_table_{mode}.csv", [r for r in tables if r["onchip_mode"] == mode])
    write_csv(out / "breakdown.csv", breakdown)
    write_csv(out / "gates.csv", gates)


def cross_job(spec):
    from .runtime import simulate
    source, mode, name, credits, setting = spec
    d = old_designs(mode)[name] if source == "126 下选出" else frozen_designs(mode)[name]
    p = configured_params(mode, credits)
    repeats, checks = [], []
    for _ in range(2):
        pred = Predictor("ours") if len(d.cores) == 2 else None
        warm = [simulate(w, d, p, dispatch="fixed",
                         predictor=pred, **setting) for w in inputs()["development"]]
        rs = [simulate(w, d, p, dispatch="fixed",
                       predictor=pred, **setting) for w in inputs()["heldout"]]
        repeats.append(rs); checks.append({"warm": digest(warm), "state": state_digest(pred)})
    assert canonical(repeats[0]) == canonical(repeats[1]) and checks[0] == checks[1]
    row = {"hardware_source": source, "onchip_mode": mode, "design": name, "credits": credits,
           "bw_GBps": p.hbm_bandwidth, "all_geomean_ms": gmean(r["latency_ms"] for r in repeats[0]),
           "geometry": d.geometry, "t_big": setting["t_big"], "large_first": setting["large_first"]}
    rawpath = HERE / "E4/cross_bw_raw" / f"{'old' if source=='126 下选出' else 'new'}_{mode}_{name}_{credits}.json.gz"
    save_gzip(rawpath, repeats[0])
    return row, {**row, "result_digest": digest(repeats[0]), "repeat_digest": digest(repeats[1]),
                 "checks": checks, "raw_file": str(rawpath.relative_to(HERE)), "raw_sha256": sha(rawpath), "exact_repeat": True}


def run_cross_bw(jobs=4):
    start = time.monotonic()
    before = freeze_sources()
    setting = json.loads((HERE / "E5/dispatch/selection.json").read_text())["chosen"]
    specs = []
    for mode in MODES:
        for credits in (256, 520):
            for name in ("B1", "B2", "best_hetero"):
                specs.append(("126 下选出", mode, name, credits, setting))
            for name in ("B1", "B2", "H51", "H42", "H33"):
                specs.append(("256 下重新搜索", mode, name, credits, setting))
    rows, checks = [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for spec, (row, rec) in zip(specs, pool.map(cross_job, specs)):
            rows.append(row); checks.append(rec)
            print({"E4_cross_bw": spec[:4], "GM_ms": row["all_geomean_ms"]}, flush=True)
    wide = []
    for source in ("126 下选出", "256 下重新搜索"):
        for mode in MODES:
            names = ("B1", "B2", "best_hetero") if source == "126 下选出" else ("B1", "B2", "H51", "H42", "H33")
            for name in names:
                vals = [r for r in rows if (r["hardware_source"], r["onchip_mode"], r["design"]) == (source, mode, name)]
                wide.append({"hardware_source": source, "onchip_mode": mode, "design": name,
                             "geometry": vals[0]["geometry"],
                             "at126_geomean_ms": next(r["all_geomean_ms"] for r in vals if r["credits"] == 256),
                             "at256_geomean_ms": next(r["all_geomean_ms"] for r in vals if r["credits"] == 520)})
    write_csv(HERE / "E4/cross_bw.csv", wide)
    write_csv(HERE / "E4/cross_bw_long.csv", rows)
    write_json(HERE / "E4/cross_bw_repeat_checks.json", checks)
    receipt(HERE / "E4/cross_bw", sys.argv, len(specs), before, time.monotonic() - start, checks)


def predictor_job(spec):
    from .runtime import simulate
    mode, name, method, credits, use_old_hw, selection = spec
    d = (old_designs(mode) if use_old_hw else frozen_designs(mode))[name]
    if use_old_hw:
        d = replace(d, flows=("WS",) * len(d.cores))
    p = configured_params(mode, credits)
    ws = inputs()
    repeats, states, warm_hashes = [], [], []
    for _ in range(2):
        pred = make_predictor(method)
        dispatch = "fixed_legacy" if use_old_hw else "fixed"
        warm = [simulate(w, d, p, dispatch=dispatch, predictor=pred, **selection) for w in ws["development"]]
        warm_hashes.append(digest(warm))
        repeats.append([simulate(w, d, p, dispatch=dispatch, predictor=pred, **selection) for w in ws["heldout"]])
        states.append(state_digest(pred))
    assert canonical(repeats[0]) == canonical(repeats[1]), (spec[:5], "预测器重复失败")
    assert warm_hashes[0] == warm_hashes[1] and states[0] == states[1]
    common = {"bw_GBps": p.hbm_bandwidth, "onchip_mode": mode, "design": name, "method": method,
              "hardware_source": "126 下选出" if use_old_hw else "256 下重新搜索",
              "dispatch_protocol": "fixed_legacy，共同WS，t3/true，复现7eb" if use_old_hw else "fixed，256开发集重新选择参数"}
    windows, tasks = [], []
    for w, r, rr in zip(ws["heldout"], *repeats):
        ts, idle_wait = prediction_samples(r, w, d, p)
        windows.append({**common, **summarize_result(w, d, p, r), "idle_wait_cycles": idle_wait,
                        "ncores": len(d.cores), "result_digest": digest(r), "repeat_digest": digest(rr)})
        tasks.extend({**common, "window_id": w["id"], "batch": w["batch"], **t} for t in ts)
    rec = {**common, "hardware": encode(d), "parameters": asdict(p), "runtime_settings": selection,
           "heldout_digest": digest(repeats[0]), "repeat_heldout_digest": digest(repeats[1]),
           "warmup_digest": warm_hashes[0], "repeat_warmup_digest": warm_hashes[1],
           "final_state_digest": states[0], "repeat_final_state_digest": states[1], "exact_repeat": True}
    rawpath = HERE / "E5/predictor/raw" / f"{credits}_{mode}_{name}_{method}.json.gz"
    save_gzip(rawpath, repeats[0])
    rec["raw_file"] = str(rawpath.relative_to(HERE))
    rec["raw_sha256"] = sha(rawpath)
    return windows, tasks, rec


def conditional_oracle_rows(windows, tasks):
    """只给 nominal 的既定行动计划填真实时长，不做反事实最优派工。

    所有结束、到达、HBM 字节和停顿原样保留。因而 oracle 延迟不是
    "完美预测器可实现的性能上界"，仅用于隔离时长估计误差。
    """
    win = [{**r, "method": "oracle", "oracle_scope": "nominal 固定行动计划的条件真实时长参考"}
           for r in windows if r["method"] == "nominal"]
    ts = [{**r, "method": "oracle", "predicted_cycles": r["actual_cycles"], "error_fraction": 0.0,
           "oracle_scope": "不改变归属、预取、供数及进度"} for r in tasks if r["method"] == "nominal"]
    return win, ts


def predictor_report(windows, tasks):
    out = HERE / "E5/predictor"
    identities = sorted({(r["bw_GBps"], r["onchip_mode"], r["design"]) for r in windows})
    table, accuracy_by_batch = [], []
    for bw, mode, name in identities:
        base = {r["window_id"]: r for r in windows if (r["bw_GBps"], r["onchip_mode"], r["design"], r["method"]) == (bw, mode, name, "nominal")}
        for method in (*METHODS, "oracle"):
            rs = [r for r in windows if (r["bw_GBps"], r["onchip_mode"], r["design"], r["method"]) == (bw, mode, name, method)]
            ts = [t for t in tasks if (t["bw_GBps"], t["onchip_mode"], t["design"], t["method"]) == (bw, mode, name, method)]
            nxt = [t for t in ts if t["eligible_next"]]
            row = {"bw_GBps": bw, "onchip_mode": mode, "design": name, "method": method}
            for batch in BATCHES:
                row[f"B{batch}"] = gmean(r["latency_ms"] for r in rs if r["batch"] == batch)
            row.update(all_geomean_ms=gmean(r["latency_ms"] for r in rs),
                       ratio_vs_no_pred=gmean(r["cycles"] / base[r["window_id"]]["cycles"] for r in rs),
                       MAE=sum(t["error_fraction"] for t in ts) / len(ts),
                       **{k: sum(t[k] for t in nxt) / len(nxt) if nxt else 0.0 for k in
                          ("success_at_2", "success_at_4", "success_at_8", "late", "late_gt_64", "late_gt_256")},
                       stall_pct=100 * sum(r["idle_wait_cycles"] for r in rs) / sum(r["ncores"] * r["cycles"] for r in rs))
            table.append(row)
            for batch in (*BATCHES, "all"):
                br = [r for r in rs if batch == "all" or r["batch"] == batch]
                bt = [t for t in ts if batch == "all" or t["batch"] == batch]
                bn = [t for t in bt if t["eligible_next"]]
                accuracy_by_batch.append({**{k: row[k] for k in ("bw_GBps", "onchip_mode", "design", "method")},
                    "batch": batch, "tasks": len(bt), "next_samples": len(bn),
                    "MAE": sum(t["error_fraction"] for t in bt) / len(bt),
                    **{k: sum(t[k] for t in bn) / len(bn) if bn else 0 for k in ("success_at_2", "success_at_4", "success_at_8", "late", "late_gt_64", "late_gt_256")},
                    "stall_pct": 100 * sum(r["idle_wait_cycles"] for r in br) / sum(r["ncores"] * r["cycles"] for r in br)})
    write_csv(out / "predictor_table.csv", table)
    write_csv(out / "accuracy_by_batch.csv", accuracy_by_batch)
    write_csv(out / "per_window.csv", windows)
    write_csv(out / "task_predictions.csv", tasks)
    md = ["# 预测器三方对照\n", "BF16；全部 135 个留出窗口；单位 ms；延迟取配对几何平均。\n",
          "`nominal` 为不学习的形状/容量成本估计，仍然使用容量感知派工与 Next 预取。有季度进度重新检查，但不更新 ETA。Next 的绑定由有限资源准入与 512 周期绑定提前量共同决定，不能靠 ETA 释放空间。\n",
          "oracle 是 nominal 固定行动计划的条件真实时长参考：其 MAE=0，但不改变任务归属、预取时机或延迟；不是最优派工，也不是收益上界。\n",
          "MAE 表中为百分比归一误差的比例值。success@W 的窗口=2/4/8×该Next任务第一投影纯计算周期/权重加载块数，含M复用、尾块、点积树排空，不含供数与控制。late 按具备 Current 结束与 Next 首块就绪两个时间戳的任务计。stall_pct 在等待区间与该核无执行 actor 区间取交集后计算，以总核周期为分母，不重复累加重叠等待。\n",
          "| 供数上限 GB/s | 模式 | 设计 | 方法 | GM ms | 对 nominal 比值 | MAE % | late % | 空转等待 % |",
          "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['bw_GBps']:.5f} | {r['onchip_mode']} | {r['design']} | {r['method']} | {r['all_geomean_ms']:.6f} | {r['ratio_vs_no_pred']:.6f} | {100*r['MAE']:.4f} | {100*r['late']:.4f} | {r['stall_pct']:.4f} |")
    md += ["\n预测准确与整层延迟是不同目标。若 ours 慢于 nominal，以下计数定位其伴随变化；它们是观测关联，不能直接相加或当成独立因果贡献。\n",
           "| 上限 GB/s | 模式 | 设计 | ours/nominal 时间 | ours−nominal HBM MiB（135窗口总计） | 重读任务增量 |",
           "|---:|---|---|---:|---:|---:|"]
    for bw, mode, name in identities:
        by = {method: [r for r in windows if (r["bw_GBps"], r["onchip_mode"], r["design"], r["method"]) == (bw, mode, name, method)] for method in ("nominal", "ours")}
        ratio = gmean(r["cycles"] / b["cycles"] for r, b in zip(by["ours"], by["nominal"]))
        delta = sum(r["hbm_bytes"] for r in by["ours"]) - sum(r["hbm_bytes"] for r in by["nominal"])
        refetch_delta = sum(r["refetch_tasks"] for r in by["ours"]) - sum(r["refetch_tasks"] for r in by["nominal"])
        md.append(f"| {bw:.5f} | {mode} | {name} | {ratio:.6f} | {delta/2**20:.3f} | {refetch_delta} |")
    (out / "PREDICTOR.md").write_text("\n".join(md) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, mode in zip(axs, MODES):
        rr = [r for r in table if r["bw_GBps"] == 256 and r["onchip_mode"] == mode and r["method"] != "oracle"]
        for name in sorted({r["design"] for r in rr}):
            vals = [next(r["ratio_vs_no_pred"] for r in rr if r["design"] == name and r["method"] == method) for method in METHODS]
            ax.plot(METHODS, vals, marker="o", label=name)
        ax.axhline(1, color="grey", linestyle="--")
        ax.set(title=mode, ylabel="Latency / nominal", xlabel="Estimator")
        ax.legend()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/fig_predictor.pdf")
    plt.close(fig)
    return table


def run_predictor(jobs=4):
    start = time.monotonic()
    before = freeze_sources()
    selection = json.loads((HERE / "E5/dispatch/selection.json").read_text())["chosen"]
    specs = []
    for credits, old in ((520, False), (256, True)):
        for mode in MODES:
            names = ("B1", "B2", "best_hetero") if old else ("B1", "B2", best_hetero_name(mode))
            for name in names:
                for method in METHODS:
                    specs.append((mode, name, method, credits, old, {"t_big": 3, "large_first": True} if old else selection))
    windows, tasks, checks = [], [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for spec, (rs, ts, rec) in zip(specs, pool.map(predictor_job, specs)):
            windows.extend(rs); tasks.extend(ts); checks.append(rec)
            print({"E5_predictor": spec[:5], "GM_ms": gmean(r["latency_ms"] for r in rs)}, flush=True)
    ow, ot = conditional_oracle_rows(windows, tasks)
    windows.extend(ow); tasks.extend(ot)
    out = HERE / "E5/predictor"
    out.mkdir(parents=True, exist_ok=True)
    save_gzip(out / "window_summaries.json.gz", windows)
    save_gzip(out / "task_predictions.json.gz", tasks)
    predictor_report(windows, tasks)
    write_json(out / "repeat_checks.json", checks)
    receipt(out, sys.argv, len(specs), before, time.monotonic() - start, checks,
            {"oracle": "nominal 固定计划，仅真实时长 accuracy 参考，不是反事实最优派工"})


def best_hetero_name(mode):
    selected = json.loads((HERE / "E4/selected_designs.json").read_text())
    return selected["best_hetero_by_mode"][mode]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("dispatch", "predictor", "cross", "all"), default="all")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.stage in ("dispatch", "all"):
        run_dispatch(args.jobs)
    if args.stage in ("predictor", "all"):
        run_predictor(args.jobs)
    if args.stage in ("cross", "all"):
        run_cross_bw(args.jobs)


if __name__ == "__main__":
    main()
