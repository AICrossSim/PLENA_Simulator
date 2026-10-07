"""Evidence-only Chinese report and auditable delivery status for round two.

This module reads the completed artifacts; it never runs a performance model,
fills missing timing, silently changes a baseline, or closes an open proof.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("pipelined", "port_tight", "fixed_issue")
BATCHES = (2, 4, 8, 16, 64, 96, 128)
BASELINES = ("B0", "B1", "B2")
FIGURES = ("headroom", "main_bars", "breakdown", "bnb_coverage", "workload_map",
           "sobol", "flip_boundary", "dataflow_grid", "me_crossover", "predictor")
REQUIRED = {
    "1": ("BRANCHES_BEFORE.md", "MERGE_LOG.md", "BRANCHES_AFTER.md", "results/E0/reproduce_check.csv"),
    "2": ("results/E0/frozen_inputs.json", "results/E0/SOURCES.md"),
    "3": tuple("results/E1/" + x for x in ("bounds_per_window.csv", "headroom_by_batch.csv", "SUMMARY.md")),
    "4": tuple("results/E2/" + x for x in ("micro.csv", "layer_grid.csv", "SUMMARY.md")),
    "5.1": tuple("results/E3/" + x for x in ("bnb_certificate.csv", "bnb_leaves.csv", "bnb_summary.json", "lb_validity.csv")),
    "5.2": ("results/E3/schedule_gaps.csv",),
    "5.3": ("results/E3/workload_map.csv", "results/E3/workload_extreme.json"),
    "5.4": ("results/E3/robust_objectives.csv", "results/E3/selection_stability.csv"),
    "5.5": ("results/E3/sobol.csv", "results/E3/flip_boundary.csv"),
    "6": tuple("results/E4/" + x for x in ("heldout_main_table.csv", "breakdown.csv", "hbm512_sensitivity.csv", "SUMMARY.md")),
    "7": tuple("results/E5/" + x for x in ("dispatch_table.csv", "predictor_table.csv", "dispatcher_state_bits.csv", "SUMMARY.md")),
    "8": tuple("results/E6/" + x for x in ("moe_layer_e2e.csv", "model_token_e2e.csv", "SUMMARY.md")),
    "9–10": ("REPORT_ZH.md",) + tuple("figures/fig_" + x + "." + ext for x in FIGURES for ext in ("pdf", "png")),
}
HEADINGS = (
    "结论", "设定与等资源账本", "下限与余量", "组会第一张表与时间分解", "证明式 DSE",
    "负载区域图", "稳健性", "敏感性", "数据流 3×3 表", "分派与预测器", "端到端", "局限", "下一步",
)


def truth(value):
    return value is True or str(value).lower() in ("true", "1", "yes")


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def fmt(value, places=4):
    x = number(value)
    return "缺失" if x is None else f"{x:.{places}f}"


def gm(values):
    values = [number(x) for x in values]
    if not values or any(x is None or x <= 0 for x in values):
        return None
    return math.exp(sum(math.log(x) for x in values) / len(values))


def table(headers, rows):
    def cell(x):
        return str(x if x is not None else "缺失").replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def git_value(root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "未记录"


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.errors = []
        self.files = {}

    def path(self, relative):
        return self.root / relative

    def rows(self, relative):
        p = self.path(relative)
        if not p.is_file():
            return []
        try:
            with p.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.files[relative] = {"rows": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            return rows
        except (OSError, csv.Error, UnicodeError) as error:
            self.errors.append(f"{relative}: {error}")
            return []

    def data(self, relative):
        p = self.path(relative)
        if not p.is_file():
            return {}
        try:
            value = json.loads(p.read_text())
            self.files[relative] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            return value
        except (OSError, ValueError) as error:
            self.errors.append(f"{relative}: {error}")
            return {}

    def text(self, relative):
        p = self.path(relative)
        return p.read_text() if p.is_file() else ""


def proof_runs(e):
    summary = e.data("results/E3/bnb_summary.json")
    runs = summary.get("runs", [])
    if runs:
        return runs
    return [{"onchip_mode": mode, "proof": proof, "result": e.data(f"results/E3/bnb_{mode}_{proof}.json")}
            for mode in MODES for proof in ("A", "B") if e.path(f"results/E3/bnb_{mode}_{proof}.json").exists()]


def test_receipts(e):
    latest = {}
    for path in sorted(e.path("results/E0/tests").glob("*/UNIT_CHECKS.json")):
        item = e.data(str(path.relative_to(e.root)))
        for suite in item.get("suites", []):
            name = suite.get("suite", "unknown")
            latest[name] = {**suite, "receipt": str(path.relative_to(e.root)), "commit": item.get("commit_at_end")}
    # Some repaired analytical suites have a standalone XML receipt rather
    # than a newer combined runner receipt. Select only the newest such file.
    standalone = list(e.path("results/E0/tests").glob("analytical*.xml"))
    if standalone:
        path = max(standalone, key=lambda p: p.stat().st_mtime_ns)
        previous = e.path(latest.get("analytical", {}).get("receipt", "missing"))
        if not previous.is_file() or path.stat().st_mtime_ns > previous.stat().st_mtime_ns:
            try:
                tree = ET.parse(path).getroot()
                suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
                values = {k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
                latest["analytical"] = {**values, "returncode": 0 if not values["failures"] and not values["errors"] else 1,
                                         "receipt": str(path.relative_to(e.root)), "commit": "standalone XML; see E0 provenance"}
            except (ET.ParseError, OSError, ValueError) as error:
                e.errors.append(f"{path}: {error}")
    required = ("research", "analytical", "rust", "main_rust")
    complete = all(name in latest and latest[name].get("returncode") == 0 and
                   not latest[name].get("failures", 0) and not latest[name].get("errors", 0) for name in required)
    return {"required_suites": required, "latest_by_suite": latest, "all_required_recorded_suites_passed": complete}


def baseline_rows(e, mode, kind="runtime"):
    rows = e.rows("results/E4/heldout_main_table.csv")
    return {r["entry"]: r for r in rows if r.get("onchip_mode") == mode and r.get("sched_type") == kind and r.get("entry") in BASELINES}


def baseline_context(e, mode, kind="runtime"):
    data = baseline_rows(e, mode, kind)
    return [fmt(data.get(name, {}).get("all_geomean")) for name in BASELINES]


def hardware_label(entry, certified=False):
    names = {"B0": "B0 原形状参考", "B1": "B1 开发集调优单核", "B2": "B2 开发集调优同构",
             "fixed_3+3": "固定 3+3", "fixed_4+2": "固定 4+2", "previous_asym": "上一轮异构",
             "U1": "U1 逐专家形状诊断", "U2": "U2 一次 SRAM 权重读诊断"}
    if entry in names:
        return names[entry]
    if entry == "best_hetero":
        return "异构族最优（模型证明已闭合）" if certified else "异构已评估候选（全族证明未闭合）"
    if entry.startswith("best_"):
        return entry.removeprefix("best_") + (" 族最优" if certified else " 已评估候选")
    return entry


def resume_command(e, mode, proof, args):
    source = f"results/E3/bnb_{mode}_{proof}.json"
    target = f"results/E3/resumed_{mode}_{proof}.json"
    # Keep the original certificate intact; a resumed result is a new file.
    code = ("import json; from pathlib import Path; from research.moe_dispatch.round2.common import inputs,write_json; "
            "from research.moe_dispatch.round2.model import Parameters; from research.moe_dispatch.round2.search import search_workloads; "
            f"b=json.loads(Path({str(e.path(source))!r}).read_text()); "
            f"r=search_workloads(inputs()['development'],Parameters(**b['resume']['parameters']),delta=b['delta'],time_limit_s={args.resume_seconds},"
            "target_families=tuple(b['resume']['families']),resume_state=b); "
            f"write_json(Path({str(e.path(target))!r}),r)")
    return shlex.quote(args.python) + " -c " + shlex.quote(code)


def delivery_status(e, args):
    frozen = e.data("results/E0/frozen_inputs.json")
    dev_ids = frozen.get("development_window_ids", [])
    held_ids = frozen.get("heldout_window_ids", [])
    repro = e.rows("results/E0/reproduce_check.csv")
    receipt = e.data("results/E0/reproduction_receipt.json")
    tests = test_receipts(e)
    runs = proof_runs(e)
    lb = e.rows("results/E3/lb_validity.csv")
    micro = e.rows("results/E2/micro.csv")
    grid = e.rows("results/E3/workload_map.csv")
    sobol = e.rows("results/E3/sobol_samples.csv")
    robust = e.rows("results/E3/robust_objectives.csv")
    stability = e.rows("results/E3/selection_stability.csv")
    main = e.rows("results/E4/heldout_main_table.csv")
    model = e.rows("results/E6/model_token_e2e.csv")
    extreme = e.data("results/E3/workload_extreme.json")
    branch = git_value(e.root, "branch", "--show-current")
    commit = git_value(e.root, "rev-parse", "HEAD")
    checks = {key: {"required_files": list(files), "missing_files": [f for f in files if not e.path(f).is_file()],
                    "status": "完成", "reasons": [], "completed_scope": {}, "remaining": {}, "resume_commands": []}
              for key, files in REQUIRED.items()}

    def partial(section, reason):
        checks[section]["reasons"].append(reason)
        if checks[section]["status"] != "失败":
            checks[section]["status"] = "部分完成"

    def fail(section, reason):
        checks[section]["status"] = "失败"
        checks[section]["reasons"].append(reason)

    for section, value in checks.items():
        if value["missing_files"]:
            partial(section, "缺少必须文件：" + "、".join(value["missing_files"]))
    cohorts = {x.get("cohort"): x for x in receipt.get("cohorts", [])}
    zero = bool(repro) and all(number(r.get("abs_diff")) == 0 for r in repro)
    checks["1"]["completed_scope"] = {"reproduction_rows": len(repro), "all_abs_diff_zero": zero, "cohorts": cohorts, "tests": tests}
    if repro and not zero:
        fail("1", "旧设计复现存在非零差异或缺失差异值")
    elif not zero:
        partial("1", "尚无可核对的逐窗口零误差复现")
    if not (cohorts.get("historical945", {}).get("rows") == 945 and truth(cohorts.get("historical945", {}).get("all_exact")) and
            cohorts.get("previous_BF16_c256_three", {}).get("rows") == 405 and truth(cohorts.get("previous_BF16_c256_three", {}).get("all_exact"))):
        partial("1", "945 历史窗口或上一轮三个设计的 405 行完整复现收据尚缺")
    if not tests["all_required_recorded_suites_passed"]:
        partial("1", "所需单元测试最新收据未全部通过或缺失；详见 E0/tests")
    if branch != "research/moe-supply-first-v3":
        fail("1", "当前分支不等于指定分支：" + branch)
    if not (len(dev_ids) == 18 and len(held_ids) == 135 and len(set(dev_ids)) == 18 and len(set(held_ids)) == 135):
        fail("2", "冻结输入必须为原 18 个开发和 135 个留出窗口")
    checks["2"]["completed_scope"] = {"development_windows": len(dev_ids), "heldout_windows": len(held_ids),
                                         "heldout_batch_counts": frozen.get("heldout_batch_counts", {}), "input_sha256": frozen.get("input_sha256", {})}
    bounds = e.rows("results/E1/bounds_per_window.csv")
    expected_bounds = {(w, name, mode) for w in held_ids for name in (*BASELINES, "fixed_3+3", "fixed_4+2", "previous_asym") for mode in MODES}
    seen_bounds = {(r.get("window_id"), r.get("design"), r.get("onchip_mode")) for r in bounds}
    checks["3"]["completed_scope"] = {"bounds_rows": len(bounds), "expected_rows": len(expected_bounds), "covered_unique_keys": len(seen_bounds & expected_bounds)}
    if seen_bounds != expected_bounds or len(bounds) != len(expected_bounds):
        partial("3", "逐窗口六设计×三模式覆盖不完整或存在重复")
    if any(number(r.get("bound")) is not None and number(r.get("latency_ms")) is not None and
           float(r["bound"]) > float(r["latency_ms"]) + 1e-9 for r in bounds):
        fail("3", "出现下界大于模型延迟的行")
    checks["4"]["completed_scope"] = {"micro_rows": len(micro), "micro_expected_rows": 6 * 3 * 2 * 11 * 3,
                                         "layer_grid_rows": len(e.rows("results/E2/layer_grid.csv"))}
    if len(micro) != 6 * 3 * 2 * 11 * 3:
        partial("4", "六形状×三数据流×两专家类型×十一 Me×三模式微实验尚未覆盖全部组合")
    layer = e.rows("results/E2/layer_grid.csv")
    if len(layer) != 3 * (3 + 9 * 3) * 8:
        partial("4", "B1 的 1×3 与三种异构的 3×3 整层数据流网格不完整")
    audits = {(r.get("sample_index", r.get("design")), r.get("window_id")) for r in lb}
    checks["5.1"]["completed_scope"] = {"proof_runs": len(runs), "lower_bound_checks": len(lb), "lower_bound_unique_checks": len(audits),
                                           "all_lb_ok": bool(lb) and all(truth(r.get("ok")) for r in lb),
                                           "seed_single_geometry_flow_counts": {m: sum(r.get("family") == "single" for r in e.data(f"results/E3/seed_points_{m}.json"))
                                                if isinstance(e.data(f"results/E3/seed_points_{m}.json"), list) else 0 for m in MODES}}
    if any(not truth(r.get("ok")) for r in lb):
        fail("5.1", "区域下界合法性检查出现违反；须停下修正")
    if len(lb) != 2000 * 18 or len(audits) != 2000 * 18:
        partial("5.1", "尚未完成 2,000 个具体设计×全部 18 开发窗口的独立下界审计")
    if {(r.get("onchip_mode"), r.get("proof")) for r in runs} != {(m, p) for m in MODES for p in ("A", "B")}:
        partial("5.1", "六个模式×证明 A/B 运行未全部记录")
    remaining = []
    for run in runs:
        result = run.get("result", {})
        families = result.get("families", {})
        for family, data in families.items():
            live = data.get("open_regions", [])
            row = {"onchip_mode": run.get("onchip_mode"), "proof": run.get("proof"), "family": family,
                   "declared_lattice_points": data.get("declared_lattice_points"), "covered_lattice_points": data.get("covered_lattice_points"),
                   "coverage_pct": data.get("coverage_pct"), "proof_complete": truth(data.get("proof_complete")),
                   "incumbent_ms": data.get("geomean_ms"), "remaining_lower_bound_ms": data.get("open_lb_ms"),
                   "certified_global_lower_bound_ms": data.get("certified_global_lb_ms"), "gap_pct": data.get("gap_pct"),
                   "open_frontiers": len(live), "open_lattice_points": sum(int(x.get("lattice_points", 0)) for x in live)}
            remaining.append(row)
            if not truth(data.get("proof_complete")) or number(data.get("coverage_pct")) != 100:
                partial("5.1", f"{run.get('onchip_mode')}/{run.get('proof')}/{family} 仍有未剪区域，族最优未证明")
        if not families:
            partial("5.1", f"{run.get('onchip_mode')}/{run.get('proof')} 缺少分族证明范围")
        if not truth(result.get("proof_complete")):
            checks["5.1"]["resume_commands"].append(resume_command(e, run.get("onchip_mode"), run.get("proof"), args))
    checks["5.1"]["remaining"] = {"family_frontiers": remaining}
    gaps = e.rows("results/E3/schedule_gaps.csv")
    checks["5.2"]["completed_scope"] = {"rows": len(gaps), "expected_rows": 3 * 3 * (len(dev_ids) + len(held_ids)),
                                           "solver_status_counts": dict(Counter(r.get("solver_status", "missing") for r in gaps))}
    if len(gaps) != 3 * 3 * (len(dev_ids) + len(held_ids)):
        partial("5.2", "三组织族×三模式×所有开发和留出窗口的调度差距覆盖不完整")
    if any(r.get("solver_status") not in ("OPTIMAL", "exact_single") for r in gaps):
        partial("5.2", "存在尚未由求解器证明最优的内层分配；报告是可执行候选调度")
    grid_ids = {r.get("point_index") for r in grid}
    verification = extreme.get("verification_full_domain_certificate", {})
    certified_grid = sum(truth(r.get("proof_complete")) for r in grid)
    checks["5.3"]["completed_scope"] = {"grid_rows": len(grid), "grid_unique_points": len(grid_ids), "planned_grid_points": 4320,
                                           "certified_grid_points": certified_grid, "CMA_evaluations": extreme.get("CMA_evaluations"),
                                           "extreme_delta0_proof_complete": truth(verification.get("proof_complete"))}
    if len(grid_ids) != 4320 or len(grid) != 4320:
        partial("5.3", "8×5×4×3×3×3 合成负载全网格尚未完成")
    if certified_grid != len(grid) or not grid:
        partial("5.3", "部分负载点硬件搜索证明仍开放；Δ 仅为已评估候选比")
    if not extreme or not truth(verification.get("proof_complete")):
        partial("5.3", "极端负载的 δ=0 全域复验未闭合")
    checks["5.3"]["remaining"] = {"extreme_open_lower_bound_ms": verification.get("open_lb_ms"), "extreme_gap_pct": verification.get("gap_pct"),
                                     "grid_uncertified_points": len(grid) - certified_grid}
    checks["5.3"]["resume_commands"] = [f"{shlex.quote(args.python)} -m research.moe_dispatch.round2.regions --stage grid --jobs {args.jobs} --point-seconds {args.point_seconds}",
                                              f"{shlex.quote(args.python)} -m research.moe_dispatch.round2.extreme --jobs {args.jobs} --max-evaluations 500 --point-seconds {args.point_seconds} --verify-seconds {args.resume_seconds}"]
    checks["5.4"]["completed_scope"] = {"objective_rows": len(robust), "stability_rows": len(stability),
                                           "bootstrap_draw_counts": sorted({r.get("bootstrap_draws") for r in stability})}
    if not stability or any(number(r.get("bootstrap_draws")) != 200 for r in stability):
        partial("5.4", "开发窗口的 200 次 bootstrap 不完整")
    if any("full proof open" in r.get("candidate_scope", "") for r in robust) or not robust:
        partial("5.4", "稳健目标只覆盖开发集已评估近优候选，尚非经证明的各族 1% 近优集合")
    checks["5.5"]["completed_scope"] = {"Saltelli_base_N": e.data("results/E3/sobol_protocol.json").get("baseN"),
                                           "sample_rows": len(sobol), "expected_sample_rows": 1792,
                                           "certified_samples": sum(truth(r.get("proof_complete")) for r in sobol)}
    if len(sobol) != 1792 or number(e.data("results/E3/sobol_protocol.json").get("baseN")) != 256:
        partial("5.5", "Saltelli N≥256 的五参数全局采样尚未完成")
    if not sobol or any(not truth(r.get("proof_complete")) for r in sobol):
        partial("5.5", "部分敏感性采样的硬件搜索未闭合，Sobol 是候选估计的指数")
    checks["6"]["completed_scope"] = {"main_table_rows": len(main), "modes": sorted({r.get("onchip_mode") for r in main}),
                                         "schedulers": sorted({r.get("sched_type") for r in main})}
    entries = (*BASELINES, "fixed_3+3", "fixed_4+2", "best_5+1", "best_4+2", "best_2+4", "best_hetero", "U1", "U2")
    seen = {(r.get("entry"), r.get("onchip_mode"), r.get("sched_type")) for r in main}
    if not {(name, m, s) for name in entries for m in MODES for s in ("milp", "runtime")} <= seen:
        partial("6", "E4 条目、三模式或 MILP/runtime 两调度的主表覆盖不完整")
    if any(r.get("entry", "").startswith("best_") and not truth(r.get("search_certified")) for r in main):
        partial("6", "族最优搜索未闭合，E4 当前按冻结已评估候选交付")
    dispatch = e.rows("results/E5/dispatch_table.csv")
    predictor = e.rows("results/E5/predictor_table.csv")
    checks["7"]["completed_scope"] = {"dispatch_rows": len(dispatch), "predictor_rows": len(predictor),
                                         "state_rows": len(e.rows("results/E5/dispatcher_state_bits.csv"))}
    if len(dispatch) != 2 * 3 * 6 * 8 or len(predictor) != 2 * 3 * 6:
        partial("7", "两冻结硬件×三模式的六分派和六预测器覆盖不完整")
    checks["8"]["completed_scope"] = {"moe_rows": len(e.rows("results/E6/moe_layer_e2e.csv")), "model_rows": len(model),
                                         "model_token_timing_rows_available": sum(number(r.get("token_ms")) is not None for r in model)}
    if not model or any(number(r.get("token_ms")) is None for r in model):
        partial("8", "缺少冻结 DeepSeek 模型匹配的 attention/router/norm 每层计时与层映射；整模型每 token 无法生成")
    checks["8"]["remaining"] = {"required_input": "matched DeepSeek non-MoE per-layer timing and layer-to-token mapping"}
    checks["8"]["resume_commands"] = [f"{shlex.quote(args.python)} -m research.moe_dispatch.round2.run --stage E6  # 提供并接入匹配非 MoE 计时后运行"]
    repeat = e.data("results/REPEAT_CHECKS.json") or e.data("REPEAT_CHECKS.json")
    repeat_status = {"receipt": repeat or None, "historical_cohorts_repeated": all(x.get("repeats") == 2 for x in cohorts.values()) and bool(cohorts),
                     "main_seed_flags": {mode: all(truth(r.get("repeat_identical")) for r in e.data(f"results/E3/seed_points_{mode}.json") if "invalid" not in r)
                         if isinstance(e.data(f"results/E3/seed_points_{mode}.json"), list) and e.data(f"results/E3/seed_points_{mode}.json") else None for mode in MODES},
                     "protocol": "model evaluations execute deterministic two-run assertions; certificates explicitly record incumbent repeat flags; frontier traversal is time-limited and may visit different nodes"}
    if e.errors:
        for section in checks:
            partial(section, "存在无法读取的证据文件；详见 evidence_read_errors")
    return {"branch": branch, "commit_at_report_generation": commit, "execution_commits": sorted({x.get("execution_commit") for n in range(7)
                     for x in e.rows(f"results/E{n}/PROVENANCE.csv") if x.get("execution_commit")}),
            "all_requested_sections_complete": all(x["status"] == "完成" for x in checks.values()), "sections": checks,
            "frozen_windows": {"development": len(dev_ids), "heldout": len(held_ids)}, "proof_runs": runs,
            "repeat_checks": repeat_status, "tests": tests, "evidence_read_errors": e.errors,
            "scientific_boundary": "BF16 phase-fluid analytical estimates; no RTL calibration or architecture-winner claim",
            "requirement": "File presence never substitutes for full scope, exact schedule solving, closed certificates, or available full-model timing"}


def render_report(e, status, args):
    frozen = e.data("results/E0/frozen_inputs.json")
    selection = e.data("results/E3/FROZEN_SELECTION.json")
    main = e.rows("results/E4/heldout_main_table.csv")
    runs = status["proof_runs"]
    grid = e.rows("results/E3/workload_map.csv")
    sobol = e.rows("results/E3/sobol.csv")
    extreme = e.data("results/E3/workload_extreme.json")
    sections = [[] for _ in HEADINGS]
    def add(index, text):
        sections[index - 1].append(text)

    for mode in MODES:
        run = next((r.get("result", {}) for r in runs if r.get("onchip_mode") == mode and r.get("proof") == "A"), {})
        if not run:
            conclusion = "证明 A 未记录"
        elif truth(run.get("proof_complete")):
            conclusion = "证明 A 全部组织族已闭合"
        elif truth(run.get("global_delta_proof")):
            conclusion = "全局资源下界已排除相对当前候选快 1.05 倍的设计；各族最优仍未证明"
        else:
            conclusion = "证明 A 仍有开放区域，尚不能排除改善 1.05 倍的设计"
        add(1, f"- {mode}：{conclusion}。")
    for mode in MODES:
        rows = [r for r in main if r.get("onchip_mode") == mode and r.get("sched_type") == "runtime" and r.get("entry", "").startswith("best_")]
        passing = [hardware_label(r["entry"], truth(r.get("search_certified"))) for r in rows if truth(r.get("gate_5pct_pass"))]
        add(1, f"- {mode}{'（非等资源参考）' if mode == 'fixed_issue' else ''}：" +
            ("、".join(passing) + "达到相对 B1/B2 都快至少 5% 的进入校准门槛。" if passing else "已记录异构候选未达到进入校准门槛。" if rows else "门槛结果尚缺。"))
    valid_grid = [r for r in grid if number(r.get("delta_vs_single_pct")) is not None]
    if valid_grid:
        best = min(valid_grid, key=lambda r: float(r["delta_vs_single_pct"]))
        add(1, f"- 合成区域：已评估最强收益点 B{best['batch']}、集中度档 {best['concentration']}、Shared {best['shared_units']}、E/top-k={best['E']}/{best['topk']}、F={best['F']}，候选 Δ={fmt(best['delta_vs_single_pct'], 2)}%；证明开放时不能称全域最强区域。")
    else:
        add(1, "- 异构收益区域尚无完整证据，不能外推到真实负载。")
    ranked = [r for r in sobol if number(r.get("ST")) is not None]
    add(1, "- 敏感性：" + (f"已评估候选指数中 {max(ranked, key=lambda r: float(r['ST']))['param']} 的 ST 最大；未闭合搜索限制其解释。" if ranked else "Sobol 指数尚缺，不能指定最敏感参数。"))
    add(1, "- 本轮无 RTL 校准；只判断是否进入校准，不宣布架构胜出。整模型每 token 计时缺失时按部分完成交付。")

    resource = frozen.get("resources", {})
    hbm = frozen.get("hbm", {})
    model = frozen.get("model", {})
    add(2, table(("项目", "冻结设定"), [
        ("输入与精度", f"{model.get('name', '缺失')}；BF16；H={model.get('H', '缺失')}，routed F={model.get('routed_F', '缺失')}，Shared F={model.get('shared_F', '缺失')}，top-k={model.get('routed_topk', '缺失')}"),
        ("窗口", f"开发 {len(frozen.get('development_window_ids', []))}；留出 {len(frozen.get('heldout_window_ids', []))}；分组 " + "/".join("B" + str(b) for b in BATCHES)),
        ("乘法器", resource.get("multipliers", "缺失")), ("总存储 B", resource.get("storage_bytes", "缺失")),
        ("W / X / 累加 bank", f"{resource.get('W_banks', '缺失')} / {resource.get('X_banks', '缺失')} / {resource.get('acc_banks', '缺失')}；每 bank {resource.get('bytes_per_bank_cycle', '缺失')} B/cycle"),
        ("HBM 主配置", f"HBM2；标称 {hbm.get('nominal_GB_per_s', '缺失')} GB/s；{hbm.get('response_cycles', '缺失')} 周期；请求 {hbm.get('request_bytes', '缺失')} B；信用 {hbm.get('main_credits', '缺失')}；供数上限 {fmt(hbm.get('main_service_cap_GB_per_s'), 3)} GB/s"),
        ("HBM 敏感性", f"信用 {hbm.get('sensitivity_credits', '缺失')}；{fmt(hbm.get('sensitivity_cap_GB_per_s'), 3)} GB/s，仅冻结候选重评估"),
        ("主基线", "B0=原形状 6x4x512 参考；B1=开发集调优单核；B2=开发集调优同构两核"),
        ("指标与门槛", "同一窗口配对延迟比的几何平均；进入校准需对 B1/B2 均降低至少 5%；本轮无校准后胜出结论"),
        ("模式", "pipelined 主表；port_tight 共享约 134.7 B/cycle W 带宽；fixed_issue 每核 30.4 拍发射，非等资源参考"),
        ("估计边界", "post-router Gate/Up、SiLU/Z、Down、combine；有限资源 phase-fluid 解析估计；假设 1 GHz"),
    ]))
    add(2, "资源独立切分：乘法器、私有 W/X/累加/Z 容量、各类 bank 和 vector 份额分别约束总量。B1/B2 每模式均由开发集选定，留出窗口只评估冻结硬件。资源格点的容量分辨率为 1 KiB、bank 与 vector 份额为正整数；这些是模型域限定。")
    add(2, f"当前分支 `{status['branch']}`；报告生成时提交 `{status['commit_at_report_generation']}`。执行提交见各结果目录 README/PROVENANCE；输入 SHA256 见 [frozen_inputs.json](results/E0/frozen_inputs.json)。")
    cohorts = status["sections"]["1"]["completed_scope"].get("cohorts", {})
    add(2, "E0 实际收据：" + "；".join(f"{name}={x.get('rows', '缺失')} 行，零误差={x.get('all_exact', '缺失')}，重复={x.get('repeats', '缺失')} 次" for name, x in cohorts.items()) + "。旧来源边界见 [SOURCES.md](results/E0/SOURCES.md)，不对旧表跨模型计算加速比。")
    add(2, "单元测试最新收据：" + "；".join(f"{name}: returncode={r.get('returncode', '缺失')}, tests={r.get('tests', '见日志')}，[{Path(r['receipt']).name}]({r['receipt']})" for name, r in status["tests"]["latest_by_suite"].items()) + "。")

    bounds = e.rows("results/E1/bounds_per_window.csv")
    head = e.rows("results/E1/headroom_by_batch.csv")
    hrows = []
    for mode in MODES:
        for batch in BATCHES:
            h = next((r for r in head if r.get("onchip_mode") == mode and r.get("batch") == str(batch)), {})
            ms = [gm(r.get("latency_ms") for r in bounds if r.get("onchip_mode") == mode and r.get("batch") == str(batch) and r.get("design") == name) for name in BASELINES]
            if h or any(x is not None for x in ms):
                hrows.append((mode, batch, *(fmt(x) for x in ms), fmt(h.get("median_headroom_pct"), 2), fmt(h.get("geomean_headroom_pct"), 2), h.get("windows", "缺失")))
    add(3, "五项下限取最大值：唯一 HBM、实际 HBM、有效 MAC、必要端口流量、最大单任务。余量=解析延迟/bound−1；忙碌项可重叠。表中 B0/B1/B2 为同一组窗口的延迟几何平均 ms。")
    add(3, table(("模式", "batch", "B0 ms", "B1 ms", "B2 ms", "B1 中位余量 %", "B1 几何余量 %", "窗口数"), hrows) if hrows else "E1 数据尚缺。")
    cells = [f"{r['onchip_mode']}/B{r['batch']}" for r in head if number(r.get("median_headroom_pct")) is not None and float(r["median_headroom_pct"]) >= 5]
    add(3, "B1 中位余量至少 5% 的格子：" + ("、".join(cells) if cells else "无已记录格子") + "。余量是可能改善空间，不能据此证明异构收益。逐窗口数据见 [bounds_per_window.csv](results/E1/bounds_per_window.csv)。")
    add(3, "![B1 延迟与下限](figures/fig_headroom.png)")

    for mode in MODES:
        add(4, f"**{mode}{'：非等资源参考' if mode == 'fixed_issue' else '：等资源解析模式'}**")
        for kind in ("runtime", "milp"):
            rr = [r for r in main if r.get("onchip_mode") == mode and r.get("sched_type") == kind]
            if not rr:
                add(4, f"{kind} 表尚缺。")
                continue
            order = {name: i for i, name in enumerate((*BASELINES, "fixed_3+3", "fixed_4+2", "best_5+1", "best_4+2", "best_2+4", "best_hetero", "U1", "U2"))}
            rr.sort(key=lambda r: order.get(r["entry"], 99))
            add(4, f"{kind}：各 batch 与 all 均为同一窗口集合的延迟几何平均 ms。")
            add(4, table(("条目", "冻结形状", *("B" + str(b) for b in BATCHES), "all ms"),
                         [(hardware_label(r["entry"], truth(r.get("search_certified"))), r["design"], *(fmt(r.get("B" + str(b))) for b in BATCHES), fmt(r.get("all_geomean"))) for r in rr]))
            add(4, table(("条目", "候选/B1", "候选/B2", "95% 速度降低下界 vs B1 %", "vs B2 %", "进入校准 5%"),
                         [(hardware_label(r["entry"], truth(r.get("search_certified"))), fmt(r.get("ratio_vs_B1")), fmt(r.get("ratio_vs_B2")), fmt(r.get("ci95_low_vs_B1"), 2), fmt(r.get("ci95_low_vs_B2"), 2),
                           "诊断参考" if r["entry"] in ("U1", "U2") else "达到" if truth(r.get("gate_5pct_pass")) else "未达到") for r in rr]))
        best = selection.get("modes", {}).get(mode, {}).get("heterogeneous", {})
        cores = best.get("design", {}).get("cores", []) if isinstance(best.get("design"), dict) else []
        if cores:
            share = min(c["pm"] * c["pn"] * c["pk"] for c in cores) / resource.get("multipliers", 12288)
            add(4, f"冻结异构候选的小核乘法器占比 {100 * share:.3f}%" + ("，不超过总数 5%。" if share <= .05 else "。"))
        for name in ("U1", "U2"):
            r = next((r for r in main if r.get("entry") == name and r.get("onchip_mode") == mode and r.get("sched_type") == "runtime"), {})
            if number(r.get("ratio_vs_B1")) is not None:
                add(4, f"{name} 相对 B1 的诊断延迟余量为 {100 * (1 - float(r['ratio_vs_B1'])):.3f}%；不同消融收益不得相加。")
    bd = e.rows("results/E4/breakdown.csv")
    selected_bd = [r for r in bd if r.get("entry") in (*BASELINES, "best_hetero") and r.get("batch") == "all" and r.get("sched_type") == "runtime"]
    add(4, table(("模式", "条目", "核0 busy", "核1 busy", "W busy", "X busy", "累加 busy", "HBM busy", "核完成差", "idle", "绑定项"),
                 [(r["onchip_mode"], hardware_label(r["entry"]), *(fmt(r.get(k), 3) for k in ("core0_compute_busy", "core1_compute_busy", "w_port_busy", "x_port_busy", "acc_port_busy", "hbm_busy_frac", "core_finish_gap", "idle_frac")), r.get("binding_term", "缺失")) for r in selected_bd]) if selected_bd else "时间分解尚缺。")
    add(4, "busy 分数和完成差的单位遵循原 CSV；重叠占用不相加成墙钟。B2/B1 的同构拆分代价可逐 batch/模式从上表配对几何平均得到；不拼接各 batch 胜格。512 信用只重评估冻结三组织族，见 [hbm512_sensitivity.csv](results/E4/hbm512_sensitivity.csv)。")
    add(4, "![各 batch 主结果](figures/fig_main_bars.png)\n\n![时间分解](figures/fig_breakdown.png)")

    summary = e.data("results/E3/bnb_summary.json")
    add(5, "完整声明域：PM=1..16、PN=1..192、PK∈{32,64,128,256,512,1024}；44 单核、41 同构双核、16,678 异构双核，共 16,763 个几何点。单核×三数据流为 132 个具体点；双核容量、bank 与 vector 独立切分形成远大于几何点数的联合格点域。枚举几何不等于穷举联合资源域。")
    add(5, "记录的枚举计数：`" + json.dumps(summary.get("geometry_counts", {}), ensure_ascii=False, sort_keys=True) + "`。证明 A 用因子 1.05 剪枝；这相当于延迟降低约 4.762%，与留出集进入校准要求的降低至少 5% 是两个判据。证明 B 用 δ=0。")
    frontiers = status["sections"]["5.1"]["remaining"].get("family_frontiers", [])
    add(5, table(("模式", "证明", "组织族", "总格点", "闭合格点", "覆盖 %", "候选 ms", "未剪 LB ms", "族全局 LB ms", "差距 %", "开放区域", "族证明"),
                 [(r["onchip_mode"], r["proof"], r["family"], r["declared_lattice_points"], r["covered_lattice_points"], fmt(r["coverage_pct"], 8), fmt(r["incumbent_ms"]), fmt(r["remaining_lower_bound_ms"]), fmt(r["certified_global_lower_bound_ms"]), fmt(r["gap_pct"], 3), r["open_frontiers"], "闭合" if r["proof_complete"] else "未完成") for r in frontiers]) if frontiers else "分族证明收据尚缺。")
    add(5, "上表的覆盖率为已剪/已完整评估格点占声明域的比例。未剪区域仍保留完整区间、下界和恢复状态；所有格点被账本追踪并不等于证明覆盖率 100%。即使全局 incumbent/资源下界已经给出 δ 证书，也不能把 B2 或某个异构比例族称为全局最优。4+2 与 2+4 在可互换角色与独立资源切分的物理域中是镜像别名。")
    audit = status["sections"]["5.1"]["completed_scope"]
    add(5, f"下界合法性实际审计 {audit['lower_bound_checks']:,} 行、独立键 {audit['lower_bound_unique_checks']:,}，全部 ok={audit['all_lb_ok']}；要求 2,000 个具体设计×18 开发窗口。随机检查支持实现可信度，不能替代理论下界证明。")
    gaps = e.rows("results/E3/schedule_gaps.csv")
    grows = []
    held = set(frozen.get("heldout_window_ids", []))
    for mode in MODES:
        for family in ("single", "homogeneous", "heterogeneous"):
            r = [x for x in gaps if x.get("onchip_mode") == mode and x.get("design") == family and x.get("window_id") in held]
            if r:
                grows.append((mode, family, *baseline_context(e, mode, "milp"), fmt(gm(float(x["T_milp_sched"]) / float(x["T_lb"]) for x in r)),
                              fmt(gm(float(x["T_runtime_eft"]) / float(x["T_milp_sched"]) for x in r)), len(r)))
    add(5, table(("模式", "调度候选族", "B0 MILP ms", "B1 MILP ms", "B2 MILP ms", "T_milp_sched/T*", "T_runtime/T_milp_sched", "留出窗口"), grows) if grows else "调度差距尚缺。")
    add(5, "这里 T* 是 CP-SAT 专家分配的资源约束松弛下界；LPT 完整流式回放是可执行调度。该模型目标的最优性不等于任意时序调度、RTL 或实芯片最优性。B0/B1/B2 三列为 E4 冻结基线上下文，没有假造 B0 的独立求解差距。")
    add(5, "![剪枝覆盖与 incumbent](figures/fig_bnb_coverage.png)")
    if status["sections"]["5.1"]["resume_commands"]:
        add(5, "继续运行保留原证书并写入新恢复结果：\n\n```sh\n" + "\n".join(status["sections"]["5.1"]["resume_commands"]) + "\n```")

    calibration = e.rows("results/E3/synthetic_calibration.csv")
    add(6, f"合成全网格已记录 {len(grid):,}/4,320 点，其中硬件证明闭合 {sum(truth(r.get('proof_complete')) for r in grid):,} 点。只在本负载区域分析使用合成路由；每点的单核/同构/异构候选重新搜索，不能把其 best_single/best_homo 偷换为真实负载冻结 B1/B2。")
    if calibration:
        kval = [float(r["me_hist_KL_real_to_synthetic"]) for r in calibration if number(r.get("me_hist_KL_real_to_synthetic")) is not None]
        dval = [abs(float(r["distinct_relative_error"])) for r in calibration if number(r.get("distinct_relative_error")) is not None]
        add(6, f"真实路由校准记录 {len(calibration)} 行；真实→合成 Me 直方图 KL 范围 [{min(kval):.5g}, {max(kval):.5g}]，distinct 数相对误差绝对值范围 [{min(dval):.3%}, {max(dval):.3%}]。完整逐窗口/浓度数据见 [synthetic_calibration.csv](results/E3/synthetic_calibration.csv)。")
    if valid_grid:
        best = min(valid_grid, key=lambda r: float(r["delta_vs_single_pct"]))
        add(6, "全网格中已评估最强候选点：`" + json.dumps({k: best.get(k) for k in ("batch", "concentration", "alpha", "shared_units", "E", "topk", "F", "bw_or_mac_scale", "delta_vs_single_pct", "delta_vs_homo_pct", "proof_complete", "open_lb_ms", "gap_pct")}, ensure_ascii=False) + "`。真实点为 E=64、top-k=6、routed F=1408、Shared=2 当量、主信用供数上限及捕获 batch；浓度只由开发路由拟合，逐窗口 KL 决定其接近程度。")
    if extreme:
        near = extreme.get("distance_from_real", {}).get("nearest_real_window", {})
        add(6, f"CMA-ES 实际评估 {extreme.get('CMA_evaluations', '缺失')} 次（上限 500）；最强已评估负载参数 `{json.dumps(extreme.get('best_evaluated_workload_parameters', {}), ensure_ascii=False)}`。δ=0 复验 Δ vs 单核={fmt(100 * extreme['delta_vs_single'], 3) if number(extreme.get('delta_vs_single')) is not None else '缺失'}%，vs 同构={fmt(100 * extreme['delta_vs_homo'], 3) if number(extreme.get('delta_vs_homo')) is not None else '缺失'}%；证明闭合={extreme.get('verification_full_domain_certificate', {}).get('proof_complete', '缺失')}。最近真实窗口 `{near.get('window_id', '缺失')}`，batch log2 距离={fmt(near.get('batch_log2_distance'))}，Me 直方图 KL={fmt(near.get('Me_hist_KL_synthetic_to_real'))}，distinct 合成/真实={near.get('distinct_synthetic', '缺失')}/{near.get('distinct_real', '缺失')}。")
    add(6, "负载点 best 与 Δ 均按各点证书解释；未闭合时只是候选估计。图中真实标记的来源与映射见图说明。\n\n![异构候选收益区域](figures/fig_workload_map.png)")

    robust = e.rows("results/E3/robust_objectives.csv")
    stability = e.rows("results/E3/selection_stability.csv")
    rr = []
    for mode in MODES:
        winners = {}
        for objective in ("geomean", "cvar10", "minimax"):
            row = next((r for r in robust if r.get("onchip_mode") == mode and r.get("selection_family") == "all" and r.get("objective") == objective and truth(r.get("selected_by_objective"))), {})
            winner = row.get("geometry", "缺失") + "/" + row.get("flows", "缺失")
            winners[objective] = winner
            top = [r for r in stability if r.get("onchip_mode") == mode and r.get("selection_family") == "all" and r.get("objective") == objective and truth(r.get("most_selected"))]
            rr.append((mode, objective, *baseline_context(e, mode, "milp"), winner, fmt(row.get("heldout_ratio_vs_frozen_single")),
                       "; ".join(r["geometry"] + "/" + r["flows"] + ":" + fmt(100 * float(r["selection_share"]), 1) + "%" for r in top) or "缺失"))
        add(7, f"{mode} 三目标的全候选诊断选择" + ("一致。" if len(set(winners.values())) == 1 and "缺失" not in next(iter(winners.values())) else "不同或证据未完整；对应 batch 比值见 robust_objectives.csv。"))
    add(7, "CVaR10=最差 ceil(0.1×窗口数) 个配对比值的算术平均；minimax=各 batch 配对几何平均的最大值。开发集 200 次 bootstrap 统计候选重选份额。留出集目标选择是诊断，主表硬件仍来自开发集冻结，不能将此诊断改成新的 headline。")
    add(7, table(("模式", "目标", "B0 MILP ms", "B1 MILP ms", "B2 MILP ms", "诊断选择", "目标值 vs 冻结 B1", "开发 bootstrap 最常选择"), rr))
    add(7, "如组织族证明开放，候选集合仅为已评估开发候选的 1% 内集合，不能称全域近优集合。各族细表见 [robust_objectives.csv](results/E3/robust_objectives.csv) 与 [selection_stability.csv](results/E3/selection_stability.csv)。")

    add(8, "每 tile 片上时间 1–30.4 拍、bank 带宽 8–32 B/cycle、点积每级 1–4 拍、信用 256–512、vector 吞吐 0.5–2 倍；Saltelli 基础 N=256，五参数一阶/总效应采样 1,792 点。每点重新搜索硬件，不固定真实负载候选。")
    add(8, table(("参数", "S1", "S1 置信半宽", "ST", "ST 置信半宽", "全采样硬件证明"),
                 [(r["param"], *(fmt(r.get(k)) for k in ("S1", "S1_ci", "ST", "ST_ci")), "闭合" if truth(r.get("all_searches_certified")) else "开放；候选指数") for r in sobol]) if sobol else "Sobol 数据尚缺。")
    flip = e.rows("results/E3/flip_boundary.csv")
    add(8, table(("参数", "Δ 目标", "边界值", "左端", "右端", "状态", "证明"), [(r["param"], r["target_delta"], fmt(r.get("value")), fmt(r.get("bracket_low")), fmt(r.get("bracket_high")), r.get("status"), r.get("proof_complete")) for r in flip]) if flip else "翻转边界数据尚缺。")
    if ranked:
        names = [r["param"] for r in sorted(ranked, key=lambda r: float(r["ST"]), reverse=True)[:2]]
        add(8, "候选指数建议后续优先校准 " + "、".join(names) + "；须先确认该排名在未剪搜索差距下仍稳定。切片未出现交叉只能表述为已评估范围无交叉，不能证明不存在边界。")
    add(8, "![Sobol 指数](figures/fig_sobol.png)\n\n![翻转切片](figures/fig_flip_boundary.png)")

    layer = e.rows("results/E2/layer_grid.csv")
    add(9, "OS：部分和驻阵列沿 K 推进、每个 M 波次重读 W；WS：W 驻留沿 M 推进、K 段部分和读改写累加 SRAM；IS：X 驻留沿 N 推进、部分和读改写。组会 RS 在 GEMM 中按 IS 实现，即 RS=IS。容量不足时重读量同时计入 HBM/SRAM。")
    for mode in MODES:
        data = [r for r in layer if r.get("onchip_mode") == mode and r.get("batch") == "all"]
        if not data:
            continue
        out = []
        for name in ("B1", "fixed_4+2", "previous_asym", "best_hetero"):
            ss = [r for r in data if r.get("design") == name]
            if not ss:
                continue
            best = min(ss, key=lambda r: float(r["geomean_ms"]))
            os = next((r for r in ss if r["df_big"] == "OS" and r.get("df_small", "") in ("", "OS")), {})
            if number(os.get("geomean_ms")) is not None:
                add(9, f"{mode}/{hardware_label(name)}：最快 {best['df_big']}/{best.get('df_small') or '-'}，OS/OS 比其慢 {100 * (float(os['geomean_ms']) / float(best['geomean_ms']) - 1):.3f}%，" + ("在最快 1% 以内。" if float(os['geomean_ms']) <= 1.01 * float(best['geomean_ms']) else "超出最快 1%。"))
            for flow in ("OS", "WS", "IS"):
                vals = []
                for small in ("OS", "WS", "IS"):
                    r = next((r for r in ss if r["df_big"] == flow and r.get("df_small", "") == ("" if name == "B1" else small)), {})
                    vals.append(fmt(r.get("ratio_vs_OS_OS")) if name != "B1" or small == "OS" else "单核不适用")
                out.append((hardware_label(name), flow, *vals, *baseline_context(e, mode)))
        add(9, f"{mode}{'（非等资源参考）' if mode == 'fixed_issue' else ''}：3×3 数字为同硬件候选/OS_OS，B0/B1/B2 列为 E4 冻结主表 ms 上下文。B0/B2 未在本节另跑数据流网格，不能当作补测结果。")
        add(9, table(("硬件", "大核 flow", "小核 OS", "小核 WS", "小核 IS", "B0 ms", "B1 ms", "B2 ms"), out))
    micro = e.rows("results/E2/micro.csv")
    pairs = defaultdict(dict)
    for r in micro:
        pairs[(r["shape"], r["expert_type"], r["Me"], r["onchip_mode"])][r["dataflow"]] = r
    for mode in MODES:
        wave = [(k, r) for k, r in pairs.items() if k[3] == mode and int(k[2]) <= int(k[0].split("x")[0]) and "OS" in r and "WS" in r]
        equal = sum(all(number(r["OS"].get(x)) == number(r["WS"].get(x)) for x in ("cycles", "w_sram_bytes", "x_sram_bytes", "acc_sram_bytes", "hbm_bytes")) for _, r in wave)
        shared = [float(r["WS"]["cycles"]) / float(r["OS"]["cycles"]) for k, r in pairs.items() if k[3] == mode and k[1] == "Shared" and int(k[2]) > int(k[0].split("x")[0]) and "OS" in r and "WS" in r]
        hot = [float(r["WS"]["cycles"]) / float(r["OS"]["cycles"]) for k, r in pairs.items() if k[3] == mode and k[1] == "routed" and int(k[2]) > int(k[0].split("x")[0]) and "OS" in r and "WS" in r]
        add(9, f"{mode}：Me≤PM 的 OS/WS 周期及全部记录流量均相同 {equal}/{len(wave)} 组；Me>PM 的 Shared WS/OS 范围 " + (f"[{min(shared):.5f}, {max(shared):.5f}]" if shared else "缺失") + "，热 routed WS/OS 范围 " + (f"[{min(hot):.5f}, {max(hot):.5f}]" if hot else "缺失") + "。小于 1 表示 WS 更快；每形状/Me 的细值保留在 micro.csv，不能把范围当所有大核同等收益。")
    add(9, "微实验是独立单专家流量与周期诊断；不把发射次数当总周期，不把单核完整私有预算映射为等资源整层收益。\n\n![数据流网格](figures/fig_dataflow_grid.png)\n\n![Me 交叉点](figures/fig_me_crossover.png)")

    dispatch = e.rows("results/E5/dispatch_table.csv")
    predictors = e.rows("results/E5/predictor_table.csv")
    add(10, "硬件冻结为各模式的异构已评估候选与固定 4+2。阈值只在开发集调优，先按同一开发窗口序列预热，再按同一留出序列统计。两次完整学习序列重放一致；随机种子冻结。")
    for mode in MODES:
        context = baseline_context(e, mode)
        dr = [r for r in dispatch if r.get("onchip_mode") == mode and r.get("batch") == "all"]
        pr = [r for r in predictors if r.get("onchip_mode") == mode]
        add(10, f"{mode}{'（非等资源参考）' if mode == 'fixed_issue' else ''}；B0/B1/B2 为 E4 主表冻结基线延迟上下文 ms，未补造基线预测器观测。")
        add(10, table(("硬件", "分派", "开发最优 T", "ms", "比 MILP-LPT", "B0 ms", "B1 ms", "B2 ms"),
                      [(hardware_label(r["design"]), r["policy"], r.get("threshold_tuned_on_dev"), fmt(r.get("geomean_ms")), fmt(r.get("ratio_vs_milp_sched")), *context) for r in dr]) if dr else "分派表尚缺。")
        add(10, table(("硬件", "预测器", "MAE %", "success %", "late %", "stall cycles", "E2E/oracle", "E2E/ours", "B0 ms", "B1 ms", "B2 ms"),
                      [(hardware_label(r["design"]), r["predictor"], *(fmt(r.get(k), 3) for k in ("mae_pct", "success_pct", "late_pct", "stall_cycles", "e2e_ratio_vs_oracle", "e2e_ratio_vs_ours")), *context) for r in pr]) if pr else "预测器表尚缺。")
        for name in ("best_hetero", "fixed_4+2"):
            ss = [r for r in dr if r.get("design") == name]
            a = next((r for r in ss if r["policy"] == "threshold_2"), {})
            b = next((r for r in ss if r["policy"].startswith("threshold_fallback")), {})
            if number(a.get("geomean_ms")) and number(b.get("geomean_ms")):
                add(10, f"{mode}/{hardware_label(name)}：纯 T=2 阈值比阈值+回退慢 {100 * (float(a['geomean_ms']) / float(b['geomean_ms']) - 1):.3f}%。")
    add(10, "MAE=平均 |预测时长−实际时长|/实际时长；success=Next 第一权重块落在 Current 结束前 W 内的比例，W 为两个权重块的纯计算服务时间（含发射、点积管线及累加依赖，排除 HBM/端口）；late=块晚于 Current 结束；stall=暴露权重等待。oracle 为相同政策两次 profile/replay 参考，若分配改变会留残差，不能称完美先知零误差。状态 bit 只是状态量估计，没有综合面积或频率。")
    add(10, "![预测器误差与端到端](figures/fig_predictor.png)")

    moe = e.rows("results/E6/moe_layer_e2e.csv")
    for mode in MODES:
        rr = [r for r in moe if r.get("onchip_mode") == mode and r.get("sched_type") == "runtime"]
        if rr:
            add(11, f"{mode}{'（非等资源参考）' if mode == 'fixed_issue' else ''}：直接取 E4 的 MoE 层 ms 与配对几何平均比值。")
            data = []
            for batch in BATCHES:
                rows = {r["design"]: r for r in rr if r.get("batch") == str(batch)}
                data.append((batch, *(fmt(rows.get(name, {}).get("moe_ms_per_layer")) for name in (*BASELINES, "best_hetero")),
                             fmt(rows.get("best_hetero", {}).get("ratio_vs_B1"))))
            add(11, table(("batch", "B0 ms", "B1 ms", "B2 ms", "异构候选 ms", "异构/B1"), data))
    modelrows = e.rows("results/E6/model_token_e2e.csv")
    available = sum(number(r.get("token_ms")) is not None for r in modelrows)
    add(11, f"整模型 token 时序可用行 {available}/{len(modelrows)}。" + ("缺少匹配冻结 DeepSeek 捕获的非 MoE 每层计时及层映射；non_moe_ms_per_layer/layers/token_ms 留空，整模型项部分完成。" if available != len(modelrows) or not modelrows else "所有 token 时间来自匹配的非 MoE 层证据。") + "不以其他模型旧 trace 拼接端到端结果，本轮不做 GPU 对比。")

    add(12, "本轮没有 RTL；片上 tile 发射、点积和端口时序未校准；解析 phase-fluid 服务与真实 bank 冲突/native HBM 行为存在模型差异。相同乘法器、存储和 bank 不等于相同硅面积；面积/能耗档关闭。fixed_issue 为非等资源参考。")
    add(12, "留出集历史上曾被访问，不称全新盲测。合成负载只用于收益区域/连续极值探索；Sobol 用真实开发窗口与参数扰动。私有容量分池、1 KiB 分辨率、正整数 vector 切分、有限任务窗为显式模型约束；任务分配 CP-SAT 松弛加 LPT 回放不是任意全时序调度最优。")
    add(12, "开放证明、未认证负载点、未认证敏感性采样和缺失整模型计时均按交付状态记录。各 batch 的胜格不拼成统一胜出；消融收益与重叠资源占用不相加。运行时间上限只终止搜索，不缩小声明域。")
    add(12, "交付状态如下；文件存在不等于完成。详细范围、残余下界、差距和继续命令见 [DELIVERY_STATUS.json](DELIVERY_STATUS.json)。")
    add(12, table(("用户节", "状态", "原因"), [(section, value["status"], "；".join(value["reasons"]) or "必须文件和本节范围检查通过") for section, value in status["sections"].items()]))

    if not status["sections"]["5.1"]["status"] == "完成":
        add(13, "- 按第 5 节恢复命令继续当前开放区间，优先缩小决定单核/同构/异构比较的残余下界差距；证明闭合前保持候选措辞。")
    if ranked:
        add(13, "- 闭合或收紧参数点的搜索误差后，优先测量 ST 最大的 " + "、".join(r["param"] for r in sorted(ranked, key=lambda r: float(r["ST"]), reverse=True)[:2]) + "，以核对结论是否跨进入校准边界。")
    if status["sections"]["8"]["status"] != "完成":
        add(13, "- 捕获与本次 DeepSeek 输入及层映射匹配的 attention/router/norm 层时序，再计算整模型每 token 时间。")
    passing = [r for r in main if r.get("entry", "").startswith("best_") and r.get("sched_type") == "runtime" and truth(r.get("gate_5pct_pass")) and r.get("onchip_mode") != "fixed_issue"]
    if passing:
        add(13, "- 对达到进入校准门槛的等资源模式冻结候选做后续时序校准；校准后仍需相对 B1/B2 至少 10% 且配对 bootstrap 95% 下界至少 5% 才能讨论胜出。")
    add(13, "复现报告（仅重读证据，不运行实验）：\n\n```sh\n" + shlex.quote(args.python) + " -m research.moe_dispatch.round2.report --root " + shlex.quote(str(e.root)) + "\n```")
    add(13, "各节命令、执行提交与输入哈希见 results/E0–E6/README.md 和 PROVENANCE.csv；[交付状态](DELIVERY_STATUS.md) 列出全部要求与继续命令。")
    lines = ["# MoE 大小核 NPU 第二轮评估", "", "证据来自本次 results/E0–E6 的 CSV/JSON。缺失值保持缺失；主指标为窗口配对延迟比几何平均。", ""]
    for index, (heading, content) in enumerate(zip(HEADINGS, sections), 1):
        lines += [f"## {index}. {heading}", "", "\n\n".join(content), ""]
    return "\n".join(lines)


def status_markdown(status):
    lines = ["# 第二轮交付状态", "", f"分支：`{status['branch']}`；报告生成时提交：`{status['commit_at_report_generation']}`。", "",
             "全部要求完成：" + ("是" if status["all_requested_sections_complete"] else "否") + "。存在文件不等于完整范围已完成。", "",
             table(("用户节", "状态", "必须文件", "原因"), [(k, v["status"], "、".join(v["required_files"]), "；".join(v["reasons"]) or "本节范围检查通过") for k, v in status["sections"].items()])]
    for section, value in status["sections"].items():
        lines += [f"## {section}", "", "已完成范围：", "", "```json", json.dumps(value["completed_scope"], ensure_ascii=False, indent=2), "```", ""]
        if value["remaining"]:
            lines += ["未完成范围的下界/差距：", "", "```json", json.dumps(value["remaining"], ensure_ascii=False, indent=2), "```", ""]
        if value["resume_commands"]:
            lines += ["继续命令（原始结果保留）：", "", "```sh", *value["resume_commands"], "```", ""]
    lines += ["## 重复与测试", "", "```json", json.dumps({"repeat_checks": status["repeat_checks"], "tests": status["tests"]}, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def generate(root, destination, args):
    e = Evidence(root)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    status = delivery_status(e, args)
    # The generated report is a concrete deliverable, even on a first pass.
    for name in ("REPORT_ZH.md",):
        if name in status["sections"]["9–10"]["missing_files"]:
            status["sections"]["9–10"]["missing_files"].remove(name)
    value = status["sections"]["9–10"]
    value["reasons"] = ["缺少必须文件：" + "、".join(value["missing_files"])] if value["missing_files"] else []
    value["status"] = "部分完成" if value["missing_files"] else "完成"
    status["all_requested_sections_complete"] = all(x["status"] == "完成" for x in status["sections"].values())
    report = render_report(e, status, args)
    status["evidence_files"] = dict(sorted(e.files.items()))
    status["evidence_read_errors"] = list(e.errors)
    (destination / "REPORT_ZH.md").write_text(report)
    (destination / "DELIVERY_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (destination / "DELIVERY_STATUS.md").write_text(status_markdown(status))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--python", default="/tmp/plena-round2-venv/bin/python")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--point-seconds", type=float, default=2)
    parser.add_argument("--resume-seconds", type=float, default=3600)
    args = parser.parse_args()
    status = generate(args.root, args.out or args.root, args)
    print(json.dumps({"report": str((args.out or args.root) / "REPORT_ZH.md"), "all_complete": status["all_requested_sections_complete"],
                      "statuses": {k: v["status"] for k, v in status["sections"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
