"""独立只读审计：近优留出诊断，不调用模拟器、不重新选择主硬件。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MODES = ("pipelined", "port_tight")
CONSTRAINTS = ("C0", "C1")
FAMILIES = ("single", "homogeneous", "5+1", "4+2", "3+3")
LABELS = dict(zip(FAMILIES, ("B1", "B2", "H51", "H42", "H33")))
OBJECTIVES = ("geomean", "cvar10", "minimax")
EXPECTED_POINTS = 187
EXPECTED_GROUP_CANDIDATES = 334
REQUIRED = ("COMPLETE.json", "RUN_RECEIPT.json", "CANDIDATE_PLAN.json",
            "repeat_checks.json", "candidate_counts.csv", "robust_objectives.csv",
            "winners.csv", "selection_stability.csv", "per_window.csv", "SUMMARY_ZH.md")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _physical_key(hardware):
    hardware = dict(hardware)
    hardware["label"] = ""
    return _canonical(hardware)


def _point_id(mode, hardware):
    return hashlib.sha256((mode + _physical_key(hardware)).encode()).hexdigest()[:20]


def _geometry(hardware):
    return "+".join(f"{c['pm']}x{c['pn']}x{c['pk']}" for c in hardware["cores"])


def _true(value):
    return value is True or value == "True"


def _equal(a, b, label):
    if _canonical(a) != _canonical(b):
        raise AssertionError(label)


def _close(actual, expected, label):
    actual, expected = float(actual), float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AssertionError(label + ": 非有限数值")
    error = abs(actual - expected)
    if error > 2e-12 * max(1., abs(expected)):
        raise AssertionError(f"{label}: {actual} != {expected}")
    return error


def _metrics(values, batches):
    """独立重算旧5.4定义，CVaR是最差10%的算术平均。"""
    a = np.asarray(values, dtype=float)
    bs = np.asarray(batches)
    if len(a) != len(bs) or not len(a) or np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise AssertionError("配对比值必须有限且为正，并与batch一一对应")
    grouped = {int(b): float(np.exp(np.log(a[bs == b]).mean())) for b in sorted(set(bs))}
    return {"geomean": float(np.exp(np.log(a).mean())),
            "cvar10": float(np.sort(a)[-max(1, math.ceil(.1 * len(a))):].mean()),
            "minimax": max(grouped.values()), "batch_geomeans": grouped}


def _bootstrap(matrix, batches, draws=200, seed=20261007):
    a = np.asarray(matrix, dtype=float)
    bs = np.asarray(batches)
    rng = np.random.default_rng(seed)
    counts = {name: [0] * len(a) for name in OBJECTIVES}
    for _ in range(draws):
        ids = rng.integers(0, a.shape[1], size=a.shape[1])
        met = [_metrics(x[ids], bs[ids]) for x in a]
        for name in OBJECTIVES:
            j = min(range(len(a)), key=lambda i: (met[i][name], i))
            counts[name][j] += 1
    return counts


def _check_point(root, point, rec, heldout, source_hashes):
    """核对原始字节与物理结果；支持新回放及经过核验的E5双遍引用。"""
    pid = point["point_id"]
    assert rec["point_id"] == pid, "点ID不符"
    _equal(rec["hardware"], point["hardware"], "点硬件与冻结计划不符")
    _equal(rec["parameters"], point["parameters"], "点参数与冻结计划不符")
    assert rec["exact_repeat"] is True, "缺少实际双遍一致标记"
    assert rec["result_digest"] == rec["repeat_digest"], "两遍digest不一致"
    raw_path = root / rec["raw_file"]
    assert _sha(raw_path) == rec["raw_sha256"], "原始gzip字节SHA不符"
    with gzip.open(raw_path, "rt") as f:
        raw = json.load(f)
    assert _digest(raw) == rec["result_digest"], "原始全结果digest不符"
    assert len(raw) == len(heldout), "原始窗口数量不符"
    assert [s["window_id"] for s in rec["solver_checks"]] == [w["id"] for w in heldout], "solver窗口顺序不符"
    if rec["kind"].startswith("new"):
        assert [x["window_id"] for x in raw] == [w["id"] for w in heldout], "原始窗口顺序不符"
        assert [x["batch"] for x in raw] == [w["batch"] for w in heldout], "原始batch顺序不符"
        physical = []
        for x, s in zip(raw, rec["solver_checks"]):
            e = x["evaluation"]
            assert e["legal"] is True, "非法物理回放"
            assert float(e["lb_cycles"]) <= float(e["milp_sched"]["cycles"]) + 1e-5, "分配下界超过物理耗时"
            expected = {"window_id": x["window_id"], "status": e["assignment"]["status"],
                        "allocation_optimal": e["assignment"]["optimal"],
                        "lb_cycles": e["lb_cycles"], "owners": e["assignment"]["owners"]}
            _equal(s, expected, "solver核验记录与原始结果不符")
            physical.append(e["milp_sched"])
    elif rec["kind"].startswith("existing"):
        source_path = root / rec["source_receipt"]
        repeat_path = source_path.parent / "repeat_checks.json"
        assert _sha(source_path) == rec["source_receipt_sha256"], "E5运行回执SHA不符"
        assert _sha(repeat_path) == rec["source_repeat_receipt_sha256"], "E5双遍回执SHA不符"
        source, repetitions = _json(source_path), _json(repeat_path)
        assert source["source_unchanged"] is True, "E5来源未冻结"
        for name in ("model.py", "runtime.py", "optimizer.py", "config.py"):
            path = str(root / name)
            assert source["source_sha256"][path] == source_hashes[path], "E5物理源版本不同"
        old_manifest = root.parent / "round2/results/E0/frozen_inputs.json"
        assert source["input_manifest_sha256"] == _sha(old_manifest), "E5输入版本不同"
        matching = [r for r in repetitions if r["dispatch"] == "milp"
                    and r["onchip_mode"] == point["onchip_mode"]
                    and _physical_key(r["hardware"]) == point["physical_key"]
                    and r["raw_file"] == rec["raw_file"]]
        assert len(matching) == 1, "E5引用必须对应唯一的实际双遍记录"
        prior = matching[0]
        assert prior["exact_repeat"] is True and prior["result_digest"] == prior["repeat_digest"], "E5未通过双遍"
        for field in ("parameters", "raw_sha256", "result_digest", "repeat_digest", "solver_checks"):
            _equal(prior[field], rec[field], "E5引用字段不符: " + field)
        physical = raw
    else:
        raise AssertionError("未知回放来源类型")
    expected_rows = []
    for w, r in zip(heldout, physical):
        assert math.isfinite(r["cycles"]) and r["cycles"] > 0 and r["hbm_bytes"] >= 0, "非法物理指标"
        expected_rows.append({"window_id": w["id"], "batch": w["batch"], "cycles": r["cycles"],
                              "latency_ms": r["cycles"] / 1e6, "hbm_bytes": r["hbm_bytes"],
                              "physical_digest": _digest(r)})
    _equal(rec["rows"], expected_rows, "逐窗口指标与原始物理结果不符")
    return expected_rows


def _diagnostic_groups(groups, points):
    result = list(groups)
    for mode in MODES:
        for constraint in CONSTRAINTS:
            family = [g for g in groups if (g["onchip_mode"], g["constraint_group"]) == (mode, constraint)]
            ids = sorted({i for g in family for i in g["candidate_ids"]}, key=lambda i: points[i]["physical_key"])
            dev = {i: next(g["development_ms"][i] for g in family if i in g["development_ms"]) for i in ids}
            result.append({"onchip_mode": mode, "constraint_group": constraint, "family": "all",
                           "candidate_ids": ids, "development_ms": dev, "frozen_id": None})
    return result


def _indexed(rows, fields, label):
    index = {tuple(str(r[f]) for f in fields): r for r in rows}
    assert len(index) == len(rows), label + ": 重复键"
    return index


def _audit_tables(out, groups, points, point_rows, selection, dev, heldout):
    """从已核验逐窗口物理输出重算全部表，包含200次开发bootstrap。"""
    gfields = ("onchip_mode", "constraint_group", "family")
    metrics = _indexed(_csv(out / "robust_objectives.csv"), (*gfields, "point_id"), "目标表")
    winners = _indexed(_csv(out / "winners.csv"), (*gfields, "objective"), "赢家表")
    stability = _indexed(_csv(out / "selection_stability.csv"), (*gfields, "point_id", "objective"), "bootstrap表")
    per_window = _indexed(_csv(out / "per_window.csv"), (*gfields, "point_id", "window_id"), "逐窗口表")
    common = {mode: _point_id(mode, selection["modes"][mode]["C0"]["B1"]) for mode in MODES}
    seen = {"metrics": set(), "winners": set(), "stability": set(), "per_window": set()}
    max_error = 0.
    for group in groups:
        mode = group["onchip_mode"]
        gt = tuple(group[f] for f in gfields)
        ids = sorted(group["candidate_ids"], key=lambda i: points[i]["physical_key"])
        baseline = np.asarray([r["latency_ms"] for r in point_rows[common[mode]]])
        ratio = [np.asarray([r["latency_ms"] for r in point_rows[i]]) / baseline for i in ids]
        computed = [_metrics(x, [w["batch"] for w in heldout]) for x in ratio]
        win = {name: min(range(len(ids)), key=lambda j: (computed[j][name], points[ids[j]]["physical_key"])) for name in OBJECTIVES}
        agree = len({ids[j] for j in win.values()}) == 1
        for j, pid in enumerate(ids):
            rowkey = (*gt, pid)
            r = metrics[rowkey]; seen["metrics"].add(rowkey)
            assert r["geometry"] == _geometry(points[pid]["hardware"]), "几何标签不符"
            _equal(json.loads(r["hardware"]), points[pid]["hardware"], "表中物理硬件不符")
            gm_ms = math.exp(sum(math.log(x["latency_ms"]) for x in point_rows[pid]) / len(heldout))
            max_error = max(max_error, _close(r["heldout_geomean_ms"], gm_ms, "延迟GM"))
            for column, name in (("GM_ratio", "geomean"), ("CVaR10_ratio", "cvar10"), ("worst_batch_ratio", "minimax")):
                max_error = max(max_error, _close(r[column], computed[j][name], column))
            actual_batch = json.loads(r["batch_ratios"])
            assert set(actual_batch) == {str(b) for b in computed[j]["batch_geomeans"]}, "batch目标覆盖不符"
            for b, value in computed[j]["batch_geomeans"].items():
                max_error = max(max_error, _close(actual_batch[str(b)], value, "batch GM"))
            assert _true(r["is_primary_frozen_design"]) == (pid == group["frozen_id"]), "主硬件标记被改写"
            for name in OBJECTIVES:
                assert _true(r["selected_by_" + name]) == (j == win[name]), "后验赢家标记不符"
            for w, expected, paired in zip(heldout, point_rows[pid], ratio[j]):
                wk = (*gt, pid, w["id"]); seen["per_window"].add(wk)
                actual = per_window[wk]
                assert int(actual["batch"]) == w["batch"] and actual["geometry"] == r["geometry"], "逐窗口batch/几何不符"
                for col, val in (("latency_ms", expected["latency_ms"]), ("hbm_MiB", expected["hbm_bytes"] / 2**20), ("ratio_vs_C0_B1", paired)):
                    max_error = max(max_error, _close(actual[col], val, "逐窗口 " + col))
        for name, j in win.items():
            k = (*gt, name); seen["winners"].add(k); r = winners[k]
            pid = ids[j]
            assert r["point_id"] == pid and r["geometry"] == _geometry(points[pid]["hardware"]), "赢家不是完整物理配置最小值"
            max_error = max(max_error, _close(r["objective_ratio"], computed[j][name], "赢家目标值"))
            assert _true(r["three_objectives_same_physical_design"]) == agree, "三目标一致性标签不符"
            assert _true(r["matches_primary_frozen_design"]) == (pid == group["frozen_id"]), "主硬件匹配标签不符"
            assert int(r["candidate_count"]) == len(ids) and _true(r["diagnostic_only"]), "赢家范围/诊断标签不符"
        devbase = np.asarray(next(g["development_ms"][common[mode]] for g in groups if g["onchip_mode"] == mode and common[mode] in g["development_ms"]))
        matrix = [np.asarray(group["development_ms"][i]) / devbase for i in ids]
        counts = _bootstrap(matrix, [w["batch"] for w in dev])
        for j, pid in enumerate(ids):
            for name in OBJECTIVES:
                k = (*gt, pid, name); seen["stability"].add(k); r = stability[k]
                assert int(r["draws"]) == 200 and int(r["selected_count"]) == counts[name][j], "开发bootstrap计数不符"
                max_error = max(max_error, _close(r["selected_share"], counts[name][j] / 200, "bootstrap份额"))
                assert _true(r["most_selected"]) == (counts[name][j] == max(counts[name])), "bootstrap最常选标记不符"
                assert _true(r["diagnostic_only"]), "bootstrap不是诊断标记"
    for name, index in (("metrics", metrics), ("winners", winners), ("stability", stability), ("per_window", per_window)):
        assert seen[name] == set(index), name + ": 缺行或多余行"
    return {"max_objective_error": max_error, "objective_rows": len(metrics), "winner_rows": len(winners),
            "bootstrap_rows": len(stability), "per_window_rows": len(per_window)}


def audit(root):
    """返回完整JSON兼容审计结果；缺证据不通过，不进行任何写入/仿真。"""
    root = Path(root).resolve()
    out = root / "E4/robust_heldout"
    answer = {"stage": "E4_near_best_heldout_robustness", "complete": False, "passed": False,
              "failures": [], "coverage": {}, "max_objective_error": 0., "checks": [],
              "evidence_sha256": {"audit_source": _sha(Path(__file__))}}
    missing = [name for name in REQUIRED if not (out / name).is_file()]
    if missing:
        answer["failures"] = ["诊断尚未闭合，缺少: " + ", ".join(missing)]
        return answer
    answer["complete"] = True
    def check(name, action):
        try:
            value = action()
            answer["checks"].append({"name": name, "passed": True})
            return value
        except Exception as error:
            message = name + ": " + type(error).__name__ + ": " + str(error)
            answer["failures"].append(message)
            answer["checks"].append({"name": name, "passed": False})
            return None
    def structural():
        plan = _json(out / "CANDIDATE_PLAN.json")
        run = _json(out / "RUN_RECEIPT.json")
        complete = _json(out / "COMPLETE.json")
        assert complete["all_unique_points_repeated"] is True and complete["diagnostic_only"] is True
        assert complete["selected_hardware_unchanged"] is True
        assert complete["receipt_sha256"] == _sha(out / "RUN_RECEIPT.json"), "完成标记回执SHA不符"
        assert run["candidate_plan_sha256"] == _sha(out / "CANDIDATE_PLAN.json"), "计划SHA不符"
        for field in ("exact_repeat_all", "source_unchanged", "selected_hardware_unchanged"):
            assert run[field] is True, "运行冻结/重复声明不符: " + field
        assert complete["unique_points"] == run["unique_points"] == EXPECTED_POINTS
        assert run["group_candidate_rows"] == EXPECTED_GROUP_CANDIDATES
        assert run["new_points"] == 175 and run["reused_points"] == 12 and run["new_physical_window_passes"] == 47250
        assert plan["selected_designs_sha256"] == _sha(root / "E4/selected_designs.json"), "冻结主硬件SHA变化"
        _equal(run["source_sha256"], plan["physical_source_sha256"], "计划/运行物理源SHA不一致")
        for path, expected in plan["physical_source_sha256"].items():
            assert _sha(path) == expected, "物理源变化: " + path
        manifest_path = root.parent / "round2/results/E0/frozen_inputs.json"
        assert run["input_manifest_sha256"] == plan["input_manifest_sha256"] == _sha(manifest_path), "输入清单SHA不符"
        manifest = _json(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            assert _sha(Path(manifest["input_directory"]) / name) == expected, "真实路由输入SHA不符: " + name
        from .common import inputs
        windows = inputs()
        dev, heldout = windows["development"], windows["heldout"]
        assert len(dev) == 18 and len(heldout) == 135
        _equal(plan["heldout_ids"], [w["id"] for w in heldout], "冻结留出窗口顺序不符")
        _equal(plan["development_ids"], [w["id"] for w in dev], "冻结开发窗口顺序不符")
        _equal(plan["heldout_ids"], manifest["heldout_window_ids"], "路由清单留出顺序不符")
        _equal(plan["development_ids"], manifest["development_window_ids"], "路由清单开发顺序不符")
        for path, expected in run["output_sha256"].items():
            assert _sha(root / path) == expected, "输出字节SHA不符: " + path
        assert set(REQUIRED) - {"RUN_RECEIPT.json", "COMPLETE.json"} <= {Path(p).name for p in run["output_sha256"]}, "输出SHA清单不完整"
        answer["evidence_sha256"].update({name: _sha(out / name) for name in REQUIRED})
        return plan, run, dev, heldout, _json(root / "E4/selected_designs.json")
    structural_result = check("冻结来源、输入、输出与完成标记", structural)
    if structural_result is None:
        return answer
    plan, run, dev, heldout, selection = structural_result
    points, groups = plan["points"], plan["groups"]
    def candidates():
        assert len(points) == EXPECTED_POINTS and len(groups) == 20
        assert len(plan["certificate_sha256"]) == 20
        expected_groups = {(m, c, f) for m in MODES for c in CONSTRAINTS for f in FAMILIES}
        assert {(g["onchip_mode"], g["constraint_group"], g["family"]) for g in groups} == expected_groups
        assert sum(len(g["candidate_ids"]) for g in groups) == EXPECTED_GROUP_CANDIDATES
        assert {i for g in groups for i in g["candidate_ids"]} == set(points), "唯一点与组成员不闭合"
        assert len({(p["onchip_mode"], p["physical_key"]) for p in points.values()}) == EXPECTED_POINTS
        for pid, point in points.items():
            assert point["point_id"] == pid == _point_id(point["onchip_mode"], point["hardware"])
            assert point["physical_key"] == _physical_key(point["hardware"])
        count_rows = _indexed(_csv(out / "candidate_counts.csv"), ("onchip_mode", "constraint_group", "family"), "候选数")
        assert set(count_rows) == expected_groups
        development_by_point = {}
        for g in groups:
            certpath = root / g["certificate_file"]
            assert _sha(certpath) == g["certificate_sha256"] == plan["certificate_sha256"][str(certpath)], "开发证书SHA不符"
            cert = _json(certpath)
            for field in ("onchip_mode", "constraint_group", "family"):
                assert cert[field] == g[field], "开发证书组不符"
            best = cert["selected"]["score_ms"]
            near = [r for r in cert["witnesses"] if r.get("status", "evaluated") == "evaluated" and r["score_ms"] <= best * 1.01 + 1e-12]
            near.sort(key=lambda r: _physical_key(r["design"]))
            ids = [_point_id(g["onchip_mode"], r["design"]) for r in near]
            _equal(ids, g["candidate_ids"], "开发1%近优集合被改写")
            assert len(ids) == len(set(ids)), "证书近优候选重复"
            _equal(g["development_ms"], {i: r["latencies_ms"] for i, r in zip(ids, near)}, "开发逐窗口结果不符")
            assert all(len(v) == 18 for v in g["development_ms"].values())
            for i in ids:
                _equal(points[i]["parameters"], cert["parameters"], "跨组物理参数不符")
                if i in development_by_point:
                    _equal(g["development_ms"][i], development_by_point[i], "同一物理点跨组开发向量不同")
                development_by_point[i] = g["development_ms"][i]
            for r in near:
                score = math.exp(sum(math.log(v) for v in r["latencies_ms"]) / 18)
                _close(r["score_ms"], score, "开发近优候选GM")
            assert g["frozen_id"] == _point_id(g["onchip_mode"], selection["modes"][g["onchip_mode"]][g["constraint_group"]][LABELS[g["family"]]])
            assert g["frozen_id"] in ids, "主硬件未包含在冻结开发近优集合"
            assert g["proof_B_closed"] == cert["proof_B_closed"]
            _close(g["development_best_ms"], best, "开发最好值")
            k = (g["onchip_mode"], g["constraint_group"], g["family"])
            assert int(count_rows[k]["near_candidates"]) == len(ids)
            assert int(count_rows[k]["naive_physical_window_passes"]) == len(ids) * 270
        all_groups = _diagnostic_groups(groups, points)
        assert len(all_groups) == 24
        return all_groups
    diagnostic_groups = check("20开发近优集合、187唯一点与24诊断组", candidates)
    if diagnostic_groups is None:
        return answer
    repeats = check("汇总双遍回执", lambda: _indexed(_json(out / "repeat_checks.json"), ("point_id",), "双遍回执"))
    if repeats is None:
        return answer
    assert_keys = lambda: _equal(sorted(k[0] for k in repeats), sorted(points), "双遍点覆盖不完整")
    if check("187点双遍覆盖", assert_keys) is None and answer["failures"]:
        return answer
    rows = {}
    new_count = reused_count = 0
    for pid, point in sorted(points.items()):
        def action():
            rec = _json(out / "points" / (pid + ".json"))
            _equal(rec, repeats[(pid,)], "单点回执与汇总不一致")
            return rec, _check_point(root, point, rec, heldout, plan["physical_source_sha256"])
        result = check("逐窗口物理证据 " + pid, action)
        if result is not None:
            rec, rows[pid] = result
            new_count += int(rec["kind"].startswith("new"))
            reused_count += int(rec["kind"].startswith("existing"))
    answer["coverage"].update({"planned_groups": 20, "diagnostic_groups": 24, "unique_points": len(points),
                               "group_candidate_rows": EXPECTED_GROUP_CANDIDATES, "raw_checks": len(rows),
                               "window_order_checks": len(rows) * 135,
                               "new_points": new_count, "reused_points": reused_count})
    if len(rows) == len(points):
        check("新增及严格复用数", lambda: _equal([new_count, reused_count], [175, 12], "实际新增/复用点数不符"))
        recomputed = check("三目标、赢家、逐窗口与200开发bootstrap独立重算",
                           lambda: _audit_tables(out, diagnostic_groups, points, rows, selection, dev, heldout))
        if recomputed is not None:
            answer["max_objective_error"] = recomputed.pop("max_objective_error")
            answer["coverage"].update(recomputed)
    answer["passed"] = not answer["failures"]
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
