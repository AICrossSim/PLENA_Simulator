"""Build dispatch-repair evidence from immutable per-window runner results.

No simulation or parameter selection happens here. All output paths are inside
the explicitly supplied directory; frozen round-two files are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


BATCHES = (2, 4, 8, 16, 64, 96, 128)
DESIGNS = ("B0", "B1", "B2", "best_hetero", "fixed_4+2")
MODES = ("pipelined", "port_tight")
DISPATCHES = ("eft_old", "fixed", "milp")
CONTROL = "eft_ours_control"
CASE_ID = "v3_captured_mixed_heldout_gpqa_t128_l13"
COMPARE_FIELDS = ("design", "onchip_mode", "dispatch", *(f"B{b}" for b in BATCHES),
                  "all_geomean_ms", "ratio_vs_milp", "ratio_vs_B1_fixed")
HBM_FIELDS = ("window_id", "design", "onchip_mode", "dispatch", "hbm_MiB",
              "hbm_MiB_milp", "extra_pct", "refetch_tasks")
REGRESSION_FIELDS = ("design", "onchip_mode", "window_id", "batch", "eft_old_cycles",
                     "fixed_cycles", "old_digest", "fixed_digest", "bit_exact")
CI_FIELDS = ("design", "onchip_mode", "baseline", "paired_windows", "ratio",
             "improvement_pct", "ci95_low_improvement_pct", "ci95_high_improvement_pct")
EXCESS_FIELDS = ("window_id", "design", "onchip_mode", "dispatch", "batch", "hbm_MiB",
                 "hbm_MiB_milp", "extra_pct", "native_unique_MiB", "refetch_tasks",
                 "zero_refetch_alternative_tasks", "all_cores_refetch_tasks",
                 "all_cores_refetch_but_lower_traffic_alternative_tasks", "chosen_above_minimum_MiB", "reasons",
                 "refetch_details")
TASK_DELTA_FIELDS = ("window_id", "design", "onchip_mode", "batch", "expert_index", "expert_id",
                     "Me", "is_shared", "fixed_core", "milp_core", "unique_hbm_bytes",
                     "fixed_hbm_bytes", "milp_hbm_bytes", "delta_bytes", "delta_MiB",
                     "fixed_refetch_factor", "milp_refetch_factor", "best_refetch_factor",
                     "best_refetch_core", "decision_kind", "bind_cycle", "candidate_comparisons")


def gmean(values: Iterable[float]) -> float:
    values = list(values)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("geometric mean requires nonempty finite positive values")
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def _input_bytes(path: Path):
    if path.exists():
        return path.read_bytes()
    compressed = path.with_name(path.name + ".gz")
    with gzip.open(compressed, "rb") as stream:
        return stream.read()


def _load(path: Path):
    return json.loads(_input_bytes(path))


def _dump_csv(path: Path, rows: Iterable[dict], fields: Iterable[str]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _key(row: dict):
    return row["design"], row["onchip_mode"], row["dispatch"], row["window_id"]


def _sort_key(row: dict):
    return (MODES.index(row["onchip_mode"]), DESIGNS.index(row["design"]),
            (*DISPATCHES, CONTROL).index(row["dispatch"]), row["batch"], row["window_id"])


def _rows(payload) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("per_window"))
    if not isinstance(payload, list):
        raise ValueError("per_window.json must contain a row list")
    rows = []
    seen = set()
    for original in payload:
        # Selection uses development only; headline tables never include it.
        if original.get("split", original.get("cohort", "heldout")) not in ("heldout", "test"):
            continue
        row = dict(original)
        if row["design"] not in DESIGNS or row["onchip_mode"] not in MODES:
            raise ValueError(f"unexpected frozen design/mode: {_key(row)}")
        if row["dispatch"] not in (*DISPATCHES, CONTROL):
            raise ValueError(f"unexpected dispatch: {row['dispatch']}")
        row["batch"] = int(row["batch"])
        if row["batch"] not in BATCHES:
            raise ValueError(f"unexpected heldout batch: {row['batch']}")
        for name in ("cycles", "latency_ms", "hbm_bytes", "native_unique_bytes"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError(f"invalid {name}: {_key(row)}")
        if not math.isclose(row["latency_ms"], row["cycles"] / 1e6, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"cycles/ms mismatch: {_key(row)}")
        row["refetch_tasks"] = int(row.get("refetch_tasks", len(row.get("refetch_details", []))))
        if _key(row) in seen:
            raise ValueError(f"duplicate heldout row: {_key(row)}")
        seen.add(_key(row))
        rows.append(row)
    if not rows:
        raise ValueError("no heldout rows")
    return sorted(rows, key=_sort_key)


def _paired(rows: list[dict], design: str, mode: str, dispatch: str,
            baseline_design: str, baseline_dispatch: str):
    candidate = {r["window_id"]: r for r in rows if
                 (r["design"], r["onchip_mode"], r["dispatch"]) == (design, mode, dispatch)}
    baseline = {r["window_id"]: r for r in rows if
                (r["design"], r["onchip_mode"], r["dispatch"]) ==
                (baseline_design, mode, baseline_dispatch)}
    if set(candidate) != set(baseline):
        raise ValueError(f"unpaired windows: {design}/{mode}/{dispatch} versus "
                         f"{baseline_design}/{baseline_dispatch}")
    for window_id in candidate:
        if candidate[window_id]["batch"] != baseline[window_id]["batch"]:
            raise ValueError(f"paired batch mismatch: {window_id}")
    return [candidate[i]["latency_ms"] / baseline[i]["latency_ms"] for i in sorted(candidate)]


def comparison_rows(rows: list[dict], dispatches=DISPATCHES):
    result = []
    for mode in MODES:
        for design in DESIGNS:
            for dispatch in dispatches:
                group = [r for r in rows if
                         (r["design"], r["onchip_mode"], r["dispatch"]) == (design, mode, dispatch)]
                if not group:
                    continue
                row = {"design": design, "onchip_mode": mode, "dispatch": dispatch}
                for batch in BATCHES:
                    values = [r["latency_ms"] for r in group if r["batch"] == batch]
                    row[f"B{batch}"] = gmean(values) if values else ""
                row["all_geomean_ms"] = gmean(r["latency_ms"] for r in group)
                row["ratio_vs_milp"] = gmean(_paired(rows, design, mode, dispatch, design, "milp"))
                row["ratio_vs_B1_fixed"] = gmean(_paired(rows, design, mode, dispatch, "B1", "fixed"))
                result.append(row)
    return result


def hbm_rows(rows: list[dict], dispatches=DISPATCHES):
    by_key = {_key(r): r for r in rows}
    result = []
    for row in rows:
        if row["dispatch"] not in dispatches:
            continue
        milp = by_key[row["design"], row["onchip_mode"], "milp", row["window_id"]]
        result.append({"window_id": row["window_id"], "design": row["design"],
                       "onchip_mode": row["onchip_mode"], "dispatch": row["dispatch"],
                       "hbm_MiB": row["hbm_bytes"] / 2**20,
                       "hbm_MiB_milp": milp["hbm_bytes"] / 2**20,
                       "extra_pct": 100 * (row["hbm_bytes"] / milp["hbm_bytes"] - 1),
                       "refetch_tasks": row["refetch_tasks"]})
    return result


def _detail_reason(detail: dict) -> str:
    if float(detail.get("best_refetch_factor", 1)) > 1:
        return "all_cores_refetch"
    if float(detail.get("refetch_factor", 1)) > 1:
        return str(detail.get("decision_kind") or "refetch_with_zero_refetch_alternative")
    return str(detail.get("decision_kind") or detail.get("reason") or "unclassified")


def excess_rows(rows: list[dict], traffic: list[dict], dispatches=("eft_old", "fixed")):
    by_key = {_key(r): r for r in rows}
    result = []
    for hbm in traffic:
        if hbm["dispatch"] not in dispatches or hbm["extra_pct"] <= 2 + 1e-10:
            continue
        row = by_key[_key(hbm)]
        details = row.get("refetch_details", [])
        reasons = Counter(_detail_reason(d) for d in details)
        if not details:
            reasons["no_per_task_refetch_detail_supplied"] = 1
        result.append({**hbm, "batch": row["batch"],
                       "native_unique_MiB": row["native_unique_bytes"] / 2**20,
                       "zero_refetch_alternative_tasks": sum(float(d.get("best_refetch_factor", 1)) <= 1 and
                                                             float(d.get("refetch_factor", 1)) > 1 for d in details),
                       "all_cores_refetch_tasks": sum(float(d.get("best_refetch_factor", 1)) > 1 for d in details),
                       "all_cores_refetch_but_lower_traffic_alternative_tasks":
                           sum(float(d.get("best_refetch_factor", 1)) > 1 and
                               float(d.get("chosen_hbm_bytes", 0)) > float(d.get("minimum_hbm_bytes", 0))
                               for d in details),
                       "chosen_above_minimum_MiB": sum(float(d.get("excess_B", 0)) for d in details) / 2**20,
                       "reasons": _json(dict(sorted(reasons.items()))), "refetch_details": _json(details)})
    return result


def regression_rows(rows: list[dict]):
    by_key = {_key(r): r for r in rows}
    result = []
    for fixed in rows:
        if fixed["design"] not in ("B0", "B1") or fixed["dispatch"] != "fixed":
            continue
        old = by_key[fixed["design"], fixed["onchip_mode"], "eft_old", fixed["window_id"]]
        detail = fixed.get("regression_detail", {})
        old_digest = detail.get("old_digest", old.get("result_digest", ""))
        fixed_digest = detail.get("fixed_digest", fixed.get("result_digest", ""))
        exact = bool(old_digest and old_digest == fixed_digest and old["cycles"] == fixed["cycles"])
        result.append({"design": fixed["design"], "onchip_mode": fixed["onchip_mode"],
                       "window_id": fixed["window_id"], "batch": fixed["batch"],
                       "eft_old_cycles": old["cycles"], "fixed_cycles": fixed["cycles"],
                       "old_digest": old_digest, "fixed_digest": fixed_digest, "bit_exact": exact})
    return result


def task_hbm_deltas(rows: list[dict]):
    """Attribute layer wire-byte changes without counting shared compulsory reads.

    Refetch details include the exact declared task bytes. A task omitted from
    one side is clean there, so its wire bytes equal the reconstructed unique
    bytes. Tasks omitted on both sides contribute zero and need no delta row.
    """
    by_key = {_key(r): r for r in rows}
    output = []
    for fixed in rows:
        if fixed["dispatch"] != "fixed":
            continue
        milp = by_key[fixed["design"], fixed["onchip_mode"], "milp", fixed["window_id"]]
        fd = {d["expert_index"]: d for d in fixed.get("refetch_details", [])}
        md = {d["expert_index"]: d for d in milp.get("refetch_details", [])}
        fixed_tasks = {t["expert_index"]: t for t in fixed.get("tasks", [])}
        milp_tasks = {t["expert_index"]: t for t in milp.get("tasks", [])}
        deltas = []
        for index in sorted(fd.keys() | md.keys()):
            fixed_detail, milp_detail = fd.get(index), md.get(index)
            detail = fixed_detail or milp_detail
            unique = float(detail["chosen_hbm_bytes"]) / float(detail["refetch_factor"])
            if not math.isclose(unique, round(unique), rel_tol=0, abs_tol=1e-3):
                raise ValueError(f"nonintegral reconstructed unique task bytes: {_key(fixed)}/{index}")
            unique = round(unique)
            if fixed_detail and milp_detail:
                other_unique = float(milp_detail["chosen_hbm_bytes"]) / float(milp_detail["refetch_factor"])
                if not math.isclose(unique, other_unique, rel_tol=0, abs_tol=1e-3):
                    raise ValueError(f"task unique bytes differ across schedules: {_key(fixed)}/{index}")
            fixed_bytes = float(fixed_detail["chosen_hbm_bytes"]) if fixed_detail else unique
            milp_bytes = float(milp_detail["chosen_hbm_bytes"]) if milp_detail else unique
            delta = fixed_bytes - milp_bytes
            deltas.append(delta)
            if delta == 0:
                continue
            ft, mt = fixed_tasks.get(index, {}), milp_tasks.get(index, {})
            output.append({"window_id": fixed["window_id"], "design": fixed["design"],
                           "onchip_mode": fixed["onchip_mode"], "batch": fixed["batch"],
                           "expert_index": index, "expert_id": detail.get("expert_id", index),
                           "Me": detail.get("Me", ""), "is_shared": detail.get("is_shared", False),
                           "fixed_core": fixed_detail.get("core") if fixed_detail else ft.get("core", ""),
                           "milp_core": milp_detail.get("core") if milp_detail else mt.get("core", ""),
                           "unique_hbm_bytes": unique, "fixed_hbm_bytes": fixed_bytes,
                           "milp_hbm_bytes": milp_bytes, "delta_bytes": delta, "delta_MiB": delta / 2**20,
                           "fixed_refetch_factor": fixed_bytes / unique,
                           "milp_refetch_factor": milp_bytes / unique,
                           "best_refetch_factor": detail.get("best_refetch_factor", ""),
                           "best_refetch_core": detail.get("best_refetch_core", ""),
                           "decision_kind": fixed_detail.get("decision_kind", "unknown") if fixed_detail else "clean",
                           "bind_cycle": fixed_detail.get("bind_cycle", "") if fixed_detail else
                               next((b.get("bind_cycle", "") for b in fixed.get("bindings", [])
                                     if b["expert_index"] == index), ""),
                           "candidate_comparisons": _json(fixed_detail.get("candidate_comparisons", []))
                               if fixed_detail else "[]"})
        layer_delta = fixed["hbm_bytes"] - milp["hbm_bytes"]
        if not math.isclose(math.fsum(deltas), layer_delta, rel_tol=0, abs_tol=1e-3):
            raise ValueError(f"task/layer HBM delta mismatch: {_key(fixed)}: "
                             f"tasks={math.fsum(deltas)}, layer={layer_delta}")
    return output


def confidence_rows(rows: list[dict]):
    from ..common import paired_ci

    result = []
    for mode in MODES:
        for design in DESIGNS:
            if not any((r["design"], r["onchip_mode"], r["dispatch"]) == (design, mode, "fixed") for r in rows):
                continue
            baselines = [("B1_fixed", "B1", "fixed"), ("B2_fixed", "B2", "fixed"),
                         ("eft_old", design, "eft_old"), ("milp", design, "milp")]
            if any((r["design"], r["onchip_mode"], r["dispatch"]) == (design, mode, CONTROL) for r in rows):
                baselines.append((CONTROL, design, CONTROL))
            for name, baseline_design, baseline_dispatch in baselines:
                ratios = _paired(rows, design, mode, "fixed", baseline_design, baseline_dispatch)
                lo, hi = paired_ci(ratios)
                ratio = gmean(ratios)
                result.append({"design": design, "onchip_mode": mode, "baseline": name,
                               "paired_windows": len(ratios), "ratio": ratio,
                               "improvement_pct": 100 * (1 - ratio),
                               "ci95_low_improvement_pct": 100 * lo,
                               "ci95_high_improvement_pct": 100 * hi})
    return result


def _fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _md_table(fields, values):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(("| " + " | ".join(fields) + " |",
                      "| " + " | ".join("---" for _ in fields) + " |",
                      *("| " + " | ".join(safe(v) for v in row) + " |" for row in values)))


def _geometry(frozen: dict, mode: str, design: str) -> str:
    value = frozen.get("modes", {}).get(mode, {}).get(design)
    if value is None:
        return "not supplied"
    if isinstance(value, str):
        return value
    cores = value.get("cores", [])
    if cores and isinstance(cores[0], dict):
        return "+".join("x".join(str(core[k]) for k in ("pm", "pn", "pk")) for core in cores)
    return "+".join("x".join(map(str, core)) for core in cores)


def case_markdown(rows: list[dict], task_deltas=None):
    case_rows = [r for r in rows if r["window_id"] == CASE_ID and r["dispatch"] in DISPATCHES]
    lines = ["# 冻结 GPQA T128 Shared 因果实例", "", f"窗口：`{CASE_ID}`。", "",
             "本实例的目标是延迟与 HBM 都在同硬件 MILP/LPT 的 2% 以内（延迟比≤1.02）；"
             "全留出集几何均值的 1.01 调度差距门槛另列。", ""]
    if not case_rows:
        return "\n".join(lines + ["未提供此窗口，不能得出实例结论。", ""])
    table = []
    for row in case_rows:
        shared = row.get("shared_tasks", [])
        cores = "; ".join(str(t.get("core", "unknown")) for t in shared) or "not supplied"
        starts = "; ".join(_fmt(float(t.get("start_ms", t.get("start_cycles", 0) / 1e6))) for t in shared) or "not supplied"
        factors = "; ".join(_fmt(t.get("refetch_factor", "unknown")) for t in shared) or "not supplied"
        chunks = "; ".join(str(t.get("z_chunks", "unknown")) for t in shared) or "not supplied"
        table.append([row["design"], row["onchip_mode"], row["dispatch"], cores, starts,
                      _fmt(row["hbm_bytes"] / 2**20, 2), _fmt(row["native_unique_bytes"] / 2**20, 2),
                      _fmt(row["latency_ms"]), factors, chunks])
    lines += [_md_table(("设计", "模式", "调度", "Shared 核", "Shared 开始 ms", "HBM MiB",
                         "原生唯一权重 MiB", "整层 ms", "Shared 重读倍数", "Shared Z 分块"), table), ""]
    by_key = {_key(r): r for r in rows}
    for row in case_rows:
        if row["dispatch"] != "fixed":
            continue
        milp = by_key[row["design"], row["onchip_mode"], "milp", row["window_id"]]
        extra = 100 * (row["hbm_bytes"] / milp["hbm_bytes"] - 1)
        ratio = row["latency_ms"] / milp["latency_ms"]
        lines.append(f"{row['onchip_mode']}/{row['design']}：fixed/MILP 延迟比={ratio:.6f}；"
                     f"HBM 超额={extra:+.3f}%。2% 延迟门槛（≤1.02）{'通过' if ratio <= 1.02 else '未通过'}；"
                     f"2% HBM 门槛{'通过' if extra <= 2 else '未通过'}。")
        if extra > 2:
            details = row.get("refetch_details", [])
            counts = Counter(_detail_reason(d) for d in details)
            lines.append("仍有重读的任务原因：" + ("；".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                                                         or "未提供任务明细") + "。")
            shared_details = [d for d in details if d.get("is_shared")]
            for detail in shared_details:
                lines.append(f"Shared 专家 {detail.get('expert_id', detail.get('expert_index'))}："
                             f"绑定核 {detail.get('core')}，重读倍数 {float(detail.get('refetch_factor', 1)):.3f}；"
                             f"所有已安装核的最小倍数={float(detail.get('best_refetch_factor', 1)):.3f}，"
                             f"对应核 {detail.get('best_refetch_core')}；决策 {_detail_reason(detail)}。"
                             f"实际流量 {float(detail.get('chosen_hbm_bytes', 0)) / 2**20:.3f} MiB，"
                             f"已安装核最小流量 {float(detail.get('minimum_hbm_bytes', 0)) / 2**20:.3f} MiB。")
        changed = [d for d in (task_deltas or []) if
                   (d["window_id"], d["design"], d["onchip_mode"]) ==
                   (row["window_id"], row["design"], row["onchip_mode"])]
        if changed:
            lines.append("以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。"
                         "两者相同的强制重读贡献为零，未列入差值表。")
            lines.extend(("", _md_table(("专家", "Shared", "Me", "Fixed 核", "MILP 核", "Fixed r",
                                          "MILP r", "差值 MiB", "Fixed 决策"),
                                         [[d["expert_id"], d["is_shared"], d["Me"], d["fixed_core"], d["milp_core"],
                                           _fmt(d["fixed_refetch_factor"], 3), _fmt(d["milp_refetch_factor"], 3),
                                           _fmt(d["delta_MiB"], 3), d["decision_kind"]] for d in changed])))
        lines.append("")
    lines += ["Shared 开始时间是模拟任务开始时间，不是原生 HBM 请求时间戳。MILP 指冻结离线专家分配后"
              "的可执行 LPT 回放，不代表证明了任意事件调度最优性。候选核的 ETA、预测完成时间与可绑定状态"
              "保留在 `task_hbm_deltas.csv` 的 candidate_comparisons 中。", ""]
    return "\n".join(lines)


def _selection_text(selection: dict):
    chosen = selection.get("chosen", {})
    counts = selection.get("bootstrap_selection_counts", [])
    lines = [f"共同策略：t_big={chosen.get('t_big', '未提供')}，"
             f"large_first={chosen.get('large_first', '未提供')}。只在 18 个开发窗口上选择，"
             "目标是全部五个冻结设计×两模式相对旧 EFT 的配对延迟比几何均值；留出窗口只用于评估。"]
    if chosen.get("large_first") is False:
        lines.append("large_first=False 时阈值与优先排序均停用，五个 t_big 值因此对应同一调度策略。"
                     "分数相同时按确定性规则选择最小 t_big；此分支的 bootstrap 胜出不能解释为阈值稳健性。")
    if isinstance(counts, list) and counts:
        draws = int(selection.get("bootstrap_draws", sum(int(c["count"]) for c in counts)))
        winner_count = sum(int(c["count"]) for c in counts if
                           (c.get("t_big"), c.get("large_first")) ==
                           (chosen.get("t_big"), chosen.get("large_first")))
        lines += [f"Bootstrap 稳定性：所选参数对在 {winner_count}/{draws} 次重采样中胜出"
                  f"（{100 * winner_count / draws:.1f}%）。这是开发集选参稳定性，不证明普遍最优阈值。", "",
                  _md_table(("t_big", "large_first", "胜出次数", "比例"),
                            [[c["t_big"], c["large_first"], c["count"],
                              _fmt(c.get("fraction", int(c["count"]) / draws))] for c in counts])]
        branch_count = sum(int(c["count"]) for c in counts if c.get("large_first") == chosen.get("large_first"))
        lines += ["", f"所选 large_first 分支胜出 {branch_count}/{draws} 次"
                  f"（{100 * branch_count / draws:.1f}%）；分支稳定性与精确阈值参数对的稳定性不同。"]
    else:
        lines.append("未提供 bootstrap 选参次数。")
    return lines


def acceptance_details(rows, comparisons, traffic, regressions, task_deltas):
    """Publish numeric gates without silently upgrading a residual to success."""
    index = {(r["design"], r["onchip_mode"], r["dispatch"]): r for r in comparisons}
    by_key = {_key(r): r for r in rows}
    gm = []
    for mode in MODES:
        for design in DESIGNS:
            row = index.get((design, mode, "fixed"))
            if row is not None:
                gm.append({"design": design, "onchip_mode": mode,
                           "fixed_ms": row["all_geomean_ms"],
                           "eft_old_ms": index[design, mode, "eft_old"]["all_geomean_ms"],
                           "milp_ms": index[design, mode, "milp"]["all_geomean_ms"],
                           "fixed_vs_milp": row["ratio_vs_milp"],
                           "latency_within_1pct": row["ratio_vs_milp"] <= 1.01})
    cases = []
    for fixed in rows:
        if fixed["dispatch"] != "fixed" or fixed["window_id"] != CASE_ID:
            continue
        old = by_key[fixed["design"], fixed["onchip_mode"], "eft_old", CASE_ID]
        milp = by_key[fixed["design"], fixed["onchip_mode"], "milp", CASE_ID]
        ratio = fixed["latency_ms"] / milp["latency_ms"]
        extra = 100 * (fixed["hbm_bytes"] / milp["hbm_bytes"] - 1)
        cases.append({"design": fixed["design"], "onchip_mode": fixed["onchip_mode"],
                      "window_id": CASE_ID, "eft_old_ms": old["latency_ms"], "fixed_ms": fixed["latency_ms"],
                      "milp_ms": milp["latency_ms"], "fixed_vs_milp": ratio,
                      "latency_within_2pct": ratio <= 1.02,
                      "eft_old_hbm_MiB": old["hbm_bytes"] / 2**20,
                      "fixed_hbm_MiB": fixed["hbm_bytes"] / 2**20,
                      "milp_hbm_MiB": milp["hbm_bytes"] / 2**20,
                      "native_unique_MiB": fixed["native_unique_bytes"] / 2**20,
                      "hbm_extra_pct": extra, "hbm_within_2pct": extra <= 2,
                      "fixed_shared_tasks": fixed.get("shared_tasks", []),
                      "task_hbm_delta_MiB": sum(d["delta_MiB"] for d in task_deltas if
                                               (d["window_id"], d["design"], d["onchip_mode"]) ==
                                               (CASE_ID, fixed["design"], fixed["onchip_mode"]))})
    b2 = [r for r in gm if r["design"] == "B2"]
    main_cases = [r for r in cases if r["design"] == "best_hetero"]
    repeated = [r for r in rows if r.get("result_digest") and r.get("repeat_digest")]
    regression_ok = bool(regressions) and all(r["bit_exact"] for r in regressions)
    repeat_ok = len(repeated) == len(rows) and all(r["result_digest"] == r["repeat_digest"] for r in repeated)
    return {
        "units": {"latency": "ms", "traffic": "MiB (2^20 bytes)", "cycle_ns": 1},
        "thresholds": {"aggregate_latency_ratio": 1.01, "gpqa_case_latency_ratio": 1.02,
                       "gpqa_case_hbm_extra_pct": 2},
        "B2_geomean_latency_gates": b2,
        "B2_both_modes_within_1pct": len(b2) == 2 and all(r["latency_within_1pct"] for r in b2),
        "all_design_geomean_latency_diagnostics": gm,
        "gpqa_t128_cases": cases,
        "best_hetero_gpqa_both_modes_latency_within_2pct": len(main_cases) == 2 and
            all(r["latency_within_2pct"] for r in main_cases),
        "best_hetero_gpqa_both_modes_hbm_within_2pct": len(main_cases) == 2 and
            all(r["hbm_within_2pct"] for r in main_cases),
        "all_single_core_bit_exact": regression_ok,
        "all_per_window_repeat_hashes_match": repeat_ok,
        "fixed_hbm_windows_above_2pct": sum(r["dispatch"] == "fixed" and r["extra_pct"] > 2 + 1e-10
                                           for r in traffic),
        "task_layer_hbm_delta_conservation": True,
        "unmet_targets_remain": not (len(b2) == 2 and all(r["latency_within_1pct"] for r in b2) and
                                     len(main_cases) == 2 and all(r["latency_within_2pct"] and
                                                                 r["hbm_within_2pct"] for r in main_cases)),
        "conclusion_scope": "runtime ordering/binding repair; no heterogeneous winner claim",
    }


def summary_markdown(rows, selection, frozen, comparisons, traffic, exceptions, regressions, cis, task_deltas):
    index = {(r["design"], r["onchip_mode"], r["dispatch"]): r for r in comparisons}
    ci_index = {(r["design"], r["onchip_mode"], r["baseline"]): r for r in cis}
    controls = {(r["design"], r["onchip_mode"]): r for r in comparison_rows(rows, (CONTROL,))}
    acceptance = acceptance_details(rows, comparisons, traffic, regressions, task_deltas)
    exception_counts = Counter((r["design"], r["onchip_mode"], r["dispatch"]) for r in exceptions)
    chosen = selection.get("chosen", {})
    counts = selection.get("bootstrap_selection_counts", [])
    draws = int(selection.get("bootstrap_draws", 200))
    wins = sum(int(c["count"]) for c in counts if (c.get("t_big"), c.get("large_first")) ==
               (chosen.get("t_big"), chosen.get("large_first")))
    branch_wins = sum(int(c["count"]) for c in counts if c.get("large_first") == chosen.get("large_first"))
    lines = ["# 容量感知运行时调度修复", "",
             "本次修复取得部分改善，但未通过全部验收目标。结论仅为冻结 E4 硬件上的运行时修复，"
             "不构成异构硬件胜出。整体留出集几何均值（GM）要求fixed/MILP≤1.01；指定GPQA实例要求"
             "延迟和HBM都在2%以内（延迟比≤1.02）。比值越小越快，95%改善下界越大越好，正值代表更快。", "",
             "## 四个问题的直接回答", ""]
    questions = [[] for _ in range(4)]
    for mode in MODES:
        gm = []
        for design in ("B2", "best_hetero", "fixed_4+2"):
            old, fixed, milp = (index.get((design, mode, d)) for d in DISPATCHES)
            if fixed and old and milp:
                gm.append(f"{design}: old/fixed GM={old['all_geomean_ms']:.4f}/{fixed['all_geomean_ms']:.4f} ms, "
                          f"MILP={milp['all_geomean_ms']:.4f} ms, fixed/MILP={fixed['ratio_vs_milp']:.6f} "
                          f"({'通过' if fixed['ratio_vs_milp'] <= 1.01 else '未通过'}1.01)")
        questions[0].append("<br>".join(gm) or "未提供")
        hbm = [f"{d}: old {exception_counts[d, mode, 'eft_old']} / fixed {exception_counts[d, mode, 'fixed']} 个"
               for d in ("B2", "best_hetero", "fixed_4+2")]
        case = next((c for c in acceptance["gpqa_t128_cases"] if c["design"] == "best_hetero" and
                     c["onchip_mode"] == mode), None)
        if case:
            hbm.append(f"GPQA best_hetero: 延迟比{case['fixed_vs_milp']:.6f}, HBM超额+{case['hbm_extra_pct']:.3f}%; "
                       f"{'通过' if case['latency_within_2pct'] and case['hbm_within_2pct'] else '未通过'}实例2%目标")
        questions[1].append("<br>".join(hbm))
        a, b = ci_index.get(("best_hetero", mode, "B1_fixed")), ci_index.get(("best_hetero", mode, "B2_fixed"))
        questions[2].append(f"best_hetero fixed/B1={a['ratio']:.6f}, 95%改善下界{a['ci95_low_improvement_pct']:.3f}%; "
                            f"fixed/B2={b['ratio']:.6f}, 下界{b['ci95_low_improvement_pct']:.3f}%"
                            if a and b else "未提供")
        questions[3].append(f"共同t_big={chosen.get('t_big')}, large_first={chosen.get('large_first')}; "
                            f"参数对胜出{wins}/{draws}, 排序分支胜出{branch_wins}/{draws}; 只用18个开发窗口")
    labels = ("1. 同构/异构在线与离线GM差距", "2. 在线HBM>2%例外与GPQA目标",
              "3. Fixed相对B1/B2的比值及配对95%下界", "4. 共同参数与bootstrap稳定性")
    lines += [_md_table(("问题", *MODES), [[label, *values] for label, values in zip(labels, questions)]), "",
              "`acceptance.json` 保留精确数值、门槛布尔值和未达标状态，所有失败项按实列出。", "",
              "## 主延迟表：同硬件在线/离线", ""]
    timing, speeds, control_table = [], [], []
    for mode in MODES:
        for design in DESIGNS:
            fixed = index.get((design, mode, "fixed"))
            if not fixed:
                continue
            old, milp = index[design, mode, "eft_old"], index[design, mode, "milp"]
            timing.append([design, mode, _fmt(old["all_geomean_ms"]), _fmt(fixed["all_geomean_ms"]),
                           _fmt(milp["all_geomean_ms"]), _fmt(old["ratio_vs_milp"], 6),
                           _fmt(fixed["ratio_vs_milp"], 6), "通过" if fixed["ratio_vs_milp"] <= 1.01 else "未通过"])
            a, b = ci_index[design, mode, "B1_fixed"], ci_index[design, mode, "B2_fixed"]
            speeds.append([design, mode, _fmt(a["ratio"], 6), _fmt(a["ci95_low_improvement_pct"], 3),
                           _fmt(b["ratio"], 6), _fmt(b["ci95_low_improvement_pct"], 3)])
            if (design, mode) in controls:
                policy = gmean(_paired(rows, design, mode, "fixed", design, CONTROL))
                predictor = gmean(_paired(rows, design, mode, CONTROL, design, "eft_old"))
                combined = gmean(_paired(rows, design, mode, "fixed", design, "eft_old"))
                control_table.append([design, mode, _fmt(policy, 6), _fmt(predictor, 6), _fmt(combined, 6)])
    lines += [_md_table(("设计", "模式", "旧EFT GM ms", "Fixed GM ms", "MILP/LPT GM ms", "Old/MILP",
                         "Fixed/MILP", "GM≤1.01"), timing), "", "## 相对B1/B2与配对置信下界", "",
              _md_table(("设计", "模式", "Fixed/B1", "相对B1改善95%下界 %", "Fixed/B2", "相对B2改善95%下界 %"), speeds), "",
              "改善=100×(1−候选/基线)。配对百分位bootstrap为2,000次、seed=20261007；完整双侧区间、"
              "配对窗口数在`paired_improvement.csv`。B0/B1为单核、B2为同构双核、另两者为异构双核。", "",
              "## 所有HBM>2%例外与未达标原因", ""]
    hbm_table = []
    for mode in MODES:
        for design in DESIGNS:
            for dispatch in ("eft_old", "fixed"):
                group = [r for r in traffic if (r["design"], r["onchip_mode"], r["dispatch"]) == (design, mode, dispatch)]
                if group:
                    hbm_table.append([design, mode, dispatch, exception_counts[design, mode, dispatch], len(group),
                                      _fmt(max(r["extra_pct"] for r in group), 3)])
    lines += [_md_table(("设计", "模式", "在线调度", ">2%例外", "窗口数", "最大超额 %"), hbm_table), "",
              "HBM超额相对同窗口/设计/模式MILP/LPT实际流量计算，原生唯一权重字节数另列。"
              "`hbm_bytes.csv`列全部主比较，`excess_hbm_windows.csv`完整列出旧EFT/Fixed所有>2%例外、"
              "原因和任务明细；控制组例外单列。", ""]
    fixed_exceptions = [r for r in exceptions if r["dispatch"] == "fixed"]
    keys = {(r["window_id"], r["design"], r["onchip_mode"]) for r in fixed_exceptions}
    changed = [d for d in task_deltas if (d["window_id"], d["design"], d["onchip_mode"]) in keys]
    positive, negative = [d for d in changed if d["delta_bytes"] > 0], [d for d in changed if d["delta_bytes"] < 0]
    clean = sum(float(d["best_refetch_factor"]) == 1 for d in positive)
    all_refetch = sum(float(d["best_refetch_factor"]) > 1 for d in positive)
    lines += [f"Fixed仍有{len(fixed_exceptions)}个>2%窗口。按同一专家比较fixed−MILP，"
              f"其中{len(positive)}个任务增加{sum(d['delta_MiB'] for d in positive):.3f} MiB，"
              f"{len(negative)}个任务减少{-sum(d['delta_MiB'] for d in negative):.3f} MiB；"
              f"增量任务中{clean}个有r=1替代核，{all_refetch}个所有核r>1。`task_hbm_deltas.csv`保留正负差值"
              "与候选核A/B的ETA/预测完成/可绑定状态，并核对每个窗口任务差值之和精确等于整层差值。"
              "两种调度相同的强制重读贡献为零，不应归因于残留。", "",
              "规则1只在候选r>1且另一个已安装核r=1时触发：若等待无重读核后的预测完成时间不晚，"
              "才拒绝当前候选；ETA认为立即重读更快时允许重读。所有核r>1时保留EFT，未保证最低流量。"
              "例如Shared128大核r=2、小核r=26，只有部分重读不可避免，额外24次仍取决于归属。"
              "这些是当前未达标原因；未来ETA不会提前授予实际资源。", "",
              "具体GPQA的旧/Fixed/MILP Shared归属、开始、HBM、延迟、2%验收及真正改变的专家流量"
              "见`gpqa_t128_case.md`。", "", "## 预测器控制", "",
              "`compare.csv`只含原E4 EFT、Fixed、MILP/LPT；双核Fixed使用未修改的predictor='ours'。"
              "原EFT+ours单列在`compare_predictor_control.csv`。Fixed/control使用同一预测算法与开发集预热协议，"
              "但各自调度会形成不同反馈状态，因此不是排除状态交互的纯因果减法。B0/B1保留predictor=None。", "",
              _md_table(("设计", "模式", "Fixed/control", "Control/old", "Fixed/old"), control_table), "",
              "## 共同参数与稳定性", "", *_selection_text(selection), "", "## 精确回归、冻结硬件与边界", ""]
    passed = sum(bool(r["bit_exact"]) for r in regressions)
    repeated = [r for r in rows if r.get("result_digest") and r.get("repeat_digest")]
    repeat_passed = sum(r["result_digest"] == r["repeat_digest"] for r in repeated)
    lines += [f"B0/B1逐窗口回归{passed}/{len(regressions)}项cycles与完整结果哈希均bit exact；"
              f"独立完整重复运行哈希一致{repeat_passed}/{len(repeated)}项，{len(rows)-len(repeated)}行缺重复哈希。"
              "详见`regression.csv`；仅延迟相同不足以通过。", "",
              _md_table(("设计", *MODES), [[d, *[_geometry(frozen, m, d) for m in MODES]] for d in DESIGNS]), "",
              "几何、私有SRAM、端口、数据流全部冻结。pipelined B2为2x24x128双核且数据流OS/WS；"
              "port_tight B2为3x16x128双核且WS/WS，同构几何不保证两核成本相同。", "",
              "范围：post-router BF16 Gate/Up、SiLU/Z、Down、combine的phase-fluid解析模拟。"
              "1假想cycle=1ns，ms=cycles/1e6，MiB=2^20字节。HBM是声明的模拟线传输量，不是实芯片测量；"
              "重叠计数不能相加当墙钟时间。不是RTL或全模型tokens/s，排除attention、router执行与完整服务。"
              "MILP为专家分配见证后的可执行LPT回放，分配最优不等于任意时序最优。", ""]
    statuses = Counter(str(r.get("solver_status", r.get("milp_solver_status", "未提供")))
                       for r in rows if r["dispatch"] == "milp")
    lines += ["离线求解状态：" + _json(dict(sorted(statuses.items()))) + "。", "",
              "只重建报告（支持无损归档per_window.json.gz）：", "```sh",
              "python -m research.moe_dispatch.round2.dispatch_fix.report --directory PATH", "```", ""]
    return "\n".join(lines)


def generate(directory: Path):
    directory = Path(directory).resolve()
    rows = _rows(_load(directory / "per_window.json"))
    selection = _load(directory / "selection.json")
    frozen = _load(directory / "frozen_designs.json")
    comparisons = comparison_rows(rows)
    control_comparisons = comparison_rows(rows, (CONTROL,))
    traffic = hbm_rows(rows)
    control_traffic = hbm_rows(rows, (CONTROL,))
    exceptions = excess_rows(rows, traffic)
    control_exceptions = excess_rows(rows, control_traffic, (CONTROL,))
    regressions = regression_rows(rows)
    task_deltas = task_hbm_deltas(rows)
    cis = confidence_rows(rows)
    outputs = {
        "compare.csv": (comparisons, COMPARE_FIELDS),
        "compare_predictor_control.csv": (control_comparisons, COMPARE_FIELDS),
        "hbm_bytes.csv": (traffic, HBM_FIELDS),
        "hbm_predictor_control.csv": (control_traffic, HBM_FIELDS),
        "excess_hbm_windows.csv": (exceptions, EXCESS_FIELDS),
        "excess_hbm_predictor_control.csv": (control_exceptions, EXCESS_FIELDS),
        "regression.csv": (regressions, REGRESSION_FIELDS),
        "paired_improvement.csv": (cis, CI_FIELDS),
        "task_hbm_deltas.csv": (task_deltas, TASK_DELTA_FIELDS),
    }
    for filename, (values, fields) in outputs.items():
        _dump_csv(directory / filename, values, fields)
    (directory / "gpqa_t128_case.md").write_text(case_markdown(rows, task_deltas))
    (directory / "SUMMARY.md").write_text(summary_markdown(rows, selection, frozen, comparisons,
                                                            traffic, exceptions, regressions, cis, task_deltas))
    (directory / "acceptance.json").write_text(json.dumps(acceptance_details(rows, comparisons, traffic,
                                                                           regressions, task_deltas),
                                                        indent=2, sort_keys=True) + "\n")
    evidence = {
        "inputs": {name: hashlib.sha256(_input_bytes(directory / name)).hexdigest()
                   for name in ("per_window.json", "selection.json", "frozen_designs.json")},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_counts": {name: len(values) for name, (values, _) in outputs.items()},
        "units": {"cycle_ns": 1, "ms_per_cycle": 1e-6, "MiB_bytes": 2**20},
        "scope": "analytical post-router BF16 layer; not RTL/full-model inference",
    }
    (directory / "report_receipt.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path,
                        help="new dispatch_fix evidence directory containing the three input JSON files")
    args = parser.parse_args(argv)
    evidence = generate(args.directory)
    print(json.dumps(evidence["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
