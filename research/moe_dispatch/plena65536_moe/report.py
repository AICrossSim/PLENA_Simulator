"""Frozen-hardware full Shared + true-route MoE tables and repeat audit.

Compute Time is the exact whole-expert schedule makespan.  Necessary external
bandwidth service and local supply rates are independent ledgers.  WS/IS are
declared operand-loop extensions on native OS PEs under ideal context ports.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from research.moe_dispatch.plena65536_shared.geometry import BUDGET, Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES
from .dataflow import FLOWS, add_rates, plena_mixed_traffic, resource_ledger
from .workloads import BATCHES, load_workloads, profile_window

MIB, KIB = 1024**2, 1024
PROFILE = PROFILES[0]


def gm(values):
    values = tuple(float(value) for value in values)
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("GM requires finite nonnegative nonempty values")
    return 0.0 if any(value == 0 for value in values) else math.exp(math.fsum(math.log(value) for value in values) / len(values))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def physical_string(cores):
    return " + ".join(f"{c.pm}×{c.pn}×{c.groups}" for c in cores)


def frame_string(cores):
    return " + ".join(f"b={c.b},R={c.r},rM={c.rm},rN={c.rn}" for c in cores)


def frozen_designs(selection):
    designs = []
    for name in ("original", "extended_single", "extended_homogeneous", "extended_heterogeneous",
                 "native_single", "native_homogeneous", "native_heterogeneous"):
        chosen = selection["families"][name]
        cores = tuple(Core(**{key: core[key] for key in ("b", "r", "rm", "rn")})
                          for core in chosen["cores"])
        designs.append((name, "native" if name == "original" or name.startswith("native_") else "extended",
                        cores, "paper_original" if name == "original" else "development_7batch_GM"))
    return designs


def point(window, cores, flows, scheduler, profile=PROFILE):
    plan = scheduler(cores, window, profile)
    traffic = plena_mixed_traffic(window, cores, flows, plan)
    rates = add_rates(traffic, plan["cycles"])
    observed = profile_window(window)
    if traffic["resource_ledger"]["multipliers"] != BUDGET:
        raise AssertionError("Frozen design violates 65,536 physical multipliers")
    if traffic["useful_macs"] != observed["total_useful_macs"]:
        raise AssertionError("Full Shared + routed useful MACs were lost")
    if traffic["external"]["cold_weight_only_unique_bytes"] != observed["total_unique_weight_bytes"]:
        raise AssertionError("Active expert identity weight ledger differs from native descriptors")
    if traffic["routed_useful_macs"] != 3 * traffic["shared_useful_macs"]:
        raise AssertionError("True Top-6 actual-F routed workload was not fully included")
    if rates["spatial_utilization_pct"] > 100.000000001:
        raise AssertionError("Useful arithmetic exceeds issued arithmetic")
    return {"plan": plan, "traffic": traffic, "rates": rates}


def result_row(design, scope, source, cores, split, window, flows, got):
    t, r, p = got["traffic"], got["rates"], got["plan"]
    e, cap = t["external"], t["capacity"]
    experts = {int(expert["id"]): expert for expert in window["experts"]}
    shared_work = sum(job["end"] - job["start"] for job in p["jobs"] if job["expert_id"] == -1)
    routed_work = sum(job["end"] - job["start"] for job in p["jobs"] if job["expert_id"] != -1)
    busy = [sum(job["end"] - job["start"] for job in p["jobs"] if job["core_id"] == core)
            for core in range(len(cores))]
    return {
        "design": design, "scope": scope, "selection_source": source,
        "split": split, "window_id": window["id"], "batch": int(window["batch"]),
        "row_type": "window", "window_count": 1, "profile": PROFILE.name,
        "physical_PM_PN_G": physical_string(cores), "native_frames": frame_string(cores),
        "multiplier_allocation": "+".join(str(c.macs) for c in cores),
        "dataflow": "/".join(flows), "schedule_contract": p.get("contract", "whole_expert_exact"),
        "active_routed_experts": t["active_routed_experts"],
        "shared_token_assignments": t["shared_tokens"], "routed_token_assignments": t["routed_token_assignments"],
        "shared_useful_macs": t["shared_useful_macs"], "routed_useful_macs": t["routed_useful_macs"],
        "total_useful_macs": t["useful_macs"], "issued_macs": t["issued_macs"],
        "compute_cycles": p["cycles"], "compute_ms": r["compute_ms"],
        "shared_core_work_ms": shared_work / 1e6, "routed_core_work_ms": routed_work / 1e6,
        "core_busy_sum_ms": sum(busy) / 1e6,
        "core0_busy_ms": busy[0] / 1e6, "core1_busy_ms": busy[1] / 1e6 if len(cores) > 1 else 0,
        "core0_shared_jobs": sum(job["expert_id"] == -1 and job["core_id"] == 0 for job in p["jobs"]),
        "core0_cold_Me_le4_jobs": sum(job["expert_id"] != -1 and job["core_id"] == 0
                                      and int(experts[job["expert_id"]]["Me"]) <= 4 for job in p["jobs"]),
        "core1_cold_Me_le4_jobs": sum(job["expert_id"] != -1 and job["core_id"] == 1
                                      and int(experts[job["expert_id"]]["Me"]) <= 4 for job in p["jobs"]),
        "useful_GFLOP_s": r["useful_GFLOP_s"],
        "spatial_utilization_pct": r["spatial_utilization_pct"],
        "wall_peak_utilization_pct": 100 * t["useful_macs"] / (p["cycles"] * BUDGET),
        "shared_unique_weight_MiB": t["shared_unique_weight_bytes"] / MIB,
        "routed_unique_weight_MiB": t["routed_unique_weight_bytes"] / MIB,
        "unique_weight_MiB": e["cold_weight_only_unique_bytes"] / MIB,
        "unique_weight_required_GB_s": r["unique_weight_required_GB_s"],
        "unique_weight_fetch_ms_at256": r["unique_weight_fetch_ms"],
        "cold_all_required_GB_s": r["cold_all_required_GB_s"],
        "cold_all_fetch_ms_at256": r["cold_all_fetch_ms"],
        "pool_weight_refill_MiB": t["pool_to_local"]["weight_masked_bytes"] / MIB,
        "pool_X_refill_MiB": t["pool_to_local"]["x_masked_bytes"] / MIB,
        "pool_context_transfer_MiB": (t["pool_to_local"]["partial_read_masked_bytes"]
                                      + t["pool_to_local"]["partial_write_masked_bytes"]) / MIB,
        "pool_weight_refill_required_GB_s": r["pool_to_local"]["weight_masked_GB_s"],
        "pool_X_refill_required_GB_s": r["pool_to_local"]["x_masked_GB_s"],
        "pool_context_required_GB_s": r["pool_to_local"]["partial_read_masked_GB_s"]
                                      + r["pool_to_local"]["partial_write_masked_GB_s"],
        "array_weight_delivery_required_GB_s": r["array_delivery"]["weight_full_GB_s"],
        "array_X_delivery_required_GB_s": r["array_delivery"]["x_full_GB_s"],
        "postprocess_pool_transfer_required_GB_s": sum(t["postprocess"][key] for key in (
            "GU_read_fp32_bytes", "Z_write_bf16_bytes", "expert_Y_combine_read_fp32_bytes",
            "combined_Y_write_fp32_bytes")) / p["cycles"],
        "unified_pool_external_weight_refetch_factor": e["unified_pool_external_weight_refetch_factor"],
        "no_pool_weight_refetch_factor": e["no_pool_stream_weight_refetch_factor"],
        "no_pool_stream_total_MiB": e["no_pool_stream_total_bytes"] / MIB,
        "no_pool_stream_required_GB_s": r["no_pool_stream_required_GB_s"],
        "no_pool_stream_fetch_ms_at256": r["no_pool_stream_fetch_ms"],
        "local_scratch_tree_upper_KiB": cap["local_scratch_tree_upper_bytes"] / KIB,
        "scheduled_scratch_upper_KiB": cap["scheduled_scratch_upper_bytes"] / KIB,
        "sum_core_max_scratch_upper_KiB": cap["sum_core_max_scratch_upper_bytes"] / KIB,
        "resident_full_payload_upper_MiB": cap["unified_resident_payload_upper_bytes"] / MIB,
        "private_resident_payload_sum_upper_MiB": cap["private_resident_payload_sum_upper_bytes"] / MIB,
        "repeat_identical": True,
    }


CAPACITY_KEYS = {"local_scratch_tree_upper_KiB", "scheduled_scratch_upper_KiB",
                 "sum_core_max_scratch_upper_KiB", "resident_full_payload_upper_MiB",
                 "private_resident_payload_sum_upper_MiB"}
IDENTITY_KEYS = {"batch", "window_id", "window_count"}
COUNT_KEYS = {"active_routed_experts", "shared_token_assignments", "routed_token_assignments",
              "core0_shared_jobs", "core0_cold_Me_le4_jobs", "core1_cold_Me_le4_jobs"}


def aggregate_rows(rows, *, batch, row_type):
    first = rows[0]
    result = dict(first)
    result.update(batch=batch, window_id="", row_type=row_type,
                  window_count=sum(row["window_count"] for row in rows))
    for key, value in first.items():
        if key in IDENTITY_KEYS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if key in CAPACITY_KEYS:
                result[key] = max(row[key] for row in rows)
            elif key in COUNT_KEYS:
                result[key] = math.fsum(row[key] for row in rows) / len(rows)
            else:
                result[key] = gm(row[key] for row in rows)
    return result


def compact_detail(design, scope, split, window, flows, got):
    t = got["traffic"]
    return {
        "design": design, "scope": scope, "split": split, "window_id": window["id"],
        "batch": int(window["batch"]), "dataflows": list(flows),
        "result_sha256": hashlib.sha256(canonical(got).encode()).hexdigest(),
        "plan": got["plan"], "rates": got["rates"],
        "external": t["external"], "pool_to_local": t["pool_to_local"],
        "array_delivery": t["array_delivery"], "postprocess": t["postprocess"],
        "capacity": t["capacity"], "per_core": t["per_core"],
        "expert_ledger": [{"expert_id": leaf["expert_id"], "is_shared": leaf["is_shared"],
                           "Me": leaf["token_count"], "H": leaf["hidden"], "F": leaf["intermediate"],
                           "useful_macs": leaf["useful_macs"], "issued_macs": leaf["issued_macs"],
                           "issues": leaf["issues"], "unique_weight_bytes": leaf["external"]["cold_weight_only_unique_bytes"],
                           "pool_to_local": leaf["pool_to_local"], "array_delivery": leaf["array_delivery"]}
                          for leaf in t["experts"]],
        "contract": t["contract"],
    }


def make_workbook(out, tables):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    path = out / "PLENA65536_Shared_MoE_完整重跑.xlsx"
    workbook = Workbook()
    workbook.remove(workbook.active)
    about = workbook.create_sheet("阅读说明")
    notes = [
        ("口径", "说明"),
        ("工作负载", "18 development / 135 heldout真实Top6窗口；Shared F2816、路由F1408；全部3 GEMM"),
        ("硬件冻结", "在development选择几何；heldout全部使用同一冻结几何；每个组织65,536物理乘法器"),
        ("Compute Time", "按完整专家任务最优双核分配的makespan；Shared/路由core-work不是顺序延迟"),
        ("计算模型", "论文拓扑一致解析时序，1GHz、理想供数/上下文端口；非原生Rust/RTL校准计时"),
        ("驻留循环", "PE始终OS，WS/IS为操作数循环扩展；G=R/b部分和上下文的流量/容量均收费"),
        ("唯一权重", "33MiB Shared + 16.5MiB×active路由专家；inactive不收费；expert_id+matrix唯一"),
        ("供数边界", "外部唯一W、共享池至本地refill、array delivery分别报告，不相加为HBM"),
        ("Fetch Time", "字节÷256GB/s服务等价；不与Compute相加；大小比较不构成完整E2E分类"),
        ("容量", "固定PE/shadow/双缓冲/树一次收费；scratch按实际任务重叠取上界；全专家中间结果保守同时驻留"),
        ("Private", "每核独立驻留容量另列；参与Shared划分的核各收费其权重所有权；无池是不同streaming系统"),
        ("聚合", "时间/吞吐/带宽按Batch内GM再七Batch等权GM；容量max；数量与owner占比算术mean，七Batch行等权mean"),
        ("重复", "每个窗口×冻结硬件×驻留组合完整重算两次并规范化比较"),
        ("非GEMM", "路由为输入；gather/scatter、激活/加权combine算术未计时；其显式数据读写另收费"),
    ]
    for row in notes:
        about.append(row)
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name[:31])
        headers = list(rows[0])
        sheet.append(headers)
        for row in rows:
            sheet.append([row[key] for key in headers])
    for sheet in workbook:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="183B56")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(wrap_text=True)
        sheet.row_dimensions[1].height = 42
        for column in sheet.columns:
            width = max(len(str(cell.value or "")) for cell in column[:80])
            sheet.column_dimensions[column[0].column_letter].width = min(65, max(13, width + 2))
    workbook.save(path)
    saved = load_workbook(path, read_only=True, data_only=True)
    checked = 0
    for name, rows in tables.items():
        cells = list(saved[name[:31]].values)
        headers = list(rows[0])
        if list(cells[0]) != headers or len(cells) != len(rows) + 1:
            raise AssertionError("Workbook header/row readback mismatch")
        for row, actual in zip(rows, cells[1:]):
            for key, value in zip(headers, actual):
                expected = row[key]
                if isinstance(expected, float):
                    if not math.isclose(expected, value, rel_tol=2e-14, abs_tol=1e-12):
                        raise AssertionError((name, key, expected, value))
                elif expected != value and not (expected == "" and value is None):
                    raise AssertionError((name, key, expected, value))
                checked += 1
    saved.close()
    return path, checked


def table(rows, columns):
    def cell(value):
        return f"{value:,.6f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(cell(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join(lines)


def bootstrap_rows(rows, *, repetitions=2000, seed=655362026):
    """Paired resampling within each Batch, preserving equal Batch weights.

    These intervals describe the finite captured-window population only, not
    independence across tokens or an inference quality/performance benchmark.
    """
    import numpy as np
    result = []
    for split in ("development", "heldout"):
        indexes = defaultdict(dict)
        for row in rows:
            if row["split"] == split and row["row_type"] == "window" and all(
                    flow == "OS" for flow in row["dataflow"].split("/")):
                indexes[row["design"]][row["window_id"]] = row
        pairs = (("extended_single", "extended_heterogeneous"),
                 ("extended_homogeneous", "extended_heterogeneous"),
                 ("extended_single", "extended_homogeneous"))
        for baseline, candidate in pairs:
            rng = np.random.default_rng(seed)
            batch_logs, all_ratios = [], []
            for batch in BATCHES:
                ids = sorted(key for key, row in indexes[baseline].items() if row["batch"] == batch)
                if not ids or any(key not in indexes[candidate] for key in ids):
                    raise AssertionError("Paired bootstrap lost matched routed windows")
                logs = np.array([math.log(indexes[baseline][key]["compute_ms"] /
                                          indexes[candidate][key]["compute_ms"]) for key in ids])
                all_ratios.extend(np.exp(logs).tolist())
                draws = rng.integers(0, len(logs), size=(repetitions, len(logs)))
                batch_logs.append(logs[draws].mean(axis=1))
            samples = np.exp(np.mean(batch_logs, axis=0))
            point_estimate = gm(gm(indexes[baseline][key]["compute_ms"] /
                                   indexes[candidate][key]["compute_ms"]
                                   for key, row in indexes[baseline].items() if row["batch"] == batch)
                                for batch in BATCHES)
            lower, upper = np.quantile(samples, (0.025, 0.975))
            result.append({"split": split, "baseline": baseline, "candidate": candidate,
                           "batch_balanced_GM_speedup": point_estimate,
                           "bootstrap_95pct_lower": float(lower), "bootstrap_95pct_upper": float(upper),
                           "matched_windows": len(all_ratios), "candidate_win_windows": sum(value > 1 for value in all_ratios),
                           "candidate_tie_windows": sum(value == 1 for value in all_ratios),
                           "candidate_loss_windows": sum(value < 1 for value in all_ratios),
                           "repetitions": repetitions, "seed": seed,
                           "resampling_contract": "paired windows within Batch; seven Batch log means equally weighted",
                           "interval_scope": "captured windows only; layers/steps may be correlated"})
    return result


def main(out, scheduler=None):
    if scheduler is None:
        from .model import schedule
        scheduler = schedule
    selection_path = out / "FROZEN_SELECTION.json"
    if not selection_path.exists():
        raise SystemExit("Complete full Shared + MoE development search before reporting")
    selection = json.loads(selection_path.read_text())
    designs = frozen_designs(selection)
    inputs = load_workloads()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "CALCULATION_SOURCE.json", {
        "report.py": sha256(Path(__file__)), "dataflow.py": sha256(Path(__file__).with_name("dataflow.py")),
        "completed_points_expected": sum(3 ** len(cores) for _, _, cores, _ in designs)
                                    * (len(inputs["development"]) + len(inputs["heldout"])),
        "repetitions_each": 2,
    })
    all_rows, aggregates, resources, sensitivity, sample_summary = [], [], [], [], []
    repeat_points = 0
    details_path = out / "cross_dataflow_details.jsonl.gz"
    with gzip.open(details_path, "wt", encoding="utf-8") as details:
        for design, scope, cores, source in designs:
            resources.append({"design": design, "scope": scope, "physical_PM_PN_G": physical_string(cores),
                              "native_frames": frame_string(cores),
                              "multiplier_allocation": "+".join(str(c.macs) for c in cores), **resource_ledger(cores)})
            for split in ("development", "heldout"):
                windows = inputs[split]
                for profile in PROFILES:
                    grouped = defaultdict(list)
                    values = []
                    for window in windows:
                        cycles = scheduler(cores, window, profile)["cycles"]
                        grouped[int(window["batch"])].append(cycles)
                        values.append(cycles)
                    sensitivity.append({"design": design, "scope": scope, "split": split,
                                        "profile": profile.name, "GM_compute_ms": gm(gm(v) for v in grouped.values()) / 1e6,
                                        "sample_GM_compute_ms": gm(values) / 1e6,
                                        "physical_PM_PN_G": physical_string(cores), "native_frames": frame_string(cores)})
                for flows in itertools.product(FLOWS, repeat=len(cores)):
                    window_rows = []
                    for window in windows:
                        got = point(window, cores, flows, scheduler)
                        repeat = point(window, cores, flows, scheduler)
                        if canonical(got) != canonical(repeat):
                            raise AssertionError("Independent repeated full-workload point differed")
                        repeat_points += 1
                        row = result_row(design, scope, source, cores, split, window, flows, got)
                        window_rows.append(row)
                        details.write(canonical(compact_detail(design, scope, split, window, flows, got)) + "\n")
                    batch_rows = [aggregate_rows([row for row in window_rows if row["batch"] == batch],
                                                 batch=batch, row_type="batch_GM") for batch in BATCHES]
                    combined = aggregate_rows(batch_rows, batch="7-Batch GM", row_type="7batch_GM")
                    all_rows.extend(window_rows + batch_rows + [combined])
                    aggregates.extend(batch_rows + [combined])
                    sample_summary.append({"design": design, "scope": scope, "split": split,
                                           "dataflow": "/".join(flows), "window_count": len(window_rows),
                                           "batch_balanced_GM_compute_ms": combined["compute_ms"],
                                           "sample_weighted_GM_compute_ms": gm(row["compute_ms"] for row in window_rows),
                                           "active_routed_experts_min": min(row["active_routed_experts"] for row in window_rows),
                                           "active_routed_experts_max": max(row["active_routed_experts"] for row in window_rows),
                                           "unique_weight_MiB_min": min(row["unique_weight_MiB"] for row in window_rows),
                                           "unique_weight_MiB_max": max(row["unique_weight_MiB"] for row in window_rows)})
                print(canonical({"design": design, "split": split, "repeat_points_completed": repeat_points}), flush=True)
    baseline = {(row["split"], str(row["batch"])): row["compute_ms"] for row in aggregates
                if row["design"] == "original" and row["dataflow"] == "OS"}
    for row in all_rows:
        if row["row_type"] != "window":
            row["speedup_vs_original_compute"] = baseline[(row["split"], str(row["batch"]))] / row["compute_ms"]
        else:
            row["speedup_vs_original_compute"] = 0.0
    original_window = {(row["split"], row["window_id"]): row["compute_ms"] for row in all_rows
                       if row["design"] == "original" and row["row_type"] == "window" and row["dataflow"] == "OS"}
    for row in all_rows:
        if row["row_type"] == "window":
            row["speedup_vs_original_compute"] = original_window[(row["split"], row["window_id"])] / row["compute_ms"]
    os_summary = [row for row in aggregates if all(flow == "OS" for flow in row["dataflow"].split("/"))]
    tradeoff = [row for row in aggregates if row["row_type"] == "7batch_GM"]
    bootstrap = bootstrap_rows(all_rows)
    crossovers = [{key: row[key] for key in ("design", "scope", "split", "batch", "dataflow", "row_type", "window_count",
                                                "compute_ms", "unique_weight_fetch_ms_at256", "no_pool_stream_fetch_ms_at256")}
                 | {"compute_service_ge_unique_weight": row["compute_ms"] >= row["unique_weight_fetch_ms_at256"],
                    "compute_service_ge_no_pool_stream": row["compute_ms"] >= row["no_pool_stream_fetch_ms_at256"],
                    "comparison_is_trace_samples_only": True, "not_full_E2E_bound_classification": True}
                 for row in aggregates]
    supply_keys = ("design", "scope", "split", "batch", "row_type", "window_id", "dataflow", "compute_ms",
                   "unique_weight_required_GB_s", "unique_weight_fetch_ms_at256", "pool_weight_refill_required_GB_s",
                   "pool_X_refill_required_GB_s", "pool_context_required_GB_s", "array_weight_delivery_required_GB_s",
                   "array_X_delivery_required_GB_s", "postprocess_pool_transfer_required_GB_s",
                   "no_pool_stream_required_GB_s", "no_pool_stream_fetch_ms_at256")
    supply = [{key: row[key] for key in supply_keys} for row in all_rows]
    for name, rows in (("cross_dataflow", all_rows), ("compute_summary", os_summary),
                       ("dataflow_tradeoff", tradeoff), ("resource_ledger", resources),
                       ("frozen_timing_sensitivity", sensitivity), ("sample_weighted_summary", sample_summary),
                       ("crossover", crossovers), ("supply_bw", supply), ("paired_bootstrap", bootstrap)):
        write_csv(out / (name + ".csv"), rows)
    return finish(out, selection_path, inputs, designs, all_rows, aggregates, resources,
                  sensitivity, sample_summary, bootstrap, crossovers, supply, repeat_points)


def finish(out, selection_path, inputs, designs, all_rows, aggregates, resources,
           sensitivity, sample_summary, bootstrap, crossovers, supply, repeat_points,
           *, finalized_from_completed_tables=False):
    os_summary = [row for row in aggregates if all(flow == "OS" for flow in row["dataflow"].split("/"))]
    tradeoff = [row for row in aggregates if row["row_type"] == "7batch_GM"]
    details_path = out / "cross_dataflow_details.jsonl.gz"
    workbook, cells = make_workbook(out, {
        "Dev全数据流": [row for row in all_rows if row["split"] == "development"],
        "Heldout全数据流": [row for row in all_rows if row["split"] == "heldout"],
        "OS计算汇总": os_summary, "七Batch数据流权衡": tradeoff,
        "反推带宽": supply, "硬件资源": resources, "时序敏感性": sensitivity,
        "样本加权对照": sample_summary, "服务交叉比较": crossovers, "配对Bootstrap": bootstrap,
    })
    heldout = [row for row in os_summary if row["split"] == "heldout"]
    heldout_gm = [row for row in heldout if row["row_type"] == "7batch_GM"]
    columns = [("design", "组织"), ("physical_PM_PN_G", "物理PM×PN×G"),
               ("multiplier_allocation", "乘法器分配"), ("compute_ms", "完整Compute ms"),
               ("speedup_vs_original_compute", "相对原始倍数"), ("spatial_utilization_pct", "MAC利用率%"),
               ("unique_weight_required_GB_s", "唯一W所需GB/s"), ("unique_weight_fetch_ms_at256", "W Fetch ms@256"),
               ("resident_full_payload_upper_MiB", "全驻留容量上界MiB")]
    lines = ["# PLENA 65,536：Shared + 真实Top-6 MoE 完整重跑", "",
             "18 development窗口选择硬件，135 heldout窗口检验。每个窗口完整执行Shared及所有活跃路由专家的Gate、Up、Down。Shared H=2048/F=2816，路由H=2048/F=1408；Shared覆盖B，路由覆盖6B，路由有用MAC严格为Shared的3倍。", "",
             "Compute Time是完整专家任务在冻结核上的精确最小makespan；Shared/路由core-work列只是工作量分解，不可相加当作多核延迟。全几何域最优结论限于整专家任务所有权，不涵盖任意token/输出切分。物理预算始终65,536乘法器，第三维G=R/b；面积与Fmax未校准。", "",
             "这是论文拓扑一致的BF16/FP32解析模型，1GHz、理想操作数与上下文端口，未声称原生Rust/RTL校准计时。PE保持OS；WS/IS是操作数循环驻留扩展，完整G部分和上下文存取与容量已收费。同硬件各流的计算时间相同，其供数及容量不同。", "",
             "唯一冷权重=33MiB Shared +16.5MiB×活跃路由专家；inactive不收费。共享池的唯一外部W、pool→local refill、array delivery分别报告；256GB/s Fetch是服务等价，不与Compute相加。路由、gather/scatter、激活、加权combine算术未计时，GU/Z/专家Y/combine的数据读写已显式收费。", "",
             "时间/吞吐/带宽GM先在各Batch内取窗口GM，再对七Batch等权GM；数量与owner占比取算术mean，七Batch行等权mean；容量聚合均取max。scratch按实际任务区间重叠取保守相位上界；全活跃权重和全专家中间结果同时驻留是保守容量包络，另列private各核容量。", "",
             "冻结异构的两个核PM均为2，乘法器分配20,480 +45,056，属于不同N/K几何及资源配比。135 heldout窗口的Shared分配给小核21次、大核114次；大核同时执行1,206个M_e≤2冷专家，小核执行891个。此次最优队列允许每核处理不同大小的任务，未形成严格Shared/热专家大核、冷专家小核分工。详见DISPATCH_QUEUE_SUMMARY.json和逐窗口真实队列。", "",
             "## Heldout：七Batch等权GM（冻结硬件，OS基线）", "", table(heldout_gm, columns), ""]
    for scope in ("native", "extended", "timing_robust"):
        scoped = [row for row in heldout if row["scope"] == scope]
        if scoped:
            lines += [f"## {scope}：七Batch与GM", "",
                      table(scoped, [("batch", "Batch")] + columns), ""]
    lines += ["## 配对真实路由窗口的速度差异", "",
              table([row for row in bootstrap if row["split"] == "heldout"], [
                  ("baseline", "基线"), ("candidate", "候选"), ("batch_balanced_GM_speedup", "GM倍数"),
                  ("bootstrap_95pct_lower", "95%下限"), ("bootstrap_95pct_upper", "95%上限"),
                  ("candidate_win_windows", "胜窗口"), ("candidate_tie_windows", "平窗口"),
                  ("candidate_loss_windows", "负窗口")]), "",
              "Bootstrap固定seed=655362026，2000次，在每个Batch内配对重采样窗口，再对七Batch等权汇总；区间仅描述捕获窗口，层/step可能相关。", "",
              "## 同冻结硬件的数据流权衡（Heldout七Batch GM）", "",
              table([row for row in tradeoff if row["split"] == "heldout"], [
                  ("design", "组织"), ("dataflow", "核0/核1驻留"), ("compute_ms", "Compute ms"),
                  ("pool_weight_refill_required_GB_s", "池W GB/s"), ("pool_X_refill_required_GB_s", "池X GB/s"),
                  ("pool_context_required_GB_s", "池上下文GB/s"),
                  ("local_scratch_tree_upper_KiB", "本地/上下文/树容量上界KiB"),
                  ("no_pool_weight_refetch_factor", "无池W重读倍数")]), "",
              "全窗口、全组织、单核3种/双核9种驻留组合的原始表在cross_dataflow.csv及Excel；每个点独立计算两次完全一致。cross_dataflow_details.jsonl.gz保留真实M_e、逐专家账本及精确任务区间。", "",
              "crossover.csv仅比较现有真实路由窗口/Batch聚合的必要服务时间，没有对未捕获的Batch编造路由分布，也不声称完整E2E compute-bound。", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [selection_path, Path(__file__), Path(__file__).with_name("dataflow.py"),
               Path(__file__).with_name("workloads.py")]
    sources.append(Path(__file__).with_name("model.py"))
    write_json(out / "REPORT_AUDIT.json", {
        "status": "PASS", "full_scope": "Shared + all active true Top6 routed experts, 3 GEMMs each",
        "development_windows": len(inputs["development"]), "heldout_windows": len(inputs["heldout"]),
        "frozen_designs": len(designs), "cross_dataflow_rows": len(all_rows),
        "distinct_repeated_points": repeat_points, "repetitions_each": 2,
        "point_comparisons_identical": True, "workbook": str(workbook),
        "workbook_readback_cells_checked": cells, "source_sha256": {path.name: sha256(path) for path in sources},
        "detail_sha256": sha256(details_path), "all_scope_arithmetic_conserved": True,
        "routing_useful_macs_equals_3x_shared": True, "native_timing_calibrated": False,
        "WS_IS_original_hardware_support_claimed": False,
        "all_cross_flow_compute_times_equal_under_ideal_ports": True,
        "capacity_is_conservative_upper_bound": True,
        "geometry_selection_heldout_access": False,
        "paired_bootstrap_repetitions": 2000, "paired_bootstrap_seed": 655362026,
        "finalized_from_completed_tables": finalized_from_completed_tables,
        "calculation_source": json.loads((out / "CALCULATION_SOURCE.json").read_text())
                              if (out / "CALCULATION_SOURCE.json").exists() else None,
    })
    print(canonical({"status": "PASS", "rows": len(all_rows), "repeat_points": repeat_points,
                     "workbook_cells": cells, "workbook": str(workbook)}), flush=True)
    for row in heldout_gm:
        print(canonical({key: row[key] for key in ("design", "physical_PM_PN_G", "compute_ms",
                                                  "speedup_vs_original_compute", "unique_weight_fetch_ms_at256")}), flush=True)


def finalize_only(out):
    """Recover artifact construction after a completed calculation export."""
    def typed(value):
        if value in ("True", "False"):
            return value == "True"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def read(name):
        with (out / (name + ".csv")).open(newline="", encoding="utf-8") as handle:
            return [{key: typed(value) for key, value in row.items()} for row in csv.DictReader(handle)]

    selection_path = out / "FROZEN_SELECTION.json"
    designs = frozen_designs(json.loads(selection_path.read_text()))
    inputs = load_workloads()
    rows = read("cross_dataflow")
    aggregates = [row for row in rows if row["row_type"] != "window"]
    repeated = [row for row in rows if row["row_type"] == "window"]
    expected = sum(3 ** len(cores) for _, _, cores, _ in designs) * (
        len(inputs["development"]) + len(inputs["heldout"]))
    if len(repeated) != expected or any(row["repeat_identical"] is not True for row in repeated):
        raise AssertionError("Complete independently repeated calculation rows are required")
    lookup = {(row["design"], row["split"], row["window_id"], row["dataflow"]): row for row in repeated}
    seen = set()
    with gzip.open(out / "cross_dataflow_details.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            detail = json.loads(line)
            key = (detail["design"], detail["split"], detail["window_id"], "/".join(detail["dataflows"]))
            row = lookup[key]
            if key in seen or row["compute_cycles"] != detail["plan"]["cycles"]:
                raise AssertionError("Detailed execution export disagrees with completed rows")
            if len(detail["result_sha256"]) != 64:
                raise AssertionError("Missing completed normalized point digest")
            seen.add(key)
    if len(seen) != expected:
        raise AssertionError("Detailed execution export is incomplete")
    # Counts are means, timing/throughput/BW are GMs, capacities are maxima.
    # Rebuild summaries from completed per-window calculations so presentation
    # corrections do not rerun or alter any full-workload computation.
    grouped = defaultdict(list)
    for row in repeated:
        grouped[(row["design"], row["split"], row["dataflow"])].append(row)
    rebuilt, aggregates = [], []
    for window_rows in grouped.values():
        batch_rows = [aggregate_rows([row for row in window_rows if row["batch"] == batch],
                                     batch=batch, row_type="batch_GM") for batch in BATCHES]
        combined = aggregate_rows(batch_rows, batch="7-Batch GM", row_type="7batch_GM")
        rebuilt.extend(window_rows + batch_rows + [combined])
        aggregates.extend(batch_rows + [combined])
    baseline = {(row["split"], str(row["batch"])): row["compute_ms"] for row in aggregates
                if row["design"] == "original" and row["dataflow"] == "OS"}
    for row in aggregates:
        row["speedup_vs_original_compute"] = baseline[(row["split"], str(row["batch"]))] / row["compute_ms"]
    previous_compute = {(row["design"], row["split"], str(row["batch"]), row["dataflow"], row["row_type"]): row["compute_ms"]
                        for row in rows if row["row_type"] != "window"}
    for row in aggregates:
        key = (row["design"], row["split"], str(row["batch"]), row["dataflow"], row["row_type"])
        if row["compute_ms"] != previous_compute[key]:
            raise AssertionError("Artifact-only aggregation changed Compute Time")
    rows = rebuilt
    os_summary = [row for row in aggregates if all(flow == "OS" for flow in row["dataflow"].split("/"))]
    tradeoff = [row for row in aggregates if row["row_type"] == "7batch_GM"]
    supply_headers = list(read("supply_bw")[0])
    supply = [{key: row[key] for key in supply_headers} for row in rows]
    for name, values in (("cross_dataflow", rows), ("compute_summary", os_summary),
                         ("dataflow_tradeoff", tradeoff), ("supply_bw", supply)):
        write_csv(out / (name + ".csv"), values)
    finish(out, selection_path, inputs, designs, rows, aggregates, read("resource_ledger"),
           read("frozen_timing_sensitivity"), read("sample_weighted_summary"),
           read("paired_bootstrap"), read("crossover"), supply, len(repeated),
           finalized_from_completed_tables=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--finalize-only", action="store_true",
                        help="Build workbook/report/audit from already completed repeated calculation tables")
    args = parser.parse_args()
    (finalize_only if args.finalize_only else main)(args.out)
