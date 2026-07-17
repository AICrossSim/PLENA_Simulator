#!/usr/bin/env python3
"""Build the auditable evidence report for the ``rtl-v1`` timing model.

The report deliberately separates three kinds of evidence:

* full-Machine RTL measurements, which can be checked cycle by cycle;
* structural extrapolations, which are supported by an RTL scaling law but
  were not measured at the exact production shape;
* unsupported RTL paths, which remain executable in the functional emulator
  but are excluded from cycle-exact validation claims.

No numerical comparison thresholds or golden tensors are changed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVIDENCE_ROOT = ROOT / "Workspace/rtl_v1_latency_validation"
DEFAULT_CALIBRATION = ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v1.json"


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def ceil_log2(value: int) -> int:
    return 0 if value <= 1 else math.ceil(math.log2(value))


def format_optional_float(value: Any, digits: int = 3) -> str:
    return "not available" if value is None else f"{float(value):.{digits}f}"


def metric_row(
    *,
    unit: str,
    point: str,
    opcode: str,
    metric: str,
    measured: int | float,
    predicted: int | float,
    provenance: str,
    holdout: bool = False,
) -> dict[str, Any]:
    error = float(predicted) - float(measured)
    return {
        "unit": unit,
        "point": point,
        "opcode": opcode,
        "metric": metric,
        "measured_cycles": measured,
        "predicted_cycles": predicted,
        "error_cycles": error,
        "abs_error_cycles": abs(error),
        "provenance": provenance,
        "holdout": holdout,
        "within_one_cycle": abs(error) <= 1.0,
    }


def matrix_rows(raw: dict[str, Any], calibration: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    config = calibration["matrix"]["mxfp"]
    for record in raw.get("measurements", []):
        point = str(record["point_tag"])
        opcode = str(record["opcode"])
        blen = int(record["blen"])
        if opcode == "M_MM":
            predicted = (
                int(config["gemm_load_blen_coefficient"]) * blen
                + int(config["gemm_load_fixed_cycles"])
            )
            for metric in ("ready_cycles", "done_cycles", "initiation_interval_cycles"):
                rows.append(
                    metric_row(
                        unit="matrix_machine",
                        point=point,
                        opcode=opcode,
                        metric=metric,
                        measured=int(record[metric]),
                        predicted=predicted,
                        provenance="full_machine_measured",
                    )
                )
        elif opcode == "M_MM_WO":
            predicted_done = (
                int(config["gemm_writeout_busy_blen_coefficient"]) * blen
                + int(config["gemm_writeout_busy_overhead_coefficient"])
                * int(config["systolic_processing_overhead_cycles"])
                + int(config["gemm_writeout_busy_fixed_cycles"])
            )
            rows.append(
                metric_row(
                    unit="matrix_machine",
                    point=point,
                    opcode=opcode,
                    metric="backend_idle_cycles",
                    measured=int(record["done_cycles"]),
                    predicted=predicted_done,
                    provenance="full_machine_measured",
                )
            )
            unsupported.append(
                {
                    "point": point,
                    "opcode": opcode,
                    "first_write_pulse_cycles": int(record["ready_cycles"]),
                    "observed_result_pulses": int(record["observed_result_pulses"]),
                    "expected_result_rows": int(record["expected_result_rows"]),
                    "row_writeback_supported": bool(record["row_writeback_supported"]),
                    "policy": "all rows conservatively become ready at backend idle",
                }
            )
    return rows, unsupported


def vector_rows(raw: dict[str, Any], calibration: dict[str, Any]) -> list[dict]:
    config = calibration["vector"]
    fixed = {
        "V_ADD_VV": int(config["add_vv_cycles"]),
        "V_ADD_VF": int(config["add_vf_cycles"]),
        "V_SUB_VV": int(config["sub_vv_cycles"]),
        "V_SUB_VF": int(config["sub_vf_cycles"]),
        "V_MUL_VV": int(config["mul_vv_cycles"]),
        "V_MUL_VF": int(config["mul_vf_cycles"]),
        "V_EXP_VV": int(config["exp_cycles"]),
        "V_RECI_VV": int(config["reciprocal_cycles"]),
    }
    rows: list[dict[str, Any]] = []
    for record in raw.get("measurements", []):
        point = str(record["point_tag"])
        opcode = str(record["opcode"])
        vlen = int(record["vlen"])
        holdout = (opcode in {"V_RED_SUM", "V_RED_MAX"} and vlen == 64) or (
            int(record["fp_exp"]) == 6 and int(record["fp_mant"]) == 5
        )
        if opcode == "V_ADD_VV_II":
            predicted = int(config["initiation_interval_cycles"])
            for metric in (
                "accepted_interval_cycles",
                "result_interval_cycles",
                "initiation_interval_cycles",
            ):
                rows.append(
                    metric_row(
                        unit="vector_machine",
                        point=point,
                        opcode=opcode,
                        metric=metric,
                        measured=int(record[metric]),
                        predicted=predicted,
                        provenance="full_machine_measured",
                        holdout=holdout,
                    )
                )
            continue
        if opcode == "V_RED_SUM":
            predicted = int(config["reduce_sum_base_cycles"]) + int(
                config["reduce_sum_per_level_cycles"]
            ) * ceil_log2(vlen + 1)
        elif opcode == "V_RED_MAX":
            predicted = int(config["reduce_max_base_cycles"]) + int(
                config["reduce_max_per_level_cycles"]
            ) * ceil_log2(vlen + 1)
        else:
            predicted = fixed.get(opcode)
        if predicted is None:
            continue
        for metric in ("ready_cycles", "done_cycles"):
            rows.append(
                metric_row(
                    unit="vector_machine",
                    point=point,
                    opcode=opcode,
                    metric=metric,
                    measured=int(record[metric]),
                    predicted=predicted,
                    provenance="full_machine_measured",
                    holdout=holdout,
                )
            )
    return rows


def scalar_rows(raw: dict[str, Any], calibration: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    config = calibration["scalar"]
    values = {
        "S_ADD_FP": (config["fp_add_ready_cycles"], config["fp_add_done_cycles"]),
        "S_SUB_FP": (config["fp_sub_ready_cycles"], config["fp_sub_done_cycles"]),
        "S_MUL_FP": (config["fp_mul_ready_cycles"], config["fp_mul_done_cycles"]),
        "S_EXP_FP": (config["fp_exp_ready_cycles"], config["fp_exp_done_cycles"]),
        "S_RECI_FP": (
            config["fp_reciprocal_ready_cycles"],
            config["fp_reciprocal_done_cycles"],
        ),
        "S_SQRT_FP": (config["fp_sqrt_ready_cycles"], config["fp_sqrt_done_cycles"]),
    }
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for record in raw.get("measurements", []):
        point = str(record["point_tag"])
        opcode = str(record["opcode"])
        holdout = int(record["fp_exp"]) == 6 and int(record["fp_mant"]) == 5
        if opcode == "S_MAX_FP":
            unsupported.append(
                {
                    "point": point,
                    "opcode": opcode,
                    "implemented": bool(record["implemented"]),
                    "policy": "functional debug only; run marked unsupported_opcodes",
                }
            )
            continue
        if opcode == "S_MAP_V_FP":
            predicted_ready = int(record["vlen"]) + int(config["map_vector_ready_fixed_cycles"])
            predicted_done = int(record["vlen"]) + int(config["map_vector_done_fixed_cycles"])
        else:
            if opcode not in values:
                continue
            predicted_ready, predicted_done = map(int, values[opcode])
        for metric, predicted in (
            ("ready_cycles", predicted_ready),
            ("done_cycles", predicted_done),
        ):
            rows.append(
                metric_row(
                    unit="scalar_machine",
                    point=point,
                    opcode=opcode,
                    metric=metric,
                    measured=int(record[metric]),
                    predicted=predicted,
                    provenance="full_machine_measured",
                    holdout=holdout,
                )
            )
    return rows, unsupported


def summarize_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for unit in sorted({row["unit"] for row in rows}):
        selected = [row for row in rows if row["unit"] == unit]
        train = [row for row in selected if not row["holdout"]]
        holdout = [row for row in selected if row["holdout"]]
        summary[unit] = {
            "metrics": len(selected),
            "max_abs_error_cycles": max((row["abs_error_cycles"] for row in selected), default=0),
            "mean_abs_error_cycles": (
                sum(row["abs_error_cycles"] for row in selected) / len(selected)
                if selected
                else 0
            ),
            "all_within_one_cycle": all(row["within_one_cycle"] for row in selected),
            "train_metrics": len(train),
            "holdout_metrics": len(holdout),
            "holdout_max_abs_error_cycles": max(
                (row["abs_error_cycles"] for row in holdout), default=0
            ),
        }
    return summary


def artifact_completeness(
    matrix: dict[str, Any], vector: dict[str, Any], scalar: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "matrix_machine": {
            "m16_b4_e8m7",
            "m32_b4_e8m7",
            "m64_b4_e8m7",
            "m32_b8_e8m7",
            "m64_b8_e8m7",
            "m64_b16_e8m7",
        },
        "vector_machine": {
            "v8_e8m7",
            "v16_e8m7",
            "v32_e8m7",
            "v64_e8m7",
            "v32_e6m5",
        },
        "scalar_machine": {
            "v8_e8m7",
            "v16_e8m7",
            "v32_e8m7",
            "v64_e8m7",
            "v32_e6m5",
        },
    }
    artifacts = {
        "matrix_machine": matrix,
        "vector_machine": vector,
        "scalar_machine": scalar,
    }
    result: dict[str, Any] = {}
    for unit, artifact in artifacts.items():
        observed = {str(row["point_tag"]) for row in artifact.get("measurements", [])}
        missing = sorted(expected[unit] - observed)
        result[unit] = {
            "expected_points": len(expected[unit]),
            "observed_points": len(observed & expected[unit]),
            "missing_points": missing,
            "complete": not missing and artifact.get("failure") is None,
        }
    return result


def pipeline_summary(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {"available": False, "all_recovery_cycles_equal_one": False, "records": []}
    records = raw.get("measurements", [])
    return {
        "available": True,
        "all_recovery_cycles_equal_one": bool(records)
        and all(int(record.get("recovery_cycles", -1)) == 1 for record in records),
        "records": records,
    }


def scheduler_summary(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {"available": False, "case_count": 0, "checks": {}}
    cases = {case["name"]: case for case in raw.get("cases", [])}
    checks: dict[str, bool] = {}
    for name, case in cases.items():
        events = case.get("events", [])
        checks[f"{name}:monotonic"] = all(
            int(event["issue_cycle"])
            <= int(event["accepted_cycle"])
            <= int(event["start_cycle"])
            <= int(event["completion_cycle"])
            for event in events
        )
        stalled = [event for event in events if int(event["accepted_cycle"]) > int(event["issue_cycle"])]
        checks[f"{name}:stalled_ops_have_recovery"] = all(
            int(event["recovery_cycles"]) == 1 for event in stalled
        )
    matrix_events = cases.get("matrix_compute_writeout_row_consumer", {}).get("events", [])
    if len(matrix_events) == 3:
        checks["matrix_writeout_backend_queue"] = (
            matrix_events[1]["accepted_cycle"] < matrix_events[1]["start_cycle"]
            and matrix_events[2]["start_cycle"] > matrix_events[1]["completion_cycle"]
        )
    mixed_events = cases.get("mixed_vector_latency_in_order", {}).get("events", [])
    if len(mixed_events) == 2:
        checks["mixed_vector_result_order"] = (
            mixed_events[1]["start_cycle"] > mixed_events[0]["result_ready_cycle"]
        )
    return {
        "available": True,
        "case_count": len(cases),
        "checks": checks,
        "all_checks_pass": len(cases) == 6 and all(checks.values()),
        "cases": list(cases.values()),
    }


def mixed_vector_summary(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {"available": False, "pass": False, "records": []}
    records = [
        record
        for record in raw.get("measurements", [])
        if record.get("opcode") == "V_MIXED_ADD_MUL_ORDER"
    ]
    passed = bool(records) and all(
        int(record.get("unsafe_order_preserved", 1)) == 0
        and int(record.get("safe_order_preserved", 0)) == 1
        and int(record.get("safe_result_count", 0)) == 2
        for record in records
    )
    return {"available": bool(records), "pass": passed, "records": records}


def opcode_work_delta(
    current_profile: dict[str, Any] | None,
    fixed_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if current_profile is None or fixed_profile is None:
        return []
    before = current_profile.get("opcodes", {})
    after = fixed_profile.get("opcodes", {})
    rows = []
    for opcode in sorted(set(before) | set(after)):
        lhs = before.get(opcode, {})
        rhs = after.get(opcode, {})
        count = int(rhs.get("count", lhs.get("count", 0)))
        lhs_cycles = float(lhs.get("total_picos", 0)) / 1000.0
        rhs_cycles = float(rhs.get("total_picos", 0)) / 1000.0
        rows.append(
            {
                "opcode": opcode,
                "count": count,
                "current_resource_work_cycles": lhs_cycles,
                "fixed_resource_work_cycles": rhs_cycles,
                "resource_work_delta_cycles": rhs_cycles - lhs_cycles,
                "per_opcode_delta_cycles": (
                    (rhs_cycles - lhs_cycles) / count if count else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: abs(row["resource_work_delta_cycles"]), reverse=True)
    return rows


def functional_executor_cycles(stats: dict[str, Any] | None) -> float | None:
    if stats is None:
        return None
    value = stats.get("functional_executor_latency_ns")
    if value is not None:
        return float(value)
    log_path = stats.get("log_path")
    if not log_path or not Path(log_path).exists():
        return None
    match = re.search(
        r"functional_executor_latency=([0-9.eE+-]+)ns",
        Path(log_path).read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def system_summary(
    current_stats: dict[str, Any] | None,
    fixed_stats: dict[str, Any] | None,
    current_profile: dict[str, Any] | None,
    fixed_profile: dict[str, Any] | None,
    current_comparison: dict[str, Any] | None,
    fixed_comparison: dict[str, Any] | None,
    functional_regression: dict[str, Any] | None,
) -> dict[str, Any]:
    if current_stats is None and fixed_stats is None:
        return {"available": False}
    source = fixed_stats or current_stats or {}
    current_timeline = (current_stats or {}).get("timeline_profile", {})
    fixed_timeline = (fixed_stats or {}).get("timeline_profile", {})
    current_cycles = (current_stats or {}).get("sim_latency_ns")
    fixed_cycles = (fixed_stats or {}).get("sim_latency_ns")
    legacy_cycles = functional_executor_cycles(current_stats)
    comparison_equal = None
    if current_comparison is not None and fixed_comparison is not None:
        keys = ("mse", "mae", "max_error", "allclose_match_rate", "allclose_pass")
        comparison_equal = all(current_comparison.get(key) == fixed_comparison.get(key) for key in keys)

    def map_delta(field: str) -> dict[str, int]:
        before = current_timeline.get(field, {}) or {}
        after = fixed_timeline.get(field, {}) or {}
        return {
            key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in sorted(set(before) | set(after))
        }

    critical_delta = map_delta("critical_path_cycles")
    latency_delta = (
        float(fixed_cycles) - float(current_cycles)
        if fixed_cycles is not None and current_cycles is not None
        else None
    )
    return {
        "available": True,
        "clock_period_ps": 1000,
        "legacy_cycles": legacy_cycles,
        "current_rtl_v1_cycles": current_cycles,
        "fixed_rtl_v1_cycles": fixed_cycles,
        "fixed_minus_current_cycles": latency_delta,
        "fixed_minus_current_pct": (
            (float(fixed_cycles) - float(current_cycles)) * 100.0 / float(current_cycles)
            if fixed_cycles is not None and current_cycles
            else None
        ),
        "fixed_minus_legacy_pct": (
            (float(fixed_cycles) - float(legacy_cycles)) * 100.0 / float(legacy_cycles)
            if fixed_cycles is not None and legacy_cycles
            else None
        ),
        "functional_output_metrics_unchanged": comparison_equal,
        "functional_regression": functional_regression,
        "current_comparison": current_comparison,
        "fixed_comparison": fixed_comparison,
        "rtl_validation": source.get("rtl_validation"),
        "resource_work_cycles": source.get("timeline_profile", {}).get("resource_work_cycles", {}),
        "critical_path_cycles": source.get("timeline_profile", {}).get("critical_path_cycles", {}),
        "current_critical_path_cycles": current_timeline.get("critical_path_cycles", {}),
        "critical_path_delta_cycles": critical_delta,
        "critical_path_delta_sum_cycles": sum(critical_delta.values()),
        "latency_delta_reconciled": latency_delta is not None
        and abs(sum(critical_delta.values()) - latency_delta) <= 1.0,
        "stall_cycles_by_reason": source.get("timeline_profile", {}).get(
            "stall_cycles_by_reason", {}
        ),
        "stall_cycle_delta_by_reason": map_delta("stall_cycles_by_reason"),
        "memory_compute_overlap_cycles": source.get("timeline_profile", {}).get(
            "memory_compute_overlap_cycles"
        ),
        "memory_compute_overlap_pct": source.get("timeline_profile", {}).get(
            "memory_compute_overlap_pct"
        ),
        "opcode_resource_work_delta": opcode_work_delta(current_profile, fixed_profile),
        "configuration": {
            "MLEN": source.get("emu_mlen"),
            "VLEN": source.get("emu_vlen"),
            "BLEN": source.get("emu_blen"),
            "HBM_CHANNELS": source.get("hbm_channels"),
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    errors = payload["microbench_error_summary"]
    system = payload["system_validation"]
    lines = [
        "# Transactional Emulator `rtl-v1` Latency Validation",
        "",
        "## Scope",
        "",
        "This report validates timing only. Numerical golden data, tolerances, and PASS criteria are unchanged.",
        "Cycles are primary; nanoseconds use `CLOCK_PERIOD_PS=1000` (1 GHz assumption), not a timing-closed fmax.",
        "Ramulator service is currently placed on a post-hoc scheduler timeline rather than online cycle-coupled co-simulation.",
        "",
        "## Full-Machine RTL Microbenchmarks",
        "",
        "| Unit | Metrics | Max abs. error (cycles) | Holdout metrics | Holdout max error | <=1 cycle |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for unit, summary in errors.items():
        complete = payload["artifact_completeness"][unit]
        lines.append(
            f"| {unit} | {summary['metrics']} | {summary['max_abs_error_cycles']:.3f} | "
            f"{summary['holdout_metrics']} | {summary['holdout_max_abs_error_cycles']:.3f} | "
            f"{'PASS' if summary['all_within_one_cycle'] and complete['complete'] else 'INCOMPLETE/FAIL'} |"
        )

    matrix_points: dict[int, dict[int, int]] = {}
    for row in payload["microbench_rows"]:
        if row["opcode"] == "M_MM" and row["metric"] == "ready_cycles":
            point = row["point"]
            mlen = int(point.split("_")[0][1:])
            blen = int(point.split("_")[1][1:])
            matrix_points.setdefault(blen, {})[mlen] = int(row["measured_cycles"])
    lines += ["", "### Matrix MLEN Independence Check", "", "| BLEN | MLEN -> measured M_MM cycles |", "|---:|---|"]
    for blen, points in sorted(matrix_points.items()):
        values = ", ".join(f"{mlen}->{cycles}" for mlen, cycles in sorted(points.items()))
        lines.append(f"| {blen} | {values} |")

    lines += [
        "",
        "`M_MM_WO` is not claimed cycle-exact for consumer readiness: current RTL emits one write-valid pulse for a BLEN-row result.",
        "The emulator therefore waits until backend idle for every output row.",
        "",
        "## Hazard Scheduler Evidence",
        "",
        f"- RTL pipeline-control recovery checks: {'PASS' if payload['pipeline_control']['all_recovery_cycles_equal_one'] else 'FAIL'}",
        f"- Scheduler differential cases: {payload['scheduler_differential']['case_count']}/6",
        f"- Scheduler invariant checks: {'PASS' if payload['scheduler_differential'].get('all_checks_pass') else 'FAIL'}",
        f"- Full-VectorMachine mixed-latency ordering: {'PASS' if payload['mixed_vector_order'].get('pass') else 'MISSING/FAIL'}",
        "",
    ]
    for record in payload["pipeline_control"].get("records", []):
        lines.append(
            f"- `{record['opcode']}`: raw stall={record.get('raw_stall_cycles')} cycles, "
            f"recovery={record.get('recovery_cycles')} cycle"
        )

    lines += ["", "## Qwen3-32B One-Layer System Result", ""]
    if system.get("available"):
        validation = system.get("rtl_validation") or {}
        regression = system.get("functional_regression") or {}
        lines += [
            "| Mode | Cycles | Time at 1 GHz |",
            "|---|---:|---:|",
            f"| Legacy functional executor | {system.get('legacy_cycles')} | {system.get('legacy_cycles')} ns |",
            f"| Previous rtl-v1 | {system.get('current_rtl_v1_cycles')} | {system.get('current_rtl_v1_cycles')} ns |",
            f"| Fixed rtl-v1 | {system.get('fixed_rtl_v1_cycles')} | {system.get('fixed_rtl_v1_cycles')} ns |",
            "",
            f"Functional comparison metrics unchanged: `{system.get('functional_output_metrics_unchanged')}`.",
            f"Decoded BF16 output bitwise identical: `{regression.get('decoded_output_bitwise_identical', 'not available')}`.",
            f"Fixed vs previous rtl-v1: `{system.get('fixed_minus_current_cycles')}` cycles "
            f"(`{format_optional_float(system.get('fixed_minus_current_pct'))}%`).",
            f"Fixed vs legacy-equivalent serial executor: "
            f"`{format_optional_float(system.get('fixed_minus_legacy_pct'))}%`.",
            f"RTL validation status: `{validation.get('status', 'not available')}`.",
            f"Validated opcode coverage: `{validation.get('validated_opcodes', 0)}/{validation.get('total_opcodes', 0)}`; "
            f"unsupported=`{validation.get('unsupported_opcodes', 0)}`, "
            f"out-of-domain=`{validation.get('out_of_domain_opcodes', 0)}`.",
            f"Unsupported tail sensitivity: `{validation.get('unsupported_makespan_sensitivity_cycles', 0)}` cycles "
            "(local tail removal only; not a rescheduled counterfactual).",
            f"Latency delta reconciled by critical-path attribution: `{system.get('latency_delta_reconciled')}` "
            f"(attributed sum=`{system.get('critical_path_delta_sum_cycles')}` cycles).",
            "",
            "### Largest Opcode Resource-Work Deltas",
            "",
            "| Opcode | Count | Total delta (cycles) | Delta/op (cycles) |",
            "|---|---:|---:|---:|",
        ]
        for row in system.get("opcode_resource_work_delta", [])[:12]:
            lines.append(
                f"| {row['opcode']} | {row['count']} | {row['resource_work_delta_cycles']:.0f} | "
                f"{row['per_opcode_delta_cycles']:.3f} |"
            )

        work = system.get("resource_work_cycles", {})
        current_critical = system.get("current_critical_path_cycles", {})
        critical = system.get("critical_path_cycles", {})
        critical_delta = system.get("critical_path_delta_cycles", {})
        resources = sorted(set(work) | set(current_critical) | set(critical))
        lines += [
            "",
            "### Resource Work and Critical Path",
            "",
            "Resource work can overlap and therefore need not sum to makespan. Critical-path contributions are mutually exclusive.",
            "",
            "| Resource | Fixed work | Previous critical path | Fixed critical path | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for resource in resources:
            lines.append(
                f"| {resource} | {int(work.get(resource, 0))} | "
                f"{int(current_critical.get(resource, 0))} | {int(critical.get(resource, 0))} | "
                f"{int(critical_delta.get(resource, 0))} |"
            )

        lines += [
            "",
            "### Largest Stall Deltas",
            "",
            "| Stall reason | Delta cycles |",
            "|---|---:|",
        ]
        for reason, delta in sorted(
            system.get("stall_cycle_delta_by_reason", {}).items(),
            key=lambda item: (-abs(int(item[1])), item[0]),
        )[:12]:
            lines.append(f"| {reason} | {int(delta)} |")

        unsupported_counts = validation.get("unsupported_opcode_counts", {})
        unsupported_cycles = validation.get("unsupported_resource_cycles", {})
        lines += [
            "",
            "### Unsupported RTL Coverage",
            "",
            "| Opcode | Count |",
            "|---|---:|",
        ]
        if unsupported_counts:
            for opcode, count in sorted(
                unsupported_counts.items(), key=lambda item: (-int(item[1]), item[0])
            ):
                lines.append(f"| {opcode} | {int(count)} |")
        else:
            lines.append("| none | 0 |")
        lines += [
            "",
            "Unsupported resource work: "
            + ", ".join(
                f"`{resource}={int(cycles)}`"
                for resource, cycles in sorted(unsupported_cycles.items())
            )
            if unsupported_cycles
            else "Unsupported resource work: none.",
        ]
    else:
        lines.append("System run artifacts were not supplied.")

    lines += [
        "",
        "## Claim Boundary",
        "",
        "Strong claims apply to tested full-Machine compute timings and the normalized pipeline-control hazards above.",
        "Runs containing `S_MAX_FP`, unsupported Matrix broadcast/writeout opcodes, or out-of-domain production shapes are labeled accordingly and are not cycle-exact RTL validation.",
        "Memory-bound accuracy remains limited by post-hoc Ramulator arrival times; online memory/compute co-simulation is deferred.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "matrix_full_machine/raw_measurements.json",
    )
    parser.add_argument(
        "--vector-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "vector_full_machine_v2/raw_measurements.json",
    )
    parser.add_argument(
        "--scalar-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "scalar_full_machine/raw_measurements.json",
    )
    parser.add_argument(
        "--pipeline-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "pipeline_control/raw_measurements.json",
    )
    parser.add_argument(
        "--scheduler-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "scheduler_differential_traces.json",
    )
    parser.add_argument(
        "--vector-mixed-artifact",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "vector_mixed_validation/raw_measurements.json",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--current-stats", type=Path)
    parser.add_argument("--fixed-stats", type=Path)
    parser.add_argument("--current-profile", type=Path)
    parser.add_argument("--fixed-profile", type=Path)
    parser.add_argument("--current-comparison", type=Path)
    parser.add_argument("--fixed-comparison", type=Path)
    parser.add_argument(
        "--functional-regression",
        type=Path,
        help="Timing-only Qwen output-hash comparison generated by run_qwen_rtl_v1_validation.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVIDENCE_ROOT / "report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless microbenchmarks, pipeline recovery, and six scheduler cases pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration = load_json(args.calibration)
    matrix = load_json(args.matrix_artifact)
    vector = load_json(args.vector_artifact)
    scalar = load_json(args.scalar_artifact)
    if not all((calibration, matrix, vector, scalar)):
        raise FileNotFoundError("calibration and all three full-Machine artifacts are required")

    matrix_metrics, matrix_unsupported = matrix_rows(matrix, calibration)
    vector_metrics = vector_rows(vector, calibration)
    scalar_metrics, scalar_unsupported = scalar_rows(scalar, calibration)
    metrics = matrix_metrics + vector_metrics + scalar_metrics
    errors = summarize_errors(metrics)
    completeness = artifact_completeness(matrix, vector, scalar)
    pipeline = pipeline_summary(load_json(args.pipeline_artifact))
    scheduler = scheduler_summary(load_json(args.scheduler_artifact))
    mixed_vector = mixed_vector_summary(load_json(args.vector_mixed_artifact))
    system = system_summary(
        load_json(args.current_stats),
        load_json(args.fixed_stats),
        load_json(args.current_profile),
        load_json(args.fixed_profile),
        load_json(args.current_comparison),
        load_json(args.fixed_comparison),
        load_json(args.functional_regression),
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "latency fidelity only; correctness gate unchanged",
        "clock_period_ps": calibration["clock_period_assumption_ps"],
        "fmax_validated": False,
        "ramulator_scheduler_coupling": "post_hoc_arrival_time",
        "calibration_source": calibration["source"],
        "microbench_error_summary": errors,
        "artifact_completeness": completeness,
        "microbench_rows": metrics,
        "unsupported_microbench_paths": matrix_unsupported + scalar_unsupported,
        "pipeline_control": pipeline,
        "scheduler_differential": scheduler,
        "mixed_vector_order": mixed_vector,
        "system_validation": system,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "rtl_v1_latency_validation.json"
    markdown_path = args.out_dir / "rtl_v1_latency_validation.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")

    strict_ok = (
        errors
        and all(summary["all_within_one_cycle"] for summary in errors.values())
        and all(summary["complete"] for summary in completeness.values())
        and pipeline["all_recovery_cycles_equal_one"]
        and scheduler.get("all_checks_pass", False)
        and mixed_vector.get("pass", False)
    )
    if args.fixed_stats is not None:
        regression = system.get("functional_regression") or {}
        strict_ok = strict_ok and bool(
            regression.get("decoded_output_bitwise_identical")
            and regression.get("comparison_metrics_identical")
        )
    return 0 if strict_ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
