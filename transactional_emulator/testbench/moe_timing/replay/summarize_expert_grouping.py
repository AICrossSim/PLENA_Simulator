#!/usr/bin/env python3
"""Audit pair-major versus expert-major routed-MoE replay results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = (
    "trace_id",
    "model_name",
    "benchmark",
    "sample_id",
    "phase",
    "layer",
    "rows",
    "hidden",
    "intermediate",
    "num_experts",
    "top_k",
    "pair_count",
    "selected_expert_count",
    "mlen",
    "blen",
    "hbm_weight_layout",
    "coalesce_hbm_bursts",
    "include_shared_expert",
    "shared_experts",
    "shared_intermediate",
    "shared_gate",
)


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    document.setdefault("include_shared_expert", False)
    document.setdefault("shared_experts", 0)
    document.setdefault("shared_intermediate", 0)
    document.setdefault("shared_gate", "none")
    strides = document.get("weight_table_strides", {})
    routed_weight_bytes = sum(int(value) for value in strides.values())
    fetches = (
        int(document["pair_count"])
        if document.get("execution_order") == "pair_major"
        else int(document["group_count"])
    )
    document.setdefault("expected_routed_weight_bytes", fetches * routed_weight_bytes)
    document.setdefault("expected_shared_weight_bytes", 0)
    functional_gate = document.get("functional_gate", document.get("zero_input_smoke_gate", {}))
    if not functional_gate.get("passed"):
        raise ValueError(f"{path}: functional gate did not pass")
    document["functional_gate"] = functional_gate
    metrics = document.get("run_metrics", {})
    if metrics.get("return_code") != 0:
        raise ValueError(f"{path}: emulator return code is not zero")
    if metrics.get("timing_model") != "serial":
        raise ValueError(f"{path}: expected serial attribution run")
    expected_bytes = int(document["expected_routed_weight_bytes"]) + int(document["expected_shared_weight_bytes"])
    measured_bytes = metrics.get("hbm_bytes_read")
    byte_gate = document.get("hbm_weight_byte_gate")
    if byte_gate is not None and not byte_gate.get("passed"):
        raise ValueError(f"{path}: HBM weight-byte gate did not pass")
    if measured_bytes != expected_bytes:
        raise ValueError(f"{path}: measured HBM bytes {measured_bytes} != formula {expected_bytes}")
    return document


def _stage_profile(document: dict[str, Any]) -> dict[str, Any]:
    path = Path(document["run_metrics"]["stage_profile_path"])
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile["cycle_accounting_status"] != "profiled_time_matches_total":
        raise ValueError(f"{path}: stage cycle accounting does not close")
    if int(profile["total_simulation_cycles"]) != int(document["run_metrics"]["sim_latency_cycles"]):
        raise ValueError(f"{path}: stage total differs from run total")
    return profile


def _check_repeat_gate(document: dict[str, Any], path: Path) -> None:
    repeat = document.get("repeat_gate")
    if not repeat or repeat.get("repeats") != 3 or not repeat.get("passed"):
        raise ValueError(f"{path}: a passing three-run repeat gate is required")


def summarize_case(
    label: str,
    pair_path: Path,
    group_path: Path,
    *,
    require_repeat_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair = _load(pair_path)
    group = _load(group_path)
    if pair.get("execution_order") != "pair_major":
        raise ValueError(f"{pair_path}: expected execution_order=pair_major")
    if group.get("execution_order") != "expert_major":
        raise ValueError(f"{group_path}: expected execution_order=expert_major")
    for field in IDENTITY_FIELDS:
        if pair.get(field) != group.get(field):
            raise ValueError(f"{label}: pair/group differ in {field}: {pair.get(field)!r} != {group.get(field)!r}")
    if require_repeat_gate:
        _check_repeat_gate(pair, pair_path)
        _check_repeat_gate(group, group_path)

    pair_profile = _stage_profile(pair)
    group_profile = _stage_profile(group)
    pair_cycles = int(pair["run_metrics"]["sim_latency_cycles"])
    group_cycles = int(group["run_metrics"]["sim_latency_cycles"])
    pair_bytes = int(pair["run_metrics"]["hbm_bytes_read"])
    group_bytes = int(group["run_metrics"]["hbm_bytes_read"])
    pair_count = int(pair["pair_count"])
    group_count = int(group["group_count"])
    routed_reuse = pair_count / group_count
    shared_bytes = int(pair["expected_shared_weight_bytes"])
    pair_routed_bytes = int(pair["expected_routed_weight_bytes"])
    group_routed_bytes = int(group["expected_routed_weight_bytes"])

    row = {
        "case": label,
        "scope": "complete_shared_moe" if pair["include_shared_expert"] else "routed_branch_only",
        "model": pair.get("model_name", pair["trace_id"].split("_swe_bench", 1)[0]),
        "benchmark": pair["benchmark"],
        "batch_tokens": pair["rows"],
        "layer": pair["layer"],
        "hidden": pair["hidden"],
        "routed_intermediate": pair["intermediate"],
        "shared_intermediate": pair["shared_intermediate"],
        "route_pairs": pair_count,
        "active_routed_experts": group_count,
        "routed_weight_reuse_factor": routed_reuse,
        "pair_major_cycles": pair_cycles,
        "expert_major_cycles": group_cycles,
        "cycle_speedup": pair_cycles / group_cycles,
        "pair_major_physical_hbm_bytes": pair_bytes,
        "expert_major_physical_hbm_bytes": group_bytes,
        "physical_hbm_byte_reduction_pct": 100.0 * (pair_bytes - group_bytes) / pair_bytes,
        "pair_major_routed_weight_bytes": pair_routed_bytes,
        "expert_major_routed_weight_bytes": group_routed_bytes,
        "shared_weight_bytes_unchanged": shared_bytes,
        "pair_replay_gate_kind": pair["functional_gate"].get("gate_kind"),
        "group_replay_gate_kind": group["functional_gate"].get("gate_kind"),
        "pair_replay_rel_rms": pair["functional_gate"]["rel_rms"],
        "group_replay_rel_rms": group["functional_gate"]["rel_rms"],
        "pair_result": str(pair_path.resolve()),
        "group_result": str(group_path.resolve()),
    }

    stages: list[dict[str, Any]] = []
    stage_names = sorted(set(pair_profile["stages"]) | set(group_profile["stages"]))
    for stage_name in stage_names:
        pair_stage = pair_profile["stages"].get(stage_name, {})
        group_stage = group_profile["stages"].get(stage_name, {})
        pair_stage_cycles = int(pair_stage.get("wall_cycles", 0))
        group_stage_cycles = int(group_stage.get("wall_cycles", 0))
        stages.append(
            {
                "case": label,
                "scope": row["scope"],
                "stage": stage_name,
                "pair_major_cycles": pair_stage_cycles,
                "expert_major_cycles": group_stage_cycles,
                "cycles_saved": pair_stage_cycles - group_stage_cycles,
                "pair_major_time_pct": 100.0 * pair_stage_cycles / pair_cycles,
                "expert_major_time_pct": 100.0 * group_stage_cycles / group_cycles,
                "pair_major_physical_hbm_bytes": int(pair_stage.get("physical_hbm_bytes_read", 0)),
                "expert_major_physical_hbm_bytes": int(group_stage.get("physical_hbm_bytes_read", 0)),
            }
        )
    return row, stages


def write_outputs(rows: list[dict[str, Any]], stages: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "expert_grouping_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stage_csv = output_dir / "expert_grouping_stage_cycles.csv"
    with stage_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)

    manifest = {
        "schema_version": "plena.moe_expert_grouping.v1",
        "measurement_scope": (
            "Rust serial simulation cycles with Ramulator-backed 64B HBM requests; "
            "fixed-route trace replay, not RTL-measured cycles"
        ),
        "cases": rows,
        "gates": {
            "same_trace_and_hardware_config": True,
            "replay_result_gate_passed": True,
            "physical_hbm_bytes_equal_formula": True,
            "stage_cycles_close_to_total": True,
        },
    }
    (output_dir / "expert_grouping_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        "| Case | Scope | Route pairs -> active experts | Per-route baseline cycles | Grouped + resident-weight cycles | Speedup | HBM bytes reduction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['case']} | {row['scope']} | {row['route_pairs']} -> {row['active_routed_experts']} | "
            f"{row['pair_major_cycles']:,} | {row['expert_major_cycles']:,} | "
            f"{row['cycle_speedup']:.3f}x | {row['physical_hbm_byte_reduction_pct']:.2f}% |"
        )
    report = """# PLENA MoE 同专家分组与权重驻留验证

## 结果

{table}

## A/B 只改变了什么

- **Pair-major baseline**：每个 `(token, expert)` route pair 独立执行 Gate/Up/Down，并重复读取该专家全部权重。
- **Expert-major grouped**：同一 layer-step 中相同 expert 的 token 紧凑打包；该专家的 Gate/Up/Down 权重各读取一次，再逐 token 应用原 route weight 并 scatter-add 回原 token。
- 两边使用相同真实 route IDs/weights、模型维度、MX 布局、64B burst、Ramulator 配置和 Rust serial timing model。
- 完整维度 timing 使用零值张量，因此只验证形状、执行与字节闭合；数值正确性由同一路径的小维度非零 BF16/MX golden 单独验证。

## 可以下的结论

这项优化减少的是重复的 routed-expert 权重读取。routed-only 的物理字节下降应严格等于 `route_pairs -> active_experts` 的去重比例；包含 shared expert 时，shared 权重流量保持不变，因此 complete-MoE 的总收益会被稀释。

## 不能下的结论

- 这是固定真实 route trace 的 Rust/Ramulator 重放，不是 device-selected router 的运行时排序实现。
- 这些不是 RTL `EXECUTION CLOCKS`，绝对周期仍需 RTL primitive 校准。
- 单个代表 layer-step 不能替代 SWE-bench 全 workload、所有 batch/layer/phase 的加权结论。
""".format(table="\n".join(table))
    (output_dir / "EXPERT_GROUPING_REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "PAIR_RESULT", "GROUP_RESULT"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-repeat-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for label, pair_path, group_path in args.case:
        row, case_stages = summarize_case(
            label,
            Path(pair_path),
            Path(group_path),
            require_repeat_gate=args.require_repeat_gate,
        )
        rows.append(row)
        stages.extend(case_stages)
    write_outputs(rows, stages, args.output_dir.resolve())
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
