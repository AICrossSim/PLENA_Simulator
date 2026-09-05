#!/usr/bin/env python3
"""Validate and summarize a fixed-capacity Matrix-SRAM panel-depth sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCOREBOARD_RE = re.compile(
    r"data_stall_picos=(?P<data>[0-9]+).*?"
    r"structural_stall_picos=(?P<structural>[0-9]+).*?"
    r"dma_wait_picos=(?P<dma_wait>[0-9]+).*?"
    r"matrix_busy_pct=(?P<matrix>[0-9.eE+-]+).*?"
    r"vector_busy_pct=(?P<vector>[0-9.eE+-]+).*?"
    r"scalar_busy_pct=(?P<scalar>[0-9.eE+-]+).*?"
    r"dma_busy_pct=(?P<dma>[0-9.eE+-]+)"
)


def _load_case(directory: Path) -> dict[str, Any]:
    result_path = directory / "moe_trace_replay_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result["run_metrics"]
    gate = result["functional_gate"]
    byte_gate = result["hbm_weight_byte_gate"]
    repeat_gate = result.get("repeat_gate")
    if not gate["passed"] or float(gate["rel_rms"]) >= 0.01:
        raise ValueError(f"{directory.name}: functional gate failed")
    if not byte_gate["passed"]:
        raise ValueError(f"{directory.name}: HBM byte gate failed")
    if repeat_gate is None or not repeat_gate["passed"]:
        raise ValueError(f"{directory.name}: repeat-3 determinism gate missing or failed")
    if metrics["timing_model"] != "scoreboard" or metrics["scoreboard_serialize"]:
        raise ValueError(f"{directory.name}: expected non-serialized scoreboard timing")

    log = Path(metrics["log_path"]).read_text(encoding="utf-8")
    match = SCOREBOARD_RE.search(log)
    if match is None:
        raise ValueError(f"{directory.name}: scoreboard counters missing")
    return {
        "case": directory.name,
        "trace_id": result["trace_id"],
        "model": result.get("model_name", "unknown"),
        "benchmark": result.get("benchmark", "unknown"),
        "phase": result.get("phase", "unknown"),
        "layer": result.get("layer"),
        "route_rows": result.get("rows"),
        "hidden": result.get("hidden"),
        "intermediate": result.get("intermediate"),
        "shared_intermediate": result.get("shared_intermediate"),
        "active_routed_experts": result.get("selected_expert_count"),
        "panel_mode": result["weight_panel_mode"],
        "panel_buffer_depth": int(result["panel_buffer_depth"]),
        "mram_tile_capacity": int(result["mram_tile_capacity"]),
        "matrix_sram_bytes": int(result["mram_tile_capacity"]) * 64 * 64 * 2,
        "cycles": int(metrics["sim_latency_cycles"]),
        "physical_hbm_bytes": int(metrics["hbm_bytes_read"]),
        "functional_gate_kind": str(gate.get("gate_kind", "unknown")),
        "functional_rel_rms": float(gate["rel_rms"]),
        "data_stall_cycles": int(match.group("data")) // 1000,
        "structural_stall_cycles": int(match.group("structural")) // 1000,
        "dma_wait_cycles": int(match.group("dma_wait")) // 1000,
        "matrix_busy_pct": float(match.group("matrix")),
        "vector_busy_pct": float(match.group("vector")),
        "scalar_busy_pct": float(match.group("scalar")),
        "dma_busy_pct": float(match.group("dma")),
        "repeat_cycles": repeat_gate["series"]["sim_latency_cycles"],
    }


def summarize(
    root: Path,
    *,
    case_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directories = sorted(
        (
            path
            for path in root.iterdir()
            if (path / "moe_trace_replay_results.json").is_file()
            and (case_prefix is None or path.name.startswith(case_prefix))
        ),
        key=lambda path: path.name,
    )
    rows = [_load_case(directory) for directory in directories]
    rows.sort(key=lambda row: row["panel_buffer_depth"])
    if not rows:
        raise ValueError(f"no completed ring-sweep cases found under {root}")
    depths = [row["panel_buffer_depth"] for row in rows]
    if len(depths) != len(set(depths)):
        raise ValueError(f"duplicate panel depths: {depths}")
    if depths[0] != 1:
        raise ValueError("the sweep requires a depth-1 blocking baseline")
    for field, label in (
        ("trace_id", "route trace"),
        ("model", "model"),
        ("hidden", "hidden dimension"),
        ("intermediate", "routed intermediate dimension"),
        ("shared_intermediate", "shared intermediate dimension"),
        ("functional_gate_kind", "functional gate kind"),
        ("physical_hbm_bytes", "physical HBM bytes"),
        ("mram_tile_capacity", "Matrix-SRAM capacity"),
    ):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"sweep changed {label}")

    baseline_cycles = rows[0]["cycles"]
    previous_cycles = None
    for row in rows:
        row["speedup_vs_depth1"] = baseline_cycles / row["cycles"]
        row["incremental_speedup_vs_previous"] = (
            1.0 if previous_cycles is None else previous_cycles / row["cycles"]
        )
        row["effective_hbm_bytes_per_cycle"] = row["physical_hbm_bytes"] / row["cycles"]
        previous_cycles = row["cycles"]
    winner = min(rows, key=lambda row: row["cycles"])
    summary = {
        "schema_version": "plena.sram_ring_sweep.v1",
        "trace_id": rows[0]["trace_id"],
        "model": rows[0]["model"],
        "benchmark": rows[0]["benchmark"],
        "phase": rows[0]["phase"],
        "layer": rows[0]["layer"],
        "route_rows": rows[0]["route_rows"],
        "hidden": rows[0]["hidden"],
        "intermediate": rows[0]["intermediate"],
        "shared_intermediate": rows[0]["shared_intermediate"],
        "active_routed_experts": rows[0]["active_routed_experts"],
        "functional_gate_kind": rows[0]["functional_gate_kind"],
        "physical_hbm_bytes": rows[0]["physical_hbm_bytes"],
        "mram_tile_capacity": rows[0]["mram_tile_capacity"],
        "matrix_sram_bytes": rows[0]["matrix_sram_bytes"],
        "winner_depth": winner["panel_buffer_depth"],
        "winner_cycles": winner["cycles"],
        "winner_speedup": winner["speedup_vs_depth1"],
        "gates": {
            "same_trace": True,
            "same_physical_hbm_bytes": True,
            "same_matrix_sram_capacity": True,
            "functional_rel_rms_below_0_01": True,
            "repeat_3_deterministic": True,
        },
        "scope": (
            "Compiler plus Rust/Ramulator event timing for panel lookahead within one "
            "expert projection at a time. Matrix-SRAM capacity and HBM bytes are fixed, "
            "but multi-expert residency and physical SRAM-bank conflicts are not modeled yet."
        ),
    }
    return rows, summary


def write_outputs(root: Path, output_dir: Path, *, case_prefix: str | None = None) -> None:
    rows, summary = summarize(root, case_prefix=case_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "sram_ring_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "sram_ring_sweep.json").write_text(
        json.dumps({**summary, "cases": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    if summary["functional_gate_kind"] == "nonzero_reference":
        value_scope = "上述维度通过非零 MX 权重与 BF16 reference；若小于原模型，不代表原模型绝对周期。"
    else:
        value_scope = (
            "上述维度只通过零值执行形状检查；非零数值正确性由同一 lowering 的缩放案例单独验证，"
            "本表只比较调度周期。"
        )

    lines = [
        "# Matrix SRAM 固定容量 Panel-Depth Sweep",
        "",
        f"- 模型：`{summary['model']}`",
        f"- 路由来源：`{summary['benchmark']}` / `{summary['phase']}` / layer `{summary['layer']}`",
        f"- 本次路由行数：`{summary['route_rows']}`，活跃 routed experts：`{summary['active_routed_experts']}`",
        f"- 执行维度：H=`{summary['hidden']}`，routed I=`{summary['intermediate']}`，shared I=`{summary['shared_intermediate']}`",
        f"- 数值验证类型：`{summary['functional_gate_kind']}`",
        f"- {value_scope}",
        "",
        "所有行使用同一条真实路由、相同 512 KiB Matrix SRAM、相同 HBM 字节和相同计算。",
        "",
        "| 同时在途 panel 深度 | Cycles | 相对 depth=1 加速 | 相对上一深度 | HBM bytes | rel-RMS | 数据等待 | DMA 等待 | Matrix busy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['panel_buffer_depth']} | {row['cycles']:,} | "
            f"{row['speedup_vs_depth1']:.3f}x | {row['incremental_speedup_vs_previous']:.4f}x | "
            f"{row['physical_hbm_bytes']:,} | "
            f"{row['functional_rel_rms']:.6f} | {row['data_stall_cycles']:,} | "
            f"{row['dma_wait_cycles']:,} | {row['matrix_busy_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- 最快深度：`{summary['winner_depth']}`，相对 blocking depth=1 为 `{summary['winner_speedup']:.3f}x`。",
            "- 每一行均通过表头标明的 functional gate、HBM byte gate 和 repeat-3 determinism gate。",
            "- 本表隔离的是更多在途 panel 的效果，不包含少搬权重带来的 grouping 收益。",
            "- 当前实现只在单个 expert projection 内增加 panel lookahead；它不是多专家同时驻留的 elastic pool 实测。",
            "- 当前 Rust SRAM 是功能存储，尚未计入真实 bank/port 冲突；因此这是 bank-aware RTL 前的比较时序，不是 SRAM 宏最终签核。",
            "",
        ]
    )
    (output_dir / "SRAM_RING_SWEEP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-prefix")
    args = parser.parse_args()
    write_outputs(
        args.root.resolve(),
        args.output_dir.resolve(),
        case_prefix=args.case_prefix,
    )
    print(args.output_dir.resolve() / "SRAM_RING_SWEEP_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
