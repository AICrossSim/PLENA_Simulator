#!/usr/bin/env python3
"""Audit and summarize blocking versus ping-pong Matrix-SRAM replays."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VARIANTS = (
    "blocking_serial",
    "blocking_scoreboard_serialized",
    "blocking_scoreboard",
    "pingpong_serial",
    "pingpong_scoreboard_serialized",
    "pingpong_scoreboard",
)

SCOREBOARD_RE = re.compile(
    r"data_stall_picos=(?P<data>[0-9]+).*?"
    r"structural_stall_picos=(?P<structural>[0-9]+).*?"
    r"dma_wait_picos=(?P<dma_wait>[0-9]+).*?"
    r"matrix_busy_pct=(?P<matrix>[0-9.eE+-]+).*?"
    r"vector_busy_pct=(?P<vector>[0-9.eE+-]+).*?"
    r"scalar_busy_pct=(?P<scalar>[0-9.eE+-]+).*?"
    r"dma_busy_pct=(?P<dma>[0-9.eE+-]+)"
)


def _load(root: Path, variant: str) -> dict[str, Any]:
    result_path = root / variant / "moe_trace_replay_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    profile_path = Path(result["run_metrics"]["stage_profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not result["functional_gate"]["passed"]:
        raise ValueError(f"{variant}: functional gate failed")
    if not result["hbm_weight_byte_gate"]["passed"]:
        raise ValueError(f"{variant}: HBM byte gate failed")

    metrics = result["run_metrics"]
    row: dict[str, Any] = {
        "variant": variant,
        "trace_id": result["trace_id"],
        "panel_mode": result["weight_panel_mode"],
        "timing_model": metrics["timing_model"],
        "scoreboard_serialize": bool(metrics["scoreboard_serialize"]),
        "cycles": int(metrics["sim_latency_cycles"]),
        "physical_hbm_bytes": int(metrics["hbm_bytes_read"]),
        "functional_rel_rms": float(result["functional_gate"]["rel_rms"]),
        "stage_profile_status": profile["cycle_accounting_status"],
        "stage_profile_additive_cycles": int(profile["total_profiled_cycles"]),
        "stage_profile_overlap_excess_cycles": max(
            0,
            int(profile["total_profiled_cycles"]) - int(metrics["sim_latency_cycles"]),
        ),
    }
    log = Path(metrics["log_path"]).read_text(encoding="utf-8")
    match = SCOREBOARD_RE.search(log)
    if metrics["timing_model"] == "scoreboard" and match is None:
        raise ValueError(f"{variant}: scoreboard counters missing from emulator log")
    if match is not None:
        row.update(
            {
                "data_stall_cycles": int(match.group("data")) // 1000,
                "structural_stall_cycles": int(match.group("structural")) // 1000,
                "dma_wait_cycles": int(match.group("dma_wait")) // 1000,
                "matrix_busy_pct": float(match.group("matrix")),
                "vector_busy_pct": float(match.group("vector")),
                "scalar_busy_pct": float(match.group("scalar")),
                "dma_busy_pct": float(match.group("dma")),
            }
        )
    return row


def summarize(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [_load(root, variant) for variant in VARIANTS]
    by_name = {row["variant"]: row for row in rows}
    if len({row["trace_id"] for row in rows}) != 1:
        raise ValueError("variants do not use the same route trace")
    if len({row["physical_hbm_bytes"] for row in rows}) != 1:
        raise ValueError("SRAM/timing variants changed physical HBM bytes")
    for panel in ("blocking", "pingpong"):
        serial = by_name[f"{panel}_serial"]
        control = by_name[f"{panel}_scoreboard_serialized"]
        if serial["cycles"] != control["cycles"]:
            raise ValueError(f"{panel}: serialized scoreboard does not reproduce serial cycles")

    blocking_serial = by_name["blocking_serial"]["cycles"]
    blocking_async = by_name["blocking_scoreboard"]["cycles"]
    pingpong_serial = by_name["pingpong_serial"]["cycles"]
    pingpong_async = by_name["pingpong_scoreboard"]["cycles"]
    summary = {
        "schema_version": "plena.sram_pingpong_ablation.v1",
        "trace_id": rows[0]["trace_id"],
        "physical_hbm_bytes": rows[0]["physical_hbm_bytes"],
        "blocking_async_speedup_over_blocking_serial": blocking_serial / blocking_async,
        "pingpong_serial_speedup_over_blocking_serial": blocking_serial / pingpong_serial,
        "pingpong_incremental_speedup_at_scoreboard": blocking_async / pingpong_async,
        "combined_speedup": blocking_serial / pingpong_async,
        "gates": {
            "same_trace": True,
            "same_physical_hbm_bytes": True,
            "functional_rel_rms_below_0_01": all(row["functional_rel_rms"] < 0.01 for row in rows),
            "serialized_scoreboard_matches_serial": True,
        },
        "interpretation": (
            "The incremental ping-pong comparison is blocking_scoreboard / "
            "pingpong_scoreboard. Additive stage cycles overlap in scoreboard mode "
            "and are not a critical-path decomposition."
        ),
    }
    return rows, summary


def write_outputs(root: Path, output_dir: Path) -> None:
    rows, summary = summarize(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "sram_pingpong_ablation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "sram_pingpong_ablation.json").write_text(
        json.dumps({**summary, "variants": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    by_name = {row["variant"]: row for row in rows}
    lines = [
        "# Matrix SRAM Blocking / Ping-Pong 消融",
        "",
        "| 面板组织 | Timing 模式 | Cycles | HBM bytes | rel-RMS | 数据等待 cycles | DMA 等待 cycles |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['panel_mode']} | {row['timing_model']}"
            f"{' (forced serial)' if row['scoreboard_serialize'] else ''} | "
            f"{row['cycles']:,} | {row['physical_hbm_bytes']:,} | "
            f"{row['functional_rel_rms']:.6f} | "
            f"{row.get('data_stall_cycles', 0):,} | {row.get('dma_wait_cycles', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 仅打开通用异步 scoreboard：`{summary['blocking_async_speedup_over_blocking_serial']:.3f}x`。",
            f"- 只改变 ping-pong 指令顺序、仍串行：`{summary['pingpong_serial_speedup_over_blocking_serial']:.3f}x`。",
            f"- 在相同异步引擎下，ping-pong 的独立额外收益：`{summary['pingpong_incremental_speedup_at_scoreboard']:.3f}x`。",
            f"- 从 blocking serial 到 ping-pong async 的组合收益：`{summary['combined_speedup']:.3f}x`。",
            "- 六种模式的 HBM 物理字节完全相同，因此这里测的是驻留/重叠，不是少搬权重。",
            "- Scoreboard 模式的 stage elapsed 区间会互相重叠；其和大于 makespan 是重叠证据，不是可闭合的 critical-path stack。",
            "",
            "## 边界",
            "",
            "这是一个真实 Qwen SWE 路由切片、缩小 H/I 的非零功能与时序消融。它验证单 context ping-pong，尚未验证多 context 弹性 SRAM；绝对周期仍待 RTL 校准。",
            "",
        ]
    )
    (output_dir / "SRAM_PINGPONG_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_outputs(args.root.resolve(), args.output_dir.resolve())
    print(args.output_dir.resolve() / "SRAM_PINGPONG_ABLATION_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
