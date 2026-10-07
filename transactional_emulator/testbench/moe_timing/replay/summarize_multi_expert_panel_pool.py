#!/usr/bin/env python3
"""Summarize fixed-capacity single- and multi-expert Matrix-SRAM buffering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RESULT_NAME = "moe_trace_replay_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sram-sweep-root", type=Path, required=True)
    parser.add_argument("--panel-pool-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    functional = document["functional_gate"]
    repeat = document["repeat_gate"]
    byte_gate = document["hbm_weight_byte_gate"]
    if functional.get("gate_kind") != "nonzero_reference" or not functional.get("passed"):
        raise ValueError(f"{path}: nonzero functional gate did not pass")
    if not repeat.get("passed") or int(repeat.get("repeats", 0)) != 3:
        raise ValueError(f"{path}: exact three-repeat gate did not pass")
    if not byte_gate.get("passed"):
        raise ValueError(f"{path}: HBM byte gate did not pass")
    if document.get("execution_order") != "expert_major":
        raise ValueError(f"{path}: expected expert-major execution")
    if document.get("hbm_weight_layout") != "tile_major":
        raise ValueError(f"{path}: expected tile-major HBM weight layout")
    if int(document.get("mram_tile_capacity", 0)) != 64:
        raise ValueError(f"{path}: expected fixed 64-tile (512 KiB) Matrix SRAM")
    return document


def model_key(document: dict[str, Any]) -> str:
    name = str(document["model_name"])
    if "Qwen" in name:
        return "qwen"
    if "DeepSeek" in name:
        return "deepseek"
    raise ValueError(f"unsupported model {name!r}")


def row_for(
    document: dict[str, Any],
    *,
    organization: str,
    slots: int,
    source: Path,
    blocking_cycles: int,
    pingpong_cycles: int,
) -> dict[str, Any]:
    metrics = document["run_metrics"]
    cycles = int(metrics["sim_latency_cycles"])
    return {
        "model": document["model_name"],
        "organization": organization,
        "resident_panel_slots": slots,
        "matrix_sram_kib": 512,
        "simulation_cycles": cycles,
        "speedup_vs_one_blocking_panel": blocking_cycles / cycles,
        "speedup_vs_one_expert_pingpong": pingpong_cycles / cycles,
        "physical_hbm_bytes": int(metrics["hbm_bytes_read"]),
        "functional_rel_rms": float(document["functional_gate"]["rel_rms"]),
        "functional_pass": bool(document["functional_gate"]["passed"]),
        "repeat_3_exact": bool(document["repeat_gate"]["passed"]),
        "source": str(source),
    }


def collect_rows(sram_root: Path, pool_root: Path) -> list[dict[str, Any]]:
    paths = {
        "qwen": {
            "blocking": sram_root / "h512_depth1" / RESULT_NAME,
            "pingpong": sram_root / "h512_depth2" / RESULT_NAME,
        },
        "deepseek": {
            "blocking": sram_root / "deep_h512_depth1" / RESULT_NAME,
            "pingpong": sram_root / "deep_h512_depth2" / RESULT_NAME,
        },
    }
    rows: list[dict[str, Any]] = []
    for key, model_paths in paths.items():
        blocking = load_result(model_paths["blocking"])
        pingpong = load_result(model_paths["pingpong"])
        if model_key(blocking) != key or model_key(pingpong) != key:
            raise ValueError(f"{key}: baseline result model mismatch")
        identity_fields = (
            "trace_id",
            "hidden",
            "intermediate",
            "shared_intermediate",
            "pair_count",
            "selected_experts",
        )
        if any(blocking[field] != pingpong[field] for field in identity_fields):
            raise ValueError(f"{key}: blocking and ping-pong workloads differ")
        blocking_cycles = int(blocking["run_metrics"]["sim_latency_cycles"])
        pingpong_cycles = int(pingpong["run_metrics"]["sim_latency_cycles"])
        expected_bytes = int(blocking["run_metrics"]["hbm_bytes_read"])
        if int(pingpong["run_metrics"]["hbm_bytes_read"]) != expected_bytes:
            raise ValueError(f"{key}: baseline HBM bytes differ")
        rows.append(
            row_for(
                blocking,
                organization="one_expert_blocking_panel",
                slots=1,
                source=model_paths["blocking"],
                blocking_cycles=blocking_cycles,
                pingpong_cycles=pingpong_cycles,
            )
        )
        rows.append(
            row_for(
                pingpong,
                organization="one_expert_pingpong",
                slots=2,
                source=model_paths["pingpong"],
                blocking_cycles=blocking_cycles,
                pingpong_cycles=pingpong_cycles,
            )
        )
        for slots in (2, 4, 8):
            path = pool_root / f"{key}_h512_slots{slots}" / RESULT_NAME
            pooled = load_result(path)
            if model_key(pooled) != key:
                raise ValueError(f"{path}: pooled result model mismatch")
            if any(blocking[field] != pooled[field] for field in identity_fields):
                raise ValueError(f"{path}: pooled workload differs from its baseline")
            if int(pooled.get("multi_expert_panel_slots", 0)) != slots:
                raise ValueError(f"{path}: panel slot count mismatch")
            if int(pooled["run_metrics"]["hbm_bytes_read"]) != expected_bytes:
                raise ValueError(f"{path}: HBM bytes differ from its baseline")
            rows.append(
                row_for(
                    pooled,
                    organization="multi_expert_panel_pool",
                    slots=slots,
                    source=path,
                    blocking_cycles=blocking_cycles,
                    pingpong_cycles=pingpong_cycles,
                )
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "multi_expert_panel_pool.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "multi_expert_panel_pool.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Multi-Expert Matrix-SRAM Panel-Pool Result",
        "",
        "All rows use the same real SWE decode route per model, exact nonzero MX weights, "
        "random nonzero activations, a fixed 512 KiB Matrix SRAM, 64B Ramulator requests, "
        "scoreboard timing, byte-exact checks, and three exactly repeatable runs.",
        "",
        "| Model | SRAM organization | Slots | Cycles | vs blocking | vs one-expert ping-pong | HBM bytes | rel-RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {organization} | {resident_panel_slots} | {simulation_cycles:,} | "
            "{speedup_vs_one_blocking_panel:.6f}x | {speedup_vs_one_expert_pingpong:.6f}x | "
            "{physical_hbm_bytes:,} | {functional_rel_rms:.6f} |".format(**row)
        )
    best_extra = max(
        row["speedup_vs_one_expert_pingpong"]
        for row in rows
        if row["organization"] == "multi_expert_panel_pool"
    )
    decision = (
        "reject a generic deeper pool as a primary contribution"
        if best_extra < 1.01
        else "retain the pool for broader validation"
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"The best incremental gain over ordinary two-panel ping-pong is {best_extra:.6f}x; "
            f"therefore the current evidence says to **{decision}**.",
            "",
            "The experiment isolates SRAM placement/lookahead only. It keeps one matrix engine, "
            "does not add HBM bandwidth, and does not model physical SRAM bank/port conflicts. "
            "Absolute cycles still require RTL primitive calibration.",
            "",
        ]
    )
    (output / "MULTI_EXPERT_PANEL_POOL_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.sram_sweep_root, args.panel_pool_root)
    write_outputs(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
