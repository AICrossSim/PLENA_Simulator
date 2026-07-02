#!/usr/bin/env python3
"""Export Window 1 P1 MoE timing runs to CSV/JSON tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import (
    DEFAULT_OUT_ROOT,
    load_json,
    summarize_run,
    write_csv,
    write_json,
)


def _default_runs(root: Path) -> list[tuple[str, Path]]:
    runs = [
        ("fixed_route_direct", root / "gpt_oss_layer0_tok1_fixed_route"),
        ("device_selected_direct", root / "gpt_oss_layer0_tok1_device_selected"),
        ("trace_replay", root / "trace_replay" / "gpt_oss_layer0_tok1_decode"),
    ]
    return [(name, path) for name, path in runs if (path / "rust_emulator_run_stats.json").exists()]


def export_results(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    run_specs = _default_runs(root)
    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for run_id, build_dir in run_specs:
        row, stack = summarize_run(run_id, build_dir)
        rows.append(row)
        stage_rows.extend(stack)

    by_id = {row["run_id"]: row for row in rows}
    tax_rows = []
    if "fixed_route_direct" in by_id and "device_selected_direct" in by_id:
        fixed = by_id["fixed_route_direct"]
        device = by_id["device_selected_direct"]
        tax_rows.append(
            {
                "tax_name": "device_selected_minus_fixed_route",
                "base_run_id": "fixed_route_direct",
                "routed_run_id": "device_selected_direct",
                "routing_tax_cycles": int(device["sim_latency_cycles"]) - int(fixed["sim_latency_cycles"]),
                "routing_tax_hbm_bytes_read": int(device["hbm_bytes_read"]) - int(fixed["hbm_bytes_read"]),
                "routing_tax_hbm_bytes_written": int(device["hbm_bytes_written"]) - int(fixed["hbm_bytes_written"]),
                "device_topk_in_emulator": True,
            }
        )

    trace_path = root / "route_traces" / "gpt_oss_layer0_tok1_trace.json"
    bytes_rows: list[dict[str, Any]] = []
    if trace_path.exists():
        trace = load_json(trace_path)
        logical = trace.get("logical_bytes", {})
        for row in rows:
            bytes_rows.append(
                {
                    "run_id": row["run_id"],
                    "logical_input_hidden_bf16_bytes": logical.get("input_hidden_bf16_bytes"),
                    "logical_routed_hidden_bf16_bytes": logical.get("routed_hidden_bf16_bytes"),
                    "physical_hbm_bytes_read": row.get("hbm_bytes_read"),
                    "physical_hbm_bytes_written": row.get("hbm_bytes_written"),
                    "physical_source": "rust_emulator WithStats 64B transfer counters",
                }
            )

    payload = {
        "schema_version": 1,
        "root": str(root),
        "runs": rows,
        "routing_tax": tax_rows,
        "stage_stack": stage_rows,
        "bytes_validation": bytes_rows,
    }
    write_json(args.out_json, payload)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_timing.csv"), rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_routing_tax.csv"), tax_rows)
    write_csv(args.out_prefix.parent / "gpqa_bfcl_routing_tax.csv", tax_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_stage_stack.csv"), stage_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_bytes_validation.csv"), bytes_rows)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_ROOT / "gpqa_bfcl_emulation_timing_summary.json")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_ROOT / "gpqa_bfcl_emulation")
    args = parser.parse_args()
    payload = export_results(args)
    print(f"exported {len(payload['runs'])} runs and {len(payload['stage_stack'])} stage rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
