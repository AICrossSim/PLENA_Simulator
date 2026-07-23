#!/usr/bin/env python3
"""Export Window 1 P2 pilot trace replay results to CSV/JSON tables."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.moe_timing.replay.utils import summarize_run
from transactional_emulator.testbench.moe_timing.qwen.utils import OUT_ROOT, load_json, write_csv, write_json


MEASUREMENT_NOTE = "self-consistent upper bound, absolute accuracy pending RTL (Window 2)"
ROUTING_STAGE_NAMES = {
    "accumulator_init",
    "gather",
    "expert_weight_address",
    "expert_route_weight",
    "scatter_combine",
}


def _result_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*") if (path / "qwen3_trace_replay_results.json").exists())


def _sum_stages(profile: dict[str, Any], names: set[str]) -> int:
    total = 0
    stages = profile.get("stages", {})
    if not isinstance(stages, dict):
        return 0
    for name in names:
        stage = stages.get(name, {})
        total += int(stage.get("wall_cycles") or 0)
    return total


def _stage_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = profile.get("stages", {})
    return stages if isinstance(stages, dict) else {}


def _mean(values: list[int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def export(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    runs = []
    stage_rows = []
    tax_rows = []
    bytes_rows = []
    skipped = []
    for build_dir in _result_dirs(root):
        result = load_json(build_dir / "qwen3_trace_replay_results.json")
        trace = load_json(build_dir / "trace.json")
        run_row, stack = summarize_run(result["trace_id"], build_dir)
        profile = load_json(Path(run_row["stage_profile_path"])) if run_row.get("stage_profile_path") else {}
        routing_tax_cycles = _sum_stages(profile, ROUTING_STAGE_NAMES)
        sim_cycles = run_row.get("sim_latency_cycles")
        if sim_cycles is None:
            skipped.append({"build_dir": str(build_dir), "reason": "missing sim_latency_cycles"})
            continue
        routed_fraction = float(routing_tax_cycles) / float(sim_cycles) if int(sim_cycles) else None
        common = {
            "trace_id": result["trace_id"],
            "benchmark": result["benchmark"],
            "model": trace["model"]["name"],
            "sample_id": result["sample_id"],
            "sample_index": trace["workload"].get("sample_index"),
            "category": trace["workload"].get("category"),
            "layer": result["layer"],
            "phase": result["phase"],
            "decode_step": trace["workload"].get("decode_step"),
            "token_count": result["rows"],
            "pair_count": result["pair_count"],
            "selected_expert_count": result["selected_expert_count"],
            "routing_source": trace["routing"].get("source"),
            "route_weight_source": trace["routing"].get("weight_source"),
            "issue_model": "in_order_blocking_no_overlap",
            "measurement_note": MEASUREMENT_NOTE,
        }
        runs.append(
            {
                **common,
                "sim_latency_cycles": int(sim_cycles),
                "hbm_bytes_read": run_row.get("hbm_bytes_read"),
                "hbm_bytes_written": run_row.get("hbm_bytes_written"),
                "functional_gate": run_row.get("functional_gate_passed"),
                "cycle_accounting_status": run_row.get("cycle_accounting_status"),
                "physical_byte_status": run_row.get("physical_byte_status"),
                "stage_total_simulation_cycles": run_row.get("stage_total_simulation_cycles"),
                "stage_total_wall_cycles": run_row.get("stage_total_wall_cycles"),
                "build_dir": str(build_dir),
                "stage_profile_path": run_row.get("stage_profile_path"),
                "run_stats_path": run_row.get("run_stats_path"),
            }
        )
        for row in stack:
            stage_rows.append({**common, **{k: v for k, v in row.items() if k != "run_id"}})
        tax_rows.append(
            {
                **common,
                "routing_tax_method": "stage_derived_accumulator_gather_address_routeweight_scatter",
                "routing_tax_cycles": routing_tax_cycles,
                "routing_tax_fraction_of_total": routed_fraction,
                "total_cycles": int(sim_cycles),
                "note": "Fixed-route trace replay excludes router GEMM/topk; routing tax here is stage-derived movement/dispatch overhead, not device-selected-minus-oracle.",
            }
        )
        logical = trace.get("logical_bytes", {})
        profile_read = profile.get("total_hbm_bytes_read")
        profile_written = profile.get("total_hbm_bytes_written")
        bytes_rows.append(
            {
                **common,
                "logical_input_hidden_bf16_bytes": logical.get("input_hidden_bf16_bytes"),
                "logical_routed_hidden_bf16_bytes": logical.get("routed_hidden_bf16_bytes"),
                "logical_route_weight_bf16_bytes": logical.get("route_weight_bf16_bytes"),
                "physical_hbm_bytes_read_run_stats": run_row.get("hbm_bytes_read"),
                "physical_hbm_bytes_written_run_stats": run_row.get("hbm_bytes_written"),
                "physical_hbm_bytes_read_stage_profile": profile_read,
                "physical_hbm_bytes_written_stage_profile": profile_written,
                "physical_read_matches": run_row.get("hbm_bytes_read") == profile_read,
                "physical_written_matches": run_row.get("hbm_bytes_written") == profile_written,
                "physical_source": "rust_emulator WithStats 64B transfer counters",
            }
        )

    summary_rows = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in runs:
        groups.setdefault((row["benchmark"], row["phase"]), []).append(row)
    for (benchmark, phase), rows in sorted(groups.items()):
        cycles = [int(row["sim_latency_cycles"]) for row in rows]
        taxes = [
            int(tax["routing_tax_cycles"]) for tax in tax_rows for row in rows if tax["trace_id"] == row["trace_id"]
        ]
        summary_rows.append(
            {
                "benchmark": benchmark,
                "phase": phase,
                "runs": len(rows),
                "cycles_min": min(cycles) if cycles else None,
                "cycles_mean": _mean(cycles),
                "cycles_max": max(cycles) if cycles else None,
                "routing_tax_cycles_min": min(taxes) if taxes else None,
                "routing_tax_cycles_mean": _mean(taxes),
                "routing_tax_cycles_max": max(taxes) if taxes else None,
                "measurement_note": MEASUREMENT_NOTE,
            }
        )

    out_prefix = args.out_prefix
    write_csv(out_prefix.with_name(out_prefix.name + "_timing.csv"), runs)
    write_csv(out_prefix.with_name(out_prefix.name + "_stage_stack.csv"), stage_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_routing_tax.csv"), tax_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_bytes_validation.csv"), bytes_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_summary_stats.csv"), summary_rows)
    payload = {
        "schema_version": 1,
        "root": str(root),
        "measurement_note": MEASUREMENT_NOTE,
        "runs": runs,
        "stage_stack": stage_rows,
        "routing_tax": tax_rows,
        "bytes_validation": bytes_rows,
        "summary_stats": summary_rows,
        "skipped": skipped,
    }
    write_json(args.out_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT_ROOT / "trace_replay")
    parser.add_argument("--out-prefix", type=Path, default=OUT_ROOT / "gpqa_bfcl_emulation")
    parser.add_argument("--out-json", type=Path, default=OUT_ROOT / "gpqa_bfcl_emulation_timing_summary.json")
    args = parser.parse_args()
    payload = export(args)
    print(f"exported {len(payload['runs'])} runs, {len(payload['stage_stack'])} stage rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
