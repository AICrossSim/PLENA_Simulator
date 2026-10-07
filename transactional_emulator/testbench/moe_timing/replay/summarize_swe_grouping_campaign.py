#!/usr/bin/env python3
"""Summarize a completed pair-major versus expert-grouped SWE campaign."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def pct(delta: int, baseline: int) -> float:
    return 100.0 * delta / baseline if baseline else 0.0


def read_stage_rows(result_path: Path, mode: str, model: str, batch: int) -> list[dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    profile = json.loads(Path(result["run_metrics"]["stage_profile_path"]).read_text(encoding="utf-8"))
    total = int(profile["total_simulation_cycles"])
    rows = []
    for stage, values in profile["stages"].items():
        cycles = int(values["wall_cycles"])
        rows.append(
            {
                "model": model,
                "batch": batch,
                "execution_mode": mode,
                "stage": stage,
                "wall_cycles": cycles,
                "total_cycles": total,
                "wall_cycle_pct": 100.0 * cycles / total if total else 0.0,
                "physical_hbm_bytes_read": int(values["physical_hbm_bytes_read"]),
                "matrix_proxy_cycles": int(values["resource_proxy_cycles"]["matrix"]),
                "vector_proxy_cycles": int(values["resource_proxy_cycles"]["vector"]),
                "scalar_proxy_cycles": int(values["resource_proxy_cycles"]["scalar"]),
                "dma_proxy_cycles": int(values["resource_proxy_cycles"]["dma"]),
            }
        )
    if sum(row["wall_cycles"] for row in rows) != total:
        raise ValueError(f"{result_path}: stage cycles do not close")
    return rows


def summarize_state(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in state["cases"]:
        grouped[(case["model"], int(case["batch"]))][case["execution_order"]] = case

    summary: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for (model, batch), modes in sorted(grouped.items()):
        if set(modes) != {"pair_major", "expert_major"}:
            raise ValueError(f"{model} B{batch}: incomplete mode pair {sorted(modes)}")
        baseline = modes["pair_major"]
        optimized = modes["expert_major"]
        if baseline["trace_id"] != optimized["trace_id"]:
            raise ValueError(f"{model} B{batch}: modes use different traces")
        pair_cycles = int(baseline["cycles"])
        grouped_cycles = int(optimized["cycles"])
        pair_bytes = int(baseline["physical_hbm_bytes"])
        grouped_bytes = int(optimized["physical_hbm_bytes"])
        summary.append(
            {
                "model": model,
                "workload": "SWE-bench full-test prompt route population",
                "timed_scope": "one exact representative decode layer-step route slice",
                "batch": batch,
                "layer": baseline["layer"],
                "step": baseline.get("step"),
                "trace_id": baseline["trace_id"],
                "route_pairs": baseline["route_pairs"],
                "unique_routed_experts": baseline["active_experts"],
                "pair_major_cycles": pair_cycles,
                "expert_grouped_cycles": grouped_cycles,
                "cycle_speedup": pair_cycles / grouped_cycles,
                "cycle_reduction_pct": pct(pair_cycles - grouped_cycles, pair_cycles),
                "pair_major_physical_hbm_bytes": pair_bytes,
                "expert_grouped_physical_hbm_bytes": grouped_bytes,
                "physical_hbm_byte_reduction_pct": pct(pair_bytes - grouped_bytes, pair_bytes),
                "functional_gate": baseline["functional_gate_kind"],
                "byte_gate": "exact",
                "cycle_accounting_gate": "closed",
            }
        )
        stages.extend(read_stage_rows(Path(baseline["result_path"]), "pair_major", model, batch))
        stages.extend(read_stage_rows(Path(optimized["result_path"]), "expert_grouped", model, batch))
    return summary, stages


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    lines = [
        "# SWE-bench Expert Grouping Compiler + Rust/Ramulator Results",
        "",
        "## Scope",
        "",
        "- Route source: the saved full-test SWE-bench prompt population (2,294 samples), with 16 saved decode tokens per sample.",
        "- Each cycle point replays one deterministic exact layer-step route slice selected from that population; it is not a full request or full agent trajectory.",
        "- Timing values are Rust simulation cycles with Ramulator-backed 64-B requests. Absolute cycles remain pending RTL calibration.",
        "- Timing uses zero-valued tensors because values do not change the instruction/HBM schedule. Nonzero random functional gates are reported separately.",
        "",
        "## Results",
        "",
        "| Model | Batch | Pair-major cycles | Expert-grouped cycles | Cycle speedup | HBM bytes reduced |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['batch']} | {row['pair_major_cycles']:,} | "
            f"{row['expert_grouped_cycles']:,} | {row['cycle_speedup']:.3f}x | "
            f"{row['physical_hbm_byte_reduction_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This ablation measures same-expert grouping plus one-load-per-expert weight reuse on the existing single matrix engine. It does not measure the proposed multi-context SRAM or multi-core execution. Therefore these speedups are the compiler/data-layout contribution, not the final architecture speedup.",
            "",
            "## Campaign Metadata",
            "",
            f"- Cases: {len(state['cases'])}",
            f"- Failures: {len(state['failures'])}",
            f"- Timing truth label: {state['timing_truth']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=16)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["failures"]:
        raise ValueError(f"campaign contains {len(state['failures'])} failures")
    if len(state["cases"]) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, found {len(state['cases'])}")
    summary, stages = summarize_state(state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "representative_cycle_results.csv", summary)
    write_csv(args.output_dir / "representative_stage_stack.csv", stages)
    write_report(args.output_dir / "REPRESENTATIVE_CYCLE_REPORT.md", summary, state)
    print(json.dumps({"summary_rows": len(summary), "stage_rows": len(stages)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
