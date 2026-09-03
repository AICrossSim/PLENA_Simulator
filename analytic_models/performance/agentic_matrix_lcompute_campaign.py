"""Replay real Nemotron Agentic routing in the Matrix L-Compute DSE.

The GPU campaign supplies workload membership, eager router decisions and a
B200 latency/energy baseline.  PLENA cycles still come exclusively from the
Compiler/Simulator model.  The two clocks are reported side by side and are
never divided into a hardware speedup claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agentic_campaign import AgenticBatchGroup, AgenticCampaign, load_agentic_campaign
from .matrix_lcompute_campaign import (
    MatrixHardwarePoint,
    MatrixVariant,
    StateMode,
    attach_real_service_evidence,
    build_physical_evidence,
    load_compiler_evidence,
    run_ablation,
)
from .nemotron3_workload import InferencePhase


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty series")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _route_statistics(campaign: AgenticCampaign, group: AgenticBatchGroup, decode_steps: int) -> dict[str, float]:
    profile = campaign.routing_profile(group, decode_steps=decode_steps)
    counts = [count for step in profile.steps for _, count in step.unique_experts_by_layer]
    return {
        "active_experts_min": min(counts),
        "active_experts_median": statistics.median(counts),
        "active_experts_p95": _percentile([float(count) for count in counts], 0.95),
        "active_experts_max": max(counts),
    }


def _group_result(
    *,
    campaign: AgenticCampaign,
    group: AgenticBatchGroup,
    decode_steps: int,
    hardware: MatrixHardwarePoint,
    compiler_root: Path,
    compiler: dict[str, Any],
    physical: dict[str, Any],
) -> dict[str, Any]:
    profile = campaign.routing_profile(group, decode_steps=decode_steps)
    result = run_ablation(
        model="nemotron3",
        phase=InferencePhase.DECODE,
        batch_size=group.batch_size,
        tokens=decode_steps,
        context_length=group.padded_context_length,
        state_mode=StateMode.PLENA_BF16,
        hardware=hardware,
        compiler_root=compiler_root,
        compiler=compiler,
        physical=physical,
        profile=profile,
    )
    records = {str(record["variant"]): record for record in result["records"]}
    variants = {str(variant): records[str(variant)] for variant in MatrixVariant}
    affine = variants[str(MatrixVariant.D_AFFINE)]
    return {
        **group.to_dict(),
        "route": _route_statistics(campaign, group, decode_steps),
        "plena": {
            "scope": "formula timeline with Compiler-emitted recurrence and symbolic weights; not RTL or silicon",
            "clock_hz_assumption": hardware.clock_hz,
            "decode_steps": decode_steps,
            "context_model": (
                "rectangular static batch at the longest prompt; padding overhead is reported explicitly"
            ),
            "cycles": {name: record["cycles"] for name, record in variants.items()},
            "bank_stall_cycles": {name: record["bank_stall_cycles"] for name, record in variants.items()},
            "speedup_D_vs_A": affine["speedup_vs_A"],
            "speedup_D_vs_B": affine["speedup_vs_B"],
            "speedup_D_vs_C": affine["speedup_vs_C_fixed"],
            "D_tpot_ms_proxy": affine["cycles"] / decode_steps / (hardware.clock_hz / 1000),
            "D_aggregate_throughput_tokens_s_proxy": (
                group.batch_size * decode_steps * hardware.clock_hz / affine["cycles"]
            ),
        },
    }


def _flatten_group(row: dict[str, Any]) -> dict[str, Any]:
    gpu = row["gpu"]
    route = row["route"]
    plena = row["plena"]
    cycles = plena["cycles"]
    stalls = plena["bank_stall_cycles"]
    return {
        "key": row["key"],
        "benchmark": row["benchmark"],
        "batch_size": row["batch_size"],
        "group_index": row["group_index"],
        "sample_ids": "|".join(row["sample_ids"]),
        "prompt_length_min": min(row["prompt_lengths"]),
        "prompt_length_max": max(row["prompt_lengths"]),
        "padded_context_length": row["padded_context_length"],
        "padding_fraction": row["padding_fraction"],
        "active_experts_median": route["active_experts_median"],
        "active_experts_p95": route["active_experts_p95"],
        "active_experts_max": route["active_experts_max"],
        "gpu_ttft_ms_median": gpu["ttft_ms_median"],
        "gpu_itl_ms_median": gpu["itl_ms_median"],
        "gpu_e2e_ms_median": gpu["e2e_ms_median"],
        "gpu_batch_throughput_tokens_s_median": gpu["batch_throughput_tokens_s_median"],
        "gpu_batch_energy_joules_median": gpu["batch_energy_joules_median"],
        "A_original_cycles": cycles[str(MatrixVariant.A_ORIGINAL)],
        "B_arlo_cycles": cycles[str(MatrixVariant.B_ARLO)],
        "C_fixed_cycles": cycles[str(MatrixVariant.C_FIXED)],
        "D_affine_cycles": cycles[str(MatrixVariant.D_AFFINE)],
        "E_overlap_cycles": cycles[str(MatrixVariant.E_AFFINE_OVERLAP)],
        "D_speedup_vs_A": plena["speedup_D_vs_A"],
        "D_speedup_vs_B": plena["speedup_D_vs_B"],
        "D_speedup_vs_C": plena["speedup_D_vs_C"],
        "C_bank_stall_cycles": stalls[str(MatrixVariant.C_FIXED)],
        "D_bank_stall_cycles": stalls[str(MatrixVariant.D_AFFINE)],
        "D_tpot_ms_proxy": plena["D_tpot_ms_proxy"],
        "D_aggregate_throughput_tokens_s_proxy": plena["D_aggregate_throughput_tokens_s_proxy"],
    }


def _summary_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = [_flatten_group(group) for group in groups]
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        buckets[(str(row["benchmark"]), int(row["batch_size"]))].append(row)
        buckets[("all", int(row["batch_size"]))].append(row)
    metrics = (
        "padding_fraction",
        "active_experts_median",
        "active_experts_p95",
        "gpu_ttft_ms_median",
        "gpu_itl_ms_median",
        "gpu_e2e_ms_median",
        "gpu_batch_throughput_tokens_s_median",
        "gpu_batch_energy_joules_median",
        "A_original_cycles",
        "B_arlo_cycles",
        "C_fixed_cycles",
        "D_affine_cycles",
        "D_speedup_vs_A",
        "D_speedup_vs_B",
        "D_speedup_vs_C",
        "D_tpot_ms_proxy",
        "D_aggregate_throughput_tokens_s_proxy",
    )
    summaries = []
    for (benchmark, batch_size), rows in sorted(buckets.items()):
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            "batch_size": batch_size,
            "group_count": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_p95"] = _percentile(values, 0.95)
        summary["D_bank_stall_cycles_max"] = max(int(row["D_bank_stall_cycles"]) for row in rows)
        summaries.append(summary)
    return summaries


def build_agentic_matrix_lcompute_campaign(
    *,
    campaign_root: Path,
    compiler_root: Path,
    decode_steps: int = 32,
    benchmarks: tuple[str, ...] | None = None,
    batch_sizes: tuple[int, ...] | None = None,
    max_groups: int | None = None,
) -> dict[str, Any]:
    campaign = load_agentic_campaign(campaign_root)
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(compiler_root))
    physical = build_physical_evidence(hardware)
    attach_real_service_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    selected = [
        group
        for group in campaign.groups
        if (benchmarks is None or group.benchmark in benchmarks)
        and (batch_sizes is None or group.batch_size in batch_sizes)
    ]
    if max_groups is not None:
        selected = selected[:max_groups]
    if not selected:
        raise ValueError("agentic DSE selection is empty")
    groups = [
        _group_result(
            campaign=campaign,
            group=group,
            decode_steps=decode_steps,
            hardware=hardware,
            compiler_root=compiler_root,
            compiler=compiler,
            physical=physical,
        )
        for group in selected
    ]
    return {
        "status": "complete",
        "contract": "nemotron-agentic-matrix-lcompute-dse-v1",
        "source": campaign.to_summary(),
        "hardware": asdict(hardware),
        "decode_steps": decode_steps,
        "group_count": len(groups),
        "groups": groups,
        "summary": _summary_rows(groups),
        "claim_boundary": {
            "gpu": "real B200 NVFP4 timing and energy baseline",
            "routing": (
                "real B1 eager traces combined according to measured length-sorted batch membership; "
                "not direct batched routing capture"
            ),
            "plena": (
                "pre-RTL formula timeline with official dimensions, Compiler-emitted recurrent schedule "
                "and symbolic weights"
            ),
            "comparison": (
                "GPU milliseconds and PLENA proxy milliseconds are not a silicon speedup ratio; RTL frequency, "
                "area and power are not available"
            ),
        },
    }


def write_agentic_artifacts(campaign: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    flat = [_flatten_group(group) for group in campaign["groups"]]
    with (output_dir / "group_results.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    with (output_dir / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(campaign["summary"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(campaign["summary"])
    readme = """# Nemotron Agentic Matrix L-Compute DSE

This directory is derived from the externally archived real-checkpoint B200
campaign. `group_results.csv` preserves every length-sorted workload group;
`summary.csv` reports medians and P95 values by benchmark and batch size.

GPU timing/energy columns are measurements. PLENA cycle/TPOT columns are
pre-RTL Compiler/Simulator estimates with symbolic weights. They are shown
side by side but must not be presented as a measured GPU speedup. Routing uses
the eager run's self-consistent token trace; the optimized timing run remains
baseline-only because its generated continuation differs for 45/48 samples.
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--benchmark", action="append", dest="benchmarks")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    campaign = build_agentic_matrix_lcompute_campaign(
        campaign_root=args.campaign_root,
        compiler_root=args.compiler_root,
        decode_steps=args.decode_steps,
        benchmarks=tuple(args.benchmarks) if args.benchmarks else None,
        batch_sizes=tuple(args.batch_sizes) if args.batch_sizes else None,
        max_groups=args.max_groups,
    )
    write_agentic_artifacts(campaign, args.output_dir)
    print(json.dumps({"status": campaign["status"], "groups": campaign["group_count"]}, indent=2))


if __name__ == "__main__":
    main()
