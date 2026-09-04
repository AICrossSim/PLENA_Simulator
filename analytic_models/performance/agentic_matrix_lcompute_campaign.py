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
from .nemotron3_workload import InferencePhase, Precision


WEIGHT_PRECISION_SCENARIOS: tuple[tuple[str, Precision | None, str], ...] = (
    (
        "checkpoint_mixed_nvfp4_bf16",
        None,
        "official NVFP4 checkpoint policy with measured BF16 exclusion modules",
    ),
    (
        "uniform_mxfp8",
        Precision.MX8,
        "PLENA MX8/MXFP8 logical storage: one byte per value plus one scale per 128 values",
    ),
    ("uniform_bf16", Precision.BF16, "uniform two-byte BF16 weight storage"),
)


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


def _timeline_endpoints(record: dict[str, Any]) -> dict[str, Any]:
    """Return a legal serial endpoint and an intentionally optimistic lower bound.

    L-Compute reuses Vector arithmetic, so its cycles share the Vector resource.
    The ideal endpoint allows HBM, Matrix and the combined Vector/L-Compute path
    to overlap completely. It is not an executable schedule and gives no credit
    for dependencies, SRAM capacity or arbitration.
    """

    resources = {
        "hbm": int(record["hbm_cycles"]),
        "matrix": int(record["matrix_cycles"]),
        "vector_plus_lcompute": int(record["vector_cycles"]) + int(record["lcompute_cycles"]),
    }
    serial = sum(resources.values())
    if serial != int(record["cycles"]):
        raise AssertionError("strict-serial timeline no longer equals the resource sum")
    return {
        "strict_serial_cycles": serial,
        "ideal_resource_overlap_lower_bound_cycles": max(resources.values()),
        "resource_cycles": resources,
    }


def _variant_records(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {str(record["variant"]): record for record in result["records"]}
    expected = {str(variant) for variant in MatrixVariant}
    if set(records) != expected:
        raise AssertionError(f"ablation variant coverage differs: {set(records)} != {expected}")
    return records


def _precision_result(result: dict[str, Any], description: str) -> dict[str, Any]:
    variants = _variant_records(result)
    baseline = variants[str(MatrixVariant.B_ARLO)]
    phased = variants[str(MatrixVariant.D_AFFINE)]
    baseline_timeline = _timeline_endpoints(baseline)
    phased_timeline = _timeline_endpoints(phased)
    return {
        "description": description,
        "weight_precision": str(result["weight_precision"]),
        "weight_precision_policy": result["weight_precision_policy"],
        "B_timeline": baseline_timeline,
        "D_timeline": phased_timeline,
        "D_speedup_vs_B_strict_serial": (
            baseline_timeline["strict_serial_cycles"] / phased_timeline["strict_serial_cycles"]
        ),
        "D_speedup_vs_B_ideal_resource_overlap": (
            baseline_timeline["ideal_resource_overlap_lower_bound_cycles"]
            / phased_timeline["ideal_resource_overlap_lower_bound_cycles"]
        ),
        "D_logical_weight_read_bytes": int(phased["logical_weight_read_bytes"]),
        "D_physical_hbm_read_bytes": int(phased["physical_hbm_read_bytes"]),
        "D_physical_hbm_write_bytes": int(phased["physical_hbm_write_bytes"]),
    }


def _d_prime_bank_control(physical: dict[str, Any], phased: dict[str, Any]) -> dict[str, Any]:
    control = physical["fixed_phased_bank_control"]["nemotron3"]
    if (
        int(control["bank_stall_cycles"]) != 0
        or float(control["compact_phase_vs_explicit_bases_bank_speedup"]) != 1.0
        or not bool(control["same_physical_coordinates_as_compact_tile_phase"])
        or int(phased["bank_stall_cycles"]) != 0
    ):
        raise AssertionError("D' is no longer a conflict-free fixed-wiring control for D")
    return {
        "scope": "bank service only; this is not a full-model timeline variant",
        "source": "recomputed by build_physical_evidence from official-shape packet coordinates",
        "mapping": control["mapping"],
        "packet_service_cycles": int(control["service_cycles"]),
        "packet_ideal_cycles": int(control["ideal_cycles"]),
        "packet_bank_stall_cycles": int(control["bank_stall_cycles"]),
        "roundtrip_values_checked": int(control["roundtrip_values_checked"]),
        "workload_matrix_service_cycles": int(phased["matrix_sram_service_cycles"]),
        "workload_bank_stall_cycles": 0,
        "D_vs_D_prime_pure_bank_speedup": 1.0,
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
    precision_runs: dict[str, tuple[dict[str, Any], str]] = {}
    for name, weight_precision, description in WEIGHT_PRECISION_SCENARIOS:
        precision_runs[name] = (
            run_ablation(
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
                strict_profile=True,
                weight_precision=weight_precision,
            ),
            description,
        )
    result = precision_runs["checkpoint_mixed_nvfp4_bf16"][0]
    variants = _variant_records(result)
    phased = variants[str(MatrixVariant.D_AFFINE)]
    timeline = {name: _timeline_endpoints(record) for name, record in variants.items()}
    ideal_d = timeline[str(MatrixVariant.D_AFFINE)]["ideal_resource_overlap_lower_bound_cycles"]
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
            "routing_mode": "strict measured eager route union; fallback is forbidden",
            "cycles": {name: record["cycles"] for name, record in variants.items()},
            "timeline_endpoints": timeline,
            "bank_stall_cycles": {name: record["bank_stall_cycles"] for name, record in variants.items()},
            "speedup_D_vs_A": phased["speedup_vs_A"],
            "speedup_D_vs_B": phased["speedup_vs_B"],
            "speedup_D_vs_C": phased["speedup_vs_C_fixed"],
            "speedup_D_vs_B_ideal_resource_overlap": (
                timeline[str(MatrixVariant.B_ARLO)]["ideal_resource_overlap_lower_bound_cycles"] / ideal_d
            ),
            "D_tpot_ms_proxy": phased["cycles"] / decode_steps / (hardware.clock_hz / 1000),
            "D_tpot_ms_ideal_resource_overlap_lower_bound": (ideal_d / decode_steps / (hardware.clock_hz / 1000)),
            "D_aggregate_throughput_tokens_s_proxy": (
                group.batch_size * decode_steps * hardware.clock_hz / phased["cycles"]
            ),
            "D_aggregate_throughput_tokens_s_ideal_resource_overlap_upper_bound": (
                group.batch_size * decode_steps * hardware.clock_hz / ideal_d
            ),
            "D_prime_bank_control": _d_prime_bank_control(physical, phased),
            "weight_precision_sensitivity": {
                name: _precision_result(precision_result, description)
                for name, (precision_result, description) in precision_runs.items()
            },
        },
    }


def _flatten_group(row: dict[str, Any]) -> dict[str, Any]:
    gpu = row["gpu"]
    route = row["route"]
    plena = row["plena"]
    cycles = plena["cycles"]
    stalls = plena["bank_stall_cycles"]
    timeline = plena["timeline_endpoints"]
    result = {
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
        "D_phased_cycles": cycles[str(MatrixVariant.D_AFFINE)],
        "E_overlap_cycles": cycles[str(MatrixVariant.E_AFFINE_OVERLAP)],
        "D_speedup_vs_A": plena["speedup_D_vs_A"],
        "D_speedup_vs_B": plena["speedup_D_vs_B"],
        "D_speedup_vs_B_ideal_resource_overlap": plena["speedup_D_vs_B_ideal_resource_overlap"],
        "D_speedup_vs_C": plena["speedup_D_vs_C"],
        "C_bank_stall_cycles": stalls[str(MatrixVariant.C_FIXED)],
        "D_bank_stall_cycles": stalls[str(MatrixVariant.D_AFFINE)],
        "D_tpot_ms_proxy": plena["D_tpot_ms_proxy"],
        "D_tpot_ms_ideal_resource_overlap_lower_bound": plena["D_tpot_ms_ideal_resource_overlap_lower_bound"],
        "D_aggregate_throughput_tokens_s_proxy": plena["D_aggregate_throughput_tokens_s_proxy"],
        "D_aggregate_throughput_tokens_s_ideal_resource_overlap_upper_bound": plena[
            "D_aggregate_throughput_tokens_s_ideal_resource_overlap_upper_bound"
        ],
        "B_ideal_resource_overlap_cycles": timeline[str(MatrixVariant.B_ARLO)][
            "ideal_resource_overlap_lower_bound_cycles"
        ],
        "D_ideal_resource_overlap_cycles": timeline[str(MatrixVariant.D_AFFINE)][
            "ideal_resource_overlap_lower_bound_cycles"
        ],
        "D_prime_workload_bank_stall_cycles": plena["D_prime_bank_control"]["workload_bank_stall_cycles"],
        "D_vs_D_prime_pure_bank_speedup": plena["D_prime_bank_control"]["D_vs_D_prime_pure_bank_speedup"],
    }
    for name, _precision, _description in WEIGHT_PRECISION_SCENARIOS:
        sensitivity = plena["weight_precision_sensitivity"][name]
        result[f"{name}_D_speedup_vs_B_serial"] = sensitivity["D_speedup_vs_B_strict_serial"]
        result[f"{name}_D_speedup_vs_B_ideal_overlap"] = sensitivity["D_speedup_vs_B_ideal_resource_overlap"]
        result[f"{name}_D_logical_weight_read_bytes"] = sensitivity["D_logical_weight_read_bytes"]
        result[f"{name}_D_physical_hbm_read_bytes"] = sensitivity["D_physical_hbm_read_bytes"]
    return result


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
        "D_phased_cycles",
        "D_speedup_vs_A",
        "D_speedup_vs_B",
        "D_speedup_vs_B_ideal_resource_overlap",
        "D_speedup_vs_C",
        "D_tpot_ms_proxy",
        "D_tpot_ms_ideal_resource_overlap_lower_bound",
        "D_aggregate_throughput_tokens_s_proxy",
        "D_aggregate_throughput_tokens_s_ideal_resource_overlap_upper_bound",
        "B_ideal_resource_overlap_cycles",
        "D_ideal_resource_overlap_cycles",
        "D_vs_D_prime_pure_bank_speedup",
        *(
            f"{name}_{suffix}"
            for name, _precision, _description in WEIGHT_PRECISION_SCENARIOS
            for suffix in (
                "D_speedup_vs_B_serial",
                "D_speedup_vs_B_ideal_overlap",
                "D_logical_weight_read_bytes",
                "D_physical_hbm_read_bytes",
            )
        ),
    )
    summaries = []
    for (benchmark, batch_size), rows in sorted(buckets.items()):
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            "batch_size": batch_size,
            "group_count": len(rows),
            "statistical_unit": "length-sorted disjoint workload group",
            "p95_status": ("descriptive_at_least_20_groups" if len(rows) >= 20 else "exploratory_low_n"),
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
        "contract": "nemotron-agentic-matrix-lcompute-dse-v2",
        "source": campaign.to_summary(),
        "hardware": asdict(hardware),
        "decode_steps": decode_steps,
        "group_count": len(groups),
        "groups": groups,
        "summary": _summary_rows(groups),
        "routing_contract": {
            "mode": "strict",
            "fallback_allowed": False,
            "profile_source": "per-step unions of independent real-checkpoint B1 eager traces",
        },
        "timeline_contract": {
            "strict_serial": "HBM + Matrix + Vector + L-Compute; current dependency-safe reported timeline",
            "ideal_resource_overlap_lower_bound": (
                "max(HBM, Matrix, Vector + L-Compute); optimistic bound only, not an emitted or replayed schedule"
            ),
            "why_both": ("the two endpoints expose whether a conclusion depends on the current no-overlap assumption"),
        },
        "weight_precision_contract": {
            name: {
                "precision_override": str(precision) if precision is not None else None,
                "description": description,
            }
            for name, precision, description in WEIGHT_PRECISION_SCENARIOS
        },
        "D_prime_bank_control": {
            "scope": "official-shape packet-level bank-only control, recomputed in this run",
            "not_a_full_timeline_variant": True,
            "evidence": physical["fixed_phased_bank_control"]["nemotron3"],
        },
        "statistics_contract": {
            "unit": "length-sorted disjoint workload group",
            "all_rows": "all groups across BFCL v3, GPQA-Diamond and SWE-bench Verified",
            "p95_policy": "reported descriptively; rows with N < 20 are labelled exploratory_low_n",
        },
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
            "D_prime": ("D' proves the fixed-wiring bank floor only; no D' whole-model cycles are fabricated"),
            "precision": (
                "NVFP4/MX8/BF16 alter logical weight traffic; dequantization compute and physical scale "
                "padding are outside this model"
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
    source = campaign["source"]
    sample_count = int(source["sample_count"])
    full_identical = int(source["timing_and_routing_tokens_identical_samples"])
    replay_identical = int(source["timing_and_routing_replay_window_identical_samples"])
    used_events = int(source["routing_event_accounting"]["used_decode_events"])
    ignored_events = int(source["routing_event_accounting"]["ignored_decode_events"])
    readme = f"""# Nemotron Agentic Matrix L-Compute DSE

This directory is derived from the externally archived real-checkpoint B200
campaign. `group_results.csv` preserves every length-sorted workload group;
`summary.csv` reports N, medians and descriptive P95 values by benchmark and
batch size. P95 rows with fewer than 20 groups are explicitly exploratory.

GPU timing/energy columns are measurements. PLENA cycle/TPOT columns are
pre-RTL Compiler/Simulator estimates with symbolic weights. They are shown
side by side but must not be presented as a measured GPU speedup. Routing uses
the eager run's self-consistent token trace; the optimized timing run remains
baseline-only. Full continuations match for {full_identical}/{sample_count}
samples; the {campaign["decode_steps"]}-step replay window matches for
{replay_identical}/{sample_count}. Exactly {used_events} fully validated decode
events enter DSE and {ignored_events} later fully validated events are excluded.

Every group includes strict-serial and ideal-resource-overlap endpoints plus
checkpoint-mixed-NVFP4, uniform-MX8/MXFP8 and uniform-BF16 weight-traffic
sensitivity. D' is materialized only as the fair packet-level bank control; no
whole-model D' timing is claimed. The source archive contains prompt token IDs
and is intentionally not part of this directory.
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
