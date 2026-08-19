"""Sweep KDA q/k/decay projection-buffer rotations for one lowered plan."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .b200_formal_campaign import build_report as build_formal_campaign_report
from .projection_scatter import (
    ScatterPlan,
    consumer_packets,
    verify_scatter_roundtrip,
)


def _rotated(plan: ScatterPlan, *, k_rotation: int, decay_rotation: int) -> ScatterPlan:
    rotations = {"k": k_rotation, "decay": decay_rotation}
    fields = tuple(
        replace(
            field,
            skew_kind=("field_constant" if rotations[field.name] else "none"),
            skew_stride=rotations[field.name],
        )
        if field.name in rotations
        else field
        for field in plan.fields
    )
    candidate = replace(plan, fields=fields, mapping_sha256="")
    candidate = replace(candidate, mapping_sha256=candidate.compute_mapping_sha256())
    candidate.validate()
    return candidate


def _metrics(
    plan: ScatterPlan,
    *,
    packets: tuple[Any, ...],
    fields: dict[str, Any],
    k_rotation: int,
    decay_rotation: int,
    write_metrics: dict[str, int | bool],
) -> dict[str, int | bool]:
    rotations = {"k": k_rotation, "decay": decay_rotation}

    def bank(field_name: str, group: int, local_row: int, lane: int) -> int:
        field = fields[field_name]
        local = local_row * field.local_lanes + lane
        if field_name in rotations:
            skew = rotations[field_name]
        elif field.skew_kind == "local_row_stride":
            skew = local_row * field.skew_stride
        elif field.skew_kind == "field_constant":
            skew = field.skew_stride
        elif field.skew_kind == "group_stride":
            skew = group * field.skew_stride
        elif field.skew_kind == "none":
            skew = 0
        else:
            raise ValueError(f"unknown projection skew {field.skew_kind!r}")
        return (local % plan.banks + skew) % plan.banks

    read_ideal = read_service = 0
    for packet in packets:
        counts = [0] * plan.banks
        for read in packet.reads:
            counts[bank(read.field, read.group, read.local_row, read.lane)] += 1
        read_ideal += math.ceil(len(packet.reads) / (plan.banks * plan.ports_per_bank))
        read_service += max(math.ceil(count / plan.ports_per_bank) for count in counts)

    read_ideal *= plan.valid_tokens
    read_service *= plan.valid_tokens
    write_ideal = int(write_metrics["write_ideal_cycles"])
    write_service = int(write_metrics["write_service_cycles"])
    return {
        "service_cycles": read_service,
        "ideal_cycles": read_ideal,
        "stall_cycles": read_service - read_ideal,
        "conflict_free": read_service == read_ideal,
        **write_metrics,
        "total_service_cycles": read_service + write_service,
        "total_ideal_cycles": read_ideal + write_ideal,
        "total_stall_cycles": read_service + write_service - read_ideal - write_ideal,
    }


def _write_metrics(plan: ScatterPlan) -> dict[str, int | bool]:
    """Count one physical producer pass.

    The swept k/decay fields each start and end on Matrix-burst boundaries, so
    a constant bank rotation cannot change their producer bank multiplicity.
    Counting writes once keeps the exhaustive 256-way read sweep suitable for
    normal regression while still rejecting a future unaligned field shape.
    """
    for name in ("k", "decay"):
        field = plan.field(name)
        field_values = plan.groups * field.values_per_group
        if field.source_offset % plan.producer_burst_values or field_values % plan.producer_burst_values:
            raise ValueError(f"{name} is not producer-burst aligned; write cost must be swept")

    source_banks = [0] * plan.source_values_per_token
    for field in plan.fields:
        for group in range(plan.groups):
            for local_row in range(field.local_rows):
                for lane in range(field.local_lanes):
                    source = plan.logical_source(field.name, group, local_row, lane)
                    source_banks[source] = plan.address(field.name, group, local_row, lane)[2]
    write_ideal = write_service = 0
    for start in range(0, len(source_banks), plan.producer_burst_values):
        packet = source_banks[start : start + plan.producer_burst_values]
        counts = [0] * plan.banks
        for target in packet:
            counts[target] += 1
        write_ideal += math.ceil(len(packet) / (plan.banks * plan.ports_per_bank))
        write_service += max(math.ceil(count / plan.ports_per_bank) for count in counts)

    write_ideal *= plan.valid_tokens
    write_service *= plan.valid_tokens
    return {
        "write_service_cycles": write_service,
        "write_ideal_cycles": write_ideal,
        "write_stall_cycles": write_service - write_ideal,
        "write_conflict_free": write_service == write_ideal,
    }


def _profile_context(campaign: dict[str, Any], scatter_reduction_percent: float) -> dict[str, Any]:
    reduction = scatter_reduction_percent / 100
    cases = []
    for case in campaign["kda"]["cases"]:
        exposed = case["matrix_path_time_fraction"]
        cases.append(
            {
                "case": case["case"],
                "b200_matrix_path_time_fraction": exposed,
                "b200_state_core_time_fraction": case["state_core_time_fraction"],
                "optimistic_speedup_ceiling": 1 / (1 - exposed * reduction),
            }
        )
    return {
        "campaign_status": campaign["campaign_status"],
        "cases": cases,
        "interpretation": (
            "The ceiling assumes every B200 Matrix-path cycle receives the full PLENA scatter-service reduction. "
            "This is deliberately optimistic: skewing does not reduce GEMM MACs or weight reads and GPU time is not PLENA time."
        ),
    }


def build_report(
    document: dict[str, Any],
    *,
    formal_campaign: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scatters = document.get("projection_scatters")
    if not isinstance(scatters, list) or not scatters:
        raise ValueError("lowered trace has no projection scatter")
    plan = ScatterPlan.from_dict(scatters[0]["plan"])
    if plan.algorithm != "kda" or plan.layout != "group_major_skewed":
        raise ValueError("KDA rotation DSE requires a group-major KDA plan")

    packets = consumer_packets(plan)
    fields = {field.name: field for field in plan.fields}
    write_metrics = _write_metrics(plan)
    results = []
    for k_rotation in range(plan.banks):
        for decay_rotation in range(plan.banks):
            results.append(
                {
                    "k_rotation": k_rotation,
                    "decay_rotation": decay_rotation,
                    **_metrics(
                        plan,
                        packets=packets,
                        fields=fields,
                        k_rotation=k_rotation,
                        decay_rotation=decay_rotation,
                        write_metrics=write_metrics,
                    ),
                }
            )
    results.sort(
        key=lambda item: (
            item["total_service_cycles"],
            item["service_cycles"],
            item["k_rotation"],
            item["decay_rotation"],
        )
    )
    baseline = next(item for item in results if item["k_rotation"] == 0 and item["decay_rotation"] == 0)
    selected_k = plan.field("k").skew_stride
    selected_decay = plan.field("decay").skew_stride
    selected = next(
        item for item in results if item["k_rotation"] == selected_k and item["decay_rotation"] == selected_decay
    )
    best_read_cycles = min(int(item["service_cycles"]) for item in results)
    best_total_cycles = int(results[0]["total_service_cycles"])
    selected_plan = _rotated(
        plan,
        k_rotation=selected_k,
        decay_rotation=selected_decay,
    )
    roundtrip = verify_scatter_roundtrip(selected_plan, tokens=1)
    baseline_plan = _rotated(plan, k_rotation=0, decay_rotation=0)
    report = {
        "schema_version": 1,
        "algorithm": "kda",
        "packet_contract": (
            "for each head and 8-key tile, consume q[8], k[8], decay[8] "
            "together; v uses independent 4-value packets and beta one value"
        ),
        "banks": plan.banks,
        "ports_per_bank": plan.ports_per_bank,
        "candidate_count": len(results),
        "baseline": {**baseline, "mapping_sha256": baseline_plan.mapping_sha256},
        "selected": {
            **selected,
            "mapping_sha256": selected_plan.mapping_sha256,
            "selection_reason": (
                "rotate k by one 8-value key packet, leave q and decay "
                "unrotated, and preserve beta group striping for writes"
            ),
            "roundtrip": roundtrip.to_dict(),
        },
        "best_service_cycles": best_read_cycles,
        "best_total_service_cycles": best_total_cycles,
        "optimal_candidate_count": sum(item["total_service_cycles"] == best_total_cycles for item in results),
        "read_service_cycle_reduction_percent": 100
        * (int(baseline["service_cycles"]) - int(selected["service_cycles"]))
        / int(baseline["service_cycles"]),
        "total_service_cycle_reduction_percent": 100
        * (int(baseline["total_service_cycles"]) - int(selected["total_service_cycles"]))
        / int(baseline["total_service_cycles"]),
        "results": results,
        "limits": [
            "the packet contract is a PLENA lane-schedule choice, not a GPU thread order",
            "rotation reuses L_SCATTER_M and changes only its descriptor/mode, not the Matrix or X_STATE opcodes",
            "RTL address-generation timing and area remain uncalibrated",
        ],
    }
    if formal_campaign is not None:
        report["formal_b200_context"] = _profile_context(
            formal_campaign,
            report["total_service_cycle_reduction_percent"],
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lowered_trace", type=Path)
    parser.add_argument("--formal-b200-campaign-summary", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.lowered_trace.read_text())
    campaign = (
        build_formal_campaign_report(args.formal_b200_campaign_summary)
        if args.formal_b200_campaign_summary is not None
        else None
    )
    rendered = json.dumps(build_report(document, formal_campaign=campaign), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
