#!/usr/bin/env python3
"""Enumerate fractional-v2 and tile-aware-v3 decompositions for one trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from analytic_models.performance.multi_chip_model import (
    estimate_multi_chip_latency,
    valid_ep_degrees,
    valid_tp_degrees,
)


def _load_json(path: Path) -> dict[str, Any]:
    handle = (
        gzip.open(path, "rt")
        if path.suffix == ".gz"
        else path.open("r")
    )
    with handle:
        return json.load(handle)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-report", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--chip-counts", default="1,2,4,8,16")
    parser.add_argument("--ports", default="1,2,4")
    parser.add_argument("--fp-width-bits", type=int, default=12)
    parser.add_argument("--kv-width-bits", type=float, default=8.125)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--append-csv", action="store_true")
    args = parser.parse_args()

    report = _load_json(args.compiler_report)
    model = _load_json(args.model_config)
    chips = tuple(int(value) for value in args.chip_counts.split(","))
    ports = tuple(int(value) for value in args.ports.split(","))
    routing_mode = (
        (report.get("trace") or {})
        .get("compiler_metadata", {})
        .get("moe_routing_mode")
    )
    rows: list[dict[str, Any]] = []
    for chip_count in chips:
        for tp_degree in valid_tp_degrees(model, chip_count):
            cp_degree = chip_count // tp_degree
            ep_values = valid_ep_degrees(
                model,
                cp_degree,
                routing_mode=routing_mode,
            )
            for ep_degree in ep_values:
                for port_count in ports:
                    common = dict(
                        report=report,
                        model=model,
                        chip_count=chip_count,
                        reference_a100_count=1,
                        parallel_model="tp-sp",
                        aggregate_hbm_bandwidth_gbps=2039.0,
                        aggregate_hbm_capacity_bytes=80_000_000_000,
                        seq_len=args.seq_len,
                        batch_size=args.batch_size,
                        fp_width_bits=args.fp_width_bits,
                        kv_width_bits=args.kv_width_bits,
                        tp_degree=tp_degree,
                        nvlink_port_count=port_count,
                    )
                    v2 = estimate_multi_chip_latency(
                        **common,
                        ep_degree=1,
                        multi_chip_model="factorized-tp-cp-v2",
                    )
                    v3 = estimate_multi_chip_latency(
                        **common,
                        ep_degree=ep_degree,
                        multi_chip_model="tile-aware-tp-cp-ep-v3",
                    )
                    rows.append(
                        {
                            "seq_len": args.seq_len,
                            "batch_size": args.batch_size,
                            "chip_count": chip_count,
                            "tp_degree": tp_degree,
                            "cp_degree": cp_degree,
                            "ep_degree": ep_degree,
                            "nvlink_port_count": port_count,
                            "fractional_v2_latency_ms": (
                                v2["latency_ns"] / 1e6
                            ),
                            "tile_aware_v3_latency_ms": (
                                v3["latency_ns"] / 1e6
                            ),
                            "v3_over_v2": (
                                v3["latency_ns"]
                                / max(v2["latency_ns"], 1e-30)
                            ),
                            "full_overlap_lower_bound_ms": (
                                v3["full_overlap_lower_bound_ns"] / 1e6
                            ),
                            "no_overlap_upper_bound_ms": (
                                v3["no_overlap_upper_bound_ns"] / 1e6
                            ),
                            "interconnect_latency_ms": (
                                v3["interconnect_latency_ns"] / 1e6
                            ),
                            "tp_collective_latency_ms": (
                                v3["tp_collective_latency_ns"] / 1e6
                            ),
                            "cp_kv_ring_latency_ms": (
                                v3["cp_kv_ring_latency_ns"] / 1e6
                            ),
                            "ep_dispatch_latency_ms": (
                                v3.get("ep_dispatch_latency_ns", 0.0) / 1e6
                            ),
                            "ep_return_latency_ms": (
                                v3.get("ep_return_latency_ns", 0.0) / 1e6
                            ),
                            "padding_cycles": v3["padding_cycles"],
                            "replicated_compute_cycles": v3[
                                "replicated_compute_cycles"
                            ],
                            "matrix_utilization_by_stage": v3[
                                "matrix_utilization_by_stage"
                            ],
                            "slowest_rank": v3["slowest_rank"],
                            "weight_replication_factor": v3[
                                "weight_replication_factor"
                            ],
                            "expert_weight_replication": v3[
                                "expert_weight_replication"
                            ],
                            "aggregate_hbm_physical_bytes": v3[
                                "aggregate_hbm_physical_bytes"
                            ],
                            "parallel_kernel_census_coverage": v3[
                                "parallel_kernel_census_coverage"
                            ],
                        }
                    )

    fastest: dict[str, dict[str, Any]] = {}
    for chip_count in chips:
        candidates = [
            row for row in rows if row["chip_count"] == chip_count
        ]
        if candidates:
            fastest[str(chip_count)] = min(
                candidates,
                key=lambda row: row["tile_aware_v3_latency_ms"],
            )
    payload = {
        "schema": "tile_aware_multichip_v3_ablation_v1",
        "compiler_report": str(args.compiler_report),
        "compiler_report_canonical_sha256": hashlib.sha256(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "model_config": str(args.model_config),
        "source": {
            "hardware": dict((report.get("trace") or {}).get("hardware") or {}),
            "workload": dict((report.get("trace") or {}).get("workload") or {}),
            "compute_timing_mode": report.get("compute_timing_mode"),
            "memory_model_version": report.get("memory_model_version"),
            "single_chip_roofline_latency_ms": (
                float(report["roofline_latency_ns"]) / 1e6
            ),
            "parallel_kernel_census_coverage": (
                (report.get("trace") or {}).get(
                    "parallel_kernel_census_coverage"
                )
            ),
        },
        "workload": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        "configuration_count": len(rows),
        "fastest_v3_by_chip_count": fastest,
        "global_fastest_v3": min(
            rows, key=lambda row: row["tile_aware_v3_latency_ms"]
        ),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    append = args.append_csv and args.output_csv.exists()
    with args.output_csv.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(
            {
                key: _csv_value(value)
                for key, value in row.items()
            }
            for row in rows
        )
    print(f"Wrote {len(rows)} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
