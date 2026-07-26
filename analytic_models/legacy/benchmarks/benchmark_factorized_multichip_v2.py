#!/usr/bin/env python3
"""Reproduce the superseded fractional TP x CP x port baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from analytic_models.performance.multi_chip_model import (
    DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS,
    estimate_multi_chip_latency,
    valid_tp_degrees,
    zigzag_context_partition,
)
from compiler.aten.plena.kv_residency import plan_kv_residency


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt")
    else:
        handle = path.open("r")
    with handle:
        return json.load(handle)


def _kv_overlay(
    *,
    seq_len: int,
    mlen: int,
    matrix_sram_tiles: int,
    matrix_sram_policy: str,
    cp_degree: int,
) -> dict[str, Any]:
    global_k_blocks = math.ceil(seq_len / mlen)
    local_tokens = int(
        zigzag_context_partition(seq_len, cp_degree)["max_local_tokens"]
    )
    local_k_blocks = math.ceil(local_tokens / mlen)
    global_plan = plan_kv_residency(
        k_blocks=global_k_blocks,
        mlen=mlen,
        matrix_sram_tiles=matrix_sram_tiles,
        policy=matrix_sram_policy,
    )
    local_plan = plan_kv_residency(
        k_blocks=local_k_blocks,
        mlen=mlen,
        matrix_sram_tiles=matrix_sram_tiles,
        policy=matrix_sram_policy,
    )
    return {
        "global_k_blocks": global_k_blocks,
        "local_k_blocks": local_k_blocks,
        "global_tile_loads": global_plan.expected_tile_loads(
            q_blocks=global_k_blocks,
            causal=True,
        ),
        "local_tile_loads": local_plan.expected_tile_loads(
            q_blocks=local_k_blocks,
            causal=True,
        ),
        "matrix_sram_policy": matrix_sram_policy,
        "resident_prefix_blocks": local_plan.resident_prefix_blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-report", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--mlen", type=int, required=True)
    parser.add_argument("--matrix-sram-tiles", type=int, default=2)
    parser.add_argument("--matrix-sram-policy", default="streaming")
    parser.add_argument("--chip-counts", default="1,2,4,8,16")
    parser.add_argument("--ports", default="1,2,4")
    parser.add_argument("--fp-width-bits", type=int, default=12)
    parser.add_argument("--kv-width-bits", type=float, default=8.125)
    args = parser.parse_args()

    report = _load_json(args.compiler_report)
    model = _load_json(args.model_config)
    if not report.get("stage_compute_opcode_work_cycles"):
        raise ValueError(
            "post-hoc TP x CP scoring requires a current CostEmitter report "
            "with stage_compute_opcode_work_cycles; legacy compact reports "
            "cannot be classified exactly"
        )

    chip_counts = tuple(int(value) for value in args.chip_counts.split(","))
    ports = tuple(int(value) for value in args.ports.split(","))
    rows: list[dict[str, Any]] = []
    for chip_count in chip_counts:
        legacy = estimate_multi_chip_latency(
            report,
            model,
            chip_count=chip_count,
            reference_a100_count=1,
            parallel_model="tp-sp",
            multi_chip_model="ideal-linear-lower-bound-v1",
            aggregate_hbm_bandwidth_gbps=2039.0,
            aggregate_hbm_capacity_bytes=80_000_000_000,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            fp_width_bits=args.fp_width_bits,
            one_way_link_bandwidth_gbps=1e30,
        )
        for tp_degree in valid_tp_degrees(model, chip_count):
            cp_degree = chip_count // tp_degree
            overlay = _kv_overlay(
                seq_len=args.seq_len,
                mlen=args.mlen,
                matrix_sram_tiles=args.matrix_sram_tiles,
                matrix_sram_policy=args.matrix_sram_policy,
                cp_degree=cp_degree,
            )
            for port_count in ports:
                result = estimate_multi_chip_latency(
                    report,
                    model,
                    chip_count=chip_count,
                    reference_a100_count=1,
                    parallel_model="tp-cp",
                    multi_chip_model="factorized-tp-cp-v2",
                    tp_degree=tp_degree,
                    aggregate_hbm_bandwidth_gbps=2039.0,
                    aggregate_hbm_capacity_bytes=80_000_000_000,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    fp_width_bits=args.fp_width_bits,
                    kv_width_bits=args.kv_width_bits,
                    nvlink_port_count=port_count,
                    nvlink_port_bidirectional_gbps=(
                        DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS
                    ),
                    interconnect_startup_ns=2_500.0,
                    kv_cache_overlay=overlay,
                )
                rows.append(
                    {
                        "chip_count": chip_count,
                        "tp_degree": tp_degree,
                        "cp_degree": cp_degree,
                        "nvlink_port_count": port_count,
                        "latency_ms": result["latency_ms"],
                        "ideal_linear_lower_bound_ms": legacy["latency_ms"],
                        "lower_bound_gap_pct": (
                            100.0
                            * (
                                result["latency_ms"]
                                / legacy["latency_ms"]
                                - 1.0
                            )
                            if legacy["latency_ms"] > 0
                            else 0.0
                        ),
                        "full_overlap_lower_bound_ms": (
                            result["full_overlap_lower_bound_ns"] / 1e6
                        ),
                        "no_overlap_upper_bound_ms": (
                            result["no_overlap_upper_bound_ns"] / 1e6
                        ),
                        "max_token_fraction": result["max_token_fraction"],
                        "max_causal_pair_fraction": result[
                            "max_causal_pair_fraction"
                        ],
                        "tp_collective_latency_ms": (
                            result["tp_collective_latency_ns"] / 1e6
                        ),
                        "cp_kv_ring_latency_ms": (
                            result["cp_kv_ring_latency_ns"] / 1e6
                        ),
                        "aggregate_hbm_physical_bytes": result[
                            "aggregate_hbm_physical_bytes"
                        ],
                        "weight_replication_factor": result[
                            "weight_replication_factor"
                        ],
                        "parallel_work_census_coverage": result[
                            "parallel_work_census_coverage"
                        ],
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fastest_by_chip = {}
    for chip_count in chip_counts:
        subset = [row for row in rows if row["chip_count"] == chip_count]
        fastest_by_chip[str(chip_count)] = min(
            subset, key=lambda row: float(row["latency_ms"])
        )
    payload = {
        "schema": "factorized_multichip_ablation_v2",
        "compiler_report": str(args.compiler_report.resolve()),
        "compiler_report_sha256": hashlib.sha256(
            args.compiler_report.read_bytes()
        ).hexdigest(),
        "workload": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "mlen": args.mlen,
            "matrix_sram_tiles": args.matrix_sram_tiles,
            "matrix_sram_policy": args.matrix_sram_policy,
        },
        "configuration_count": len(rows),
        "single_chip_baseline": fastest_by_chip["1"],
        "fastest_by_chip_count": fastest_by_chip,
        "global_fastest": min(
            rows, key=lambda row: float(row["latency_ms"])
        ),
        "assumptions": {
            "multi_chip_model": "factorized-tp-cp-v2",
            "nvlink_bandwidth_semantics": "architectural_peak_assumption",
            "nvlink_oneway_gbps_per_port": 450.0,
            "startup_us": 2.5,
            "pipeline_parallel_degree": 1,
        },
    }
    if args.smoke_summary:
        smoke = _load_json(args.smoke_summary)
        resources = dict(smoke.get("worker_resource_summary") or {})
        payload["dse_smoke"] = {
            "path": str(args.smoke_summary.resolve()),
            "target_complete_trials": smoke.get("target_complete_trials"),
            "attempt_count": smoke.get("attempt_count"),
            "completed": smoke.get("completed"),
            "pruned": smoke.get("pruned"),
            "failed": smoke.get("failed"),
            "effective_complete_rate": smoke.get("effective_complete_rate"),
            "canonical_hardware_domain_size": smoke.get(
                "canonical_hardware_domain_size"
            ),
            "maximum_dynamic_concurrency": resources.get(
                "maximum_dynamic_concurrency"
            ),
            "peak_active_process_tree_rss_gib": resources.get(
                "peak_active_process_tree_rss_gib"
            ),
            "peak_worker_rss_gib": resources.get("peak_worker_rss_gib"),
            "parent_termination_count": resources.get(
                "parent_termination_count"
            ),
        }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
