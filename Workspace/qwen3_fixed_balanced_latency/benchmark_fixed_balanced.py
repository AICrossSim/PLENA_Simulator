#!/usr/bin/env python3
"""Benchmark the latency-only Qwen3 dense/MoE CostEmitter paths.

The script deliberately emits a compact result instead of serializing the
full CostTrace schedule.  Use ``--scheduled-shadow`` only for validation: it
enables global-row-state V4 and the compressed scoreboard replay, while the
default path is the fast stage-roofline estimator intended for DSE.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from compiler.aten.cost_frontend import clear_cost_trace_cache

from analytic_models.performance.compiler_cost_model import (
    clear_v4_work_cache,
    compile_and_evaluate_compiler_cost,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed/plena_settings.toml"
V4 = ROOT / "analytic_models/performance/calibration/hbm_dma_service_v4.json"
MODELS = {
    "qwen3-32b": {
        "config": ROOT / "Workspace/qwen3_32b_dense_analytic/qwen3-32b.json",
        "layers": 64,
        "routing_mode": "static-indices",
        "scaling": "single-layer",
    },
    "qwen3-235b-a22b": {
        "config": ROOT / "Workspace/qwen3_235b_a22b_analytic/raw_hf_config.json",
        "layers": 94,
        "routing_mode": "fixed-balanced",
        "scaling": "repeat-fixed-balanced",
    },
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _compact_result(
    model_name: str,
    trace: Any,
    report: Any,
    *,
    cold_wall_sec: float,
    warm_wall_sec: list[float],
    scheduled_shadow: bool,
) -> dict[str, Any]:
    metadata = trace.metadata
    compatibility = report.compatibility
    return {
        "model": model_name,
        "workload": metadata.get("workload"),
        "hardware": metadata.get("hardware"),
        "num_layers": metadata.get("num_layers"),
        "routing": {
            "mode": metadata.get("moe_routing_mode"),
            "fidelity": metadata.get("routing_fidelity"),
            "layer_scaling_fidelity": metadata.get("layer_scaling_fidelity"),
            "summary_hash": metadata.get("routing_summary_hash"),
            "summary_algorithm": metadata.get("routing_summary_algorithm"),
            "route_count": metadata.get("route_count"),
            "route_count_per_layer": metadata.get("route_count_per_layer"),
            "decoder_route_count": metadata.get("decoder_route_count"),
            "materialized_route_count": metadata.get("materialized_route_count"),
            "active_expert_count": metadata.get("active_expert_count"),
            "routes_per_expert": metadata.get("routes_per_expert"),
            "expert_bucket_rows": metadata.get("expert_bucket_rows"),
            "runtime_arg_topk_included": metadata.get("runtime_arg_topk_included"),
            "exact_token_addresses": metadata.get("exact_token_addresses"),
            "latency_only": metadata.get("latency_only"),
        },
        "trace": {
            "dynamic_instruction_count": trace.dynamic_instruction_count,
            "static_instruction_count": trace.static_instruction_count,
            "memory_stream_count": len(trace.memory_events),
            "dynamic_hbm_opcodes": {
                opcode: int(trace.dynamic_opcodes.get(opcode, 0))
                for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
            },
            "cost_cache_hit": metadata.get("cost_cache_hit"),
        },
        "runtime": {
            "cold_wall_sec": cold_wall_sec,
            "warm_samples_sec": warm_wall_sec,
            "warm_median_sec": (
                statistics.median(warm_wall_sec) if warm_wall_sec else None
            ),
            "warm_p95_sec": (
                _percentile(warm_wall_sec, 0.95) if warm_wall_sec else None
            ),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "latency": {
            "compute_resource_work_ns": report.compute_latency_ns,
            "v4_memory_work_ns": report.memory_latency_ns,
            "stage_roofline_ns": report.roofline_latency_ns,
            "serial_upper_bound_ns": report.serial_latency_ns,
            "one_layer_compute_resource_work_ns": (
                report.one_layer_compute_resource_work_cycles
                * compatibility.get("clock_period_ps", 1000)
                / 1000.0
            ),
            "one_layer_v4_memory_work_ns": sum(
                report.one_layer_hbm_stage_latency_ns.values()
            ),
            "stage_compute_ns": report.stage_compute_latency_ns,
            "stage_v4_memory_ns": report.hbm_stage_latency_ns,
            "stage_roofline_ns_breakdown": report.stage_roofline_latency_ns,
            "stage_bound": report.stage_bound,
            "scheduled_shadow_makespan_cycles": (
                report.scheduled_shadow_makespan_cycles
            ),
            "scheduled_shadow_ns": report.scheduled_shadow_latency_ns,
            "scheduled_shadow_status": report.scheduled_shadow.get("status"),
            "scheduled_shadow_fidelity": report.scheduled_shadow.get("fidelity"),
            "scheduled_shadow_reason": report.scheduled_shadow.get("reason"),
            "scheduled_shadow_validation": report.scheduled_shadow.get(
                "validation"
            ),
        },
        "memory_fidelity": {
            "model_version": report.memory_model_version,
            "evaluation_mode": compatibility.get("memory_evaluation_mode"),
            "row_state_runtime": compatibility.get("dma_row_state_runtime"),
            "occurrence_count": compatibility.get("occurrence_count"),
            "calibration_in_domain": report.calibration_in_domain,
            "domain_issues": compatibility.get("domain_issues"),
            "max_extrapolation_ratio": compatibility.get(
                "max_extrapolation_ratio"
            ),
            "work_cache_hit": compatibility.get("v4_work_cache_hit"),
        },
        "scheduled_shadow_requested": scheduled_shadow,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    spec = MODELS[args.model]
    layers = 1 if args.one_layer else int(spec["layers"])
    scaling = (
        "single-layer"
        if layers == 1
        else str(spec["scaling"])
    )
    memory_mode = (
        "full-cached-occurrence"
        if args.scheduled_shadow
        else "full-global-stateful"
        if args.global_v4
        else "one-layer-cached-occurrence-scaled"
    )
    kwargs = {
        "model_config": spec["config"],
        "transactional_settings": SETTINGS,
        "hbm_calibration": V4,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "num_layers": layers,
        "layer_idx": 0,
        "moe_routing_mode": spec["routing_mode"],
        "moe_layer_scaling": scaling,
        "compute_timing_mode": "rtl-v1",
        "scheduled_shadow": args.scheduled_shadow,
        "v4_memory_evaluation": memory_mode,
    }

    clear_cost_trace_cache()
    clear_v4_work_cache()
    started = time.perf_counter()
    trace, report = compile_and_evaluate_compiler_cost(
        **kwargs,
        use_trace_cache=False,
        use_v4_work_cache=False,
    )
    cold_wall = time.perf_counter() - started

    warm_times: list[float] = []
    if not args.scheduled_shadow and not args.global_v4:
        # Populate both caches once before recording steady-state samples.
        compile_and_evaluate_compiler_cost(**kwargs)
        for _ in range(args.warm_repeats):
            started = time.perf_counter()
            trace, report = compile_and_evaluate_compiler_cost(**kwargs)
            warm_times.append(time.perf_counter() - started)

    return _compact_result(
        args.model,
        trace,
        report,
        cold_wall_sec=cold_wall,
        warm_wall_sec=warm_times,
        scheduled_shadow=args.scheduled_shadow,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--seq-len", type=int, default=482)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--one-layer", action="store_true")
    parser.add_argument("--scheduled-shadow", action="store_true")
    parser.add_argument("--global-v4", action="store_true")
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seq_len <= 0 or args.batch_size <= 0 or args.warm_repeats < 0:
        parser.error("sequence, batch, and warm-repeat counts must be nonnegative")
    if args.scheduled_shadow and args.global_v4:
        parser.error("--scheduled-shadow and --global-v4 are mutually exclusive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
