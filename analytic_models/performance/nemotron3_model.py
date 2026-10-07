"""Command-line entry point for the Nemotron 3 workload and hardware DSE models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import load_model_config

from .b200_formal_campaign import build_report as build_formal_campaign_report
from .nemotron3_dse import (
    HardwareDesign,
    Nemotron3DseModel,
    ProjectionLayout,
    StateCachePolicy,
    sweep_designs,
)
from .nemotron3_gpu_microprofile import build_report as build_microprofile_report
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    WorkloadScenario,
    WeightPrecisionPolicy,
    formal_nemotron_nvfp4_weight_policy,
    storage_bytes,
)


MIB = 1024 * 1024
MODEL_KEY = "nemotron3_nano_30b_a3b"


def _enum_list(raw: str, enum_type: type) -> tuple:
    return tuple(enum_type(item.strip()) for item in raw.split(",") if item.strip())


def _int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _bool_list(raw: str) -> tuple[bool, ...]:
    values = {"0": False, "false": False, "1": True, "true": True}
    try:
        return tuple(values[item.strip().lower()] for item in raw.split(",") if item.strip())
    except KeyError as error:
        raise argparse.ArgumentTypeError("boolean lists use 0/1 or false/true") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nemotron 3 Nano workload model and uncalibrated PLENA Mamba DSE",
    )
    parser.add_argument("--mode", choices=("workload", "dse", "sweep"), default="sweep")
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--phase", type=InferencePhase, choices=InferencePhase, default=InferencePhase.DECODE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--scan-strategy", type=ScanStrategy, choices=ScanStrategy, default=ScanStrategy.SEQUENTIAL)
    parser.add_argument("--body-only", action="store_true", help="Exclude embedding and LM head")
    parser.add_argument("--moe-unique-experts", type=int)
    parser.add_argument("--activation-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--weight-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--state-precision", type=Precision, choices=Precision, default=Precision.FP32)
    parser.add_argument(
        "--gpu-microprofile-dir",
        type=Path,
        help="Validate and attach a standalone Nemotron Mamba GPU microprofile without treating it as PLENA timing",
    )
    parser.add_argument(
        "--formal-b200-campaign-summary",
        type=Path,
        help="Attach full-model B200 latency, layer traffic, and routing evidence without fitting PLENA cycles",
    )
    parser.add_argument(
        "--formal-checkpoint-weight-map",
        action="store_true",
        help="Use the campaign's real mixed NVFP4/BF16 checkpoint exclusions instead of one uniform weight dtype",
    )

    parser.add_argument("--frequency-mhz", type=int, default=1000)
    parser.add_argument("--matrix-macs-per-cycle", type=int, default=4096)
    parser.add_argument("--vector-ops-per-cycle", type=int, default=256)
    parser.add_argument("--conv-macs-per-cycle", type=int, default=256)
    parser.add_argument("--exp-ops-per-cycle", type=int, default=16)
    parser.add_argument("--hbm-bytes-per-cycle", type=int, default=64)
    parser.add_argument("--projection-buffer-banks", type=int, default=16)
    parser.add_argument("--projection-buffer-ports-per-bank", type=int, default=1)
    parser.add_argument("--matrix-result-burst-values", type=int, default=64)
    parser.add_argument("--projection-buffer-write-values-per-cycle", type=int, default=16)
    parser.add_argument("--projection-fifo-values", type=int, default=64)
    parser.add_argument("--disable-projection-bypass", action="store_true")
    parser.add_argument("--projection-consumer-start-cycles", type=int, default=0)
    parser.add_argument("--projection-consume-values-per-cycle", type=int, default=16)
    parser.add_argument("--head-lanes", type=int, default=8)
    parser.add_argument("--head-dim-lanes", type=int, default=4)
    parser.add_argument("--state-dim-lanes", type=int, default=8)

    parser.add_argument("--layout", type=ProjectionLayout, choices=ProjectionLayout, default=ProjectionLayout.ROW_MAJOR)
    parser.add_argument("--bc-broadcast", action="store_true")
    parser.add_argument("--state-cache-mib", type=int, default=0)
    parser.add_argument(
        "--state-cache-policy", type=StateCachePolicy, choices=StateCachePolicy, default=StateCachePolicy.NONE
    )

    parser.add_argument("--sweep-layouts", default="row_major,group_major_skewed")
    parser.add_argument("--sweep-broadcasts", default="0,1")
    parser.add_argument("--sweep-cache-mib", default="0,16,32,64")
    parser.add_argument("--sweep-cache-policies", default="none,lru,pinned")
    parser.add_argument("--sweep-state-dim-lanes", default="8,16")
    parser.add_argument("--sweep-bypasses", default="1")
    parser.add_argument("--sweep-fifo-values", default="64")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    return parser


def _scenario(args: argparse.Namespace) -> WorkloadScenario:
    sequence_length = args.sequence_length
    if sequence_length is None:
        sequence_length = 1 if args.phase == InferencePhase.DECODE else 2048
    return WorkloadScenario(
        phase=args.phase,
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        context_length=args.context_length,
        decode_tokens=args.decode_tokens,
        scan_strategy=args.scan_strategy,
        include_embedding=not args.body_only,
        include_lm_head=not args.body_only,
        moe_unique_experts=args.moe_unique_experts,
    )


def _base_design(args: argparse.Namespace) -> HardwareDesign:
    return HardwareDesign(
        frequency_hz=args.frequency_mhz * 1_000_000,
        matrix_macs_per_cycle=args.matrix_macs_per_cycle,
        vector_ops_per_cycle=args.vector_ops_per_cycle,
        conv_macs_per_cycle=args.conv_macs_per_cycle,
        exp_ops_per_cycle=args.exp_ops_per_cycle,
        hbm_bytes_per_cycle=args.hbm_bytes_per_cycle,
        projection_buffer_banks=args.projection_buffer_banks,
        projection_buffer_ports_per_bank=args.projection_buffer_ports_per_bank,
        matrix_result_burst_values=args.matrix_result_burst_values,
        projection_buffer_write_values_per_cycle=args.projection_buffer_write_values_per_cycle,
        projection_fifo_values=args.projection_fifo_values,
        projection_direct_bypass=not args.disable_projection_bypass,
        projection_consumer_start_cycles=args.projection_consumer_start_cycles,
        projection_consume_values_per_cycle=args.projection_consume_values_per_cycle,
        head_lanes=args.head_lanes,
        head_dim_lanes=args.head_dim_lanes,
        state_dim_lanes=args.state_dim_lanes,
        projection_layout=args.layout,
        bc_broadcast=args.bc_broadcast,
        state_cache_bytes=args.state_cache_mib * MIB,
        state_cache_policy=args.state_cache_policy,
    )


def _metadata(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_key": args.model_key,
        "model_id": model_id,
        "calibration": {
            "status": "uncalibrated_no_gpu_or_rtl",
            "meaning": "Use for relative design comparison, not final latency or PPA claims.",
        },
        "assumptions": [
            "HBM and compute can overlap within a stage; stages execute in order.",
            "Persistent Mamba state uses a dedicated cache and does not occupy Matrix SRAM.",
            "Mamba projection bank timing models a new L-compute projection buffer on the Matrix-result path, not current Matrix SRAM.",
            "Projection bypass is transparent to ISA: state-consumed fields may use FIFO, while the Vector-consumed gate always spills.",
            "Projection spill bytes are Vector-SRAM traffic and are not added to logical HBM bytes.",
            "projection_consumer_start_cycles is measured from the projection stage start and must be swept until RTL supplies ready timing.",
            "Weights are charged once per active layer and decode step; no weight-cache capacity is assumed.",
            "Pinned cache reserves a deterministic subset; LRU follows the real layer-major decode order.",
            "B/C broadcast also computes prefill's group-shared CxB product once per group instead of once per head.",
            "State precision is explicit: the model config requests FP32, but GPU profiling must record the actual cache dtype.",
        ],
    }


def _attach_b200_decode_guardrail(metadata: dict[str, Any]) -> None:
    if "formal_b200_campaign" not in metadata or not metadata.get("results"):
        return
    scenario = metadata["scenario"]
    if scenario["phase"] != InferencePhase.DECODE:
        return
    best = metadata["results"][0]
    metrics = best["metrics"]
    design = best["design"]
    gpu_itl_ms = metadata["formal_b200_campaign"]["latency"]["decode_s2048_128"]["itl_median_ms"]
    hbm_bytes = metrics["hbm_bytes_per_step"]
    gpu_seconds = gpu_itl_ms / 1000
    metadata["formal_b200_guardrail"] = {
        "gpu": "NVIDIA B200",
        "gpu_decode_itl_median_ms": gpu_itl_ms,
        "candidate_latency_ms_per_step": metrics["latency_us_per_step"] / 1000,
        "candidate_latency_over_gpu": metrics["latency_us_per_step"] / 1000 / gpu_itl_ms,
        "candidate_logical_hbm_gib_per_step": hbm_bytes / (1024**3),
        "minimum_hbm_bytes_per_cycle_to_match_gpu_ignoring_compute": hbm_bytes
        / (design["frequency_hz"] * gpu_seconds),
        "minimum_hbm_gib_per_second_to_match_gpu_ignoring_compute": hbm_bytes
        / gpu_seconds
        / (1024**3),
        "interpretation": (
            "This is a rejection guardrail, not calibrated speedup: the candidate cycles are analytic, and the "
            "bandwidth floor optimistically ignores every compute and synchronization cycle."
        ),
    }


def build_document(args: argparse.Namespace) -> dict[str, Any]:
    config = load_model_config(args.model_key)
    if args.formal_checkpoint_weight_map and args.formal_b200_campaign_summary is None:
        raise ValueError("--formal-checkpoint-weight-map requires --formal-b200-campaign-summary")
    scenario = _scenario(args)
    metadata = _metadata(args, config.model_id)
    weight_policy: WeightPrecisionPolicy | None = None
    precisions = {
        "activation": args.activation_precision,
        "weight": args.weight_precision,
        "state": args.state_precision,
    }
    metadata["scenario"] = asdict(scenario)
    metadata["precisions"] = dict(precisions)
    if args.gpu_microprofile_dir is not None:
        gpu_validation = build_microprofile_report(args.gpu_microprofile_dir)
        observed_state = gpu_validation["observed_precision"]["ssm_state"]
        requested_state = str(args.state_precision)
        metadata["gpu_validation"] = gpu_validation
        metadata["calibration"] = {
            "status": "gpu_workload_validated_plena_uncalibrated",
            "meaning": (
                "The official GPU shape, standalone latency, staged kernel time, and BF16 recurrent-state read are measured. "
                "PLENA cycles and FPGA PPA remain uncalibrated."
            ),
            "state_precision_match": requested_state == "bf16" and observed_state == "bfloat16",
        }
    if args.formal_b200_campaign_summary is not None:
        campaign = build_formal_campaign_report(args.formal_b200_campaign_summary)
        metadata["formal_b200_campaign"] = campaign["nemotron"]
        if args.formal_checkpoint_weight_map:
            if args.weight_precision != Precision.NVFP4:
                raise ValueError("--formal-checkpoint-weight-map requires --weight-precision nvfp4")
            weight_policy = formal_nemotron_nvfp4_weight_policy(
                config.arch,
                campaign["nemotron"]["checkpoint_quantization"],
            )
            metadata["precisions"]["weight_policy"] = weight_policy.to_dict()
        moe = config.arch.moe
        assert moe is not None
        moe_layers = config.arch.count_layers("moe")
        routed_expert_bytes = storage_bytes(
            2 * config.arch.hidden_size * moe.intermediate_size,
            Precision.NVFP4,
        )
        shared_expert_bytes = storage_bytes(
            2 * config.arch.hidden_size * moe.shared_intermediate_size,
            Precision.NVFP4,
        )
        metadata["formal_routing_capacity_guardrail"] = {
            "weight_format": "NVFP4 logical bytes excluding global scales and padding",
            "expert_mlp": "up -> ReLU2 -> down (two matrices)",
            "routed_expert_mib_per_layer": routed_expert_bytes / MIB,
            "one_hottest_routed_expert_for_all_moe_layers_mib": (
                routed_expert_bytes * moe_layers / MIB
            ),
            "all_shared_experts_for_all_moe_layers_mib": (
                shared_expert_bytes * moe.shared_experts * moe_layers / MIB
            ),
            "one_hottest_expert_per_layer_assignment_coverage": campaign["nemotron"][
                "routing"
            ]["one_hottest_expert_per_layer_assignment_coverage"],
            "interpretation": (
                "This is a static capacity/frequency bound, not a cache-hit prediction. "
                "Exact reuse and Expert/M/K scheduling require the raw per-step top-k trace."
            ),
        }
        metadata["calibration"].update(
            {
                "status": "gpu_system_evidence_plena_uncalibrated",
                "campaign_status": campaign["campaign_status"],
                "routing": "3013_real_layer_step_events_non_uniform",
                "formal_state_precision_match": args.state_precision == Precision.FP32,
                "formal_weight_precision_match": weight_policy is not None,
                "transactional_weight_precision_match": False,
                "plena_timing": "uncalibrated_gpu_time_is_not_plena_cycles",
            }
        )
        metadata["assumptions"].extend(
            [
                "The formal NVFP4 run observes FP32 Mamba state; FP32 is the baseline unless a numerical study permits less.",
                "NVFP4 logical weight bytes include packed E2M1 values and one FP8 scale per 16 values; global scales and alignment padding are excluded.",
                "MoE expert traffic is non-uniform: the aggregate decode hotspot reaches about 21x mean slot load.",
                "All six Nemotron NCU reports are present; their eager/replay counters calibrate workload traffic, not PLENA cycles.",
                "The formal checkpoint mixes default NVFP4 linear weights with explicit BF16 projections, conv, norms, embedding, and lm_head.",
                "NVFP4 is modeled only as logical storage traffic; the current Compiler binary and transactional Matrix path remain MXFP8.",
            ]
        )

    if args.mode == "workload":
        report = Nemotron3WorkloadModel(
            config.arch,
            activation_precision=args.activation_precision,
            weight_precision=args.weight_precision,
            state_precision=args.state_precision,
            weight_precision_policy=weight_policy,
        ).build(scenario)
        metadata["mode"] = "workload"
        metadata["workload"] = report.to_dict()
        return metadata

    model = Nemotron3DseModel(config.arch)
    base = _base_design(args)
    if args.mode == "dse":
        result = model.evaluate(
            scenario,
            base,
            **{f"{name}_precision": value for name, value in precisions.items()},
            weight_precision_policy=weight_policy,
        )
        metadata["mode"] = "dse"
        metadata["results"] = [result.to_dict()]
        _attach_b200_decode_guardrail(metadata)
        return metadata

    designs = sweep_designs(
        replace(base, state_cache_bytes=0, state_cache_policy=StateCachePolicy.NONE),
        layouts=_enum_list(args.sweep_layouts, ProjectionLayout),
        broadcasts=_bool_list(args.sweep_broadcasts),
        cache_sizes=tuple(value * MIB for value in _int_list(args.sweep_cache_mib)),
        cache_policies=_enum_list(args.sweep_cache_policies, StateCachePolicy),
        state_dim_lanes=_int_list(args.sweep_state_dim_lanes),
        bypasses=_bool_list(args.sweep_bypasses),
        fifo_values=_int_list(args.sweep_fifo_values),
    )
    results = [
        model.evaluate(
            scenario,
            design,
            **{f"{name}_precision": value for name, value in precisions.items()},
            weight_precision_policy=weight_policy,
        )
        for design in designs
    ]
    results.sort(key=lambda result: (result.total_cycles, result.hbm_read_bytes + result.hbm_write_bytes))
    metadata["mode"] = "sweep"
    metadata["design_count"] = len(results)
    metadata["results"] = [result.to_dict(include_stages=False) for result in results]
    _attach_b200_decode_guardrail(metadata)
    return metadata


def _print_summary(document: dict[str, Any], top_k: int) -> None:
    scenario = document["scenario"]
    gpu_validation = "YES" if "gpu_validation" in document or "formal_b200_campaign" in document else "NO"
    print(
        f"Nemotron 3 Nano | {scenario['phase']} | batch={scenario['batch_size']} | "
        f"context={scenario['context_length']} | GPU workload validation={gpu_validation} | PLENA calibration=NO"
    )
    if document["mode"] == "workload":
        workload = document["workload"]
        totals = workload["totals"]
        print(f"layers={workload['layer_counts']}  FLOPs={totals['flops']:,}")
        print(
            f"logical HBM read={totals['logical_hbm_read_bytes'] / MIB:,.2f} MiB  "
            f"write={totals['logical_hbm_write_bytes'] / MIB:,.2f} MiB"
        )
        return

    headings = "rank design                                             us/step  Mamba us  HBM MiB spill MiB hit% bank stall"
    print(headings)
    print("-" * len(headings))
    for rank, result in enumerate(document["results"][:top_k], start=1):
        design = result["design"]
        metrics = result["metrics"]
        steps = scenario["decode_tokens"] if scenario["phase"] == InferencePhase.DECODE else 1
        mamba_us = metrics["cycle_breakdown"].get("mamba", 0) / steps / design["frequency_hz"] * 1e6
        name = design["name"][:50]
        print(
            f"{rank:>4} {name:<50} {metrics['latency_us_per_step']:>8.1f} "
            f"{mamba_us:>9.1f} {metrics['hbm_bytes_per_step'] / MIB:>8.1f} "
            f"{metrics['projection_spill_bytes'] / steps / MIB:>9.2f} "
            f"{100 * metrics['state_cache_hit_rate']:>4.1f} {metrics['bank_stall_cycles_per_step']:>10.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(args)
    _print_summary(document, args.top_k)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"JSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
