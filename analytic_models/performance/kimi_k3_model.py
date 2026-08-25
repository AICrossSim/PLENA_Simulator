"""Command-line entry point for the Kimi K3 KDA-only workload contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .b200_formal_campaign import build_report as build_formal_campaign_report
from .kimi_k3_gpu_microprofile import build_report as build_microprofile_report
from .kimi_k3_workload import (
    KimiK3Architecture,
    KimiK3HybridWorkloadModel,
    KimiK3KdaWorkloadModel,
    default_kimi_k3_scenario,
)
from .nemotron3_workload import InferencePhase, Precision


MIB = 1024 * 1024
KIMI_K3_PROFILE_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


def _rtx5090_profile_summary() -> dict:
    """Pinned metadata supplied with the standalone official KDA mixer run."""
    return {
        "source_status": "summary_ingested_raw_nsys_ncu_not_local",
        "gpu": "RTX 5090",
        "revision": KIMI_K3_PROFILE_REVISION,
        "weights": "deterministic_random_bf16",
        "scope": "official_KimiDeltaAttention_single_mixer",
        "shape": {
            "hidden_size": 7168,
            "heads": 96,
            "head_dim": 128,
            "conv_kernel": 4,
        },
        "numerical_validation": {
            "flashkda_reference_max_error": 0.0,
            "flashkda_reference_exact": True,
            "fla_relative_error_range_percent": [0.17, 0.515],
        },
        "latency": [
            {"phase": "prefill", "batch": 1, "sequence": 128, "median_us": 922.0, "p95_us": 930.0},
            {"phase": "prefill", "batch": 1, "sequence": 512, "median_us": 2610.0, "p95_us": 2622.0},
            {"phase": "prefill", "batch": 1, "sequence": 2048, "median_us": 9985.0, "p95_us": 10033.0},
            {"phase": "prefill", "batch": 1, "sequence": 8192, "median_us": 40214.0, "p95_us": 40254.0},
            {"phase": "decode", "batch": 1, "sequence": 1, "median_us": 565.0, "p95_us": 567.0},
            {"phase": "decode", "batch": 4, "sequence": 1, "median_us": 625.0, "p95_us": 628.0},
            {"phase": "decode", "batch": 8, "sequence": 1, "median_us": 659.0, "p95_us": 722.0},
            {"phase": "decode", "batch": 16, "sequence": 1, "median_us": 738.0, "p95_us": 742.0},
        ],
        "observed_state": {
            "recurrent_shape": [1, 96, 128, 128],
            "recurrent_precision": "fp32",
            "recurrent_mib_per_layer": 6.0,
            "conv_precision": "bf16",
            "conv_mib_per_layer": 0.28125,
            "persistent_mib_for_69_layers": 433.40625,
        },
        "backend": {
            "prefill": "fla_chunk_kda",
            "decode": "fla_fused_recurrent_kda",
            "flashkda_golden_compiled_on_sm120": True,
            "flashkda_used_by_profiled_prefill_mixer": False,
        },
        "not_calibrated": [
            "PLENA stage cycles",
            "PLENA end-to-end latency",
            "full-mixer projection/conv/gate per-stage GPU time",
            "projection/conv/gate/out-projection measured DRAM bytes",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimi K3 workload model")
    parser.add_argument("--scope", choices=("kda", "full"), default="kda")
    parser.add_argument("--phase", type=InferencePhase, choices=InferencePhase, default=InferencePhase.DECODE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--activation-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--weight-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--state-precision", type=Precision, choices=Precision, default=Precision.FP32)
    parser.add_argument(
        "--conv-state-precision",
        type=Precision,
        choices=Precision,
        default=Precision.BF16,
    )
    parser.add_argument(
        "--gpu-microprofile-dir",
        type=Path,
        help=(
            "Validate and attach the pinned B200 KDA microprofile without treating physical GPU traffic as PLENA timing"
        ),
    )
    parser.add_argument(
        "--formal-b200-campaign-summary",
        type=Path,
        help="Attach the complete formal B200 stage campaign as DSE evidence, not as PLENA cycle calibration",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def build_document(args: argparse.Namespace) -> dict:
    arch = KimiK3Architecture()
    scenario = default_kimi_k3_scenario(
        args.phase,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        context_length=args.context_length,
    )
    model_type = KimiK3KdaWorkloadModel if args.scope == "kda" else KimiK3HybridWorkloadModel
    report = model_type(
        arch,
        activation_precision=args.activation_precision,
        weight_precision=args.weight_precision,
        state_precision=args.state_precision,
        conv_state_precision=args.conv_state_precision,
    ).build(scenario)
    document = {
        "schema_version": 1,
        "model_id": "moonshotai/Kimi-K3",
        "scope": ("text_backbone_kda_mixers_only" if args.scope == "kda" else "full_93_layer_text_backbone"),
        "calibration": {
            "state_contract": "rtx5090_real_shape_random_weight_layer_profile",
            "plena_timing": "uncalibrated_no_rtl_or_stage_counter_fit",
        },
        "gpu_profile": _rtx5090_profile_summary(),
        "excluded": (
            ["MLA", "LatentMoE", "dense FFN", "AttnRes", "vision tower"] if args.scope == "kda" else ["vision tower"]
        ),
        "architecture": {
            "text_layers": arch.num_layers,
            "kda_layers": len(arch.kda_layer_numbers),
            "mla_layers": len(arch.mla_layer_numbers),
            "moe_layers": len(arch.moe_layer_numbers),
            "dense_ffn_layers": len(arch.dense_ffn_layer_numbers),
            "attn_res_blocks": len(range(0, arch.num_layers, arch.attn_res_block_size)),
            "mla_cache_elements_per_token_per_layer": arch.mla_cache_elements_per_token,
            "kda_state_mib_per_request": arch.recurrent_state_bytes(args.state_precision) / MIB,
            "kda_conv_state_mib_per_request": arch.conv_state_bytes(args.conv_state_precision) / MIB,
            "state_precision": args.state_precision.value,
            "conv_state_precision": args.conv_state_precision.value,
        },
        "modeling_assumptions": {
            "mla_cache": "576 elements/token/layer: 512 latent KV plus 64 RoPE key",
            "decode_moe_unique_experts": "min(896, tokens * 16) unless explicitly supplied",
            "prefill_moe_unique_experts": "defaults to all experts reached by token assignments",
            "weight_cache": "not modeled; logical weight reads are a no-weight-residency upper bound",
            "attn_res": "structural work equation, not a bit-exact kernel replay",
            "latency": "no PLENA cycle claim until transactional counters are calibrated by RTL",
        },
        "workload": report.to_dict(),
    }
    if args.gpu_microprofile_dir is not None:
        validation = build_microprofile_report(args.gpu_microprofile_dir)
        document["b200_gpu_validation"] = validation
        document["calibration"] = {
            "state_contract": "b200_official_wrapper_bit_exact_fp32_state",
            "physical_traffic": "b200_directional_dram_and_l2_counters",
            "logical_traffic": "architecture_derived_tensor_bytes",
            "plena_timing": "uncalibrated_no_rtl_or_stage_counter_fit",
            "state_precision_match": args.state_precision == Precision.FP32,
        }
    if args.formal_b200_campaign_summary is not None:
        campaign = build_formal_campaign_report(args.formal_b200_campaign_summary)
        document["formal_b200_campaign"] = campaign["kda"]
        document["calibration"].update(
            {
                "full_layer_stage_dominance": "b200_ncu_replay_validated",
                "campaign_status": campaign["campaign_status"],
                "formal_state_precision_match": args.state_precision == Precision.FP32,
                "plena_timing": "uncalibrated_gpu_time_is_not_plena_cycles",
            }
        )
        document["modeling_assumptions"]["kda_bottleneck"] = (
            "The full-layer model must include Matrix/weight streaming. On B200 the qkv+gate+out path "
            "takes 62-74% of profiled kernel time while the recurrent core takes 5-15%."
        )
    return document


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(args)
    totals = document["workload"]["totals"]
    arch = document["architecture"]
    gpu_evidence = "b200_gpu_validation" in document or "formal_b200_campaign" in document
    print(
        f"Kimi K3 {args.scope} | {args.phase} | batch={args.batch_size} | "
        f"KDA/MLA={arch['kda_layers']}/{arch['mla_layers']} | "
        f"B200 evidence={'YES' if gpu_evidence else 'NO'}, "
        "PLENA timing=NO"
    )
    print(f"FLOPs={totals['flops']:,}")
    print(
        f"logical HBM read={totals['logical_hbm_read_bytes'] / MIB:,.2f} MiB  "
        f"write={totals['logical_hbm_write_bytes'] / MIB:,.2f} MiB"
    )
    print(
        f"persistent KDA state={arch['kda_state_mib_per_request']:.2f} MiB  "
        f"conv state={arch['kda_conv_state_mib_per_request']:.2f} MiB/request"
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"JSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
