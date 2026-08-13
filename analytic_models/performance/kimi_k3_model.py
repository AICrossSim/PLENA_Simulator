"""Command-line entry point for the Kimi K3 KDA-only workload contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kimi_k3_workload import KimiK3Architecture, KimiK3KdaWorkloadModel, default_kimi_k3_scenario
from .nemotron3_workload import InferencePhase, Precision


MIB = 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimi K3 KDA-only workload model")
    parser.add_argument("--phase", type=InferencePhase, choices=InferencePhase, default=InferencePhase.DECODE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--activation-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--weight-precision", type=Precision, choices=Precision, default=Precision.BF16)
    parser.add_argument("--state-precision", type=Precision, choices=Precision, default=Precision.BF16)
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
    report = KimiK3KdaWorkloadModel(
        arch,
        activation_precision=args.activation_precision,
        weight_precision=args.weight_precision,
        state_precision=args.state_precision,
    ).build(scenario)
    return {
        "schema_version": 1,
        "model_id": "moonshotai/Kimi-K3",
        "scope": "text_backbone_kda_mixers_only",
        "calibration": "uncalibrated_no_gpu_or_rtl",
        "excluded": ["MLA", "LatentMoE", "dense FFN", "AttnRes", "vision tower"],
        "architecture": {
            "text_layers": arch.num_layers,
            "kda_layers": len(arch.kda_layer_numbers),
            "mla_layers": len(arch.mla_layer_numbers),
            "moe_layers": len(arch.moe_layer_numbers),
            "dense_ffn_layers": len(arch.dense_ffn_layer_numbers),
            "kda_state_mib_per_request": arch.recurrent_state_bytes(args.state_precision) / MIB,
            "kda_conv_state_mib_per_request": arch.conv_state_bytes(args.state_precision) / MIB,
        },
        "workload": report.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(args)
    totals = document["workload"]["totals"]
    arch = document["architecture"]
    print(
        f"Kimi K3 KDA-only | {args.phase} | batch={args.batch_size} | "
        f"KDA/MLA={arch['kda_layers']}/{arch['mla_layers']} | calibration=NO"
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
