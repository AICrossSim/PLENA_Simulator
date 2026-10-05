"""Command line for the stacked-DRAM model.

Examples (from the repository root)::

    python -m analytic_models.stacked_dram describe \\
        --profile analytic_models/stacked_dram/examples/fictional_stacked_dram.json --total-layers 4,8,12

    python -m analytic_models.stacked_dram estimate \\
        --model llama-3.1-8b --model-lib PLENA_Compiler/doc/Model_Lib \\
        --config plena_settings.toml --isa-lib analytic_models/performance/customISA_lib.json \\
        --profile analytic_models/stacked_dram/examples/fictional_stacked_dram.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml

from .estimate import (
    OVERLAP_POLICIES,
    DecoderLatencyEstimate,
    DecoderShape,
    HbmStoragePrecision,
    estimate_decoder_latency,
)
from .memory import FixedBandwidthMemory
from .model import StackedDramModel
from .profile import load_memory_profile


def _int_list(text: str) -> list[int]:
    try:
        values = [int(item) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _gb(num_bytes: float) -> str:
    return f"{num_bytes / 1e9:.3f}"


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f}"


def _describe(args: argparse.Namespace) -> int:
    memory = load_memory_profile(args.profile)
    if isinstance(memory, StackedDramModel):
        totals = args.total_layers or [memory.config.total_layers]
        variants = []
        for total in totals:
            if args.connected_layers:
                connected_choices = [count for count in args.connected_layers if count <= total]
            elif args.total_layers:
                connected_choices = [total]
            else:
                connected_choices = [memory.config.connected_layers]
            variants.extend(memory.with_layers(total, connected) for connected in connected_choices)
        if not variants:
            raise SystemExit("no --connected-layers value fits within any --total-layers value")
    elif args.total_layers or args.connected_layers:
        raise SystemExit("--total-layers/--connected-layers apply to stacked_dram profiles only")
    else:
        variants = [memory]
    rows = [variant.describe() for variant in variants]

    if args.json:
        print(json.dumps(rows if len(rows) > 1 else rows[0], indent=2))
        return 0

    print(f"{memory.name} [{memory.kind}]")
    if isinstance(memory, StackedDramModel):
        print(
            f"{'layers':>7} {'connected':>9} {'peak GB/s':>11} {'efficiency':>10} {'usable GB/s':>11} {'capacity GB':>11} {'clock scale':>11}"
        )
        for row in rows:
            print(
                f"{row['total_layers']:>7} {row['connected_layers']:>9} {_gb(row['peak_bandwidth_bytes_per_s']):>11} "
                f"{row['connectivity_efficiency']:>10.4f} {_gb(row['usable_bandwidth_bytes_per_s']):>11} "
                f"{_gb(row['capacity_bytes']):>11} {row['compute_frequency_scale']:>11.4f}"
            )
    else:
        row = rows[0]
        capacity = "not modeled" if row["capacity_bytes"] is None else f"{_gb(row['capacity_bytes'])} GB"
        print(f"usable bandwidth {_gb(row['usable_bandwidth_bytes_per_s'])} GB/s, capacity {capacity}")
    return 0


def _model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path)
    if not args.model_lib:
        raise SystemExit("--model-lib is required with --model")
    path = Path(args.model_lib) / f"{args.model}.json"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in Path(args.model_lib).glob("*.json")))
        raise SystemExit(f"model {args.model!r} not found in {args.model_lib}; available: {available}")
    return path


def _memory(args: argparse.Namespace) -> Any:
    if args.profile:
        memory = load_memory_profile(args.profile)
    else:
        capacity = None if args.fixed_capacity_gib is None else round(args.fixed_capacity_gib * 2**30)
        memory = FixedBandwidthMemory(
            name="command-line fixed bandwidth",
            bandwidth_bytes_per_s=args.fixed_bandwidth_gbs * 1e9,
            capacity_bytes=capacity,
            provenance={"source": "command line"},
        )
    if args.total_layers is not None or args.connected_layers is not None:
        if not isinstance(memory, StackedDramModel):
            raise SystemExit("--total-layers/--connected-layers apply to stacked_dram profiles only")
        total = memory.config.total_layers if args.total_layers is None else args.total_layers
        memory = memory.with_layers(total, args.connected_layers)
    return memory


def _print_estimate(result: DecoderLatencyEstimate) -> None:
    memory = result.memory
    capacity = "not modeled" if memory["capacity_bytes"] is None else f"{_gb(memory['capacity_bytes'])} GB"
    print(
        f"Model           {result.model} (batch {result.batch_size}, input {result.input_seq_len}, output {result.output_seq_len})"
    )
    print(
        f"Memory          {memory['name']} [{memory['kind']}]: usable {_gb(memory['usable_bandwidth_bytes_per_s'])} GB/s, capacity {capacity}"
    )
    print(f"Compute clock   {result.compute_frequency_hz / 1e9:.4f} GHz")
    print(f"Overlap policy  {result.overlap_policy}" + (" (LM head included)" if result.include_lm_head else ""))
    print()
    print(f"{'phase':<14} {'time ms':>12} {'compute ms':>12} {'memory ms':>12} {'mem-bound':>10} {'DRAM GB':>10}")
    for label, phase in (
        ("prefill", result.prefill),
        ("first decode", result.first_token_decode),
        (f"decode x{result.output_seq_len}", result.decode),
    ):
        bound = phase.memory_bound_seconds / phase.seconds if phase.seconds else 0.0
        print(
            f"{label:<14} {_ms(phase.seconds):>12} {_ms(phase.compute_seconds):>12} {_ms(phase.memory_seconds):>12} "
            f"{bound:>10.1%} {_gb(phase.read_bytes + phase.write_bytes):>10}"
        )
    print()
    print(f"TTFT            {_ms(result.ttft_seconds)} ms")
    print(f"TPS             {result.tps:.2f} tokens/s")
    total = result.weight_footprint_bytes + result.kv_cache_footprint_bytes
    verdict = {None: "capacity not modeled", True: "fits", False: "exceeds capacity"}[result.fits_in_memory]
    print(
        f"Footprint       weights {_gb(result.weight_footprint_bytes)} GB + KV {_gb(result.kv_cache_footprint_bytes)} GB "
        f"= {_gb(total)} GB ({verdict})"
    )
    for warning in result.warnings:
        print(f"warning: {warning}")


def _estimate(args: argparse.Namespace) -> int:
    memory = _memory(args)
    shape = DecoderShape.from_json(_model_path(args))
    perf = PerfModel(load_hardware_config_from_toml(args.config), args.isa_lib)
    precision = HbmStoragePrecision.from_settings(args.config)
    result = estimate_decoder_latency(
        shape,
        perf,
        precision,
        memory,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        frequency_hz=args.frequency_hz,
        overlap_policy=args.overlap,
        include_lm_head=args.include_lm_head,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_estimate(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m analytic_models.stacked_dram", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe", help="bandwidth, capacity and clock scale derived from a memory profile")
    describe.add_argument("--profile", required=True, help="memory profile JSON")
    describe.add_argument(
        "--total-layers", type=_int_list, help="comma-separated stack heights to tabulate (fully connected by default)"
    )
    describe.add_argument("--connected-layers", type=_int_list, help="comma-separated connected-layer counts")
    describe.add_argument("--json", action="store_true", help="print JSON")
    describe.set_defaults(handler=_describe)

    estimate = commands.add_parser("estimate", help="memory-aware TTFT/TPS of a dense decoder on one PLENA chip")
    memory = estimate.add_mutually_exclusive_group(required=True)
    memory.add_argument("--profile", help="memory profile JSON (stacked_dram or fixed_bandwidth)")
    memory.add_argument("--fixed-bandwidth-gbs", type=float, help="ad-hoc fixed-bandwidth memory, decimal GB/s")
    estimate.add_argument("--fixed-capacity-gib", type=float, help="capacity of the ad-hoc fixed-bandwidth memory")
    model = estimate.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="model name in --model-lib")
    model.add_argument("--model-path", help="HuggingFace-style config JSON")
    estimate.add_argument("--model-lib", help="directory of model config JSONs")
    estimate.add_argument("--config", required=True, help="plena_settings.toml")
    estimate.add_argument("--isa-lib", required=True, help="customISA_lib.json")
    estimate.add_argument("--batch-size", type=int, default=4)
    estimate.add_argument("--input-seq", type=int, default=2048)
    estimate.add_argument("--output-seq", type=int, default=1024)
    estimate.add_argument("--frequency-hz", type=float, default=1e9, help="nominal compute clock (default 1 GHz)")
    estimate.add_argument("--overlap", choices=OVERLAP_POLICIES, default="stage-roofline")
    estimate.add_argument("--include-lm-head", action="store_true", help="also charge the LM head projection")
    estimate.add_argument("--total-layers", type=int, help="override the stacked_dram profile's stack height")
    estimate.add_argument("--connected-layers", type=int, help="override the connected layers (default: all)")
    estimate.add_argument("--json", action="store_true", help="print JSON")
    estimate.set_defaults(handler=_estimate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "fixed_capacity_gib", None) is not None and not getattr(args, "fixed_bandwidth_gbs", None):
        raise SystemExit("--fixed-capacity-gib requires --fixed-bandwidth-gbs")
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        # Invalid profiles, layer counts and paths are user input: report them without a traceback.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
