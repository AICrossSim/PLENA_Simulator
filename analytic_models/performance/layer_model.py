"""
Per-Layer Analytical Model CLI for PLENA Simulator.

Runs a SINGLE PerfModel layer/op and prints its analytical cycle count (and
time at a given frequency). This is the per-layer counterpart to llama_model.py,
which composes the same methods into whole-model TTFT/TPS.

Usage:
    python layer_model.py feed_forward --config ./plena_settings.toml \
        --isa-lib ./customISA_lib.json --hidden-size 2048 --intermediate-size 8192 \
        --seq-len 2048 --batch-size 1 --mode prefill

    python layer_model.py --list   # list available layers and their parameters
"""

import argparse
import json

from perf_model import PerfModel, load_hardware_config_from_toml


# Each entry maps a layer name to the PerfModel method and the argument names it
# consumes (resolved from the parsed CLI namespace via the LAYERS table below).
# Keeping this declarative makes `--list` self-documenting and keeps dispatch in
# one place.
LAYERS = {
    "rms_layer": ("hidden_size", "seq_len", "batch_size", "mode"),
    "embeddings": ("hidden_size", "seq_len", "batch_size", "mode"),
    "residual": ("hidden_size", "seq_len", "batch_size", "mode"),
    "feed_forward": ("hidden_size", "intermediate_size", "seq_len", "batch_size", "mode"),
    "projection": (
        "hidden_size",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "seq_len",
        "batch_size",
        "mode",
    ),
    "flash_attention": (
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "seq_len",
        "kv_size",
        "batch_size",
        "mode",
    ),
    "self_attention": (
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "seq_len",
        "kv_size",
        "batch_size",
        "mode",
    ),
    "mlp_moe": (
        "hidden_size",
        "seq_len",
        "batch_size",
        "num_experts",
        "expert_per_token",
        "intermediate_size",
        "mode",
    ),
    "lm_head": ("hidden_size", "vocab_size", "batch_size"),
    "lm_head_full_seq": ("hidden_size", "vocab_size", "seq_len", "batch_size"),
    "softmax_full_seq": ("vocab_size", "seq_len", "batch_size"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-layer analytical cycle model for PLENA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available layers: " + ", ".join(sorted(LAYERS)),
    )
    parser.add_argument("layer", nargs="?", help="Layer/op name (see --list)")
    parser.add_argument("--list", action="store_true", help="List available layers and exit")

    parser.add_argument("--config", "-c", help="Path to hardware config TOML")
    parser.add_argument("--isa-lib", help="Path to customISA_lib.json")
    parser.add_argument("--frequency", type=float, default=1e9, help="Clock frequency in Hz (default: 1e9)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    # Layer dimension parameters (shared across layers; each layer uses a subset).
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--kv-size", type=int, default=None, help="KV length for attention (default: seq-len)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=None, help="default: hidden-size // num-attention-heads")
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-per-token", type=int, default=2)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    return parser


def resolve_kwargs(args: argparse.Namespace) -> dict:
    """Map the shared CLI flags onto the canonical PerfModel argument names."""
    head_dim = args.head_dim if args.head_dim is not None else args.hidden_size // args.num_attention_heads
    kv_size = args.kv_size if args.kv_size is not None else args.seq_len
    return {
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "seq_len": args.seq_len,
        "kv_size": kv_size,
        "batch_size": args.batch_size,
        "num_attention_heads": args.num_attention_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": args.vocab_size,
        "num_experts": args.num_experts,
        "expert_per_token": args.expert_per_token,
        "mode": args.mode,
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list:
        print("Available layers (parameters consumed):")
        for name in sorted(LAYERS):
            print(f"  {name:20s} {', '.join(LAYERS[name])}")
        return

    if not args.layer:
        parser.error("a layer name is required (use --list to see options)")
    if args.layer not in LAYERS:
        parser.error(f"unknown layer '{args.layer}'. Available: {', '.join(sorted(LAYERS))}")
    if not args.config or not args.isa_lib:
        parser.error("--config and --isa-lib are required")

    hardware_config = load_hardware_config_from_toml(args.config)
    perf = PerfModel(hardware_config, args.isa_lib)

    all_kwargs = resolve_kwargs(args)
    call_kwargs = {k: all_kwargs[k] for k in LAYERS[args.layer]}
    cycles = getattr(perf, args.layer)(**call_kwargs)
    time_s = cycles / args.frequency

    if args.json:
        print(json.dumps({"layer": args.layer, "params": call_kwargs, "cycles": cycles, "time_seconds": time_s}, indent=2))
    else:
        print("=" * 60)
        print(f"Layer: {args.layer}")
        print("-" * 60)
        for k in LAYERS[args.layer]:
            print(f"  {k:20s} {call_kwargs[k]}")
        print("-" * 60)
        print(f"  MLEN={perf.mlen}  BLEN={perf.blen}  VLEN={perf.vlen}  HLEN={perf.hlen}")
        print("=" * 60)
        print(f"Cycles: {cycles:,}")
        print(f"Time:   {time_s * 1e6:.3f} us  (@ {args.frequency / 1e9:.3f} GHz)")
        print("=" * 60)


if __name__ == "__main__":
    main()
