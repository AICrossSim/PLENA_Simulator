#!/usr/bin/env python3
"""Build a Qwen3 route trace without the 30B checkpoint.

Every real trace comes from ``generate_true_routing_with_weights``, which loads
Qwen3-30B-A3B and runs a forward pass. That is why ``qwen3_trace_replay`` -- the
only decoder-level program that produces timing numbers -- has never had a CI
job: there is no trace to give it.

This draws router logits from a normal distribution and takes the same
``topk`` + ``softmax`` of them that the real capture does, then hands the result
to :func:`trace_from_record`, so the output goes through the same construction
and the same schema validation as a captured trace. What is synthetic is the
routing *distribution*, not the trace's shape -- so this is a harness for
plumbing and semantics, not a substitute for measuring real routing skew.

Gaussian logits are the point rather than a convenience: they put near-ties in
the top-k tail, which is where the BF16 reconstruction in :mod:`router_logits`
is hardest and where a device/reference disagreement would show up first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from transactional_emulator.testbench.moe_timing.qwen.build_route_traces import trace_from_record
from transactional_emulator.testbench.moe_timing.qwen.utils import MODEL_CONFIGS, ensure_paths


def synthetic_record(
    *,
    tokens: int,
    layer: int = 0,
    seed: int = 0,
    model_key: str = "qwen3",
    benchmark: str = "synthetic",
    sample_id: str = "synthetic0",
    phase: str = "prefill",
) -> dict[str, Any]:
    """A routing record shaped exactly like one row of the true-routing JSONL."""
    config = MODEL_CONFIGS[model_key]
    num_experts = int(config["num_experts"])
    top_k = int(config["top_k"])
    if tokens < 1:
        raise ValueError(f"tokens must be at least 1, got {tokens}")

    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, num_experts, generator=generator)
    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    top_weights = torch.softmax(top_values, dim=-1)

    return {
        "model_key": model_key,
        "benchmark": benchmark,
        "sample_id": sample_id,
        "layer": layer,
        "phase": phase,
        "tokens": tokens,
        "input_tokens": tokens,
        "category": "synthetic",
        "sample_index": 0,
        "routing_source": f"synthetic_gaussian_logits_seed{seed}",
        "routes": top_indices.tolist(),
        "route_weights": [[float(value) for value in row] for row in top_weights.tolist()],
    }


def synthetic_trace(
    *,
    tokens: int,
    layer: int = 0,
    seed: int = 0,
    mlen: int = 64,
    blen: int = 4,
    emu_threads: int = 1,
) -> dict[str, Any]:
    """A schema-valid route trace. Validation happens inside ``trace_from_record``."""
    return trace_from_record(
        synthetic_record(tokens=tokens, layer=layer, seed=seed),
        mlen=mlen,
        blen=blen,
        emu_threads=emu_threads,
        allow_uniform_weights=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ensure_paths()
    trace = synthetic_trace(
        tokens=args.tokens,
        layer=args.layer,
        seed=args.seed,
        mlen=args.mlen,
        blen=args.blen,
        emu_threads=args.emu_threads,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
