"""Run CPU recurrent-state storage error experiments for Kimi K3 KDA."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from analytic_models.reference.state_precision import (
    StateStorage,
    quantize_state,
    storage_bytes,
)

from .nemotron3_precision import PrecisionResult, _metrics


def run_kda_precision_experiment(
    *,
    tokens: int = 128,
    num_heads: int = 8,
    key_dim: int = 128,
    value_dim: int = 128,
    seed: int = 29,
) -> dict:
    if min(tokens, num_heads, key_dim, value_dim) <= 0:
        raise ValueError("all dimensions must be positive")
    generator = torch.Generator().manual_seed(seed)
    shape = (1, num_heads, value_dim, key_dim)
    a_log = torch.linspace(-0.8, 0.8, num_heads)
    dt_bias = torch.linspace(-0.5, 0.5, key_dim).repeat(num_heads, 1)
    reference_state = torch.zeros(shape)
    candidates = {
        storage: torch.zeros(shape)
        for storage in (
            StateStorage.BF16,
            StateStorage.FP16,
            StateStorage.MX8_B128,
        )
    }
    reference_outputs = []
    candidate_outputs = {storage: [] for storage in candidates}
    scale = 1.0 / math.sqrt(key_dim)

    def update(state: torch.Tensor, q, k, v, gate, beta):
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
        log_decay = -5.0 * torch.sigmoid(torch.exp(a_log)[None, :, None] * (gate + dt_bias[None]))
        decayed = state * torch.exp(log_decay)[:, :, None, :]
        prediction = torch.einsum("bhvk,bhk->bhv", decayed, k)
        error = torch.sigmoid(beta)[:, :, None] * (v - prediction)
        updated = decayed + error[:, :, :, None] * k[:, :, None, :]
        output = scale * torch.einsum("bhvk,bhk->bhv", updated, q)
        return updated, output

    for _ in range(tokens):
        q = torch.randn(1, num_heads, key_dim, generator=generator) * 0.2
        k = torch.randn(1, num_heads, key_dim, generator=generator) * 0.2
        v = torch.randn(1, num_heads, value_dim, generator=generator) * 0.2
        gate = torch.randn(1, num_heads, key_dim, generator=generator) * 0.2
        beta = torch.randn(1, num_heads, generator=generator)
        reference_state, output = update(reference_state, q, k, v, gate, beta)
        reference_outputs.append(output)
        for storage, stored in candidates.items():
            updated, output = update(stored, q, k, v, gate, beta)
            candidate_outputs[storage].append(output)
            candidates[storage] = quantize_state(updated, storage)

    elements = math.prod(shape)
    fp32_bytes = storage_bytes(elements, StateStorage.FP32)
    reference_output = torch.stack(reference_outputs)
    results = []
    for storage, state in candidates.items():
        byte_count = storage_bytes(elements, storage)
        result = PrecisionResult(
            storage=storage.value,
            state_bytes=byte_count,
            compression_vs_fp32=fp32_bytes / byte_count,
            state_error=_metrics(state, reference_state),
            output_error=_metrics(torch.stack(candidate_outputs[storage]), reference_output),
        )
        results.append(
            {
                **asdict(result),
                "state_error": asdict(result.state_error),
                "output_error": asdict(result.output_error),
            }
        )
    return {
        "experiment": {
            "tokens": tokens,
            "batch_size": 1,
            "num_heads": num_heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "seed": seed,
            "arithmetic": "FP32 KDA update/reduction; quantized persistent recurrent state only",
            "normalization": "rsqrt(sum(x^2) + 1e-6), matching FlashKDA",
            "mx8_definition": "E4M3FN values + one power-of-two scale per 128-key block",
        },
        "fp32_state_bytes": fp32_bytes,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--real-shape",
        action="store_true",
        help="use Kimi K3 H=96, K=128, V=128",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.real_shape:
        args.heads, args.key_dim, args.value_dim = 96, 128, 128
    report = run_kda_precision_experiment(
        tokens=args.tokens,
        num_heads=args.heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
