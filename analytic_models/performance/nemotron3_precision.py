"""Run CPU state-storage error experiments for the Nemotron 3 recurrence."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

from analytic_models.reference.state_precision import StateStorage, quantize_state, storage_bytes


@dataclass(frozen=True)
class ErrorMetrics:
    mae: float
    rmse: float
    max_abs: float
    relative_l2: float
    cosine: float


@dataclass(frozen=True)
class PrecisionResult:
    storage: str
    state_bytes: int
    compression_vs_fp32: float
    state_error: ErrorMetrics
    output_error: ErrorMetrics


def _metrics(actual: Tensor, reference: Tensor) -> ErrorMetrics:
    actual = actual.double().reshape(-1)
    reference = reference.double().reshape(-1)
    error = actual - reference
    reference_norm = torch.linalg.vector_norm(reference)
    denominator = max(float(reference_norm), 1.0e-30)
    cosine = 1.0 if torch.equal(actual, reference) else float(functional.cosine_similarity(actual, reference, dim=0))
    return ErrorMetrics(
        mae=float(error.abs().mean()),
        rmse=float(torch.sqrt(error.square().mean())),
        max_abs=float(error.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(error)) / denominator,
        cosine=cosine,
    )


def run_state_precision_experiment(
    *,
    tokens: int = 512,
    num_heads: int = 8,
    head_dim: int = 8,
    state_dim: int = 128,
    groups: int = 1,
    seed: int = 17,
) -> dict:
    """Compare storage formats while keeping update/reduction arithmetic FP32."""
    if min(tokens, num_heads, head_dim, state_dim, groups) <= 0:
        raise ValueError("all dimensions must be positive")
    if num_heads % groups:
        raise ValueError("num_heads must be divisible by groups")

    generator = torch.Generator().manual_seed(seed)
    state_shape = (1, num_heads, head_dim, state_dim)
    a_log = torch.linspace(-0.8, 1.2, num_heads)
    dt_bias = torch.linspace(-3.0, -1.0, num_heads)
    d_skip = torch.randn(num_heads, generator=generator) * 0.1
    reference_state = torch.zeros(state_shape)
    candidate_states = {
        storage: torch.zeros(state_shape) for storage in (StateStorage.BF16, StateStorage.FP16, StateStorage.MX8_B128)
    }
    reference_outputs: list[Tensor] = []
    candidate_outputs: dict[StateStorage, list[Tensor]] = {storage: [] for storage in candidate_states}

    heads_per_group = num_heads // groups
    for _ in range(tokens):
        x = torch.randn(1, num_heads, head_dim, generator=generator) * 0.15
        b_group = torch.randn(1, groups, state_dim, generator=generator) * 0.08
        c_group = torch.randn(1, groups, state_dim, generator=generator) * 0.08
        b = b_group.repeat_interleave(heads_per_group, dim=1)
        c = c_group.repeat_interleave(heads_per_group, dim=1)
        dt_raw = torch.randn(1, num_heads, generator=generator) * 0.25
        dt = functional.softplus(dt_raw + dt_bias)
        decay = torch.exp(dt[:, :, None, None] * -torch.exp(a_log)[None, :, None, None])
        update = dt[:, :, None, None] * b[:, :, None, :] * x[:, :, :, None]

        reference_state = reference_state * decay + update
        reference_y = (reference_state * c[:, :, None, :]).sum(-1) + x * d_skip[None, :, None]
        reference_outputs.append(reference_y)
        for storage, stored_state in candidate_states.items():
            updated = stored_state * decay + update
            y = (updated * c[:, :, None, :]).sum(-1) + x * d_skip[None, :, None]
            candidate_outputs[storage].append(y)
            candidate_states[storage] = quantize_state(updated, storage, block_size=128)

    elements = math.prod(state_shape)
    fp32_bytes = storage_bytes(elements, StateStorage.FP32)
    reference_output = torch.stack(reference_outputs)
    results = []
    for storage, state in candidate_states.items():
        byte_count = storage_bytes(elements, storage)
        results.append(
            PrecisionResult(
                storage=storage.value,
                state_bytes=byte_count,
                compression_vs_fp32=fp32_bytes / byte_count,
                state_error=_metrics(state, reference_state),
                output_error=_metrics(torch.stack(candidate_outputs[storage]), reference_output),
            )
        )
    return {
        "experiment": {
            "tokens": tokens,
            "batch_size": 1,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "state_dim": state_dim,
            "groups": groups,
            "seed": seed,
            "arithmetic": "FP32 update and reduction; quantized persistent state only",
            "mx8_definition": "E4M3FN values + one power-of-two scale per 128-state block (PLENA MX8-B128)",
        },
        "fp32_state_bytes": fp32_bytes,
        "results": [
            {
                **asdict(result),
                "state_error": asdict(result.state_error),
                "output_error": asdict(result.output_error),
            }
            for result in results
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--real-shape", action="store_true", help="use Nemotron H=64, P=64, N=128, G=8")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.real_shape:
        args.heads, args.head_dim, args.state_dim, args.groups = 64, 64, 128, 8
    report = run_state_precision_experiment(
        tokens=args.tokens,
        num_heads=args.heads,
        head_dim=args.head_dim,
        state_dim=args.state_dim,
        groups=args.groups,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
