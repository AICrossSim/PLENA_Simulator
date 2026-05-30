#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.utils.math import (
    gelu_with_bf16_intermediates,
    quantize_to_mxfp,
)


def gelu_tanh_with_bf16_intermediates(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation with BF16 truncation at hardware-visible steps."""
    x_f32 = x.float()

    x2 = (x_f32 * x_f32).to(torch.bfloat16)
    x3 = (x2.float() * x_f32).to(torch.bfloat16)
    cubic_term = (0.044715 * x3.float()).to(torch.bfloat16)
    poly = (x_f32 + cubic_term.float()).to(torch.bfloat16)

    z = (0.7978845608028654 * poly.float()).to(torch.bfloat16)  # sqrt(2/pi)
    two_z = (z.float() + z.float()).to(torch.bfloat16)
    exp_2z = torch.exp(two_z.float()).to(torch.bfloat16)

    num = (exp_2z.float() - 1.0).to(torch.bfloat16)
    den = (num.float() + 2.0).to(torch.bfloat16)
    tanh_z = (num.float() * (1.0 / den.float())).to(torch.bfloat16)

    one_plus_tanh = (1.0 + tanh_z.float()).to(torch.bfloat16)
    half_x = (0.5 * x_f32).to(torch.bfloat16)
    return (half_x.float() * one_plus_tanh.float()).to(torch.bfloat16)


def pairwise_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Compute simple numeric agreement metrics between two tensors."""
    a_f = a.float()
    b_f = b.float()
    diff = torch.abs(a_f - b_f)
    return {
        "mae": float(diff.mean().item()),
        "max_error": float(diff.max().item()),
        "match_1e2": float((torch.isclose(a_f, b_f, atol=1e-2, rtol=1e-2).float().mean() * 100).item()),
        "match_2e2": float((torch.isclose(a_f, b_f, atol=2e-2, rtol=2e-2).float().mean() * 100).item()),
    }


def print_metrics(title: str, m: dict[str, float]) -> None:
    print(title)
    print(f"  MAE: {m['mae']:.6f}")
    print(f"  Max Error: {m['max_error']:.6f}")
    print(f"  Match Rate @ atol=rtol=1e-2: {m['match_1e2']:.2f}%")
    print(f"  Match Rate @ atol=rtol=2e-2: {m['match_2e2']:.2f}%")


if __name__ == "__main__":
    seed = int(os.environ.get("SIGLIP_GELU_SEED", "42"))
    batch = int(os.environ.get("SIGLIP_GELU_BATCH", "4"))
    hidden = int(os.environ.get("SIGLIP_GELU_HIDDEN", "128"))
    use_mxfp_input = os.environ.get("SIGLIP_GELU_USE_MXFP_INPUT", "1") != "0"

    torch.manual_seed(seed)
    x = torch.randn(batch, hidden, dtype=torch.bfloat16)
    x_in = quantize_to_mxfp(x).to(torch.bfloat16) if use_mxfp_input else x

    gelu_tanh_hw = gelu_tanh_with_bf16_intermediates(x_in)
    gelu_sigmoid_hw = gelu_with_bf16_intermediates(x_in)
    gelu_pytorch_tanh = F.gelu(x_in.float(), approximate="tanh").to(torch.bfloat16)

    print("================================================")
    print("GELU Variant Comparison")
    print("================================================")
    print(f"Input shape: {tuple(x_in.shape)}")
    print(f"Input mode: {'MXFP-quantized BF16' if use_mxfp_input else 'BF16'}")
    print("")

    print_metrics("[tanh_hw] vs [sigmoid_hw]", pairwise_metrics(gelu_tanh_hw, gelu_sigmoid_hw))
    print("")
    print_metrics("[tanh_hw] vs [pytorch_tanh]", pairwise_metrics(gelu_tanh_hw, gelu_pytorch_tanh))
    print("")
    print_metrics("[sigmoid_hw] vs [pytorch_tanh]", pairwise_metrics(gelu_sigmoid_hw, gelu_pytorch_tanh))
