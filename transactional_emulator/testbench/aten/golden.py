"""Hardware-accurate golden reference functions for ATen testbenches.

Each function matches the emulator's precision path:
  HBM (MXFP8) → VRAM (BF16) → step-by-step quantized compute → VRAM (BF16)

Re-exports from sliced_layer_test_builder where the functions originate.
"""

from __future__ import annotations

import torch

from transactional_emulator.testbench.sliced_layer_test_builder import (
    _active_precision_settings,
    _flash_attn_ref,
    _rms_norm_vector_ref,
    golden_ffn,
    quantize_to_mxfp,
    quantize_to_vector_fp,
)

__all__ = [
    "golden_embedding_add",
    "golden_ffn",
    "golden_flash_attention",
    "golden_layer_norm",
    "golden_linear",
    "golden_rms_norm",
    "golden_rope",
    "golden_softmax",
    "quantize_to_mxfp",
    "quantize_to_vector_fp",
]


def golden_linear(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """MXFP8 quantised inputs → BF16 matmul."""
    X_q = quantize_to_mxfp(X)
    W_q = quantize_to_mxfp(W)
    return torch.matmul(X_q.float(), W_q.float()).to(torch.bfloat16)


def golden_rms_norm(X: torch.Tensor, eps: float) -> torch.Tensor:
    """Step-by-step BF16 RMS norm matching emulator ISA path."""
    precision = _active_precision_settings()
    return _rms_norm_vector_ref(X, eps, precision)


def golden_layer_norm(X: torch.Tensor, eps: float) -> torch.Tensor:
    """Step-by-step BF16 layer norm (mean-center + RMS norm)."""
    precision = _active_precision_settings()

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    x = qvfp(X)
    mean = qvfp(x.float().mean(dim=-1, keepdim=True))
    centered = qvfp(x - mean)
    rms = qvfp(torch.rsqrt(centered.float().pow(2).mean(-1, keepdim=True) + eps))
    return qvfp(centered * rms)


def golden_softmax(X: torch.Tensor, scale: float) -> torch.Tensor:
    """Step-by-step BF16 softmax matching emulator ISA path."""
    precision = _active_precision_settings()

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    scores = qvfp(X.float() * scale)
    row_max = qvfp(scores.max(dim=-1, keepdim=True).values)
    shifted = qvfp(scores - row_max)
    exp_shifted = qvfp(shifted.float().exp())
    row_sum = qvfp(exp_shifted.sum(dim=-1, keepdim=True))
    return qvfp(exp_shifted / row_sum)


def golden_rope(Q: torch.Tensor, Q_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Step-by-step BF16 RoPE: (Q * cos) + (Q_rot * sin)."""
    precision = _active_precision_settings()

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    return qvfp(qvfp(Q.float() * cos.float()) + qvfp(Q_rot.float() * sin.float()))


def golden_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """Step-by-step BF16 flash attention matching emulator ISA path."""
    precision = _active_precision_settings()
    return _flash_attn_ref(Q, K, V, scale, precision=precision)


def golden_embedding_add(X: torch.Tensor, POS: torch.Tensor) -> torch.Tensor:
    """BF16 element-wise add (simple, no MXFP quantization needed)."""
    return (X.float() + POS.float()).to(torch.bfloat16)
