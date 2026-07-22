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
    _load_to_vector_fp,
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
    X_hbm = _load_to_vector_fp(X, precision["HBM_V_ACT_TYPE"], precision)
    return _rms_norm_vector_ref(X_hbm, eps, precision)


def golden_layer_norm(X: torch.Tensor, eps: float) -> torch.Tensor:
    """Step-by-step BF16 layer norm (mean-center + RMS norm)."""
    precision = _active_precision_settings()
    X_hbm = _load_to_vector_fp(X, precision["HBM_V_ACT_TYPE"], precision)

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    mean = qvfp(X_hbm.float().mean(dim=-1, keepdim=True))
    centered = qvfp(X_hbm - mean)
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
    """Step-by-step BF16 RoPE: (Q * cos) + (Q_rot * sin), with HBM MXFP8 quantization."""
    precision = _active_precision_settings()
    hbm_act = precision["HBM_V_ACT_TYPE"]

    def load_hbm(t):
        return _load_to_vector_fp(t, hbm_act, precision)

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    Q_q, Qr_q, cos_q, sin_q = load_hbm(Q), load_hbm(Q_rot), load_hbm(cos), load_hbm(sin)
    return qvfp(qvfp(Q_q * cos_q) + qvfp(Qr_q * sin_q))


def golden_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """Step-by-step BF16 flash attention matching emulator ISA path."""
    precision = _active_precision_settings()
    return _flash_attn_ref(Q, K, V, scale, precision=precision)


def golden_embedding_add(X: torch.Tensor, POS: torch.Tensor) -> torch.Tensor:
    """HBM MXFP8 load to BF16, then BF16 vector add."""
    precision = _active_precision_settings()
    hbm_act = precision["HBM_V_ACT_TYPE"]
    x_hbm = _load_to_vector_fp(X, hbm_act, precision)
    pos_hbm = _load_to_vector_fp(POS, hbm_act, precision)
    return quantize_to_vector_fp(x_hbm.float() + pos_hbm.float(), precision)
