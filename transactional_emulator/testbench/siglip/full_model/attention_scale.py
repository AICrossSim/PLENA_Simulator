"""Shared attention-scale contract for SigLIP full-model harness paths."""

from __future__ import annotations

import math


def compute_visible_head_dim(config: dict) -> int:
    """Return visible per-head dimension from model config."""
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    if num_heads <= 0:
        raise ValueError(f"num_attention_heads must be > 0, got {num_heads}")
    if hidden_size % num_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads for attention scale, "
            f"got hidden_size={hidden_size}, num_attention_heads={num_heads}"
        )
    return hidden_size // num_heads


def compute_attention_scale(config: dict) -> float:
    """Compute QK scale as 1/sqrt(visible_head_dim)."""
    head_dim = compute_visible_head_dim(config)
    return 1.0 / math.sqrt(float(head_dim))
