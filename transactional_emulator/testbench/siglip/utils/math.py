"""Shared projection and quantization helpers for SigLIP tests."""

from __future__ import annotations

import torch

from compiler.asm_templates._k_split import k_chunks
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP format matching hardware-visible loads."""
    quantized, _, _, _ = _mx_fp_quantize_hardware(
        tensor,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[8],
    )
    return quantized.reshape(tensor.shape)


def matmul_bf16_visible(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul then cast to BF16-visible output, matching VRAM observations."""
    return torch.matmul(a.float(), b.float()).to(torch.bfloat16).float()


def projection_matmul_k_split_visible(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    mlen: int,
    matrix_sram_size: int = 1024,
) -> torch.Tensor:
    """Mirror projection_asm K-split accumulation order in golden modeling."""
    if a.shape[1] != w.shape[0]:
        raise ValueError(f"Incompatible matmul shapes: a={tuple(a.shape)}, w={tuple(w.shape)}")
    if a.shape[1] % mlen != 0:
        raise ValueError(f"K ({a.shape[1]}) must be divisible by MLEN ({mlen})")

    max_k_tiles = max(1, matrix_sram_size // mlen)
    num_k_tiles = a.shape[1] // mlen
    if num_k_tiles <= max_k_tiles:
        return matmul_bf16_visible(a, w)

    acc = None
    for chunk_start_tile, chunk_count in k_chunks(num_k_tiles, max_k_tiles):
        k_start = chunk_start_tile * mlen
        k_end = (chunk_start_tile + chunk_count) * mlen
        part = matmul_bf16_visible(a[:, k_start:k_end], w[k_start:k_end, :])
        if acc is None:
            acc = part
        else:
            acc = (acc + part).to(torch.bfloat16).float()

    if acc is None:
        raise RuntimeError("K-split produced no chunks")
    return acc


def quantize_flattened_like_hbm(tensor: torch.Tensor) -> torch.Tensor:
    """Match create_mem_for_sim behavior: quantize flattened [1, N], then reshape."""
    flat = tensor.reshape(1, -1).float()
    quantized = quantize_to_mxfp(flat).to(torch.bfloat16)
    return quantized.reshape(tensor.shape)


def gqa_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    hq: int,
    hkv: int,
    kv_valid_len: int | None = None,
) -> torch.Tensor:
    """Grouped-query scaled dot-product attention.

    Args:
        q: [batch, seq_q, num_q_heads, head_dim]
        k: [batch, seq_kv, num_kv_heads, head_dim]
        v: [batch, seq_kv, num_kv_heads, head_dim]
        scale: Attention scale (typically 1/sqrt(head_dim)).
        hq: Number of query heads.
        hkv: Number of KV heads.
        kv_valid_len: Optional real KV length for masking padded KV tokens.

    Returns:
        [batch, seq_q, num_q_heads, head_dim]
    """
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).repeat_interleave(hq // hkv, dim=1).to(q_t.dtype)
    v_t = v.transpose(1, 2).repeat_interleave(hq // hkv, dim=1).to(q_t.dtype)

    attn_mask = None
    if kv_valid_len is not None and kv_valid_len < k_t.shape[-2]:
        batch_size, q_len, kv_len = q_t.shape[0], q_t.shape[-2], k_t.shape[-2]
        attn_mask = torch.zeros((batch_size, hq, q_len, kv_len), dtype=q_t.dtype, device=q_t.device)
        attn_mask[:, :, :, kv_valid_len:] = float("-inf")

    o = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return o.transpose(1, 2)

__all__ = [
    "gqa_sdpa",
    "matmul_bf16_visible",
    "projection_matmul_k_split_visible",
    "quantize_flattened_like_hbm",
    "quantize_to_mxfp",
]
