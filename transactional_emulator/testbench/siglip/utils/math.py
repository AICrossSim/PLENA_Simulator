"""Shared projection and quantization helpers for SigLIP tests."""

from __future__ import annotations

import torch

from compiler.asm_templates._k_split import k_chunks
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


MXFP_BLOCK_SIZE = 8
MXFP_REAL_DATA_RATIO = (
    (MXFP_BLOCK_SIZE * MXFP_BLOCK_SIZE + MXFP_BLOCK_SIZE)
    / (MXFP_BLOCK_SIZE * MXFP_BLOCK_SIZE)
)


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
    matrix_sram_size: int = 4096,
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


def gelu_with_bf16_intermediates(tensor: torch.Tensor) -> torch.Tensor:
    """Approximate GELU while truncating key intermediates to BF16.

    This mirrors the staged arithmetic used by the hardware-visible GELU path.
    """
    tensor_f32 = tensor.float()
    step1 = (1.702 * tensor_f32).to(torch.bfloat16)
    step2 = (-step1.float()).to(torch.bfloat16)
    step3 = torch.exp(step2.float()).to(torch.bfloat16)
    step4 = (1.0 + step3.float()).to(torch.bfloat16)
    step5 = (1.0 / step4.float()).to(torch.bfloat16)
    return (tensor_f32 * step5.float()).to(torch.bfloat16)


def gelu_fp_preload(
    *,
    size: int = 64,
    one_slot: int = 1,
    coeff_slot: int = 4,
    coeff: float = 1.702,
) -> list[float]:
    """Build a common FP preload vector for GELU-heavy testbench scripts."""
    fp_preload = [0.0] * size
    fp_preload[one_slot] = 1.0
    fp_preload[coeff_slot] = coeff
    return fp_preload


def compute_hbm_size_aligned(
    num_elements: int,
    real_data_ratio: float = MXFP_REAL_DATA_RATIO,
    align_boundary: int = 64,
) -> int:
    """Compute HBM allocation size with quantization overhead and alignment.

    Args:
        num_elements: Number of input elements (before quantization)
        real_data_ratio: Quantization expansion factor (default MXFP 1.125)
        align_boundary: Alignment boundary in element units (default 64)

    Returns:
        Total allocation size in elements, aligned to boundary
    """
    size_elems = int(num_elements * real_data_ratio)
    return ((size_elems + align_boundary - 1) // align_boundary) * align_boundary


def mha_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    *,
    hq: int | None = None,
    hkv: int | None = None,
    kv_valid_len: int | None = None,
) -> torch.Tensor:
    """Multi-head scaled dot-product attention.

    Args:
        q: [batch, seq_q, num_heads, head_dim]
        k: [batch, seq_kv, num_heads, head_dim]
        v: [batch, seq_kv, num_heads, head_dim]
        scale: Attention scale (typically 1/sqrt(head_dim)).
        num_heads: Number of attention heads.
        num_kv_heads: Optional KV head count for legacy callsites. Must equal
            num_heads in this MHA helper.
        hq: Legacy alias for num_heads.
        hkv: Legacy alias for num_kv_heads.
        kv_valid_len: Optional real KV length for masking padded KV tokens.

    Returns:
        [batch, seq_q, num_heads, head_dim]
    """
    if num_heads is None:
        num_heads = hq
    elif hq is not None and hq != num_heads:
        raise ValueError(f"mha_sdpa got conflicting head counts: num_heads={num_heads}, hq={hq}")

    if num_kv_heads is None:
        num_kv_heads = hkv
    elif hkv is not None and hkv != num_kv_heads:
        raise ValueError(
            f"mha_sdpa got conflicting KV head counts: num_kv_heads={num_kv_heads}, hkv={hkv}"
        )

    if num_heads is None:
        raise ValueError("mha_sdpa requires num_heads (or legacy alias hq)")

    if num_kv_heads is not None and num_kv_heads != num_heads:
        raise ValueError(
            f"mha_sdpa expects equal Q/KV heads, got num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )

    if q.shape[2] != num_heads or k.shape[2] != num_heads or v.shape[2] != num_heads:
        raise ValueError(
            "Head count mismatch for mha_sdpa: "
            f"q={q.shape[2]}, k={k.shape[2]}, v={v.shape[2]}, num_heads={num_heads}"
        )

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).to(q_t.dtype)
    v_t = v.transpose(1, 2).to(q_t.dtype)

    attn_mask = None
    if kv_valid_len is not None and kv_valid_len < k_t.shape[-2]:
        batch_size, q_len, kv_len = q_t.shape[0], q_t.shape[-2], k_t.shape[-2]
        attn_mask = torch.zeros((batch_size, num_heads, q_len, kv_len), dtype=q_t.dtype, device=q_t.device)
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


def gqa_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    hq: int,
    hkv: int,
    kv_valid_len: int | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for legacy callsites.

    SigLIP testbenches currently run MHA (hq == hkv). For true GQA, callsites
    should pass expanded K/V heads explicitly before calling mha_sdpa.
    """
    if hq != hkv:
        raise ValueError(f"gqa_sdpa is deprecated in this codepath; expected hq==hkv, got hq={hq}, hkv={hkv}")
    return mha_sdpa(q, k, v, scale=scale, num_heads=hq, num_kv_heads=hkv, kv_valid_len=kv_valid_len)

__all__ = [
    "MXFP_BLOCK_SIZE",
    "MXFP_REAL_DATA_RATIO",
    "gelu_fp_preload",
    "gelu_with_bf16_intermediates",
    "mha_sdpa",
    "gqa_sdpa",
    "matmul_bf16_visible",
    "projection_matmul_k_split_visible",
    "quantize_flattened_like_hbm",
    "quantize_to_mxfp",
]
