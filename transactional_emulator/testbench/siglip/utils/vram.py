"""Shared VRAM tensor packing and loading helpers for SigLIP tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_vram_bf16(vram_bin_file: Path, num_elements: int, start_elem: int = 0) -> torch.Tensor:
    with open(vram_bin_file, "rb") as f:
        vram_bytes = f.read()
    vram_uint16 = np.frombuffer(vram_bytes, dtype=np.uint16)
    region = vram_uint16[start_elem : start_elem + num_elements]
    return torch.from_numpy(region.copy()).view(torch.bfloat16).float()


def pack_seq_to_chunk_major(seq_tensor: torch.Tensor, mlen: int) -> torch.Tensor:
    """Pack [seq, hidden] into chunk-major flat [chunks, seq, mlen]."""
    seq_len, hidden_dim = seq_tensor.shape
    if hidden_dim % mlen != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by mlen={mlen}")
    chunks = hidden_dim // mlen
    return seq_tensor.reshape(seq_len, chunks, mlen).permute(1, 0, 2).contiguous().reshape(-1)


def unpack_chunk_major_to_seq(
    flat_tensor: torch.Tensor,
    seq_len: int,
    hidden_dim: int,
    mlen: int,
    *,
    seq_stride: int | None = None,
) -> torch.Tensor:
    """Unpack chunk-major flat [chunks, seq_stride, mlen] into [seq, hidden]."""
    if hidden_dim % mlen != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by mlen={mlen}")
    resolved_seq_stride = seq_len if seq_stride is None else seq_stride
    if resolved_seq_stride < seq_len:
        raise ValueError(
            f"seq_stride={resolved_seq_stride} must be >= seq_len={seq_len}"
        )
    expected = resolved_seq_stride * hidden_dim
    if flat_tensor.numel() != expected:
        raise ValueError(f"Expected {expected} elements, got {flat_tensor.numel()}")
    chunks = hidden_dim // mlen
    return (
        flat_tensor.reshape(chunks, resolved_seq_stride, mlen)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(resolved_seq_stride, hidden_dim)[:seq_len]
    )


def load_vram_chunk_major_to_seq(
    vram_bin_file: Path,
    *,
    start_elem: int,
    seq_len: int,
    hidden_dim: int,
    mlen: int,
    seq_stride: int | None = None,
) -> torch.Tensor:
    resolved_seq_stride = seq_len if seq_stride is None else seq_stride
    flat = load_vram_bf16(
        vram_bin_file,
        num_elements=resolved_seq_stride * hidden_dim,
        start_elem=start_elem,
    )
    return unpack_chunk_major_to_seq(
        flat,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        mlen=mlen,
        seq_stride=resolved_seq_stride,
    )


def load_vram_seq_major_to_seq(
    vram_bin_file: Path,
    *,
    start_elem: int,
    seq_len: int,
    hidden_dim: int,
) -> torch.Tensor:
    flat = load_vram_bf16(vram_bin_file, num_elements=seq_len * hidden_dim, start_elem=start_elem)
    return flat.reshape(seq_len, hidden_dim)


def load_vram_head_major_q_to_seq(
    vram_bin_file: Path,
    *,
    start_elem: int,
    s_q: int,
    hq: int,
    d_padded: int,
) -> torch.Tensor:
    flat = load_vram_bf16(vram_bin_file, num_elements=s_q * hq * d_padded, start_elem=start_elem)
    return flat.reshape(hq, s_q, d_padded).permute(1, 0, 2).contiguous().reshape(s_q, hq * d_padded)

__all__ = [
    "load_vram_bf16",
    "load_vram_chunk_major_to_seq",
    "load_vram_head_major_q_to_seq",
    "load_vram_seq_major_to_seq",
    "pack_seq_to_chunk_major",
    "unpack_chunk_major_to_seq",
]
