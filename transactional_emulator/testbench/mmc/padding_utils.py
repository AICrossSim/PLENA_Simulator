from __future__ import annotations

import math

import torch


def align_up(value: int, alignment: int) -> int:
    """Round value up to nearest multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def pad_sequence_to_kernel(seq_tensor: torch.Tensor, seq_len_kernel: int) -> torch.Tensor:
    """Pad [S, D] sequence tensor with zero rows up to seq_len_kernel."""
    seq_len_valid, feature_dim = seq_tensor.shape
    if seq_len_kernel < seq_len_valid:
        raise ValueError(
            f"seq_len_kernel ({seq_len_kernel}) must be >= seq_len_valid ({seq_len_valid})"
        )
    if seq_len_kernel == seq_len_valid:
        return seq_tensor

    padded = torch.zeros(seq_len_kernel, feature_dim, dtype=seq_tensor.dtype)
    padded[:seq_len_valid, :] = seq_tensor
    return padded


def choose_kernel_grid(seq_len_kernel: int, target_rows: int, target_cols: int, blen: int) -> tuple[int, int]:
    """Pick a factor pair close to target shape, preferring rows divisible by BLEN."""
    candidates: list[tuple[int, int, int]] = []
    for rows in range(1, int(math.isqrt(seq_len_kernel)) + 1):
        if seq_len_kernel % rows != 0:
            continue
        cols = seq_len_kernel // rows
        for r, c in ((rows, cols), (cols, rows)):
            blen_penalty = 0 if (r % blen == 0) else 10_000
            shape_penalty = abs(r - target_rows) + abs(c - target_cols)
            score = blen_penalty + shape_penalty
            candidates.append((score, r, c))

    if not candidates:
        raise ValueError(f"Failed to choose kernel grid for seq_len_kernel={seq_len_kernel}")

    candidates.sort(key=lambda x: x[0])
    _, rows_kernel, cols_kernel = candidates[0]
    return rows_kernel, cols_kernel
