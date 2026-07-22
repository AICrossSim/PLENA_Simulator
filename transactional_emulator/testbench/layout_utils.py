"""Shared tensor layout helpers for transactional emulator testbenches."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def _materialize(value: Any) -> Any:
    materialize = getattr(value, "materialize", None)
    if callable(materialize):
        return materialize()
    return value


def infer_hbm_tensor_layouts(input_tensors: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    """Infer explicit per-row HBM layouts for 2D tensors.

    The legacy writer defaults to one MLEN-wide source row when no sidecar is
    present.  That is fine for toy one-tile tensors but corrupts wide model
    weights.  Use the tensor's real trailing dimension as the row width unless a
    caller provides a more specific physical layout.
    """
    layouts: dict[str, dict[str, object]] = {}
    for name, value in input_tensors.items():
        tensor = _materialize(value)
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) != 2:
            continue
        rows = int(shape[0])
        cols = int(shape[1])
        layouts[str(name)] = {
            "source_shape": [rows, cols],
            "storage_shape": [rows, cols],
            "source_rows": rows,
            "storage_rows": rows,
            "source_row_elements": cols,
            "storage_row_elements": cols,
        }
    return layouts


def prestage_bf16_vram_matrix(
    *,
    prog,
    name: str,
    tensor: torch.Tensor,
    vram_addr: int,
    physical_shape: tuple[int, int],
    vram_preload: torch.Tensor,
):
    """Preload a BF16 matrix into VRAM's column-block-major matrix layout."""
    rows, cols = tensor.shape
    physical_rows, physical_cols = physical_shape
    if physical_rows < rows or physical_cols < cols:
        raise ValueError(f"{name}: physical_shape={physical_shape} smaller than tensor shape={tuple(tensor.shape)}")

    padded = torch.zeros(physical_rows, physical_cols, dtype=torch.bfloat16)
    padded[:rows, :cols] = tensor.to(torch.bfloat16)
    num_col_blocks = math.ceil(physical_cols / prog.mlen)
    layout_size = num_col_blocks * physical_rows * prog.mlen
    end = vram_addr + layout_size
    if end > vram_preload.numel():
        raise ValueError(f"{name}: VRAM preload too small for [{vram_addr}, {end})")

    for col_block in range(num_col_blocks):
        col_start = col_block * prog.mlen
        col_end = min(col_start + prog.mlen, physical_cols)
        width = col_end - col_start
        for row in range(physical_rows):
            dst = vram_addr + (col_block * physical_rows + row) * prog.mlen
            vram_preload[dst : dst + width] = padded[row, col_start:col_end]

    bias_input = prog.input(
        name,
        shape=(rows, cols),
        hbm_addr=0,
        prestaged_vram_addr=vram_addr,
        physical_shape=physical_shape,
    )
    return prog.load_batch(bias_input, name=name)
