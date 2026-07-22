"""Shared helpers for the GPT-OSS routed-MoE testbench scripts.

These helpers were previously copy-pasted (byte-identical) across the
``routed_moe/`` and ``models/gpt_oss/`` test scripts. They are consolidated
here so there is a single source of truth. Names keep their historical leading
underscore so existing call sites in the test scripts need no changes beyond
importing them from this module.

Only helpers that were identical across every defining file live here; helpers
that legitimately differ between tests (e.g. ``_activation_golden``,
``_linear_projection_golden``) are intentionally left in their own files.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from transactional_emulator.testbench.aten.golden import quantize_to_mxfp


def _activation_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """BF16 GPT-OSS clamp-gated SwiGLU activation, matching the vector-ISA emit sequence."""
    neg_alpha = torch.tensor(-1.702, dtype=torch.bfloat16).float()

    gate = _bf16(torch.clamp(gate.float(), max=7.0))
    up = _bf16(torch.clamp(up.float(), max=7.0))
    up = _bf16(torch.clamp(up.float(), min=-7.0))

    sigmoid = _bf16(gate.float())
    sigmoid = _bf16(sigmoid.float() * neg_alpha)
    sigmoid = _bf16(torch.exp(torch.clamp(sigmoid.float(), -88.0, 88.0)))
    sigmoid = _bf16(sigmoid.float() + 1.0)
    sigmoid = _bf16(torch.reciprocal(sigmoid.float()))

    glu = _bf16(gate.float() * sigmoid.float())
    up_plus_one = _bf16(up.float() + 1.0)
    return _bf16(up_plus_one.float() * glu.float())


def _linear_projection_golden(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    mlen: int,
    mram_tile_capacity: int = 4,
    hbm_input: bool = True,
) -> torch.Tensor:
    """Hardware-aware linear projection golden, including compiler K-split.

    HBM-loaded activations and weights are MX-quantized before matmul. VRAM
    intermediate activations are already BF16 and must not be MX-quantized
    again. When K exceeds MRAM tile capacity, the compiler emits partial matmuls
    and BF16 VRAM adds; mirror that rounding order here.
    """
    x_q = quantize_to_mxfp(x) if hbm_input else x.to(torch.bfloat16)
    w_q = quantize_to_mxfp(w)
    k_total = x_q.shape[1]
    chunk = mlen * mram_tile_capacity

    acc = None
    for k_start in range(0, k_total, chunk):
        k_end = min(k_start + chunk, k_total)
        partial = torch.matmul(x_q[:, k_start:k_end].float(), w_q[k_start:k_end, :].float()).to(torch.bfloat16)
        acc = partial if acc is None else (acc.float() + partial.float()).to(torch.bfloat16)
    assert acc is not None
    return acc


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _exact_mxfp8_tensor(shape: tuple[int, ...], *, stride: int, offset: int = 0) -> torch.Tensor:
    values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
    idx = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.long)
    return values[(idx * stride + offset) % values.numel()].reshape(shape)


def _decode_bf16_dump(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u2")
    return torch.tensor(raw.astype(np.uint16), dtype=torch.uint16).view(torch.bfloat16)


def _decode_u32_dump(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u4")
    return torch.tensor(raw.astype(np.uint32), dtype=torch.int64)


def _expanded_bias(row_bias: torch.Tensor, rows: int) -> torch.Tensor:
    return row_bias.to(torch.bfloat16).reshape(1, -1).repeat(rows, 1)


def _stats_dict(stats) -> dict:
    return {
        "rel_rms": stats.rel_rms,
        "atol": stats.atol,
        "rtol": stats.rtol,
        "allclose": stats.allclose,
        "pass_rate": stats.pass_rate,
        "max_abs_error": stats.max_abs_error,
    }


def _comparison_params_for(output_vram, *, rows: int, hidden: int, mlen: int, golden: torch.Tensor) -> dict:
    output_vram_addr = output_vram._program._compiler.get_vram_addr(output_vram.name)
    output_physical_rows = output_vram.physical_shape[0]
    chunks_per_batch = math.ceil(hidden / mlen)
    rows_to_read = (chunks_per_batch - 1) * output_physical_rows + rows
    return {
        "start_row_idx": output_vram_addr // mlen,
        "num_rows": rows_to_read,
        "num_batches": rows,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "physical_rows": output_physical_rows,
        "atol": float((golden.float().std(unbiased=False) * 0.01).item()),
        "rtol": 0.02,
    }
