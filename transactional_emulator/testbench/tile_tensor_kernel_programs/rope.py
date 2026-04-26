"""TileTensorProgram rewrite of `tilelang_kernels.rope`.

This version keeps the activation tensor in the original `[B, S, H, D]` layout
and models RoPE with one explicit `parallel_region3d` over `(S, H, D)`.
RoPE coefficients are assumed to be pre-expanded to full-lane shape so the
runtime does not need one separate half-index repeat load pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    venv_lib = parent / ".venv" / "lib"
    if not venv_lib.is_dir():
        continue
    for site_pkg in venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(site_pkg))

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram
from tile_tensor_kernel_programs.testbench_runner import emit_single_output_testbench


def build_rope_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    head_count: int,
    half_dim: int,
    batch_size: int = 2,
) -> tuple[TileTensorProgram, tuple[object, object], tuple[object, object]]:
    full_dim = half_dim * 2
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=mlen // blen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    shape = (batch_size, seq_len, head_count, full_dim)

    xq_in = prog.input("XQ_IN", shape)
    xk_in = prog.input("XK_IN", shape)
    cos_in = prog.input("COS_IN", shape)
    sin_in = prog.input("SIN_IN", shape)
    neg_sin_in = prog.input("NEG_SIN_IN", shape)

    q_out_buf = prog.input("Q_OUT", shape)
    k_out_buf = prog.input("K_OUT", shape)

    xq = prog.tensor("XQ", shape)
    xk = prog.tensor("XK", shape)
    cos_t = prog.tensor("COS", shape)
    sin_t = prog.tensor("SIN", shape)
    neg_sin_t = prog.tensor("NEG_SIN", shape)
    q_out = prog.tensor("Q", shape)
    k_out = prog.tensor("K", shape)

    for batch_index in range(batch_size):
        prog.copy(xq_in[batch_index, :, :, :], xq[batch_index, :, :, :])
        prog.copy(xk_in[batch_index, :, :, :], xk[batch_index, :, :, :])
        prog.copy(cos_in[batch_index, :, :, :], cos_t[batch_index, :, :, :])
        prog.copy(sin_in[batch_index, :, :, :], sin_t[batch_index, :, :, :])
        prog.copy(neg_sin_in[batch_index, :, :, :], neg_sin_t[batch_index, :, :, :])

        with prog.parallel_region3d((seq_len, head_count, full_dim), name=f"rope_q_b{batch_index}") as (s, h, d):
            q_out[batch_index, s, h, d] = prog.if_then_else(
                d % 2 == 0,
                xq[batch_index, s, h, d] * cos_t[batch_index, s, h, d]
                + xq[batch_index, s, h, prog.pair(d)] * neg_sin_t[batch_index, s, h, d],
                xq[batch_index, s, h, prog.pair(d)] * sin_t[batch_index, s, h, d]
                + xq[batch_index, s, h, d] * cos_t[batch_index, s, h, d],
            )

        with prog.parallel_region3d((seq_len, head_count, full_dim), name=f"rope_k_b{batch_index}") as (s, h, d):
            k_out[batch_index, s, h, d] = prog.if_then_else(
                d % 2 == 0,
                xk[batch_index, s, h, d] * cos_t[batch_index, s, h, d]
                + xk[batch_index, s, h, prog.pair(d)] * neg_sin_t[batch_index, s, h, d],
                xk[batch_index, s, h, prog.pair(d)] * sin_t[batch_index, s, h, d]
                + xk[batch_index, s, h, d] * cos_t[batch_index, s, h, d],
            )

        prog.copy(q_out[batch_index, :, :, :], q_out_buf[batch_index, :, :, :])
        prog.copy(k_out[batch_index, :, :, :], k_out_buf[batch_index, :, :, :])

    return (
        prog,
        (q_out, k_out),
        (q_out_buf, k_out_buf),
    )


def build_rope_golden(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, heads, seq_len, dim = xq.shape
    half_dim = dim // 2

    fc = freqs_cis.squeeze(1)
    cos_half = fc[..., 0, 0].unsqueeze(1).expand(bsz, heads, seq_len, half_dim)
    neg_sin_half = fc[..., 0, 1].unsqueeze(1).expand(bsz, heads, seq_len, half_dim)
    sin_half = fc[..., 1, 0].unsqueeze(1).expand(bsz, heads, seq_len, half_dim)

    cos_full = torch.repeat_interleave(cos_half, repeats=2, dim=-1)
    neg_sin_full = torch.repeat_interleave(neg_sin_half, repeats=2, dim=-1)
    sin_full = torch.repeat_interleave(sin_half, repeats=2, dim=-1)

    pair_index = torch.arange(dim, device=xq.device) ^ 1
    xq_pair = xq.index_select(-1, pair_index)
    xk_pair = xk.index_select(-1, pair_index)
    even_mask = (torch.arange(dim, device=xq.device) % 2 == 0).view(1, 1, 1, dim)

    q_even_formula = xq * cos_full + xq_pair * neg_sin_full
    q_odd_formula = xq_pair * sin_full + xq * cos_full
    k_even_formula = xk * cos_full + xk_pair * neg_sin_full
    k_odd_formula = xk_pair * sin_full + xk * cos_full

    q_out = torch.where(even_mask, q_even_formula, q_odd_formula)
    k_out = torch.where(even_mask, k_even_formula, k_odd_formula)
    return q_out, k_out


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    batch_size = 2
    seq_len = 128
    head_count = 8
    half_dim = 8
    full_dim = half_dim * 2

    prog, _, (_q_out_buf, k_out_buf) = build_rope_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        head_count=head_count,
        half_dim=half_dim,
        batch_size=batch_size,
    )

    xq_data = torch.randn(batch_size, seq_len, head_count, full_dim, dtype=torch.float32) * 0.25
    xk_data = torch.randn(batch_size, seq_len, head_count, full_dim, dtype=torch.float32) * 0.25

    pos = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1)
    dim = torch.arange(half_dim, dtype=torch.float32).view(1, 1, 1, half_dim)
    theta = pos * torch.pow(10000.0, -2.0 * dim / full_dim)
    cos_half = torch.cos(theta)
    sin_half = torch.sin(theta)

    freqs_cis = torch.zeros(batch_size, 1, seq_len, half_dim, 2, 2, dtype=torch.float32)
    freqs_cis[..., 0, 0] = cos_half
    freqs_cis[..., 0, 1] = -sin_half
    freqs_cis[..., 1, 0] = sin_half
    freqs_cis[..., 1, 1] = cos_half

    cos_full = torch.repeat_interleave(cos_half.permute(0, 2, 1, 3), repeats=2, dim=-1)
    sin_full = torch.repeat_interleave(sin_half.permute(0, 2, 1, 3), repeats=2, dim=-1)
    cos_full = cos_full.expand(batch_size, seq_len, head_count, full_dim).contiguous()
    sin_full = sin_full.expand(batch_size, seq_len, head_count, full_dim).contiguous()
    neg_sin_full = -sin_full

    xq_golden = xq_data.permute(0, 2, 1, 3).contiguous()
    xk_golden = xk_data.permute(0, 2, 1, 3).contiguous()
    _q_golden, k_golden = build_rope_golden(xq_golden, xk_golden, freqs_cis)
    k_golden_bshd = k_golden.permute(0, 2, 1, 3).contiguous().reshape(
        batch_size * seq_len,
        head_count * full_dim,
    )

    emit_single_output_testbench(
        prog=prog,
        out_buf=k_out_buf,
        input_tensors={
            "XQ_IN": xq_data,
            "XK_IN": xk_data,
            "COS_IN": cos_full,
            "SIN_IN": sin_full,
            "NEG_SIN_IN": neg_sin_full,
            "Q_OUT": torch.zeros(batch_size, seq_len, head_count, full_dim, dtype=torch.float32),
            "K_OUT": torch.zeros(batch_size, seq_len, head_count, full_dim, dtype=torch.float32),
        },
        golden_output=k_golden_bshd,
        asm_name="tile_tensor_kernel_rope",
        artifact_prefix="tile_tensor_kernel_rope",
        build_dir=_TESTBENCH_DIR / "build",
    )
