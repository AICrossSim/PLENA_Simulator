"""TileTensorProgram rewrite of `tilelang_kernels.rmsnorm`."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram
from tile_tensor_kernel_programs.testbench_runner import emit_single_output_testbench


def build_rmsnorm_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
    eps: float = 1.0e-6,
) -> tuple[TileTensorProgram, object, object]:
    batch_size = 1
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hidden_size if mlen % hidden_size == 0 else mlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    x_in = prog.input("X_IN", (batch_size, seq_len, 1, hidden_size))
    scale_in = prog.input("SCALE_IN", (batch_size, seq_len, 1, hidden_size))
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, hidden_size))

    x = prog.tensor("X", (batch_size, seq_len, 1, hidden_size))
    scale = prog.tensor("SCALE", (batch_size, seq_len, 1, hidden_size))
    y = prog.tensor("Y", (batch_size, seq_len, 1, hidden_size))
    work = prog.alloc_fragment(prog._auto_name("WORK"), (batch_size, seq_len, 1, hidden_size))
    sq = prog.alloc_fragment(prog._auto_name("SQ"), (batch_size, seq_len, 1, hidden_size))
    row_sq = prog.alloc_fragment(prog._auto_name("ROW_SQ"), (batch_size, 1, seq_len))
    inv_rms = prog.alloc_fragment(prog._auto_name("INV_RMS"), (batch_size, 1, seq_len))

    recip_hidden = prog.constant(prog._auto_name("recip_hidden"), 1.0 / float(hidden_size), size=seq_len)
    eps_vec = prog.constant(prog._auto_name("eps"), float(eps), size=seq_len)

    prog.copy(x_in, x)
    prog.copy(scale_in, scale)
    prog.copy(x, work)

    work_head = work[0, :, 0:1, :]
    sq_head = sq[0, :, 0:1, :]

    prog.fill(row_sq[0, 0, :], 0.0)
    prog.atomic_mul(work, work, sq)
    prog.row_op(sq_head, op="reduce_sum", out=row_sq[0, 0, :], dim=-1)
    prog.pure_fp_compute(
        row_sq[0, 0, :],
        row_sq[0, 0, :],
        src2=recip_hidden,
        control="mul",
        task_id="rmsnorm.mean_sq",
    )
    prog.pure_fp_compute(
        row_sq[0, 0, :],
        row_sq[0, 0, :],
        src2=eps_vec,
        control="add",
        task_id="rmsnorm.eps",
    )
    prog.fp_sqrt(row_sq[0, 0, :], inv_rms[0, 0, :])
    prog.fp_reci(inv_rms[0, 0, :], inv_rms[0, 0, :])
    prog.row_op(work_head, inv_rms[0, 0, :], "mul", dim=-1)
    prog.atomic_mul(work, scale, work)

    prog.copy(work, y)
    prog.copy(y, out_buf)
    return prog, y, out_buf


def build_rmsnorm_golden(x: torch.Tensor, scale: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms * scale


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    seq_len = 128
    hidden_size = 64
    eps = 1.0e-6

    prog, _, out_buf = build_rmsnorm_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
    )
    x_data = torch.randn(1, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
    scale_data = torch.randn(1, seq_len, 1, hidden_size, dtype=torch.float32) * 0.1 + 1.0
    golden = build_rmsnorm_golden(x_data, scale_data, eps).reshape(seq_len, hidden_size)

    emit_single_output_testbench(
        prog=prog,
        out_buf=out_buf,
        input_tensors={
            "X_IN": x_data,
            "SCALE_IN": scale_data,
            "OUT": torch.zeros(1, seq_len, 1, hidden_size, dtype=torch.float32),
        },
        golden_output=golden,
        asm_name="tile_tensor_kernel_rmsnorm",
        artifact_prefix="tile_tensor_kernel_rmsnorm",
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
