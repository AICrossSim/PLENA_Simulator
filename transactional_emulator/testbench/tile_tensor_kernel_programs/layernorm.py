"""TileTensorProgram rewrite of `tilelang_kernels.layernorm`."""

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


def build_layernorm_program(
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
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, hidden_size))

    x = prog.tensor("X", (batch_size, seq_len, 1, hidden_size))
    y = prog.tensor("Y", (batch_size, seq_len, 1, hidden_size))
    centered = prog.alloc_fragment(prog._auto_name("CENTERED"), (batch_size, seq_len, 1, hidden_size))
    sq = prog.alloc_fragment(prog._auto_name("SQ"), (batch_size, seq_len, 1, hidden_size))
    mean = prog.alloc_fragment(prog._auto_name("MEAN"), (batch_size, 1, seq_len))
    var = prog.alloc_fragment(prog._auto_name("VAR"), (batch_size, 1, seq_len))
    inv_std = prog.alloc_fragment(prog._auto_name("INV_STD"), (batch_size, 1, seq_len))

    recip_hidden = prog.constant(prog._auto_name("recip_hidden"), 1.0 / float(hidden_size), size=seq_len)
    eps_vec = prog.constant(prog._auto_name("eps"), float(eps), size=seq_len)

    prog.copy(x_in, x)
    prog.copy(x, centered)

    x_head = centered[0, :, 0:1, :]
    sq_head = sq[0, :, 0:1, :]

    prog.fill(mean[0, 0, :], 0.0)
    prog.fill(var[0, 0, :], 0.0)
    prog.row_op(x_head, op="reduce_sum", out=mean[0, 0, :], dim=-1)
    prog.pure_fp_compute(
        mean[0, 0, :],
        mean[0, 0, :],
        src2=recip_hidden,
        control="mul",
        task_id="layernorm.mean_scale",
    )
    prog.row_op(x_head, mean[0, 0, :], "sub", dim=-1)

    prog.atomic_mul(centered, centered, sq)
    prog.row_op(sq_head, op="reduce_sum", out=var[0, 0, :], dim=-1)
    prog.pure_fp_compute(
        var[0, 0, :],
        var[0, 0, :],
        src2=recip_hidden,
        control="mul",
        task_id="layernorm.var_scale",
    )
    prog.pure_fp_compute(
        var[0, 0, :],
        var[0, 0, :],
        src2=eps_vec,
        control="add",
        task_id="layernorm.var_eps",
    )
    prog.fp_sqrt(var[0, 0, :], inv_std[0, 0, :])
    prog.fp_reci(inv_std[0, 0, :], inv_std[0, 0, :])
    prog.row_op(x_head, inv_std[0, 0, :], "mul", dim=-1)

    prog.copy(centered, y)
    prog.copy(y, out_buf)
    return prog, y, out_buf


def build_layernorm_golden(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    seq_len = 128
    hidden_size = 64
    eps = 1.0e-6

    prog, _, out_buf = build_layernorm_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
    )
    x_data = torch.randn(1, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
    golden = build_layernorm_golden(x_data, eps).reshape(seq_len, hidden_size)

    emit_single_output_testbench(
        prog=prog,
        out_buf=out_buf,
        input_tensors={
            "X_IN": x_data,
            "OUT": torch.zeros(1, seq_len, 1, hidden_size, dtype=torch.float32),
        },
        golden_output=golden,
        asm_name="tile_tensor_kernel_layernorm",
        artifact_prefix="tile_tensor_kernel_layernorm",
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
