"""TileTensorProgram rewrites of `tilelang_kernels.activations`."""

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


def build_gelu_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
    use_tanh_approximation: bool = True,
) -> tuple[TileTensorProgram, object, object]:
    """Build GELU using the sigmoid approximation `x * sigmoid(1.702 * x)`.

    The tanh approximation is not supported by the current TileTensorProgram
    runtime because there is no tensor-domain tanh primitive.
    """
    if use_tanh_approximation:
        raise NotImplementedError(
            "build_gelu_program currently supports the sigmoid approximation only; "
            "pass use_tanh_approximation=False"
        )

    batch_size = 2
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

    scale_const = prog.constant(prog._auto_name("scale"), 1.702)
    one_const = prog.constant(prog._auto_name("one"), 1.0)
    neg_one_const = prog.constant(prog._auto_name("neg_one"), -1.0)

    for batch_index in range(batch_size):
        scaled = prog.alloc_fragment(prog._auto_name("SCALED"), (1, seq_len, 1, hidden_size))
        prog.copy(x_in[batch_index, :, :, :], x[batch_index, :, :, :])
        prog.copy(x[batch_index, :, :, :], scaled)
        prog.row_op(scaled, scale_const, "mul", dim=-1)
        prog.row_op(scaled, neg_one_const, "mul", dim=-1)
        prog.row_op(scaled, op="exp", dim=-1)
        prog.row_op(scaled, one_const, "add", dim=-1)
        prog.row_op(scaled, op="reci", dim=-1)
        prog.atomic_mul(x[batch_index, :, :, :], scaled, y[batch_index, :, :, :])
        prog.copy(y[batch_index, :, :, :], out_buf[batch_index, :, :, :])
        prog.free_tensor_tile(scaled)
    return prog, y, out_buf


def build_silu_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
) -> tuple[TileTensorProgram, object, object]:
    """Build SiLU (Sigmoid Linear Unit) program."""
    batch_size = 2
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

    one_const = prog.constant(prog._auto_name("one"), 1.0)
    neg_one_const = prog.constant(prog._auto_name("neg_one"), -1.0)

    for batch_index in range(batch_size):
        sigmoid = prog.alloc_fragment(prog._auto_name("SIGMOID"), (1, seq_len, 1, hidden_size))
        prog.copy(x_in[batch_index, :, :, :], x[batch_index, :, :, :])
        prog.copy(x[batch_index, :, :, :], sigmoid)
        prog.row_op(sigmoid, neg_one_const, "mul", dim=-1)
        prog.row_op(sigmoid, op="exp", dim=-1)
        prog.row_op(sigmoid, one_const, "add", dim=-1)
        prog.row_op(sigmoid, op="reci", dim=-1)
        prog.atomic_mul(x[batch_index, :, :, :], sigmoid, y[batch_index, :, :, :])
        prog.copy(y[batch_index, :, :, :], out_buf[batch_index, :, :, :])
        prog.free_tensor_tile(sigmoid)
    return prog, y, out_buf


def build_gelu_golden(x: torch.Tensor, use_tanh_approximation: bool = True) -> torch.Tensor:
    """Golden reference for GELU."""
    if use_tanh_approximation:
        # tanh approximation
        return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    else:
        # sigmoid approximation (what we implement)
        return x * torch.sigmoid(1.702 * x)


def build_silu_golden(x: torch.Tensor) -> torch.Tensor:
    """Golden reference for SiLU."""
    return x * torch.sigmoid(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    batch_size = 2
    seq_len = 128
    hidden_size = 64
    
    # Test GELU
    print("Testing GELU...")
    prog_gelu, _, out_buf_gelu = build_gelu_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        hidden_size=hidden_size,
        use_tanh_approximation=False,
    )
    x_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
    golden_gelu = build_gelu_golden(x_data, use_tanh_approximation=False).reshape(batch_size * seq_len, hidden_size)
    
    emit_single_output_testbench(
        prog=prog_gelu,
        out_buf=out_buf_gelu,
        input_tensors={
            "X_IN": x_data,
            "OUT": torch.zeros(batch_size, seq_len, 1, hidden_size, dtype=torch.float32),
        },
        golden_output=golden_gelu,
        asm_name="tile_tensor_kernel_gelu",
        artifact_prefix="tile_tensor_kernel_gelu",
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
    
    # Test SiLU
    print("Testing SiLU...")
    prog_silu, _, out_buf_silu = build_silu_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )
    golden_silu = build_silu_golden(x_data).reshape(batch_size * seq_len, hidden_size)
    
    emit_single_output_testbench(
        prog=prog_silu,
        out_buf=out_buf_silu,
        input_tensors={
            "X_IN": x_data,
            "OUT": torch.zeros(batch_size, seq_len, 1, hidden_size, dtype=torch.float32),
        },
        golden_output=golden_silu,
        asm_name="tile_tensor_kernel_silu",
        artifact_prefix="tile_tensor_kernel_silu",
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
    
    print("GELU and SiLU implementations created successfully!")
