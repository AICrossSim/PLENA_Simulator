"""TileTensorProgram rewrites of `tilelang_kernels.elementwise`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram
from tile_tensor_kernel_programs.testbench_runner import emit_single_output_testbench


def build_modulate_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
) -> tuple[TileTensorProgram, object, object]:
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
    scale_in = prog.input("SCALE_IN", (batch_size, seq_len, 1, hidden_size))
    shift_in = prog.input("SHIFT_IN", (batch_size, seq_len, 1, hidden_size))
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, hidden_size))

    x = prog.tensor("X", (batch_size, seq_len, 1, hidden_size))
    scale = prog.tensor("SCALE", (batch_size, seq_len, 1, hidden_size))
    shift = prog.tensor("SHIFT", (batch_size, seq_len, 1, hidden_size))
    out = prog.tensor("OUT_T", (batch_size, seq_len, 1, hidden_size))
    one = prog.constant(prog._auto_name("one"), 1.0)

    for batch_index in range(batch_size):
        scale_plus_one = prog.alloc_fragment(prog._auto_name("SCALE_PLUS_ONE"), (1, seq_len, 1, hidden_size))
        tmp = prog.alloc_fragment(prog._auto_name("TMP"), (1, seq_len, 1, hidden_size))
        prog.copy(x_in[batch_index, :, :, :], x[batch_index, :, :, :])
        prog.copy(scale_in[batch_index, :, :, :], scale[batch_index, :, :, :])
        prog.copy(shift_in[batch_index, :, :, :], shift[batch_index, :, :, :])
        prog.copy(scale[batch_index, :, :, :], scale_plus_one)
        prog.row_op(scale_plus_one, one, "add", dim=-1)
        prog.atomic_mul(scale_plus_one, x[batch_index, :, :, :], tmp)
        prog.atomic_add(tmp, shift[batch_index, :, :, :], tmp)
        prog.copy(tmp, out[batch_index, :, :, :])
        prog.copy(out[batch_index, :, :, :], out_buf[batch_index, :, :, :])
        prog.free_tensor_tile(scale_plus_one)
        prog.free_tensor_tile(tmp)
    return prog, out, out_buf


def build_modulate_golden(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return (1.0 + scale) * x + shift


def build_residual_gate_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
) -> tuple[TileTensorProgram, object, object]:
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
    gate_in = prog.input("GATE_IN", (batch_size, seq_len, 1, hidden_size))
    y_in = prog.input("Y_IN", (batch_size, seq_len, 1, hidden_size))
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, hidden_size))

    x = prog.tensor("X", (batch_size, seq_len, 1, hidden_size))
    gate = prog.tensor("GATE", (batch_size, seq_len, 1, hidden_size))
    y = prog.tensor("Y", (batch_size, seq_len, 1, hidden_size))
    out = prog.tensor("OUT_T", (batch_size, seq_len, 1, hidden_size))
    for batch_index in range(batch_size):
        gated = prog.alloc_fragment(prog._auto_name("GATED"), (1, seq_len, 1, hidden_size))
        prog.copy(x_in[batch_index, :, :, :], x[batch_index, :, :, :])
        prog.copy(gate_in[batch_index, :, :, :], gate[batch_index, :, :, :])
        prog.copy(y_in[batch_index, :, :, :], y[batch_index, :, :, :])
        prog.atomic_mul(gate[batch_index, :, :, :], y[batch_index, :, :, :], gated)
        prog.atomic_add(x[batch_index, :, :, :], gated, gated)
        prog.copy(gated, out[batch_index, :, :, :])
        prog.copy(out[batch_index, :, :, :], out_buf[batch_index, :, :, :])
        prog.free_tensor_tile(gated)
    return prog, out, out_buf


def build_residual_gate_golden(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + gate * y


def _normalized_kind() -> str:
    kind = os.getenv("TILE_TENSOR_ELEMENTWISE_KIND", "modulate").strip().lower()
    aliases = {
        "modulate": "modulate",
        "adaln_modulate": "modulate",
        "ada_ln_modulate": "modulate",
        "residual_gate": "residual_gate",
        "residual-gate": "residual_gate",
        "gate": "residual_gate",
    }
    if kind not in aliases:
        raise ValueError(
            "Unsupported TILE_TENSOR_ELEMENTWISE_KIND="
            f"{kind!r}; expected one of {sorted(set(aliases.keys()))}"
        )
    return aliases[kind]


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    batch_size = 2
    seq_len = 128
    hidden_size = 64
    kind = _normalized_kind()

    if kind == "modulate":
        prog, _, out_buf = build_modulate_program(
            mlen=mlen,
            blen=blen,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )
        x_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
        scale_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.1
        shift_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.05
        golden = build_modulate_golden(x_data, scale_data, shift_data).reshape(batch_size * seq_len, hidden_size)
        input_tensors = {
            "X_IN": x_data,
            "SCALE_IN": scale_data,
            "SHIFT_IN": shift_data,
            "OUT": torch.zeros(batch_size, seq_len, 1, hidden_size, dtype=torch.float32),
        }
        asm_name = "tile_tensor_kernel_modulate"
        artifact_prefix = "tile_tensor_kernel_modulate"
    else:
        prog, _, out_buf = build_residual_gate_program(
            mlen=mlen,
            blen=blen,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )
        x_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
        gate_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.1
        y_data = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
        golden = build_residual_gate_golden(x_data, gate_data, y_data).reshape(batch_size * seq_len, hidden_size)
        input_tensors = {
            "X_IN": x_data,
            "GATE_IN": gate_data,
            "Y_IN": y_data,
            "OUT": torch.zeros(batch_size, seq_len, 1, hidden_size, dtype=torch.float32),
        }
        asm_name = "tile_tensor_kernel_residual_gate"
        artifact_prefix = "tile_tensor_kernel_residual_gate"

    emit_single_output_testbench(
        prog=prog,
        out_buf=out_buf,
        input_tensors=input_tensors,
        golden_output=golden,
        asm_name=asm_name,
        artifact_prefix=artifact_prefix,
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
