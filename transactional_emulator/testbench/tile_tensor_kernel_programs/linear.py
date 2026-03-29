"""TileTensorProgram rewrite of `tilelang_kernels.linear`."""

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


def build_linear_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    with_bias: bool = False,
) -> tuple[TileTensorProgram, object, object]:
    batch_size = 1
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=mlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    x_in = prog.input("X_IN", (batch_size, seq_len, 1, in_features))
    w_in = prog.input("W_IN", (batch_size, in_features, 1, out_features))
    if with_bias:
        bias_in = prog.input("BIAS_IN", (batch_size, seq_len, 1, out_features))
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, out_features))

    x = prog.tensor("X", (batch_size, seq_len, 1, in_features))
    w = prog.tensor("W", (batch_size, in_features, 1, out_features))
    if with_bias:
        bias = prog.tensor("BIAS", (batch_size, seq_len, 1, out_features))
    y = prog.tensor("Y", (batch_size, seq_len, 1, out_features))
    prog.copy(x_in, x)
    prog.copy(w_in, w)

    y_group = prog.alloc_fragment(prog._auto_name("Y_GROUP"), (batch_size, seq_len, 1, out_features))
    prog.matmul(x, w, y_group)

    if with_bias:
        prog.copy(bias_in, bias)
        prog.atomic_add(y_group, bias, y_group)

    prog.copy(y_group, y)
    prog.copy(y, out_buf)
    return prog, y, out_buf


def build_linear_golden(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    y = torch.einsum("bshk,bkhd->bshd", x, weight)
    if bias is not None:
        y = y + bias
    return y


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    seq_len = 128
    in_features = 128
    out_features = 128

    prog, _, out_buf = build_linear_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        with_bias=True,
    )
    x_data = torch.randn(1, seq_len, 1, in_features, dtype=torch.float32) * 0.25
    w_data = torch.randn(1, in_features, 1, out_features, dtype=torch.float32) * 0.25
    bias_head = torch.randn(1, 1, 1, out_features, dtype=torch.float32) * 0.1
    bias_data = bias_head.expand(1, seq_len, 1, out_features).contiguous()
    golden = build_linear_golden(x_data, w_data, bias_data).reshape(seq_len, out_features)

    emit_single_output_testbench(
        prog=prog,
        out_buf=out_buf,
        input_tensors={
            "X_IN": x_data,
            "W_IN": w_data,
            "BIAS_IN": bias_data,
            "OUT": torch.zeros(1, seq_len, 1, out_features, dtype=torch.float32),
        },
        golden_output=golden,
        asm_name="tile_tensor_kernel_linear",
        artifact_prefix="tile_tensor_kernel_linear",
        build_dir=_TESTBENCH_DIR / "build",
    )
