import json
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

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.sim_env_utils import create_mem_for_sim
from tile_tensor_program import TileTensorProgram
from tile_tensor_test_helper import build_input_feed, stage_input_tensor_for_stride_compare
from transactional_emulator.tools.create_sim_env import create_sim_env


def build_program(
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
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

    x_in = prog.input("X_IN", (batch_size, seq_len, 1, hidden_size))
    w_in = prog.input("W_IN", (batch_size, hidden_size, 1, hidden_size))
    bias_in = prog.input("BIAS_IN", (batch_size, seq_len, 1, hidden_size))
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, hidden_size))

    x = prog.tensor("X", (batch_size, seq_len, 1, hidden_size))
    w = prog.tensor("W", (batch_size, hidden_size, 1, hidden_size))
    bias = prog.tensor("BIAS", (batch_size, seq_len, 1, hidden_size))
    y = prog.tensor("Y", (batch_size, seq_len, 1, hidden_size))
    prog.copy(x_in, x)
    prog.copy(w_in, w)
    prog.copy(bias_in, bias)

    y_group = prog.alloc_fragment(prog._auto_name("Y_GROUP"), (batch_size, seq_len, 1, hidden_size))
    prog.matmul(x, w, y_group)
    prog.atomic_add(y_group, bias, y_group)
    prog.copy(y_group, y)

    prog.copy(y, out_buf)
    return prog, y, out_buf


def build_golden(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = torch.einsum("bshk,bkhd->bshd", x, w)
    return y + bias


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    seq_len = 128
    hidden_size = 128

    prog, out, out_buf = build_program(
        mlen=mlen,
        blen=blen,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )
    comparison_params = stage_input_tensor_for_stride_compare(out_buf)
    staging_isa = comparison_params.pop("staging_isa", "")
    gen_code = prog.compile()
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    x_data = torch.randn(1, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
    w_data = torch.randn(1, hidden_size, 1, hidden_size, dtype=torch.float32) * 0.25
    bias_head = torch.randn(1, 1, 1, hidden_size, dtype=torch.float32) * 0.1
    bias_data = bias_head.expand(1, seq_len, 1, hidden_size).contiguous()
    golden_y = build_golden(x_data, w_data, bias_data)

    input_feed, input_order = build_input_feed(
        prog,
        {
            "X_IN": x_data,
            "W_IN": w_data,
            "BIAS_IN": bias_data,
            "OUT": torch.zeros(1, seq_len, 1, hidden_size, dtype=torch.float32),
        },
    )
    golden_result = {"original_output": golden_y.reshape(seq_len, hidden_size)}

    create_sim_env(
        input_tensor=input_feed,
        generated_code=gen_code,
        golden_result=golden_result,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="tile_tensor_group_head_linear_test",
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "tile_tensor_group_head_linear_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    if np is not None:
        np.save(
            build_dir / "tile_tensor_group_head_linear_golden_fp32.npy",
            golden_y.reshape(seq_len, hidden_size).detach().cpu().numpy().astype(np.float32),
        )

    prog.write_operation_report(build_dir / "tile_tensor_group_head_linear_ops.txt")
    prog.write_tile_distribution_report(build_dir / "tile_tensor_group_head_linear_tile_report.txt")
