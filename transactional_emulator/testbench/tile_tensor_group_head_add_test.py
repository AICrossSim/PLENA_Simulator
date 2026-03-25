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
    hlen: int,
    seq_len: int,
    head_count: int,
) -> tuple[TileTensorProgram, object, object]:
    batch_size = 1
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    a_in = prog.input("A_IN", (batch_size, seq_len, head_count, hlen))
    b_in = prog.input("B_IN", (batch_size, seq_len, head_count, hlen))
    out_buf = prog.input("OUT", (batch_size, seq_len, head_count, hlen))

    a = prog.tensor("A", (batch_size, seq_len, head_count, hlen))
    b = prog.tensor("B", (batch_size, seq_len, head_count, hlen))
    prog.copy(a_in, a)
    prog.copy(b_in, b)

    prog.atomic_add(a, b, b)
    prog.copy(b, out_buf)
    return prog, b, out_buf


def build_golden(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    hlen = 16
    seq_len = 128
    head_count = 8

    prog, out, out_buf = build_program(
        mlen=mlen,
        blen=blen,
        hlen=hlen,
        seq_len=seq_len,
        head_count=head_count,
    )
    comparison_params = stage_input_tensor_for_stride_compare(out_buf)
    staging_isa = comparison_params.pop("staging_isa", "")
    gen_code = prog.compile()
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    a_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.25
    b_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.25
    golden_y = build_golden(a_data, b_data)

    input_feed, input_order = build_input_feed(
        prog,
        {
            "A_IN": a_data,
            "B_IN": b_data,
            "OUT": torch.zeros(1, seq_len, head_count, hlen, dtype=torch.float32),
        },
    )
    golden_result = {"original_output": golden_y.reshape(seq_len, head_count * hlen)}

    create_sim_env(
        input_tensor=input_feed,
        generated_code=gen_code,
        golden_result=golden_result,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="tile_tensor_group_head_add_test",
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "tile_tensor_group_head_add_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    if np is not None:
        np.save(
            build_dir / "tile_tensor_group_head_add_golden_fp32.npy",
            golden_y.reshape(seq_len, head_count * hlen).detach().cpu().numpy().astype(np.float32),
        )

    prog.write_operation_report(build_dir / "tile_tensor_group_head_add_ops.txt")
    prog.write_tile_distribution_report(build_dir / "tile_tensor_group_head_add_tile_report.txt")
