"""GPT-OSS MoE split gate/up clamp emulator smoke.

This is the second device-side step for fixed-routing MoE v0.  GPT-OSS stores
gate/up projection weights interleaved in the reference model, but PLENA's
current vector-scalar min/max applies to whole vector rows or coarse head
chunks, not even/odd lanes.  For exact v0 semantics we split the packed
gate_up projection into separate gate and up projections:

    gate = X @ W_gate
    up   = X @ W_up
    gate = min(gate, 7)
    up   = max(min(up, 7), -7)

The final compared tensor is gate + up, only to put both clamp paths into one
VRAM output for this smoke.  Gated activation and down projection remain out of
scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw
from transactional_emulator.testbench.aten.golden import golden_linear
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _exact_mxfp8_tensor,
)


def run_split_clamp_smoke(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    rows, batch_size, seq_len = resolve_rows(args, default_seq=blen)
    hidden = args.hidden_size or mlen
    intermediate = args.intermediate_size or mlen

    if hidden % mlen != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by MLEN ({mlen})")
    if intermediate % mlen != 0:
        raise ValueError(f"intermediate ({intermediate}) must be divisible by MLEN ({mlen})")

    build_dir = args.build_dir
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        "GPT-OSS MoE split gate/up clamp emulator smoke "
        f"(mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})"
    )
    print("=" * 80)

    x = _exact_mxfp8_tensor((rows, hidden), stride=1)
    w_gate = _exact_mxfp8_tensor((hidden, intermediate), stride=2, offset=1)
    w_up = _exact_mxfp8_tensor((hidden, intermediate), stride=3, offset=2)

    gate = torch.matmul(x.float(), w_gate.float()).to(torch.bfloat16)
    up = torch.matmul(x.float(), w_up.float()).to(torch.bfloat16)
    if not torch.equal(gate, golden_linear(x, w_gate)):
        raise AssertionError("gate projection lost Golden-A equivalence under exact MXFP8 inputs")
    if not torch.equal(up, golden_linear(x, w_up)):
        raise AssertionError("up projection lost Golden-A equivalence under exact MXFP8 inputs")

    golden = (torch.clamp(gate.float(), max=7.0) + torch.clamp(up.float(), min=-7.0, max=7.0)).to(torch.bfloat16)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    w_gate_input = prog.input("W_gate_e0", shape=(hidden, intermediate))
    w_up_input = prog.input("W_up_e0", shape=(hidden, intermediate))
    limit_pos = prog.fp_var("gpt_oss_limit_pos", size=1)
    limit_neg = prog.fp_var("gpt_oss_limit_neg", size=1)

    x_vram = prog.load_batch(x_input, name="X")
    gate_vram = prog.linear_projection(x_vram, w_gate_input, name="gate_e0")
    up_vram = prog.linear_projection(x_vram, w_up_input, name="up_e0")

    active_rows = list(range(rows))
    num_col_blocks = intermediate // mlen
    # Program-level tile_row_*_fp maps multiple rows to consecutive FPRAM
    # offsets. GPT-OSS clamp needs the same scalar limit on every row, so emit
    # one single-row op at a time and pass FPVar objects to keep bounds checks.
    for col_block in range(num_col_blocks):
        for row_idx in active_rows:
            prog.tile_row_min_fp(gate_vram, limit_pos, row_idx=row_idx, tile_col_idx=col_block)
            prog.tile_row_min_fp(up_vram, limit_pos, row_idx=row_idx, tile_col_idx=col_block)
            prog.tile_row_max_fp(up_vram, limit_neg, row_idx=row_idx, tile_col_idx=col_block)
    prog.vram_add(gate_vram, up_vram, num_rows=rows)
    isa = prog.compile()

    input_tensors = {"X": x, "W_gate_e0": w_gate, "W_up_e0": w_up}
    golden_result = {"original_output": golden}
    fp_preload = [7.0, -7.0, 0.0, 1e-6, 1.0 / hidden] + [0.0] * 5

    create_sim_env(input_tensors, isa, golden_result, fp_preload, build_dir=str(build_dir))

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_moe_clamp",
        data=None,
        specified_data_order=["X", "W_gate_e0", "W_up_e0"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    gate_vram_addr = prog._compiler.get_vram_addr(gate_vram.name)
    comparison_params = {
        "start_row_idx": gate_vram_addr // mlen,
        "num_rows": (rows * intermediate) // mlen,
        "num_batches": rows,
        "elements_per_batch": intermediate,
        "row_dim": mlen,
        "atol": 0.0,
        "rtol": 0.0,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa)

    print(f"Generated {len(isa.splitlines())} lines of ISA")
    print(f"gate+up clamped output VRAM row: {gate_vram_addr // mlen}")
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False}

    metrics = run_and_assert(build_dir, "gpt_oss_moe_clamp", mlen=mlen, blen=blen)
    return {"build_dir": str(build_dir), "ran": True, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_moe_clamp",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    run_split_clamp_smoke(args)


if __name__ == "__main__":
    main()
