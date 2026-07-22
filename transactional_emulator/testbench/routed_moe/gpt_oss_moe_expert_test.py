"""GPT-OSS MoE single-expert emulator smoke.

This extends the activation smoke by adding the expert down projection:

    gate = min(X @ W_gate, 7)
    up   = max(min(X @ W_up, 7), -7)
    hidden = (up + 1) * gate * sigmoid(1.702 * gate)
    out = hidden @ W_down

Routing and multi-expert combine remain out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw
from transactional_emulator.testbench.aten.golden import golden_linear, quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _activation_golden,
    _exact_mxfp8_tensor,
)


def _emit_gpt_oss_activation(
    prog: PlenaCompiler,
    gate_vram,
    up_vram,
    sigmoid_vram,
    rows: int,
    mlen: int,
):
    active_rows = list(range(rows))
    num_col_blocks = sigmoid_vram.shape[1] // mlen
    _zero = prog.fp_var("zero", size=1)
    limit_pos = prog.fp_var("gpt_oss_limit_pos", size=rows)
    limit_neg = prog.fp_var("gpt_oss_limit_neg", size=rows)
    one = prog.fp_var("one", size=rows)
    neg_alpha = prog.fp_var("neg_alpha", size=rows)

    for col_block in range(num_col_blocks):
        prog.tile_row_min_fp(gate_vram, limit_pos, rows=active_rows, tile_col_idx=col_block)
        prog.tile_row_min_fp(up_vram, limit_pos, rows=active_rows, tile_col_idx=col_block)
        prog.tile_row_max_fp(up_vram, limit_neg, rows=active_rows, tile_col_idx=col_block)

    prog.vram_fill_zero(sigmoid_vram, rows=active_rows)
    prog.vram_add(sigmoid_vram, gate_vram, num_rows=rows)

    for col_block in range(num_col_blocks):
        prog.tile_row_mul_fp(sigmoid_vram, neg_alpha, rows=active_rows, tile_col_idx=col_block)
        prog.tile_row_exp(sigmoid_vram, rows=active_rows, tile_col_idx=col_block)
        prog.tile_row_add_fp(sigmoid_vram, one, rows=active_rows, tile_col_idx=col_block)
        prog.tile_row_reci(sigmoid_vram, rows=active_rows, tile_col_idx=col_block)
    prog.vram_mul(gate_vram, sigmoid_vram, num_rows=rows)

    for col_block in range(num_col_blocks):
        prog.tile_row_add_fp(up_vram, one, rows=active_rows, tile_col_idx=col_block)
    prog.vram_mul(up_vram, gate_vram, num_rows=rows)

    fp_preload = [0.0] + [7.0] * rows + [-7.0] * rows + [1.0] * rows + [-1.702] * rows + [0.0] * 8
    return fp_preload


def run_single_expert_smoke(args: argparse.Namespace) -> dict:
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
        "GPT-OSS MoE single-expert emulator smoke "
        f"(mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})"
    )
    print("=" * 80)

    x = _exact_mxfp8_tensor((rows, hidden), stride=1)
    w_gate = _exact_mxfp8_tensor((hidden, intermediate), stride=2, offset=1)
    w_up = _exact_mxfp8_tensor((hidden, intermediate), stride=3, offset=2)
    w_down = _exact_mxfp8_tensor((intermediate, hidden), stride=4, offset=3)

    gate = torch.matmul(x.float(), w_gate.float()).to(torch.bfloat16)
    up = torch.matmul(x.float(), w_up.float()).to(torch.bfloat16)
    if not torch.equal(gate, golden_linear(x, w_gate)):
        raise AssertionError("gate projection lost Golden-A equivalence under exact MXFP8 inputs")
    if not torch.equal(up, golden_linear(x, w_up)):
        raise AssertionError("up projection lost Golden-A equivalence under exact MXFP8 inputs")

    hidden_golden = _activation_golden(gate, up)
    # The down projection consumes BF16 hidden values already resident in VRAM.
    # Unlike HBM-loaded inputs, this activation must not be MX-quantized again;
    # only W_down follows the matrix-weight MX path.
    golden = torch.matmul(hidden_golden.float(), quantize_to_mxfp(w_down).float()).to(torch.bfloat16)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    w_gate_input = prog.input("W_gate_e0", shape=(hidden, intermediate))
    w_up_input = prog.input("W_up_e0", shape=(hidden, intermediate))
    w_down_input = prog.input("W_down_e0", shape=(intermediate, hidden))

    x_vram = prog.load_batch(x_input, name="X")
    gate_vram = prog.linear_projection(x_vram, w_gate_input, name="gate_e0")
    up_vram = prog.linear_projection(x_vram, w_up_input, name="up_e0")
    sigmoid_vram = prog.alloc(
        "gate_sigmoid",
        rows=rows,
        cols=intermediate,
        physical_shape=(max(mlen, math.ceil(rows / mlen) * mlen), intermediate),
        strict=False,
    )

    fp_preload = _emit_gpt_oss_activation(prog, gate_vram, up_vram, sigmoid_vram, rows, mlen)
    output_vram = prog.linear_projection(up_vram, w_down_input, name="expert_e0")
    isa = prog.compile()

    input_tensors = {"X": x, "W_gate_e0": w_gate, "W_up_e0": w_up, "W_down_e0": w_down}
    golden_result = {"original_output": golden}

    create_sim_env(input_tensors, isa, golden_result, fp_preload, build_dir=str(build_dir))

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_moe_expert",
        data=None,
        specified_data_order=["X", "W_gate_e0", "W_up_e0", "W_down_e0"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    output_vram_addr = prog._compiler.get_vram_addr(output_vram.name)
    comparison_params = {
        "start_row_idx": output_vram_addr // mlen,
        "num_rows": (rows * hidden) // mlen,
        "num_batches": rows,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "atol": 0.0,
        "rtol": 0.0,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa)

    print(f"Generated {len(isa.splitlines())} lines of ISA")
    print(f"expert output VRAM row: {output_vram_addr // mlen}")
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False}

    metrics = run_and_assert(build_dir, "gpt_oss_moe_expert", mlen=mlen, blen=blen)
    return {"build_dir": str(build_dir), "ran": True, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_moe_expert",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    run_single_expert_smoke(args)


if __name__ == "__main__":
    main()
