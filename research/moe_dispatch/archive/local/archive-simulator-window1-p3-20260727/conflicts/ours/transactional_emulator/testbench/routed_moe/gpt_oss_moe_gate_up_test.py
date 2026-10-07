"""GPT-OSS MoE gate-up projection emulator smoke.

This is the first device-side step for fixed-routing MoE v0.  It only runs the
selected expert's gate_up matmul:

    X[tokens, hidden] @ W_gate_up_e[hidden, 2 * intermediate]

Bias, clamp, gated activation, down projection, and combine deliberately stay
out of this smoke.  The input values are chosen to be exactly representable by
the current PLENA MXFP8 HBM format, so the hardware-aware golden equals the
pure BF16 mathematical golden for this stage.
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


def run_gate_up_matmul_smoke(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    rows, batch_size, seq_len = resolve_rows(args, default_seq=blen)
    hidden = args.hidden_size or mlen
    out_features = args.out_features or 2 * mlen

    if hidden % mlen != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by MLEN ({mlen})")
    if out_features % mlen != 0:
        raise ValueError(f"out_features ({out_features}) must be divisible by MLEN ({mlen})")

    build_dir = args.build_dir
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        "GPT-OSS MoE gate_up matmul emulator smoke "
        f"(mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})"
    )
    print("=" * 80)

    x = _exact_mxfp8_tensor((rows, hidden), stride=1)
    w_gate_up_e = _exact_mxfp8_tensor((hidden, out_features), stride=2, offset=1)

    pure_golden = torch.matmul(x.float(), w_gate_up_e.float()).to(torch.bfloat16)
    hardware_golden = golden_linear(x, w_gate_up_e)
    if not torch.equal(pure_golden, hardware_golden):
        max_diff = (pure_golden.float() - hardware_golden.float()).abs().max().item()
        raise AssertionError(f"Exact-value gate_up smoke lost Golden-A equivalence; max diff={max_diff}")

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    w_input = prog.input("W_gate_up_e0", shape=(hidden, out_features))
    x_vram = prog.load_batch(x_input, name="X")
    gate_up = prog.linear_projection(x_vram, w_input, name="gate_up_e0")
    isa = prog.compile()

    input_tensors = {"X": x, "W_gate_up_e0": w_gate_up_e}
    golden_result = {"original_output": pure_golden}
    fp_preload = [0.0, 1e-6, 1.0 / hidden] + [0.0] * 7

    create_sim_env(input_tensors, isa, golden_result, fp_preload, build_dir=str(build_dir))

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_moe_gate_up",
        data=None,
        specified_data_order=["X", "W_gate_up_e0"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    gate_up_vram_addr = prog._compiler.get_vram_addr(gate_up.name)
    comparison_params = {
        "start_row_idx": gate_up_vram_addr // mlen,
        "num_rows": (rows * out_features) // mlen,
        "num_batches": rows,
        "elements_per_batch": out_features,
        "row_dim": mlen,
        "atol": 0.0,
        "rtol": 0.0,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa)

    print(f"Generated {len(isa.splitlines())} lines of ISA")
    print(f"gate_up output VRAM row: {gate_up_vram_addr // mlen}")
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False}

    metrics = run_and_assert(build_dir, "gpt_oss_moe_gate_up", mlen=mlen, blen=blen)
    return {"build_dir": str(build_dir), "ran": True, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--out-features", type=int, default=None)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_moe_gate_up",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    run_gate_up_matmul_smoke(args)


if __name__ == "__main__":
    main()
