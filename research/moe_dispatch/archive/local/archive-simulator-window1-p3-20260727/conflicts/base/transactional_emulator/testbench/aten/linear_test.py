"""ATen-style Linear Projection Test.

python linear_test.py [--mlen 128] [--blen 16] [--batch-size 8]
"""

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw
from transactional_emulator.testbench.aten.golden import golden_linear
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--out-features", type=int, default=None)
    args = parser.parse_args()

    mlen = args.mlen
    blen = args.blen
    vlen = args.vlen or mlen
    # Total token rows = batch_size * seq_len (unified [batch, seq, hidden] interface).
    # linear is per-row independent, so the rows are flattened to [rows, in_features].
    rows, batch_size, seq_len = resolve_rows(args, default_seq=mlen)
    in_features = args.hidden_size or mlen
    out_features = args.out_features or 2 * mlen

    if in_features % mlen != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by MLEN ({mlen})")
    if out_features % mlen != 0:
        raise ValueError(f"out_features ({out_features}) must be divisible by MLEN ({mlen})")

    build_dir = Path(__file__).parent / "build" / "linear"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        f"ATen-style Linear Projection Test  (mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})"
    )
    print("=" * 80)

    torch.manual_seed(args.seed)
    X = torch.randn(rows, in_features)
    W = torch.randn(in_features, out_features)
    print(f"\nInput X: {X.shape}, W: {W.shape}")

    print("\n--- Hardware-Accurate Golden Reference (MXFP8 + BF16) ---")
    golden_Y = golden_linear(X, W)
    print(f"  golden_Y: {golden_Y.shape}")
    print(f"  golden_Y[0,:4]: {golden_Y[0, :4].tolist()}")

    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    x_input = prog.input("X", shape=(rows, in_features))
    w_input = prog.input("W", shape=(in_features, out_features))
    X_batch = prog.load_batch(x_input, name="X")
    Y = ops.linear(prog, X_batch, w_input)

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    input_tensors = {"X": X, "W": W}
    golden_result = {"original_output": golden_Y}
    fp_preload = [0.0, 1e-6, 1.0 / in_features] + [0.0] * 7

    create_sim_env(input_tensors, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    # Place each tensor at the compiler's actual HBM address. At MLEN>=256 the
    # compiler tile-aligns HBM allocations (gaps between tensors); a contiguous
    # writer would put weights where the prefetch never reads -> zero weights.
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    y_vram_addr = prog._compiler.get_vram_addr(Y.name)

    comparison_params = {
        "start_row_idx": y_vram_addr // mlen,
        "num_rows": (rows * out_features) // mlen,
        "num_batches": rows,
        "elements_per_batch": out_features,
        "row_dim": mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Y location: VRAM row {y_vram_addr // mlen}")
    run_and_assert(build_dir, "linear", mlen=mlen, blen=blen)
