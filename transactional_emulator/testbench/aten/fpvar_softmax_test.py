"""ATen-style Online Softmax Test.

    python fpvar_softmax_test.py [--mlen 128] [--blen 16]

This is the ATen-style version of fpvar_softmax_test.py.
Instead of manually calling PlenaCompiler methods inline, we use:

    import compiler.aten.ops as ops
    result_var = ops.softmax(prog, X_batch, scale=1.0)

The operator is dispatched through the PLENA ATen-style registry to
softmax_plena(), which encapsulates the online softmax algorithm.

To compare with the CPU (golden) reference:
    registry.set_backend(Backend.CPU)
    golden = ops.softmax(X_tensor, scale=1.0)
"""

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import golden_softmax
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    args = parser.parse_args()

    mlen = args.mlen
    blen = args.blen
    # softmax operates on a [rows, mlen] attention score matrix; rows = batch_size * seq_len.
    # The default (batch_size=1, seq_len=mlen) reproduces the prior [mlen, mlen] score matrix.
    rows, batch_size, seq_len = resolve_rows(args, default_seq=mlen)
    scale = 1.0 / math.sqrt(mlen)

    build_dir = Path(__file__).parent / "build" / "fpvar_softmax"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        f"ATen-style Online Softmax Test  (mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows}, scale={scale:.4f})"
    )
    print("=" * 80)

    torch.manual_seed(args.seed)

    # ========================================================================
    # Test data
    # ========================================================================
    X = torch.randn(rows, mlen) * 0.5
    print(f"\nInput X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")

    # ========================================================================
    # Hardware-accurate golden reference
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference ---")
    golden_P = golden_softmax(X, scale)
    print(f"  golden_P: {golden_P.shape}")
    print(f"  golden_P[0,:4]: {golden_P[0, :4].tolist()}")
    print(f"  golden_P[0,:].sum(): {golden_P[0, :].sum():.6f}  (should be ~1.0)")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # Generates ISA code using PlenaCompiler under the hood.
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    # Register input tensor
    x_input = prog.input("X", shape=(rows, mlen))
    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: softmax_plena() is called with (prog, X_batch, scale)
    S = ops.softmax(prog, X_batch, scale=scale)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    input_tensors = {"X": X}
    golden_result = {"original_output": golden_P}

    # FP SRAM preload: [0]=0.0, [1]=scale, [2]=-inf
    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 7

    create_sim_env(input_tensors, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    # Place each tensor at the compiler's actual HBM address (tile-aligned at MLEN>=256).
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="fpvar_softmax_test",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    s_vram_addr = prog._compiler.get_vram_addr(S.name)

    comparison_params = {
        "start_row_idx": s_vram_addr // mlen,
        "num_rows": (rows * mlen) // mlen,
        "num_batches": rows,
        "elements_per_batch": mlen,
        "row_dim": mlen,
        "use_stride_mode": True,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  S location: VRAM row {s_vram_addr // mlen}")
    run_and_assert(build_dir, "softmax", mlen=mlen, blen=blen)
