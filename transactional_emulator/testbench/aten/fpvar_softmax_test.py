"""
ATen-style Online Softmax Test

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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json

# ATen-style imports
from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

# Existing infrastructure (unchanged)
from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Online Softmax Test  (plena.ops.softmax)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    X = torch.randn(mlen, mlen) * 0.5
    print(f"\nInput X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # CPU golden reference (via registry, Backend.CPU)
    # ========================================================================
    print("\n--- CPU Golden Reference ---")
    registry.set_backend(Backend.CPU)
    golden_P = ops.softmax(X, scale=scale)
    print(f"  golden_P: {golden_P.shape}")
    print(f"  golden_P[0,:4]: {golden_P[0, :4].tolist()}")
    print(f"  golden_P[0,:].sum(): {golden_P[0, :].sum():.6f}  (should be ≈1.0)")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # Generates ISA code using PlenaCompiler under the hood.
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Register input tensor
    x_input = prog.input("X", shape=(mlen, mlen))
    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: softmax_plena() is called with (prog, X_batch, scale)
    S = ops.softmax(prog, X_batch, scale=scale)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment (same as original test)
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"X": X}
    golden_result = {"original_output": golden_P}

    # FP SRAM preload: [0]=0.0, [1]=scale, [2]=-inf
    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="fpvar_softmax_test",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
    )

    s_vram_addr = prog._compiler.get_vram_addr(S.name)

    comparison_params = {
        "start_row_idx": s_vram_addr // mlen,
        "num_rows": (mlen * mlen) // mlen,
        "num_batches": mlen,
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
