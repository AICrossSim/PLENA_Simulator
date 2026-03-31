"""
ATen-style Linear Projection Test

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    Y = ops.linear(prog, X_batch, w_input)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden_Y = ops.linear(X_tensor, W_tensor)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Linear Projection Test  (plena.ops.linear)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    in_features = 128
    out_features = 256
    batch_size = 64  # must be multiple of mlen
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    X = torch.randn(batch_size, in_features)
    W = torch.randn(in_features, out_features)
    print(f"\nInput X: {X.shape}, W: {W.shape}")

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
    golden_Y = ops.linear(X, W)
    print(f"  golden_Y: {golden_Y.shape}")
    print(f"  golden_Y[0,:4]: {golden_Y[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs: activation in VRAM (load_batch), weight stays in HBM
    x_input = prog.input("X", shape=(batch_size, in_features))
    w_input = prog.input("W", shape=(in_features, out_features))
    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: linear_plena() is called with (prog, X_batch, w_input)
    Y = ops.linear(prog, X_batch, w_input)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"X": X, "W": W}
    golden_result = {"original_output": golden_Y}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/in_features
    fp_preload = [0.0, 1e-6, 1.0 / in_features] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
    )

    y_vram_addr = prog._compiler.get_vram_addr(Y.name)

    comparison_params = {
        "start_row_idx": y_vram_addr // mlen,
        "num_rows": (batch_size * out_features) // mlen,
        "num_batches": batch_size,
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
