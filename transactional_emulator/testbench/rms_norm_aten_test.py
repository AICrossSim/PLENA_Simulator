"""
ATen-style RMS Normalization Test

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    result = ops.rms_norm(prog, X_batch)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.rms_norm(X_tensor)
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
    print("ATen-style RMS Normalization Test  (plena.ops.rms_norm)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 128
    batch_size = 4
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    X = torch.randn(batch_size, hidden_size)
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
    golden_out = ops.rms_norm(X)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0,:4].tolist()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Load activation into VRAM, then apply RMS norm in-place
    x_input = prog.input("X", shape=(batch_size, hidden_size))
    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: rms_norm_plena() is called with (prog, X_batch)
    result = ops.rms_norm(prog, X_batch)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"X": X}
    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/hidden_size
    fp_preload = [0.0, 1e-6, 1.0 / hidden_size] + [0.0] * 7

    create_sim_env(
        input_tensor, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir)
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="rms_norm_aten",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
    )

    # RMS norm is in-place: result is at same VRAM location as input
    symbol_table = prog._compiler.symbol_table.table
    x_info = symbol_table[X_batch.name]

    comparison_params = {
        "start_row_idx": x_info.vram_addr // mlen,
        "num_rows": (batch_size * hidden_size) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_info.vram_addr // mlen} (in-place)")
    run_and_assert(build_dir, "rms_norm", mlen=mlen, blen=blen)
