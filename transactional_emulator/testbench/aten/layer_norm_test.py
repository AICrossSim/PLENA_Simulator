"""ATen-style Layer Normalization Test.

    python layer_norm_test.py [--mlen 128] [--blen 16] [--batch-size 8]

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.layer_norm(prog, X_batch)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.layer_norm(X_tensor)
"""

import argparse
import json
import os
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler
from verification.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from plena_utils import load_precision_from_toml
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    args = parser.parse_args()

    mlen = args.mlen
    blen = args.blen
    batch_size = max(args.batch_size or blen, blen)
    hidden_size = 2 * mlen

    if batch_size % blen != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by BLEN ({blen})")
    if hidden_size % mlen != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by MLEN ({mlen})")

    build_dir = Path(__file__).parent / "build" / "layer_norm"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"ATen-style Layer Normalization Test  (mlen={mlen}, blen={blen}, batch={batch_size})")
    print("=" * 80)

    torch.manual_seed(args.seed)

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
    golden_out = ops.layer_norm(X)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")
    print(f"  golden_out[0,:].mean(): {golden_out[0, :].mean():.6f}  (should be ~0.0)")
    print(f"  golden_out[0,:].std():  {golden_out[0, :].std():.6f}   (should be ~1.0)")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    # Load activation into VRAM, then apply Layer norm in-place
    x_input = prog.input("X", shape=(batch_size, hidden_size))
    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: layer_norm_plena() is called with (prog, X_batch)
    result = ops.layer_norm(prog, X_batch)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    input_tensor = {"X": X}
    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/hidden_size
    fp_preload = [0.0, 1e-6, 1.0 / hidden_size] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    toml_path = os.environ.get("PLENA_SETTINGS_TOML", str(Path(__file__).parents[3] / "plena_settings.toml"))
    precision_settings = load_precision_from_toml(toml_path, mode="TRANSACTIONAL")

    create_mem_for_sim(
        precision_settings=precision_settings,
        data_size=256,
        mode="behave_sim",
        asm="layer_norm_aten",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
    )

    # Layer norm is in-place: result is at same VRAM location as input
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
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
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (in-place)")
    run_and_assert(build_dir, "layer_norm", mlen=mlen, blen=blen)
