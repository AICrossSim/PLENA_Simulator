"""ATen-style RMS Normalization Test.

    python rms_norm_test.py [--mlen 128] [--blen 16] [--batch-size 8]

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.rms_norm(prog, X_batch)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.rms_norm(X_tensor)
"""

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import golden_rms_norm
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
    # Total token rows = batch_size * seq_len (unified [batch, seq, hidden] interface).
    # rms_norm is per-row independent, so the rows are flattened to [rows, hidden_size].
    # default_seq=blen preserves the prior default row count (max(batch_size or blen, blen)).
    rows, batch_size, seq_len = resolve_rows(args, default_seq=blen)
    hidden_size = args.hidden_size or mlen

    if hidden_size % mlen != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by MLEN ({mlen})")

    build_dir = Path(__file__).parent / "build" / "rms_norm"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"ATen-style RMS Normalization Test  (mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})")
    print("=" * 80)

    torch.manual_seed(args.seed)

    # ========================================================================
    # Test data
    # ========================================================================
    X = torch.randn(rows, hidden_size)
    print(f"\nInput X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")

    # ========================================================================
    # Hardware-accurate golden reference
    # ========================================================================
    eps = 1e-6
    print("\n--- Hardware-Accurate Golden Reference ---")
    golden_out = golden_rms_norm(X, eps)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    # Load activation into VRAM, then apply RMS norm in-place
    x_input = prog.input("X", shape=(rows, hidden_size))
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
    input_tensors = {"X": X}
    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/hidden_size
    fp_preload = [0.0, 1e-6, 1.0 / hidden_size] + [0.0] * 7

    create_sim_env(input_tensors, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="rms_norm_aten",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir,
        input_tensors=input_tensors,
    )

    # RMS norm is in-place: result is at same VRAM location as input
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (rows * hidden_size) // mlen,
        "num_batches": rows,
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
    run_and_assert(build_dir, "rms_norm", mlen=mlen, blen=blen)
