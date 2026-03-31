"""
ATen-style Flash Attention Test

This is the ATen-style version of flash_attention_plena_test.py.
Instead of manually calling PLENAProgram methods inline, we use:

    import plena.ops as ops
    O = ops.flash_attention(prog, Q_batch, k_input, v_input, scale)

The operator is dispatched through the PLENA ATen-style registry to
flash_attention_plena(), which encapsulates the Online Softmax algorithm.

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden_O = ops.flash_attention(Q, K, V, scale)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
import math

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Flash Attention Test  (plena.ops.flash_attention)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    seq_len = 128  # 2 * mlen
    head_dim = 128  # 2 * mlen
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    Q = torch.randn(seq_len, head_dim) * 0.5
    K = torch.randn(seq_len, head_dim) * 0.5
    V = torch.randn(seq_len, head_dim) * 0.5

    print("\nInput shapes:")
    print(f"  Q: {Q.shape}  K: {K.shape}  V: {V.shape}")
    print(f"  scale: {scale:.6f}")

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
    golden_O = ops.flash_attention(Q, K, V, scale)
    print(f"  golden_O: {golden_O.shape}")
    print(f"  golden_O[0,:4]: {golden_O[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # Generates ISA code using PLENAProgram under the hood.
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs:
    #   Q → loaded to VRAM (full sequence), tiled by flash_attention_plena
    #   K, V → remain in HBM, loaded block-by-block during the attention loop
    q_input = prog.input("Q", shape=(seq_len, head_dim))
    k_input = prog.input("K", shape=(seq_len, head_dim))
    v_input = prog.input("V", shape=(seq_len, head_dim))

    Q_batch = prog.load_batch(q_input, name="Q")

    # ATen-style dispatch: flash_attention_plena() is called with
    # (prog, Q_batch, k_input, v_input, scale)
    O = ops.flash_attention(prog, Q_batch, k_input, v_input, scale)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "Q": Q.reshape(1, -1),
        "K": K.reshape(1, -1),
        "V": V.reshape(1, -1),
    }
    golden_result = {"original_output": golden_O}

    # FP SRAM preload: [0]=0.0, [1]=scale, [2]=-inf
    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_aten",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * head_dim) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": head_dim,
        "row_dim": mlen,
        "use_stride_mode": True,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  O location: VRAM row {o_vram_addr // mlen}")
    run_and_assert(build_dir, "flash_attention", mlen=mlen, blen=blen)
