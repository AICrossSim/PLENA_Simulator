"""
ATen-style Learned Positional Embedding Add Test

SigLIP vision encoder step:
    embeddings = patch_embeds + position_embedding(position_ids)

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.embedding_add(prog, patch_var, pos_var)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.embedding_add(X, pos_weight)
"""

import sys
from pathlib import Path


import torch
import json

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Embedding Add Test  (plena.ops.embedding_add)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 128  # vision encoder hidden dim
    seq_len = 4  # number of patches (batch dimension)
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Test data: patch embeddings + position embedding table
    # ========================================================================
    X = torch.randn(seq_len, hidden_size)  # patch embeddings
    pos_weight = torch.randn(seq_len, hidden_size)  # learned position embeddings

    print(f"\nInput X:         {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")
    print(f"pos_weight:      {pos_weight.shape}, range [{pos_weight.min():.3f}, {pos_weight.max():.3f}]")

    # ========================================================================
    # CPU golden reference
    # ========================================================================
    print("\n--- CPU Golden Reference ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_out = ops.embedding_add(X, pos_weight)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    x_input = prog.input("X", shape=(seq_len, hidden_size))
    pe_input = prog.input("POS", shape=(seq_len, hidden_size))

    X_batch = prog.load_batch(x_input, name="X")
    PE_batch = prog.load_batch(pe_input, name="POS")

    result = ops.embedding_add(prog, X_batch, PE_batch)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build" / "embedding_add"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"X": X, "POS": pos_weight}
    golden_result = {"original_output": golden_out}

    create_sim_env(input_tensor, gen_code, golden_result, [], build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="embedding_add_aten",
        data=None,
        specified_data_order=["X", "POS"],
        build_path=build_dir,
    )

    # embedding_add is in-place: result is at same VRAM location as X
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (seq_len * hidden_size) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (in-place on X)")
    run_and_assert(build_dir, "embedding_add", mlen=mlen, blen=blen)
