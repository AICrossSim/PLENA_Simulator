"""ATen-style Learned Positional Embedding Add Test.

    python embedding_add_test.py [--mlen 128] [--blen 16] [--batch-size 8] [--seq-len 4]

SigLIP vision encoder step:
    embeddings = patch_embeds + position_embedding(position_ids)

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.embedding_add(prog, patch_var, pos_var)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.embedding_add(X, pos_weight)
"""

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import golden_embedding_add
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
    hidden_size = args.hidden_size or mlen
    # Total token rows = batch_size * seq_len (unified [batch, seq, hidden] interface).
    # embedding_add is per-row independent, so the rows are flattened to [rows, hidden].
    rows, batch_size, seq_len = resolve_rows(args, default_seq=max(4, mlen // 16))

    if hidden_size % mlen != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by MLEN ({mlen})")

    build_dir = Path(__file__).parent / "build" / "embedding_add"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"ATen-style Embedding Add Test  (mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows}, hidden={hidden_size})")
    print("=" * 80)

    torch.manual_seed(args.seed)

    # ========================================================================
    # Test data: patch embeddings + position embedding table
    # ========================================================================
    X = torch.randn(rows, hidden_size)  # patch embeddings
    pos_weight = torch.randn(rows, hidden_size)  # learned position embeddings

    print(f"\nInput X:         {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")
    print(f"pos_weight:      {pos_weight.shape}, range [{pos_weight.min():.3f}, {pos_weight.max():.3f}]")

    # ========================================================================
    # Hardware-accurate golden reference
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference ---")
    golden_out = golden_embedding_add(X, pos_weight)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    x_input = prog.input("X", shape=(rows, hidden_size))
    pe_input = prog.input("POS", shape=(rows, hidden_size))

    X_batch = prog.load_batch(x_input, name="X")
    PE_batch = prog.load_batch(pe_input, name="POS")

    result = ops.embedding_add(prog, X_batch, PE_batch)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    input_tensors = {"X": X, "POS": pos_weight}
    golden_result = {"original_output": golden_out}

    create_sim_env(input_tensors, gen_code, golden_result, [], build_dir=str(build_dir))

    # Place each tensor at the compiler's actual HBM address. At MLEN>=256 the
    # compiler tile-aligns HBM allocations (gaps between tensors); a contiguous
    # writer would put POS where the prefetch never reads -> zero/garbage.
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="embedding_add_aten",
        data=None,
        specified_data_order=["X", "POS"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    # embedding_add is in-place: result is at same VRAM location as X
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
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (in-place on X)")
    run_and_assert(build_dir, "embedding_add", mlen=mlen, blen=blen)
