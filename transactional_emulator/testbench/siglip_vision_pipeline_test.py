"""
SigLIP Vision Pipeline Test

End-to-end pipeline chaining four ops in a single PLENAProgram:

    patch_embeds + pos_weight  →  embedding_add
    layer_norm                 →  layer_norm
    flash_attention            →  flash_attention  (Q = X_batch)
    ffn                        →  ffn

All ops run in-place on a single VRAMMatrixVar (X_batch).

FPRAM slot layout (avoids conflict between flash_attention slot 2 and layer_norm):
    slot 0 = 0.0
    slot 1 = attn_scale
    slot 2 = -inf          (flash_attention: init_online_softmax running max)
    slot 3 = eps           (layer_norm: uses eps_offset=3)
    slot 4 = 1/hidden_size (layer_norm: uses reci_hid_offset=4)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    X_gold = ops.embedding_add(patch_embeds, pos_weight)
    X_gold = ops.layer_norm(X_gold)
    X_gold = ops.flash_attention(X_gold, K, V, scale)
    X_gold = ops.ffn(X_gold, W_gate, W_up, W_down)
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
    print("SigLIP Vision Pipeline Test  (embedding_add → layer_norm → flash_attention → ffn)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    seq_len     = 64   # = mlen, one Q block for flash_attention
    hidden_size = 64   # = mlen (works with load_batch vlen=64)
    head_dim    = 64   # = hidden_size (Q shape = seq_len x head_dim)
    inter_dim   = 64   # FFN intermediate dim (must be divisible by mlen)
    mlen        = 64
    blen        = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    patch_embeds = torch.randn(seq_len, hidden_size) * 0.1
    pos_weight   = torch.randn(seq_len, hidden_size) * 0.1
    K            = torch.randn(seq_len, head_dim) * 0.1
    V            = torch.randn(seq_len, head_dim) * 0.1
    W_gate       = torch.randn(hidden_size, inter_dim) * 0.1
    W_up         = torch.randn(hidden_size, inter_dim) * 0.1
    W_down       = torch.randn(inter_dim, hidden_size) * 0.1

    print(f"\nInput shapes:")
    print(f"  patch_embeds: {patch_embeds.shape},  range [{patch_embeds.min():.3f}, {patch_embeds.max():.3f}]")
    print(f"  pos_weight:   {pos_weight.shape},  range [{pos_weight.min():.3f}, {pos_weight.max():.3f}]")
    print(f"  K:            {K.shape},  range [{K.min():.3f}, {K.max():.3f}]")
    print(f"  V:            {V.shape},  range [{V.min():.3f}, {V.max():.3f}]")
    print(f"  W_gate:       {W_gate.shape},  range [{W_gate.min():.3f}, {W_gate.max():.3f}]")
    print(f"  W_up:         {W_up.shape},  range [{W_up.min():.3f}, {W_up.max():.3f}]")
    print(f"  W_down:       {W_down.shape},  range [{W_down.min():.3f}, {W_down.max():.3f}]")
    print(f"  scale:        {scale:.6f}")

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

    X_gold = ops.embedding_add(patch_embeds, pos_weight)
    X_gold = ops.layer_norm(X_gold)
    X_gold = ops.flash_attention(X_gold, K, V, scale)
    X_gold = ops.ffn(X_gold, W_gate, W_up, W_down)
    golden_out = X_gold

    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0,:4].tolist()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs (order determines HBM layout)
    x_input     = prog.input("X",      shape=(seq_len, hidden_size))
    pos_input   = prog.input("POS",    shape=(seq_len, hidden_size))
    k_input     = prog.input("K",      shape=(seq_len, head_dim))
    v_input     = prog.input("V",      shape=(seq_len, head_dim))
    wgate_input = prog.input("W_gate", shape=(hidden_size, inter_dim))
    wup_input   = prog.input("W_up",   shape=(hidden_size, inter_dim))
    wdown_input = prog.input("W_down", shape=(inter_dim, hidden_size))

    # Load activations to VRAM
    X_batch   = prog.load_batch(x_input,   name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")

    # Pipeline
    ops.embedding_add(prog, X_batch, POS_batch)                        # X += POS (in-place)
    prog.layer_norm(X_batch, eps_offset=3, reci_hid_offset=4)         # normalize (slots 3,4, in-place)
    O = ops.flash_attention(prog, X_batch, k_input, v_input, scale)    # attention → new O var
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)              # ffn (in-place on O)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "X":      patch_embeds,
        "POS":    pos_weight,
        "K":      K,
        "V":      V,
        "W_gate": W_gate,
        "W_up":   W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FPRAM: slot0=0.0, slot1=scale, slot2=-inf, slot3=eps, slot4=1/hidden_size
    fp_preload = [0.0, scale, float("-inf"), 1e-6, 1.0 / hidden_size, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensor, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir)
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="siglip_vision_pipeline",
        data=None,
        specified_data_order=["X", "POS", "K", "V", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # Output is at O's VRAM location (flash_attention allocates O separately)
    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
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
    print(f"  Result location: VRAM row {o_vram_addr // mlen} (O from flash_attention)")
    run_and_assert(build_dir, "siglip_vision_pipeline", mlen=mlen, blen=blen)
