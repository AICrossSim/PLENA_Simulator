"""
SmolLM2 Decoder Layer Pipeline Test

Full single-layer decoder pipeline chaining:
    embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

All operations run in-place on a single VRAMMatrixVar (X_batch).

FPRAM conflict resolution:
  - flash_attention uses FPRAM slot 2 = -inf
  - rms_norm uses FPRAM slot 1 = eps, slot 2 = 1/hidden (default)
  - SOLUTION: Call prog.rms_norm(X_batch, eps_offset=3, reci_hid_offset=4)
    directly (NOT ops.rms_norm) for ALL rms_norm calls
  - fp_preload = [0.0, attn_scale, float("-inf"), 1e-6, 1.0/hidden_size] + [0.0]*5

Parameters:
    seq_len     = 64   (= mlen, one Q block for flash_attention)
    hidden_size = 64   (= head_dim = mlen)
    head_dim    = 64
    inter_dim   = 64   (FFN intermediate, must be divisible by mlen=64)
    mlen        = 64
    blen        = 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import json

import torch

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


# ============================================================================
# Helper functions
# ============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """rotate_half: [-x[d//2:], x[:d//2]]"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def make_rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE cos/sin tables, shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)          # (seq_len, half)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    # Duplicate for both halves (standard RoPE)
    cos = torch.cat([cos_half, cos_half], dim=-1)   # (seq_len, head_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def rms_norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CPU reference: RMS normalization (float32)."""
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


def flash_attn_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """CPU reference: scaled dot-product attention."""
    import torch.nn.functional as F
    scores = (Q @ K.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def ffn_ref(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """CPU reference: SwiGLU FFN matching PLENA hardware (SiLU applied to W_up projection)."""
    import torch.nn.functional as F
    gate = x @ W_gate           # gate projection (no silu)
    up = F.silu(x @ W_up)      # silu applied to up projection (matches hardware)
    return (up * gate) @ W_down


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SmolLM2 Decoder Layer Pipeline Test")
    print("  embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    seq_len    = 64   # = mlen, one Q block for flash_attention
    hidden_size = 64  # = head_dim = mlen
    head_dim   = 64
    inter_dim  = 64   # FFN intermediate (must be divisible by mlen=64)
    mlen       = 64
    blen       = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    token_embeds = torch.randn(seq_len, hidden_size)
    pos_weight   = torch.randn(seq_len, hidden_size)
    K            = torch.randn(seq_len, head_dim)
    V            = torch.randn(seq_len, head_dim)
    W_gate       = torch.randn(hidden_size, inter_dim)
    W_up         = torch.randn(hidden_size, inter_dim)
    W_down       = torch.randn(inter_dim, hidden_size)

    cos, sin = make_rope_tables(seq_len, head_dim)

    # Precompute Q_rot from bfloat16-approximated intermediate.
    # PLENA computes embedding_add + rms_norm in bfloat16; Q_rot must match that
    # or the rope mismatch propagates catastrophically through flash_attention.
    X_embed_bf16 = token_embeds.to(torch.bfloat16) + pos_weight.to(torch.bfloat16)
    rms_bf16 = torch.rsqrt(
        X_embed_bf16.float().pow(2).mean(-1, keepdim=True) + 1e-6
    ).to(torch.bfloat16)
    X_norm_bf16 = (X_embed_bf16 * rms_bf16)  # bfloat16 rms_norm approximation
    Q_rot = rotate_half(X_norm_bf16.float())

    print(f"\ntoken_embeds: {token_embeds.shape}, range [{token_embeds.min():.3f}, {token_embeds.max():.3f}]")
    print(f"pos_weight:   {pos_weight.shape},   range [{pos_weight.min():.3f}, {pos_weight.max():.3f}]")
    print(f"Q_rot:        {Q_rot.shape},         range [{Q_rot.min():.3f}, {Q_rot.max():.3f}]")
    print(f"cos:          {cos.shape},            range [{cos.min():.3f}, {cos.max():.3f}]")
    print(f"sin:          {sin.shape},            range [{sin.min():.3f}, {sin.max():.3f}]")
    print(f"K:            {K.shape},              range [{K.min():.3f}, {K.max():.3f}]")
    print(f"V:            {V.shape},              range [{V.min():.3f}, {V.max():.3f}]")
    print(f"W_gate:       {W_gate.shape},         range [{W_gate.min():.3f}, {W_gate.max():.3f}]")
    print(f"W_up:         {W_up.shape},            range [{W_up.min():.3f}, {W_up.max():.3f}]")
    print(f"W_down:       {W_down.shape},          range [{W_down.min():.3f}, {W_down.max():.3f}]")
    print(f"\nattn_scale: {scale:.6f}")

    # ========================================================================
    # CPU Golden Reference
    # NOTE: We use manual functions (not registry) so we can capture the
    # intermediate Q_rot that PLENA needs as a pre-loaded input.
    # ========================================================================
    print("\n--- CPU Golden Reference ---")

    X_gold = token_embeds.clone()
    X_gold = X_gold + pos_weight                            # embedding_add
    # Use bfloat16 rms_norm to match PLENA's quantised intermediate (for Q_rot consistency)
    X_gold_bf16 = X_gold.to(torch.bfloat16)
    rms_gold = torch.rsqrt(
        X_gold_bf16.float().pow(2).mean(-1, keepdim=True) + 1e-6
    ).to(torch.bfloat16)
    X_gold = (X_gold_bf16 * rms_gold).float()              # rms_norm (bfloat16)
    Q_rot_gold = rotate_half(X_gold)                       # consistent Q_rot
    X_gold = X_gold * cos + Q_rot_gold * sin               # rope
    X_gold = flash_attn_ref(X_gold, K, V, scale)           # flash_attention
    X_gold = ffn_ref(X_gold, W_gate, W_up, W_down)        # ffn
    X_gold = rms_norm_ref(X_gold)                          # final rms_norm

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0,:4].tolist()}")

    # ========================================================================
    # PLENA Backend (ISA generation)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs — order determines HBM layout
    x_input     = prog.input("X",      shape=(seq_len, hidden_size))
    pos_input   = prog.input("POS",    shape=(seq_len, hidden_size))
    qrot_input  = prog.input("QROT",   shape=(seq_len, head_dim))
    cos_input   = prog.input("COS",    shape=(seq_len, head_dim))
    sin_input   = prog.input("SIN",    shape=(seq_len, head_dim))
    k_input     = prog.input("K",      shape=(seq_len, head_dim))
    v_input     = prog.input("V",      shape=(seq_len, head_dim))
    wgate_input = prog.input("W_gate", shape=(hidden_size, inter_dim))
    wup_input   = prog.input("W_up",   shape=(hidden_size, inter_dim))
    wdown_input = prog.input("W_down", shape=(inter_dim, hidden_size))

    # Load to VRAM
    X_batch    = prog.load_batch(x_input,    name="X")
    POS_batch  = prog.load_batch(pos_input,  name="POS")
    Qrot_batch = prog.load_batch(qrot_input, name="QROT")
    Cos_batch  = prog.load_batch(cos_input,  name="COS")
    Sin_batch  = prog.load_batch(sin_input,  name="SIN")

    # Pipeline
    ops.embedding_add(prog, X_batch, POS_batch)                    # X += POS (in-place)
    prog.rms_norm(X_batch, eps_offset=3, reci_hid_offset=4)       # normalize (in-place, slots 3,4)
    ops.rope(prog, X_batch, Qrot_batch, Cos_batch, Sin_batch)     # RoPE (in-place)
    O = ops.flash_attention(prog, X_batch, k_input, v_input, scale)  # attention → new O var
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)          # ffn (in-place on O)
    prog.rms_norm(O, eps_offset=3, reci_hid_offset=4)             # final normalize (in-place on O)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "X":      token_embeds,
        "POS":    pos_weight,
        "QROT":   Q_rot,
        "COS":    cos,
        "SIN":    sin,
        "K":      K,
        "V":      V,
        "W_gate": W_gate,
        "W_up":   W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FPRAM layout:
    #   slot 0 = 0.0        (reserved)
    #   slot 1 = attn_scale (flash_attention)
    #   slot 2 = -inf       (flash_attention softmax mask)
    #   slot 3 = eps        (rms_norm, offset=3)
    #   slot 4 = 1/hidden   (rms_norm, offset=4)
    #   slots 5-9 = 0.0     (padding)
    fp_preload = [0.0, scale, float("-inf"), 1e-6, 1.0 / hidden_size, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensor, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir)
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="smollm2_decoder_pipeline",
        data=None,
        specified_data_order=["X", "POS", "QROT", "COS", "SIN", "K", "V", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # Result is at O's VRAM location (flash_attention allocates O separately)
    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    comparison_params = {
        "start_row_idx":      o_vram_addr // mlen,
        "num_rows":           (seq_len * hidden_size) // mlen,
        "num_batches":        seq_len,
        "elements_per_batch": hidden_size,
        "row_dim":            mlen,
        "use_stride_mode":    hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {o_vram_addr // mlen} (O from flash_attention)")

    run_and_assert(build_dir, "smollm2_decoder_pipeline", mlen=mlen, blen=blen)
