"""End-to-end testbench for the LLaMA Pre-Norm batch-decode pipeline.

Pairs with `decoder_decode_asm_gen.py` to:
  1. Build the per-layer ISA
  2. Compute a PyTorch golden reference for the same pipeline
  3. Pack the HBM data + VRAM preload + FPRAM constants the emulator needs
  4. Run the Rust transactional emulator and diff its VRAM output vs golden

This script is the ground-truth used by `disagg_sim_validate.py` to validate
the analytical model (`perf_model.PerfModel.decoder_layer_decode`).  The
golden compares numerical correctness; the sim cycle count is what the
validation sweep diffs against the analytical estimate.

KV cache layout in HBM
    Cache rows have shape (kv_size, mlen) -- last s_q rows are this step's new
    K_new / V_new (K is RoPE'd before being written).  Only the first
    hkv*h_qkv = 16 columns hold real data; columns 16..mlen are zero pad.

Prestaged VRAM tensors
    QROT, COS, SIN, KROT are pushed into VRAM at fixed addresses via the
    `vram_preload` binary -- this avoids issuing multiple HBM->VRAM prefetch
    sequences (X is the only tensor loaded that way at runtime).

FPRAM preload
    Six canonical slots (see FPRAMAllocator docstring in
    compiler/aten/plena/memory.py):

      0 = 0.0           4 = 1/hidden
      1 = attn_scale    5 = 1.0 (FFN SiLU)
      2 = -inf          6+ = GQA softmax state (pre-allocated by the asm-gen)
      3 = eps
"""

import json
import math
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F

from sim_env_utils import create_mem_for_sim
from plena_utils.load_config import load_precision_from_toml
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.misc.decoder_decode_asm_gen import (
    generate_decode_asm,
    DECODE_BATCH,
    FP_SLOT_ZERO,
    FP_SLOT_ATTN_SCALE,
    FP_SLOT_NEG_INF,
    FP_SLOT_RMS_EPS,
    FP_SLOT_RMS_RECI_HID,
    FP_SLOT_SILU_ONE,
)


def rms_norm_ref(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / rms


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Hardware RoPE convention: rotate consecutive pairs (x[2i], x[2i+1])."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s   = cos[..., 0::2], sin[..., 0::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    """The QROT/KROT layout the hardware rope ISA expects:
       out[..., 0::2] = -x[..., 1::2];  out[..., 1::2] =  x[..., 0::2]"""
    out = torch.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] =  x[..., 0::2]
    return out


def gqa_sdpa_ref(q, k, v, scale, hq, hkv):
    """GQA attention via SDPA with KV-head broadcast.  Returns (s_q, hq*h_qkv)."""
    ratio = hq // hkv
    q_t = q.unsqueeze(0).transpose(1, 2)
    k_t = k.unsqueeze(0).transpose(1, 2).repeat_interleave(ratio, dim=1)
    v_t = v.unsqueeze(0).transpose(1, 2).repeat_interleave(ratio, dim=1)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    return o.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)


def decoder_decode_golden(
    x, w_q, w_k_real, w_v_real, w_o,
    k_old_real, v_old_real,
    cos_q, sin_q, cos_k_real, sin_k_real,
    w_gate, w_up, w_down,
    hq, hkv, h_qkv, qk_scale, eps=1e-5,
):
    """Pure-PyTorch golden matching the ISA (rms_norm + QKV proj + RoPE +
    KV-append + GQA + W_O + residual + rms_norm + FFN + residual).  No final
    RMSNorm (Algorithm 2 applies that only after the full L-layer stack)."""
    s_q, hidden = x.shape
    x_bf16 = x.to(torch.bfloat16)

    # 1. Pre-attention RMSNorm
    x_norm = rms_norm_ref(x_bf16.float(), eps=eps).to(torch.bfloat16)

    # 2. Q, K, V projections
    q     = (x_norm.float() @ w_q.float()).to(torch.bfloat16)
    k_new = (x_norm.float() @ w_k_real.float()).to(torch.bfloat16)
    v_new = (x_norm.float() @ w_v_real.float()).to(torch.bfloat16)

    # 3. RoPE on Q and new K
    q_rot     = rope_ref(q, cos_q, sin_q)
    k_new_rot = rope_ref(k_new, cos_k_real, sin_k_real)

    # 4. KV cache append (concat new at tail)
    k_cache_full = torch.cat([k_old_real, k_new_rot], dim=0)
    v_cache_full = torch.cat([v_old_real, v_new],     dim=0)

    # 5. GQA attention
    q_heads = q_rot.reshape(s_q, hq, h_qkv).float()
    k_heads = k_cache_full.reshape(-1, hkv, h_qkv).float()
    v_heads = v_cache_full.reshape(-1, hkv, h_qkv).float()
    attn_raw = gqa_sdpa_ref(q_heads, k_heads, v_heads, qk_scale, hq, hkv).to(torch.bfloat16)

    # 6. W_O
    o_proj = (attn_raw.float() @ w_o.float()).to(torch.bfloat16)

    # 7. Post-attention residual
    x_prime = x_bf16 + o_proj

    # 8. Pre-FFN RMSNorm
    x_prime_n = rms_norm_ref(x_prime.float(), eps=eps).to(torch.bfloat16)

    # 9. SwiGLU FFN
    gate_out  = (x_prime_n.float() @ w_gate.float()).to(torch.bfloat16)
    up_out    = (x_prime_n.float() @ w_up.float()).to(torch.bfloat16)
    silu_gate = F.silu(gate_out.float()).to(torch.bfloat16)
    ffn_out   = ((silu_gate.float() * up_out.float()).to(torch.bfloat16).float()
                 @ w_down.float()).to(torch.bfloat16)

    # 10. Post-FFN residual -- no final norm
    return x_prime + ffn_out, k_cache_full, v_cache_full


# -----------------------------------------------------------------------------
# Testbench main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-layer batch-decode testbench (disaggregated decode, Algorithm 2)"
    )
    ap.add_argument("--kv-size", type=int, default=128,
        help="TOTAL KV cache length INCLUDING the s_q new tokens "
             "(multiple of mlen=64). Default: 128 = 64 old + 64 new.")
    ap.add_argument("--inter", type=int, default=128,
        help="FFN intermediate width (multiple of mlen=64). Default: 128.")
    args = ap.parse_args()

    # -- Hardware / model config (matches the ASM generator) ------------------
    kv_size  = args.kv_size
    mlen     = 64
    blen     = 4
    s_q      = DECODE_BATCH       # 64
    hq       = blen               # 4
    hkv      = 1
    h_qkv    = mlen // blen       # 16
    hidden   = hq * h_qkv         # 64
    inter    = args.inter
    qk_scale = 1.0 / math.sqrt(h_qkv)
    eps      = 1e-5
    reci_hid = 1.0 / hidden

    assert kv_size > s_q,         f"kv_size ({kv_size}) must exceed s_q ({s_q})"
    assert kv_size % mlen == 0,   f"kv_size ({kv_size}) must be a multiple of mlen ({mlen})"
    assert inter % mlen == 0,     f"inter ({inter}) must be a multiple of mlen ({mlen})"
    kv_old = kv_size - s_q

    build_dir = Path(__file__).parent.parent / "build" / "decoder_decode"
    build_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Generate ASM ------------------------------------------------------
    print("[1/4] Generating decode ASM ...")
    asm_info = generate_decode_asm(
        kv_size=kv_size, hidden=hidden, inter=inter, head_dim=h_qkv,
        build_dir=str(build_dir),
    )
    gen_assembly_code = asm_info["isa"]
    o_proj_vram_addr  = asm_info["o_proj_vram_addr"]
    qrot_vram_addr    = asm_info["qrot_vram_addr"]
    cos_vram_addr     = asm_info["cos_vram_addr"]
    sin_vram_addr     = asm_info["sin_vram_addr"]
    krot_vram_addr    = asm_info["krot_vram_addr"]

    # -- 2. Random inputs -----------------------------------------------------
    print("[2/4] Building HBM data ...")
    torch.manual_seed(42)

    # Activation for the s_q new tokens.
    x = torch.randn(s_q, hidden, dtype=torch.bfloat16) * 0.5

    # Projection weights.  W_K / W_V have only total_kv_dim_real = hkv*h_qkv
    # real columns (16); we zero-pad them to mlen=64 because the hardware
    # linear projection requires out_features % mlen == 0.
    total_q_dim      = hq * h_qkv
    total_kv_dim_real = hkv * h_qkv

    w_q       = torch.randn(hidden, total_q_dim,       dtype=torch.bfloat16) * 0.1
    w_k_real  = torch.randn(hidden, total_kv_dim_real, dtype=torch.bfloat16) * 0.1
    w_v_real  = torch.randn(hidden, total_kv_dim_real, dtype=torch.bfloat16) * 0.1
    w_o       = torch.randn(total_q_dim, hidden,       dtype=torch.bfloat16) * 0.1

    # Pre-existing KV cache (already RoPE'd) -- last s_q rows will be filled
    # with this step's new K/V after the golden runs.
    k_old_real = torch.randn(kv_old, total_kv_dim_real, dtype=torch.bfloat16) * 0.5
    v_old_real = torch.randn(kv_old, total_kv_dim_real, dtype=torch.bfloat16) * 0.5

    # FFN weights
    w_gate = torch.randn(hidden, inter, dtype=torch.bfloat16) * 0.1
    w_up   = torch.randn(hidden, inter, dtype=torch.bfloat16) * 0.1
    w_down = torch.randn(inter, hidden, dtype=torch.bfloat16) * 0.1

    # RoPE tables (paired layout: cos[2i] == cos[2i+1] so rope_ref matches HW).
    # Q and K share COS / SIN -- K's padding cols multiply zero, so only the
    # first total_kv_dim_real values are observed.
    cos_half = torch.rand(s_q, hidden // 2, dtype=torch.bfloat16)
    sin_half = torch.rand(s_q, hidden // 2, dtype=torch.bfloat16)
    cos_q = cos_half.repeat_interleave(2, dim=1)
    sin_q = sin_half.repeat_interleave(2, dim=1)
    cos_k_real = cos_q[:, :total_kv_dim_real]
    sin_k_real = sin_q[:, :total_kv_dim_real]

    # -- 3. Golden reference --------------------------------------------------
    decoder_out_golden, k_cache_full_real, v_cache_full_real = decoder_decode_golden(
        x, w_q, w_k_real, w_v_real, w_o,
        k_old_real, v_old_real,
        cos_q, sin_q, cos_k_real, sin_k_real,
        w_gate, w_up, w_down,
        hq=hq, hkv=hkv, h_qkv=h_qkv, qk_scale=qk_scale, eps=eps,
    )

    # -- 4. Pack HBM tensors, VRAM preload, FPRAM preload ---------------------
    # Precompute QROT / KROT to match the BF16 path the ISA will take.
    x_norm_bf16 = rms_norm_ref(x.float(), eps=eps).to(torch.bfloat16)
    q_pre_rope  = (x_norm_bf16.float() @ w_q.float()).to(torch.bfloat16)
    k_new_pre   = (x_norm_bf16.float() @ w_k_real.float()).to(torch.bfloat16)
    qrot        = rotate_half_pairs(q_pre_rope)
    krot_real   = rotate_half_pairs(k_new_pre)

    def _pad_cols(t: torch.Tensor, target_cols: int) -> torch.Tensor:
        pad = target_cols - t.shape[-1]
        return t.contiguous() if pad <= 0 else F.pad(t, (0, pad)).contiguous()

    krot_padded = _pad_cols(krot_real, mlen)
    w_k_padded  = _pad_cols(w_k_real,  mlen)
    w_v_padded  = _pad_cols(w_v_real,  mlen)

    # Pad K/V cache rows from (kv_size, total_kv_dim_real) to (kv_size, mlen).
    num_kv_slots = mlen // h_qkv
    k_padded = torch.zeros(kv_size, num_kv_slots, h_qkv, dtype=torch.bfloat16)
    v_padded = torch.zeros(kv_size, num_kv_slots, h_qkv, dtype=torch.bfloat16)
    k_padded[:, :hkv, :] = k_cache_full_real.reshape(kv_size, hkv, h_qkv)
    v_padded[:, :hkv, :] = v_cache_full_real.reshape(kv_size, hkv, h_qkv)

    input_tensor = {
        "X":      x,
        "QROT":   qrot,
        "COS":    cos_q,
        "SIN":    sin_q,
        "KROT":   krot_padded,
        "W_Q":    w_q,
        "W_K":    w_k_padded,
        "W_V":    w_v_padded,
        "W_O":    w_o,
        "K":      k_padded.reshape(kv_size, mlen),
        "V":      v_padded.reshape(kv_size, mlen),
        "W_gate": w_gate,
        "W_up":   w_up,
        "W_down": w_down,
    }
    golden_result = {"original_output": decoder_out_golden.reshape(s_q, hidden)}

    # -- VRAM preload ---------------------------------------------------------
    # The asm-gen reserves prestaged VRAM addresses for the four RoPE helpers
    # (so load_batch on them emits no transfer ISA).  Build a flat fp16
    # buffer that lays those tensors at the addresses returned by asm_info.
    # X is loaded normally, so we leave VRAM[0..s_q*hidden) as zeros -- the
    # H_PREFETCH_V instructions will fill that region at runtime.
    preload_len = krot_vram_addr + s_q * mlen
    vram_preload = torch.zeros(preload_len, dtype=torch.float16)
    def _stage(t: torch.Tensor, addr: int) -> None:
        flat = t.contiguous().reshape(-1).to(torch.float16)
        vram_preload[addr : addr + flat.numel()] = flat
    _stage(qrot,        qrot_vram_addr)
    _stage(cos_q,       cos_vram_addr)
    _stage(sin_q,       sin_vram_addr)
    _stage(krot_padded, krot_vram_addr)

    # -- FPRAM preload (canonical 6-slot layout) ------------------------------
    fp_preload = [0.0] * 6
    fp_preload[FP_SLOT_ZERO]         = 0.0
    fp_preload[FP_SLOT_ATTN_SCALE]   = qk_scale
    fp_preload[FP_SLOT_NEG_INF]      = float("-inf")
    fp_preload[FP_SLOT_RMS_EPS]      = eps
    fp_preload[FP_SLOT_RMS_RECI_HID] = reci_hid
    fp_preload[FP_SLOT_SILU_ONE]     = 1.0

    # -- Materialise sim artifacts --------------------------------------------
    create_sim_env(
        input_tensor, gen_assembly_code, golden_result, fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
    )
    # Precision (block size, element bits, scale bits) is read from
    # plena_settings.toml so the test stays in sync with whatever
    # MXFP / MXINT format the emulator is configured for.
    # NOTE: create_mem_for_sim now loads PRECISION from plena_settings.toml
    # internally (see sim_env_utils.env_setup), so we no longer pass
    # precision_settings explicitly — doing so would be an unexpected kwarg.
    create_mem_for_sim(
        data_size=256, mode="behave_sim", asm="decoder_decode", data=None,
        specified_data_order=[
            "X", "QROT", "COS", "SIN", "KROT",
            "W_Q", "W_K", "W_V", "W_O",
            "K", "V",
            "W_gate", "W_up", "W_down",
        ],
        build_path=build_dir,
    )

    comparison_params = {
        "start_row_idx":      o_proj_vram_addr // mlen,
        "num_rows":           (s_q * hidden) // mlen,
        "num_batches":        s_q,
        "elements_per_batch": hidden,
        "row_dim":            mlen,
        "use_stride_mode":    False,
        "use_slice_mode":     False,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    # -- 5. Run Rust simulator and diff vs golden -----------------------------
    print("[3/4] Running Rust simulator and comparing output ...")
    run_and_assert(build_dir, "decoder_decode", mlen=mlen, blen=blen)
    print("[4/4] Done.")
