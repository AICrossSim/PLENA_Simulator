"""Emits the ISA (assembly) for ONE decoder layer of a batched LLaMA decode step.

This is the emulator-side, cycle-accurate ground truth for the decode part of
disaggregated serving. Pair file: `decoder_decode_test.py` (PyTorch golden +
Rust-sim runner). The cycle/byte counts it produces are what we use to CALIBRATE
the analytic model in `analytic_models/performance/disagg_decode.py`.

Decoder layer pipeline:

    RMSNorm -> Q,K,V proj -> RoPE(Q,K) -> KV-cache append -> GQA flash-attn
            -> W_O proj   -> +residual  -> RMSNorm -> SwiGLU FFN -> +residual

(LM head / embedding / final RMSNorm are per-token-once, not per-layer, so they
are NOT in this layer template — the analytic model adds them separately.)

FPRAM constant slots (6 pre-allocated, so GQA softmax state starts at slot 6):
    0 = 0.0          3 = eps        (RMSNorm; shared by both norms)
    1 = attn_scale   4 = 1/hidden   (RMSNorm; shared)
    2 = -inf         5 = 1.0        (FFN SiLU)

KV cache layout: the K/V HBM tensors already include this step's new tokens at
the tail — rows [0 : kv_size-s_q] are the old context, [kv_size-s_q :] are the
new K/V. The ISA also recomputes K_new/V_new and `store`s them (so the cache-
append memory write is counted), while attention reads the pre-filled cache so
sim and golden stay in sync.

Fixed hardware shape (GQA fused template): mlen=64, blen=4, hq=4, hkv=1,
h_qkv=16, hidden=64  (ratio = hq/hkv = blen = 4; ratio*h_qkv = mlen = 64).
"""

from pathlib import Path
import argparse
import math

from compiler.aten.ops.registry import OpRegistry, Backend
from compiler.aten.plena import PlenaCompiler as _PlenaCompiler
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.asm_templates import preload_addr_reg_asm


# -----------------------------------------------------------------------------
# FPRAM slot layout -- single source of truth for the test harness preload.
# -----------------------------------------------------------------------------
FP_SLOT_ZERO         = 0   # 0.0
FP_SLOT_ATTN_SCALE   = 1   # 1/sqrt(h_qkv)
FP_SLOT_NEG_INF      = 2   # -inf
FP_SLOT_RMS_EPS      = 3   # RMSNorm epsilon  (shared across all RMSNorms)
FP_SLOT_RMS_RECI_HID = 4   # 1/hidden         (shared across all RMSNorms)
FP_SLOT_SILU_ONE     = 5   # 1.0 (FFN SiLU)

DECODE_BATCH = 64  # queries per decode step (one new token per batch element)


# PlenaCompiler subclass: point GQA flash-attention at FPRAM slots 1 (attn_scale)
# and 2 (-inf). The stock method defaults to slots 5/0, which would collide with
# our shared constant slots.
class PlenaCompiler(_PlenaCompiler):
    def _flash_attention_gqa_fused(
        self, Q, K, V, scale, hq, hkv, h_qkv,
        *, batch_size: int = 1, seq_len: int | None = None, kv_seq_len: int | None = None,
    ):
        # batch_size / seq_len / kv_seq_len are accepted to match the compiler's
        # current flash_attention() call signature, but this decode-step override
        # bakes batch into s_q (one new token per sequence) and derives the q/kv
        # lengths from the tensor shapes below, so they are intentionally unused.
        ratio = hq // hkv
        mlen, blen, vlen = self.mlen, self.blen, self.mlen

        if ratio != blen:
            raise ValueError(
                f"GQA ratio hq/hkv={ratio} must equal blen={blen} "
                "(hardware packs heads into blen)."
            )
        if ratio * h_qkv != mlen:
            raise ValueError(
                f"GQA constraint: (hq/hkv)*h_qkv = {ratio * h_qkv} must equal mlen={mlen}."
            )

        s_q, _ = Q.shape
        s_kv, _ = K.shape
        if scale is None:
            scale = 1.0 / math.sqrt(h_qkv)

        self._ensure_hbm_sub_matrix_registered(K)
        self._ensure_hbm_sub_matrix_registered(V)
        alloc = self.register_allocator
        k_addr, v_addr = alloc.allocate_addr(2)
        gp_for_preload = alloc.allocate_gp(2)
        self.emit(preload_addr_reg_asm(
            addr_reg_to_set=[k_addr, v_addr],
            available_registers=gp_for_preload,
            addr_reg_val=[K.hbm_addr, V.hbm_addr],
        ))
        alloc.free_gp(gp_for_preload)

        q_vram_base = self.get_vram_addr(Q.name)
        from compiler.aten.plena.vars import VRAMMatrixVar
        s_name  = self._scoped_name("_gqa_S")
        pv_name = self._scoped_name("_gqa_PV")
        o_name  = self._scoped_name("O")
        self.allocate_vram_matrix(name=s_name,  rows=mlen * ratio, cols=mlen,         strict=False)
        self.allocate_vram_matrix(name=pv_name, rows=mlen * ratio, cols=mlen,         strict=False)
        self.allocate_vram_matrix(name=o_name,  rows=s_q,          cols=hq * h_qkv,   strict=False)

        br = min(mlen, s_q)
        fp_info = self.add_fpram_object(name="_gqa_softmax_state", size=3 * br * ratio)

        # Pin flash_attn_asm's scratch (S/PV) and output (O) to the buffers we
        # allocated above. Otherwise it writes O to an internal default address
        # that the following W_O projection wouldn't read from.
        self.emit(flash_attn_asm(
            mlen=mlen, vlen=vlen, blen=blen,
            batch=1, hq=hq, hkv=hkv, d=h_qkv,
            q_len=s_q, kv_len=s_kv,
            alive_registers_int=list(range(1, 16)),
            alive_registers_fp=list(range(1, 8)),
            vector_sram_base_address=q_vram_base,
            fp_sram_start_address=fp_info.fpram_addr,
            k_base_hbm_offset_reg=k_addr,
            v_base_hbm_offset_reg=v_addr,
            attn_scale_fp_address=FP_SLOT_ATTN_SCALE,  # slot 1 (canonical)
            inf_fp_address=FP_SLOT_NEG_INF,            # slot 2 (canonical)
            scratch_base_address=self.get_vram_addr(s_name),
            output_base_address=self.get_vram_addr(o_name),
        ))
        alloc.free_addr([k_addr, v_addr])

        O = VRAMMatrixVar(self, o_name, (s_q, hq * h_qkv), display_name="O")
        self._tensors[o_name] = O
        return O


# -----------------------------------------------------------------------------
# Reserve FPRAM slots 0..5 up-front so the GQA softmax-state allocation
# lands at slot 6+ and never overwrites our six shared constants.
# -----------------------------------------------------------------------------
def _reserve_fpram_constants(prog: PlenaCompiler) -> None:
    """Allocate FPRAM slots 0..5 in canonical order.  Their values come from
    the fp_preload bin produced by the test (decoder_decode_test.py)."""
    fp_allocs = prog.fpram_allocator
    for name in (
        "fp_zero",         # slot 0
        "fp_attn_scale",   # slot 1
        "fp_neg_inf",      # slot 2
        "fp_rms_eps",      # slot 3
        "fp_rms_reci_hid", # slot 4
        "fp_silu_one",     # slot 5
    ):
        if name not in fp_allocs.allocations:
            fp_allocs.allocate(name=name, size=1)


def generate_decode_asm(
    kv_size: int,
    hidden: int,
    inter: int,
    head_dim: int,            # accepted for CLI symmetry; HW fixes h_qkv = mlen // blen
    build_dir: str = "./build",
) -> dict:
    """Generate the per-layer batch-decode ISA and write it under build_dir.

    Args:
        kv_size:   TOTAL KV cache length (must be divisible by mlen=64 and > s_q).
                   The last s_q rows represent this step's new K/V -- see the
                   module docstring for the cache-append convention.
        hidden:    Total hidden size; must equal hq * h_qkv = 4 * 16 = 64.
        inter:     FFN intermediate width.
        head_dim:  Accepted but ignored (HW forces h_qkv = mlen // blen = 16).

    Returns:
        dict with keys:
            isa                 -- generated assembly text
            mlen, blen, s_q     -- hardware shape used
            hidden              -- echo of input
            o_proj_vram_addr    -- VRAM address where the per-layer output lands
            qrot/cos/sin/krot_vram_addr -- prestaged-tensor VRAM addresses
                                          (used by the test harness vram_preload)
    """
    del head_dim  # forced by hardware below

    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)

    # Hardware (mlen=64, blen=4 keeps the GQA fused template's PV region
    # under the IMM2 bound; mlen=128 would overflow).
    mlen, blen          = 64, 4
    real_data_ratio     = (8 * 8 + 8) / (8 * 8)
    s_q                 = DECODE_BATCH
    hq, hkv             = blen, 1
    h_qkv               = mlen // blen           # 16
    # GQA requirement: hq/hkv == blen, and ratio * h_qkv == mlen (= 4 * 16 = 64).
    total_q_dim         = hq * h_qkv             # 64
    k_padded_cols       = mlen                   # HW K cache row = mlen even though only hkv*h_qkv cols are real
    scale               = 1.0 / math.sqrt(h_qkv)

    assert hidden == total_q_dim,    f"hidden ({hidden}) must equal hq*h_qkv ({total_q_dim})"
    assert kv_size % mlen == 0,      f"kv_size ({kv_size}) must be a multiple of mlen ({mlen})"
    assert kv_size > s_q,            f"kv_size ({kv_size}) must exceed s_q ({s_q})"

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)
    _reserve_fpram_constants(prog)

    # -------------------------------------------------------------------------
    # HBM inputs. The RoPE helpers (QROT, COS, SIN, KROT) are prestaged into
    # VRAM (the test's vram_preload), so load_batch on them emits no ISA. Only X
    # is fetched at runtime via H_PREFETCH_V. Prestaged addresses pack right
    # after X, which occupies VRAM[0 .. s_q*hidden).
    # -------------------------------------------------------------------------
    qrot_vram = s_q * hidden
    cos_vram  = qrot_vram + s_q * hidden
    sin_vram  = cos_vram  + s_q * hidden
    krot_vram = sin_vram  + s_q * hidden       # KROT has mlen cols == hidden here

    x_input     = prog.input("X",      shape=(s_q, hidden))
    qrot_input  = prog.input("QROT",   shape=(s_q, hidden),       prestaged_vram_addr=qrot_vram)
    cos_input   = prog.input("COS",    shape=(s_q, hidden),       prestaged_vram_addr=cos_vram)
    sin_input   = prog.input("SIN",    shape=(s_q, hidden),       prestaged_vram_addr=sin_vram)
    krot_input  = prog.input("KROT",   shape=(s_q, mlen),         prestaged_vram_addr=krot_vram)
    wq_input    = prog.input("W_Q",    shape=(hidden, total_q_dim))
    wk_input    = prog.input("W_K",    shape=(hidden, mlen))       # padded out_features
    wv_input    = prog.input("W_V",    shape=(hidden, mlen))       # padded out_features
    wo_input    = prog.input("W_O",    shape=(total_q_dim, hidden))
    k_input     = prog.input("K",      shape=(kv_size, k_padded_cols))
    v_input     = prog.input("V",      shape=(kv_size, k_padded_cols))
    wgate_input = prog.input("W_gate", shape=(hidden, inter))
    wup_input   = prog.input("W_up",   shape=(hidden, inter))
    wdown_input = prog.input("W_down", shape=(inter, hidden))

    # -------------------------------------------------------------------------
    # Load activations.  X uses the HBM->VRAM prefetch ISA; the four prestaged
    # tensors register themselves at the addresses set above without emitting
    # transfer instructions.
    # -------------------------------------------------------------------------
    X_batch    = prog.load_batch(x_input,    name="X")
    Qrot_batch = prog.load_batch(qrot_input, name="QROT")
    Cos_batch  = prog.load_batch(cos_input,  name="COS")
    Sin_batch  = prog.load_batch(sin_input,  name="SIN")
    Krot_batch = prog.load_batch(krot_input, name="KROT")

    # Snapshot X for the post-attention residual BEFORE we mutate it in-place
    # with RMSNorm.  Fresh VRAM is zero-initialised, so `X_residual += X` is
    # an exact copy.
    X_residual = prog.alloc("X_residual", s_q, hidden)
    prog.vram_add(X_residual, X_batch)

    # X_n = RMSNorm(X).  Scratchpad pinned high enough to avoid the
    # prestaged tensors that live in VRAM[s_q*hidden .. 5*s_q*hidden].
    prog.rms_norm(X_batch,
                  eps_offset=FP_SLOT_RMS_EPS,
                  reci_hid_offset=FP_SLOT_RMS_RECI_HID,
                  scratchpad_vram_addr=5 * s_q * hidden + s_q * mlen)

    # Q, K, V projections
    Q     = prog.linear_projection(X_batch, wq_input, "Q")        # (s_q, total_q_dim)
    K_new = prog.linear_projection(X_batch, wk_input, "K_new")    # (s_q, mlen), first total_kv_dim_real cols real
    V_new = prog.linear_projection(X_batch, wv_input, "V_new")

    # RoPE on Q and K_new (paired layout: x * cos + rotate_half(x) * sin).
    # K reuses COS/SIN; its padded tail multiplies zero, so values there are
    # irrelevant.  See compiler/asm_templates/rope_asm.py for the ISA layout
    # (3 V_ ops per VLEN chunk: V_MUL_VV, V_MUL_VV, V_ADD_VV).
    prog.rope(Q,     Qrot_batch, Cos_batch, Sin_batch)
    prog.rope(K_new, Krot_batch, Cos_batch, Sin_batch)

    # GQA Repeat-Group prep: KV-cache append.  The destination HBM
    # bytes are accounted for but the data is mirrored into the K/V inputs
    # for this step by the test harness, so the attention sees a consistent
    # cache without a costly in-HBM copy.
    prog.store(K_new, name="K_appended")
    prog.store(V_new, name="V_appended")

    # GQA flash attention -- softmax(QK^T/sqrt(d)) . V.
    # Implementation in compiler/asm_templates/flashattn/overall.py.  For
    # ratio == blen the M_BTMM (batched) path is used; otherwise M_TMM
    # per-head.  Our ratio=4=blen so M_BTMM applies.
    O = prog.flash_attention(Q, k_input, v_input, scale,
                             hq=hq, hkv=hkv, h_qkv=h_qkv, causal_mask=None)

    # W_O projection
    O_proj = prog.linear_projection(O, wo_input, "O_proj")        # (s_q, hidden)

    # post-attention residual.  O_proj = attn_out + X.
    prog.vram_add(O_proj, X_residual)

    # Snapshot X' for the post-FFN residual.  X' = O_proj at this point.
    ffn_residual = prog.alloc("ffn_residual", s_q, hidden)
    prog.vram_add(ffn_residual, O_proj)

    # pre-FFN RMSNorm (reuses slots 3 and 4 -- same eps, same 1/hidden).
    prog.rms_norm(O_proj,
                  eps_offset=FP_SLOT_RMS_EPS,
                  reci_hid_offset=FP_SLOT_RMS_RECI_HID)

    # SwiGLU FFN in-place on O_proj.
    # ISA template lives in compiler/asm_templates/ffn_asm.py and includes
    # K-split when intermediate_size exceeds MRAM capacity.
    prog.ffn(O_proj, wgate_input, wup_input, wdown_input)

    # post-FFN residual.  O_proj = ffn_out + X'.
    prog.vram_add(O_proj, ffn_residual)

    # The per-layer pass ends here. The final RMSNorm + LM head run once after
    # the full L-layer stack (not per layer), so they are not emitted here; the
    # analytic model (disagg_decode.py) accounts for them separately per token.

    # -------------------------------------------------------------------------
    # Emit ASM + return metadata so the test knows where to read the result
    # from.
    # -------------------------------------------------------------------------
    gen_code = prog.compile()
    asm_path = build_path / "generated_asm_code.asm"
    asm_path.write_text(gen_code)

    o_proj_addr = prog.get_vram_addr(O_proj.name)
    print(f"Decode ASM -- s_q={s_q}, kv_size={kv_size}, hidden={hidden}, inter={inter}")
    print(f"GQA params:  hq={hq}, hkv={hkv}, h_qkv={h_qkv}, scale={scale:.4f}")
    print(f"Output X'' at VRAM[{o_proj_addr}] (row {o_proj_addr // mlen})")
    print(f"Generated {len(gen_code.splitlines())} lines of ISA -> {asm_path}")

    return {
        "isa": gen_code,
        "mlen": mlen,
        "blen": blen,
        "s_q": s_q,
        "hidden": hidden,
        "o_proj_vram_addr": o_proj_addr,
        "qrot_vram_addr": qrot_vram,
        "cos_vram_addr":  cos_vram,
        "sin_vram_addr":  sin_vram,
        "krot_vram_addr": krot_vram,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-layer batch-decode ISA generator (disaggregated decode)"
    )
    parser.add_argument("--kv-size",  type=int, default=128,
        help="TOTAL KV cache length INCLUDING the s_q new tokens (multiple of mlen=64)")
    parser.add_argument("--hidden",   type=int, default=64,
        help="Hidden size = hq * h_qkv = 4 * 16 = 64")
    parser.add_argument("--inter",    type=int, default=128, help="FFN intermediate width")
    parser.add_argument("--head-dim", type=int, default=16,  help="(forced) per-head dim h_qkv")
    parser.add_argument("--build-dir", type=str, default="./build/decode")
    args = parser.parse_args()

    generate_decode_asm(
        kv_size=args.kv_size, hidden=args.hidden, inter=args.inter,
        head_dim=args.head_dim, build_dir=args.build_dir,
    )
