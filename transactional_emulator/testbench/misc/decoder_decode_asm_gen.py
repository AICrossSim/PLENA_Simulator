"""Emits the ISA (assembly) for ONE decoder layer of a batched LLaMA decode step.

This is the transactional-emulator cycle and traffic reference for the decode
part of disaggregated serving. Pair file: `decoder_decode_test.py` (PyTorch
golden + Rust-sim runner). Its cycle/byte counts calibrate the analytic model in
`analytic_models/performance/disagg_decode.py`; they are not matched-layer RTL
measurements.

Decoder layer pipeline:

    RMSNorm -> Q,K,V proj -> RoPE(Q,K) -> KV-cache append -> GQA flash-attn
            -> W_O proj   -> +residual  -> RMSNorm -> SwiGLU FFN -> +residual

The final RMSNorm and LM head are emitted once after this one-layer stack so the
test exercises the complete epilogue. Analytic per-layer accounting keeps that
once-per-token work separate from the repeated decoder-layer body.

FPRAM constant slots (6 pre-allocated, so GQA softmax state starts at slot 6):
    0 = 0.0          3 = eps        (RMSNorm; shared by both norms)
    1 = attn_scale   4 = 1/hidden   (RMSNorm; shared)
    2 = -inf         5 = 1.0        (FFN SiLU)

KV cache layout: the K/V HBM tensors already include this step's new tokens at
the tail — rows [0 : kv_size-s_q] are the old context, [kv_size-s_q :] are the
new K/V. The ISA also recomputes K_new/V_new and `store`s them (so the cache-
append memory write is counted), while attention reads the pre-filled cache so
sim and golden stay in sync.

Hardware GQA group: mlen=64, blen=4, h_qkv=16 and hq/hkv=4. The residual
stream stays MLEN-wide while W_Q expands to one MLEN-wide query group per KV
head and W_O maps those groups back to the residual width.
"""

from pathlib import Path
import argparse
import json
import math

from compiler.aten.ops.registry import OpRegistry, Backend
from compiler.aten.plena import PlenaCompiler as _PlenaCompiler
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.asm_templates.flashattn.overall import (
    softmax_row_tile,
    softmax_state_slots,
)
from compiler.asm_templates import preload_addr_reg_asm
from compiler.asm_templates.lm_head import lm_head_asm, lm_head_vocab_padding


# -----------------------------------------------------------------------------
# FPRAM slot layout -- single source of truth for the test harness preload.
# -----------------------------------------------------------------------------
FP_SLOT_ZERO         = 0   # 0.0
FP_SLOT_ATTN_SCALE   = 1   # 1/sqrt(h_qkv)
FP_SLOT_NEG_INF      = 2   # -inf
FP_SLOT_RMS_EPS      = 3   # RMSNorm epsilon  (shared across all RMSNorms)
FP_SLOT_RMS_RECI_HID = 4   # 1/hidden         (shared across all RMSNorms)
FP_SLOT_SILU_ONE     = 5   # 1.0 (FFN SiLU)
FP_SLOT_COUNT        = 6   # shared constants reserved ahead of dynamic use

# Consecutive query positions sharing one synthetic cache in this test harness.
DECODE_BATCH = 64
# Vocabulary the testbench validates the LM head over. A production vocabulary
# is orders of magnitude wider; the lowering is identical and only the tile
# count changes, and the logits of a real one would not fit this VRAM.
DECODE_VOCAB = 256


def decode_geometry(
    settings_toml: str | None = None,
    *,
    kv_heads: int = 1,
    kv_head_reuse: bool = False,
    row_tile: int | None = None,
) -> dict:
    """Derive the decode-layer shape from the emulator's array geometry.

    The emulator and the generated assembly must agree on MLEN/BLEN/HLEN, so
    both read the TRANSACTIONAL section of the settings file rather than
    carrying their own constants.

    HLEN is the head lane width, so it is the head dimension. The array holds
    MLEN/HLEN head lanes, which is the number of query heads one broadcast
    matmul covers and therefore the GQA ratio. Each KV head owns one such
    MLEN-wide query/output group, while the residual stream remains MLEN-wide.

    BLEN is independent of all of that: it is the systolic block length. The
    scalar FP SRAM sets the query-row tile, which is snapped to whole BLEN
    blocks so the final matrix issue does not compute unused rows.
    """
    import toml

    from runtime_paths import settings_path

    path = settings_toml or str(settings_path())
    config = toml.load(path).get("TRANSACTIONAL", {}).get("CONFIG", {})

    def value(name: str) -> int:
        entry = config.get(name)
        if not isinstance(entry, dict) or "value" not in entry:
            raise ValueError(f"TRANSACTIONAL.CONFIG.{name} missing from {path}")
        return int(entry["value"])

    mlen, blen, hlen, vlen = value("MLEN"), value("BLEN"), value("HLEN"), value("VLEN")
    if vlen != mlen:
        raise ValueError(f"VLEN ({vlen}) must equal MLEN ({mlen})")
    if mlen % hlen or mlen % blen:
        raise ValueError(f"MLEN ({mlen}) must be divisible by HLEN and BLEN")
    if not blen <= hlen <= mlen:
        raise ValueError(f"BLEN <= HLEN <= MLEN violated ({blen}, {hlen}, {mlen})")
    broadcast = value("BROADCAST_AMOUNT")
    if broadcast != mlen // hlen:
        raise ValueError(
            f"BROADCAST_AMOUNT ({broadcast}) must equal MLEN/HLEN ({mlen // hlen})"
        )

    if not 1 <= kv_heads <= broadcast:
        raise ValueError(
            f"kv_heads must be in [1, BROADCAST_AMOUNT={broadcast}], got {kv_heads}"
        )

    head_dim = hlen
    query_heads = broadcast * kv_heads
    # VRAM matrices are registered in whole MLEN-row tiles, so the packed
    # decode batch covers a whole query tile.
    batch = max(DECODE_BATCH, mlen)
    batch += (-batch) % mlen
    # The online softmax keeps three running scalars per query row per query
    # head on top of the shared constants. Sweeping every query row at once
    # makes that grow with MLEN * ratio, so the rows are tiled to the largest
    # count whose state fits the scalar FP SRAM the hardware provides.
    fp_sram_depth = value("FP_SRAM_DEPTH")
    query_rows = min(batch, mlen)
    live_head_lanes = broadcast * (kv_heads if kv_head_reuse else 1)
    if row_tile is None:
        row_tile = softmax_row_tile(
            query_rows,
            live_head_lanes,
            fp_sram_depth,
            constant_slots=FP_SLOT_COUNT,
            blen=blen,
        )
        flash_fp_sram_depth = fp_sram_depth
    else:
        if not 0 < row_tile <= query_rows:
            raise ValueError(
                f"row_tile must be in [1, {query_rows}], got {row_tile}"
            )
        if row_tile % blen:
            raise ValueError(
                f"row_tile ({row_tile}) must be a multiple of BLEN ({blen})"
            )
        flash_fp_sram_depth = softmax_state_slots(
            row_tile, live_head_lanes, constant_slots=FP_SLOT_COUNT
        )
        if flash_fp_sram_depth > fp_sram_depth:
            raise ValueError(
                f"row_tile={row_tile} needs {flash_fp_sram_depth} scalar FP SRAM "
                f"slots, but the hardware provides {fp_sram_depth}"
            )

    fp_sram_required = softmax_state_slots(
        row_tile, live_head_lanes, constant_slots=FP_SLOT_COUNT
    )
    return {
        "mlen": mlen,
        "blen": blen,
        "hlen": hlen,
        "hbm_v_prefetch_amount": value("HBM_V_Prefetch_Amount"),
        "hbm_v_writeback_amount": value("HBM_V_Writeback_Amount"),
        "broadcast_amount": broadcast,
        "head_dim": head_dim,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "hidden": mlen,
        "total_q_dim": query_heads * head_dim,
        "total_kv_dim": kv_heads * head_dim,
        "packed_group_layout": kv_heads > 1,
        "q_group_stride": batch * mlen if kv_heads > 1 else None,
        "o_group_stride": batch * mlen if kv_heads > 1 else None,
        "batch": batch,
        "fp_sram_depth": fp_sram_depth,
        "flash_fp_sram_depth": flash_fp_sram_depth,
        "softmax_row_tile": row_tile,
        "live_softmax_heads": live_head_lanes,
        "fp_sram_required": fp_sram_required,
    }


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
        broadcast_amount = mlen // h_qkv

        # The batched QK^T applies one K slice to `broadcast_amount` query-head
        # lanes at once, so the GQA group must fit the lanes the array has.
        if ratio > broadcast_amount:
            raise ValueError(
                f"GQA ratio hq/hkv={ratio} exceeds the {broadcast_amount} "
                f"broadcast lanes (MLEN/head_dim = {mlen}/{h_qkv})."
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
        geometry = self.decode_geometry
        row_tile = geometry["softmax_row_tile"]
        live_head_lanes = ratio * (
            hkv if getattr(self, "kv_head_reuse", False) else 1
        )
        fp_info = self.add_fpram_object(
            name="_gqa_softmax_state", size=3 * row_tile * live_head_lanes
        )

        # Pin flash_attn_asm's scratch (S/PV) and output (O) to the buffers we
        # allocated above. Otherwise it writes O to an internal default address
        # that the following W_O projection wouldn't read from.
        flash_kwargs = {}
        if hkv > 1:
            flash_kwargs = {
                "packed_group_layout": True,
                "q_group_stride": Q.physical_shape[0] * mlen,
                "o_group_stride": s_q * mlen,
            }
        self.emit(flash_attn_asm(
            mlen=mlen, vlen=vlen, blen=blen,
            batch=1, hq=hq, hkv=hkv, d=h_qkv,
            q_len=s_q, kv_len=s_kv,
            broadcast_amount=broadcast_amount,
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
            fp_sram_depth=geometry["flash_fp_sram_depth"],
            kv_head_reuse=getattr(self, "kv_head_reuse", False),
            **flash_kwargs,
        ))
        alloc.free_addr([k_addr, v_addr])

        O = VRAMMatrixVar(self, o_name, (s_q, hq * h_qkv), display_name="O")
        self._tensors[o_name] = O
        return O

    def lm_head(self, hidden_states, weight, name: str = "logits"):
        """Emit the hidden-to-vocabulary projection over the final hidden states.

        `weight` carries the checkpoint's native `(vocab_size, hidden_size)`
        row-major layout, so this lowers to the transposed projection and needs
        no transpose pass over the weight matrix.
        """
        from compiler.aten.plena.vars import VRAMMatrixVar

        batch, hidden_size = hidden_states.shape
        vocab_size, weight_hidden = weight.shape
        if weight_hidden != hidden_size:
            raise ValueError(
                f"lm_head weight is (vocab={vocab_size}, hidden={weight_hidden}) "
                f"but the hidden states are {hidden_size} wide"
            )

        self._ensure_hbm_sub_matrix_registered(weight)
        alloc = self.register_allocator
        (weight_addr,) = alloc.allocate_addr(1)
        gp_for_preload = alloc.allocate_gp(2)
        self.emit(preload_addr_reg_asm(
            addr_reg_to_set=[weight_addr],
            available_registers=gp_for_preload,
            addr_reg_val=[weight.hbm_addr],
        ))
        alloc.free_gp(gp_for_preload)

        padded_vocab = lm_head_vocab_padding(vocab_size, self.blen)
        logits_name = self._scoped_name(name)
        self.allocate_vram_matrix(
            name=logits_name, rows=batch, cols=padded_vocab, strict=False
        )
        gp_regs = alloc.allocate_gp(6)
        self.emit(lm_head_asm(
            mlen=self.mlen,
            blen=self.blen,
            batch=batch,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            alive_registers=gp_regs,
            lm_head_weight_hbm_offset_reg=weight_addr,
            activation_base_address=self.get_vram_addr(hidden_states.name),
            result_base_address=self.get_vram_addr(logits_name),
        ))
        alloc.free_gp(gp_regs)
        alloc.free_addr([weight_addr])

        logits = VRAMMatrixVar(
            self, logits_name, (batch, padded_vocab), display_name=name
        )
        self._tensors[logits_name] = logits
        return logits


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
    head_dim: int,            # validated against the hardware HLEN
    build_dir: str = "./build",
    vocab: int = DECODE_VOCAB,
    kv_head_reuse: bool = False,
    kv_heads: int = 1,
    row_tile: int | None = None,
    settings_toml: str | None = None,
    activation_element_bits: int = 8,
    weight_element_bits: int = 8,
    kv_element_bits: int = 8,
    block_size: int = 8,
    scale_bits: int = 8,
    verbose: bool = True,
) -> dict:
    """Generate the per-layer batch-decode ISA and write it under build_dir.

    Args:
        kv_size:   TOTAL KV cache length (must be divisible by mlen=64 and > s_q).
                   The last s_q rows represent this step's new K/V -- see the
                   module docstring for the cache-append convention.
        hidden:    Residual-stream width; must equal mlen.
        inter:     FFN intermediate width.
        head_dim:  Per-head width; must match the hardware HLEN.
        vocab:     Vocabulary width of the LM head.
        kv_heads:  Number of packed KV heads; each adds one MLEN-wide W_Q/W_O group.
        row_tile:  Optional fixed query-row tile for controlled comparisons.
        settings_toml: Hardware configuration used by both lowering and emulator.

    Returns:
        dict with keys:
            isa                 -- generated assembly text
            mlen, blen, s_q     -- hardware shape used
            hidden              -- echo of input
            o_proj_vram_addr    -- VRAM address where the per-layer output lands
            logits_vram_addr    -- VRAM address where the LM head logits land
            vocab, padded_vocab -- LM head widths
            qrot/cos/sin/krot_vram_addr -- prestaged-tensor VRAM addresses
                                          (used by the test harness vram_preload)
    """
    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)

    geometry            = decode_geometry(
        settings_toml,
        kv_heads=kv_heads,
        kv_head_reuse=kv_head_reuse,
        row_tile=row_tile,
    )
    mlen, blen          = geometry["mlen"], geometry["blen"]
    real_data_ratio     = (8 * 8 + 8) / (8 * 8)
    s_q                 = geometry["batch"]
    hq, hkv             = geometry["query_heads"], geometry["kv_heads"]
    h_qkv               = geometry["head_dim"]
    total_q_dim         = hq * h_qkv
    k_padded_cols       = mlen                   # HW K cache row = mlen even though only hkv*h_qkv cols are real
    scale               = 1.0 / math.sqrt(h_qkv)

    assert head_dim == h_qkv,       f"head_dim ({head_dim}) must equal HLEN ({h_qkv})"
    assert hidden == mlen,           f"hidden ({hidden}) must equal MLEN ({mlen})"
    assert kv_size % mlen == 0,      f"kv_size ({kv_size}) must be a multiple of mlen ({mlen})"
    assert kv_size > s_q,            f"kv_size ({kv_size}) must exceed s_q ({s_q})"

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        real_data_ratio=real_data_ratio,
        hbm_v_prefetch_amount=geometry["hbm_v_prefetch_amount"],
        hbm_v_writeback_amount=geometry["hbm_v_writeback_amount"],
        hbm_element_width=activation_element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
    )
    prog.hlen = h_qkv
    prog.broadcast_amount = geometry["broadcast_amount"]
    prog.kv_head_reuse = kv_head_reuse
    prog.decode_geometry = geometry
    # Size the FP SRAM to what this geometry's softmax state needs. The RTL
    # declares FP_SRAM_DEPTH = 512 in definitions/configuration.svh, so a
    # requirement above that does not fit the current hardware.
    prog.fpram_allocator.total_size = geometry["fp_sram_required"]
    prog.fpram_allocator._vmm.total_size = geometry["fp_sram_required"]
    _reserve_fpram_constants(prog)

    # -------------------------------------------------------------------------
    # HBM inputs. The RoPE helpers (QROT, COS, SIN, KROT) are prestaged into
    # VRAM (the test's vram_preload), so load_batch on them emits no ISA. Only X
    # is fetched at runtime via H_PREFETCH_V. Prestaged addresses pack right
    # after X, which occupies VRAM[0 .. s_q*hidden).
    # -------------------------------------------------------------------------
    qrot_vram = s_q * hidden
    cos_vram  = qrot_vram + s_q * total_q_dim
    sin_vram  = cos_vram  + s_q * total_q_dim
    krot_vram = sin_vram  + s_q * total_q_dim

    activation_storage = {
        "hbm_element_width": activation_element_bits,
        "hbm_block_size": block_size,
        "hbm_scale_width": scale_bits,
        "precision_role": "activation",
    }
    weight_storage = {
        "hbm_element_width": weight_element_bits,
        "hbm_block_size": block_size,
        "hbm_scale_width": scale_bits,
        "precision_role": "weight",
    }
    kv_storage = {
        "hbm_element_width": kv_element_bits,
        "hbm_block_size": block_size,
        "hbm_scale_width": scale_bits,
    }
    x_input = prog.input("X", shape=(s_q, hidden), **activation_storage)
    qrot_input = prog.input(
        "QROT", shape=(s_q, total_q_dim),
        prestaged_vram_addr=qrot_vram, **activation_storage,
    )
    cos_input = prog.input(
        "COS", shape=(s_q, total_q_dim),
        prestaged_vram_addr=cos_vram, **activation_storage,
    )
    sin_input = prog.input(
        "SIN", shape=(s_q, total_q_dim),
        prestaged_vram_addr=sin_vram, **activation_storage,
    )
    krot_input = prog.input(
        "KROT", shape=(s_q, mlen),
        prestaged_vram_addr=krot_vram, **activation_storage,
    )
    wq_input = prog.input("W_Q", shape=(hidden, total_q_dim), **weight_storage)
    wk_input = prog.input("W_K", shape=(hidden, mlen), **weight_storage)
    wv_input = prog.input("W_V", shape=(hidden, mlen), **weight_storage)
    wo_input = prog.input("W_O", shape=(total_q_dim, hidden), **weight_storage)
    k_input = prog.input("K", shape=(kv_size, k_padded_cols), precision_role="key", **kv_storage)
    v_input = prog.input("V", shape=(kv_size, k_padded_cols), precision_role="value", **kv_storage)
    wgate_input = prog.input("W_gate", shape=(hidden, inter), **weight_storage)
    wup_input = prog.input("W_up", shape=(hidden, inter), **weight_storage)
    wdown_input = prog.input("W_down", shape=(inter, hidden), **weight_storage)
    lm_head_input = prog.input("W_lm_head", shape=(vocab, hidden), **weight_storage)

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

    # X_n = RMSNorm(X), written to its own buffer so X survives as the
    # post-attention residual without a copy pass.  Scratchpad pinned high
    # enough to avoid the prestaged tensors and X_norm itself.
    X_norm = prog.alloc("X_norm", s_q, hidden)
    rms_scratch_vram = krot_vram + s_q * mlen + s_q * hidden
    prog.rms_norm(X_batch,
                  eps_offset=FP_SLOT_RMS_EPS,
                  reci_hid_offset=FP_SLOT_RMS_RECI_HID,
                  scratchpad_vram_addr=rms_scratch_vram,
                  destination_var=X_norm)

    # Q, K, V projections
    Q     = prog.linear_projection(X_norm, wq_input, "Q")        # (s_q, total_q_dim)
    K_new = prog.linear_projection(X_norm, wk_input, "K_new")    # (s_q, mlen), first total_kv_dim_real cols real
    V_new = prog.linear_projection(X_norm, wv_input, "V_new")

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
    prog.store(
        K_new,
        name="K_appended",
        precision=1,
        hbm_element_width=kv_element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="key",
    )
    prog.store(
        V_new,
        name="V_appended",
        precision=1,
        hbm_element_width=kv_element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="value",
    )

    # GQA flash attention -- softmax(QK^T/sqrt(d)) . V.
    # Implementation in compiler/asm_templates/flashattn/overall.py.  For
    # ratio == blen the M_BTMM (batched) path is used; otherwise M_TMM
    # per-head.  Our ratio=4=blen so M_BTMM applies.
    O = prog.flash_attention(Q, k_input, v_input, scale,
                             hq=hq, hkv=hkv, h_qkv=h_qkv, causal_mask=None)

    # W_O projection, then the post-attention residual in place:
    # ffn_residual = W_O(attn_out) + X.  It stays live through the FFN branch,
    # which normalizes out of place, so it is the residual without a copy.
    ffn_residual = prog.linear_projection(O, wo_input, "ffn_residual")  # (s_q, hidden)
    prog.vram_add(ffn_residual, X_batch)

    # pre-FFN RMSNorm (reuses slots 3 and 4 -- same eps, same 1/hidden).
    O_proj = prog.alloc("O_proj", s_q, hidden)
    prog.rms_norm(ffn_residual,
                  eps_offset=FP_SLOT_RMS_EPS,
                  reci_hid_offset=FP_SLOT_RMS_RECI_HID,
                  destination_var=O_proj)

    # SwiGLU FFN in-place on the normalized copy.
    # ISA template lives in compiler/asm_templates/ffn_asm.py and includes
    # K-split when intermediate_size exceeds MRAM capacity.
    prog.ffn(O_proj, wgate_input, wup_input, wdown_input)

    # post-FFN residual.  O_proj = ffn_out + X'.
    prog.vram_add(O_proj, ffn_residual)

    # The per-layer pass ends here. The final RMSNorm and the LM head run once
    # after the full L-layer stack; with one layer emitted they follow directly.
    final_norm = prog.alloc("final_norm", s_q, hidden)
    prog.rms_norm(O_proj,
                  eps_offset=FP_SLOT_RMS_EPS,
                  reci_hid_offset=FP_SLOT_RMS_RECI_HID,
                  destination_var=final_norm)
    logits = prog.lm_head(final_norm, lm_head_input)

    # -------------------------------------------------------------------------
    # Emit ASM + return metadata so the test knows where to read the result
    # from.
    # -------------------------------------------------------------------------
    compilation_artifact = prog.compile_with_trace()
    gen_code = compilation_artifact.assembly
    asm_path = build_path / "generated_asm_code.asm"
    asm_path.write_text(gen_code)
    compiler_artifact_path = build_path / "compilation_artifact.json"
    compiler_artifact_path.write_text(
        json.dumps(
            compilation_artifact.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    o_proj_addr = prog.get_vram_addr(O_proj.name)
    logits_addr = prog.get_vram_addr(logits.name)
    if verbose:
        print(f"Decode ASM -- s_q={s_q}, kv_size={kv_size}, hidden={hidden}, inter={inter}")
        print(f"GQA params:  hq={hq}, hkv={hkv}, h_qkv={h_qkv}, scale={scale:.4f}")
        print(f"Softmax row tile: {geometry['softmax_row_tile']}; "
              f"FP SRAM required: {geometry['fp_sram_required']} f16 slots "
              f"(hardware depth = {geometry['fp_sram_depth']})")
        print(f"Output X'' at VRAM[{o_proj_addr}] (row {o_proj_addr // mlen})")
        print(f"Generated {len(gen_code.splitlines())} lines of ISA -> {asm_path}")

    return {
        "isa": gen_code,
        "compilation_artifact": compilation_artifact,
        "compilation_artifact_path": compiler_artifact_path,
        "mlen": mlen,
        "blen": blen,
        "s_q": s_q,
        "hidden": hidden,
        "o_proj_vram_addr": o_proj_addr,
        "logits_vram_addr": logits_addr,
        "vocab": vocab,
        "padded_vocab": logits.shape[1],
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
    parser.add_argument("--hidden",   type=int, default=None,
        help="Residual hidden size. Defaults to MLEN.")
    parser.add_argument("--inter",    type=int, default=128, help="FFN intermediate width")
    parser.add_argument("--head-dim", type=int, default=16,  help="(forced) per-head dim h_qkv")
    parser.add_argument("--kv-heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--kv-head-reuse", action="store_true")
    parser.add_argument("--row-tile", type=int, default=None,
        help="Optional fixed query-row tile (multiple of BLEN)")
    parser.add_argument("--build-dir", type=str, default="./build/decode")
    args = parser.parse_args()

    cli_geometry = decode_geometry(
        kv_heads=args.kv_heads,
        kv_head_reuse=args.kv_head_reuse,
        row_tile=args.row_tile,
    )
    generate_decode_asm(
        kv_size=args.kv_size,
        hidden=args.hidden if args.hidden is not None else cli_geometry["hidden"],
        inter=args.inter,
        head_dim=args.head_dim, build_dir=args.build_dir,
        kv_heads=args.kv_heads, kv_head_reuse=args.kv_head_reuse,
        row_tile=args.row_tile,
    )
