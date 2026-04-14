"""PLENA backend implementation for Flash Attention operator."""


def flash_attention_plena(prog, Q, K, V, scale=None, hq=1, hkv=1, h_qkv=None):
    """PLENA backend: Flash Attention via PLENAProgram.

    Dispatches to one of two codegen paths based on shape:
      * MHA  (hq == hkv == 1) — online-softmax loop using PLENAProgram primitives
      * GQA  (hq // hkv > 1)  — fused codegen via `flash_attn_asm` template, packs
        `hq/hkv` Q heads into blen of M_BTMM (hardware GQA fusion, matches main)

    Args:
        prog:   PLENAProgram instance
        Q:      VRAMMatrixVar — Q in VRAM, shape (seq_len, hq*h_qkv)
        K:      InputVar — K in HBM, shape (seq_len, hkv*h_qkv_padded)
        V:      InputVar — V in HBM, shape (seq_len, hkv*h_qkv_padded)
        scale:  Attention scale (default 1/sqrt(head_dim))
        hq, hkv, h_qkv: GQA params. Defaults treat input as single-head MHA.

    Returns:
        VRAMMatrixVar for O, shape matching Q.
    """
    import math

    # Detect MHA vs GQA
    if hq == 1 and hkv == 1:
        return _flash_attention_mha(prog, Q, K, V, scale)

    # GQA: dispatch to fused codegen
    if h_qkv is None:
        raise ValueError("GQA mode requires h_qkv to be specified")
    return _flash_attention_gqa_fused(prog, Q, K, V, scale, hq, hkv, h_qkv)


def _flash_attention_mha(prog, Q, K, V, scale):
    """Single-head flash attention using PLENAProgram primitives (online softmax)."""
    import math

    seq_len, head_dim = Q.shape
    mlen = prog.mlen

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    num_q_blocks = seq_len // mlen
    num_k_blocks = seq_len // mlen

    S_block = prog.alloc("S", mlen, mlen)
    PV = prog.alloc("PV", mlen, head_dim)
    O = prog.alloc("O", seq_len, head_dim)

    for q_idx in range(num_q_blocks):
        prog.init_online_softmax(q_idx, O)

        for k_idx in range(num_k_blocks):
            prog.vram_sub_projection_T_to(
                Q,
                q_idx,
                K,
                k_idx,
                S_block,
                target_row_idx=0,
                target_col_idx=0,
            )
            prog.online_softmax_block(S_block, scale)
            prog.compute_pv(S_block, V, k_idx, PV, head_dim)
            prog.scale_o_row(O, q_idx)
            prog.vram_add(O, PV, dst_row_offset=q_idx * mlen)

        prog.final_scale_o(q_idx, O)

    return O


def _flash_attention_gqa_fused(prog, Q, K, V, scale, hq, hkv, h_qkv):
    """GQA flash attention using fused M_BTMM codegen (main branch template).

    Packs `hq/hkv` Q heads into the `blen` systolic dimension — one M_BTMM
    produces `ratio` head outputs in parallel.  Requires:
      (hq // hkv) == prog.blen, and (hq // hkv) * h_qkv == prog.mlen.
    """
    import math
    from compiler.asm_templates.flashattn import flash_attn_asm
    from compiler.asm_templates import preload_addr_reg_asm, reset_reg_asm

    ratio = hq // hkv
    mlen = prog.mlen
    blen = prog.blen
    vlen = mlen

    if ratio != blen:
        raise ValueError(
            f"GQA ratio hq/hkv={ratio} must equal blen={blen} (hardware packs "
            f"heads into blen).  Use ratio=4 with default blen."
        )
    if ratio * h_qkv != mlen:
        raise ValueError(
            f"GQA constraint: (hq/hkv)*h_qkv = {ratio * h_qkv} must equal "
            f"mlen={mlen}.  E.g. hq=4, hkv=1, h_qkv=16 with mlen=64."
        )

    s_q, q_total_dim = Q.shape
    s_kv, k_total_dim = K.shape

    if scale is None:
        scale = 1.0 / math.sqrt(h_qkv)

    # Allocate HBM addr registers for K and V (C_SET_ADDR_REG aN)
    # Make sure K/V HBM subst-matrix registry knows about them
    prog._ensure_hbm_sub_matrix_registered(K)
    prog._ensure_hbm_sub_matrix_registered(V)
    alloc = prog._compiler.register_allocator
    k_addr, v_addr = alloc.allocate_addr(2)
    gp_for_preload = alloc.allocate_gp(2)
    setup = preload_addr_reg_asm(
        addr_reg_to_set=[k_addr, v_addr],
        available_registers=gp_for_preload,
        addr_reg_val=[K.hbm_addr, V.hbm_addr],
    )
    alloc.free_gp(gp_for_preload)
    prog._compiler.generated_code += setup

    # Allocate VRAM buffers mirroring main's layout.
    #   S, PV each require mlen*mlen*ratio elements; O is s_q * (hq*h_qkv).
    # We let PLENAProgram's allocator handle placement — main's template uses
    # vector_sram_base_address + computed offsets for S/PV/O, so they must be
    # allocated contiguously starting right after Q.  allocate_vram_matrix
    # bump-allocates, which preserves that contiguity.
    q_vram_base = prog._compiler.get_vram_addr(Q.name)
    s_name = prog._scoped_name("_gqa_S")
    pv_name = prog._scoped_name("_gqa_PV")
    o_name = prog._scoped_name("O")

    # Express sizes as (rows, cols) that multiply to the required counts.
    prog._compiler.allocate_vram_matrix(name=s_name, rows=mlen * ratio, cols=mlen, strict=False)
    prog._compiler.allocate_vram_matrix(name=pv_name, rows=mlen * ratio, cols=mlen, strict=False)
    prog._compiler.allocate_vram_matrix(name=o_name, rows=s_q, cols=hq * h_qkv, strict=False)

    # Reserve FPRAM for multi-head softmax state.  main's flash_attn_asm
    # assumes slots 0..2 hold constants (0.0, scale, -inf, preloaded by the
    # test harness) and softmax state starts at fp_sram_start_address, with
    # 3 triples (m_old, m_res, l_old) per head, strided 3*br apart.
    # Reserve slots 0..2 first so our bump-allocated state lives at offset 3+.
    br = min(mlen, s_q)
    fp_allocs = prog._compiler.sub_matrix_manager.fpram_allocator
    if "_gqa_fp_const_zero" not in fp_allocs.allocations:
        fp_allocs.allocate(name="_gqa_fp_const_zero", size=1)
        fp_allocs.allocate(name="_gqa_fp_const_scale", size=1)
        fp_allocs.allocate(name="_gqa_fp_const_neg_inf", size=1)
    fp_state_size = 3 * br * ratio
    fp_start = prog._compiler.allocate_fpram(name="_gqa_softmax_state", size=fp_state_size)

    # Call main's fused GQA template
    asm = flash_attn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=1,
        hq=hq,
        hkv=hkv,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        alive_registers_int=list(range(1, 16)),
        alive_registers_fp=list(range(1, 8)),
        vector_sram_base_address=q_vram_base,
        fp_sram_start_address=fp_start,
        k_base_hbm_offset_reg=k_addr,
        v_base_hbm_offset_reg=v_addr,
    )
    prog._compiler.generated_code += asm

    # Release HBM addr regs (they're only needed during the call)
    alloc.free_addr([k_addr, v_addr])

    # Return O as a VRAMMatrixVar the caller can consume
    from plena_program import VRAMMatrixVar

    O = VRAMMatrixVar(prog, o_name, (s_q, hq * h_qkv), display_name="O")
    prog._tensors[o_name] = O
    return O
