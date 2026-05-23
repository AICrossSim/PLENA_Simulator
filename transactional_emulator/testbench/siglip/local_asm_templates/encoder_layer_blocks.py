from compiler.asm_templates.normalization_asm import layer_norm_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.gelu_asm import gelu_asm
from compiler.asm_templates.elementwise_add_vram_asm import elementwise_add_vram_asm
from compiler.asm_templates._imm import load_large_int_str
from compiler.asm_templates._imm import addi_large_int_str
from compiler.asm_templates.preload_addr_reg import preload_addr_reg_asm
from compiler.asm_templates.reset_reg_asm import reset_reg_asm
from compiler.asm_templates.flashattn.overall import flash_attn_asm
from compiler.asm_templates.flashattn.reset import reset_vssram_code


def build_mlp_block(
    *,
    mlen,
    blen,
    vlen,
    batch,
    hidden_size,
    inter_dim,
    w1_hbm_offset_reg,
    w2_hbm_offset_reg,
    activation_base,
    mlp_inter_base,
    mlp_out_base,
    scratch_base,
    gelu_one_fp_slot,
    gelu_1702_fp_slot,
    include_gelu=True,
):
    """Emit the MLP sub-block: proj-up -> GELU -> proj-down."""
    asm = ""

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        vlen=vlen,
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w1_hbm_offset_reg,
        activation_base_address=activation_base,
        result_base_address=mlp_inter_base,
        out_features=inter_dim,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

    if include_gelu:
        asm += gelu_asm(
            const_one_fp_address=gelu_one_fp_slot,
            const_1702_fp_address=gelu_1702_fp_slot,
            alive_registers=[5, 6, 7],
            activation_base_address=mlp_inter_base,
            scratchpad_base_address=scratch_base,
            vlen=vlen,
            batch_size=batch,
            hidden_dim=inter_dim,
        )
        asm += reset_reg_asm(alive_registers=[5, 6, 7])

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=inter_dim,
        vlen=vlen,
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w2_hbm_offset_reg,
        activation_base_address=mlp_inter_base,
        result_base_address=mlp_out_base,
        out_features=hidden_size,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

    return asm


def _pack_ln1_seq_to_head_major_q(
    *,
    s_q,
    hq,
    d_padded,
    x_base,
    q_base,
):
    """Pack LN1 output from seq-major [s_q, hq, d] to head-major [hq, s_q, d]."""
    asm = "; Repack LN1 output to flash-attn Q (head-major)\n"

    asm += reset_vssram_code(
        reset_start_address=q_base,
        vect_dim=d_padded,
        per_stride_dim=s_q * hq,
        reset_stride=d_padded,
        reset_amount=1,
        alive_registers_int=[10, 11, 12],
    )

    # Register map:
    # - gp10/gp11: current dst/src vector pointers
    # - gp12/gp13: head-base dst/src pointers for outer loop
    # - gp14/gp15: outer/inner loop counters
    # - gp9: temp register for large immediate adds
    dst_reg = 10
    src_reg = 11
    dst_head_base_reg = 12
    src_head_base_reg = 13
    outer_loop_reg = 14
    inner_loop_reg = 15
    temp_reg = 9

    asm += load_large_int_str(dst_head_base_reg, q_base)
    asm += load_large_int_str(src_head_base_reg, x_base)

    if hq > 0 and s_q > 0:
        asm += f"C_LOOP_START gp{outer_loop_reg}, {hq}\n"
        asm += f"S_ADDI_INT gp{dst_reg}, gp{dst_head_base_reg}, 0\n"
        asm += f"S_ADDI_INT gp{src_reg}, gp{src_head_base_reg}, 0\n"

        asm += f"C_LOOP_START gp{inner_loop_reg}, {s_q}\n"
        asm += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        asm += addi_large_int_str(dst_reg, dst_reg, d_padded, temp_reg)
        asm += addi_large_int_str(src_reg, src_reg, hq * d_padded, temp_reg)
        asm += f"C_LOOP_END gp{inner_loop_reg}\n"

        asm += addi_large_int_str(dst_head_base_reg, dst_head_base_reg, s_q * d_padded, temp_reg)
        asm += addi_large_int_str(src_head_base_reg, src_head_base_reg, d_padded, temp_reg)
        asm += f"C_LOOP_END gp{outer_loop_reg}\n"

    asm += reset_reg_asm(alive_registers=[9, 10, 11, 12, 13, 14, 15])
    return asm


def _pack_chunk_major_to_head_major_q(
    *,
    s_q,
    hq,
    d_padded,
    x_base,
    q_base,
):
    """Pack chunk-major [hq, s_q, d] to head-major [hq, s_q, d]."""
    asm = "; Repack projection output chunk-major -> flash-attn Q (head-major)\n"

    asm += reset_vssram_code(
        reset_start_address=q_base,
        vect_dim=d_padded,
        per_stride_dim=s_q * hq,
        reset_stride=d_padded,
        reset_amount=1,
        alive_registers_int=[10, 11, 12],
    )

    # Source and destination layouts are both [hq, s_q, d_padded], so this is a
    # contiguous vector copy using nested loops.
    dst_reg = 10
    src_reg = 11
    outer_loop_reg = 14
    inner_loop_reg = 15
    temp_reg = 9

    asm += load_large_int_str(dst_reg, q_base)
    asm += load_large_int_str(src_reg, x_base)

    if hq > 0 and s_q > 0:
        asm += f"C_LOOP_START gp{outer_loop_reg}, {hq}\n"
        asm += f"C_LOOP_START gp{inner_loop_reg}, {s_q}\n"
        asm += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        asm += addi_large_int_str(dst_reg, dst_reg, d_padded, temp_reg)
        asm += addi_large_int_str(src_reg, src_reg, d_padded, temp_reg)
        asm += f"C_LOOP_END gp{inner_loop_reg}\n"
        asm += f"C_LOOP_END gp{outer_loop_reg}\n"

    asm += reset_reg_asm(alive_registers=[9, 10, 11, 14, 15])
    return asm


def _pack_seq_major_to_block_major(
    *,
    seq_len,
    num_blocks,
    block_size,
    src_base,
    dst_base,
    comment,
):
    """Pack [seq_len, num_blocks, block_size] -> [num_blocks, seq_len, block_size]."""
    asm = f"; {comment}\n"

    asm += reset_vssram_code(
        reset_start_address=dst_base,
        vect_dim=block_size,
        per_stride_dim=seq_len * num_blocks,
        reset_stride=block_size,
        reset_amount=1,
        alive_registers_int=[10, 11, 12],
    )

    # Register map:
    # - gp10/gp11: current dst/src vector pointers
    # - gp12/gp13: block-base dst/src pointers for outer loop
    # - gp14/gp15: outer/inner loop counters
    # - gp9: temp register for large immediate adds
    dst_reg = 10
    src_reg = 11
    dst_block_base_reg = 12
    src_block_base_reg = 13
    outer_loop_reg = 14
    inner_loop_reg = 15
    temp_reg = 9

    asm += load_large_int_str(dst_block_base_reg, dst_base)
    asm += load_large_int_str(src_block_base_reg, src_base)

    if num_blocks > 0 and seq_len > 0:
        asm += f"C_LOOP_START gp{outer_loop_reg}, {num_blocks}\n"
        asm += f"S_ADDI_INT gp{dst_reg}, gp{dst_block_base_reg}, 0\n"
        asm += f"S_ADDI_INT gp{src_reg}, gp{src_block_base_reg}, 0\n"

        asm += f"C_LOOP_START gp{inner_loop_reg}, {seq_len}\n"
        asm += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        asm += addi_large_int_str(dst_reg, dst_reg, block_size, temp_reg)
        asm += addi_large_int_str(src_reg, src_reg, num_blocks * block_size, temp_reg)
        asm += f"C_LOOP_END gp{inner_loop_reg}\n"

        asm += addi_large_int_str(dst_block_base_reg, dst_block_base_reg, seq_len * block_size, temp_reg)
        asm += addi_large_int_str(src_block_base_reg, src_block_base_reg, block_size, temp_reg)
        asm += f"C_LOOP_END gp{outer_loop_reg}\n"

    asm += reset_reg_asm(alive_registers=[9, 10, 11, 12, 13, 14, 15])
    return asm


def build_encoder_layer_asm(
    *,
    mlen,
    blen,
    vlen,
    batch,
    s_q,
    s_kv,
    s_kv_valid=None,
    hq,
    hkv,
    h_qkv,
    hidden_size,
    inter_dim,
    x_base,
    attn_base,
    residual_base,
    mlp_inter_base,
    mlp_out_base,
    scratch_base,
    k_hbm_offset,
    v_hbm_offset,
    w1_hbm_offset,
    w2_hbm_offset,
    ln_eps_fp_slot,
    ln_reci_hid_fp_slot,
    gelu_one_fp_slot,
    gelu_1702_fp_slot,
    attn_scale_fp_slot,
    attn_ninf_fp_slot,
    flash_temp_fp_start,
    q_base=None,
    q_seq_base=None,
    wq_hbm_offset=None,
    q_bias_base=None,
    debug_flash_tile_trace_base=None,
    debug_attn_snapshot_base=None,
    include_final_residual=True,
    include_gelu=True,
):
    """Emit one SigLIP encoder layer in SRAM-resident pipeline form."""
    asm = "; SigLIP Encoder Layer (ASM) Test\n"
    asm += "; LayerNorm1 -> FlashAttn -> Residual -> LayerNorm2 -> MLP -> Residual\n"

    # Shape legend used below:
    # - S = s_q (query sequence length)
    # - H = hidden_size
    # - V = vlen
    # - NB = H // V (number of V-sized hidden blocks)
    # - D = h_qkv (per-head hidden, padded to mlen when needed)
    # - HQ = hq
    # Memory layouts used in this block:
    # - chunk-major: [NB, S, V]  (flat index: ((b * S) + t) * V + j)
    # - token-major: [S, NB, V]  (flat index: ((t * NB) + b) * V + j)
    # - head-major Q: [HQ, S, D] (flat index: ((h * S) + t) * D + j)
    # Stage outputs in order:
    # - Input X at x_base: chunk-major [NB, S, V]
    # - Stage 0 residual snapshot at residual_base: token-major [S, NB, V]
    # - Stage 1 LN1 output at x_base: chunk-major [NB, S, V]
    # - Stage 1.5 Q projection (if enabled): q_seq_base chunk-major [NB, S, V], q_base head-major [HQ, S, D]
    # - Stage 2 flash-attn output at attn_base: token-major [S, NB, V]
    # - Stage 3 x_res1 at attn_base: token-major [S, NB, V]
    # - Stage 5 repack x_res1 to ln2_input_base/x_base: chunk-major [NB, S, V]
    # - Stage 4 residual snapshot for final add at residual_base: chunk-major [NB, S, V]
    # - Stage 5 LN2 output at x_base: chunk-major [NB, S, V]
    # - Stage 6 MLP output at mlp_out_base: chunk-major [NB, S, V]
    # - Stage 7 final output at mlp_out_base: chunk-major [NB, S, V]

    if wq_hbm_offset is None:
        asm += preload_addr_reg_asm(
            addr_reg_to_set=[1, 2, 3, 4],
            available_registers=[1, 2, 3, 4],
            addr_reg_val=[k_hbm_offset, v_hbm_offset, w1_hbm_offset, w2_hbm_offset],
        )
    else:
        asm += preload_addr_reg_asm(
            addr_reg_to_set=[1, 2, 3, 4, 5],
            available_registers=[1, 2, 3, 4, 5],
            addr_reg_val=[k_hbm_offset, v_hbm_offset, w1_hbm_offset, w2_hbm_offset, wq_hbm_offset],
        )
    asm += reset_reg_asm(alive_registers=[1])

    # Stage 0: Save X residual in token-major layout before LN1.
    # Shape after Stage 0 (residual_base): token-major [S, NB, V].
    # x_base holds X in chunk-major [hidden//vlen, s_q, vlen].
    # Flash-attn writes its output at attn_base in token-major [s_q, hidden//vlen, vlen].
    # We must save the residual in the same token-major layout so Stage 3
    # (attn_out += residual) performs the flat element-wise add correctly.
    asm += _pack_seq_major_to_block_major(
        seq_len=hidden_size // vlen,
        num_blocks=s_q,
        block_size=vlen,
        src_base=x_base,
        dst_base=residual_base,
        comment="Save X residual in token-major layout (chunk->token) for Stage 3 residual add",
    )

    # Stage 1: LayerNorm1 in-place on X.
    # Shape after Stage 1 (x_base): chunk-major [NB, S, V].
    asm += layer_norm_asm(
        _eps_offset=ln_eps_fp_slot,
        reci_hid_offset=ln_reci_hid_fp_slot,
        alive_registers=[5, 6, 7],
        activation_base_address=x_base,
        scratchpad_base_address=scratch_base,
        vlen=vlen,
        batch_size=s_q,
        hidden_dim=hidden_size,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7])

    # Stage 1.5: Build flash-attn Q on-chip from LN1 output.
    # LN1 seq-major X at x_base -> projection_asm with WQ -> q_seq_base,
    # optional bias add from q_bias_base, then repack to head-major at q_base.
    # Shape after Stage 1.5:
    # - q_seq_base: chunk-major [NB, S, V]
    # - q_base: head-major [HQ, S, D]
    if q_base is not None:
        if q_seq_base is not None and wq_hbm_offset is not None:
            asm += projection_asm(
                mlen=mlen,
                blen=blen,
                batch=s_q,
                hidden_size=hidden_size,
                vlen=vlen,
                alive_registers=[4, 5, 6, 7, 8, 9],
                w_base_hbm_offset_reg=5,
                activation_base_address=x_base,
                result_base_address=q_seq_base,
                out_features=hidden_size,
                scratch_base_address=scratch_base,
                rope_enabled=False,
            )
            asm += reset_reg_asm(alive_registers=[4, 5, 6, 7, 8, 9])

            if q_bias_base is not None:
                asm += elementwise_add_vram_asm(
                    vlen=vlen,
                    num_vectors=(s_q * hidden_size) // vlen,
                    alive_registers=[10, 11],
                    dst_base_address=q_seq_base,
                    src_base_address=q_bias_base,
                )
                asm += reset_reg_asm(alive_registers=[10, 11])

            asm += _pack_chunk_major_to_head_major_q(
                s_q=s_q,
                hq=hq,
                d_padded=h_qkv,
                x_base=q_seq_base,
                q_base=q_base,
            )
        else:
            asm += _pack_ln1_seq_to_head_major_q(
                s_q=s_q,
                hq=hq,
                d_padded=h_qkv,
                x_base=x_base,
                q_base=q_base,
            )

    # Stage 2: Flash attention.
    # Shape after Stage 2 (attn_base/o_old_base): token-major [S, NB, V].
    asm += "; Flash attention block\n"
    alive_int = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    alive_fp = [1, 2, 3, 4, 5, 6]
    asm += flash_attn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        hq=hq,
        hkv=hkv,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        kv_valid_len=s_kv_valid,
        alive_registers_int=alive_int,
        alive_registers_fp=alive_fp,
        vector_sram_base_address=(x_base if q_base is None else q_base),
        fp_sram_start_address=flash_temp_fp_start,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
        attn_scale_fp_address=attn_scale_fp_slot,
        inf_fp_address=attn_ninf_fp_slot,
        causal_mask=False,
        debug_tile_trace_base=debug_flash_tile_trace_base,
    )

    if debug_attn_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_attn_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_attn_snapshot_base,
            src_base_address=attn_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 3: Residual 1, attn_out += residual.
    # Shape after Stage 3 (attn_base): token-major [S, NB, V].
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=attn_base,
        src_base_address=residual_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    # Stage 5: Repack x_res1 (token-major at attn_base) -> chunk-major at x_base for LN2/MLP.
    # Shape after repack (ln2_input_base/x_base): chunk-major [NB, S, V].
    # Must run BEFORE Stage 4 so we save the chunk-major version as residual for Stage 7.
    ln2_input_base = x_base
    asm += _pack_seq_major_to_block_major(
        seq_len=s_q,
        num_blocks=hidden_size // vlen,
        block_size=vlen,
        src_base=attn_base,
        dst_base=ln2_input_base,
        comment="Repack x_res1 (token-major) -> chunk-major for LN2/MLP",
    )

    # Stage 4: Save chunk-major x_res1 (now at x_base/ln2_input_base) as residual for
    # the final MLP residual add (Stage 7).  The token-major X saved at Stage 0 is no
    # longer needed — it has been consumed by Stage 3.
    # Shape after Stage 4 (residual_base): chunk-major [NB, S, V].
    asm += reset_vssram_code(
        reset_start_address=residual_base,
        vect_dim=vlen,
        per_stride_dim=hidden_size // vlen,
        reset_stride=hidden_size,
        reset_amount=s_q,
        alive_registers_int=[10, 11, 12],
    )
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=residual_base,
        src_base_address=x_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 5: LayerNorm2 in-place.
    # Shape after Stage 5 (ln2_input_base/x_base): chunk-major [NB, S, V].
    asm += layer_norm_asm(
        _eps_offset=ln_eps_fp_slot,
        reci_hid_offset=ln_reci_hid_fp_slot,
        alive_registers=[5, 6, 7],
        activation_base_address=ln2_input_base,
        scratchpad_base_address=scratch_base,
        vlen=vlen,
        batch_size=s_q,
        hidden_dim=hidden_size,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7])

    # Stage 6: MLP block.
    # Shape after Stage 6 (mlp_out_base): chunk-major [NB, S, V].
    asm += build_mlp_block(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=s_q,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        w1_hbm_offset_reg=3,
        w2_hbm_offset_reg=4,
        activation_base=ln2_input_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        gelu_one_fp_slot=gelu_one_fp_slot,
        gelu_1702_fp_slot=gelu_1702_fp_slot,
        include_gelu=include_gelu,
    )

    if include_final_residual:
        # Stage 7: Residual 2, mlp_out += residual.
        # Shape after Stage 7 (mlp_out_base): chunk-major [NB, S, V].
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=mlp_out_base,
            src_base_address=residual_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])

    return asm
