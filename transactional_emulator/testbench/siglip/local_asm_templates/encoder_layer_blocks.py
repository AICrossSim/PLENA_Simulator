from compiler.asm_templates.normalization_asm import layer_norm_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.gelu_asm import gelu_asm
from compiler.asm_templates.elementwise_add_vram_asm import elementwise_add_vram_asm
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
):
    """Emit the MLP sub-block: proj-up -> GELU -> proj-down."""
    asm = ""

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w1_hbm_offset_reg,
        activation_base_address=activation_base,
        result_base_address=mlp_inter_base,
        out_features=inter_dim,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

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
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w2_hbm_offset_reg,
        activation_base_address=mlp_inter_base,
        result_base_address=mlp_out_base,
        out_features=hidden_size,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

    return asm


def build_encoder_layer_asm(
    *,
    mlen,
    blen,
    vlen,
    batch,
    s_q,
    s_kv,
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
):
    """Emit one SigLIP encoder layer in SRAM-resident pipeline form."""
    asm = "; SigLIP Encoder Layer (ASM) Test\n"
    asm += "; LayerNorm1 -> FlashAttn -> Residual -> LayerNorm2 -> MLP -> Residual\n"

    asm += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2, 3, 4],
        available_registers=[1, 2, 3, 4],
        addr_reg_val=[k_hbm_offset, v_hbm_offset, w1_hbm_offset, w2_hbm_offset],
    )
    asm += reset_reg_asm(alive_registers=[1])

    # Stage 0: Save residual before LN1.
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

    # Stage 1: LayerNorm1 in-place on X.
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

    # Stage 2: Flash attention.
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
        alive_registers_int=alive_int,
        alive_registers_fp=alive_fp,
        vector_sram_base_address=0,
        fp_sram_start_address=flash_temp_fp_start,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
        attn_scale_fp_address=attn_scale_fp_slot,
        inf_fp_address=attn_ninf_fp_slot,
        causal_mask=False,
    )

    # Stage 3: Residual 1, attn_out += residual.
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=attn_base,
        src_base_address=residual_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    # Stage 4: Save residual buffer before LN2.
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
        src_base_address=attn_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 5: LayerNorm2 in-place on attention output.
    asm += layer_norm_asm(
        _eps_offset=ln_eps_fp_slot,
        reci_hid_offset=ln_reci_hid_fp_slot,
        alive_registers=[5, 6, 7],
        activation_base_address=attn_base,
        scratchpad_base_address=scratch_base,
        vlen=vlen,
        batch_size=s_q,
        hidden_dim=hidden_size,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7])

    # Stage 6: MLP block.
    asm += build_mlp_block(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=s_q,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        w1_hbm_offset_reg=3,
        w2_hbm_offset_reg=4,
        activation_base=attn_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        gelu_one_fp_slot=gelu_one_fp_slot,
        gelu_1702_fp_slot=gelu_1702_fp_slot,
    )

    # Stage 7: Residual 2, mlp_out += residual.
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=mlp_out_base,
        src_base_address=residual_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    return asm
