from compiler.asm_templates import (
    layer_norm_asm,
    preload_act_asm,
    reset_reg_asm,
)


def build_layernorm_inplace_asm(
    *,
    title,
    effective_batch,
    hidden_size,
    vlen,
    eps_fp_slot,
    reci_hid_fp_slot,
    affine_weight_vram_offset=None,
    affine_bias_vram_offset=None,
):
    gen_assembly_code = f"; {title}\\n"
    gen_assembly_code += f"; Shape: ({effective_batch}, {hidden_size})\\n"

    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=effective_batch,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    scratch_vram_addr = effective_batch * hidden_size
    gen_assembly_code += layer_norm_asm(
        _eps_offset=eps_fp_slot,
        reci_hid_offset=reci_hid_fp_slot,
        alive_registers=[1, 2, 3],
        activation_base_address=0,
        scratchpad_base_address=scratch_vram_addr,
        vlen=vlen,
        batch_size=effective_batch,
        hidden_dim=hidden_size,
        affine_weight_base_address=affine_weight_vram_offset,
        affine_bias_base_address=affine_bias_vram_offset,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

    return gen_assembly_code
