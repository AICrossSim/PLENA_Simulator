from compiler.asm_templates import (
    gelu_asm,
    preload_act_asm,
    preload_addr_reg_asm,
    projection_asm,
    reset_reg_asm,
)


def build_mlp_pipeline_asm(
    *,
    title,
    effective_batch,
    hidden_size,
    aligned_intermediate,
    real_data_ratio,
    mlen,
    blen,
    vlen,
    weight_up_numel,
    weight_hbm_reg,
    weight_down_hbm_reg,
    gelu_one_fp_slot,
    gelu_1702_fp_slot,
):
    gen_assembly_code = f"; {title}\\n"
    gen_assembly_code += f"; Shape: ({effective_batch}, {hidden_size}) -> ({effective_batch}, {hidden_size})\\n"
    gen_assembly_code += f"; Intermediate padded to {aligned_intermediate}\\n"

    act_hbm_size = int(hidden_size * effective_batch * real_data_ratio)
    act_hbm_size = ((act_hbm_size + 63) // 64) * 64
    weight_hbm_offset = act_hbm_size
    weight_up_size = int(weight_up_numel * real_data_ratio)
    weight_up_size = ((weight_up_size + 63) // 64) * 64
    weight_down_hbm_offset = ((weight_hbm_offset + weight_up_size + 63) // 64) * 64

    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[weight_hbm_reg, weight_down_hbm_reg],
        available_registers=[weight_hbm_reg, weight_down_hbm_reg],
        addr_reg_val=[weight_hbm_offset, weight_down_hbm_offset],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[weight_hbm_reg])

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

    intermediate_vram_offset = effective_batch * hidden_size
    final_vram_offset = intermediate_vram_offset + effective_batch * aligned_intermediate
    scratch_vram_addr = final_vram_offset + effective_batch * hidden_size

    gen_assembly_code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=effective_batch,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=weight_hbm_reg,
        activation_base_address=0,
        result_base_address=intermediate_vram_offset,
        rope_enabled=False,
        out_features=aligned_intermediate,
        scratch_base_address=scratch_vram_addr,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

    gen_assembly_code += gelu_asm(
        const_one_fp_address=gelu_one_fp_slot,
        const_1702_fp_address=gelu_1702_fp_slot,
        alive_registers=[1, 2, 3],
        activation_base_address=intermediate_vram_offset,
        scratchpad_base_address=scratch_vram_addr,
        vlen=vlen,
        batch_size=effective_batch,
        hidden_dim=aligned_intermediate,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

    gen_assembly_code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=effective_batch,
        hidden_size=aligned_intermediate,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=weight_down_hbm_reg,
        activation_base_address=intermediate_vram_offset,
        result_base_address=final_vram_offset,
        rope_enabled=False,
        out_features=hidden_size,
        scratch_base_address=scratch_vram_addr,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

    return gen_assembly_code, final_vram_offset
