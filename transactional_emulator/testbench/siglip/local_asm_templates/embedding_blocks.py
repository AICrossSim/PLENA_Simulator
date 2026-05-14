from compiler.asm_templates import (
    elementwise_add_vram_asm,
    preload_act_asm,
    preload_addr_reg_asm,
    projection_asm,
    reset_reg_asm,
)


def build_embedding_projection_asm(
    *,
    title,
    shape_batch,
    in_features,
    out_features,
    effective_batch,
    mlen,
    blen,
    vlen,
    weight_hbm_offset,
    weight_hbm_end,
):
    gen_assembly_code = f"; {title}\\n"
    gen_assembly_code += f"; Shape: ({shape_batch}, {in_features}) @ ({in_features}, {out_features})\\n"

    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[weight_hbm_offset, weight_hbm_end],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=effective_batch,
        hidden_size=in_features,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=in_features,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])

    result_vram_offset = in_features * shape_batch

    gen_assembly_code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=effective_batch,
        hidden_size=in_features,
        out_features=out_features,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=result_vram_offset,
        rope_enabled=False,
    )

    return gen_assembly_code, result_vram_offset


def append_position_add_asm(
    *,
    gen_assembly_code,
    result_vram_offset,
    position_vram_offset,
    batch,
    out_features,
    vlen,
):
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=batch,
        hidden_size=out_features,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=position_vram_offset,
        activation_offset_reg=2,
        stride_size=out_features,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    num_result_vectors = (batch * out_features) // vlen
    gen_assembly_code += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=num_result_vectors,
        alive_registers=[1, 2],
        dst_base_address=result_vram_offset,
        src_base_address=position_vram_offset,
    )
    return gen_assembly_code
