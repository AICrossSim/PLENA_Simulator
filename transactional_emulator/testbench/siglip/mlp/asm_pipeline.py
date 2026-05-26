from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.asm_templates.siglip import build_mlp_block


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
    """Build standalone MLP pipeline ASM using compiler-owned SigLIP block."""
    gen_assembly_code = f"; {title}\\n"
    gen_assembly_code += (
        f"; Shape: ({effective_batch}, {hidden_size}) -> ({effective_batch}, {hidden_size})\\n"
    )
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

    gen_assembly_code += build_mlp_block(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=effective_batch,
        hidden_size=hidden_size,
        inter_dim=aligned_intermediate,
        w1_hbm_offset_reg=weight_hbm_reg,
        w2_hbm_offset_reg=weight_down_hbm_reg,
        activation_base=0,
        mlp_inter_base=intermediate_vram_offset,
        mlp_out_base=final_vram_offset,
        scratch_base=scratch_vram_addr,
        gelu_one_fp_slot=gelu_one_fp_slot,
        gelu_1702_fp_slot=gelu_1702_fp_slot,
        include_gelu=True,
    )

    return gen_assembly_code, final_vram_offset
