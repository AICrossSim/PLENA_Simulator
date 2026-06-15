"""Output computation assembly code generation for Flash Attention."""

IMM2_BOUND = 2**18 - 1


def computing_o_code(
    mlen: int,
    stage: str,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
    m_res_base_address: int,
    pv_base_address: int,
    o_old_base_address: int,
    head_dim: int,
    q_head_num: int,
) -> str:
    """
    Args:
    head_dim: the head dimension
    mlen: the number of row of the QKT result
    alive_registers_int: the list of alive registers for fix point operations
    alive_registers_fp: the list of alive registers for floating point operations
    m_res_address: the address of the m_res
    pv_result_address: the address of the PV result
    o_old_base_address: the base address of the old O
    Description:
        This part of asm is for the computing of the O operation, mapping to line 10 process
        Assume the C_SET_V_MASK_REG is set already by the PV operation.

    """
    m_res_vector_address_register = alive_registers_int[0]
    o_old_vector_address_register = alive_registers_int[1]
    pv_vector_address_register = alive_registers_int[2]
    loop_register = alive_registers_int[3]

    m_res_fp_register = alive_registers_fp[0]
    generated_code = "; Computing O Code \n"
    assert head_dim <= mlen, "head_dim must be less than or equal to mlen"
    # break diag(MLEN) * (MLEN * Head_dim) into diag(MLEN) * [(MLEN * MLEN) ... (MLEN * MLEN)]

    # load o_old base address
    assert o_old_base_address < IMM2_BOUND, f"o_old_base_address must be less than {IMM2_BOUND}"
    generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp0, {o_old_base_address} \n"

    # reload m_res base address
    assert m_res_base_address < IMM2_BOUND, f"m_res_base_address must be less than {IMM2_BOUND}"
    generated_code += f"S_ADDI_INT gp{m_res_vector_address_register}, gp0, {m_res_base_address} \n"

    # load pv base address
    assert pv_base_address < IMM2_BOUND, f"pv_base_address must be less than {IMM2_BOUND}"
    generated_code += f"S_ADDI_INT gp{pv_vector_address_register}, gp0, {pv_base_address} \n"

    if stage == "prefill":
        # loop over different row of m_res using hardware loop
        generated_code += f"C_LOOP_START gp{loop_register}, {mlen} \n"
        # load m_res (using indirect addressing)
        generated_code += f"S_LD_FP f{m_res_fp_register}, gp{m_res_vector_address_register}, 0 \n"
        # boardcast m_res to multiply with a row of a block of O_old and write to o_old
        generated_code += (
            f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{m_res_fp_register}, 1 \n"
        )
        # # add pv row to o_old
        generated_code += f"V_ADD_VV gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, gp{pv_vector_address_register}, 1 \n"
        # # update o_old base address
        generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, {q_head_num * head_dim} \n"
        # # update pv base address
        generated_code += f"S_ADDI_INT gp{pv_vector_address_register}, gp{pv_vector_address_register}, {mlen} \n"
        # # update m_res address
        generated_code += f"S_ADDI_INT gp{m_res_vector_address_register}, gp{m_res_vector_address_register}, 1 \n"
        generated_code += f"C_LOOP_END gp{loop_register} \n"
        # now o_old should contain the result of the current o, diag(exp(m_res)) * O_old + PV
    else:
        # load m_res (using indirect addressing)
        generated_code += f"S_LD_FP f{m_res_fp_register}, gp{m_res_vector_address_register}, 0 \n"
        # boardcast m_res to multiply with a row of a block of O_old and write to o_old
        generated_code += (
            f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{m_res_fp_register}, 1 \n"
        )
        # # add pv row to o_old
        generated_code += f"V_ADD_VV gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, gp{pv_vector_address_register}, 1 \n"
    return generated_code


def computing_row_wise_scaling_code(
    mlen: int,
    stage: str,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
    o_old_base_address: int,
    l_old_base_address: int,
    o_row_stride: int,
    use_mask: bool = True,
) -> str:
    """
    line 12 in flash attention algorithm
    mlen: the number of row of the QKT result
    alive_registers_int: the list of alive registers for fix point operations
    alive_registers_fp: the list of alive registers for floating point operations
    o_old_base_address: the base address of the old O
    l_old_base_address: the base address of the l values
    o_row_stride: the stride between consecutive rows in O (= q_head_num * head_dim)
    use_mask: whether to use V_MASK for selective head scaling (default True for packed format)
    """
    o_old_vector_address_register = alive_registers_int[0]
    l_old_vector_address_register = alive_registers_int[1]
    loop_register = alive_registers_int[2]
    l_old_fp_register = alive_registers_fp[0]

    mask_en = 1 if use_mask else 0

    generated_code = "; Row-wise Scaling Code (1/l normalization) \n"
    # load l_old base address
    generated_code += f"S_ADDI_INT gp{l_old_vector_address_register}, gp0, {l_old_base_address} \n"
    # load o_old base address
    generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp0, {o_old_base_address} \n"

    if stage == "prefill":
        # loop over different row of Br using hardware loop
        generated_code += f"C_LOOP_START gp{loop_register}, {mlen} \n"
        # load l_old (using indirect addressing through l_old_vector_address_register)
        generated_code += f"S_LD_FP f{l_old_fp_register}, gp{l_old_vector_address_register}, 0 \n"
        # compute the inverse of l_old
        generated_code += f"S_RECI_FP f{l_old_fp_register}, f{l_old_fp_register} \n"
        # multiply o_old with the inverse of l_old (use mask to select head's elements)
        generated_code += f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{l_old_fp_register}, {mask_en} \n"
        # update o_old base address
        generated_code += (
            f"S_ADDI_INT gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, {o_row_stride} \n"
        )
        # update l_old address
        generated_code += f"S_ADDI_INT gp{l_old_vector_address_register}, gp{l_old_vector_address_register}, 1 \n"
        generated_code += f"C_LOOP_END gp{loop_register} \n"
    else:
        # load l_old (using indirect addressing through l_old_vector_address_register)
        generated_code += f"S_LD_FP f{l_old_fp_register}, gp{l_old_vector_address_register}, 0 \n"
        # compute the inverse of l_old
        generated_code += f"S_RECI_FP f{l_old_fp_register}, f{l_old_fp_register} \n"
        # multiply o_old with the inverse of l_old (use mask to select head's elements)
        generated_code += f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{l_old_fp_register}, {mask_en} \n"

    return generated_code
