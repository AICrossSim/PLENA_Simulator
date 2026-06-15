"""QKT multiplication assembly code generation for Flash Attention."""

IMM2_BOUND = 2**18 - 1


def qkt_multiply(
    d: int,
    mlen: int,
    stage: str,
    alive_registers: list[int],
    q_base_address: int,
    k_base_hbm_offset_reg: int,
    q_head_index: int,
    k_head_index: int,
    s_base_address: int = 0,
) -> str:
    """
    Args:
        mlen: the number of rows in the first matrix.
        (mlen // hlen): the number of Q heads that could process with the same K head.
        hlen is assumed to be equal to d
        d: the head dimension
        alive_registers: the list of alive registers.
        q_base_address: the base address of the query.
        k_base_address: the base address of the key.
    Description:
        This part of asm code gen template is used to compute QKT result.
        Assuming Q is in dim of (1, MLEN, MLEN//HLEN, HLEN) for prefill and (1, 1, MLEN//HLEN, HLEN) for decode, K is in dim of (1, MLEN, 1, HLEN)
        This template will perform, single batch, MLEN tiled, per KV head, QKT multiplication.
        Producing the results in shape of [1, MLEN//HLEN, MLEN, MLEN] for prefill and [1, MLEN//HLEN, MLEN, 1] for decode.
    """
    q_base_register = alive_registers[0]
    k_base_register = alive_registers[1]
    s_base_register = q_base_register
    generated_code = "; QKT Per KV Head Multiplication \n"

    # Set Q row stride for M_BTMM
    # M_BTMM uses mm_load_stride to determine Q row spacing: v_addr + i * mlen * stride_len
    # For Q layout [s_q, hq, d], each token has hq * d elements, so stride = (hq * d) / mlen

    # Prefetch K from HBM
    generated_code += f"S_ADDI_INT gp{q_base_register}, gp0, {q_base_address + q_head_index * d} \n"
    generated_code += f"S_ADDI_INT gp{k_base_register}, gp0, {k_head_index * d} \n"

    # Use stride_en=0 for contiguous prefetch to avoid 64-byte alignment issues
    # When stride < 64 elements, strided access causes unaligned HBM reads
    # Parameter order: rd, rs1, rs2, rstride(stride_en), funct1(scale_en)
    generated_code += f"H_PREFETCH_M gp0, gp{k_base_register}, a{k_base_hbm_offset_reg}, 0, 1 \n"

    # QKT multiply
    if stage == "prefill":
        generated_code += f"M_BTMM 0, gp{q_base_register}, gp0 \n"
        assert s_base_address + q_head_index * mlen * mlen < IMM2_BOUND, "S base address is too large"
        generated_code += f"S_ADDI_INT gp{s_base_register}, gp0, {s_base_address + q_head_index * mlen * mlen} \n"
        generated_code += f"M_BMM_WO gp{s_base_register}, 0 \n"
    else:
        generated_code += f"M_BTMV 0, gp{q_base_register}, gp0 \n"
        assert s_base_address + q_head_index * mlen < IMM2_BOUND, "S base address is too large"
        generated_code += f"S_ADDI_INT gp{s_base_register}, gp0, {s_base_address + q_head_index * mlen} \n"
        generated_code += f"M_BMV_WO gp{s_base_register}, 0 \n"

    return generated_code
