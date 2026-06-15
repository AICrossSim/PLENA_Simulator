"""PV multiplication assembly code generation for Flash Attention."""

IMM2_BOUND = 2**18 - 1


def computing_pv_code(
    head_dim: int,
    blen: int,
    mlen: int,
    vlen: int,
    stage: str,
    alive_registers: list[int],
    p_base_address: int,
    v_base_hbm_offset_reg: int,
    q_head_index: int,
    v_head_index: int,
    output_base_address: int,
    head_offset: int,
    v_msram_base: int = 0,  # MSRAM base address for V (can be 0, prefetched after K is used)
) -> str:
    """
    Compute PV = P @ V and write directly to packed output format.

    Single Head PV Multiplication. (MLEN, MLEN) @ (MLEN, H_QKV) -> (MLEN, H_QKV) Prefill Version
    Single Head PV Multiplication. (1, MLEN) @ (MLEN, H_QKV) -> (MLEN, H_QKV) Decode Version

    Args:
        head_dim: dimension per head
        q_head_num: total number of Q heads
        blen: block length (systolic array dimension)
        mlen: number of rows
        vlen: vector length
        alive_registers: available registers
        p_base_address: base address of softmax scores P
        v_base_hbm_offset_reg: HBM address register for V
        q_head_index: current Q head index
        v_head_index: current KV head index
        output_base_address: base address for packed output
        head_offset: offset within each row for this head (= q_head_index * head_dim)
        v_msram_base: MSRAM base address for V (can be 0 since K was already used)
    """
    generated_code = "; PV Per KV Head Multiplication (packed output) \n"
    p_base_register = alive_registers[0]
    v_base_register = alive_registers[1]
    out_base_register = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]
    out_col_register = alive_registers[5]

    assert p_base_address + q_head_index * mlen * mlen < IMM2_BOUND, f"p_base_address must be less than {IMM2_BOUND}"
    assert v_head_index * head_dim < IMM2_BOUND, f"v_base_address must be less than {IMM2_BOUND}"
    assert output_base_address < IMM2_BOUND, f"output_base_address must be less than {IMM2_BOUND}"
    assert v_msram_base < IMM2_BOUND, f"v_msram_base must be less than {IMM2_BOUND}"

    # Prefetch V from HBM (MLEN, head_dim) to MSRAM at v_msram_base
    # NOTE: We ALWAYS prefetch V because K prefetch in qkt_multiply uses MSRAM 0,
    # which overwrites any previously prefetched V. Even though all heads share
    # the same V data (same KV head), we must re-prefetch after each K prefetch.
    generated_code += f"S_ADDI_INT gp{v_base_register}, gp0, {v_head_index * head_dim} \n"
    # Use v_msram_base as MSRAM destination (can be 0 since K was already used)
    generated_code += f"S_ADDI_INT gp{out_base_register}, gp0, {v_msram_base} \n"
    # Use stride_en=0 for contiguous prefetch to avoid 64-byte alignment issues
    generated_code += f"H_PREFETCH_M gp{out_base_register}, gp{v_base_register}, a{v_base_hbm_offset_reg}, 0, 1 \n"

    # P address for this head's softmax scores
    p_start_address = p_base_address + q_head_index * mlen * mlen
    generated_code += f"S_ADDI_INT gp{p_base_register}, gp0, {p_start_address} \n"

    # V is prefetched to MSRAM at v_msram_base
    generated_code += f"S_ADDI_INT gp{v_base_register}, gp0, {v_msram_base} \n"

    # Output starts at output_base + head_offset (for this head's column position)
    generated_code += f"S_ADDI_INT gp{out_base_register}, gp0, {output_base_address + head_offset * head_dim} \n"

    if stage == "prefill":
        # Loop structure:
        # - outer loop over column blocks of V (head_dim // blen iterations)
        # - inner loop over row blocks (mlen // blen iterations)
        outer_loop_count = head_dim // blen  # Number of column blocks
        inner_loop_count = mlen // blen  # Number of row blocks
        generated_code += f"S_ADDI_INT gp{out_col_register}, gp0, {output_base_address + head_offset * head_dim} \n"
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {outer_loop_count} \n"
        generated_code += f"C_LOOP_START gp{inner_loop_register}, {inner_loop_count} \n"

        # M_MM: V_col_block @ P_row_block -> output block
        generated_code += f"M_MM 0, gp{v_base_register}, gp{p_base_register} \n"
        # Write to packed output position
        generated_code += f"M_MM_WO gp{out_base_register}, gp0, 0 \n"

        # Advance P to next row block (4 rows of mlen elements each)
        generated_code += f"S_ADDI_INT gp{p_base_register}, gp{p_base_register}, {blen * mlen} \n"
        # Advance output to next row block (stride is vlen * blen for packed format)
        generated_code += f"S_ADDI_INT gp{out_base_register}, gp{out_base_register}, {vlen * blen} \n"

        generated_code += f"C_LOOP_END gp{inner_loop_register} \n"

        # After inner loop: reset P, advance to next column block
        generated_code += f"S_ADDI_INT gp{p_base_register}, gp0, {p_start_address} \n"
        # Move to next column position (blen elements to the right)
        generated_code += f"S_ADDI_INT gp{out_col_register}, gp{out_col_register}, {blen} \n"
        generated_code += f"S_ADDI_INT gp{out_base_register}, gp{out_col_register}, 0 \n"
        # Move V to next column block
        generated_code += f"S_ADDI_INT gp{v_base_register}, gp{v_base_register}, {blen} \n"

        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"

    else:
        # Loop structure:
        loop_count = head_dim // blen  # Number of column blocks
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {loop_count} \n"
        # M_MM: V_col_block @ P_row_block -> output block
        generated_code += f"M_MV 0, gp{v_base_register}, gp{p_base_register} \n"
        # Write to packed output position
        generated_code += f"M_MV_WO gp{out_base_register}, gp0, 0 \n"
        generated_code += f"S_ADDI_INT gp{out_base_register}, gp{out_base_register}, {blen} \n"
        generated_code += f"S_ADDI_INT gp{v_base_register}, gp{v_base_register}, {blen} \n"
        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"
    return generated_code
