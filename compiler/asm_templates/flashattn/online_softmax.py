"""Online softmax assembly code generation for Flash Attention."""

IMM2_BOUND = 2**18 - 1

"""
Memory Layout in FP SRAM:
- m_last (MLEN)
- m_res (MLEN)
- l_old (MLEN)
"""


def online_softmax_code(
    mlen: int,
    stage: str,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
    s_address: int,
    m_start_address: int,
    qk_scale_address: int = 1,
) -> str:
    """
    Args:
    s_address: the starting address of the QKT result
    alive_registers_int: the list of alive registers for fix point operations
    alive_registers_fp: the list of alive registers for floating point operations
    mlen: also Br: the number of row of the QKT result
    qk_scale_address: the FP SRAM address containing qk_scale (default 1)
    Description:
        This part of asm is for the inner loop of the flash attention, mapping to line 9 to line 10 process,
        which requires per row level computation, hence with the loop mlen times.
    """
    # get two registers from alive_registers, 1 as m_last address, 1 as m_curr address
    m_last_register = alive_registers_fp[0]
    l_old_register = alive_registers_fp[1]
    tmp_fp_register = alive_registers_fp[2]
    sum_p_register = alive_registers_fp[3]
    qk_scale_register = alive_registers_fp[4]

    # get a general address register
    s_address_register = alive_registers_int[0]  # general address register
    m_last_address_register = alive_registers_int[1]
    m_res_address_register = alive_registers_int[2]  # m_res address register
    l_old_address_register = alive_registers_int[3]  # l_old address register
    loop_register = alive_registers_int[4]  # loop counter register

    generated_code = "; Online Softmax Code \n"

    if stage == "prefill":
        # Presettings
        # Load the starting address of S, which is the QKT result of the current head, in shape of (MLEN, MLEN)
        assert m_start_address < IMM2_BOUND, f"m_start_address must be less than {IMM2_BOUND}"
        generated_code += f"S_ADDI_INT gp{s_address_register}, gp0, {s_address} \n"
        generated_code += f"S_ADDI_INT gp{m_last_address_register}, gp0, {m_start_address} \n"
        generated_code += f"S_ADDI_INT gp{m_res_address_register}, gp{m_last_address_register}, {mlen} \n"
        generated_code += f"S_ADDI_INT gp{l_old_address_register}, gp{m_res_address_register}, {mlen} \n"

        # Load qk_scale from FP SRAM (preloaded at address 1)
        generated_code += f"S_LD_FP f{qk_scale_register}, gp0, {qk_scale_address} \n"

        # Hardware loop over mlen rows
        generated_code += f"C_LOOP_START gp{loop_register}, {mlen} \n"

        # Scale S row by qk_scale: S = S * qk_scale
        generated_code += f"V_MUL_VF gp{s_address_register}, gp{s_address_register}, f{qk_scale_register}, 0 \n"

        # load m_last (using indirect addressing with offset 0)
        # Note: m_last is already in scaled space from previous tiles (or -inf for first tile)
        generated_code += f"S_LD_FP f{m_last_register}, gp{m_last_address_register}, 0 \n"
        # # copy m_last to a tmp fp register
        generated_code += f"S_ADD_FP f{tmp_fp_register}, f{m_last_register}, f0 \n"

        # m_curr = max(S_scaled[row], m_last) and store at m_curr
        m_curr_register = m_last_register
        generated_code += f"V_RED_MAX f{m_curr_register}, gp{s_address_register}, 0 \n"

        # m_res = m_last - m_curr
        m_res_register = tmp_fp_register
        generated_code += f"S_SUB_FP f{m_res_register}, f{tmp_fp_register}, f{m_curr_register} \n"

        # # exp(m_res)
        generated_code += f"S_EXP_FP f{m_res_register}, f{m_res_register} \n"

        # # store m_res (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{tmp_fp_register}, gp{m_res_address_register}, 0 \n"

        # store m_curr (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{m_curr_register}, gp{m_last_address_register}, 0 \n"

        # # S' = S - m_curr
        generated_code += f"V_SUB_VF gp{s_address_register}, gp{s_address_register}, f{m_curr_register}, 0, 0 \n"

        # P = exp(S')
        generated_code += f"V_EXP_V gp{s_address_register}, gp{s_address_register}, 0 \n"

        # load l_old (using indirect addressing with offset 0)
        generated_code += f"S_LD_FP f{l_old_register}, gp{l_old_address_register}, 0 \n"

        # P = sum(P)
        generated_code += f"S_ADD_FP  f{sum_p_register}, f0, f0 \n"
        generated_code += f"V_RED_SUM f{sum_p_register}, gp{s_address_register} \n"

        # l_s = l_old * exp(m_res)
        generated_code += f"S_MUL_FP f{l_old_register}, f{l_old_register}, f{tmp_fp_register} \n"
        l_s_register = l_old_register

        # l_s = l_old * exp(m_res) + sum(P)
        generated_code += f"S_ADD_FP f{l_s_register}, f{sum_p_register}, f{l_old_register} \n"

        # store l_s (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{l_s_register}, gp{l_old_address_register}, 0 \n"

        # # next row of S
        generated_code += f"S_ADDI_INT gp{s_address_register}, gp{s_address_register}, {mlen} \n"
        # # increment m_last, m_res, l_old addresses
        generated_code += f"S_ADDI_INT gp{m_last_address_register}, gp{m_last_address_register}, 1 \n"
        generated_code += f"S_ADDI_INT gp{m_res_address_register}, gp{m_res_address_register}, 1 \n"
        generated_code += f"S_ADDI_INT gp{l_old_address_register}, gp{l_old_address_register}, 1 \n"

        generated_code += f"C_LOOP_END gp{loop_register} \n"
    else:
        # Presettings
        # Load the starting address of S, which is the QKT result of the current head, in shape of (MLEN, MLEN)
        assert m_start_address < IMM2_BOUND, f"m_start_address must be less than {IMM2_BOUND}"
        generated_code += f"S_ADDI_INT gp{s_address_register}, gp0, {s_address} \n"
        generated_code += f"S_ADDI_INT gp{m_last_address_register}, gp0, {m_start_address} \n"
        generated_code += f"S_ADDI_INT gp{m_res_address_register}, gp{m_last_address_register}, {1} \n"
        generated_code += f"S_ADDI_INT gp{l_old_address_register}, gp{m_res_address_register}, {1} \n"

        # Load qk_scale from FP SRAM (preloaded at address 1)
        generated_code += f"S_LD_FP f{qk_scale_register}, gp0, {qk_scale_address} \n"

        # Scale S row by qk_scale: S = S * qk_scale
        generated_code += f"V_MUL_VF gp{s_address_register}, gp{s_address_register}, f{qk_scale_register}, 0 \n"

        # load m_last (using indirect addressing with offset 0)
        # Note: m_last is already in scaled space from previous tiles (or -inf for first tile)
        generated_code += f"S_LD_FP f{m_last_register}, gp{m_last_address_register}, 0 \n"
        # # copy m_last to a tmp fp register
        generated_code += f"S_ADD_FP f{tmp_fp_register}, f{m_last_register}, f0 \n"

        # m_curr = max(S_scaled[row], m_last) and store at m_curr
        m_curr_register = m_last_register
        generated_code += f"V_RED_MAX f{m_curr_register}, gp{s_address_register}, 0 \n"

        # m_res = m_last - m_curr
        m_res_register = tmp_fp_register
        generated_code += f"S_SUB_FP f{m_res_register}, f{tmp_fp_register}, f{m_curr_register} \n"

        # # exp(m_res)
        generated_code += f"S_EXP_FP f{m_res_register}, f{m_res_register} \n"

        # # store m_res (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{tmp_fp_register}, gp{m_res_address_register}, 0 \n"

        # store m_curr (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{m_curr_register}, gp{m_last_address_register}, 0 \n"

        # # S' = S - m_curr
        generated_code += f"V_SUB_VF gp{s_address_register}, gp{s_address_register}, f{m_curr_register}, 0, 0 \n"

        # P = exp(S')
        generated_code += f"V_EXP_V gp{s_address_register}, gp{s_address_register}, 0 \n"

        # load l_old (using indirect addressing with offset 0)
        generated_code += f"S_LD_FP f{l_old_register}, gp{l_old_address_register}, 0 \n"

        # P = sum(P)
        generated_code += f"S_ADD_FP  f{sum_p_register}, f0, f0 \n"
        generated_code += f"V_RED_SUM f{sum_p_register}, gp{s_address_register} \n"

        # l_s = l_old * exp(m_res)
        generated_code += f"S_MUL_FP f{l_old_register}, f{l_old_register}, f{tmp_fp_register} \n"
        l_s_register = l_old_register

        # l_s = l_old * exp(m_res) + sum(P)
        generated_code += f"S_ADD_FP f{l_s_register}, f{sum_p_register}, f{l_old_register} \n"

        # store l_s (using indirect addressing with offset 0)
        generated_code += f"S_ST_FP f{l_s_register}, gp{l_old_address_register}, 0 \n"
    return generated_code
