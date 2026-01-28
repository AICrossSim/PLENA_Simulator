from ipaddress import v4_int_to_packed
from typing import List
import math
from .reset_reg_asm import reset_reg_asm, reset_fpreg_asm, reset_vmask_asm

IMM2_BOUND = 2**18 - 1

def qkt_multiply(
    d: int,
    mlen: int,
    alive_registers: List[int],
    q_base_address: int,
    k_base_hbm_offset_reg: int,
    q_head_index: int,
    k_head_index: int,
    s_base_address: int = 0,
) -> str:
    """
    Args:
        mlen: the number of rows in the first matrix.
        blen: the number of columns in the second matrix.
        d: the head dimension
        q_len: the query length
        alive_registers: the list of alive registers.
        q_base_address: the base address of the query.
        k_base_address: the base address of the key.
    Description:
        This part of asm code gen template is used to compute QKT result.
        Assuming Q is in dim of (B, S, Hq, D), K is in dim of (B, S, Hkv, D)
        The num of Hq // Hkv of Q heads share the same K head.
        This template will perform, single batch, MLEN tiled, per KV head, QKT multiplication.
        (MLEN, Hq // Hkv, D) @ broadcast(D, 1, MLEN) = (Hq // Hkv, MLEN, MLEN)
    """
    q_base_register = alive_registers[0]
    k_base_register = alive_registers[1]
    s_base_register = q_base_register
    generated_code = "; QKT Per KV Head Multiplication \n"

    # Prefetch K from HBM
    generated_code += f"S_ADDI_INT gp{q_base_register}, gp0, {q_base_address + q_head_index * d} \n"
    generated_code += f"S_ADDI_INT gp{k_base_register}, gp0, {k_head_index * d} \n"
    generated_code += f"H_PREFETCH_M gp{k_base_register}, gp{k_base_register}, a{k_base_hbm_offset_reg}, 1, 1 \n"

    # QKT multiply
    generated_code += f"M_BTMM 0, gp{q_base_register}, gp{k_base_register} \n"
    generated_code += f"S_ADDI_INT gp{s_base_register}, gp0, {s_base_address + q_head_index * mlen * mlen} \n"
    generated_code += f"M_BMM_WO gp{s_base_register}, 0 \n"

    return generated_code


def _online_softmax_code(
    mlen: int,
    alive_registers_int: List[int],
    alive_registers_fp: List[int],
    s_address: int,
    m_start_address: int
) -> str:
    """
    Args:
    s_address: the starting address of the QKT result
    alive_registers_int: the list of alive registers for fix point operations
    alive_registers_fp: the list of alive registers for floating point operations
    mlen: also Br: the number of row of the QKT result
    address_of_mlen: the address that contains the mlen (number of row of the QKT result) value 
    Description:
        This part of asm is for the inner loop of the flash attention, mapping to line 9 to line 10 process,
        which requires per row level computation, hence with the loop mlen times.
    """
    # get two registers from alive_registers, 1 as m_last address, 1 as m_curr address
    m_last_register = alive_registers_fp[0]
    l_old_register = alive_registers_fp[1]
    tmp_fp_register = alive_registers_fp[2]
    sum_p_register = alive_registers_fp[3]

    # get a general address register
    s_address_register      = alive_registers_int[0] # general address register
    m_last_address_register = alive_registers_int[1]
    m_res_address_register  = alive_registers_int[2] # m_res address register
    l_old_address_register  = alive_registers_int[3] # l_old address register
    general_address_register = alive_registers_int[4] # general address register
    

    generated_code = "; Online Softmax Code \n"

    # Presettings
    # Load the starting address of S, which is the QKT result of the current head, in shape of (MLEN, MLEN)
    generated_code += f"S_ADDI_INT gp{s_address_register}, gp0, {s_address} \n"
    generated_code += f"S_ADDI_INT gp{m_last_address_register}, gp0, {m_start_address} \n"
    generated_code += f"S_ADDI_INT gp{m_res_address_register}, gp{m_last_address_register}, {mlen} \n"
    generated_code += f"S_ADDI_INT gp{l_old_address_register}, gp{m_res_address_register}, {mlen} \n"

    for i in range(mlen):
        # load m_last
        assert m_start_address < IMM2_BOUND, f"m_start_address must be less than {IMM2_BOUND}"

        generated_code += f"S_LD_FP f{m_last_register}, gp{m_last_address_register}, {i} \n"
        # copy m_last to a tmp fp register
        generated_code += f"S_ADD_FP f{tmp_fp_register}, f{m_last_register}, f0 \n"
        
        # m_curr = max(P[x4], m_last) and store at m_curr
        m_curr_register = m_last_register
        generated_code += f"V_RED_MAX f{m_curr_register}, gp{s_address_register}, {0} \n"

        # m_res = m_last - m_curr
        m_res_register = tmp_fp_register
        generated_code += f"S_SUB_FP f{m_res_register}, f{tmp_fp_register}, f{m_curr_register} \n"

        # exp(m_res)
        generated_code += f"S_EXP_FP f{m_res_register}, f{m_res_register}, 0 \n"

        # store m_res
        generated_code += f"S_ST_FP f{tmp_fp_register}, gp{m_res_address_register}, {i} \n"

        # store m_curr
        generated_code += f"S_ST_FP f{m_curr_register}, gp{m_last_address_register}, {i} \n"
        
        # S' = S - m_curr
        generated_code += f"V_SUB_VF gp{s_address_register}, gp{s_address_register}, f{m_curr_register}, 0, 0 \n"

        # P = exp(S')
        generated_code += f"V_EXP_V gp{s_address_register}, gp{s_address_register}, 0, 0 \n"

        # load l_old 
        generated_code += f"S_LD_FP f{l_old_register}, gp{l_old_address_register}, {i} \n"

        # P = sum(P)
        generated_code += f"S_ADD_FP  f{sum_p_register}, f0, f0 \n"

        generated_code += f"V_RED_SUM f{sum_p_register}, gp{s_address_register}, 0, 0 \n"

        # l_s = l_old * exp(m_res)
        generated_code += f"S_MUL_FP f{l_old_register}, f{l_old_register}, f{tmp_fp_register} \n"
        l_s_register = l_old_register

        # l_s = l_old * exp(m_res) + sum(P)
        generated_code += f"S_ADD_FP f{l_s_register}, f{sum_p_register}, f{l_old_register} \n"

        # store l_s
        generated_code += f"S_ST_FP f{l_s_register}, gp{l_old_address_register}, {i} \n"

        # next row of S
        generated_code += f"S_ADDI_INT gp{s_address_register}, gp{s_address_register}, {mlen} \n"

    return generated_code

def _computing_pv_code(
    head_dim: int,
    q_head_num: int,
    blen: int,
    mlen: int,
    alive_registers: List[int],
    p_base_address: int,
    v_base_hbm_offset_reg: int,
    q_head_index: int,
    v_head_index: int,
    pv_base_address: int,
    same_v_head: bool,
) -> str:
    """
    Args:
    
    Description:
        This part of asm is for the computing of the PV operation, mapping to line 10 process,
        which requires per head dimension level computation, hence with the loop head_dim // mlen times.
        (mlen, mlen) @ (mlen, head_dim) = (mlen, head_dim)
    """
    generated_code = "; PV Per KV Head Multiplication \n"
    p_base_register = alive_registers[0]
    v_base_register = alive_registers[1]
    pv_base_register = alive_registers[2]
    mm_wo_stride_register = alive_registers[3]
    assert p_base_address + q_head_index * head_dim < IMM2_BOUND, f"p_base_address must be less than {IMM2_BOUND}"
    assert v_head_index * head_dim < IMM2_BOUND, f"v_base_address must be less than {IMM2_BOUND}"
    assert pv_base_address < IMM2_BOUND, f"pv_base_address must be less than {IMM2_BOUND}"
    # q_row_ratio = (q_head_num * head_dim) // mlen


    assert q_head_index <= (mlen // head_dim) and v_head_index <= (mlen // head_dim), "q_head_index and v_head_index must be less than mlen // head_dim"
    # Prefetch K from HBM (MLEN, MLEN)
    if not same_v_head:    
        generated_code += f"S_ADDI_INT gp{v_base_register}, gp0, {v_head_index * head_dim} \n"
        generated_code += f"H_PREFETCH_M gp0, gp{v_base_register}, a{v_base_hbm_offset_reg}, 1, 1 \n"

    # Address Settings
    generated_code += f"S_ADDI_INT gp{p_base_register}, gp0, {p_base_address + q_head_index * mlen * mlen} \n"

    generated_code += f"S_ADDI_INT gp{v_base_register}, gp0, {v_head_index * head_dim} \n"
    generated_code += f"S_ADDI_INT gp{pv_base_register}, gp0, {pv_base_address + q_head_index * mlen * mlen} \n"
    # generated_code += f"S_ADDI_INT gp{mm_wo_stride_register}, gp0, {((q_head_num * head_dim) // mlen)} \n"

    # PV mult
    for i in range (head_dim // blen):
        for j in range (mlen // blen):
            # Keep V stationary in the inner loop, shift across p 
            generated_code += f"M_MM 0, gp{v_base_register}, gp{p_base_register} \n"
            generated_code += f"M_MM_WO gp{pv_base_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{p_base_register}, gp{p_base_register}, {blen * mlen} \n"
            generated_code += f"S_ADDI_INT gp{pv_base_register}, gp{pv_base_register}, {mlen * blen} \n"
        # Update v_base_register and reset p_base_register and pv_base_register
        generated_code += f"S_ADDI_INT gp{p_base_register}, gp0, {p_base_address + q_head_index * mlen * mlen} \n"
        generated_code += f"S_ADDI_INT gp{pv_base_register}, gp0, {pv_base_address + q_head_index * head_dim + (i+1) * blen} \n"
        generated_code += f"S_ADDI_INT gp{v_base_register}, gp{v_base_register}, {blen} \n"

    return generated_code

def _computing_o_code(
    mlen: int,
    alive_registers_int: List[int],
    alive_registers_fp: List[int],
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

    m_res_fp_register = alive_registers_fp[0]
    generated_code = "; Computing O Code \n"
    assert head_dim < mlen, "head_dim must be less than mlen"
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

    # loop over different row of m_res
    for j in range(mlen):
        # load m_res
        generated_code += f"S_LD_FP f{m_res_fp_register}, gp{m_res_vector_address_register}, {j} \n"
        # boardcast m_res to multiply with a row of a block of O_old and write to o_old
        generated_code += f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{m_res_vector_address_register}, 1 \n"
        # add pv row to o_old
        generated_code += f"V_ADD_VV gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, gp{pv_vector_address_register}, 1 \n"
        # # update o_old base address
        generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, {q_head_num * head_dim} \n"
        # # update pv base address
        generated_code += f"S_ADDI_INT gp{pv_vector_address_register}, gp{pv_vector_address_register}, {mlen} \n"
    # now o_old should contain the result of the current o, diag(exp(m_res)) * O_old + PV
    return generated_code


def _computing_row_wise_scaling_code(
    mlen: int,
    alive_registers_int: List[int],
    alive_registers_fp: List[int],
    o_old_base_address: int,
    l_old_base_address: int,
) -> str:
    """ 
    line 12 in flash attention algorithm
    mlen: the number of row of the QKT result
    alive_registers_int: the list of alive registers for fix point operations
    alive_registers_fp: the list of alive registers for floating point operations
    o_old_base_address: the base address of the old O
    """
    o_old_vector_address_register = alive_registers_int[0]
    l_old_vector_address_register = alive_registers_int[1]
    l_old_fp_register = alive_registers_fp[0]

    generated_code = ""
    # load l_old base address
    generated_code += f"S_ADDI_INT gp{l_old_vector_address_register}, gp0, {l_old_base_address} \n"
    # load o_old base address
    generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp0, {o_old_base_address} \n"

    # loop over different row of Br
    for i in range(mlen):
        # load l_old
        generated_code += f"S_LD_FP f{l_old_fp_register}, gp{l_old_vector_address_register}, {i} \n"
        # compute the inverse of l_old
        generated_code += f"S_RECI_FP f{l_old_fp_register}, f{l_old_fp_register}, 0 \n"
        # multiply o_old with the inverse of l_old
        generated_code += f"V_MUL_VF gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, f{l_old_fp_register} \n"
        # update o_old base address
        generated_code += f"S_ADDI_INT gp{o_old_vector_address_register}, gp{o_old_vector_address_register}, {mlen} \n"

    return generated_code

def _reset_fpsram_code(
    reset_start_address: int,
    per_stride_dim: int,
    reset_stride: int,
    reset_amount: int,
    reset_val_address: int,
    alive_registers_fp: List[int],
    alive_registers_int: List[int],
) -> str:
    """
    Args:
    reset_start_address: the start address of the reset
    per_stride_dim: the dimension of the reset per stride
    reset_stride: the stride of the reset
    reset_amount: the amount of the reset
    reset_val_address: the address of the reset value
    """
    generated_code = f"; Reset FPSRAM Code from {reset_start_address} to {reset_start_address + reset_amount * reset_stride} with value {reset_val_address}\n"
    
    generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {reset_start_address} \n"
    generated_code += f"S_LD_FP f{alive_registers_fp[0]}, gp0, {reset_val_address} \n"
    for i in range(reset_amount):
        for j in range(per_stride_dim):
            generated_code += f"S_ST_FP f{alive_registers_fp[0]}, gp{alive_registers_int[0]}, {i * reset_stride + j} \n"
    return generated_code

def _reset_vssram_code(
    reset_start_address: int,
    vect_dim: int,
    per_stride_dim: int,
    reset_stride: int,
    reset_amount: int,
    alive_registers_int: List[int],
) -> str:
    """
    Args:
    reset_start_address: the start address of the reset
    per_stride_dim: the dimension of the reset per stride
    reset_stride: the stride of the reset
    reset_amount: the amount of the reset
    reset_val_address: the address of the reset value
    """
    generated_code = f"; Reset VSSRAM Code from {reset_start_address} to {reset_start_address + reset_amount * reset_stride} with value 0\n"
    generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {reset_start_address} \n"
    for i in range(reset_amount):
        for j in range(per_stride_dim):
            generated_code += f"V_MUL_VF gp{alive_registers_int[0]}, gp{alive_registers_int[0]}, f0, 0\n"
            generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp{alive_registers_int[0]}, {vect_dim} \n"
    return generated_code

def _reset_kv_prefetch(
    hkv: int,
    d: int,
    kv_len: int,
    batch: int,
    alive_registers_int: List[int],
) -> str:
    generated_code = f"; Reset KV Prefetch Code \n"
    assert hkv * d * kv_len * batch < IMM2_BOUND, f"hkv * d * kv_len * batch must be less than {IMM2_BOUND}"
    assert hkv * d * kv_len * batch < IMM2_BOUND, f"hkv * d * kv_len * batch must be less than {IMM2_BOUND}"
    generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {hkv *d * kv_len * batch} \n"
    generated_code += f"C_SET_SCALE_REG gp{alive_registers_int[0]} \n"
    generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {hkv * d * batch} \n"
    generated_code += f"C_SET_STRIDE_REG gp{alive_registers_int[0]} \n"
    return generated_code


def flash_attn_asm(
    mlen: int,
    blen: int,
    batch: int,
    hq: int,
    hkv: int,
    d: int,
    q_len: int,
    kv_len: int,
    alive_registers_int: List[int],
    alive_registers_fp: List[int],
    vector_sram_base_address: int,
    fp_sram_start_address: int,
    k_base_hbm_offset_reg: int,
    v_base_hbm_offset_reg: int,
) -> str:
    """
    Args:
    vector_sram_base_address: the base address of the vector SRAM
    fp_sram_start_address: the start address of the fp SRAM
    k_base_hbm_offset_reg: the offset register of the k base address in HBM
    v_base_hbm_offset_reg: the offset register of the v base address in HBM
    Description:
        This part of asm takes the multi-loops, looping over kv head, then two loops for the flash atten, with small loops over q head per kv head within the inner loop.
    """
    # Iteration Settings
    q_seq_iteration_number = (q_len + mlen - 1) // mlen
    k_seq_iteration_number = (kv_len + mlen - 1) // mlen
    q_index_2_kv_index_ratio = hq // hkv

    # Memory Layout:
    # -- FP SRAM --
    # Defalt 0 - zero
    # 1 - infinity
    # fp_sram_start_address - 1 - qk_scale
    # per head dimension * q_index_2_kv_index_ratio level {
    m_fp_sram_start_address = fp_sram_start_address
    # - m old (MLEN) - 0
    # - m res (MLEN) - 1
    # - l old (MLEN) - 2
    # }
    print("=" * 5, "VSRAM Memory Layout", "=" * 5)
    # -- Vector SRAM --
    # Q  (q_len, hq * hkv) Since the Q is computed directly from the projection, where per token dim all heads connecting together.
    q_base_address = vector_sram_base_address
    print(f"Q Base Address: {q_base_address}")
    # tmp S (MLEN, MLEN, hq // hkv) and also tmp P.
    s_base_address = q_base_address + hq * hkv * q_len 
    print(f"S Base Address: {s_base_address}")
    # PV (q_index_2_kv_index_ratio, mlen, mlen)
    pv_base_address = s_base_address + mlen * mlen * q_index_2_kv_index_ratio
    print(f"PV Base Address: {pv_base_address}")
    # O_Old (q_len, HEAD_DIM * Hq * batch)
    o_old_base_address = pv_base_address + mlen * mlen * q_index_2_kv_index_ratio
    print(f"O_Old Base Address: {o_old_base_address}")
    generated_code = "; Flash Attention Generation \n"
    generated_code += _reset_kv_prefetch(
        hkv=hkv,
        d=d,
        kv_len=kv_len,
        batch=batch,
        alive_registers_int=alive_registers_int[0:1],
    )

    # loop over kv heads
    for kv_head_index in range(hkv):
        # loop over per kv head kv_len // MLEN
        for i in range(k_seq_iteration_number):
            # Reset m old for every q_index_2_kv_index_ratio q heads with -inf
            generated_code += _reset_fpsram_code(
                reset_start_address =   m_fp_sram_start_address,
                per_stride_dim      =   mlen,
                reset_stride        =   3 * mlen,
                reset_amount        =   q_index_2_kv_index_ratio,
                reset_val_address   =   2,
                alive_registers_fp  =   alive_registers_fp[0:1], 
                alive_registers_int =   alive_registers_int[0:1],
            )

            # Reset l with zeros
            generated_code += _reset_fpsram_code(
                reset_start_address =   m_fp_sram_start_address + 2 * mlen,
                per_stride_dim      =   mlen,
                reset_stride        =   3 *mlen,
                reset_amount        =   q_index_2_kv_index_ratio,
                reset_val_address   =   0,
                alive_registers_fp  =   alive_registers_fp[0:1],
                alive_registers_int =   alive_registers_int[0:1],
            )

            # Reset O_old with zeros
            generated_code += _reset_vssram_code(
                reset_start_address =   o_old_base_address,
                vect_dim            =   mlen,
                per_stride_dim      =   d,
                reset_stride        =   q_index_2_kv_index_ratio * mlen,
                reset_amount        =   q_index_2_kv_index_ratio,
                alive_registers_int =   alive_registers_int[0:1],
            )

            # # loop over per q_index_2_kv_index_ratio q heads (q_len // MLEN), compute q_index_2_kv_index_ratio heads in parallel.
            for j in range(q_seq_iteration_number):
                # Compute S = QKT result
                generated_code += qkt_multiply(
                    d=d,
                    mlen=mlen,
                    alive_registers         =   alive_registers_int[0:2],
                    q_base_address          =   q_base_address + q_index_2_kv_index_ratio * kv_head_index * mlen,
                    k_base_hbm_offset_reg   =   k_base_hbm_offset_reg,
                    q_head_index            =   j * q_index_2_kv_index_ratio,
                    k_head_index            =   kv_head_index,
                    s_base_address          =   s_base_address,
                )

                generated_code += reset_reg_asm(alive_registers_int[0:2])
                stored_m_fp_res_address = m_fp_sram_start_address + mlen
                
                for inner_q_head_index in range(hq // hkv):
                    # Per Q head level online softmax
                    generated_code += _online_softmax_code(
                        mlen=mlen,
                        alive_registers_int=alive_registers_int[0:5],
                        alive_registers_fp=alive_registers_fp[0:4],
                        s_address=s_base_address + inner_q_head_index * mlen * mlen,
                        m_start_address=m_fp_sram_start_address
                    )

                    m_fp_sram_start_address += mlen * 3
                    generated_code += reset_fpreg_asm(alive_registers_fp[0:5])
                    generated_code += reset_reg_asm(alive_registers_int[0:4])
                    generated_code += _computing_pv_code(
                        head_dim=d,
                        q_head_num=hq,
                        blen=blen,
                        mlen=mlen,
                        alive_registers=alive_registers_int[0:4],
                        p_base_address=s_base_address,
                        v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                        q_head_index=inner_q_head_index,
                        v_head_index=kv_head_index,
                        pv_base_address=pv_base_address + inner_q_head_index * mlen * mlen,
                        same_v_head=(inner_q_head_index != 0),
                    )

                    generated_code += reset_reg_asm(alive_registers_int[0:4])
                    generated_code += reset_vmask_asm(alive_registers_int[0], 1 << inner_q_head_index)

                    generated_code += _computing_o_code(
                        mlen=mlen,
                        alive_registers_int=alive_registers_int[0:3],
                        alive_registers_fp=alive_registers_fp[0:1],
                        m_res_base_address=stored_m_fp_res_address,
                        pv_base_address=pv_base_address + inner_q_head_index * mlen * mlen,
                        o_old_base_address=o_old_base_address + kv_head_index * mlen,
                        head_dim=d,
                        q_head_num=hq,
                    )
                    stored_m_fp_res_address += 3 * mlen
                    break
                break
            break
        break

        # update q base address
        q_base_address += q_index_2_kv_index_ratio * q_len * d

        # generated_code += _computing_row_wise_scaling_code(
        #     mlen=mlen,
        #     alive_registers_int=alive_registers_int,
        #     alive_registers_fp=alive_registers_fp,
        #     o_old_base_address=o_old_address,
        #     l_old_base_address=l_old_base_address,
        # )
    
    return generated_code