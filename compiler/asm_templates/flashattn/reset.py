"""Reset/initialization assembly code generation for Flash Attention."""

IMM2_BOUND = 2**18 - 1


def reset_fpsram_code(
    reset_start_address: int,
    per_stride_dim: int,
    stride_dist: int,
    reset_amount: int,
    reset_val_address: int,
    alive_registers_fp: list[int],
    alive_registers_int: list[int],
) -> str:
    """
    Args:
    reset_start_address: the start address of the reset
    per_stride_dim: the dimension of the reset per stride
    stride_dist: the stride distance between two consecutive resets
    reset_amount: the amount of the consecutive resets
    reset_val_address: the address of the reset value
    """
    generated_code = f"; Reset FPSRAM Code from {reset_start_address} to {reset_start_address + reset_amount * per_stride_dim} with value {reset_val_address}\n"

    addr_register = alive_registers_int[0]
    outer_loop_register = alive_registers_int[1]
    inner_loop_register = alive_registers_int[2]
    offset_register = alive_registers_int[3]
    fp_val_register = alive_registers_fp[0]

    generated_code += f"S_ADDI_INT gp{addr_register}, gp0, {reset_start_address} \n"
    generated_code += f"S_ADDI_INT gp{offset_register}, gp0, {stride_dist} \n"
    generated_code += f"S_LD_FP f{fp_val_register}, gp0, {reset_val_address} \n"

    # Total iterations = reset_amount * per_stride_dim
    if reset_amount > 1:
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {reset_amount} \n"
        if per_stride_dim > 1:
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {per_stride_dim} \n"
            generated_code += f"S_ST_FP f{fp_val_register}, gp{addr_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{addr_register}, gp{addr_register}, 1 \n"
            generated_code += f"C_LOOP_END gp{inner_loop_register} \n"
        else:
            generated_code += f"S_ST_FP f{fp_val_register}, gp{addr_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{addr_register}, gp{addr_register}, 1 \n"

        generated_code += f"S_ADDI_INT gp{offset_register}, gp{offset_register}, {stride_dist} \n"
        generated_code += f"S_ADD_INT gp{addr_register}, gp0, gp{offset_register} \n"
        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"
    else:
        if per_stride_dim > 1:
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {per_stride_dim} \n"
            generated_code += f"S_ST_FP f{fp_val_register}, gp{addr_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{addr_register}, gp{addr_register}, 1 \n"
            generated_code += f"C_LOOP_END gp{inner_loop_register} \n"
        else:
            generated_code += f"S_ST_FP f{fp_val_register}, gp{addr_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{addr_register}, gp{addr_register}, 1 \n"

    return generated_code


def reset_vssram_code(
    reset_start_address: int,
    vect_dim: int,
    per_stride_dim: int,
    reset_stride: int,
    reset_amount: int,
    alive_registers_int: list[int],
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

    addr_register = alive_registers_int[0]
    outer_loop_register = alive_registers_int[1]
    inner_loop_register = alive_registers_int[2]

    generated_code += f"S_ADDI_INT gp{addr_register}, gp0, {reset_start_address} \n"

    total_iterations = reset_amount * per_stride_dim
    if total_iterations > 0:
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {reset_amount} \n"
        generated_code += f"C_LOOP_START gp{inner_loop_register}, {per_stride_dim} \n"
        generated_code += f"V_MUL_VF gp{addr_register}, gp{addr_register}, f0, 0\n"
        generated_code += f"S_ADDI_INT gp{addr_register}, gp{addr_register}, {vect_dim} \n"
        generated_code += f"C_LOOP_END gp{inner_loop_register} \n"
        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"

    return generated_code


def reset_kv_prefetch(
    hkv: int,
    d: int,
    kv_len: int,
    batch: int,
    mlen: int,
    alive_registers_int: list[int],
) -> str:
    generated_code = "; Reset KV Prefetch Code \n"
    assert hkv * d * kv_len * batch < IMM2_BOUND, f"hkv * d * kv_len * batch must be less than {IMM2_BOUND}"
    assert hkv * d * kv_len * batch < IMM2_BOUND, f"hkv * d * kv_len * batch must be less than {IMM2_BOUND}"

    if hkv * d < mlen:
        generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {mlen * kv_len * batch} \n"
        generated_code += f"C_SET_SCALE_REG gp{alive_registers_int[0]} \n"
        generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {mlen} \n"
        generated_code += f"C_SET_STRIDE_REG gp{alive_registers_int[0]} \n"
    else:
        generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {hkv * d * kv_len * batch} \n"
        generated_code += f"C_SET_SCALE_REG gp{alive_registers_int[0]} \n"
        generated_code += f"S_ADDI_INT gp{alive_registers_int[0]}, gp0, {hkv * d * batch} \n"
        generated_code += f"C_SET_STRIDE_REG gp{alive_registers_int[0]} \n"
    return generated_code
