import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import math

def batched_matmul_asm(
    mlen: int,
    blen: int,
    b: int,
    m: int,
    k: int,
    n: int,
    alive_registers: List[int],
    w_base_hbm_offset_reg: int,
    w_prefetch_amount: int,
    # w_precision: int, # in bytes
    a_base_hbm_offset_reg: int,
    a_prefetch_amount: int,
    # a_precision: int, # in bytes
    # on_chip_mem_space: int,
    result_base_address: int,
) -> str:
    """
    Generates assembly code for a general batched matrix multiplication operation.
    activation(Batch, M, K) @ weight(Batched, K, N) -> result(Batch, M, N)

    Args:
        (blen (int), mlen (int)) formating the shape of the matrix. 
        b, m, k, n are the dimensions of the matrix.
        alive_registers are the registers that are alive.
        w_base_hbm_offset_reg is the base hbm offset of the weight matrix.
        a_base_hbm_offset_reg is the base hbm offset of the activation matrix.
        result_base_address is the base address of the result matrix.
    Assumption: 
        Assuming the two tenssors are stored in the continous memory in mx data format, with the scales stored at the end of each tensor.
    Returns:
        str: Generated assembly code for projection, including dot product and RoPE(cond)
    """
    generated_code = "; Batched Matrix Multiplication Generation \n"
    assert k % mlen == 0, "k must be divisible by mlen"
    assert m % blen == 0, "m must be divisible by blen"
    assert n % blen == 0, "n must be divisible by blen"
    print(f"b = {b}, m = {m}, k = {k}, n = {n}")
    print(f"mlen = {mlen}, blen = {blen}")

    a_actual_register   = alive_registers[0]
    w_actual_register = alive_registers[1]
    result_actual_register = alive_registers[2]


    generated_code += f"S_ADDI_INT gp{result_actual_register}, gp0, {result_base_address} \n"
    for batch in range(1, b + 1):
        # preload the activation matrix
        for i in range(math.ceil(n // blen)):
            assert w_prefetch_amount >= k, "w_prefetch_amount must be greater than or equal to k"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {m * k * batch} \n"
            generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {n} \n"
            generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
            generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{w_actual_register}, a{w_base_hbm_offset_reg}, 1, 0 \n"
            
            for j in range(math.ceil(m // blen)):
                if j == 0:
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, { k * n * batch} \n"
                    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {k} \n"
                    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
                if j % math.ceil((a_prefetch_amount * mlen) // k) == 0:
                    generated_code += f"H_PREFETCH_V gp{a_actual_register}, gp{a_actual_register}, a{a_base_hbm_offset_reg}, 1, 0 \n"
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0 \n"
                for g in range(math.ceil(k // mlen)):
                    generated_code += f"M_MM 0, gp{w_actual_register}, gp{a_actual_register} \n"
                    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen * mlen} \n"
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {blen * mlen} \n"
                generated_code += f"M_MM_WO {result_actual_register}, gp0, 0 \n"
                generated_code += f"S_ADDI_INT gp{result_actual_register}, gp0, {j * n + i * blen} \n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"

    return generated_code
