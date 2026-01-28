import os
from typing import Dict, List, Any, Optional
from pathlib import Path

def rms_norm_asm(
    _eps_offset: int,
    reci_hid_offset: int,
    alive_registers: List[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int
) -> str:
    """
    Generate assembly code for RMS normalization.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]

    generated_code = "; RMS Norm generation \n"
    generated_code += f"S_ADDI_INT gp{scratchpad_addr}, gp0, {scratchpad_base_address} \n"

    # Load eps into f1
    generated_code += f"S_LD_FP f1, gp0, {_eps_offset} \n"
    # Reset f2 as accumulator for reduction
    generated_code += "S_ADD_FP f2, f0, f0 \n"
    # Load the 1/hidden_dim into f3
    generated_code += f"S_LD_FP f3, gp0, {reci_hid_offset} \n"

    for batch in range(batch_size):
        # Set act_addr to start of current batch
        generated_code += f"S_ADDI_INT gp{act_addr}, gp0, {activation_base_address + vlen * batch} \n"
        # Set stats_addr to same position for iteration
        generated_code += f"S_ADDI_INT gp{stats_addr}, gp0, {activation_base_address + vlen * batch} \n"

        # First loop: compute sum of squares using stats_addr
        for i in range(hidden_dim // vlen):
            # Compute square of the activation vector and summation
            generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
            generated_code += f"V_RED_SUM f2, gp{scratchpad_addr} \n"

            # Move stats pointer to next vector
            generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"

        # Taking the avg
        generated_code += f"S_MUL_FP f2, f2, f3 \n"

        # Plus epsilon
        generated_code += f"S_ADD_FP f2, f2, f1 \n"

        # Compute square root
        generated_code += "S_SQRT_FP f2, f2 \n"

        # Compute reciprocal
        generated_code += "S_RECI_FP f2, f2 \n"

        # Second loop: normalize using act_addr
        for i in range(hidden_dim // vlen):
            # Normalize the activation vector
            generated_code += f"V_MUL_VF gp{act_addr}, gp{act_addr}, f2, 0 \n"

            # Move to next vector
            generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"

        # Reset accumulator for next batch
        generated_code += "S_ADD_FP f2, f0, f0 \n"

    return generated_code

def layer_norm_asm(
    _eps_offset: int,
    reci_hid_offset: int,
    alive_registers: List[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int
) -> str:
    """
    Generate assembly code for layer normalization.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]

    generated_code = "; Layer Norm generation \n"
    generated_code += f"S_ADDI_INT gp{scratchpad_addr}, gp0, {scratchpad_base_address} \n"

    # Load constants
    generated_code += f"S_LD_FP f1, gp0, {_eps_offset} \n"          # epsilon
    generated_code += "S_ADD_FP f2, f0, f0 \n"                      # sum(x) accumulator
    generated_code += "S_ADD_FP f3, f0, f0 \n"                      # sum(x^2) accumulator
    generated_code += f"S_LD_FP f4, gp0, {reci_hid_offset} \n"      # 1/hidden_dim

    for batch in range(batch_size):
        # Set act_addr to start of current batch
        generated_code += f"S_ADDI_INT gp{act_addr}, gp0, {activation_base_address + vlen * batch} \n"
        # Set stats_addr to same position for iteration
        generated_code += f"S_ADDI_INT gp{stats_addr}, gp0, {activation_base_address + vlen * batch} \n"

        # First loop: compute sum(x) and sum(x^2) using stats_addr
        for i in range(hidden_dim // vlen):
            # sum(x)
            generated_code += f"V_RED_SUM f2, gp{stats_addr} \n"

            # sum(x^2)
            generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
            generated_code += f"V_RED_SUM f3, gp{scratchpad_addr} \n"

            # Move stats pointer to next vector
            generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"

        # f2 = sum(x) * (1/hidden_dim) = mean(x)
        generated_code += f"S_MUL_FP f2, f2, f4 \n"

        # f3 = sum(x^2) * (1/hidden_dim) = mean(x^2)
        generated_code += f"S_MUL_FP f3, f3, f4 \n"

        # f5 = mean(x)^2
        generated_code += f"S_MUL_FP f5, f2, f2 \n"

        # f5 = mean(x^2) - mean(x)^2 = variance
        generated_code += f"S_SUB_FP f5, f3, f5 \n"

        # f5 = variance + epsilon
        generated_code += f"S_ADD_FP f5, f5, f1 \n"

        # f5 = sqrt(variance + epsilon) = std
        generated_code += f"S_SQRT_FP f5, f5 \n"

        # f5 = 1/std
        generated_code += f"S_RECI_FP f5, f5 \n"

        # Second loop: normalize using act_addr (still at batch start)
        for i in range(hidden_dim // vlen):
            # normalized = (x - mean) * (1/std)
            # Store (x - mean) in scratchpad first
            generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{act_addr}, f2, 0, 0 \n"
            # Then multiply by 1/std and write back to activation
            generated_code += f"V_MUL_VF gp{act_addr}, gp{scratchpad_addr}, f5, 0 \n"

            # Move to next vector
            generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"

        # Reset accumulators for next batch
        generated_code += "S_ADD_FP f2, f0, f0 \n"
        generated_code += "S_ADD_FP f3, f0, f0 \n"

    return generated_code