from typing import List


def silu_asm(
    const_one_fp_address: int,
    alive_registers: List[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int
) -> str:
    """
    Generate assembly code for SiLU activation.

    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))

    Args:
        const_one_fp_address: FP SRAM address containing constant 1.0
        alive_registers: List of available integer registers
        activation_base_address: VRAM base address for input activations
        scratchpad_base_address: VRAM base address for intermediate results
        vlen: Vector length (number of elements per vector)
        batch_size: Batch size dimension
        hidden_dim: Hidden dimension size

    Returns:
        Generated assembly code string
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    loop_reg = alive_registers[2]

    num_vectors = (batch_size * hidden_dim) // vlen

    generated_code = "; SiLU Activation Generation\n"
    generated_code += f"S_ADDI_INT gp{act_addr}, gp0, {activation_base_address}\n"
    generated_code += f"S_ADDI_INT gp{scratchpad_addr}, gp0, {scratchpad_base_address}\n"

    # Load constant 1.0 into f1
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    generated_code += f"C_LOOP_START gp{loop_reg}, {num_vectors}\n"

    # SiLU computation: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    # Step 1: -x (negate using f0=0, with reverse order flag)
    generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{act_addr}, f0, 0, 1\n"
    # Step 2: exp(-x)
    generated_code += f"V_EXP_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # Step 3: 1 + exp(-x)
    generated_code += f"V_ADD_VF gp{scratchpad_addr}, gp{scratchpad_addr}, f1, 0\n"
    # Step 4: 1 / (1 + exp(-x)) = sigmoid(x)
    generated_code += f"V_RECI_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # Step 5: x * sigmoid(x) = silu(x), store in-place
    generated_code += f"V_MUL_VV gp{act_addr}, gp{scratchpad_addr}, gp{act_addr}, 0\n"

    # Move to next vector
    generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_reg}\n"

    return generated_code
