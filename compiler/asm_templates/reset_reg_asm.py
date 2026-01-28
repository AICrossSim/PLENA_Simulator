from typing import Dict, List

def reset_reg_asm(
    alive_registers: List[int],
) -> str:
    """
    Generates assembly code for resetting registers.
    """
    generated_code = f"; Reset Registers [{alive_registers}] \n"
    for register in alive_registers:
        generated_code += f"S_ADDI_INT gp{register}, gp0, 0 \n"
    return generated_code


def reset_fpreg_asm(
    alive_registers: List[int],
) -> str:
    """
    Generates assembly code for resetting floating point registers.
    """
    generated_code = f"; Reset Floating Point Registers [{alive_registers}] \n"
    for register in alive_registers:
        generated_code += f"S_ADD_FP f{register}, f0, f0 \n"
    return generated_code

def reset_vmask_asm(
    alive_register: int,
    vmask: int,
) -> str:
    """
    Generates assembly code for resetting vector mask.
    """
    generated_code = f"; Reset Vector Mask [{vmask}] \n"
    generated_code += f"S_ADDI_INT gp{alive_register}, gp0, {vmask} \n"
    generated_code += f"C_SET_V_MASK_REG gp{alive_register} \n"
    return generated_code