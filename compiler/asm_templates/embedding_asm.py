import os
from typing import Dict, List, Any, Optional
from pathlib import Path



def embedding_asm(
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    alive_registers: List[int],
    voc_table_row_size: int,
    activation_base_address: int,
    voc_table_base_addr_reg_index: int,
    input_ids: list[int]
) -> str:
    """
    Generates assembly code for embedding lookup operation.
    Returns:
        str: elementwise add, previous layer's activation add with the current layer's activation.
    """
    assert len(input_ids) == batch, "Input IDs length must match batch"
    generated_code = "; Embedding_asm generation \n"
    indx_reg = alive_registers[0]
    table_entry_addr = alive_registers[1] 
    load_v_on_chip_addr = alive_registers[2]
    load_m_on_chip_addr = alive_registers[3]
    hidden_size = hidden_size

    generated_code += f"S_ADDI_INT gp{table_entry_addr}, gp0, {voc_table_row_size} \n"
    generated_code += f"S_ADDI_INT gp{load_v_on_chip_addr}, gp0, {activation_base_address} \n"

    # Need to perform dot product with dim (hidden_size, hidden_size) @ (hidden_size, batch_size)
    for m in range(hidden_size // blen):
        for j in range(hidden_size // mlen):
            for i in range(blen):
                if m == 0:
                    # Load to on-chip memory
                    input_id = input_ids[i]
                    generated_code += f"S_ADDI_INT gp{indx_reg}, gp0, {input_id} \n"
                    generated_code += f"S_MUL_INT gp{indx_reg}, gp{indx_reg}, gp{table_entry_addr} \n"
                    generated_code += f"H_PREFETCH_V gp{load_v_on_chip_addr}, gp{indx_reg}, a{voc_table_base_addr_reg_index}, 0, 0 \n"
                    generated_code += f"S_ADDI_INT gp{load_v_on_chip_addr}, gp{load_v_on_chip_addr}, {mlen} \n"
                generated_code += f"H_PREFETCH_M gp{load_m_on_chip_addr}, gp{indx_reg}, a{voc_table_base_addr_reg_index}, 0, 0 \n"
                generated_code += f"S_ADDI_INT gp{load_m_on_chip_addr}, gp{load_m_on_chip_addr}, {mlen} \n"
            generated_code += f"M_MM gp{load_m_on_chip_addr}, gp{load_m_on_chip_addr}, gp{load_v_on_chip_addr} \n"
        generated_code += f"M_MM_WO gp{load_v_on_chip_addr}, {0} \n"
    return generated_code
