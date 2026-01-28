import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from compiler.asm_templates import preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from create_sim_env import create_sim_env
import math
from sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # Testing H_STORE_V: Preload activations from HBM and store them back
    in_features = 128
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6]

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, in_features)
    
    print(f"H_STORE_V Test: ({batch_size}, {in_features}) activations")
    print("original_output shape:", act_tensor.shape)
    print("original_output is:\n", act_tensor)

    input_tensor = {
        "act_tensor": act_tensor,
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": act_tensor  # Store activations as golden output
    }

    gen_assembly_code = "; H_STORE_V Test Generation\n"
    gen_assembly_code += f"; Preload activations from HBM and store them back\n"
    gen_assembly_code += f"; Shape: ({batch_size}, {in_features})\n"

    # Calculate HBM offsets
    # Layout in HBM: [activations | stored_copy]
    act_hbm_size = int(in_features * batch_size * real_data_ratio)
    stored_copy_hbm_offset = act_hbm_size  # Store copy after original activations

    # Set HBM address register for activation source (a0)
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[0],
        available_registers=[0],
        addr_reg_val=[0]  # Activations start at offset 0
    )

    # Set HBM address register for storing copy (a1)
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[stored_copy_hbm_offset]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3]
    )

    # Set scale register (required for mx data type)
    # Scale offset = batch * hidden_size for activations
    scale_offset = int(batch_size * in_features)
    gen_assembly_code += f"S_ADDI_INT gp8, gp0, {scale_offset}\n"
    gen_assembly_code += f"C_SET_SCALE_REG gp8\n"

    # Gen Activation Preload from HBM to VRAM
    gen_assembly_code += preload_act_asm(
        vlen=64,
        preload_len=batch_size,  # Preload all batches
        batch=batch_size,
        hidden_size=in_features,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=in_features
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4,5]
    )

    # Set stride register for storing (hidden_size for batch-wise storage)
    gen_assembly_code += f"S_ADDI_INT gp7, gp0, {in_features}\n"
    gen_assembly_code += f"C_SET_STRIDE_REG gp7\n"

    # Store activations from VRAM back to HBM using H_STORE_V
    # H_STORE_V rd, rs1, rs2, rstride, precision
    # rd: VRAM source address (0, where we loaded activations)
    # rs1: HBM offset (0, relative to a1 base)
    # rs2: HBM address register (a1)
    # rstride: 1 (use STRIDE_REG)
    # precision: 0 (Activation)
    vlen = 64
    store_v_amount = batch_size  # HBM_V_Writeback_Amount
    
    # Calculate number of H_STORE_V calls needed
    # Each H_STORE_V stores STORE_V_AMOUNT * VLEN elements
    # We need to store batch_size * in_features elements
    total_elements = batch_size * in_features
    num_store_calls = (total_elements + store_v_amount * vlen - 1) // (store_v_amount * vlen)
    
    # Store activations in chunks
    for j in range(math.ceil(in_features // vlen)):
        for i in range(math.ceil(batch_size / store_v_amount)):
            vram_addr = j * vlen + i * store_v_amount * vlen
            hbm_offset = j * vlen + i * store_v_amount * vlen
            gen_assembly_code += f"S_ADDI_INT gp9, gp0, {vram_addr}\n"  # VRAM source
            gen_assembly_code += f"S_ADDI_INT gp10, gp0, {hbm_offset}\n"  # HBM offset
            gen_assembly_code += f"H_STORE_V gp9, gp10, a1, 1, 0\n"  # Store with stride

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="h_store", data=None, specified_data_order=["act_tensor"])

    # Save comparison parameters for checking HBM content
    import json
    stored_copy_hbm_start_byte = stored_copy_hbm_offset
    stored_copy_hbm_size_bytes = int(batch_size * in_features * real_data_ratio)
    
    # Calculate VRAM row indices for compatibility with view_mem.py
    # Activations are stored in VRAM starting at offset 0
    vram_start_row = 0
    vram_num_rows = (batch_size * in_features + vlen - 1) // vlen  # Round up
    
    comparison_params = {
        # Standard VRAM viewing parameters (for compatibility with view_mem.py)
        "start_row_idx": vram_start_row,
        "num_rows": vram_num_rows,
        "num_batches": batch_size,
        "elements_per_batch": in_features,
        
        # HBM-specific parameters
        "result_hbm_start_byte": stored_copy_hbm_start_byte,
        "result_hbm_size_bytes": stored_copy_hbm_size_bytes,
        "scale_offset": scale_offset,  # Distance from elements to scales in HBM
        "vlen": vlen,
        "check_hbm": True,  # Flag to indicate this test checks HBM
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating assembly code")
    print(f"Original activations: byte 0, size {act_hbm_size} bytes")
    print(f"Stored copy location: byte {stored_copy_hbm_start_byte}, size {stored_copy_hbm_size_bytes} bytes")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
