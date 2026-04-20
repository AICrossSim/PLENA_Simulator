import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

from compiler.asm_templates import preload_act_asm, reset_reg_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8])
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    hidden_size = 64
    batch_size = 4
    vlen = 64

    # Shift amount to test
    shift_amount = 2

    # FP SRAM layout: [0]=0.0, [1]=1.0
    fp_preload = [0.0, 1.0]

    torch.manual_seed(42)
    # Use integer-like values for clearer shift testing
    act_tensor = torch.randint(1, 16, (batch_size, hidden_size), dtype=torch.bfloat16)

    print("Input tensor shape:", act_tensor.shape)
    print("Input tensor (first 8 values):", act_tensor[0, :8])

    # Quantize input to MXFP to match hardware precision
    act_mxfp = quantize_to_mxfp(act_tensor).to(act_tensor.dtype)

    # Compute golden output: element shift right by shift_amount
    # [a0, a1, a2, ...] -> [0, 0, ..., a0, a1, a2, ...]
    # Each row (vlen elements) is shifted independently
    original_output = act_mxfp.clone()
    for batch_idx in range(batch_size):
        for row_start in range(0, hidden_size, vlen):
            row_end = row_start + vlen
            row = act_mxfp[batch_idx, row_start:row_end]
            if shift_amount >= vlen:
                # All zeros
                original_output[batch_idx, row_start:row_end] = 0
            elif shift_amount == 0:
                # No shift
                original_output[batch_idx, row_start:row_end] = row
            else:
                # Shift elements right, fill with zeros from the left
                shifted = torch.cat([torch.zeros(shift_amount, dtype=row.dtype), row[:vlen - shift_amount]])
                original_output[batch_idx, row_start:row_end] = shifted

    print(f"Shift amount: {shift_amount}")
    print("Input tensor (first 8 values):", act_mxfp[0, :8])
    print("Output tensor (first 8 values):", original_output[0, :8])

    input_tensor = {
        "act_tensor": act_mxfp,
    }

    golden_result = {"input_tensor": input_tensor, "original_output": original_output}

    gen_assembly_code = "; V_SHIFT_V Test Generation\n"

    # Reset registers
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

    # Preload activations to VRAM starting at address 0
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=batch_size,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size,
    )

    # Reset registers
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])

    # Set up registers for V_SHIFT_V:
    # gp1 = source/dest vector address (0)
    # gp2 = shift amount
    gen_assembly_code += "S_ADDI_INT gp1, gp0, 0\n"  # gp1 = 0 (source vector address)
    gen_assembly_code += f"S_ADDI_INT gp2, gp0, {shift_amount}\n"  # gp2 = shift amount

    # Apply V_SHIFT_V to each vector row
    # V_SHIFT_V rd, rs1, rs2
    # rd = destination address register, rs1 = source address register, rs2 = shift amount register
    total_vectors = (batch_size * hidden_size) // vlen
    for i in range(total_vectors):
        gen_assembly_code += "V_SHIFT_V gp1, gp1, gp2\n"  # In-place shift
        gen_assembly_code += f"S_ADDI_INT gp1, gp1, {vlen}\n"  # Move to next vector

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="v_shft_v",
        data=None,
        specified_data_order=["act_tensor"],
        build_path=build_path,
    )

    # Save comparison parameters for view_mem.py
    import json

    result_vram_offset = 0  # In-place computation
    result_start_row = result_vram_offset // vlen
    num_result_rows = (batch_size * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating V_SHIFT_V test assembly code")
    print(f"Shift amount: {shift_amount}")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
