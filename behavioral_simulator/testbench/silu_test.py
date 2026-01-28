import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn
from compiler.asm_templates import preload_act_asm, reset_reg_asm, silu_asm
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Uses the same quantizer as the behavioral simulator's memory loader.
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 4
    vlen = 64

    # FP SRAM layout: [0]=0.0, [1]=1.0 (for sigmoid computation)
    fp_preload = [0.0, 1.0]

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)

    print("Input tensor shape:", act_tensor.shape)
    print("Input tensor (first 8 values):", act_tensor[0, :8])

    # Quantize input to MXFP to match hardware precision
    act_mxfp = quantize_to_mxfp(act_tensor).to(act_tensor.dtype)

    # Compute golden output using PyTorch with quantized input
    original_output = nn.functional.silu(act_mxfp)

    print("Output tensor (first 8 values):", original_output[0, :8])

    input_tensor = {
        "act_tensor": act_mxfp,  # Use MXFP-quantized to match simulator
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; SiLU Test Generation\n"

    # Reset registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1, 2, 3]
    )

    # Preload activations
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=batch_size,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size
    )

    # Reset registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1, 2, 3, 4]
    )

    # SiLU computation
    gen_assembly_code += silu_asm(
        const_one_fp_address=1,
        alive_registers=[1, 2, 3, 4, 5],
        activation_base_address=0,
        scratchpad_base_address=hidden_size * batch_size,
        vlen=vlen,
        batch_size=batch_size,
        hidden_dim=hidden_size
    )

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="silu", data=None, specified_data_order=["act_tensor"])

    # Save comparison parameters for view_mem.py
    import json
    result_vram_offset = 0  # In-place computation
    result_start_row = result_vram_offset // vlen
    num_result_rows = (batch_size * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating SiLU test assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
