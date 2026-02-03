import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
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
    # Testing the operation (hidden_size, hidden_size) @ (hidden_size, batch_size)
    hidden_size = 128
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6, 1/hidden_size]

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    original_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, dtype=torch.bfloat16)
    weights = original_layer.state_dict()

    # Quantize inputs to MXFP to match hardware precision
    act_mxfp = quantize_to_mxfp(act_tensor).to(act_tensor.dtype)
    weights_mxfp = quantize_to_mxfp(weights['weight'].t()).to(act_tensor.dtype)

    # Compute golden with MXFP-quantized inputs
    original_output = torch.mm(act_mxfp, weights_mxfp)
    print("original_output is:\n", original_output)

    input_tensor = {
        "act_tensor": act_tensor,
        "weights": weights['weight'].t(),
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; Linear Test with Loop Instructions \n"

    # Set the addr offset for weight and bias
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[int(hidden_size * batch_size * real_data_ratio), int((hidden_size * (batch_size + 1) + hidden_size * hidden_size) * real_data_ratio)]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3]
    )

    # Gen Activation Preload
    gen_assembly_code += preload_act_asm(
        vlen=64,
        preload_len=4,
        batch=4,
        hidden_size=128,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size
    )

    # Reset the registers - need 7 for loop version
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4,5,6,7]
    )

    # Use the projection assembly
    gen_assembly_code += projection_asm(
        mlen=64,
        blen=4,
        batch=4,
        hidden_size=128,
        alive_registers=[1,2,3,4,5,6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=hidden_size * batch_size,
        rope_enabled=False
    )

    # Print generated assembly for comparison
    print("=" * 60)
    print("Generated Assembly Code (Loop Version):")
    print("=" * 60)
    print(gen_assembly_code)
    print("=" * 60)

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order=["act_tensor", "weights"], build_path=build_path)

    # Save comparison parameters for view_mem.py
    import json
    result_start_row = (hidden_size * batch_size) // 64  # Row where results start
    num_result_rows = (batch_size * hidden_size) // 64
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
    print("Finished generating assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
