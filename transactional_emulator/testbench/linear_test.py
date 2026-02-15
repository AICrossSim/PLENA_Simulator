import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn
from compiler.asm_templates import projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Uses the same quantizer as the transactional emulator's memory loader.
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    # Testing rectangular linear: (batch, in_features) @ (in_features, out_features) -> (batch, out_features)
    in_features = 128
    out_features = 256  # Rectangular matrix test
    batch_size = 8
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6, 1/in_features]

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
    original_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False, dtype=torch.bfloat16)
    weights = original_layer.state_dict()

    # Quantize inputs to MXFP (E4M3) to match hardware precision
    # Hardware stores data in HBM as MXFP, then loads to VRAM as dequantized values
    # bm_x is the dequantized value that hardware computes with
    act_mxfp = quantize_to_mxfp(act_tensor).to(act_tensor.dtype)  # (batch_size, in_features)
    weights_mxfp = quantize_to_mxfp(weights['weight'].t()).to(act_tensor.dtype)  # (in_features, out_features)

    # Compute golden with MXFP-quantized inputs (float32 accumulation like hardware)
    original_output = torch.mm(act_mxfp, weights_mxfp)

    print(f"Linear: ({batch_size}, {in_features}) @ ({in_features}, {out_features}) -> ({batch_size}, {out_features})")
    print("original_output shape:", original_output.shape)
    print("original_output is:\n", original_output)

    # Weight is stored as (out_features, in_features) in PyTorch, we transpose for our layout
    # Our layout: (in_features, out_features) for matmul: act @ weight
    # Store original (non-quantized) tensors - they will be quantized when loaded to HBM
    input_tensor = {
        "act_tensor": act_mxfp,  # Use MXFP-quantized to match simulator
        "weights": weights_mxfp,  # Use MXFP-quantized to match simulator
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; Linear Test Generation (Rectangular Matrix)\n"
    gen_assembly_code += f"; Shape: ({batch_size}, {in_features}) @ ({in_features}, {out_features}) -> ({batch_size}, {out_features})\n"

    # Calculate HBM offsets
    # Layout in HBM: [activations | weights]
    act_hbm_size = int(in_features * batch_size * real_data_ratio)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((in_features * batch_size + in_features * out_features) * real_data_ratio)

    # Set the addr offset for weight
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[weight_hbm_offset, weight_hbm_end]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3]
    )

    # Gen Activation Preload
    gen_assembly_code += preload_act_asm(
        vlen=64,
        preload_len=4,
        batch=batch_size,
        hidden_size=in_features,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=in_features
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4]
    )

    # Result is stored after activation in VRAM
    result_vram_offset = in_features * batch_size

    gen_assembly_code += projection_asm(
        mlen=64,
        blen=4,
        batch=batch_size,
        hidden_size=in_features,      # in_features (input dimension)
        out_features=out_features,     # out_features (output dimension) - rectangular support!
        alive_registers=[1,2,3,4,5,6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=result_vram_offset,
        rope_enabled=False
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order=["act_tensor", "weights"], build_path=build_path)

    # Save comparison parameters for view_mem.py
    import json
    result_start_row = result_vram_offset // 64  # Row where results start
    num_result_rows = (batch_size * out_features) // 64
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": out_features
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
