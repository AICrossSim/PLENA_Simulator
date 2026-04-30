import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

from compiler.asm_templates import columnwise_scan_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


def quantize_to_mxfp(tensor):
    """Quantize tensor to MXFP format matching hardware."""
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8])
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    # Siglip 14x14 test parameter 
    rows = 14
    cols = 14
    feature_dim = 64
    vlen = 64

    torch.manual_seed(42)
    input_tensor = torch.randn(rows * cols, feature_dim, dtype=torch.bfloat16)
    input_mxfp = quantize_to_mxfp(input_tensor).to(torch.bfloat16)

    # Logical reshape: (rows, cols, D) -> (cols, rows, D)
    original_output = input_mxfp.view(rows, cols, feature_dim).transpose(0, 1).contiguous().view(rows * cols, feature_dim)
    print("Difference between original output and input:")
    print(sum(original_output - input_mxfp))

    print(f"Columnwise scan test: ({rows} * {cols}, {feature_dim}) -> ({cols} * {rows}, {feature_dim})")
    print("Input shape:", input_mxfp.shape)
    print("Output shape:", original_output.shape)

    golden_result = {"input_tensor": input_mxfp, "original_output": original_output}

    gen_assembly_code = "; Columnwise Scan Test Generation\n"
    gen_assembly_code += f"; Shape: ({rows}, {cols}, {feature_dim}) -> ({cols}, {rows}, {feature_dim})\n"
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[0],
        available_registers=[0],
        addr_reg_val=[0],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])
    gen_assembly_code += columnwise_scan_asm(
        mlen=64,
        vlen=vlen,
        rows=rows,
        cols=cols,
        feature_dim=feature_dim,
        alive_registers=[1, 2, 3, 4],
        input_hbm_base_addr_reg=0,
        output_vram_base=0,
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(
        {"input_tensor": input_mxfp},
        gen_assembly_code,
        golden_result,
        [0.0, 1e-6],
        build_dir=build_path,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="columnwise_scan",
        data=None,
        specified_data_order=["input_tensor"],
        build_path=build_path,
    )

    comparison_params = {
        "start_row_idx": 0,
        "num_rows": (rows * cols * feature_dim) // vlen,
        "num_batches": rows * cols,
        "elements_per_batch": feature_dim,
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating columnwise scan test assembly code")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
