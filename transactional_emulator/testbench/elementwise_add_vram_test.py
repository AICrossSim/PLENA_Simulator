import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

from compiler.asm_templates import elementwise_add_vram_asm, preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


def quantize_to_mxfp(tensor):
    """Quantize tensor to MXFP format matching hardware."""
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8])
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    # Test parameters: elementwise add on a single batch with [seq_len, feature_dim]
    # VRAM layout: B,S,H scheme where B=batch, S=sequence, H=hidden
    batch_size = 1
    seq_len = 14 * 14
    feature_dim = 128
    vlen = 64
    prefetch_len = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)
    
    # Create two input tensors [seq_len, feature_dim]
    vector_a = torch.randn(seq_len, feature_dim, dtype=torch.bfloat16)
    vector_b = torch.randn(seq_len, feature_dim, dtype=torch.bfloat16)
    
    # Quantize to MXFP
    vector_a_mxfp = quantize_to_mxfp(vector_a).to(torch.bfloat16)
    vector_b_mxfp = quantize_to_mxfp(vector_b).to(torch.bfloat16)
    
    # Compute golden output (elementwise addition)
    golden_output = vector_a_mxfp + vector_b_mxfp
    
    golden_result = {
        "vector_a": vector_a_mxfp,
        "vector_b": vector_b_mxfp,
        "original_output": golden_output,
    }

    gen_assembly_code = "; Elementwise Add Vector Test Generation\n"
    gen_assembly_code += f"; Shape: (seq_len={seq_len}, feature_dim={feature_dim})\n"
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[0, 1],
        available_registers=[0, 1],
        addr_reg_val=[0, int(seq_len * feature_dim * real_data_ratio)],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6, 7])

    # Preload both inputs into VRAM using the standard activation layout.
    # For preload_act_asm: treat seq_len as the "batch" dimension (number of rows to load)
    # hidden_size is the feature dimension per row
    total_elements = seq_len * feature_dim
    scaled_offset = int(seq_len * feature_dim * real_data_ratio)  # Account for scale data in MXFP
    
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=prefetch_len,
        batch=seq_len,  # Number of rows in S dimension
        hidden_size=feature_dim,  # Number of elements (H) per row
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=feature_dim,
    )
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=prefetch_len,
        batch=seq_len,  # Number of rows in S dimension
        hidden_size=feature_dim,  # Number of elements (H) per row
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=scaled_offset,
        activation_offset_reg=1,
        stride_size=feature_dim,
    )

    num_vectors = (seq_len * feature_dim) // vlen
    gen_assembly_code += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=num_vectors,
        alive_registers=[1, 2],
        dst_base_address=0,
        src_base_address=scaled_offset,
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(
        {"vector_a": vector_a_mxfp, "vector_b": vector_b_mxfp},
        gen_assembly_code,
        golden_result,
        [0.0, 1e-6],
        build_dir=build_path,
    )
    create_mem_for_sim(
        data_size=512,
        mode="behave_sim",
        asm="elementwise_add",
        data=None,
        specified_data_order=["vector_a", "vector_b"],
        build_path=build_path,
    )

    comparison_params = {
        "start_row_idx": 0,
        "num_rows": seq_len * feature_dim // vlen,
        "num_batches": seq_len,  
        "elements_per_batch": feature_dim,  
        "use_stride_mode": True,  
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating elementwise_add test assembly code")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}, Feature dim: {feature_dim}")
    print(f"Vector A shape: {vector_a_mxfp.shape}")
    print(f"Vector B shape: {vector_b_mxfp.shape}")
    print(f"Output shape: {golden_output.shape}")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
