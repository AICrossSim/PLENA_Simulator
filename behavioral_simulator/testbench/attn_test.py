
# This test is about the prefilling stage of the flash attention process.

import sys
import math
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from torch import Tensor, nn
from kernels import get_kernel
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import flash_attn_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from create_sim_env import create_sim_env
from transformers.modeling_flash_attention_utils import _flash_attention_forward as _flash_attention_forward_ref
from aria_lm_ops.models.llama import flash_attn2_gemv
from sim_env_utils import create_mem_for_sim


# Reshape q, k, v so that each batch's data stays together and rows are of length mem_row_size
# def flatten_and_split_by_rows(x, mem_row_size):
#     b = x.shape[0]
#     x_flat = x.reshape(b, -1)
#     total = x_flat.shape[1]
#     pad = (mem_row_size - (total % mem_row_size)) % mem_row_size
#     # Pad each batch along feature dim so divisible
#     if pad > 0:
#         x_flat = torch.cat([x_flat, torch.zeros((b, pad), dtype=x.dtype, device=x.device)], dim=1)
#     x_2d = x_flat.reshape(b, -1, mem_row_size)
#     return x_2d


if __name__ == "__main__":
    # Currently single batch test
    batch_size = 1
    s_q =64
    s_kv = 64
    num_q_heads = 16
    num_kv_heads = 4
    h_qkv = 16
    hidden_size = h_qkv * num_q_heads
    mlen = 64
    vlen = 64
    blen = 4
    qk_scale = 1.0 / math.sqrt(h_qkv)
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, qk_scale, -float("inf")]
    mem_row_size = 512

    # Set device - use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("fp preload:", fp_preload)

    torch.manual_seed(42)
    # in shape of b, s, h, d
    q = torch.randn(batch_size, s_q, num_q_heads, h_qkv, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device) 

    # Set print options to avoid "..." truncation for high-dimensional tensors
    torch.set_printoptions(edgeitems=20, threshold=20000, linewidth=200)

    input_tensor = {
        "q": q.reshape(batch_size, -1),
        "k": k.reshape(batch_size, -1),
        "v": v.reshape(batch_size, -1)
    }

    print("q reshaped shape:", q.reshape(batch_size, -1)[:, : num_q_heads * h_qkv - 1])

    weights = torch.zeros(h_qkv)

    # TODO: Add a reference flashattention output to ensure the flash_attn2_gemv is correct.
    
    print("q now shape:", q.shape)
    original_output = flash_attn2_gemv(
        q,
        k,
        v,
        qk_scale=qk_scale,
        s_q=s_q,
        s_kv=s_kv,
        h_qkv=h_qkv,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        Bc=mlen,
        Br=mlen
    )

    golden_result = {
        "input_tensor": input_tensor,
        "weights": weights,
        "original_output": original_output
    }

    gen_assembly_code = "; FlashAttn Test Generation \n"

    # Set the Q, K, V addr offset for weight and bias
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[int((h_qkv * num_q_heads) * batch_size * s_q * real_data_ratio), int((h_qkv * num_kv_heads * batch_size * s_kv + h_qkv * num_q_heads * batch_size * s_q) * real_data_ratio)]
    )

    # Gen Activation Preload Q
    gen_assembly_code += preload_act_asm(
        vlen=mlen,
        preload_len=4,
        batch=batch_size,
        hidden_size=h_qkv * num_q_heads * s_q,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4,5]
    )

    # Start the flash attention process
    gen_assembly_code += flash_attn_asm(
        mlen=mlen,
        blen=blen,
        batch=batch_size,
        hq=num_q_heads,
        hkv=num_kv_heads,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        alive_registers_int=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        alive_registers_fp=[1,2,3,4,5,6,7],
        vector_sram_base_address=0,
        fp_sram_start_address=3,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2
    )


    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order = ["q", "k", "v"])

    import json
    result_vram_offset = 0  # activation_base_address
    effective_batch = batch_size * s_q
    result_start_row = result_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
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