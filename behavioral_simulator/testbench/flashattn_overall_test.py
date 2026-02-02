"""Overall Flash Attention test for flashattn.overall module.

Tests the complete flash_attn_asm function which orchestrates:
- QKT multiplication
- Online softmax
- PV multiplication
- Output accumulation and normalization

This is the end-to-end test for the full flash attention pipeline.
"""

import sys
import json
import math
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compiler.asm_templates import preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from compiler.asm_templates.flashattn import flash_attn_asm
from create_sim_env import create_sim_env
from aria_lm_ops.models.llama import flash_attn2_gemv
from sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # Test configuration
    batch_size = 1
    s_q = 64
    s_kv = 64
    num_q_heads = 4
    num_kv_heads = 1
    h_qkv = 16
    hidden_size = h_qkv * num_q_heads
    mlen = 64
    vlen = 64
    blen = 4
    qk_scale = 1.0 / math.sqrt(h_qkv)
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, qk_scale, -float("inf")]

    device = torch.device("cpu")
    print(f"flashattn.overall Full Test Config:")
    print(f"  batch_size={batch_size}, s_q={s_q}, s_kv={s_kv}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, h_qkv={h_qkv}")
    print(f"  hidden_size={hidden_size}, qk_scale={qk_scale:.6f}")
    print(f"  fp_preload: {fp_preload}")

    torch.manual_seed(42)
    q = torch.randn(batch_size, s_q, num_q_heads, h_qkv, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device)

    torch.set_printoptions(edgeitems=20, threshold=20000, linewidth=200)
    if num_kv_heads < (mlen // h_qkv):
        k_reshaped = torch.zeros(batch_size, s_kv, (mlen // h_qkv), h_qkv, dtype=k.dtype, device=k.device)
        v_reshaped = torch.zeros(batch_size, s_kv, (mlen // h_qkv), h_qkv, dtype=v.dtype, device=v.device)
        k_reshaped[:, :, :num_kv_heads, :] = k
        v_reshaped[:, :, :num_kv_heads, :] = v
        # The remaining extra heads are left as zeros
    elif num_kv_heads == (mlen // h_qkv):
        k_reshaped = k
        v_reshaped = v
    else:
        raise ValueError("num_kv_heads > num_q_heads not supported for zero padding logic.")

    input_tensor = {
        "q": q.reshape(batch_size, -1),
        "k": k_reshaped.reshape(batch_size, -1),
        "v": v_reshaped.reshape(batch_size, -1)
    }

    print(f"\nTensor shapes:")
    print(f"  q: {q.shape} -> reshaped: {q.reshape(batch_size, -1).shape}")
    print(f"  k: {k.shape} -> reshaped: {k.reshape(batch_size, -1).shape}")
    print(f"  v: {v.shape} -> reshaped: {v.reshape(batch_size, -1).shape}")

    # Compute golden output using reference implementation with intermediates
    print("\nComputing golden output with intermediate values...")
    original_output, all_intermediates = flash_attn2_gemv(
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
        Br=mlen,
        debug=True,
        return_intermediates=True
    )

    # Reshape output for golden comparison
    original_output_flat = original_output.reshape(batch_size * s_q, hidden_size)
    print(f"  original_output shape: {original_output.shape}")
    print(f"  original_output_flat shape: {original_output_flat.shape}")

    # Extract PV value for head 0, tile (batch=0, q_tile=0, kv_tile=0)
    # golden_pv_head0 = all_intermediates[0]["intermediates"][(0, 0, 0)]["pv"]
    # print(f"\nGolden PV for head 0 shape: {golden_pv_head0.shape}")
    # print(f"Golden PV for head 0:\n{golden_pv_head0}")


    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output.reshape(s_q, num_q_heads * h_qkv)
    }

    # Generate assembly
    gen_assembly_code = "; flashattn.overall Full Flash Attention Test \n"
    gen_assembly_code += f"; Config: batch={batch_size}, s_q={s_q}, s_kv={s_kv}, hq={num_q_heads}, hkv={num_kv_heads}, d={h_qkv}\n"

    # Calculate HBM offsets for K and V
    q_hbm_size = int(s_q * num_q_heads * h_qkv * batch_size * real_data_ratio)
    k_hbm_size = int(s_kv * (mlen // h_qkv) * h_qkv * batch_size * real_data_ratio)
    k_hbm_offset = q_hbm_size
    v_hbm_offset = q_hbm_size + k_hbm_size

    print(f"\nHBM Layout:")
    print(f"  Q: 0 - {q_hbm_size} (size: {q_hbm_size})")
    print(f"  K: {k_hbm_offset} - {k_hbm_offset + k_hbm_size} (size: {k_hbm_size})")
    print(f"  V: {v_hbm_offset} (size: {k_hbm_size})")

    # Set the K, V addr offset registers
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[k_hbm_offset, v_hbm_offset]
    )

    # Preload Q to VSRAM
    gen_assembly_code += preload_act_asm(
        vlen=mlen,
        preload_len=4,
        batch=batch_size,
        hidden_size=h_qkv * num_q_heads * s_q,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0
    )

    # Reset registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1, 2, 3, 4, 5]
    )

    # Run full flash attention
    gen_assembly_code += flash_attn_asm(
        mlen=mlen,
        blen=blen,
        batch=batch_size,
        hq=num_q_heads,
        hkv=num_kv_heads,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        alive_registers_int=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        alive_registers_fp=[1, 2, 3, 4, 5, 6, 7],
        vector_sram_base_address=0,
        fp_sram_start_address=3,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2
    )

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order=["q", "k", "v"])

    # Calculate VRAM memory layout to find O_old base address
    q_index_2_kv_index_ratio = num_q_heads // num_kv_heads
    q_base_address = 0
    s_base_address = q_base_address + s_q * num_q_heads * h_qkv
    pv_base_address = s_base_address + mlen * mlen * q_index_2_kv_index_ratio
    o_old_base_address = pv_base_address + mlen * mlen * q_index_2_kv_index_ratio

    # Output is stored at o_old_base_address
    result_vram_offset = o_old_base_address
    effective_batch = batch_size * s_q
    result_start_row = result_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen

    # Extract golden FPSRAM values for head 0
    golden_l_new = all_intermediates[0]["intermediates"][(0, 0, 0)]["l_new"]  # [mlen]
    golden_exp_m_res = all_intermediates[0]["intermediates"][(0, 0, 0)]["exp_m_res"]  # [mlen]

    print(f"\nGolden FPSRAM values for head 0:")
    print(f"  l_new shape: {golden_l_new.shape}, values: {golden_l_new[:8]}")
    print(f"  exp_m_res shape: {golden_exp_m_res.shape}, values: {golden_exp_m_res[:8]}")

    # FPSRAM layout: fp_sram_start_address=3
    # head 0: m_old at 3, m_res at 3+mlen, l_old at 3+2*mlen
    fp_sram_start = 3
    fpsram_m_res_start = fp_sram_start + mlen  # m_res (stores exp(m_old - m_new))
    fpsram_l_start = fp_sram_start + 2 * mlen  # l_old/l_new location

    comparison_params = {
        # VRAM comparison params (default)
        "start_row_idx": result_start_row,
        "num_rows": mlen,
        "num_batches": 1,
        "elements_per_batch": mlen * h_qkv,
        "row_dim": vlen,
        "use_stride_mode": False,
        "use_slice_mode": True,
        "slice_per_row": h_qkv,
        # FPSRAM comparison flag and params
        "compare_fpsram": True,
        "fpsram_m_res_start": fpsram_m_res_start,
        "fpsram_l_start": fpsram_l_start,
        "fpsram_num_elements": mlen,
    }

    # Save golden FPSRAM values to file
    build_dir = Path(__file__).parent / "build"
    torch.save({
        "golden_l_new": golden_l_new,
        "golden_exp_m_res": golden_exp_m_res,
        "fpsram_m_res_start": fpsram_m_res_start,
        "fpsram_l_start": fpsram_l_start,
    }, build_dir / "golden_fpsram.pt")
    print(f"Saved golden FPSRAM values to: {build_dir / 'golden_fpsram.pt'}")

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("\n" + "=" * 60)
    print("VRAM Memory Layout:")
    print(f"  Q Base Address: {q_base_address}")
    print(f"  S Base Address: {s_base_address}")
    print(f"  PV Base Address: {pv_base_address}")
    print(f"  O_Old Base Address: {o_old_base_address}")
    print("=" * 60)
    print("Finished generating assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("=" * 60)
