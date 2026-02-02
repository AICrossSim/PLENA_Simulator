"""PV multiplication test for flashattn.pv module.

Tests the computing_pv_code function which computes:
P @ V where P is the softmax attention scores and V is the value matrix.
Output is written in packed format with heads interleaved.

Memory layout:
- HBM: P (q_index_2_kv_index_ratio, MLEN, MLEN) at 0, then V (MLEN, h_qkv)
- VSRAM: P preloaded, then PV output
"""

import sys
import json
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compiler.asm_templates import preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from compiler.asm_templates.flashattn import computing_pv_code, reset_kv_prefetch
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # Test configuration
    batch_size = 1
    num_q_heads = 4
    num_kv_heads = 1
    h_qkv = 16
    mlen = 64
    vlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    s_q = 64
    s_kv = 64

    q_index_2_kv_index_ratio = num_q_heads // num_kv_heads
    hidden_size = h_qkv * num_q_heads

    device = torch.device("cpu")
    print(f"PV Test Config:")
    print(f"  batch_size={batch_size}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, h_qkv={h_qkv}")
    print(f"  q_index_2_kv_index_ratio={q_index_2_kv_index_ratio}")
    print(f"  blen={blen}, mlen={mlen}")

    torch.manual_seed(42)
    # P shape: (q_index_2_kv_index_ratio, mlen, mlen) - softmax output for each Q head
    # Using random positive values to simulate softmax output (unnormalized)
    p = torch.rand(batch_size, q_index_2_kv_index_ratio, mlen, mlen, dtype=torch.bfloat16, device=device)
    # V shape: (batch, s_kv, num_kv_heads, h_qkv) - value matrix
    v = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device)

    print(f"\nTensor shapes:")
    print(f"  P: {p.shape}")
    print(f"  V: {v.shape}")

    # Reshape V to match num_q_heads (zero padding for GQA)
    if num_kv_heads < num_q_heads:
        v_reshaped = torch.zeros(batch_size, s_kv, num_q_heads, h_qkv, dtype=v.dtype, device=v.device)
        v_reshaped[:, :, :num_kv_heads, :] = v
        # The remaining extra heads are left as zeros
    elif num_kv_heads == num_q_heads:
        v_reshaped = v
    else:
        raise ValueError("num_kv_heads > num_q_heads not supported for zero padding logic.")

    print(f"V reshaped to {v_reshaped.shape}")

    input_tensor = {
        "p": p.reshape(batch_size, -1),
        "v": v_reshaped.reshape(batch_size, -1),
    }

    # Compute golden output: PV = P @ V for each Q head
    golden_pv_list = []
    for q_head in range(q_index_2_kv_index_ratio):
        kv_head = q_head // q_index_2_kv_index_ratio  # Map Q head to KV head
        p_head = p[0, q_head, :, :]  # [mlen, mlen]
        v_2d = v[0, :, kv_head, :]  # [s_kv, h_qkv]
        pv = torch.matmul(p_head, v_2d)  # [mlen, h_qkv]
        golden_pv_list.append(pv)
        print(f"  Q head {q_head} (KV head {kv_head}) -> PV shape: {pv.shape}")

    print(f"\nGolden PV computed for {len(golden_pv_list)} heads")

    # Memory layout in VSRAM
    p_base_address = 0
    p_total_size = q_index_2_kv_index_ratio * mlen * mlen
    pv_base_address = p_base_address + p_total_size

    print(f"\nVSRAM Layout:")
    print(f"  P Base: {p_base_address}")
    print(f"  P Total Size: {p_total_size}")
    print(f"  PV Output Base: {pv_base_address}")

    # HBM layout
    p_hbm_size = int(q_index_2_kv_index_ratio * mlen * mlen * batch_size * real_data_ratio)
    v_hbm_offset = p_hbm_size

    print(f"\nHBM Layout:")
    print(f"  P: 0 - {p_hbm_size}")
    print(f"  V offset: {v_hbm_offset}")

    # Generate assembly
    gen_assembly_code = "; PV Test Generation \n"

    # Set V addr offset register
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1, 2],
        addr_reg_val=[v_hbm_offset]
    )

    # Preload P to VSRAM
    gen_assembly_code += preload_act_asm(
        vlen=mlen,
        preload_len=4,
        batch=batch_size * q_index_2_kv_index_ratio * mlen,
        hidden_size=mlen,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    # Set up V prefetch
    gen_assembly_code += reset_kv_prefetch(
        hkv=num_kv_heads,
        d=h_qkv,
        mlen=mlen,
        kv_len=s_kv,
        batch=batch_size,
        alive_registers_int=[1],
    )

    # PV multiplication for each Q head
    gen_assembly_code += f"; === Q head {q_head} ===\n"
    for q_head in range(q_index_2_kv_index_ratio):
        gen_assembly_code += computing_pv_code(
            head_dim=h_qkv,
            blen=blen,
            mlen=mlen,
            vlen=mlen,
            alive_registers=[1, 2, 3, 4, 5, 6],
            p_base_address=p_base_address + q_head * mlen * mlen,
            v_base_hbm_offset_reg=1,
            q_head_index=q_head,
            v_head_index=0,
            output_base_address=pv_base_address + q_head * mlen * h_qkv,
            head_offset=q_head * h_qkv,
        )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

    # Golden result - PV for head 0 only (first 16 elements per row)
    # Output is [mlen, h_qkv] = [64, 16]
    golden_pv_head0 = golden_pv_list[0]  # [mlen, h_qkv]

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden_pv_head0.reshape(-1).unsqueeze(0)
    }

    print(f"\nGolden PV head0 shape: {golden_pv_head0.shape}")
    print(f"Golden output flattened shape: {golden_pv_head0.reshape(-1).shape}")

    fp_preload = [0.0]
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order=["p", "v"])

    result_start_row = pv_base_address // vlen
    num_result_rows = mlen  # 64 rows of output

    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": 1,
        "elements_per_batch": mlen * h_qkv,
        "row_dim": vlen,
        "use_stride_mode": False,
        "use_slice_mode": True,  # Extract first h_qkv elements from each vlen-wide row
        "slice_per_row": h_qkv
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print(f"\nResult location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("=" * 60)
