
# Online Softmax Test
# Tests the online_softmax_code function which computes:
# - Row-wise max (m_curr) with running max tracking (m_last)
# - Row-wise exp(S - m_curr) = P (softmax probabilities before normalization)
# - Running sum l = l_old * exp(m_last - m_curr) + sum(P)
#
# Configuration: h_qkv=16, num_q_heads=16, num_kv_heads=4, seq_len=64, batch=1
#
# Key constraints:
# - h_qkv must equal hardware HLEN (16) for correct M_BTMM operation
# - M_BTMM processes MLEN//HLEN = 64//16 = 4 Q heads in parallel per KV head
# - HBM stride alignment: num_kv_heads * h_qkv >= 64 (so num_kv_heads >= 4 with h_qkv=16)
# - With num_kv_heads=4, h_qkv=16: stride = 64, which is 64-byte aligned

import sys
import json
import math
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compiler.asm_templates import preload_act_asm, reset_reg_asm, preload_addr_reg_asm, reset_fpreg_asm
from compiler.asm_templates.flash_attn_asm import qkt_multiply, _reset_kv_prefetch
from compiler.asm_templates.flashattn import online_softmax_code, reset_fpsram_code
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # Multi-head test configuration
    # h_qkv must equal hardware HLEN (16) for correct M_BTMM operation
    # M_BTMM processes MLEN//HLEN = 4 Q heads in parallel per KV head
    # HBM stride = num_kv_heads * h_qkv = 4 * 16 = 64 (meets 64-byte alignment)
    batch_size = 1
    s_q = 64
    s_kv = 64
    num_q_heads = 4       # 16 Q heads total
    num_kv_heads = 1       # 4 KV heads (stride = 4 * 16 = 64, aligned)
    h_qkv = 16             # Must equal hardware HLEN = 16
    mlen = 64
    vlen = 64
    blen = 4
    qk_scale = 1.0 / math.sqrt(h_qkv)
    real_data_ratio = (8*8 + 8) / (8 * 8)

    q_index_2_kv_index_ratio = num_q_heads // num_kv_heads  # 4 Q heads per KV head

    device = torch.device("cpu")
    print(f"Multi-Head Online Softmax Test Config:")
    print(f"  batch_size={batch_size}, s_q={s_q}, s_kv={s_kv}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, h_qkv={h_qkv}")
    print(f"  q_index_2_kv_index_ratio={q_index_2_kv_index_ratio}")
    print(f"  qk_scale={qk_scale:.6f}")

    torch.manual_seed(42)
    # Q shape: (batch, s_q, num_q_heads, h_qkv) = (1, 64, 16, 16)
    # qkt_multiply expects Q in standard torch layout - no permutation needed
    # M_BTMM uses strided access internally
    q = torch.randn(batch_size, s_q, num_q_heads, h_qkv, dtype=torch.bfloat16, device=device)
    # K shape: (batch, s_kv, num_kv_heads, h_qkv) = (1, 64, 4, 16)
    k = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16, device=device)

    print(f"\nTensor shapes (standard torch layout, no permutation):")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")

    if num_kv_heads < num_q_heads:
        k_reshaped = torch.zeros(batch_size, s_kv, num_q_heads, h_qkv, dtype=k.dtype, device=k.device)
        k_reshaped[:, :, :num_kv_heads, :] = k
        # The remaining extra heads are left as zeros
    elif num_kv_heads == num_q_heads:
        k_reshaped = k
    else:
        raise ValueError("num_kv_heads > num_q_heads not supported for zero padding logic.")

    print(f"K reshaped to {k_reshaped.shape}")
    input_tensor = {
        "q": q.reshape(batch_size, -1),
        "k": k_reshaped.reshape(batch_size, -1),
    }

    # Compute golden output using real online softmax algorithm
    # Online softmax iteratively computes:
    # - m_new = max(m_old, rowmax(S_j))
    # - P = exp(S_j - m_new)
    # - l = exp(m_old - m_new) * l + rowsum(P)
    Br = s_q  # Block row size
    Bc = s_kv  # Block column size (single tile for this test)
    Tc = s_kv // mlen  # Number of K tiles

    golden_p_list = []
    for kv_head in range(num_kv_heads):
        for q_head_offset in range(q_index_2_kv_index_ratio):
            q_head = kv_head * q_index_2_kv_index_ratio + q_head_offset
            q_i = q[0, :, q_head, :]  # [Br, h_qkv] = [64, 16]

            # Initialize online softmax state
            m = torch.full((Br,), float("-inf"), device=q.device)
            l = torch.zeros((Br,), device=q.device)

            # Iterate over K tiles (Tc=1 for this test, but general structure)
            for j in range(Tc):
                k_j = k[0, j * Bc : (j + 1) * Bc, kv_head, :]  # [Bc, h_qkv]
                s_j = q_i @ k_j.transpose(0, 1) * qk_scale  # Q @ Kj^T, [Br, Bc]

                rowmax_s_j = s_j.max(dim=1).values  # [Br]
                m_new = torch.maximum(m, rowmax_s_j)  # [Br]
                print(f"m_new: {m_new}")
                # Subtract m_new from each row of s_j
                s_j_shifted = s_j - m_new.unsqueeze(1)  # [Br, Bc]
                p = torch.exp(s_j_shifted)  # exp(Sj - m_new), [Br, Bc]
                p = p.to(torch.bfloat16)

                m_res = m - m_new  # [Br]
                m = m_new
                l_scale = torch.exp(m_res)  # [Br]
                print(f"l_scale: {l_scale}")
                p_row_sum = p.sum(dim=1)  # [Br]
                l = l_scale * l + p_row_sum  # [Br]

            golden_p_list.append(p)
            print(f"  Q head {q_head} x K head {kv_head} -> P shape: {p.shape}")

    # Stack all P results
    golden_p_all = torch.stack(golden_p_list, dim=0)  # (4, 64, 64)
    print(f"\nGolden P all heads shape: {golden_p_all.shape}")
    print(f"Golden P sample (head 0, first 4x4):\n{golden_p_all[0, :4, :4]}")

    # Memory layout in VSRAM
    # Q is stored in standard torch layout: [s_q, num_q_heads, h_qkv] = [64, 16, 16]
    # Each token has num_q_heads * h_qkv = 256 elements
    # qkt_multiply uses q_base_address + kv_head * q_per_kv * d + q_head_index * d internally
    q_base_address = 0
    q_total_size = s_q * num_q_heads * h_qkv  # 64 * 16 * 16 = 16384
    s_base_address = q_base_address + q_total_size
    fp_sram_start = 3  # After 0=zero, 1=qk_scale, 2=-inf

    print(f"\nVSRAM Layout:")
    print(f"  Q Base: {q_base_address}")
    print(f"  Q Total Size: {q_total_size}")
    print(f"  Q layout: [{s_q}, {num_q_heads}, {h_qkv}]")
    print(f"  S Base (QKT result): {s_base_address}")
    print(f"  S Size per head: {mlen * mlen}")
    print(f"  S Total for test (4 heads): {mlen * mlen * q_index_2_kv_index_ratio}")
    print(f"  FP SRAM start: {fp_sram_start}")

    # HBM layout
    q_hbm_size = int(s_q * num_q_heads * h_qkv * batch_size * real_data_ratio)
    k_hbm_offset = q_hbm_size

    print(f"\nHBM Layout:")
    print(f"  Q: 0 - {q_hbm_size}")
    print(f"  K: {k_hbm_offset}")

    # Generate assembly
    gen_assembly_code = "; Multi-Head Online Softmax Test Generation \n"

    # Set K addr offset register
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1, 2],
        addr_reg_val=[k_hbm_offset]
    )

    # Preload Q to VSRAM
    gen_assembly_code += preload_act_asm(
        vlen=mlen,
        preload_len=4,
        batch=batch_size * s_q,
        hidden_size=h_qkv * num_q_heads,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    # Set scale and stride for K prefetch
    gen_assembly_code += _reset_kv_prefetch(
        hkv=num_kv_heads,
        d=h_qkv,
        mlen=mlen,
        kv_len=s_kv,
        batch=batch_size,
        alive_registers_int=[1],
    )


    gen_assembly_code += reset_fpsram_code(
        alive_registers_int=[1, 2, 3, 4],
        alive_registers_fp=[1],
        reset_start_address=2,
        per_stride_dim=mlen,
        stride_dist = 3 * mlen,
        reset_amount = q_index_2_kv_index_ratio,
        reset_val_address=2,
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])
    gen_assembly_code += reset_fpreg_asm(alive_registers=[1])

    # Test with single KV head first to validate M_BTMM operation
    # M_BTMM processes 4 Q heads in parallel (MLEN//HLEN = 64//16 = 4)
    # Memory usage: Q(16384) + S(16384 for 4 heads) = 32768 < 65536 VSRAM capacity
    test_kv_head = 0  # Test first KV head only

    q_head_start = test_kv_head * q_index_2_kv_index_ratio
    q_head_end = q_head_start + q_index_2_kv_index_ratio - 1
    gen_assembly_code += f"; === KV head {test_kv_head} (Q heads {q_head_start}-{q_head_end}) ===\n"

    # Q base for this KV head group
    # Q layout is [s_q, num_q_heads, h_qkv] = [64, 16, 16]
    # qkt_multiply internally computes: q_base + kv_head * q_per_kv * d + q_head_index * d
    q_base_for_kv_head = q_base_address + test_kv_head * q_index_2_kv_index_ratio * h_qkv

    for kv_head_index in range(num_kv_heads):
        gen_assembly_code += qkt_multiply(
            d=h_qkv,
            mlen=mlen,
            alive_registers=[1, 2],
            q_base_address=q_base_address + kv_head_index * q_index_2_kv_index_ratio * h_qkv,
            k_base_hbm_offset_reg=1,
            q_head_index=kv_head_index * q_index_2_kv_index_ratio,
            k_head_index=kv_head_index,
            s_base_address=s_base_address + kv_head_index * mlen * mlen
        )
        gen_assembly_code += reset_reg_asm(alive_registers=[1, 2])

        # Apply online softmax for this head
        gen_assembly_code += online_softmax_code(
            mlen=mlen,
            alive_registers_int=[1, 2, 3, 4, 5],
            alive_registers_fp=[1, 2, 3, 4, 5],
            s_address=s_base_address + kv_head_index * mlen * mlen,
            m_start_address=fp_sram_start,
            qk_scale_address=1,
        )
        gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    # Golden result - only first 4 Q heads (KV head 0) for this test
    # M_BMM_WO writes in [heads, seq_q, seq_k] layout (confirmed from behavioral_simulator/src/main.rs):
    #   for j in 0..broadcast_amount (heads):
    #     for i in 0..mlen (seq_q):
    #       write tensor at vec_base + (j * mlen + i) * mlen
    # This means head 0 rows 0-63 come first, then head 1 rows 0-63, etc.
    # golden_p_all has shape [heads, seq_q, seq_k] = [16, 64, 64] - this MATCHES M_BMM_WO layout!
    num_test_heads = q_index_2_kv_index_ratio  # 4 heads
    golden_p_test = golden_p_all[:num_test_heads]  # [4, 64, 64] = [heads, seq_q, seq_k]

    # No transposition needed - golden layout matches M_BMM_WO output layout
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden_p_test.reshape(-1).unsqueeze(0)  # (1, 4*64*64)
    }

    print(f"\nGolden P test shape: {golden_p_test.shape}")
    print(f"Golden layout: [heads, seq_q, seq_k] = [4, 64, 64] (matches M_BMM_WO output)")
    print(f"Golden output flattened shape: {golden_p_test.reshape(-1).shape}")

    fp_preload = [0.0, qk_scale, -float("inf")]
    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order=["q", "k"], build_path=build_path)

    # Result is at s_base_address, shape (4, mlen, mlen) = (4, 64, 64) for 4 Q heads
    result_start_row = s_base_address // vlen
    num_result_rows = (mlen * mlen) // vlen

    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": 1,
        "elements_per_batch": mlen * mlen,
        "row_dim": vlen,
        "use_stride_mode": False
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print(f"\nResult location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("=" * 60)
