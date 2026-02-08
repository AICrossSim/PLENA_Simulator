"""
Flash Attention 实现 - 使用 PLENAProgram API

测试场景：
- 输入：Q, K, V (seq_len x head_dim)
- 输出：O = softmax(Q @ K^T / sqrt(d)) @ V

Flash Attention 算法（Online Softmax）：
for q_block in Q_blocks:
    init m_old = -inf, l_old = 0, O_row = 0
    for k_block in K_blocks:
        S = Q[q_block] @ K[k_block]^T
        S = S * scale
        Online Softmax: update m, l, P
        PV = P @ V[k_block]
        O_row = diag(m_res) * O_row + PV
        m_old, l_old = m_new, l_new
    O[q_block] = O_row / l_old
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import math
import json
from plena_program import PLENAProgram
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    print("=" * 80)
    print("Flash Attention 测试 - PLENAProgram API")
    print("=" * 80)

    # ========================================================================
    # 参数设置
    # ========================================================================
    seq_len = 128  # 2 * mlen
    head_dim = 128  # 2 * mlen
    mlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)

    num_q_blocks = seq_len // mlen  # 2
    num_k_blocks = seq_len // mlen  # 2
    scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    # ========================================================================
    # 创建测试数据
    # ========================================================================
    Q = torch.randn(seq_len, head_dim) * 0.5
    K = torch.randn(seq_len, head_dim) * 0.5
    V = torch.randn(seq_len, head_dim) * 0.5

    print(f"\n输入数据：")
    print(f"  Q: {Q.shape} ({num_q_blocks} 行子块 x {head_dim//mlen} 列子块)")
    print(f"  K: {K.shape} ({num_k_blocks} 行子块 x {head_dim//mlen} 列子块)")
    print(f"  V: {V.shape} ({num_k_blocks} 行子块 x {head_dim//mlen} 列子块)")
    print(f"  scale: {scale:.6f}")

    # ========================================================================
    # Golden 计算 (标准 Attention)
    # ========================================================================
    print("\n--- Golden Computation ---")
    
    golden_S = torch.matmul(Q.float(), K.float().T)  # (seq_len, seq_len)
    golden_S_scaled = golden_S * scale
    golden_P = torch.softmax(golden_S_scaled, dim=-1)
    golden_O = torch.matmul(golden_P, V.float())

    print(f"  Golden O: {golden_O.shape}")
    print(f"  O[0, :4]: {golden_O[0, :4].tolist()}")

    # ========================================================================
    # 使用 PLENAProgram 实现 Flash Attention
    # ========================================================================
    print("\n--- PLENAProgram Implementation ---")
    
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # 输入声明
    q_input = prog.input("Q", shape=(seq_len, head_dim))
    k_input = prog.input("K", shape=(seq_len, head_dim))
    v_input = prog.input("V", shape=(seq_len, head_dim))

    # 加载 Q 到 VRAM，注册为 VRAM sub-matrix
    Q_batch = prog.load_batch(q_input, name="Q")
    Q_sub = prog.register_vram_sub_matrix(Q_batch)

    # K, V 注册为 HBM sub-matrix（按需加载）
    K_sub = prog.register_sub_matrix(k_input)
    V_sub = prog.register_sub_matrix(v_input)

    # 分配 VRAM 矩阵
    S_block = prog.alloc("S", mlen, mlen)      # 当前 Q-K 块乘法结果
    PV = prog.alloc("PV", mlen, head_dim)      # P @ V 的临时结果
    O = prog.alloc("O", seq_len, head_dim)     # 最终输出

    print(f"  已分配 VRAM 矩阵：")
    print(f"    S_block: {mlen}x{mlen}")
    print(f"    PV: {mlen}x{head_dim}")
    print(f"    O: {seq_len}x{head_dim}")

    # ========================================================================
    # Flash Attention 主循环
    # ========================================================================
    print(f"\n--- Flash Attention 主循环 ({num_q_blocks} Q blocks x {num_k_blocks} K blocks) ---")

    for q_idx in range(num_q_blocks):
        print(f"\n  Q block {q_idx}:")
        
        # 初始化 Online Softmax 状态: m=-inf, l=0, O_row=0
        prog.init_online_softmax(q_idx, O)
        
        for k_idx in range(num_k_blocks):
            print(f"    K block {k_idx}: ", end="")
            
            # 重置 MRAM（清空之前加载的子块）
            prog.reset_mram()
            
            # 加载 K[k_idx][:] 到 MRAM
            K_sub.load_row(k_idx)
            
            # S = Q[q_idx][:] @ K[k_idx][:]^T
            prog.vram_sub_projection_T_to(
                Q_sub.row(q_idx),
                K_sub.row(k_idx),
                S_block,
                target_row_idx=0,
                target_col_idx=0
            )
            print("S=Q@K^T", end=" ")
            
            # Online Softmax: scale + softmax, 更新 m, l
            prog.online_softmax_block(S_block, scale)
            print("→ OnlineSoftmax", end=" ")
            
            # PV = P @ V[k_idx]
            prog.compute_pv(S_block, V_sub, k_idx, PV)
            print("→ PV=P@V", end=" ")
            
            # O[q_idx] = O[q_idx] * m_res (Online Softmax 校正)
            prog.scale_o_row(O, q_idx)
            
            # O[q_idx] += PV
            prog.vram_add(O, PV, dst_row_offset=q_idx * mlen)
            print("→ O+=PV")
        
        # 最终归一化: O[q_idx] = O_row / l
        prog.final_scale_o(q_idx, O)
        print(f"    Final scale O[{q_idx}] /= l")

    # ========================================================================
    # 生成 ISA 代码
    # ========================================================================
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\n生成 {len(lines)} 行 ISA 代码")

    # ========================================================================
    # 打印 Symbol Table
    # ========================================================================
    print("\n--- Symbol Table ---")
    prog.print_symbol_table()

    # ========================================================================
    # 创建仿真环境
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "Q": Q.reshape(1, -1),
        "K": K.reshape(1, -1),
        "V": V.reshape(1, -1),
    }

    golden_result = {
        "original_output": golden_O,
    }

    # FP SRAM 预加载: [0]=0.0, [1]=scale, [2]=-inf
    fp_preload = [0.0, scale, float('-inf')] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_plena",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir
    )

    # 保存 comparison params
    symbol_table = prog._compiler.symbol_table.table
    o_info = symbol_table[O.name]

    comparison_params = {
        "start_row_idx": o_info.vram_addr // mlen,
        "num_rows": (seq_len * head_dim) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": head_dim,
        "row_dim": mlen,
        "use_stride_mode": True
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Flash Attention - PLENAProgram 实现完成")
    print("=" * 80)
    print(f"✓ 算法: Online Softmax (避免存储完整 S 矩阵)")
    print(f"✓ Q blocks: {num_q_blocks} (每块 {mlen}x{head_dim})")
    print(f"✓ K blocks: {num_k_blocks} (每块 {mlen}x{head_dim})")
    print(f"✓ V blocks: {num_k_blocks} (每块 {mlen}x{head_dim})")
    print(f"✓ Scale factor: {scale:.6f}")
    print(f"✓ 生成 ISA 代码: {len(lines)} 行")
    print(f"✓ O 位置: VRAM row {o_info.vram_addr // mlen}")
    print("=" * 80)
    
    print(f"\n仿真环境已创建: {build_dir}")
    print(f"\n运行仿真:")
    print(f"  cd PLENA_Simulator && just behave_sim flash_attention_plena")

