"""
Flash Attention 展开版本测试

使用伪代码语法 + SimpleCompiler，手动设置 golden 和比较。
不使用 AutoCompilerHelper。

伪代码展开版本 (来自文档):
- for 循环遍历子块
- S = Q @ K^T (分块计算)
- Online Softmax
- O = P @ V (分块累加)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

import torch
import math
import json
from simple_compiler import SimpleCompiler
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    # ========================================================================
    # 测试参数
    # ========================================================================
    seq_len = 128  # 2 * mlen，测试 2x2 子块
    head_dim = 128  # 2 * mlen
    mlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    
    torch.manual_seed(42)
    
    # ========================================================================
    # 创建测试数据（数值稍大一些，便于调试）
    # ========================================================================
    Q = torch.randn(seq_len, head_dim) * 0.5
    K = torch.randn(seq_len, head_dim) * 0.5
    V = torch.randn(seq_len, head_dim) * 0.5
    
    print("=" * 60)
    print("Flash Attention Expanded Test (Pseudocode + SimpleCompiler)")
    print("=" * 60)
    print(f"Q: {Q.shape} -> {seq_len//mlen} 行子块 x {head_dim//mlen} 列子块")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    
    # ========================================================================
    # 手动计算 Golden Result: O = softmax(Q @ K^T / sqrt(d)) @ V
    # ========================================================================
    scale = 1.0 / math.sqrt(head_dim)
    golden_S = torch.matmul(Q.float(), K.float().T)  # (seq_len, seq_len)
    golden_S_scaled = golden_S * scale
    golden_P = torch.softmax(golden_S_scaled, dim=-1)
    golden_O = torch.matmul(golden_P, V.float())
    
    print(f"\nGolden S (Q @ K^T): {golden_S.shape}")
    print(f"Golden O (attention output): {golden_O.shape}")
    print(f"Golden O (first 5x5):\n{golden_O[:5, :5]}")
    
    # ========================================================================
    # 定义伪代码 - 展开版本
    # 
    # 由于完整的 Online Softmax 需要很多向量操作，这里先测试 S = Q @ K^T 部分
    # ========================================================================
    
    # 先定义 HBM 地址（手动分配）
    q_hbm_size = int(seq_len * head_dim * real_data_ratio)
    k_hbm_size = int(seq_len * head_dim * real_data_ratio)
    v_hbm_size = int(seq_len * head_dim * real_data_ratio)
    
    q_hbm_addr = 0
    k_hbm_addr = q_hbm_addr + q_hbm_size
    v_hbm_addr = k_hbm_addr + k_hbm_size
    
    # ========================================================================
    # 完整 Flash Attention 伪代码
    # 
    # O = softmax(Q @ K^T / sqrt(d)) @ V
    # 
    # 分块计算，使用 Online Softmax 避免存储完整的 S 矩阵
    # ========================================================================
    num_q_blocks = seq_len // mlen  # 2
    num_k_blocks = seq_len // mlen  # 2
    num_v_col_blocks = head_dim // mlen  # 2 (V 的列块数)
    scale_factor = 1.0 / math.sqrt(head_dim)
    
    # ========================================================================
    # 展开版本 Flash Attention 伪代码 - 完整 Online Softmax
    # 
    # Flash Attention 核心算法：
    # for q_block in Q_blocks:
    #     init m_old = -inf, l = 0, O_row = 0
    #     for k_block in K_blocks:
    #         S = Q[q_block] @ K[k_block]^T    # (mlen, mlen)
    #         S = S * scale
    #         Online Softmax: P, m_new, l_new = softmax_online(S, m_old, l)
    #         PV = P @ V[k_block]              # (mlen, head_dim)
    #         O_row = diag(m_res) * O_row + PV
    #         m_old, l = m_new, l_new
    #     O[q_block] = O_row / l
    # ========================================================================
    
    code = f"""
    # ========== Load tensors ==========
    Q_batch = Load_Batch(Q)
    Q_sub = Register_VRAMSubMatrix(Q_batch)
    
    K_sub = Register_SubMatrix(K)
    V_sub = Register_SubMatrix(V)
    
    # ========== Allocate outputs ==========
    # S: 当前 Q-K 块乘法结果 (mlen x mlen)
    S_block = Allocate_VRAMMatrix({mlen}, {mlen})
    
    # PV: P @ V 的临时结果 (mlen x head_dim)
    PV = Allocate_VRAMMatrix({mlen}, {head_dim})
    
    # O: 最终输出 (seq_len x head_dim)
    O = Allocate_VRAMMatrix({seq_len}, {head_dim})
    
    # ========== Flash Attention Main Loop ==========
    for q_idx in range(0, {num_q_blocks}, 1):
        # 初始化: m=-inf, l=0, O_row=0
        Init_Online_Softmax(q_idx, O)
        
        for k_idx in range(0, {num_k_blocks}, 1):
            Reset_MRAM()
            
            # S = Q[q_idx] @ K[k_idx]^T
            Load_SubMatrix_Row(K_sub, k_idx)
            S_block[0][0] = VRAMSubMatrix(Q_sub, q_idx) @T SubMatrix(K_sub, k_idx)
            
            # Online Softmax: scale + softmax, 更新 m, l
            OnlineSoftmax_Block(S_block, {scale_factor})
            
            # PV = P @ V[k_idx]
            Load_SubMatrix_Row(V_sub, k_idx)
            Compute_PV(S_block, V_sub, k_idx, PV)
            
            # O[q_idx] = O[q_idx] * m_res (Online Softmax 特有)
            Scale_O_Row(O, q_idx)
            
            # O[q_idx] += PV (通用矩阵加法，行偏移 = q_idx * mlen)
            VRAM_Add(O, PV, q_idx * {mlen})
        endfor
        
        # O[q_idx] = O_row / l
        Final_Scale_O(q_idx, O)
    endfor
    
    Result O
    """
    
    # ========================================================================
    # 使用 SimpleCompiler 解析伪代码
    # ========================================================================
    print("\n" + "=" * 60)
    print("Parsing Pseudocode with SimpleCompiler...")
    print("=" * 60)
    
    # 创建配置文件（手动）
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    
    config_content = f"""# Input tensors
Q: hbm_addr={q_hbm_addr}, hbm_size={q_hbm_size}, shape=({seq_len}, {head_dim})
K: hbm_addr={k_hbm_addr}, hbm_size={k_hbm_size}, shape=({seq_len}, {head_dim})
V: hbm_addr={v_hbm_addr}, hbm_size={v_hbm_size}, shape=({seq_len}, {head_dim})

# Code
{code}
"""
    
    config_file = build_dir / "flash_attention_expand_config.txt"
    with open(config_file, "w") as f:
        f.write(config_content)
    
    # 使用 SimpleCompiler
    compiler = SimpleCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)
    gen_code = compiler.parse_file(str(config_file))
    
    # ========================================================================
    # 打印 Symbol Table
    # ========================================================================
    print("\n" + "=" * 60)
    print("Symbol Table:")
    print("=" * 60)
    compiler.compiler.print_symbol_table()
    
    # ========================================================================
    # 设置仿真环境
    # ========================================================================
    
    # 准备 input tensors
    q_flat = Q.reshape(1, -1)
    k_flat = K.reshape(1, -1)
    v_flat = V.reshape(1, -1)
    
    input_tensor = {
        "Q": q_flat,
        "K": k_flat,
        "V": v_flat,
    }
    
    # Golden result - 完整 Flash Attention 输出 O
    golden_result = {
        "original_output": golden_O  # (seq_len, head_dim)
    }
    
    # FP preload: [0]=0.0, [1]=scale_factor, [2]=-inf
    fp_preload = [0.0, scale_factor, float('-inf')]
    
    # 创建仿真环境
    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))
    
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_expand",
        data=None,
        specified_data_order=["Q", "K", "V"]
    )
    
    # 保存 comparison params
    o_info = compiler.compiler.symbol_table["O"]
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
    
    # 保存代码副本
    with open(build_dir / "generated_asm_code_before_run.asm", "w") as f:
        f.write(gen_code)
    
    # ========================================================================
    # 打印生成的 ISA 代码
    # ========================================================================
    print("\n" + "=" * 60)
    print("Generated ISA Code (first 100 lines):")
    print("=" * 60)
    lines = gen_code.splitlines()
    for i, line in enumerate(lines[:100]):
        print(f"{i+1:4d}: {line}")
    if len(lines) > 100:
        print(f"... ({len(lines) - 100} more lines)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("Flash Attention Expanded Test - Complete with Online Softmax")
    print("=" * 60)
    print(f"✓ Pseudocode parsed successfully")
    print(f"✓ Q: ({seq_len}, {head_dim}) -> Load_Batch")
    print(f"✓ K: ({seq_len}, {head_dim}) -> Load_Matrix (HBM)")
    print(f"✓ V: ({seq_len}, {head_dim}) -> Load_Matrix (HBM)")
    print(f"✓ O = FlashAttention(Q, K, V)")
    print(f"✓ Algorithm (Online Softmax):")
    print(f"    for q_block in [0, {num_q_blocks}):")
    print(f"        init m=-inf, l=0, O_row=0")
    print(f"        for k_block in [0, {num_k_blocks}):")
    print(f"            S = Q[q_block] @ K[k_block]^T")
    print(f"            S = S * {scale_factor:.6f}")
    print(f"            P = OnlineSoftmax(S, m, l)")
    print(f"            PV = P @ V[k_block]")
    print(f"            O_row = diag(m_res)*O_row + PV")
    print(f"        O[q_block] = O_row / l")
    print(f"✓ Generated {len(lines)} lines of ISA code")
    print(f"✓ Golden O shape: {golden_O.shape}")
    print("=" * 60)
    
    print("\n下一步:")
    print("  1. cargo run          # 运行仿真")
    print("  2. python view_mem.py # 查看比较结果")
