#!/usr/bin/env python3
"""
Flash Attention 调试测试
1. seq_len=64 (单 block) - 应该工作
2. seq_len=128 (2 blocks) - 分析问题

主要检查点：
1. QK multiply 的 S 矩阵是否正确
2. Online Softmax 的 m_old, m_res, l 是否正确
3. PV multiply 的结果是否正确
4. O 的累加是否正确
"""

import sys
import os
import torch
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "compiler", "asm_templates"))


def analyze_online_softmax():
    """分析 Online Softmax 的中间结果"""
    print("="*60)
    print("Online Softmax 分析")
    print("="*60)
    
    mlen = 64
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)  # 0.125
    
    # 测试 seq_len = 128
    seq_len = 128
    
    torch.manual_seed(42)
    Q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    K = torch.randn(seq_len, head_dim, dtype=torch.float32)
    V = torch.randn(seq_len, head_dim, dtype=torch.float32)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # 分块计算
    q_blocks = (seq_len + mlen - 1) // mlen  # 2
    k_blocks = (seq_len + mlen - 1) // mlen  # 2
    
    print(f"q_blocks = {q_blocks}, k_blocks = {k_blocks}")
    
    # 对于 Q block 0，模拟 Flash Attention
    q_block_idx = 0
    Q_block = Q[q_block_idx * mlen : (q_block_idx + 1) * mlen, :]  # (64, 64)
    
    print(f"\n=== Q Block {q_block_idx} ===")
    print(f"Q_block shape: {Q_block.shape}")
    
    # 初始化
    m_old = torch.full((mlen,), float('-inf'))
    l_old = torch.zeros(mlen)
    O = torch.zeros(mlen, head_dim)
    
    for k_block_idx in range(k_blocks):
        K_block = K[k_block_idx * mlen : (k_block_idx + 1) * mlen, :]  # (64, 64)
        V_block = V[k_block_idx * mlen : (k_block_idx + 1) * mlen, :]  # (64, 64)
        
        print(f"\n--- K/V Block {k_block_idx} ---")
        print(f"K_block shape: {K_block.shape}")
        print(f"V_block shape: {V_block.shape}")
        
        # 1. S = Q @ K^T
        S = torch.matmul(Q_block, K_block.T)  # (64, 64)
        print(f"S shape: {S.shape}")
        print(f"S[0, :5]: {S[0, :5].tolist()}")
        print(f"S range: [{S.min():.4f}, {S.max():.4f}]")
        
        # 2. S *= scale
        S = S * scale
        print(f"S after scale: [{S.min():.4f}, {S.max():.4f}]")
        
        # 3. Online Softmax
        row_max = S.max(dim=1).values  # (64,)
        print(f"row_max[:5]: {row_max[:5].tolist()}")
        
        # m_curr = max(row_max, m_old)
        m_curr = torch.maximum(row_max, m_old)
        print(f"m_old[:5] (before): {m_old[:5].tolist()}")
        print(f"m_curr[:5]: {m_curr[:5].tolist()}")
        
        # m_res = exp(m_old - m_curr)
        m_res = torch.exp(m_old - m_curr)
        print(f"m_res[:5]: {m_res[:5].tolist()}")
        
        # 检查 m_res 是否有异常值
        if torch.any(torch.isnan(m_res)) or torch.any(torch.isinf(m_res)):
            print(f"⚠️ WARNING: m_res 有异常值!")
            print(f"  NaN count: {torch.isnan(m_res).sum()}")
            print(f"  Inf count: {torch.isinf(m_res).sum()}")
        
        # S' = S - m_curr
        S_prime = S - m_curr.unsqueeze(1)
        print(f"S_prime range: [{S_prime.min():.4f}, {S_prime.max():.4f}]")
        
        # P = exp(S')
        P = torch.exp(S_prime)
        print(f"P range: [{P.min():.4f}, {P.max():.4f}]")
        
        if torch.any(torch.isnan(P)) or torch.any(torch.isinf(P)):
            print(f"⚠️ WARNING: P 有异常值!")
        
        # l_new = l_old * m_res + sum(P)
        l_new = l_old * m_res + P.sum(dim=1)
        print(f"l_new[:5]: {l_new[:5].tolist()}")
        
        if torch.any(torch.isnan(l_new)) or torch.any(torch.isinf(l_new)):
            print(f"⚠️ WARNING: l_new 有异常值!")
        
        # PV = P @ V
        PV = torch.matmul(P, V_block)  # (64, 64)
        print(f"PV range: [{PV.min():.4f}, {PV.max():.4f}]")
        
        if torch.any(torch.isnan(PV)) or torch.any(torch.isinf(PV)):
            print(f"⚠️ WARNING: PV 有异常值!")
        
        # O = diag(m_res) * O_old + PV
        O = m_res.unsqueeze(1) * O + PV
        print(f"O range: [{O.min():.4f}, {O.max():.4f}]")
        
        if torch.any(torch.isnan(O)) or torch.any(torch.isinf(O)):
            print(f"⚠️ WARNING: O 有异常值!")
        
        # 更新状态
        m_old = m_curr
        l_old = l_new
    
    # 最终缩放
    O = O / l_old.unsqueeze(1)
    print(f"\n=== Final O ===")
    print(f"O shape: {O.shape}")
    print(f"O range: [{O.min():.4f}, {O.max():.4f}]")
    print(f"O[0, :5]: {O[0, :5].tolist()}")
    
    if torch.any(torch.isnan(O)) or torch.any(torch.isinf(O)):
        print(f"⚠️ WARNING: Final O 有异常值!")
    
    # 计算标准 attention 作为参考
    print("\n=== 标准 Attention 参考 ===")
    S_full = torch.matmul(Q, K.T) * scale
    P_full = torch.softmax(S_full, dim=-1)
    O_full = torch.matmul(P_full, V)
    
    # 只比较 Q block 0 的部分
    O_ref = O_full[q_block_idx * mlen : (q_block_idx + 1) * mlen, :]
    
    print(f"O_ref shape: {O_ref.shape}")
    print(f"O_ref range: [{O_ref.min():.4f}, {O_ref.max():.4f}]")
    print(f"O_ref[0, :5]: {O_ref[0, :5].tolist()}")
    
    # 比较
    diff = torch.abs(O - O_ref)
    print(f"\n=== 差异分析 ===")
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    if diff.max() < 0.01:
        print("✓ Flash Attention 计算正确")
    else:
        print("✗ Flash Attention 计算有误差")


def check_hardware_constraints():
    """检查硬件约束"""
    print("\n" + "="*60)
    print("硬件约束检查")
    print("="*60)
    
    mlen = 64
    blen = 4
    head_dim = 64
    
    # M_TMM 约束
    print("\n--- M_TMM 约束 ---")
    print(f"每次操作: ({blen}, {mlen}) @ ({blen}, {mlen})^T = ({blen}, {blen})")
    print(f"需要的 tile 数: {(mlen // blen) * (mlen // blen)} = {(mlen // blen) ** 2}")
    
    # K MSRAM 偏移
    print("\n--- K MSRAM 偏移 ---")
    for k_row in range(mlen // blen):
        k_msram_offset = k_row * blen * mlen
        print(f"k_row={k_row}: offset={k_msram_offset} (必须 < {mlen * mlen} = 4096)")
    
    # M_MM 约束
    print("\n--- M_MM 约束 (for PV) ---")
    print(f"mat_offset 必须 < mlen = {mlen}")
    for v_col in range(mlen // blen):
        v_msram_offset = v_col * blen
        status = "✓" if v_msram_offset < mlen else "✗"
        print(f"v_col={v_col}: offset={v_msram_offset} {status}")


def check_hbm_offset():
    """检查 HBM 偏移计算"""
    print("\n" + "="*60)
    print("HBM 偏移检查")
    print("="*60)
    
    mlen = 64
    head_dim = 64
    seq_len = 128
    
    k_blocks = (seq_len + mlen - 1) // mlen
    
    print(f"seq_len = {seq_len}")
    print(f"mlen = {mlen}")
    print(f"head_dim = {head_dim}")
    print(f"k_blocks = {k_blocks}")
    
    for k_block_idx in range(k_blocks):
        # 元素偏移
        k_elem_offset = k_block_idx * mlen * head_dim
        # 字节偏移 (fp16 = 2 bytes)
        k_byte_offset = k_elem_offset * 2
        
        print(f"\nK block {k_block_idx}:")
        print(f"  Element offset: {k_elem_offset}")
        print(f"  Byte offset (fp16): {k_byte_offset}")
        print(f"  K rows: [{k_block_idx * mlen} : {(k_block_idx + 1) * mlen}]")
    
    # 检查 developer_compiler 中的计算
    print("\n--- developer_compiler 中的计算 ---")
    print("k_hbm_offset = k_block_idx * mlen * head_dim")
    print("这是 **元素偏移**，需要检查 H_PREFETCH_M 是否正确处理")


if __name__ == "__main__":
    analyze_online_softmax()
    check_hardware_constraints()
    check_hbm_offset()

