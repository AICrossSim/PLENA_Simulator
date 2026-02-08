#!/usr/bin/env python3
"""
分段测试 1: 只测试 QK multiply (S = Q @ K^T)
不包含 Softmax，不包含 PV，只测试矩阵乘法是否正确
"""

import sys
import os
import torch
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "compiler", "asm_templates"))

from developer_compiler import DeveloperCompiler
from symbol_table import SymbolTable

def test_qk_multiply():
    """测试 QK multiply: S = Q @ K^T"""
    
    # 参数设置
    mlen = 64
    blen = 4
    head_dim = 64
    
    # 测试两种情况
    # 1. seq_len = 64 (单个 K block)
    # 2. seq_len = 128 (两个 K blocks，需要分别测试)
    
    for seq_len in [64, 128]:
        print(f"\n{'='*60}")
        print(f"Testing QK multiply with seq_len = {seq_len}")
        print(f"{'='*60}")
        
        # 创建测试数据
        torch.manual_seed(42)
        Q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        K = torch.randn(seq_len, head_dim, dtype=torch.float32)
        
        # 计算 golden
        S_golden = torch.matmul(Q, K.T)  # (seq_len, seq_len)
        
        print(f"Q shape: {Q.shape}")
        print(f"K shape: {K.shape}")
        print(f"S_golden shape: {S_golden.shape}")
        
        # 分块测试
        q_blocks = (seq_len + mlen - 1) // mlen
        k_blocks = (seq_len + mlen - 1) // mlen
        
        print(f"q_blocks = {q_blocks}, k_blocks = {k_blocks}")
        
        for q_block_idx in range(q_blocks):
            for k_block_idx in range(k_blocks):
                # 提取 Q 和 K 的块
                q_start = q_block_idx * mlen
                q_end = min(q_start + mlen, seq_len)
                k_start = k_block_idx * mlen
                k_end = min(k_start + mlen, seq_len)
                
                Q_block = Q[q_start:q_end, :]  # (mlen, head_dim)
                K_block = K[k_start:k_end, :]  # (mlen, head_dim)
                
                # 计算 S 块的 golden
                S_block_golden = torch.matmul(Q_block, K_block.T)  # (mlen, mlen)
                
                print(f"\n--- Q block {q_block_idx}, K block {k_block_idx} ---")
                print(f"Q_block: [{q_start}:{q_end}, :], K_block: [{k_start}:{k_end}, :]")
                print(f"S_block_golden shape: {S_block_golden.shape}")
                print(f"S_block_golden[0, :5]: {S_block_golden[0, :5].tolist()}")
                print(f"S_block_golden[:5, 0]: {S_block_golden[:5, 0].tolist()}")
                print(f"S_block_golden range: [{S_block_golden.min():.4f}, {S_block_golden.max():.4f}]")
        
        # 检查完整 S 矩阵
        print(f"\nFull S_golden:")
        print(f"S_golden[0, :10]: {S_golden[0, :10].tolist()}")
        print(f"S_golden[64, :10] (if seq_len > 64): {S_golden[64, :10].tolist() if seq_len > 64 else 'N/A'}")
        print(f"S_golden range: [{S_golden.min():.4f}, {S_golden.max():.4f}]")

def test_qk_with_simulator():
    """使用仿真器测试 QK multiply"""
    
    from auto_compiler_helper import AutoCompilerHelper
    
    mlen = 64
    blen = 4
    head_dim = 64
    seq_len = 64  # 先测试单 block
    
    print(f"\n{'='*60}")
    print(f"Testing QK multiply with simulator (seq_len = {seq_len})")
    print(f"{'='*60}")
    
    # 创建测试数据
    torch.manual_seed(42)
    Q = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    K = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    
    # 创建伪代码 - 只测试 QK multiply
    # 需要自定义 ISA 来只执行 QK 部分
    pseudo_code = f"""
Batch Q[{seq_len}, {head_dim}] fp16
Matrix K[{seq_len}, {head_dim}] fp16
QK_only S = Q @ K^T
Result S
"""
    
    # 由于 QK_only 不是标准指令，我们需要手动生成代码
    # 这里先打印出需要的配置
    
    print(f"Q: {Q.shape}, dtype={Q.dtype}")
    print(f"K: {K.shape}, dtype={K.dtype}")
    
    # Golden
    Q_f32 = Q.float()
    K_f32 = K.float()
    S_golden = torch.matmul(Q_f32, K_f32.T)
    
    print(f"\nGolden S[0, :10]: {S_golden[0, :10].tolist()}")
    print(f"Golden S range: [{S_golden.min():.4f}, {S_golden.max():.4f}]")
    
    # 检查 K 的 HBM 地址计算
    # 对于 seq_len=128, k_block_idx=1:
    # k_hbm_offset = 1 * 64 * 64 = 4096 elements
    # 如果每个元素 2 bytes (fp16/bf16)，则 byte 偏移 = 4096 * 2 = 8192 bytes
    print(f"\n--- HBM 偏移检查 ---")
    print(f"seq_len=128, k_block_idx=1:")
    print(f"  k_hbm_offset (elements) = {1 * 64 * 64}")
    print(f"  k_hbm_offset (bytes, fp16) = {1 * 64 * 64 * 2}")

if __name__ == "__main__":
    test_qk_multiply()
    # test_qk_with_simulator()  # 需要进一步适配

