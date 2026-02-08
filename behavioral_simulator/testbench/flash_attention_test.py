"""
Flash Attention Test: 使用高层伪代码测试 Flash Attention

测试场景：
- Q, K, V: (seq_len, head_dim) 的矩阵
- O = softmax(Q @ K^T / sqrt(d)) @ V

伪代码语法（展开版本）：
```
Q = Load_Batch(q)
K = Load_Matrix(k)
V = Load_Matrix(v)

# 分块计算
for q_block in range(0, seq_len, mlen):
    # 初始化 online softmax 状态
    m_old = Init(-inf, mlen)
    l_old = Init(0, mlen)
    O_block = Init(0, mlen, head_dim)
    
    for kv_block in range(0, seq_len, mlen):
        # 加载 K, V 子块
        Load_SubMatrix_Row(K_sub, kv_block)
        Load_SubMatrix_Row(V_sub, kv_block)
        
        # S = Q[q_block][:] @ K[kv_block][:]^T
        S = VRAMSubMatrix(Q_sub, q_block) @T SubMatrix(K_sub, kv_block)
        
        # Online Softmax
        S = Scale(S, 1/sqrt(head_dim))
        m_cur = RowMax(S)
        m_new = Max(m_old, m_cur)
        m_res = Exp(m_old - m_new)
        P = Exp(S - m_new)
        l_new = l_old * m_res + RowSum(P)
        
        # P @ V
        PV = P @ SubMatrix(V_sub, kv_block)
        
        # Update O
        O_block = m_res * O_block + PV
        
        m_old = m_new
        l_old = l_new
    endfor
    
    # Final scaling
    O_block = O_block / l_old
endfor

Result O
```

简化版本（推荐）：
```
Q = Load_Batch(q)
K = Load_Matrix(k)
V = Load_Matrix(v)
O = FlashAttention(Q, K, V)
Result O
```
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
import math
from auto_compiler_helper import AutoCompilerHelper


def compute_flash_attention_golden(Q, K, V, head_dim=None):
    """
    计算 Flash Attention 的 golden result
    
    O = softmax(Q @ K^T / sqrt(d)) @ V
    """
    if head_dim is None:
        head_dim = Q.shape[-1]
    
    scale = 1.0 / math.sqrt(head_dim)
    
    # S = Q @ K^T / sqrt(d)
    S = scale * torch.matmul(Q.float(), K.float().T)
    
    # P = softmax(S)
    P = torch.softmax(S, dim=-1)
    
    # O = P @ V
    O = torch.matmul(P, V.float())
    
    return O


if __name__ == "__main__":
    # ========================================================================
    # 测试参数
    # ========================================================================
    # 注意：seq_len 和 head_dim 都必须是 64 的倍数
    seq_len = 64  # 序列长度（也是 batch size）
    head_dim = 64  # Head dimension
    mlen = 64  # 子块大小
    blen = 4   # batch 分块大小
    real_data_ratio = (8*8 + 8) / (8 * 8)  # 1.125
    
    torch.manual_seed(42)
    
    # ========================================================================
    # 创建测试数据
    # ========================================================================
    # Q: (seq_len, head_dim) - 将被加载到 VRAM
    Q = torch.randn(seq_len, head_dim) * 0.1  # 小值避免 overflow
    
    # K: (seq_len, head_dim) - 在 HBM
    K = torch.randn(seq_len, head_dim) * 0.1
    
    # V: (seq_len, head_dim) - 在 HBM
    V = torch.randn(seq_len, head_dim) * 0.1
    
    # ========================================================================
    # 计算 golden result
    # ========================================================================
    golden_O = compute_flash_attention_golden(Q, K, V, head_dim)
    print("=" * 60)
    print("Flash Attention Test")
    print("=" * 60)
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Golden O shape: {golden_O.shape}")
    print(f"Golden O (first 5x5):\n{golden_O[:5, :5]}")
    
    # ========================================================================
    # 定义伪代码（简化版本）
    # ========================================================================
    code = """
    Q_batch = Load_Batch(Q)
    K_mat = Load_Matrix(K)
    V_mat = Load_Matrix(V)
    O = FlashAttention(Q_batch, K_mat, V_mat)
    Result O
    """
    
    # ========================================================================
    # 使用 AutoCompilerHelper 自动处理
    # ========================================================================
    helper = AutoCompilerHelper(real_data_ratio=real_data_ratio, mlen=mlen, blen=blen)
    
    # 添加 tensors
    helper.add_tensor("Q", Q, is_batch=True)
    helper.add_tensor("K", K, is_batch=False)
    helper.add_tensor("V", V, is_batch=False)
    
    print("\nTensor Layout (auto-allocated):")
    for name in ["Q", "K", "V"]:
        h, w = helper.tensor_shapes[name]
        hbm_addr = helper.hbm_layout[name]
        hbm_size = helper.hbm_sizes[name]
        print(f"  {name}: shape=({h}, {w}), hbm_addr={hbm_addr}, hbm_size={hbm_size}")
    
    # 编译并自动设置环境
    result = helper.compile_and_setup(
        code, 
        asm_name="flash_attention_test",
        fp_preload=[0.0, 1e-6, float('-inf')]  # 0=zero, 1=epsilon, 2=-inf for softmax
    )
    
    compiler = result["compiler"]
    gen_assembly_code = result["generated_code"]
    symbol_table = result["symbol_table"]
    golden_result = result["golden_result"]
    build_dir = result["build_dir"]
    
    # ========================================================================
    # 打印结果
    # ========================================================================
    print("\n" + "=" * 60)
    print("Symbol Table:")
    print("=" * 60)
    compiler.print_symbol_table()
    
    # 打印 golden result 比较
    print("\n" + "=" * 60)
    print("Golden Result Comparison:")
    print("=" * 60)
    print(f"AutoCompilerHelper golden shape: {golden_result['original_output'].shape}")
    print(f"AutoCompilerHelper golden (first 5x5):\n{golden_result['original_output'][:5, :5]}")
    
    # 比较两个 golden
    if torch.allclose(golden_result['original_output'], golden_O, rtol=1e-4, atol=1e-4):
        print("\n✓ Golden results match!")
    else:
        print("\n✗ Golden results DO NOT match!")
        diff = (golden_result['original_output'] - golden_O).abs().max()
        print(f"  Max difference: {diff}")
    
    # 检查生成的代码
    print(f"\n✓ Generated code saved to: {build_dir / 'generated_asm_code_before_run.asm'}")
    print(f"  Total lines: {len(gen_assembly_code.splitlines())}")
    
    # ========================================================================
    # 打印生成的 ISA 代码片段（前 100 行）
    # ========================================================================
    print("\n" + "=" * 60)
    print("Generated ISA Code (first 100 lines):")
    print("=" * 60)
    lines = gen_assembly_code.splitlines()
    for i, line in enumerate(lines[:100]):
        print(f"{i+1:4d}: {line}")
    if len(lines) > 100:
        print(f"... ({len(lines) - 100} more lines)")
    
    # ========================================================================
    # 打印摘要
    # ========================================================================
    print("\n" + "=" * 60)
    print("Flash Attention Test Summary")
    print("=" * 60)
    print(f"✓ Q: ({seq_len}, {head_dim}) loaded to VRAM")
    print(f"✓ K: ({seq_len}, {head_dim}) declared in HBM")
    print(f"✓ V: ({seq_len}, {head_dim}) declared in HBM")
    print(f"✓ O = FlashAttention(Q, K, V)")
    print(f"✓ Output shape: {golden_result['original_output'].shape}")
    print("=" * 60)
