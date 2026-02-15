"""
Attention QKV + QK^T Test using Auto Compiler Helper
使用 AutoCompilerHelper 自动处理 HBM 地址、环境搭建和 golden value 计算

流程：
1. 计算 Q = X @ W_Q, K = X @ W_K, V = X @ W_V
2. Store K to HBM
3. Load K as Matrix (会自动转置)
4. 计算 S = Q @ K^T (使用 TMM_Matmul)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

import torch
import json
from auto_compiler_helper import AutoCompilerHelper


if __name__ == "__main__":
    # Attention QKV Projection Parameters
    # batch_size 需要 >= 64
    batch_size = 64  # B*T
    d = 64  # embedding dimension (input features)
    d_k = 128  # key/query dimension (output features)
    d_v = 128  # value dimension
    real_data_ratio = (8*8 + 8) / (8 * 8)  # 1.125

    torch.manual_seed(42)
    
    # ========== 用户只需要提供这些 ==========
    # Input: X : (batch_size, d)
    X = torch.randn(batch_size, d)
    
    # Parameters: W_Q, W_K, W_V
    # 注意：projection 使用的是 X @ W，所以 W 的形状是 (d, d_k)
    W_Q = torch.randn(d, d_k)  # (d, d_k)
    W_K = torch.randn(d, d_k)  # (d, d_k)
    W_V = torch.randn(d, d_v)  # (d, d_v)
    
    # 定义伪代码：
    # 1. 计算 Q, K, V
    # 2. Store K to HBM
    # 3. Load K as Matrix (转置效果)
    # 4. 计算 S = Q @ K^T
    code = """
    X_batch = Load_Batch(X)
    W_Q_mat = Load_Matrix(W_Q)
    W_K_mat = Load_Matrix(W_K)
    W_V_mat = Load_Matrix(W_V)
    Q = X_batch @ W_Q_mat
    K = X_batch @ W_K_mat
    V = X_batch @ W_V_mat
    Store(K, k_stored)
    K_mat = Load_Matrix(k_stored)
    S = Q @T K_mat
    Result S
    """
    # ======================================
    
    # 使用 AutoCompilerHelper 自动处理一切
    print("="*60)
    print("Attention QKV + QK^T Test")
    print("="*60)
    print(f"Input X: shape {X.shape}")
    print(f"Weights: W_Q {W_Q.shape}, W_K {W_K.shape}, W_V {W_V.shape}")
    print(f"Expected Q shape: ({batch_size}, {d_k})")
    print(f"Expected K shape: ({batch_size}, {d_k})")
    print(f"Expected S = Q @ K^T shape: ({batch_size}, {batch_size})")
    
    helper = AutoCompilerHelper(real_data_ratio=real_data_ratio, mlen=64, blen=4)
    
    # 添加 tensors（会自动分配 HBM 地址）
    helper.add_tensor("X", X, is_batch=True)
    helper.add_tensor("W_Q", W_Q, is_batch=False)
    helper.add_tensor("W_K", W_K, is_batch=False)
    helper.add_tensor("W_V", W_V, is_batch=False)
    
    print("\nTensor Layout (auto-allocated):")
    for name in ["X", "W_Q", "W_K", "W_V"]:
        h, w = helper.tensor_shapes[name]
        hbm_addr = helper.hbm_layout[name]
        hbm_size = helper.hbm_sizes[name]
        print(f"  {name}: shape=({h}, {w}), hbm_addr={hbm_addr}, hbm_size={hbm_size}")
    
    # 编译并自动设置环境
    result = helper.compile_and_setup(code, asm_name="attention_qkt")
    
    compiler = result["compiler"]
    gen_assembly_code = result["generated_code"]
    symbol_table = result["symbol_table"]
    golden_result = result["golden_result"]
    build_dir = result["build_dir"]
    
    # 打印符号表状态
    print("\nSymbol Table:")
    compiler.print_symbol_table()
    
    # 打印 golden result
    print(f"\nGolden Result (S = Q @ K^T) shape: {golden_result['original_output'].shape}")
    print(f"Golden Result S (first 5x5):\n{golden_result['original_output'][:5, :5]}")
    
    # 验证计算结果
    Q_expected = X @ W_Q
    K_expected = X @ W_K
    V_expected = X @ W_V
    S_expected = Q_expected @ K_expected.T
    print(f"\nExpected Q shape: {Q_expected.shape}")
    print(f"Expected K shape: {K_expected.shape}")
    print(f"Expected V shape: {V_expected.shape}")
    print(f"Expected S = Q @ K^T shape: {S_expected.shape}")
    print(f"Expected S (first 5x5):\n{S_expected[:5, :5]}")
    
    # 检查生成的代码
    print(f"\n✓ Generated code saved to: {build_dir / 'generated_asm_code_before_run.asm'}")
    print(f"  Total lines: {len(gen_assembly_code.splitlines())}")
    
    # 检查 H_PREFETCH_M 对齐问题
    alignment_warnings = result["alignment_warnings"]
    if alignment_warnings:
        print("\n⚠️  WARNINGS found in generated code:")
        for w in alignment_warnings:
            print(f"  {w}")
    else:
        print("✓ No alignment issues found in H_PREFETCH_M instructions")
    
    # 打印摘要
    print("\n" + "="*60)
    print("Attention QKV + QK^T Test Summary")
    print("="*60)
    x_info = symbol_table["X_batch"]
    q_info = symbol_table["Q"]
    k_info = symbol_table["K"]
    v_info = symbol_table["V"]
    s_info = symbol_table["S"]
    
    print(f"✓ Load_Batch X_batch: VRAM={x_info.vram_addr}, shape={x_info.shape}")
    print(f"✓ Projection Q = X @ W_Q -> VRAM={q_info.vram_addr}, shape={q_info.shape}")
    print(f"✓ Projection K = X @ W_K -> VRAM={k_info.vram_addr}, shape={k_info.shape}")
    print(f"✓ Projection V = X @ W_V -> VRAM={v_info.vram_addr}, shape={v_info.shape}")
    print(f"✓ Store K to HBM as 'k_stored'")
    print(f"✓ Load K_mat from 'k_stored' (as Matrix, transposed)")
    print(f"✓ S = Q @T K_mat -> VRAM={s_info.vram_addr}, shape={s_info.shape}")
    print(f"✓ Result S location: row {result['comparison_params']['start_row_idx']}, {result['comparison_params']['num_rows']} rows")
    print(f"✓ Comparison params saved to: {build_dir / 'comparison_params.json'}")
    print("="*60)
