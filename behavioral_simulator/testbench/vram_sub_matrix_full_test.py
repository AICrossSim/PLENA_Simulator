"""
VRAM Sub Matrix Full Test: 测试 (128x64) @ (64x128) = (128x128) 的分块计算
使用 AutoCompilerHelper 自动处理 HBM 地址、环境搭建和 golden value 计算

测试场景：
1. A: (128, 64) 在 VRAM 中，分成 2 行 x 1 列子块
2. W: (64, 128) 在 MRAM 中，分成 1 行 x 2 列子块
3. 计算 C = A @ W = (128, 128)

分块计算：
- C[0][0] = A[0][:] @ W[:][0] = (64, 64)
- C[0][1] = A[0][:] @ W[:][1] = (64, 64)
- C[1][0] = A[1][:] @ W[:][0] = (64, 64)
- C[1][1] = A[1][:] @ W[:][1] = (64, 64)

MRAM 约束：
- W: 64x128 -> 1 行 x 2 列子块
- 加载 W[:][0] + W[:][1] = 2 个子块 = 8192 < 16384，够用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

import torch
import json
from auto_compiler_helper import AutoCompilerHelper


if __name__ == "__main__":
    # ========================================================================
    # 测试参数
    # ========================================================================
    batch_size = 128  # A 的行数 (2 个 64 行子块)
    hidden_size = 64  # A 的列数 = W 的行数 (1 个子块)
    out_features = 128  # W 的列数 (2 个子块)
    mlen = 64  # 子块大小
    blen = 4   # batch 分块大小
    real_data_ratio = (8*8 + 8) / (8 * 8)  # 1.125
    
    # 矩阵布局:
    # A: (128, 64) -> 2 行 x 1 列子块
    # W: (64, 128) -> 1 行 x 2 列子块
    # C: (128, 128) -> 2 行 x 2 列子块
    
    torch.manual_seed(42)
    
    # ========================================================================
    # 创建测试数据
    # ========================================================================
    A = torch.randn(batch_size, hidden_size)   # (128, 64)
    W = torch.randn(hidden_size, out_features)  # (64, 128)
    
    # ========================================================================
    # 定义伪代码
    # ========================================================================
    # 计算完整的 A @ W = (128, 128)
    # 使用新语法：分配大矩阵，然后将子块结果写入指定位置
    # - C[0][0] = A[0][:] @ W[:][0]
    # - C[0][1] = A[0][:] @ W[:][1]
    # - C[1][0] = A[1][:] @ W[:][0]
    # - C[1][1] = A[1][:] @ W[:][1]
    code = f"""
    A_batch = Load_Batch(A)
    A_sub = Register_VRAMSubMatrix(A_batch)
    W_sub = Register_SubMatrix(W)
    Load_SubMatrix(W_sub, 0)
    Load_SubMatrix(W_sub, 1)
    C = Allocate_VRAMMatrix(128, 128)
    C[0][0] = VRAMSubMatrix(A_sub, 0) @ SubMatrix(W_sub, 0)
    C[0][1] = VRAMSubMatrix(A_sub, 0) @ SubMatrix(W_sub, 1)
    C[1][0] = VRAMSubMatrix(A_sub, 1) @ SubMatrix(W_sub, 0)
    C[1][1] = VRAMSubMatrix(A_sub, 1) @ SubMatrix(W_sub, 1)
    Result C
    """
    
    # ========================================================================
    # 使用 AutoCompilerHelper 自动处理
    # ========================================================================
    print("=" * 60)
    print("VRAM Sub Matrix Full Test: (128x64) @ (64x128) = (128x128)")
    print("=" * 60)
    print(f"Input A: shape {A.shape}")
    print(f"Weight W: shape {W.shape}")
    print(f"\n分块计算:")
    print(f"  C00 = A[0][:] @ W[:][0] = (64,64) @ (64,64) -> (64,64)")
    print(f"  C01 = A[0][:] @ W[:][1] = (64,64) @ (64,64) -> (64,64)")
    print(f"  C10 = A[1][:] @ W[:][0] = (64,64) @ (64,64) -> (64,64)")
    print(f"  C11 = A[1][:] @ W[:][1] = (64,64) @ (64,64) -> (64,64)")
    print(f"\n组合后 C = [[C00, C01], [C10, C11]] = (128, 128)")
    
    helper = AutoCompilerHelper(real_data_ratio=real_data_ratio, mlen=mlen, blen=blen)
    
    # 添加 tensors
    helper.add_tensor("A", A, is_batch=True)
    helper.add_tensor("W", W, is_batch=False)
    
    print("\nTensor Layout (auto-allocated):")
    for name in ["A", "W"]:
        h, w = helper.tensor_shapes[name]
        hbm_addr = helper.hbm_layout[name]
        hbm_size = helper.hbm_sizes[name]
        print(f"  {name}: shape=({h}, {w}), hbm_addr={hbm_addr}, hbm_size={hbm_size}")
    
    # 编译并自动设置环境
    result = helper.compile_and_setup(code, asm_name="vram_sub_matrix_full_test")
    
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
    
    # 打印 MRAM 子矩阵布局
    print("\n" + "=" * 60)
    print("MRAM Sub Matrix Layout (W_sub):")
    print("=" * 60)
    compiler.compiler.print_sub_matrix_layout("W_sub")
    
    # 打印 VRAM 子矩阵布局
    print("\n" + "=" * 60)
    print("VRAM Sub Matrix Layout (A_sub):")
    print("=" * 60)
    compiler.compiler.print_vram_sub_matrix_layout("A_sub")
    
    # ========================================================================
    # 手动验证计算结果
    # ========================================================================
    print("\n" + "=" * 60)
    print("手动验证各子块:")
    print("=" * 60)
    
    # 完整矩阵乘法
    C_full = A @ W  # (128, 128)
    print(f"完整 C = A @ W shape: {C_full.shape}")
    
    # 分块结果
    C00_expected = A[0:64, :] @ W[:, 0:64]   # (64, 64)
    C01_expected = A[0:64, :] @ W[:, 64:128]  # (64, 64)
    C10_expected = A[64:128, :] @ W[:, 0:64]  # (64, 64)
    C11_expected = A[64:128, :] @ W[:, 64:128] # (64, 64)
    
    print(f"\nC00 = A[0:64,:] @ W[:,0:64]: {C00_expected.shape}")
    print(f"  (first 3x3):\n{C00_expected[:3, :3]}")
    
    print(f"\nC01 = A[0:64,:] @ W[:,64:128]: {C01_expected.shape}")
    print(f"  (first 3x3):\n{C01_expected[:3, :3]}")
    
    print(f"\nC10 = A[64:128,:] @ W[:,0:64]: {C10_expected.shape}")
    print(f"  (first 3x3):\n{C10_expected[:3, :3]}")
    
    print(f"\nC11 = A[64:128,:] @ W[:,64:128]: {C11_expected.shape}")
    print(f"  (first 3x3):\n{C11_expected[:3, :3]}")
    
    # 验证分块组合等于完整乘法
    C_combined = torch.zeros(128, 128)
    C_combined[0:64, 0:64] = C00_expected
    C_combined[0:64, 64:128] = C01_expected
    C_combined[64:128, 0:64] = C10_expected
    C_combined[64:128, 64:128] = C11_expected
    
    if torch.allclose(C_full, C_combined, rtol=1e-5, atol=1e-5):
        print("\n✓ 分块组合 == 完整乘法!")
    else:
        print("\n✗ 分块组合 != 完整乘法!")
        diff = (C_full - C_combined).abs().max()
        print(f"  Max difference: {diff}")
    
    # 验证 golden (输出是完整的 C 矩阵)
    print("\n" + "=" * 60)
    print("Golden Result (完整矩阵 C):")
    print("=" * 60)
    print(f"Golden Result shape: {golden_result['original_output'].shape}")
    print(f"Golden Result (first 3x3):\n{golden_result['original_output'][:3, :3]}")
    print(f"Golden Result C[64:67, 64:67]:\n{golden_result['original_output'][64:67, 64:67]}")
    
    if torch.allclose(golden_result['original_output'], C_full, rtol=1e-4, atol=1e-4):
        print("\n✓ Golden C matches A @ W!")
    else:
        print("\n✗ Golden C DOES NOT match A @ W!")
        diff = (golden_result['original_output'] - C_full).abs().max()
        print(f"  Max difference: {diff}")
    
    # ========================================================================
    # 打印摘要
    # ========================================================================
    print("\n" + "=" * 60)
    print("VRAM Sub Matrix Full Test Summary")
    print("=" * 60)
    
    a_info = symbol_table["A_batch"]
    c_info = symbol_table["C"]
    
    print(f"✓ A: ({batch_size}, {hidden_size}) -> {batch_size//mlen} 行 x {hidden_size//mlen} 列子块")
    print(f"✓ W: ({hidden_size}, {out_features}) -> {hidden_size//mlen} 行 x {out_features//mlen} 列子块")
    print(f"✓ C = Allocate_VRAMMatrix(128, 128)")
    print(f"  VRAM={c_info.vram_addr}, shape={c_info.shape}")
    print(f"✓ C[0][0] = A_sub[0][:] @ W_sub[:][0]")
    print(f"✓ C[0][1] = A_sub[0][:] @ W_sub[:][1]")
    print(f"✓ C[1][0] = A_sub[1][:] @ W_sub[:][0]")
    print(f"✓ C[1][1] = A_sub[1][:] @ W_sub[:][1]")
    print("=" * 60)
    
    # ========================================================================
    # 打印生成的 ISA 代码片段
    # ========================================================================
    print("\n" + "=" * 60)
    print("Generated ISA Code (first 100 lines):")
    print("=" * 60)
    lines = gen_assembly_code.splitlines()
    for i, line in enumerate(lines[:100]):
        print(f"{i+1:4d}: {line}")
    if len(lines) > 100:
        print(f"... ({len(lines) - 100} more lines)")

