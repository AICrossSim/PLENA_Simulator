"""
VRAM Sub Matrix Test: 测试 VRAM 子块与 MRAM SubMatrix 的乘法（包括转置）
使用 AutoCompilerHelper 自动处理 HBM 地址、环境搭建和 golden value 计算

测试场景：
1. 128x128 的矩阵 A 在 VRAM 中，被分成 2x2 个 64x64 的子块
2. 128x128 的矩阵 W 在 MRAM 中，被分成 2x2 个 64x64 的子块
3. 非转置: C = A[1][:] @ W[:][1]
   即 C = A[64:128, :] @ W[:, 64:128] -> (64, 64)
4. 转置: C_T = A[1][:] @ W[0][:]^T
   即 C_T = A[64:128, :] @ W[0:64, :].T -> (64, 64)
5. 验证结果与 golden value 一致

MRAM 约束：
- MRAM 总大小: 16384（4 个 64x64 子块）
- Load_SubMatrix 列: 2 个子块 = 8192
- Load_SubMatrix_Row 行: 2 个子块 = 8192
- 总计刚好用满 MRAM

关键特性：
- VRAM 子块: A[row_idx][:] 是 VRAM 矩阵的某一行子块
- MRAM 子块: W[:][col_idx] 是 MRAM 矩阵的某一列子块（非转置）
- MRAM 子块: W[row_idx][:] 是 MRAM 矩阵的某一行子块（转置）
- 所有地址在 compiler 阶段预计算
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
    # 注意：VRAM 大小有限（~65536），MRAM 大小有限（16384 = 4 个子块）
    # A: batch_size × hidden_size
    # W: hidden_size × out_features
    # MRAM 约束：Load_SubMatrix 列 + Load_SubMatrix_Row 行 <= 4 个子块
    batch_size = 128  # A 的行数（必须是 64 的倍数）
    hidden_size = 128  # A 的列数 = W 的行数（2 个子块）
    out_features = 128  # W 的列数（2 个子块）
    mlen = 64  # 子块大小
    blen = 4   # batch 分块大小
    real_data_ratio = (8*8 + 8) / (8 * 8)  # 1.125
    
    # MRAM 使用计算:
    # - W 是 128x128，分成 2x2 个子块
    # - Load_SubMatrix 列: 2 个子块 = 8192
    # - Load_SubMatrix_Row 行: 2 个子块 = 8192
    # - 总计: 16384 = MRAM 总大小，刚好够
    
    # 子块索引
    vram_row_idx = 1  # VRAM A 的第 1 行子块（A[64:128, :]）
    mram_col_idx = 1  # MRAM W 的第 1 列子块（W[:, 64:128]）用于非转置
    mram_row_idx = 0  # MRAM W 的第 0 行子块（W[0:64, :]）用于转置
    
    torch.manual_seed(42)
    
    # ========================================================================
    # 创建测试数据
    # ========================================================================
    # Input: A : (batch_size, hidden_size) - 将被加载到 VRAM
    A = torch.randn(batch_size, hidden_size)
    
    # Weight: W : (hidden_size, out_features) - 将被加载到 MRAM
    # 注意：这里 W 形状是 (hidden_size, out_features)，不是 (out_features, hidden_size)
    W = torch.randn(hidden_size, out_features)
    
    # ========================================================================
    # 定义伪代码
    # ========================================================================
    # 流程：
    # 1. 加载 A 到 VRAM
    # 2. 将 VRAM 中的 A 注册为子块管理
    # 3. 注册 W 为 MRAM 子矩阵
    # 4. 加载 W[:][mram_col_idx] 到 MRAM（列子块，用于非转置）
    # 5. 加载 W[mram_row_idx][:] 到 MRAM（行子块，用于转置）
    # 6. 计算 C = A[vram_row_idx][:] @ W[:][mram_col_idx]（非转置）
    # 7. 计算 C_T = A[vram_row_idx][:] @ W[mram_row_idx][:]^T（转置）
    code = f"""
    A_batch = Load_Batch(A)
    A_sub = Register_VRAMSubMatrix(A_batch)
    W_sub = Register_SubMatrix(W)
    Load_SubMatrix(W_sub, {mram_col_idx})
    Load_SubMatrix_Row(W_sub, {mram_row_idx})
    C = VRAMSubMatrix(A_sub, {vram_row_idx}) @ SubMatrix(W_sub, {mram_col_idx})
    C_T = VRAMSubMatrix(A_sub, {vram_row_idx}) @T SubMatrix(W_sub, {mram_row_idx})
    Result C_T
    """
    
    # ========================================================================
    # 使用 AutoCompilerHelper 自动处理
    # ========================================================================
    print("=" * 60)
    print("VRAM Sub Matrix Test: 非转置 + 转置")
    print("=" * 60)
    print(f"Input A: shape {A.shape}")
    print(f"Weight W: shape {W.shape}")
    print(f"\n操作1 (非转置): C = A[{vram_row_idx}][:] @ W[:][{mram_col_idx}]")
    print(f"  = A[{vram_row_idx*mlen}:{(vram_row_idx+1)*mlen}, :] @ W[:, {mram_col_idx*mlen}:{(mram_col_idx+1)*mlen}]")
    print(f"  A 行子块形状: ({mlen}, {hidden_size})")
    print(f"  W 列子块形状: ({hidden_size}, {mlen})")
    print(f"  期望结果形状: ({mlen}, {mlen})")
    print(f"\n操作2 (转置): C_T = A[{vram_row_idx}][:] @T W[{mram_row_idx}][:]")
    print(f"  = A[{vram_row_idx*mlen}:{(vram_row_idx+1)*mlen}, :] @ W[{mram_row_idx*mlen}:{(mram_row_idx+1)*mlen}, :].T")
    print(f"  A 行子块形状: ({mlen}, {hidden_size})")
    print(f"  W 行子块形状 (转置前): ({mlen}, {out_features}) -> 转置后 ({out_features}, {mlen})")
    print(f"  期望结果形状: ({mlen}, {mlen})")
    
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
    result = helper.compile_and_setup(code, asm_name="vram_sub_matrix_test")
    
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
    
    # 打印 golden result
    print("\n" + "=" * 60)
    print("Golden Result (最终输出 C_T):")
    print("=" * 60)
    print(f"Golden Result shape: {golden_result['original_output'].shape}")
    print(f"Golden Result C_T (first 5x5):\n{golden_result['original_output'][:5, :5]}")
    
    # 手动验证计算结果
    # ========================================================================
    # 非转置: A[row_idx][:] @ W[:][col_idx]
    # ========================================================================
    A_sub_row = A[vram_row_idx*mlen:(vram_row_idx+1)*mlen, :]  # (mlen, hidden_size)
    W_sub_col = W[:, mram_col_idx*mlen:(mram_col_idx+1)*mlen]  # (hidden_size, mlen)
    C_expected = A_sub_row @ W_sub_col  # (mlen, mlen)
    print(f"\n手动计算 Expected C (非转置) shape: {C_expected.shape}")
    print(f"Expected C (first 5x5):\n{C_expected[:5, :5]}")
    
    # ========================================================================
    # 转置: A[row_idx][:] @ W[row_idx][:]^T
    # ========================================================================
    W_sub_row = W[mram_row_idx*mlen:(mram_row_idx+1)*mlen, :]  # (mlen, hidden_size)
    C_T_expected = A_sub_row @ W_sub_row.T  # (mlen, mlen)
    print(f"\n手动计算 Expected C_T (转置) shape: {C_T_expected.shape}")
    print(f"Expected C_T (first 5x5):\n{C_T_expected[:5, :5]}")
    
    # 检查 golden 和 expected 是否一致（最终输出是 C_T）
    if torch.allclose(golden_result['original_output'], C_T_expected, rtol=1e-4, atol=1e-4):
        print("\n✓ Golden result matches expected C_T!")
    else:
        print("\n✗ Golden result DOES NOT match expected C_T!")
        diff = (golden_result['original_output'] - C_T_expected).abs().max()
        print(f"  Max difference: {diff}")
    
    # 检查生成的代码
    print(f"\n✓ Generated code saved to: {build_dir / 'generated_asm_code_before_run.asm'}")
    print(f"  Total lines: {len(gen_assembly_code.splitlines())}")
    
    # ========================================================================
    # 打印摘要
    # ========================================================================
    print("\n" + "=" * 60)
    print("VRAM Sub Matrix Test Summary")
    print("=" * 60)
    
    a_info = symbol_table["A_batch"]
    c_info = symbol_table["C"]
    ct_info = symbol_table["C_T"]
    
    print(f"✓ Load_Batch A_batch: VRAM={a_info.vram_addr}, shape={a_info.shape}")
    print(f"✓ Register_VRAMSubMatrix A_sub: {batch_size}x{hidden_size} -> {batch_size//mlen} rows x {hidden_size//mlen} cols blocks")
    print(f"✓ Register_SubMatrix W_sub: {hidden_size}x{out_features} -> {hidden_size//mlen}x{out_features//mlen} blocks")
    print(f"✓ Load_SubMatrix W_sub[:][{mram_col_idx}] -> MRAM (列子块)")
    print(f"✓ Load_SubMatrix_Row W_sub[{mram_row_idx}][:] -> MRAM (行子块)")
    print(f"✓ VRAM Sub Projection (非转置): A_sub[{vram_row_idx}][:] @ W_sub[:][{mram_col_idx}]")
    print(f"  C -> VRAM={c_info.vram_addr}, shape={c_info.shape}")
    print(f"✓ VRAM Sub Projection T (转置): A_sub[{vram_row_idx}][:] @T W_sub[{mram_row_idx}][:]")
    print(f"  C_T -> VRAM={ct_info.vram_addr}, shape={ct_info.shape}")
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

