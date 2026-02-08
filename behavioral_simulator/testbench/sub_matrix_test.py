"""
Sub Matrix Test: 测试子矩阵分块加载和计算
使用 AutoCompilerHelper 自动处理 HBM 地址、环境搭建和 golden value 计算

测试场景：
1. 256x256 的大矩阵 W 被分成 4x4 个 64x64 的子块
2. 加载 W[:][col_idx] 到 MRAM（一整列子块）
3. 计算 C = A @ W[:][col_idx]（子块投影）
   即 C = A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
4. 验证结果与 golden value 一致

关键特性：
- sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
- 需要加载 W 的第 col_idx 列子块（所有行的第 col_idx 列）
- 所有地址在 compiler 阶段预计算
- HBM 格式: [rows, cols] 行主序
- VRAM 格式: [batch, mlen, hidden/mlen] 列块优先
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
    batch_size = 64  # batch size (必须 >= 64)
    hidden_size = 256  # hidden dimension (必须是 64 的倍数)
    out_features = 256  # output features (必须是 64 的倍数，且 == hidden_size for sub_projection)
    mlen = 64  # 子块大小
    blen = 4   # batch 分块大小
    real_data_ratio = (8*8 + 8) / (8 * 8)  # 1.125
    
    # 子块列索引（测试 sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen]）
    sub_col_idx = 1  # 取 W 的第 1 列子块（即 W[:, 64:128]）
    
    torch.manual_seed(42)
    
    # ========================================================================
    # 创建测试数据
    # ========================================================================
    # Input: A : (batch_size, hidden_size)
    A = torch.randn(batch_size, hidden_size)
    
    # Weight: W : (out_features, hidden_size)
    # 注意：W 会被分成 (out_features/mlen) x (hidden_size/mlen) 个子块
    # 对于 sub_projection，我们取 W[:, col_idx*mlen:(col_idx+1)*mlen]
    W = torch.randn(out_features, hidden_size)
    
    # ========================================================================
    # 定义伪代码
    # ========================================================================
    # 流程：
    # 1. 加载 A 到 VRAM
    # 2. 注册 W 为子矩阵（自动分块）
    # 3. 加载 W[:][col_idx] 到 MRAM（一整列子块）
    # 4. 计算 C = A @ SubMatrix(W, col_idx)
    #    即 C = A @ W[:, col_idx*mlen:(col_idx+1)*mlen]
    code = f"""
    A_batch = Load_Batch(A)
    W_sub = Register_SubMatrix(W)
    Load_SubMatrix(W_sub, {sub_col_idx})
    C = A_batch @ SubMatrix(W_sub, {sub_col_idx})
    Result C
    """
    
    # ========================================================================
    # 使用 AutoCompilerHelper 自动处理
    # ========================================================================
    print("=" * 60)
    print("Sub Matrix Test: sub_projection")
    print("=" * 60)
    print(f"Input A: shape {A.shape}")
    print(f"Weight W: shape {W.shape}")
    print(f"\n操作: C = A @ W[:, col_idx*mlen:(col_idx+1)*mlen]")
    print(f"Sub column index: {sub_col_idx}")
    print(f"取 W 的列范围: W[:, {sub_col_idx*mlen}:{(sub_col_idx+1)*mlen}]")
    print(f"W 子块形状: ({out_features}, {mlen})")
    print(f"期望结果形状: ({batch_size}, {mlen})")
    
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
    result = helper.compile_and_setup(code, asm_name="sub_matrix_test")
    
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
    
    # 打印子矩阵布局
    print("\n" + "=" * 60)
    print("Sub Matrix Layout:")
    print("=" * 60)
    compiler.compiler.print_sub_matrix_layout("W_sub")
    
    # 打印 golden result
    print("\n" + "=" * 60)
    print("Golden Result:")
    print("=" * 60)
    print(f"Golden Result (C = A @ W[:, {sub_col_idx*mlen}:{(sub_col_idx+1)*mlen}]) shape: {golden_result['original_output'].shape}")
    print(f"Golden Result C (first 5x5):\n{golden_result['original_output'][:5, :5]}")
    
    # 手动验证计算结果
    # sub_projection: A @ SubMatrix(W, idx) = A @ W[:, idx*mlen:(idx+1)*mlen]
    W_sub_col = W[:, sub_col_idx*mlen:(sub_col_idx+1)*mlen]  # (out_features, mlen)
    C_expected = A @ W_sub_col  # (batch_size, mlen)
    print(f"\n手动计算 Expected C shape: {C_expected.shape}")
    print(f"Expected C (first 5x5):\n{C_expected[:5, :5]}")
    
    # 检查 golden 和 expected 是否一致
    if torch.allclose(golden_result['original_output'], C_expected, rtol=1e-4, atol=1e-4):
        print("\n✓ Golden result matches expected!")
    else:
        print("\n✗ Golden result DOES NOT match expected!")
        diff = (golden_result['original_output'] - C_expected).abs().max()
        print(f"  Max difference: {diff}")
    
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
    
    # ========================================================================
    # 打印摘要
    # ========================================================================
    print("\n" + "=" * 60)
    print("Sub Matrix Test Summary")
    print("=" * 60)
    
    a_info = symbol_table["A_batch"]
    c_info = symbol_table["C"]
    
    print(f"✓ Load_Batch A_batch: VRAM={a_info.vram_addr}, shape={a_info.shape}")
    print(f"✓ Register_SubMatrix W_sub: {out_features}x{hidden_size} -> {out_features//mlen}x{hidden_size//mlen} blocks")
    print(f"✓ Load_SubMatrix W_sub[:][{sub_col_idx}] -> MRAM (加载列子块)")
    print(f"✓ Sub Projection C = A_batch @ W[:, {sub_col_idx*mlen}:{(sub_col_idx+1)*mlen}]")
    print(f"  C -> VRAM={c_info.vram_addr}, shape={c_info.shape}")
    print(f"✓ Result C location: row {result['comparison_params']['start_row_idx']}, {result['comparison_params']['num_rows']} rows")
    print(f"✓ Comparison params saved to: {build_dir / 'comparison_params.json'}")
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
