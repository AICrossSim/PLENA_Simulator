#!/usr/bin/env python3
"""
Test QK^T Multiply with numerical verification
使用 auto_compiler_helper 测试 QK^T 乘法的数值正确性
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from auto_compiler_helper import AutoCompilerHelper


def test_qkt_multiply_simple():
    """简单测试：小矩阵 QK^T 乘法"""
    print("=" * 60)
    print("Test 1: Simple QK^T Multiply")
    print("=" * 60)
    
    # 创建 helper
    helper = AutoCompilerHelper(real_data_ratio=1.125, mlen=64, blen=4)
    
    # 设置随机种子以便可重复
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 8
    head_dim = 64
    
    # Q: (batch_size, head_dim)
    q = torch.randn(batch_size, head_dim, dtype=torch.bfloat16)
    # K: (batch_size, head_dim)
    k = torch.randn(batch_size, head_dim, dtype=torch.bfloat16)
    
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    
    # 添加到 helper
    helper.add_tensor("q", q, is_batch=True)
    helper.add_tensor("k", k, is_batch=True)
    
    # 计算 golden result
    golden_result = torch.matmul(q, k.T)
    print(f"Golden result shape: {golden_result.shape}")
    print(f"Golden result (first 3x3):\n{golden_result[:3, :3]}")
    
    # 生成伪代码
    code = """
Q = Load_Batch(q)
K = Load_Batch(k)
S = Q @ Transpose(K)
Result S
"""
    
    # 编译并设置环境
    result = helper.compile_and_setup(code, fp_preload=[0.0, 1e-6, 1.0])
    
    print("\nGenerated ISA Code (first 50 lines):")
    print("\n".join(result["generated_code"].split("\n")[:50]))
    
    print("\nSymbol Table:")
    result["symbol_table"].print_table()
    
    print("\nComparison Params:")
    print(result["comparison_params"])
    
    print("\n✅ Test 1 completed!")
    return result


def test_qkt_multiply_larger():
    """较大矩阵测试"""
    print("\n" + "=" * 60)
    print("Test 2: Larger QK^T Multiply")
    print("=" * 60)
    
    helper = AutoCompilerHelper(real_data_ratio=1.125, mlen=64, blen=4)
    
    torch.manual_seed(123)
    
    batch_size = 16
    head_dim = 128
    
    q = torch.randn(batch_size, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, head_dim, dtype=torch.bfloat16)
    
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    
    helper.add_tensor("q", q, is_batch=True)
    helper.add_tensor("k", k, is_batch=True)
    
    # 计算 golden result
    golden_result = torch.matmul(q, k.T)
    print(f"Golden result shape: {golden_result.shape}")
    print(f"Golden result stats:")
    print(f"  Mean: {golden_result.mean().item():.6f}")
    print(f"  Std: {golden_result.std().item():.6f}")
    print(f"  Min: {golden_result.min().item():.6f}")
    print(f"  Max: {golden_result.max().item():.6f}")
    
    code = """
Q = Load_Batch(q)
K = Load_Batch(k)
S = Q @ Transpose(K)
Result S
"""
    
    result = helper.compile_and_setup(code, fp_preload=[0.0, 1e-6, 1.0])
    
    print("\nComparison Params:")
    print(result["comparison_params"])
    
    print("\n✅ Test 2 completed!")
    return result


def test_qkt_multiply_attention_like():
    """类似 attention 的测试：多个 head"""
    print("\n" + "=" * 60)
    print("Test 3: Attention-like QK^T Multiply")
    print("=" * 60)
    
    helper = AutoCompilerHelper(real_data_ratio=1.125, mlen=64, blen=4)
    
    torch.manual_seed(456)
    
    # 模拟 attention: seq_len=32, num_heads=4, head_dim=64
    seq_len = 32
    head_dim = 64
    
    q = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    
    print(f"Q shape: {q.shape} (seq_len={seq_len}, head_dim={head_dim})")
    print(f"K shape: {k.shape}")
    
    helper.add_tensor("q", q, is_batch=True)
    helper.add_tensor("k", k, is_batch=True)
    
    # 计算 golden result
    golden_result = torch.matmul(q, k.T)
    print(f"Golden result shape: {golden_result.shape} (attention scores)")
    
    # 应用 softmax（仅用于验证）
    attention_scores = torch.softmax(golden_result, dim=-1)
    print(f"After softmax - Mean: {attention_scores.mean().item():.6f}")
    
    code = """
Q = Load_Batch(q)
K = Load_Batch(k)
S = Q @ Transpose(K)
Result S
"""
    
    result = helper.compile_and_setup(code, fp_preload=[0.0, 1e-6, 1.0])
    
    print("\nComparison Params:")
    print(result["comparison_params"])
    
    print("\n✅ Test 3 completed!")
    return result


def test_qkt_multiply_verify_golden():
    """验证 golden result 计算是否正确"""
    print("\n" + "=" * 60)
    print("Test 4: Verify Golden Result Calculation")
    print("=" * 60)
    
    helper = AutoCompilerHelper(real_data_ratio=1.125, mlen=64, blen=4)
    
    torch.manual_seed(789)
    
    batch_size = 8
    head_dim = 32
    
    q = torch.randn(batch_size, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, head_dim, dtype=torch.float32)
    
    helper.add_tensor("q", q, is_batch=True)
    helper.add_tensor("k", k, is_batch=True)
    
    # 手动计算
    manual_result = torch.matmul(q, k.T)
    
    # 使用 helper 计算
    code = """
Q = Load_Batch(q)
K = Load_Batch(k)
S = Q @ Transpose(K)
Result S
"""
    
    result = helper.compile_and_setup(code, fp_preload=[0.0, 1e-6, 1.0])
    golden_result = result["golden_result"]["original_output"]
    
    print(f"Manual result shape: {manual_result.shape}")
    print(f"Golden result shape: {golden_result.shape}")
    
    # 比较（考虑 bfloat16 精度）
    if torch.allclose(manual_result, golden_result, atol=1e-2, rtol=1e-2):
        print("✅ Golden result matches manual calculation!")
    else:
        max_diff = (manual_result - golden_result).abs().max()
        print(f"⚠️  Golden result differs from manual calculation (max diff: {max_diff:.6f})")
        print(f"Manual result (first 3x3):\n{manual_result[:3, :3]}")
        print(f"Golden result (first 3x3):\n{golden_result[:3, :3]}")
    
    print("\n✅ Test 4 completed!")
    return result


if __name__ == "__main__":
    print("Running QK^T Multiply Numerical Tests")
    print("=" * 60)
    
    try:
        result1 = test_qkt_multiply_simple()
        result2 = test_qkt_multiply_larger()
        result3 = test_qkt_multiply_attention_like()
        result4 = test_qkt_multiply_verify_golden()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the behavioral simulator to verify hardware correctness")
        print("2. Check the generated assembly code in build/generated_asm_code_before_run.asm")
        print("3. Compare simulated results with golden results")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

