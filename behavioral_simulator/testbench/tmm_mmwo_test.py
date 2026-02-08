#!/usr/bin/env python3
"""
Test for M_TMM + M_MM_WO using AutoCompilerHelper.

Computes: Q @ K^T where Q=[batch, hidden], K=[out_features, hidden] → result=[batch, out_features]

Usage:
    1. python tmm_mmwo_test.py      # 生成代码和环境
    2. cargo run (behavioral sim)   # 运行仿真器
    3. python view_mem.py           # 查看比较结果
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTBENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(TESTBENCH_DIR))

import torch
from config_utils import update_plena_config
from auto_compiler_helper import AutoCompilerHelper


def main() -> None:
    update_plena_config(vlen=64, mlen=64, blen=4, verbose=False)

    mlen = 64
    blen = 4

    # -----------------------
    # Test dimensions
    # -----------------------
    batch = 16          # Number of rows in Q (must be multiple of blen)
    hidden_size = 128   # Shared dimension
    out_features = 128  # Output dimension (must be multiple of blen)

    # Constraints check
    assert hidden_size % mlen == 0, f"hidden_size ({hidden_size}) must be multiple of mlen ({mlen})"
    assert batch % blen == 0, f"batch ({batch}) must be multiple of blen ({blen})"
    assert out_features % blen == 0, f"out_features ({out_features}) must be multiple of blen ({blen})"
    
    num_out_blocks = (out_features + mlen - 1) // mlen
    k_padded_rows = num_out_blocks * mlen

    # -----------------------
    # Test data
    # -----------------------
    torch.manual_seed(42)
    
    # Q: (batch, hidden_size)
    # K: (out_features, hidden_size) - for TMM matmul Q @T K = Q @ K^T
    q_orig = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    k_orig = torch.randn(out_features, hidden_size, dtype=torch.bfloat16)
    
    # Pad K if needed (pad rows to mlen boundary)
    k_padded = torch.zeros(k_padded_rows, hidden_size, dtype=torch.bfloat16)
    k_padded[:out_features, :] = k_orig

    print("=" * 60)
    print("M_TMM + M_MM_WO Test (via AutoCompilerHelper)")
    print("=" * 60)
    print(f"Q shape: [{batch}, {hidden_size}]")
    print(f"K original shape: [{out_features}, {hidden_size}]")
    print(f"K padded shape: [{k_padded_rows}, {hidden_size}]")
    print(f"Result shape: [{batch}, {out_features}]")
    print(f"num_out_blocks: {num_out_blocks}")

    # -----------------------
    # Use AutoCompilerHelper
    # -----------------------
    helper = AutoCompilerHelper(mlen=mlen, blen=blen)
    helper.add_tensor("q", q_orig)
    helper.add_tensor("k", k_padded)
    
    code = """
Q = Load_Batch(q)
K = Load_Matrix(k)
R = Q @T K
Result R
"""
    
    print("\nTensor Layout (auto-allocated):")
    for name in ["q", "k"]:
        h, w = helper.tensor_shapes[name]
        hbm_addr = helper.hbm_layout[name]
        hbm_size = helper.hbm_sizes[name]
        print(f"  {name}: shape=({h}, {w}), hbm_addr={hbm_addr}, hbm_size={hbm_size}")
    
    result = helper.compile_and_setup(
        code=code,
        fp_preload=[0.0, 1e-6, 1.0],
        asm_name="tmm_mmwo_test"
    )
    
    # Print symbol table
    print("\n" + "=" * 60)
    print("Symbol Table")
    print("=" * 60)
    for name, info in result["symbol_table"].table.items():
        print(f"  {name}: {info}")
    
    # Print alignment warnings if any
    if result.get("alignment_warnings"):
        print("\n⚠️ Alignment Warnings:")
        for warning in result["alignment_warnings"]:
            print(f"  {warning}")
    else:
        print("\n✓ No alignment issues found in H_PREFETCH_M instructions")
    
    print("\n" + "=" * 60)
    print("Generated Assembly (first 50 lines)")
    print("=" * 60)
    for i, line in enumerate(result["generated_code"].split('\n')[:50]):
        print(f"{i+1:3}: {line}")

    # -----------------------
    # Print summary (like linear_compiler_test.py)
    # -----------------------
    build_dir = result["build_dir"]
    params = result["comparison_params"]
    
    print("\n" + "=" * 60)
    print("TMM MatMul Test Summary")
    print("=" * 60)
    
    symbol_table = result["symbol_table"]
    q_info = symbol_table["Q"]
    k_info = symbol_table["K"]
    r_info = symbol_table["R"]
    
    print(f"✓ Load_Batch Q: VRAM={q_info.vram_addr}, shape={q_info.shape}")
    print(f"✓ Load_Matrix K: HBM addr={k_info.hbm_addr}, HBM size={k_info.hbm_size}")
    print(f"✓ TMM MatMul R = Q @T K -> VRAM={r_info.vram_addr}, shape={r_info.shape}")
    print(f"✓ Result location: row {params['start_row_idx']}, {params['num_rows']} rows")
    print(f"✓ Comparison params: num_batches={params['num_batches']}, elements_per_batch={params['elements_per_batch']}")
    print(f"✓ Generated code saved to: {build_dir / 'generated_asm_code_before_run.asm'}")
    print(f"✓ Comparison params saved to: {build_dir / 'comparison_params.json'}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  1. Run behavioral simulator: cargo run")
    print("  2. View comparison results: python view_mem.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
