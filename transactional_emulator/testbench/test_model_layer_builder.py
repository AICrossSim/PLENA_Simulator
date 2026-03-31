"""
Unit tests for model_layer_test_builder.py

Run with: python test_model_layer_builder.py
No HuggingFace model downloads required (pure function tests).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import math

from model_layer_test_builder import (
    ModelDims,
    slice_dims_for_sim,
    quantize_to_mxfp,
    golden_ffn,
    HIDDEN_SLICE,
    INTER_SLICE,
)


def test_slice_dims_clips_to_limits():
    """slice_dims_for_sim clips hidden and inter to sim limits."""
    dims = ModelDims(
        hidden_size=576,
        inter_dim=1536,
        num_heads=9,
        num_kv_heads=3,
        head_dim=64,
        model_id="test",
    )
    sliced = slice_dims_for_sim(dims)
    assert sliced.hidden_size == HIDDEN_SLICE, f"Expected {HIDDEN_SLICE}, got {sliced.hidden_size}"
    assert sliced.inter_dim == INTER_SLICE, f"Expected {INTER_SLICE}, got {sliced.inter_dim}"
    # Non-dim fields preserved
    assert sliced.num_heads == dims.num_heads
    assert sliced.num_kv_heads == dims.num_kv_heads
    assert sliced.head_dim == dims.head_dim
    assert sliced.model_id == dims.model_id
    print("  PASS test_slice_dims_clips_to_limits")


def test_slice_dims_preserves_small_dims():
    """slice_dims_for_sim does not grow dims that are already small."""
    dims = ModelDims(
        hidden_size=64,
        inter_dim=64,
        num_heads=1,
        num_kv_heads=1,
        head_dim=64,
        model_id="test",
    )
    sliced = slice_dims_for_sim(dims)
    assert sliced.hidden_size == 64
    assert sliced.inter_dim == 64
    print("  PASS test_slice_dims_preserves_small_dims")


def test_quantize_to_mxfp_shape_preserved():
    """quantize_to_mxfp preserves tensor shape."""
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    xq = quantize_to_mxfp(x)
    assert xq.shape == x.shape, f"Shape mismatch: {xq.shape} != {x.shape}"
    print("  PASS test_quantize_to_mxfp_shape_preserved")


def test_quantize_to_mxfp_reduces_precision():
    """quantize_to_mxfp introduces quantization error vs float32 original."""
    torch.manual_seed(0)
    x = torch.randn(8, 128)
    xq = quantize_to_mxfp(x)
    diff = (x - xq).abs().max().item()
    # Should have non-zero error (quantization happened) but be small (MXFP8 is decent precision)
    assert diff > 0, "quantize_to_mxfp should not be lossless"
    assert diff < 1.0, f"quantize_to_mxfp error too large: {diff:.4f}"
    print(f"  PASS test_quantize_to_mxfp_reduces_precision  (max_diff={diff:.4f})")


def test_golden_ffn_output_shape():
    """golden_ffn returns correct output shape (batch, hidden)."""
    torch.manual_seed(42)
    batch, hidden, inter = 4, 128, 256
    X = torch.randn(batch, hidden)
    W_gate = torch.randn(hidden, inter)
    W_up = torch.randn(hidden, inter)
    W_down = torch.randn(inter, hidden)
    out = golden_ffn(X, W_gate, W_up, W_down)
    assert out.shape == (batch, hidden), f"Expected ({batch},{hidden}), got {out.shape}"
    print(f"  PASS test_golden_ffn_output_shape  (out={out.shape})")


def test_golden_ffn_dtype_bfloat16():
    """golden_ffn output is bfloat16 (matches VRAM storage convention)."""
    torch.manual_seed(1)
    batch, hidden, inter = 2, 64, 128
    X = torch.randn(batch, hidden)
    W_gate = torch.randn(hidden, inter)
    W_up = torch.randn(hidden, inter)
    W_down = torch.randn(inter, hidden)
    out = golden_ffn(X, W_gate, W_up, W_down)
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"
    print("  PASS test_golden_ffn_dtype_bfloat16")


def test_golden_ffn_deterministic():
    """golden_ffn is deterministic (same inputs → same output)."""
    torch.manual_seed(7)
    batch, hidden, inter = 4, 128, 256
    X = torch.randn(batch, hidden)
    W_gate = torch.randn(hidden, inter)
    W_up = torch.randn(hidden, inter)
    W_down = torch.randn(inter, hidden)
    out1 = golden_ffn(X, W_gate, W_up, W_down)
    out2 = golden_ffn(X, W_gate, W_up, W_down)
    assert torch.equal(out1, out2), "golden_ffn must be deterministic"
    print("  PASS test_golden_ffn_deterministic")


def test_golden_ffn_not_all_zeros():
    """golden_ffn output is non-trivial."""
    torch.manual_seed(3)
    batch, hidden, inter = 4, 128, 256
    X = torch.randn(batch, hidden)
    W_gate = torch.randn(hidden, inter)
    W_up = torch.randn(hidden, inter)
    W_down = torch.randn(inter, hidden)
    out = golden_ffn(X, W_gate, W_up, W_down)
    assert out.abs().max().item() > 0, "golden_ffn output must not be all zeros"
    print("  PASS test_golden_ffn_not_all_zeros")


if __name__ == "__main__":
    print("=" * 60)
    print("model_layer_test_builder unit tests")
    print("=" * 60)

    tests = [
        test_slice_dims_clips_to_limits,
        test_slice_dims_preserves_small_dims,
        test_quantize_to_mxfp_shape_preserved,
        test_quantize_to_mxfp_reduces_precision,
        test_golden_ffn_output_shape,
        test_golden_ffn_dtype_bfloat16,
        test_golden_ffn_deterministic,
        test_golden_ffn_not_all_zeros,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        import sys

        sys.exit(1)
    print("All unit tests PASSED")
