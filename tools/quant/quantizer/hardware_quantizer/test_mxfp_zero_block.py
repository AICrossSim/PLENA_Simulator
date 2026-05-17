"""Regression tests for MXFP packer zero-block handling.

Background:
    _mx_fp_quantize_hardware previously crashed (via pack_fp_to_bin assert) when
    a quantization block contained all zeros AND the input dtype was fp16.
    Root cause:
        per_block_max = 0 + 1e-9        # 1e-9 underflows in fp16
        per_block_exponent_bias = clamp(floor(log2(0)), -128, 127) = -128
        px = px / 2**(-128)              # 2**(-128) underflows to 0 in fp16
                                         # -> 0/0 = NaN, propagates everywhere
        pack_fp_to_bin: assert NaN >= 0  # fires

    Fix: in _mx_fp_quantize_hardware, detect zero blocks (max == 0) and
    substitute per_block_max = 1.0 for those blocks before log2. Non-zero
    blocks still take per_block_max = max + 1e-9 (original behavior).

Goals of these tests:
    1. Zero blocks (fp16 and fp32) round-trip cleanly through quantize +
       pack_fp_to_bin without raising, and decode back to 0.
    2. Non-zero blocks still produce bit-identical output to a reference
       computed without the zero-block path (regression guard).
    3. Mixed tensors (some zero blocks + some non-zero blocks) handle both
       kinds correctly in a single call.

Run:
    PYTHONPATH=tools:.venv/lib/python3.13/site-packages \
        python test_mxfp_zero_block.py
    # or
    PYTHONPATH=... pytest test_mxfp_zero_block.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# When this script is executed directly, Python prepends its own directory to
# sys.path. That directory contains a `utils.py` module which shadows the
# `tools/utils/` package needed by quant.quantizer.* internals. Prepend the
# tools root explicitly so `utils` resolves to the package (not the local
# module).
_TOOLS_ROOT = Path(__file__).resolve().parents[3]
# Always prepend (don't gate on `not in sys.path`): even if PYTHONPATH already
# contains tools/, the script's own directory is at sys.path[0] and would
# shadow `utils` with the local hardware_quantizer/utils.py module.
sys.path.insert(0, str(_TOOLS_ROOT))

import torch

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from utils.torch_fp_conversion import pack_fp_to_bin


# Quantization config used by PLENA HBM_V_ACT_TYPE / HBM_M_WEIGHT_TYPE
# (BEHAVIOR.PRECISION in plena_settings.toml): MXFP-8 = E8M0 scale + E4M3 element.
ELEM_WIDTH = 8         # total bits per element (sign + exp + mant)
ELEM_EXP_WIDTH = 4
ELEM_MAN_WIDTH = 3     # = 8 - 4 - 1
SCALE_EXP_WIDTH = 8    # E8M0 shared exponent
BLOCK_SIZE = [1, 8]


def _quantize_and_pack(x: torch.Tensor):
    """Helper that runs the same pipeline as build_env.create_mem_for_sim."""
    bm_x, per_exp, per_mant, per_scale = _mx_fp_quantize_hardware(
        x,
        width=ELEM_WIDTH,
        exponent_width=ELEM_EXP_WIDTH,
        exponent_bias_width=SCALE_EXP_WIDTH,
        block_size=BLOCK_SIZE,
    )
    # rand_gen.quantize_tensor packs one block at a time
    n_blocks = per_exp.shape[0]
    bin_blocks = []
    for i in range(n_blocks):
        bin_block = pack_fp_to_bin(
            per_exp[i], per_mant[i], ELEM_EXP_WIDTH, ELEM_MAN_WIDTH
        )
        bin_blocks.append(bin_block.tolist())
    return bm_x, bin_blocks, per_scale.flatten().tolist()


# ============================================================================
# Test 1+2: zero blocks should not crash and should decode to 0
# ============================================================================


def test_zero_block_fp16():
    """A single all-zero block in fp16 must pack without raising."""
    x = torch.zeros(1, 8, dtype=torch.float16)
    bm_x, bins, scales = _quantize_and_pack(x)
    assert torch.all(bm_x == 0), f"fp16 zero block did not decode to 0: {bm_x}"
    assert bins == [[0] * 8], f"fp16 zero block bin pattern wrong: {bins}"
    # scale is the biased E8M0 exponent; any valid value is acceptable for a
    # zero block since mantissa is 0 (decoded value is 0 regardless).
    assert all(0 <= s <= 255 for s in scales), f"scale out of E8M0 range: {scales}"


def test_zero_block_fp32():
    """fp32 path already worked; verify still passes after patch."""
    x = torch.zeros(1, 8, dtype=torch.float32)
    bm_x, bins, scales = _quantize_and_pack(x)
    assert torch.all(bm_x == 0), f"fp32 zero block did not decode to 0: {bm_x}"
    assert bins == [[0] * 8], f"fp32 zero block bin pattern wrong: {bins}"


# ============================================================================
# Test 3: non-zero block behavior unchanged
# ============================================================================


def _expected_nonzero_block_via_original_path(x: torch.Tensor):
    """Reference implementation that mirrors the *original* (pre-patch)
    code path for the non-zero block case. We use this to assert that the
    new code returns bit-identical results when no zero block is present.
    """
    from quant.quantizer.hardware_quantizer.minifloat import (
        _minifloat_ieee_quantize_hardware,
    )
    from quant.quantizer.utils import my_clamp

    block_size = BLOCK_SIZE
    x_shape = x.shape
    pad_size_0 = (block_size[0] - (x_shape[-2] % block_size[0])) % block_size[0]
    pad_size_1 = (block_size[1] - (x_shape[-1] % block_size[1])) % block_size[1]
    import torch.nn.functional as f
    px = f.pad(x, (0, pad_size_1, 0, pad_size_0), "constant", 0)
    px_shape = px.shape
    px = px.view(
        -1, px_shape[-2] // block_size[0], block_size[0], px_shape[-1] // block_size[1], block_size[1]
    ).permute(0, 1, 3, 2, 4).reshape(-1, block_size[0] * block_size[1])

    # Original pre-patch line: NO zero-block check, just `+1e-9`
    per_block_max = px.abs().max(dim=-1, keepdim=True).values + 1e-9
    per_block_exponent_bias = my_clamp(
        torch.floor(torch.log2(per_block_max)),
        -(2 ** (SCALE_EXP_WIDTH - 1)),
        2 ** (SCALE_EXP_WIDTH - 1) - 1,
    )
    px = px / 2**per_block_exponent_bias
    _, per_exp, per_mant = _minifloat_ieee_quantize_hardware(
        px, width=ELEM_WIDTH, exponent_width=ELEM_EXP_WIDTH
    )
    bias_bias = 2 ** (SCALE_EXP_WIDTH - 1) - 1
    per_block_exponent_bias_biased = per_block_exponent_bias + bias_bias
    return per_exp, per_mant, per_block_exponent_bias_biased


def test_nonzero_block_unchanged():
    """Patched _mx_fp_quantize_hardware must be bit-identical to the
    pre-patch implementation on tensors without any zero blocks."""
    torch.manual_seed(0)
    # 4 non-zero blocks of size 8
    x = torch.randn(1, 32, dtype=torch.float32) * 2.5

    # Reference path
    ref_exp, ref_mant, ref_scale = _expected_nonzero_block_via_original_path(x)
    # Patched path
    _, new_exp, new_mant, new_scale = _mx_fp_quantize_hardware(
        x, width=ELEM_WIDTH, exponent_width=ELEM_EXP_WIDTH,
        exponent_bias_width=SCALE_EXP_WIDTH, block_size=BLOCK_SIZE,
    )
    assert torch.equal(new_exp, ref_exp), "per_exp diverged on non-zero blocks"
    assert torch.equal(new_mant, ref_mant), "per_mant diverged on non-zero blocks"
    assert torch.equal(new_scale, ref_scale), "per_scale diverged on non-zero blocks"


def test_nonzero_block_unchanged_fp16():
    """Same as above but fp16 input (verify dtype handling didn't regress)."""
    torch.manual_seed(1)
    x = (torch.randn(1, 32) * 2.0).to(torch.float16)
    ref_exp, ref_mant, ref_scale = _expected_nonzero_block_via_original_path(x)
    _, new_exp, new_mant, new_scale = _mx_fp_quantize_hardware(
        x, width=ELEM_WIDTH, exponent_width=ELEM_EXP_WIDTH,
        exponent_bias_width=SCALE_EXP_WIDTH, block_size=BLOCK_SIZE,
    )
    assert torch.equal(new_exp, ref_exp), "fp16 per_exp diverged on non-zero blocks"
    assert torch.equal(new_mant, ref_mant), "fp16 per_mant diverged on non-zero blocks"
    assert torch.equal(new_scale, ref_scale), "fp16 per_scale diverged on non-zero blocks"


# ============================================================================
# Test 4: mixed (zero + non-zero blocks in the same call)
# ============================================================================


def test_mixed_zero_and_nonzero_blocks():
    """Build a tensor with 3 blocks: [zeros, random_values, zeros].
    Each block of 8 elements must be handled correctly without crashing."""
    torch.manual_seed(2)
    block_zero = torch.zeros(8)
    block_nonzero = torch.randn(8) * 1.5
    x = torch.cat([block_zero, block_nonzero, block_zero]).unsqueeze(0)  # (1, 24)
    assert x.shape == (1, 24)

    for dtype, name in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
        bm_x, bins, scales = _quantize_and_pack(x.to(dtype))
        # _mx_fp_quantize_hardware returns shape (N, x.shape[-2], x.shape[-1])
        # for a (1, 24) input -> (1, 1, 24). Flatten the trailing batch dims
        # so we can compare per-element.
        bm_flat = bm_x.reshape(-1)
        assert bm_flat.numel() == 24, (
            f"{name}: unexpected bm_x size {bm_flat.numel()}, shape {bm_x.shape}"
        )
        # Block 0: all zero -> decoded all zero
        assert torch.all(bm_flat[0:8] == 0), f"{name}: block 0 not decoded to 0"
        # Block 1: non-zero -> decoded close to original (MXFP precision)
        decoded = bm_flat[8:16].to(torch.float32)
        original = block_nonzero
        # Relative tolerance loose because MXFP-8 has limited precision
        rel_err = (decoded - original).abs() / (original.abs() + 1e-6)
        assert rel_err.max() < 0.5, (
            f"{name}: block 1 decoded too far from original: "
            f"original={original.tolist()}, decoded={decoded.tolist()}"
        )
        # Block 2: all zero -> decoded all zero
        assert torch.all(bm_flat[16:24] == 0), f"{name}: block 2 not decoded to 0"
        # bin patterns: zero blocks should be all-zero
        assert bins[0] == [0] * 8, f"{name}: bin[0] not all-zero: {bins[0]}"
        assert bins[2] == [0] * 8, f"{name}: bin[2] not all-zero: {bins[2]}"


# ============================================================================
# Test 5: regression for the specific RoPE failure mode
#   sin_half[pos=0, :] = sin(0) = 0 across the whole head_dim/2 row
# ============================================================================


def test_rope_sin_position_zero_fp16():
    """Exactly the production failure shape: sin_half row at position 0
    (all zeros) in fp16, head_dim/2 = 64 -> 8 zero blocks of size 8."""
    sin_pos0 = torch.zeros(1, 64, dtype=torch.float16)
    bm_x, bins, scales = _quantize_and_pack(sin_pos0)
    assert torch.all(bm_x == 0), "RoPE sin[pos=0] regression: nonzero decode"
    assert all(b == [0] * 8 for b in bins), \
        f"RoPE sin[pos=0] regression: nonzero bins: {bins}"


# ============================================================================
# Direct-run entry point (so this file works without pytest installed)
# ============================================================================


def _run_all():
    tests = [
        test_zero_block_fp16,
        test_zero_block_fp32,
        test_nonzero_block_unchanged,
        test_nonzero_block_unchanged_fp16,
        test_mixed_zero_and_nonzero_blocks,
        test_rope_sin_position_zero_fp16,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed == 0:
        print(f"All {len(tests)} MXFP zero-block regression tests PASSED.")
        return 0
    print(f"{failed}/{len(tests)} tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(_run_all())
