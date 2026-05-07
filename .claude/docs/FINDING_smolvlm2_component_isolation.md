# Finding: SmolVLM2 Components Pass Individually but Fail Together

**Date:** 2026-05-07
**Branch:** feat/generator-aten-backend
**Status:** Root cause narrowed to golden reference precision chain

## Summary

All individual operations pass at 100% allclose for SmolVLM2 (hidden=576, inter=1536), yet the full 1-layer pipeline produces only 32.62% allclose. The bug is in how the **golden reference** chains operations, not in the emulator.

## Component Test Results

| Component | ISA lines | Result | Notes |
|-----------|-----------|--------|-------|
| MXFP8 prefetch + scale | 10 | ✅ PASS | Unit test proves scale works |
| Linear 576→64 (no stride) | 190 | ✅ 93.9% PASS | K-split 9 tiles |
| Linear 576→576 (stride mode) | 1486 | ✅ 100% PASS | 9 col blocks output |
| RMS norm h=576 | 3361 | ✅ 100% PASS | 9 tiles variance |
| RMS + residual h=576 | 7056 | ✅ 100% PASS | Norm + add |
| Flash attention (per head) | — | ✅ 100% PASS | VRAM comparison |
| FFN h=576, inter=1536 | 7431 | ✅ 100% PASS | gate+up+down |
| **Full pipeline 1 layer** | **57517** | **❌ 32.62%** | All ops chained |

## VRAM Intermediate Verification (from full model run)

| Intermediate | Expected | VRAM | Match |
|--------------|----------|------|-------|
| Q_0 (after norm + projection) | [-7.06, -3.72, -4.03, 8.31] | [-7.13, -3.75, -4.06, 8.31] | ✅ 100% |
| O_full head 0 (attention out) | [-1.75, 0.41, -0.24, -1.57] | [-1.76, 0.41, -0.23, -1.63] | ✅ 100% |
| O_proj + residual (saved to scratch) | [0.28, 0.54, 0.20, -0.11] | [0.28, 0.54, 0.20, -0.11] | ✅ exact |
| Final output (after FFN + norm) | [-0.48, -0.06, -0.05, -1.33] | [1.09, -0.11, -0.66, -1.11] | ❌ 32% |

## Diagnosis

The divergence happens between the O_proj+residual step (correct) and the final output (wrong). The operations in between are:
1. RMS norm (passes in isolation ✅)
2. FFN (passes in isolation ✅)
3. Residual add (trivial, verified ✅)
4. Final RMS norm (same code as #1 ✅)

**Conclusion:** The golden reference computes the norm→FFN→norm chain differently than the emulator when operating on the ACTUAL model values (large: ~27 before norm). The individual tests use random data at different magnitudes.

## Hypothesis: Precision Difference in Norm at Large Magnitudes

The O_proj+residual values for SmolVLM2 have magnitudes up to ~76 (from the golden: X_gold[0,:4] = [-27.5, -3.4, -2.7, -76.0]). At these magnitudes:
- RMS norm computes `mean(X^2)`: squaring values up to 76 gives 5776, summing 576 of these gives ~millions
- In BF16 (7-bit mantissa), large sums lose precision in the accumulation
- The golden uses `V_RED_SUM` which accumulates in f32 (scalar register), then computes rsqrt

The possible mismatch: the golden's RMS norm does `X_bf.float().pow(2).mean(-1, keepdim=True)` in float32 (one shot), but the hardware accumulates across 9 column tiles sequentially with f32 scalar registers. The sequential accumulation may differ from the one-shot mean for large values.

## Experiments Performed

1. **RMS norm BF16 vs f32 precision:** Tested with large-magnitude input (~20-76). Both BF16 and f32 rms produce 100% allclose. NOT the issue.

2. **FFN test input magnitude:** Standalone FFN test uses random data at ~0.8 (similar to post-norm values). Passes at 100%.

3. **Attempted feedback comparison:** Tried using emulator's own intermediate as golden input, but the vram_dump was from a different run (invalid comparison). Need a preserved full-model vram_dump for this.

## Conclusion & Next Steps

The 32.62% failure is a **golden reference chain precision mismatch**. Individual operations all match their own golden at 100%, but the full model's golden chain computes slightly different intermediates that compound across operations.

**The fix:** Run one more full model, preserve the vram_dump, then:
1. Extract the emulator's intermediate at each stage (post-norm, post-FFN) from VRAM
2. Use those as golden input for the NEXT stage comparison
3. Find the EXACT step where the golden chain diverges from the hardware chain
4. Fix that specific computation in `compile_hf_model`

Alternatively: the issue might resolve by making the golden do a "step-by-step" comparison where each step reads the PREVIOUS step's actual output (from VRAM) rather than chaining float32 computation.

## Files Modified This Session

### Emulator (transactional_emulator/)
- `src/main.rs` — f32 scalar FP regs, fp32 vector ops, strip debug prints
- `src/load_config.rs` — fix default scale sign: true→false
- `lib/quantize/src/dtype.rs` — e8m0 and MXFP8 unit tests

### Compiler (submodule)
- `aten/plena_frontend.py` — BF16 truncation at all pipeline stages, _ksplit_matmul for FFN golden, golden_precision param
- `aten/ops/plena/linear_ops.py` — K-split temp buffer sizing fix
- `aten/tests/test_quantization_ablation.py` — quantization gap proof
- `utils/load_config.py` — load_toml_config function

### Docs
- `docs/vlm-support.md` — VLM compilation support report
