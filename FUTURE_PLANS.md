# PLENA Simulator — Future Development Plans

Priority order based on profiling data (see `smolvlm2_profile.md`).

---

## High Priority (Bottleneck-Driven)

### 1. Flash Attention Optimization
Flash attention = 63% of all decode cycles (30-layer decoder).
- KV-cache support for autoregressive decode (currently only prefill)
- Better Q-block tiling (currently seq_len must equal mlen=64)
- Bandwidth-aware scheduling for K/V loads from HBM
- Multi-head support (currently head_dim=64=mlen, single head)

### 2. RMS Norm Performance
RMS norm = 24% of decoder cycles (surprisingly high).
- Profile whether it is compute-bound or memory-bandwidth-bound
- Investigate fused rms_norm + linear (pre-norm fusion)
- Check if vector hardware is fully utilized (currently 14.9% of instructions are scalar)

### 3. RoPE
RoPE = 12% of decoder cycles.
- Currently implemented as vector elementwise ops
- Potential for fused rope + linear projection

---

## Medium Priority (Feature Completeness)

### 4. ATen Compiler → Flash Attention
The ATen compiler (`plena/compiler/aten_compiler.py`) handles:
- ✅ linear / mm
- ✅ rms_norm / layer_norm
- ✅ FFN fusion (linear→silu→mul→linear)
- ❌ flash_attention / SDPA (not yet)
- ❌ RoPE (not yet)
- ❌ embedding_add (not yet)

Adding SDPA would enable full decoder compilation via `torch.export`.

### 5. Real-Scale Decoder
Current sim caps: hidden=64 (mlen limit), inter=128 (VRAM conflict).
Real SmolVLM2-256M: hidden=576, inter=1536.
- Stride mode (`use_stride_mode=True`) should allow hidden>64
- Need VRAM layout rework for larger activations
- K-split already handles large weight matrices

### 6. Multi-Layer ATen Compiler
Currently ATen compiler handles single ops/single layer.
- Compile full N-layer transformer end-to-end from `torch.export`
- Layer-loop codegen (vs N copies of single-layer ASM)
- KV-cache memory planning across layers

---

## Low Priority

### 7. Conv2d Hardware / Optimization
Conv2d = ~1% of VQA end-to-end cost (breakeven at ~0.54 output tokens).
Only worth revisiting for:
- Video inference (64+ frames per query)
- Very short generation (≤5 output tokens, e.g. classification)
Current implementation: im2col + systolic matmul, K-split for K_col > 256.

### 8. FFN at Real Dims
At sim dims (hidden=64, inter=128), FFN = 1.1% of decoder cycles.
At real dims (hidden=576, inter=1536), FFN contribution will be larger.
Revisit after real-scale decoder is working.

---

## Paper Benchmarks (Next Immediate Task)

See `smolvlm2_profile.md` for all profiling tables.

Remaining benchmark work:
- [ ] Clean up all op test outputs (consistent formatting, pass rates)
- [ ] Roofline analysis at real hardware params (MLEN=512, BLEN=32, VLEN=512 from plena_settings.toml)
- [ ] Utilization model results for SmolVLM2 and Llama-3.1-8B
- [ ] ASM profile for vision encoder + decoder combined (full image pass)
- [ ] Accuracy tables: allclose % for each op test at sim dims
