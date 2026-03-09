# SmolVLM2-256M PLENA Hardware Profile

Generated from PLENA ISA simulator. Hardware constants: MLEN=VLEN=64, BLEN=4.
All cycle estimates use PLENA cost model (S_*=1, V_*=64, M_MM=64, H_PREFETCH=64).

---

## 1. Vision Encoder Pipeline (Conv2d + 1 Decoder Layer)

Single forward pass: conv2d patch embed + 1 transformer layer.

| Section | Instructions | % | Est. Cycles | % |
|---|---|---|---|---|
| conv2d (patch embed) | 22,751 | 83.1% | 603,737 | 86.4% |
| flash_attention | 2,933 | 10.7% | 67,571 | 9.7% |
| rms_norm (×2) | 1,544 | 5.6% | 25,736 | 3.7% |
| ffn | 118 | 0.4% | 1,189 | 0.2% |
| other (data loading, embedding) | 31 | 0.1% | 157 | 0.0% |
| **TOTAL** | **27,377** | | **698,390** | |

**Key finding**: conv2d patch embedding dominates vision encoder at 83% instructions / 86% cycles.

---

## 2. LM Decoder — Single Layer (SmolLM2-135M backbone)

Parameters: seq_len=64, hidden_size=64, inter_dim=128, head_dim=64.

| Section | Instructions | % | Est. Cycles | % |
|---|---|---|---|---|
| flash_attention | 2,951 | 56.9% | 67,715 | 62.8% |
| rms_norm_2 (post-FFN) | 772 | 14.9% | 12,868 | 11.9% |
| rms_norm_1 (pre-attn) | 772 | 14.9% | 12,868 | 11.9% |
| rope | 449 | 8.7% | 12,545 | 11.6% |
| ffn | 118 | 2.3% | 1,189 | 1.1% |
| data_loading | 110 | 2.1% | 425 | 0.4% |
| embedding_add | 18 | 0.3% | 144 | 0.1% |
| **TOTAL** | **5,190** | | **107,754** | |

---

## 3. LM Decoder — 30 Layers (Full SmolVLM2-256M Text Model)

30 × single-layer, representing one complete autoregressive decode step.

| Section | Instructions | % | Est. Cycles | % |
|---|---|---|---|---|
| flash_attention | 88,530 | 56.9% | 2,031,450 | 62.8% |
| rms_norm_1 (pre-attn) | 23,160 | 14.9% | 386,040 | 11.9% |
| rms_norm_2 (post-FFN) | 23,160 | 14.9% | 386,040 | 11.9% |
| rope | 13,470 | 8.7% | 376,350 | 11.6% |
| ffn | 3,540 | 2.3% | 35,670 | 1.1% |
| data_loading | 3,300 | 2.1% | 12,750 | 0.4% |
| embedding_add | 540 | 0.3% | 4,320 | 0.1% |
| **TOTAL** | **155,700** | | **3,232,620** | |

Instruction type breakdown:

| Type | Count | % | Description |
|---|---|---|---|
| S_* (scalar) | 104,580 | 67.2% | Address calc, loop control |
| V_* (vector) | 32,880 | 21.1% | Elementwise ops (norm, rope, silu) |
| M_* (matmul) | 15,630 | 10.0% | Systolic array (QK, PV, linear) |
| C_* (control) | 2,280 | 1.5% | Branches, config |
| H_* (HBM) | 330 | 0.2% | Memory prefetch |

---

## 4. Full Model Cost Breakdown (SmolVLM2-256M)

Architecture: 1 conv2d patch embed + 12 SigLIP vision encoder layers + 30 LM decoder layers.

| Component | Count | Est. Cycles | % of Total |
|---|---|---|---|
| Conv2d patch embed | 1× | 603,737 | 12% |
| Vision encoder transformer layers | 12× | 1,135,836 | 23% |
| Text LM decoder layers | 30× | 3,232,620 | 65% |
| **TOTAL (per decode step)** | | **4,972,193** | |

*Vision encoder cost per layer: ~94,653 cycles (4,626 instructions)*

---

## 5. Vision Encoder vs LM Decoder — Proportion by Output Tokens

Single image. Vision encoder runs once; LM decoder runs once per output token.
Vision encoder total cost: 1,739,573 cycles (conv2d + 12 transformer layers).
LM decoder cost per token: 3,232,620 cycles (30 layers).

| Output tokens | Vision enc (cycles) | LM decoder (cycles) | Vision % | Decoder % |
|---|---|---|---|---|
| 1 | 1,739,573 | 3,232,620 | 35% | 65% |
| 5 | 1,739,573 | 16,163,100 | 10% | 90% |
| 10 | 1,739,573 | 32,326,200 | 5% | 95% |
| 15 | 1,739,573 | 48,489,300 | 3.5% | 96.5% |
| 20 | 1,739,573 | 64,652,400 | 2.6% | 97.4% |
| 30 | 1,739,573 | 96,978,600 | 1.8% | 98.2% |
| 50 | 1,739,573 | 161,631,000 | 1.1% | 98.9% |
| 100 | 1,739,573 | 323,262,000 | 0.5% | 99.5% |

**Breakeven**: vision encoder ≈ 0.54 output tokens of decode cost.

---

## 6. VQA Workload Analysis (15–30 Output Tokens)

Typical VQA output length: 15–30 tokens (short answer to open-ended description).

| Tokens | Flash attn cycles | Vision enc cycles | Flash attn / Vision enc |
|---|---|---|---|
| 15 | 30,471,750 | 1,739,573 | 17.5× |
| 20 | 40,629,000 | 1,739,573 | 23.4× |
| 30 | 60,943,500 | 1,739,573 | 35.0× |

At 20 output tokens (typical VQA), flash attention alone costs **23× more** than the entire vision encoder pass.

### Decoder Bottleneck Priority (20-token VQA workload)

| Op | % of total end-to-end cycles |
|---|---|
| Flash attention (decoder) | ~62% |
| RMS norm (decoder, ×2 per layer) | ~15% |
| RoPE (decoder) | ~12% |
| FFN/SwiGLU (decoder) | ~1% |
| Vision encoder (all) | ~2.6% |
| Conv2d patch embed | ~0.9% |

---

## 7. Multi-Frame Video Analysis

Vision encoder re-runs every frame; LM decoder runs once per output token.
Assuming 100 output tokens for captioning.

| Frames | Vision enc total | LM decoder (100 tok) | Vision % |
|---|---|---|---|
| 1 | 1.74M | 323M | 0.5% |
| 16 | 27.8M | 323M | 8% |
| 64 | 111M | 323M | 26% |

Conv2d hardware acceleration becomes meaningful only at 64+ frames.

---

## Methodology Notes

- Profiled using `analytic_models/roofline/asm_profiler.py` on PLENA ISA-generated assembly
- Single-layer measurements from actual emulator-verified tests (98-100% allclose vs PyTorch golden)
- Multi-layer (30×) computed by tiling single-layer ASM via `smolvlm2_multilayer_decoder_profile.py`
- Vision encoder layer count (12) inferred from SigLIP-B/16 base config (hidden=768, heads=12)
- Hardware parameters scaled to sim limits: hidden=64, inter=128 (vs real hidden=576, inter=1536)
- Cycle model is instruction-count based; does not model memory bandwidth or pipeline stalls
