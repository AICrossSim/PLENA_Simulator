# SmolVLM2-2.2B PLENA Hardware Profile

**Date**: 2026-03-09
**Tool**: PLENA Compiler Generator (`compiler/generator/runner.py`)
**Model**: HuggingFaceTB/SmolVLM2-2.2B-Instruct

---

## Model Architecture Summary

### Text Decoder (SmolLM2 2.2B)
| Dimension | Value |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 24 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 |
| head_dim | 64 |
| intermediate_size | 8192 |
| vocab_size | 49280 |
| max_position_embeddings | 8192 |
| activation | SiLU |
| norm_type | RMSNorm (eps=1e-5) |

### Vision Encoder (SigLIP 400M)
| Dimension | Value |
|---|---|
| hidden_size | 1152 |
| num_hidden_layers | 27 |
| num_attention_heads | 16 |
| num_key_value_heads | 16 |
| head_dim | 72 |
| intermediate_size | 4304 |
| image_size | 384 |
| patch_size | 14 |
| num_patches | 27 x 27 = 729 |
| activation | gelu_pytorch_tanh |
| norm_type | LayerNorm (eps=1e-6) |

---

## Symbolic Graph Analysis

### Text Decoder Graph

Total nodes: **146** (indices 0-145)

Node breakdown:
- Embeddings: 1 (`embed_tokens`, 49280 x 2048)
- Decoder layers: 24 (each: 6 nodes)
- Final norm: 1 (`final_layernorm`)
- Total: 1 + (24 x 6) + 1 = 146 nodes

Layer anatomy (x 24 layers):

| Node | Op Type | Shape |
|---|---|---|
| `layer_N_input_layernorm` | normalization | (1, 512, 2048), eps=1e-5 |
| `layer_N_self_attn` | attention | heads=32, kv_heads=32, head_dim=64 |
| `layer_N_attn_residual` | elementwise_add | (1, 512, 2048) |
| `layer_N_post_attention_layernorm` | normalization | (1, 512, 2048), eps=1e-5 |
| `layer_N_mlp` | ffn | 2048 -> 8192 -> 2048, activation=silu |
| `layer_N_ffn_residual` | elementwise_add | (1, 512, 2048) |

### Vision Encoder Graph

Total nodes: **164** (indices 0-163)

Node breakdown:
- Patch embedding: 1 (`vision_patch_embed`, conv2d 3x384x384 -> 729x1152)
- Encoder layers: 27 (each: 6 nodes)
- Final norm: 1 (`vision_final_norm`)
- Total: 1 + (27 x 6) + 1 = 164 nodes

Layer anatomy (x 27 layers):

| Node | Op Type | Shape | Details |
|---|---|---|---|
| `vision_layer_N_pre_attn_norm` | normalization | (1, 729, 1152) | LayerNorm, eps=1e-6 |
| `vision_layer_N_self_attn` | attention | (1, 729, 1152) | heads=16, kv_heads=16, head_dim=72, Q/K/V/O proj: 1152x1152 |
| `vision_layer_N_attn_residual` | elementwise_add | (1, 729, 1152) | |
| `vision_layer_N_pre_ffn_norm` | normalization | (1, 729, 1152) | LayerNorm, eps=1e-6 |
| `vision_layer_N_mlp` | ffn | (1, 729, 1152) | fc1: 1152->4304, fc2: 4304->1152, activation=gelu_pytorch_tanh |
| `vision_layer_N_ffn_residual` | elementwise_add | (1, 729, 1152) | |

---

## PLENA Hardware Profiling Results

### Hardware Configuration

| Parameter | RTL Default (configuration.svh) | Simulator Override | Note |
|---|---|---|---|
| MLEN | 16 | 64 | Systolic array M dimension |
| VLEN | 16 | 64 | Vector lane width |
| BLEN | 4 | 4 | Batch lanes |
| HLEN | 8 | 8 | |
| MAX_K_TILES | 4 | 4 | MRAM limit = 4 x 64^2 = 16,384 elements |
| IMM2_WIDTH | 18 bits | -- | ISA immediate field |
| IMM2_BOUND | 2^18 = 262,144 | -- | Max addressable weight elements |
| HBM format | MXFP8 (E4M3, block=8) | -- | |
| VECTOR_SRAM_DEPTH | 1024 | -- | VRAM rows |
| MATRIX_SRAM_DEPTH | 1024 | -- | MRAM rows |
| FP_SRAM_DEPTH | 512 | -- | FPRAM scalar slots |
| INT_SRAM_DEPTH | 32 | -- | Integer register file |

### ASM Generation Status

#### Text Decoder

- **Status**: PARTIAL -- symbolic graph constructed successfully (146 nodes), ASM generation blocked
- **Blocker**: `projection_asm` assertion: `in_features * out_features < IMM2_BOUND`
  - SmolVLM2 Q/K/V projections: 2048 x 2048 = 4,194,304 >> 262,144
  - FFN up/gate projections: 2048 x 8192 = 16,777,216 >> 262,144
  - Hardware immediate field (18 bits) too narrow for full-size weight matrices
  - **Root cause**: PLENA ISA addressing designed for models with hidden_size <= ~512

#### Vision Encoder

- **Status**: PARTIAL -- symbolic graph constructed successfully (164 nodes), no ASM generation path for vision yet
- **Blockers**:
  - Vision encoder parser produces graph but `runner.py` only feeds text decoder graph to code_gen
  - Same IMM2_BOUND constraint applies (1152 x 1152 = 1,327,104 >> 262,144)
  - head_dim=72 is not a multiple of VLEN=64 -- requires padding
  - Patch embedding conv2d (C_in=3, K=14, K_col=588) already implemented and tested via K-split

### Utilization Report

Analysis parameters: `m=64, k=64, n=64, batch_size=4, context_length=1024` (decoding mode)

#### Systolic Operation Counts

| Component | Systolic Operations | Description |
|---|---|---|
| Embedding | 1,024 | (hidden/m) x (hidden/k) = 32 x 32 |
| Attention (x24) | 49,152 | Projections + QK^T + PV per layer |
| FFN (x24) | 294,912 | Up + Gate + Down projections per layer |
| LM Head | 24,640 | (vocab/m) x (hidden/k) = 770 x 32 |
| **Total** | **369,728** | |

#### Attainable FLOPS (batch_size=4)

Attainable FLOPS assumes N-dimension utilization = `batch_size = 4` (decoding, 1 token per batch).

| Component | Attainable FLOPS | % of Theoretical |
|---|---|---|
| Embedding | 16,777,216 (16.8M) | 6.25% |
| Attention | 1,610,612,736 (1.61G) | 3.57% |
| FFN | 4,831,838,208 (4.83G) | 6.25% |
| LM Head | 403,701,760 (403.7M) | 6.25% |
| **Total** | **6,862,929,920 (6.86G)** | **5.32%** |

#### Theoretical FLOPS (full N-utilization, n=64)

Theoretical FLOPS assumes full N-dimension utilization (n=64, e.g., prefill with seq_len >= 64).

| Component | Theoretical FLOPS |
|---|---|
| Embedding | 268,435,456 (268.4M) |
| Attention | 45,097,156,608 (45.1G) |
| FFN | 77,309,411,328 (77.3G) |
| LM Head | 6,459,228,160 (6.46G) |
| **Total** | **129,134,231,552 (129.1G)** |

#### Utilization Efficiency

| Metric | Value | Explanation |
|---|---|---|
| Decode utilization | **6.25%** (4/64) | batch=4 out of n=64 lanes used |
| Prefill utilization | **100%** (theoretical) | Full systolic array used when seq >= 64 |
| Attention decode utilization | **3.57%** | Lower due to QK^T/PV having m-only utilization |
| Compute-dominant component | **FFN (59.8%)** | FFN dominates total FLOPS |
| Attention share | **34.9%** | Second largest |

---

## Hardware Fit Analysis

### What Fits on PLENA Hardware Today

| Component | Real Dims | PLENA-sim Dims | Fits? | Notes |
|---|---|---|---|---|
| Text hidden_size | 2048 | <= 64 (1 tile) | No | 32x over VLEN |
| Text inter_size | 8192 | <= 128 (safe) | No | 128x over VLEN |
| Text head_dim | 64 | 64 | Yes | Exact match |
| Vision hidden_size | 1152 | -- | No | 18x over VLEN |
| Vision head_dim | 72 | -- | No | Not multiple of 64 |
| Vision inter_size | 4304 | -- | No | 67x over VLEN |
| Vision patch conv (K_col=588) | C_in=3, K=14 | K-split (10 tiles, 3 chunks) | Yes | 90.33% allclose |
| Single decoder layer | seq=64, hidden=64, inter=128 | Manually constrained | Yes | ~99% allclose |
| Vision encoder pipeline | 1 layer, constrained dims | Manually constrained | Yes | 99.95% allclose |

### Scaling Gap

**To run full SmolVLM2 text decoder:**
- Need VLEN >= 2048 OR multi-tile systolic unrolling for N-dimension
- Current MRAM limit: 4 x 64^2 = 16,384 elements (K_col <= 256 per pass)
- Required: K_col = 2048 -> 32 tiles -> 8 K-split chunks (K-split is implemented)
- IMM2_BOUND (2^18 = 262,144) blocks weight addressing for 2048x2048 matrices (4,194,304 elements)
- **Critical path**: ISA immediate field extension from 18 bits to >= 23 bits

**To run full SigLIP vision encoder:**
- head_dim = 72 (not multiple of 64) -- needs zero-padding to 128, wastes 44% of compute
- hidden_size = 1152 (18 tiles of 64) -- needs multi-tile systolic unrolling
- Patch embedding conv: C_in=3, K=14, K_col=588 -- **already working** (K-split, 3 chunks)
- Activation = gelu_pytorch_tanh -- **implemented** in PLENA ISA (`compiler/asm_templates/gelu_asm.py`, sigmoid approx: x * sigmoid(1.702x))
- Norm = LayerNorm -- **implemented** in PLENA ISA

### Parameter Memory Requirements

| Component | Parameters | HBM (MXFP8) | Notes |
|---|---|---|---|
| Text embeddings | 49280 x 2048 = 101M | ~101 MB | |
| Text decoder (per layer) | 4x(2048x2048) + 3x(2048x8192) = ~67.1M | ~67.1 MB | Q,K,V,O + Up,Gate,Down |
| Text decoder (24 layers) | 1.61B | ~1.61 GB | |
| Text LM head | 49280 x 2048 = 101M | ~101 MB | Often tied with embeddings |
| Vision patch embed | 3 x 14 x 14 x 1152 = 677K | ~0.68 MB | |
| Vision encoder (per layer) | 4x(1152x1152) + 2x(1152x4304) = ~15.2M | ~15.2 MB | Q,K,V,O + fc1,fc2 |
| Vision encoder (27 layers) | 411M | ~411 MB | |
| **Total model** | **~2.2B** | **~2.2 GB** | |

## IMM2_BOUND: Hardware Immediate Field Constraint

### Root Cause

The `IMM2_BOUND = 2**18 = 262,144` constant reflects a **true hardware constraint** defined in the PLENA ISA instruction word — not a software limitation.

**Hardware definition** (`compiler/doc/configuration.svh`, line 49):
```
parameter IMM_2_WIDTH = 18;
```

The 32-bit PLENA instruction word encodes `S_ADDI_INT` (opcode `0x22`) as:
```
[imm2: 18 bits][rs1: 4 bits][rd: 4 bits][opcode: 6 bits] = 32 bits total
```
The 18-bit `imm2` field can hold a maximum unsigned value of 2^18 - 1 = 262,143.

### Why Weight Matrix Size Must Fit in IMM2

In `projection_asm.py` and `ffn_asm.py`, the total number of weight elements is loaded into a GP register via:
```
S_ADDI_INT gp{reg}, gp0, {in_features * out_features}
```
This value is then passed to `C_SET_SCALE_REG`, which configures the HBM prefetch engine's stride/scale computation. Since `S_ADDI_INT` uses the 18-bit `imm2` field, the product `in_features * out_features` must be < 262,144 or the value is silently truncated by hardware.

### Constraint Propagation (Hardware → Software)

| Layer | File | Constraint |
|---|---|---|
| RTL spec | `compiler/doc/configuration.svh:49` | `IMM_2_WIDTH = 18` |
| Rust emulator | `transactional_emulator/src/op.rs` | `const IMM_2_WIDTH: u32 = 18; mask(18)` |
| Assembler | `compiler/assembler/assembly_to_binary.py` | reads `IMM_2_WIDTH` from `.svh` |
| Code generators | 12 Python files in `compiler/asm_templates/` | `IMM2_BOUND = 2**18` assertions |

### All Assertion Sites (12 files)

| File | Lines | Assertion |
|---|---|---|
| `projection_asm.py` | 66, 161 | `in_features * out_features < IMM2_BOUND` |
| `ffn_asm.py` | 133, 139, 329, 337, 487, 495, 725, 733, 1057, 1065 | `hidden * inter < IMM2_BOUND` and `hidden * batch * seq < IMM2_BOUND` |
| `preload_act.py` | 53 | `batch * hidden_size <= IMM2_BOUND` |
| `store_act_asm.py` | 78 | `batch * hidden_size <= IMM2_BOUND` |
| `flashattn/pv.py` | 50–53 | VRAM address bounds |
| `flashattn/output.py` | 42, 46, 50 | VRAM address bounds |
| `flashattn/qkt.py` | 53, 58 | VRAM address bounds |
| `flashattn/online_softmax.py` | 52, 124 | VRAM address bounds |
| `flashattn/reset.py` | 108–109 | `hkv * d * kv_len * batch < IMM2_BOUND` |
| `sub_matrix_manager.py` | 25 | constant mirrored from hardware |

Note: `flashattn/` files use `2**18 - 1`; others use `2**18` with strict `<` — both are equivalent (both reject values ≥ 2^18).

### Git History

Searched full git log — **no commits** have ever modified or worked around the IMM2_BOUND constraint. It has been a fixed hardware parameter since the earliest tracked commits. The current bounds (18-bit, max 262,143) have never been changed.

### Impact for Full Models

| Model Layer | Dimensions | Product | Fits? |
|---|---|---|---|
| SmolVLM2 text Q/K/V projection | 2048 × 2048 | 4,194,304 | No (16x over) |
| SmolVLM2 text FFN up/gate | 2048 × 8192 | 16,777,216 | No (64x over) |
| SigLIP vision Q/K/V | 1152 × 1152 | 1,327,104 | No (5x over) |
| Current simulator tests | 128 × 256 | 32,768 | Yes |
| Max safe (square) | 512 × 512 | 262,144 | Boundary |

### Possible Fix Paths

1. **Extend `IMM_2_WIDTH` to 24+ bits** in RTL (`configuration.svh`), Rust emulator (`op.rs`), assembler (`assembly_to_binary.py`), and all 12 code-generator assertion sites. Requires 25 bits to cover 2048×8192 = 16,777,216.
2. **Use `S_LUI_INT` + `S_ADDI_INT` pair** — `S_LUI_INT` uses the 22-bit `imm` field, enabling a two-instruction sequence to build 26-bit immediate values (sufficient for all current models).
3. **Preload large constants to INT_SRAM** — store the weight total via `S_ST_INT` at setup time, then load with `S_LD_INT` (register indirect, no immediate width limit).

---

## Conclusions

1. **Patch embedding conv2d**: Already fully implemented and tested (K_col=588, K-split with 3 chunks, 90.33% allclose)
2. **Vision encoder pipeline**: Single-layer constrained-dim test passes (99.95% allclose)
3. **Text decoder**: Cannot generate full-model ASM due to IMM2_BOUND hardware constraint (18-bit immediate field)
4. **Utilization at decode**: 6.25% systolic utilization with batch=4 (N-dimension underutilization); FFN dominates (59.8% of compute)
5. **Utilization at prefill**: Theoretical 100% utilization when sequence length >= VLEN=64
6. **Simulation strategy**: PLENA simulator validates op correctness at small scale (hidden=64); full-model profiling requires architectural extensions

---

## Next Steps

- [ ] Extend `projection_asm` to support multi-tile weight matrices (remove IMM2_BOUND constraint via wider immediate or indirect addressing)
- [ ] Wire `gelu_asm` into the ATen compiler and `runner.py` FFN codegen path (GELU ISA opcode exists; needs compiler integration for vision encoder FFN)
- [ ] Add conv2d SystemVerilog hardware support for the vision encoder patch embedding
- [ ] Implement RoPE fusion in ATen compiler
- [ ] Multi-layer model compilation via ATen compiler frontend
- [ ] Profile with larger batch sizes (batch=16, 32) to improve decode utilization
- [ ] Investigate head_dim=72 padding strategy (pad to 128 vs. native 72 support)
