# PLENA VLM Support — SmolVLM2

This document describes Vision-Language Model (VLM) support in PLENA, covering
the two compilation pipelines, the new ops and ASM templates added for vision
encoders, and how to run every relevant test.

## 1. Compilation Pipelines

PLENA has two routes from a HuggingFace model to ISA:

### Pipeline 1 — ATen (recommended)

```
HF model (nn.Module) --> PlenaCompiler + ops.* --> ISA --> Rust emulator --> golden comparison
```

- Entry: `compiler/generator/runner.py` (`aten` mode) or `compiler/aten/plena_frontend.py`
- Walks the actual `nn.Module` tree, extracts real weight tensors
- Generates ISA by calling `PlenaCompiler` methods and `ops.*` dispatch
- Supports native model dimensions (hidden=384, 576, etc.)
- Immediate numerical verification: three-way comparison of HF float32 vs
  golden (MXFP8+BF16) vs emulator output
- Current accuracy: 98-100% allclose on text decoders, 99.95% on vision encoder

### Pipeline 2 — Generator (structural)

```
HF config --> LLMModelParser --> symbolic graph --> scheduler --> code_gen --> ASM
```

- Entry: `compiler/generator/runner.py` (`codegen` mode)
- Reads `AutoConfig`, builds a JSON symbolic graph, schedules VRAM/HBM, emits ASM
- Fixed to simulator dimensions (hidden=64 default)
- Numerical verification deferred
- Supports vision graph prepending for VLM models

### Comparison

| Aspect | Generator (Pipeline 2) | ATen (Pipeline 1) |
|--------|----------------------|-------------------|
| Input | HF config (AutoConfig) | HF model (nn.Module with weights) |
| Dimensions | Fixed sim-scale (hidden=64) | Native model dimensions |
| Numerical verification | Deferred | Immediate, 98-100% allclose |
| Memory management | Static JSON library files | Dynamic VirtualMemoryManager |
| Multi-head attention | Fused flash_attn_asm node | Per-head loop with on-chip K/V + RoPE |
| VLM vision support | Symbolic graph nodes | conv2d ops, emulator-verified |

## 2. Tests and Scripts

### Unit Tests (no HF download, seconds)

| Test | Command | Covers |
|------|---------|--------|
| VLM parser | `cd compiler && PYTHONPATH=. pytest generator/tests/test_vlm_parser.py -v` | Config extraction, text+vision dims, MQA, symbolic graphs |
| VLM codegen | `cd compiler && PYTHONPATH=.:.. pytest generator/tests/test_vlm_code_gen.py -v` | No M_MM_VV, vision nodes present, assembles to binary |
| PlenaCompiler | `cd compiler && PYTHONPATH=.:tools python3 aten/tests/test_plena_compiler.py` | VRAM ops, large immediate fix, alloc |

### Integration Tests (require HF download)

| Test | Command | Covers |
|------|---------|--------|
| Vision encoder e2e | `bash run.sh test-vision-encoder-smolvlm2` | Conv2d patch embed + ViT + FFN with real weights, emulator verified (99.95% allclose) |
| 30-layer decoder profile | `bash run.sh multilayer-decoder-profile smolvlm2` | Full decoder ISA (160,920 lines) + profiling |
| ATen text e2e | `bash run.sh test-generator-aten AICrossSim/clm-60m 64 22` | 22-layer text decoder through ATen path |

### Conv2d Tests

| Test | Command |
|------|---------|
| Basic im2col | `bash run.sh test-conv2d` |
| Multi-tile K=4 | `bash run.sh test-conv2d-tiled` |
| SigLIP K=8 | `bash run.sh test-conv2d-siglip` |
| SigLIP K=14 K-split | `bash run.sh test-conv2d-siglip-real` |

### Other Operator Tests

| Test | Command |
|------|---------|
| Softmax | `bash run.sh test-softmax` |
| Linear | `bash run.sh test-linear` |
| RMS norm | `bash run.sh test-rms-norm` |
| Layer norm | `bash run.sh test-layer-norm` |
| FFN | `bash run.sh test-ffn` |
| Flash attention | `bash run.sh test-flash-attention` |
| Embedding add | `bash run.sh test-embedding-add` |
| RoPE | `bash run.sh test-rope` |

## 3. New Ops and ASM Templates for Vision

### ASM Templates (`compiler/asm_templates/`)

| Template | File | Purpose |
|----------|------|---------|
| im2col (shift) | `im2col_asm.py` | On-chip im2col using `V_SHIFT_V`. Requires 64-aligned pixel columns. |
| im2col (no shift) | `im2col_asm_no_shift.py` | Fallback using basis-vector extraction. Handles non-aligned columns. |
| GELU | `gelu_asm.py` | `x * sigmoid(1.702 * x)` approximation for ViT FFN. |
| LayerNorm | `normalization_asm.py` | Mean-subtract + variance-normalize. SigLIP/ViT uses this vs RMSNorm. |

### ATen Ops (`compiler/aten/ops/`, registered in `native_ops.yaml`)

| Op | Backend | Purpose |
|----|---------|---------|
| conv2d | `plena/conv_ops.py` | im2col + linear matmul. `CONV_USE_SHIFT` env var toggles shift variant. |
| embedding_add | `plena/embedding_ops.py` | Learned positional embedding: `input += pos_weight`. SigLIP vision encoder. |
| rope | `plena/embedding_ops.py` | RoPE: `x = x*cos + rotate_half(x)*sin`. Text decoder. |
| layer_norm | `plena/norm_ops.py` | LayerNorm for ViT layers. |

### Symbolic Graph Nodes (Generator path, `llm_parser.py`)

| Node | Purpose |
|------|---------|
| conv2d | Patch embedding with in/out channels, kernel, stride, num_patches |
| vision_projection | Pixel-shuffle + linear connector (vision to text hidden dim) |
| Bidirectional attention | `causal_mask: False` for SigLIP/ViT self-attention |
| ViT FFN | `arch: "vit"` triggers fc1 + GELU + fc2 instead of gated gate/up/down |

### Code Gen Extensions (`compiler/generator/passes/code_gen.py`)

- `_generate_conv2d_code()` — im2col + projection, auto-selects shift vs no-shift
- `_generate_vision_projection_code()` — pixel-shuffle (zero-cost reshape) + linear
- Extended attention — bidirectional when `causal_mask=False`, disables RoPE for vision
- Extended FFN — `arch: "vit"` dispatches to fc1 + GELU + fc2
- Extended normalization — `layer_norm_asm` for vision, `rms_norm_asm` for text

## 4. SmolVLM2 Architecture

```
SmolVLM2-256M
├── Vision Encoder (SigLIP)
│   ├── Conv2d patch embed (image_size=512, patch_size=16)
│   ├── Learned positional embedding (embedding_add)
│   ├── 12x ViT layers
│   │   ├── LayerNorm
│   │   ├── Bidirectional self-attention (12 heads, head_dim=64)
│   │   ├── LayerNorm
│   │   └── FFN (fc1 → GELU → fc2, inter=3072)
│   └── Vision-to-text connector (pixel-shuffle + linear projection)
└── Text Decoder (Llama-style)
    ├── Token embedding
    ├── 30x decoder layers
    │   ├── RMSNorm
    │   ├── Causal GQA (9 heads, 3 KV heads, head_dim=64)
    │   ├── RMSNorm
    │   └── Gated FFN (gate/up/down, inter=1536, SiLU)
    └── LM head
```

## 5. Current Accuracy Results

| Model | Scope | Pipeline | Allclose |
|-------|-------|----------|----------|
| clm-60m | 5-layer text decoder | ATen | 100% |
| SmolVLM2-256M | Vision encoder (conv2d + ViT + FFN) | ATen | 99.95% |
| SmolVLM2-256M | 30-layer decoder ISA | Generator | Generated (160K lines) |
| clm-60m | 22-layer text decoder | ATen | Running |

## 6. Key Files

```
compiler/
├── aten/
│   ├── plena_frontend.py          # Native-dim ATen frontend
│   ├── plena_compiler.py          # ISA emitter + virtual memory manager
│   ├── native_ops.yaml            # Op registry (9 ops)
│   └── ops/plena/
│       ├── conv_ops.py            # conv2d
│       ├── embedding_ops.py       # embedding_add, rope
│       └── norm_ops.py            # layer_norm, rms_norm
├── asm_templates/
│   ├── im2col_asm.py              # V_SHIFT_V im2col
│   ├── im2col_asm_no_shift.py     # Basis-vector im2col
│   ├── gelu_asm.py                # GELU activation
│   └── normalization_asm.py       # LayerNorm + RMSNorm
├── generator/
│   ├── runner.py                  # Unified CLI (codegen/aten modes)
│   ├── aten_runner.py             # ATen e2e runner
│   ├── parser/llm_parser.py       # VLM-aware parser with vision graph
│   ├── passes/code_gen.py         # Conv2d, vision_projection, ViT FFN dispatch
│   └── tests/
│       ├── test_vlm_parser.py     # 3 SmolVLM2 parser tests
│       └── test_vlm_code_gen.py   # 3 SmolVLM2 codegen tests
└── doc/Model_Lib/
    ├── smolvlm2-2.2b-text.json    # SmolVLM2 text decoder config
    └── smolvlm2-2.2b-vision.json  # SmolVLM2 vision encoder config

transactional_emulator/testbench/
├── conv/
│   └── smolvlm2_vision_encoder_test.py  # Vision e2e with real weights
└── models/
    └── multi_model_multilayer_decoder_profile.py  # 30-layer profiling
```
