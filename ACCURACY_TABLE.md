# PLENA Simulator Accuracy Table

**Match Rate** indicates the percentage of output elements within tolerance (atol=0.2, rtol=0.2) when comparing PLENA simulator outputs against PyTorch/NumPy reference implementations.

## Primitive Operators

Synthetic weights with simulator dimensions (hidden=64, seq=64):

| Operator | Configuration | Match Rate |
|----------|---------------|------------|
| softmax | — | 100.00% |
| linear | — | 93.03% |
| rms_norm | — | 100.00% |
| layer_norm | — | 100.00% |
| ffn | — | 100.00% |
| flash_attention | — | 100.00% |
| conv2d | K_col=64 (baseline) | 95.58% |
| conv2d (tiled) | K_col=128 (2 tiles) | 93.07% |
| conv2d (SigLIP approx) | K_col=192 (3 tiles) | 91.24% |
| conv2d (SigLIP real kernel) | K=14, K_col=588 (10 tiles, 3 K-split chunks) | 90.33% |
| embedding_add | — | 100.00% |
| rope | — | 100.00% |

## Real-Model End-to-End

Real HuggingFace weights, sliced to simulator dimensions:

| Model | Task | Configuration | Match Rate |
|-------|------|---------------|------------|
| SmolLM2-135M | FFN | hidden=128, inter=256 | 100.00% |
| clm-60m | FFN | hidden=128, inter=256 | 100.00% |
| SmolLM2-135M | Decoder pipeline | seq=64, hidden=64, inter=128 | 98.97% |
| SmolVLM2-256M | Vision encoder pipeline | conv2d + 1 decoder layer | 99.95% |

## ATen Compiler

Automatic PLENA ISA generation from `torch.export`:

| Operation | Match Rate |
|-----------|------------|
| linear | 100.00% |
| decoder (embedding + rms_norm + rope + flash_attn + ffn) | 99.05% |

---

## Notes

**Tolerance and Quantization**: All results use atol=0.2, rtol=0.2 tolerance. This accounts for MXFP8 quantization applied to HBM-stored weights (k, v tensors in flash attention, linear layer weights). Activation tensors in VRAM remain in full precision.

**Conv2d Accuracy Scaling**: Convolution accuracy decreases with K_col (number of output channels per kernel) due to accumulated MXFP8 quantization errors across K-split chunks when K_col exceeds 256 (hardware MRAM tile limit). For example, the SigLIP real kernel test (K_col=588, split into 3 chunks across 10 tiles) shows 90.33% match rate compared to 95.58% for the baseline (K_col=64).
