"""
LLaDA-8B Single-Layer Decoder Pipeline Test — Real Model Weights

LLaDA uses a LLaMA-3-8B backbone with bidirectional (non-causal) attention
for masked diffusion. The PLENA flash_attention op already has no causal mask,
so LLaDA runs through the standard decoder pipeline unchanged.

Full single-layer decoder pipeline with real GSAI-ML/LLaDA-8B-Instruct weights:
    embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

Parameters:
    seq_len     = 64  (= mlen)
    hidden_size = 64  (= head_dim = mlen, required for flash_attention)
    head_dim    = 64
    inter_dim   = 128 (sliced from 14336; 256 causes VRAM conflict with flash_attn O)
    mlen        = 64
    blen        = 4

Note on LLaDA:
    LLaDA-8B uses architecture LlamaForMaskedDiffusion (trust_remote_code=True).
    The backbone is identical to LLaMA-3-8B — same layer structure, same weight names.
    Inference: T denoising steps x full-sequence prefill (no autoregressive decode).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_layer_test_builder import build_and_run_decoder_test

if __name__ == "__main__":
    build_dir = Path(__file__).parent / "build"
    build_and_run_decoder_test(
        model_id           = "GSAI-ML/LLaDA-8B-Instruct",
        asm_name           = "llada_8b_decoder",
        build_dir          = build_dir,
        layer_idx          = 0,
        seq_len            = 64,
        hidden_size        = 64,
        inter_dim          = 128,
        trust_remote_code  = True,
        partial_load       = True,
    )
