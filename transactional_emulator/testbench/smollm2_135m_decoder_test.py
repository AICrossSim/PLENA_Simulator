"""
SmolLM2-135M Single-Layer Decoder Pipeline Test — Real Model Weights

Full single-layer decoder pipeline with real HuggingFaceTB/SmolLM2-135M weights:
    embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

K and V are precomputed from real W_k, W_v weights applied to random context.
FFN uses real sliced weights (hidden=64, inter=128).

Parameters:
    seq_len     = 64  (= mlen)
    hidden_size = 64  (= head_dim = mlen, required for flash_attention)
    head_dim    = 64
    inter_dim   = 128 (sliced from 1536; 256 causes VRAM conflict with flash_attn O)
    mlen        = 64
    blen        = 4

Note on inter_dim constraint:
    The FFN intermediate storage occupies VRAM addresses:
      up_result  = batch * hidden = 4096
      gate_result = 4096 + inter * batch = 4096 + inter * 64
    The flash_attention output O sits at VRAM row 449 (addr=28736).
    With inter=256: gate_result reaches 20480 + 2*4096 = 28672, overlapping O.
    With inter=128: gate_result max = 20539, safely below O at 28736.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_layer_test_builder import build_and_run_decoder_test

if __name__ == "__main__":
    build_dir = Path(__file__).parent / "build"
    build_and_run_decoder_test(
        model_id="HuggingFaceTB/SmolLM2-135M",
        asm_name="smollm2_135m_decoder",
        build_dir=build_dir,
        layer_idx=0,
        seq_len=64,
        hidden_size=64,
        inter_dim=128,
    )
