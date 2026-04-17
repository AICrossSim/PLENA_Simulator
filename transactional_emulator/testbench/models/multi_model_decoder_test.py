"""
Multi-Model Real-Weight Decoder Pipeline Test

Validates the full PLENA decoder pipeline across multiple HuggingFace
model architectures by loading layer-0 weights and comparing against a
hardware-accurate golden reference.

Pipeline:
    embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

Each model's hidden_size and inter_dim are sliced to simulator limits
(hidden=64, inter=128).

Note on inter_dim=128 constraint:
    The FFN intermediate storage in VRAM overlaps with flash_attention's
    output O at inter=256.  inter=128 keeps gate_result max safely below O.

Note on LLaDA-8B:
    Uses architecture LlamaForMaskedDiffusion (trust_remote_code=True).
    Backbone is identical to LLaMA-3-8B; the bidirectional attention
    masking happens at runtime, not in the ISA — so the ISA path is
    identical to a standard LLaMA decoder.

Add new models by extending the MODELS list — no extra test files needed.
"""

from pathlib import Path

from transactional_emulator.testbench.model_layer_test_builder import build_and_run_decoder_test


# (model_id, asm_name, extra kwargs to build_and_run_decoder_test)
MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", "smollm2_135m_decoder", {}),
    (
        "GSAI-ML/LLaDA-8B-Instruct",
        "llada_8b_decoder",
        {"trust_remote_code": True, "partial_load": True},
    ),
]


if __name__ == "__main__":
    for model_id, asm_name, extra in MODELS:
        print(f"\n{'=' * 80}\nDecoder test: {model_id}\n{'=' * 80}")
        build_dir = Path(__file__).parent / "build" / asm_name
        build_and_run_decoder_test(
            model_id=model_id,
            asm_name=asm_name,
            build_dir=build_dir,
            layer_idx=0,
            seq_len=64,
            hidden_size=64,
            inter_dim=128,
            **extra,
        )
