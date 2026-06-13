"""
Multi-Model Real-Weight Decoder Pipeline Test

Validates the simulator-sliced PLENA decoder layer pipeline across multiple
HuggingFace model architectures by loading layer-0 weights and comparing
against a hardware-accurate golden reference.

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

import argparse
from pathlib import Path

from transactional_emulator.testbench.sliced_layer_test_builder import build_and_run_sliced_decoder_layer_test


# model key -> (model_id, asm_name, extra kwargs to build_and_run_sliced_decoder_layer_test)
MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "smollm2_135m_decoder", {}),
    "llada-8b": (
        "GSAI-ML/LLaDA-8B-Instruct",
        "llada_8b_decoder",
        {"trust_remote_code": True, "partial_load": True},
    ),
}

ALIASES = {
    "smollm2": "smollm2-135m",
    "smollm2-135m": "smollm2-135m",
    "llada": "llada-8b",
    "llada-8b": "llada-8b",
}


def _selected_models(selection: str):
    if selection == "all":
        return MODELS.items()
    return [(ALIASES[selection], MODELS[ALIASES[selection]])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        choices=["all", *sorted(ALIASES)],
        help="Model preset to run (default: all)",
    )
    args = parser.parse_args()

    for _key, (model_id, asm_name, extra) in _selected_models(args.model):
        print(f"\n{'=' * 80}\nDecoder test: {model_id}\n{'=' * 80}")
        build_dir = Path(__file__).parent / "build" / asm_name
        build_and_run_sliced_decoder_layer_test(
            model_id=model_id,
            asm_name=asm_name,
            build_dir=build_dir,
            layer_idx=0,
            seq_len=64,
            hidden_size=64,
            inter_dim=128,
            **extra,
        )