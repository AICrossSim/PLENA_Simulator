"""
Multi-Model Real-Weight FFN Test

Validates the PLENA FFN operator across multiple HuggingFace model
architectures by loading layer-0 MLP weights and comparing against a
hardware-accurate golden reference.

Each model's hidden_size / inter_dim are sliced to simulator limits
(hidden=128, inter=256).  Add new models by extending the MODELS list —
no extra test files needed.
"""

import argparse
from pathlib import Path

from transactional_emulator.testbench.sliced_layer_test_builder import build_and_run_sliced_ffn_test


# model key -> (model_id, asm_name, extra kwargs to build_and_run_sliced_ffn_test)
MODELS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "smollm2_135m_ffn", {}),
    "clm60m": ("AICrossSim/clm-60m", "clm60m_ffn", {}),
    "smolvlm2": (
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "smolvlm2_256m_ffn",
        {},
    ),
}

ALIASES = {
    "smollm2": "smollm2-135m",
    "smollm2-135m": "smollm2-135m",
    "clm60m": "clm60m",
    "clm-60m": "clm60m",
    "smolvlm2": "smolvlm2",
    "smolvlm2-256m": "smolvlm2",
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
        print(f"\n{'=' * 80}\nFFN test: {model_id}\n{'=' * 80}")
        build_dir = Path(__file__).parent / "build" / asm_name
        build_and_run_sliced_ffn_test(
            model_id=model_id,
            asm_name=asm_name,
            build_dir=build_dir,
            layer_idx=0,
            batch_size=4,
            **extra,
        )