"""
Multi-Model Real-Weight FFN Test

Validates the PLENA FFN operator across multiple HuggingFace model
architectures by loading layer-0 MLP weights and comparing against a
hardware-accurate golden reference.

Each model's hidden_size / inter_dim are sliced to simulator limits
(hidden=128, inter=256).  Add new models by extending the MODELS list —
no extra test files needed.
"""

from pathlib import Path

from transactional_emulator.testbench.model_layer_test_builder import build_and_run_ffn_test


# (model_id, asm_name, extra kwargs to build_and_run_ffn_test)
MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", "smollm2_135m_ffn", {}),
    ("AICrossSim/clm-60m", "clm60m_ffn", {}),
    ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "smolvlm2_256m_ffn", {}),
]


if __name__ == "__main__":
    for model_id, asm_name, extra in MODELS:
        print(f"\n{'=' * 80}\nFFN test: {model_id}\n{'=' * 80}")
        build_dir = Path(__file__).parent / "build" / asm_name
        build_and_run_ffn_test(
            model_id=model_id,
            asm_name=asm_name,
            build_dir=build_dir,
            layer_idx=0,
            batch_size=4,
            **extra,
        )
