"""
AICrossSim/clm-60m FFN Test — Real Model Weights

Loads REAL weights from AICrossSim/clm-60m (layer 0 MLP)
and validates the PLENA FFN operator against a hardware-accurate golden reference.

Model dimensions:
    hidden_size = 384
    inter_dim   = 1408

Sliced to simulator limits:
    hidden_size = 128
    inter_dim   = 256
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_layer_test_builder import build_and_run_ffn_test

if __name__ == "__main__":
    build_dir = Path(__file__).parent / "build"
    build_and_run_ffn_test(
        model_id="AICrossSim/clm-60m",
        asm_name="clm60m_ffn",
        build_dir=build_dir,
        layer_idx=0,
        batch_size=4,
    )
