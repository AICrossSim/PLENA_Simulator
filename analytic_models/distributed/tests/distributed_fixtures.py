"""Shared inputs of the multi-chip model tests.

Model dimensions are public HuggingFace configs; the NoC values are
DeepStack's illustrative example values (tile-ai/DeepStack,
examples/custom_noc.py) and the memories are fictional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analytic_models.distributed import ModelSpec, NocProfile, noc_from_dict
from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml
from analytic_models.stacked_dram import FixedBandwidthMemory, HbmStoragePrecision

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "analytic_models" / "distributed" / "examples"
STACKED_DRAM_EXAMPLES = REPO_ROOT / "analytic_models" / "stacked_dram" / "examples"
SETTINGS = REPO_ROOT / "plena_settings.toml"
ISA_LIB = REPO_ROOT / "analytic_models" / "performance" / "customISA_lib.json"

UNBOUNDED = FixedBandwidthMemory(name="unbounded", bandwidth_bytes_per_s=1e30, capacity_bytes=None)
FICTIONAL_HBM = FixedBandwidthMemory(name="fictional", bandwidth_bytes_per_s=1.234e12, capacity_bytes=None)

# Public Llama-3.1-8B dimensions (HuggingFace config), as in PLENA_Compiler's Model_Lib.
LLAMA_3_1_8B: dict[str, Any] = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}

# Public gpt-oss-20b dimensions (HuggingFace config), as in PLENA_Compiler's Model_Lib.
GPT_OSS_20B: dict[str, Any] = {
    "model_type": "gpt_oss",
    "hidden_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "num_hidden_layers": 24,
    "intermediate_size": 2880,
    "vocab_size": 201088,
    "tie_word_embeddings": False,
    "num_local_experts": 32,
    "experts_per_token": 4,
    "num_experts_per_tok": 4,
    "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 12,
}

# A small fictional MoE decoder whose expert layout (128 experts, top-8)
# matches DeepStack's qwen3_235b routing trace.
FICTIONAL_MOE_128: dict[str, Any] = {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 768,
    "moe_intermediate_size": 256,
    "vocab_size": 32000,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}


def llama() -> ModelSpec:
    return ModelSpec.from_hf_config(LLAMA_3_1_8B, name="llama-3.1-8b")


def gpt_oss() -> ModelSpec:
    return ModelSpec.from_hf_config(GPT_OSS_20B, name="gpt-oss-20b")


def perf_model() -> PerfModel:
    return PerfModel(load_hardware_config_from_toml(SETTINGS), str(ISA_LIB))


def precision() -> HbmStoragePrecision:
    return HbmStoragePrecision.from_settings(SETTINGS)


def _switch(size: int, hop_ns: float, gbs: float) -> dict[str, Any]:
    layer: dict[str, Any] = {
        "kind": "switch",
        "shape": [1, size],
        "hop_latency_ns": hop_ns,
        "link_bandwidth_gbytes_per_s": gbs,
    }
    if size > 1:
        layer["switch_center_in_gbytes_per_s"] = gbs * size
        layer["switch_center_out_gbytes_per_s"] = gbs * size
    return layer


def switch_noc(devices: int, *, groups: int = 1, energy: bool = True) -> NocProfile:
    """``groups`` switches of ``devices // groups`` devices, joined by a slower switch."""

    data: dict[str, Any] = {
        "schema_version": 1,
        "kind": "noc_hierarchy",
        "name": f"test-switch-{groups}x{devices // groups}",
        "layers": {
            "L3": _switch(1, 0, 50),
            "L2": _switch(groups, 1000, 50),
            "L1": _switch(devices // groups, 200, 100),
        },
    }
    if energy:
        data["energy_pj_per_bit"] = {"l1": 1.0, "l2": 2.0, "l3": 3.0}
    return noc_from_dict(data)
