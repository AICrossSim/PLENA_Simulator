"""Conspicuously fictional configurations shared by the stacked-DRAM tests.

The DRAM, timing and thermal values follow DeepStack's public interface tests
(tile-ai/DeepStack tests/test_custom_arch_profile.py and
tests/test_custom_dram_dse_policy.py); none of them describes a real device.
"""

from __future__ import annotations

from pathlib import Path

from analytic_models.stacked_dram import DramTimingConfig, StackedDramConfig, ThermalPolicy

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "analytic_models" / "stacked_dram" / "examples"
SETTINGS = REPO_ROOT / "plena_settings.toml"
ISA_LIB = REPO_ROOT / "analytic_models" / "performance" / "customISA_lib.json"

# Public Llama-3.1-8B dimensions (HuggingFace config), as in PLENA_Compiler's Model_Lib.
LLAMA_3_1_8B = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}


def fictional_dram(**changes: object) -> StackedDramConfig:
    values: dict[str, object] = {
        "total_layers": 10,
        "connected_layers": 3,
        "channels_per_connected_layer": 7,
        "bytes_per_channel_transfer": 5.5,
        "transfers_per_memory_clock": 1.25,
        "memory_frequency_hz": 345_678_901.0,
        "capacity_per_layer_bytes": 123_456_789,
        "transaction_bytes": 96,
        "fully_connected_efficiency": 0.37,
    }
    values.update(changes)
    return StackedDramConfig(**values)


def fictional_timing(**changes: object) -> DramTimingConfig:
    values: dict[str, object] = {
        "row_bytes": 4_070,
        "sector_bytes": 74,
        "sector_cycles": 3,
        "recharge_cycles": 41,
        "round_trip_latency_cycles": 137.5,
        "latency_clock_hz": 654_321_987.0,
    }
    values.update(changes)
    return DramTimingConfig(**values)


def fictional_thermal(**changes: object) -> ThermalPolicy:
    values: dict[str, object] = {
        "resistance_base_c_per_w": 0.42,
        "resistance_per_layer_c_per_w": 0.007,
        "baseline_layers": 3,
        "design_power_w": 73.0,
        "static_power_w": 8.0,
        "dynamic_power_exponent": 2.5,
    }
    values.update(changes)
    return ThermalPolicy(**values)
