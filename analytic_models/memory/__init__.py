"""
Memory Analytic Models for PLENA Simulator.

This module provides memory footprint, bandwidth, and traffic analysis
for LLM inference workloads on PLENA hardware.

Key Classes:
- MemoryModel: Per-layer memory traffic computation
- LLMMemoryModel: End-to-end LLM memory analysis with utilization metrics

Usage:
    from analytic_models.memory import LLMMemoryModel, load_memory_config_from_toml

    # Load config and create model
    memory_config = load_memory_config_from_toml("plena_settings.toml")
    model = LLMMemoryModel(model_config_path, memory_config, ...)

    # Compute utilization with execution cycles from performance model
    prefill_util = model.compute_prefill_utilization(prefill_cycles)
    decode_util = model.compute_decode_utilization(decode_cycles)

    # Print results
    model.print_utilization("prefill", prefill_util)
    model.print_utilization("decode", decode_util)
"""

from .llm_memory_model import (
    LLMMemoryModel,
    PhaseUtilizationAnalysis,
)
from .memory_model import (
    MemoryConfig,
    MemoryModel,
    MemoryTraffic,
    load_memory_config_from_toml,
)

__all__ = [
    "LLMMemoryModel",
    "MemoryConfig",
    "MemoryModel",
    "MemoryTraffic",
    "PhaseUtilizationAnalysis",
    "load_memory_config_from_toml",
]
