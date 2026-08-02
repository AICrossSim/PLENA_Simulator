"""Model-config frontends for compiler-derived latency estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from compiler.aten.cost_frontend import CompilerHardwareSpec, compile_dense_decoder_trace
from compiler.aten.program_sink import COST_TRACE_GRANULARITY_SUMMARY

from .compute import estimate_compute_latency
from .memory import ConfiguredBandwidthMemoryProvider
from .model import estimate_latency
from .schemas import LatencyReport
from .timing import MainTimingConfig


def _config_value(section: dict[str, Any], name: str, default: int) -> int:
    raw = section.get(name, {"value": default})
    return int(raw.get("value", default) if isinstance(raw, dict) else raw)


def estimate_dense_prefill(
    model_config: str | Path,
    settings_toml: str | Path,
    *,
    seq_len: int,
    batch_size: int = 1,
    timing_provider: str = "main",
    memory_bandwidth_gbps: float = 2_039.0,
    config_section: str = "ANALYTIC",
) -> LatencyReport:
    """Compile and estimate one complete dense prefill workload."""

    timing = MainTimingConfig.from_toml(settings_toml, section=config_section)
    settings = toml.load(settings_toml)[config_section]["CONFIG"]
    compiler_hardware = CompilerHardwareSpec(
        mlen=timing.mlen,
        blen=timing.blen,
        mram_tile_capacity=_config_value(settings, "MATRIX_SRAM_SIZE", 4),
        hlen=timing.hlen,
        broadcast_amount=timing.broadcast_amount,
        attention_head_packing=False,
        hbm_v_prefetch_amount=_config_value(settings, "HBM_V_Prefetch_Amount", 4),
        hbm_v_writeback_amount=_config_value(settings, "HBM_V_Writeback_Amount", 4),
    )
    compiled = compile_dense_decoder_trace(
        model_config,
        compiler_hardware,
        seq_len=seq_len,
        batch_size=batch_size,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )
    compute = estimate_compute_latency(compiled.trace, timing, timing_provider)
    memory = ConfiguredBandwidthMemoryProvider(memory_bandwidth_gbps).estimate(compiled.trace)
    return estimate_latency(compiled.trace, compute, memory)


__all__ = ["estimate_dense_prefill"]
