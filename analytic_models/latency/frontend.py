"""Model-config frontends for compiler-derived latency estimation."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import toml

from compiler.aten.cost_frontend import CompilerHardwareSpec, compile_dense_decoder_trace
from compiler.aten.program_sink import COST_TRACE_GRANULARITY_SUMMARY

from .compute import estimate_compute_latency
from .hbm_v4 import (
    DEFAULT_HBM_V4_CALIBRATION,
    HbmPrecisionConfig,
    HbmServiceModelV4,
    HbmV4Config,
    HbmV4MemoryProvider,
)
from .memory import ConfiguredBandwidthMemoryProvider
from .model import estimate_latency
from .schemas import LatencyReport
from .timing import MainTimingConfig


def _config_value(section: dict[str, Any], name: str, default: int) -> int:
    raw = section.get(name, {"value": default})
    return int(raw.get("value", default) if isinstance(raw, dict) else raw)


@lru_cache(maxsize=1)
def _compiler_source_fingerprint() -> str:
    """Hash the compiler implementation that determines final schedules."""

    aten_root = Path(__file__).resolve().parents[2] / "PLENA_Compiler" / "aten"
    digest = sha256()
    for source in sorted(aten_root.rglob("*.py")):
        relative = source.relative_to(aten_root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_fingerprint(path: str | Path) -> tuple[str, str]:
    resolved = Path(path).resolve()
    return str(resolved), sha256(resolved.read_bytes()).hexdigest()


def _estimate_dense_prefill_uncached(
    model_config: str | Path,
    settings_toml: str | Path,
    *,
    seq_len: int,
    batch_size: int = 1,
    timing_provider: str = "main",
    memory_provider: str = "hbm-v4",
    memory_bandwidth_gbps: float = 2_039.0,
    hbm_v4_channels: int = 32,
    hbm_v4_calibration: str | Path = DEFAULT_HBM_V4_CALIBRATION,
    config_section: str = "ANALYTIC",
) -> LatencyReport:
    """Compile and estimate one complete dense prefill workload."""

    timing = MainTimingConfig.from_toml(settings_toml, section=config_section)
    section = toml.load(settings_toml)[config_section]
    settings = section["CONFIG"]
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
        compiler_hash=_compiler_source_fingerprint(),
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )
    compute = estimate_compute_latency(compiled.trace, timing, timing_provider)
    if memory_provider == "configured-bandwidth":
        memory = ConfiguredBandwidthMemoryProvider(memory_bandwidth_gbps).estimate(compiled.trace)
    elif memory_provider == "hbm-v4":
        memory = HbmV4MemoryProvider(
            HbmServiceModelV4.load(hbm_v4_calibration),
            HbmPrecisionConfig.from_settings(section["PRECISION"]),
            HbmV4Config(hbm_v4_channels),
            aggregation="sufficient-statistics",
        ).estimate(compiled.trace)
    else:
        raise ValueError(f"unsupported memory provider {memory_provider!r}")
    return estimate_latency(compiled.trace, compute, memory)


@lru_cache(maxsize=4)
def _estimate_dense_prefill_cached(
    model_config: str,
    model_fingerprint: str,
    settings_toml: str,
    settings_fingerprint: str,
    compiler_fingerprint: str,
    seq_len: int,
    batch_size: int,
    timing_provider: str,
    memory_provider: str,
    memory_bandwidth_gbps: float,
    hbm_v4_channels: int,
    hbm_v4_calibration: str,
    calibration_fingerprint: str,
    config_section: str,
) -> LatencyReport:
    # Fingerprints are explicit cache-key fields. The uncached implementation
    # consumes the corresponding paths and records compiler provenance.
    del model_fingerprint, settings_fingerprint, compiler_fingerprint
    del calibration_fingerprint
    return _estimate_dense_prefill_uncached(
        model_config,
        settings_toml,
        seq_len=seq_len,
        batch_size=batch_size,
        timing_provider=timing_provider,
        memory_provider=memory_provider,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        hbm_v4_channels=hbm_v4_channels,
        hbm_v4_calibration=hbm_v4_calibration,
        config_section=config_section,
    )


def estimate_dense_prefill(
    model_config: str | Path,
    settings_toml: str | Path,
    *,
    seq_len: int,
    batch_size: int = 1,
    timing_provider: str = "main",
    memory_provider: str = "hbm-v4",
    memory_bandwidth_gbps: float = 2_039.0,
    hbm_v4_channels: int = 32,
    hbm_v4_calibration: str | Path = DEFAULT_HBM_V4_CALIBRATION,
    config_section: str = "ANALYTIC",
) -> LatencyReport:
    """Compile and estimate one dense prefill workload with a four-entry LRU."""

    model_path, model_fingerprint = _file_fingerprint(model_config)
    settings_path, settings_fingerprint = _file_fingerprint(settings_toml)
    calibration_path, calibration_fingerprint = _file_fingerprint(hbm_v4_calibration)
    return _estimate_dense_prefill_cached(
        model_path,
        model_fingerprint,
        settings_path,
        settings_fingerprint,
        _compiler_source_fingerprint(),
        int(seq_len),
        int(batch_size),
        str(timing_provider),
        str(memory_provider),
        float(memory_bandwidth_gbps),
        int(hbm_v4_channels),
        calibration_path,
        calibration_fingerprint,
        str(config_section),
    )


def clear_dense_prefill_cache() -> None:
    """Clear process-local semantic reports, primarily for validation tools."""

    _estimate_dense_prefill_cached.cache_clear()


def dense_prefill_cache_info():
    """Return the standard functools cache statistics for telemetry."""

    return _estimate_dense_prefill_cached.cache_info()


__all__ = [
    "clear_dense_prefill_cache",
    "dense_prefill_cache_info",
    "estimate_dense_prefill",
]
