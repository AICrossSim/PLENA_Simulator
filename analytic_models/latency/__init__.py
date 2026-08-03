"""Compiler-derived latency estimation for PLENA final schedules."""

from .compute import estimate_compute_latency
from .frontend import (
    clear_dense_prefill_cache,
    dense_prefill_cache_info,
    estimate_dense_prefill,
)
from .hbm_v4 import (
    DEFAULT_HBM_V4_CALIBRATION,
    HbmPrecisionConfig,
    HbmServiceModelV4,
    HbmV4Config,
    HbmV4MemoryProvider,
    MemoryFormat,
    estimate_hbm_v4,
)
from .memory import ConfiguredBandwidthMemoryProvider
from .model import estimate_latency
from .schemas import (
    ComputeLatencyReport,
    LatencyReport,
    MemoryLatencyReport,
    OpcodeLatencyEntry,
    StageLatency,
)
from .timing import IdealII1TimingProvider, MainTimingConfig, MainTimingProvider
from .validation import (
    AssemblyParityReport,
    EmulatorParityReport,
    dynamic_assembly_opcode_counts,
    validate_compute_against_emulator_profile,
    validate_detailed_trace_against_assembly,
)

__all__ = [
    "AssemblyParityReport",
    "ComputeLatencyReport",
    "ConfiguredBandwidthMemoryProvider",
    "DEFAULT_HBM_V4_CALIBRATION",
    "EmulatorParityReport",
    "HbmPrecisionConfig",
    "HbmServiceModelV4",
    "HbmV4Config",
    "HbmV4MemoryProvider",
    "IdealII1TimingProvider",
    "LatencyReport",
    "MainTimingConfig",
    "MainTimingProvider",
    "MemoryFormat",
    "MemoryLatencyReport",
    "OpcodeLatencyEntry",
    "StageLatency",
    "clear_dense_prefill_cache",
    "dense_prefill_cache_info",
    "dynamic_assembly_opcode_counts",
    "estimate_compute_latency",
    "estimate_dense_prefill",
    "estimate_hbm_v4",
    "estimate_latency",
    "validate_compute_against_emulator_profile",
    "validate_detailed_trace_against_assembly",
]
