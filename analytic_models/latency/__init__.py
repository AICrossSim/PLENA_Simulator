"""Compiler-derived latency estimation for PLENA final schedules."""

from .compute import estimate_compute_latency
from .frontend import (
    clear_dense_prefill_cache,
    dense_prefill_cache_info,
    estimate_dense_prefill,
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
    "EmulatorParityReport",
    "IdealII1TimingProvider",
    "LatencyReport",
    "MainTimingConfig",
    "MainTimingProvider",
    "MemoryLatencyReport",
    "OpcodeLatencyEntry",
    "StageLatency",
    "clear_dense_prefill_cache",
    "dense_prefill_cache_info",
    "dynamic_assembly_opcode_counts",
    "estimate_compute_latency",
    "estimate_dense_prefill",
    "estimate_latency",
    "validate_compute_against_emulator_profile",
    "validate_detailed_trace_against_assembly",
]
