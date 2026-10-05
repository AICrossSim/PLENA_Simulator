"""3D-stacked DRAM memory model for PLENA's analytic flow.

The memory mechanisms are ported from DeepStack (Mo et al., "DeepStack:
Facilitating Co-Design Exploration of 3D DRAM-Stacked Accelerators for
Distributed LLM Inference", arXiv:2604.04750); each module names its source.
No hardware values ship with the package: a design is described by the caller,
in Python or in a JSON profile. See README.md in this directory.
"""

from .config import BufferingPolicy, DramEnergyConfig, DramTimingConfig, StackedDramConfig, ThermalPolicy
from .estimate import (
    OVERLAP_POLICIES,
    DecoderLatencyEstimate,
    DecoderShape,
    HbmStoragePrecision,
    PhaseEstimate,
    PhaseTotals,
    decode_block_traffic,
    element_bits,
    estimate_decoder_latency,
    prefill_block_traffic,
    price_stage,
)
from .memory import FixedBandwidthMemory, MemorySystem
from .model import LittlesLawResult, StackedDramModel, dram_connectivity_efficiency
from .profile import SCHEMA_VERSION, load_memory_profile, memory_from_dict

__all__ = [
    "OVERLAP_POLICIES",
    "SCHEMA_VERSION",
    "BufferingPolicy",
    "DecoderLatencyEstimate",
    "DecoderShape",
    "DramEnergyConfig",
    "DramTimingConfig",
    "FixedBandwidthMemory",
    "HbmStoragePrecision",
    "LittlesLawResult",
    "MemorySystem",
    "PhaseEstimate",
    "PhaseTotals",
    "StackedDramConfig",
    "StackedDramModel",
    "ThermalPolicy",
    "decode_block_traffic",
    "dram_connectivity_efficiency",
    "element_bits",
    "estimate_decoder_latency",
    "load_memory_profile",
    "memory_from_dict",
    "prefill_block_traffic",
    "price_stage",
]
