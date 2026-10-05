"""Multi-chip PLENA performance model built on DeepStack's network and parallelism model.

Each device is a PLENA chip (``PerfModel`` compute and a ``stacked_dram``
memory system); devices are connected by a DeepStack NoC hierarchy and run a
TP/EP/DP/PP/CP plan. DeepStack (Mo et al., "DeepStack: Facilitating Co-Design
Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference",
MICRO 2026, arXiv:2604.04750) is vendored as the ``DeepStack`` git submodule.
See README.md in this directory.
"""

from .comm import NO_COMM, CommCost, Network
from .model import (
    COLLECTIVES,
    COMM_OVERLAP_POLICIES,
    DistributedEstimate,
    PipelinePass,
    StageReport,
    estimate_distributed,
)
from .moe import ROUTING_MODES, ExpertLoad, expert_load, routing_rows
from .noc import NocProfile, load_noc_profile, noc_from_dict
from .plan import MOE_TP_MODES, ParallelPlan
from .workload import FAMILIES, ModelSpec

__all__ = [
    "COLLECTIVES",
    "COMM_OVERLAP_POLICIES",
    "FAMILIES",
    "MOE_TP_MODES",
    "NO_COMM",
    "ROUTING_MODES",
    "CommCost",
    "DistributedEstimate",
    "ExpertLoad",
    "ModelSpec",
    "Network",
    "NocProfile",
    "ParallelPlan",
    "PipelinePass",
    "StageReport",
    "estimate_distributed",
    "expert_load",
    "load_noc_profile",
    "noc_from_dict",
    "routing_rows",
]
