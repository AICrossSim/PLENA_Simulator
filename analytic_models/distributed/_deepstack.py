"""Access to DeepStack, the network and parallelism model behind ``analytic_models.distributed``.

DeepStack (Mo et al., "DeepStack: Facilitating Co-Design Exploration of 3D
DRAM-Stacked Accelerators for Distributed LLM Inference", MICRO 2026,
arXiv:2604.04750) is vendored as the ``DeepStack`` git submodule
(https://github.com/tile-ai/DeepStack). This module puts its source tree on
``sys.path`` and imports only open-source DeepStack modules: hierarchical NoC
topologies and routing, collective algorithms, parallel rank grouping and MoE
routing statistics. None of them loads DeepStack's bundled reference binaries.

Set ``PLENA_DEEPSTACK_ROOT`` to use another DeepStack checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "PLENA_DEEPSTACK_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE_BINARIES = ("mosaic.arch._reference_model", "mosaic.cost._capacity", "mosaic.noc._model_support")


def deepstack_root() -> Path:
    root = Path(os.environ.get(ENV_VAR) or _REPO_ROOT / "DeepStack")
    if not (root / "src" / "deepstack" / "mosaic").is_dir():
        raise ModuleNotFoundError(
            f"DeepStack sources not found at {root}: run `git submodule update --init DeepStack` "
            f"or point {ENV_VAR} at a DeepStack checkout"
        )
    return root


def _ensure_on_path() -> Path:
    root = deepstack_root().resolve()
    for sub in ("src/tilesight", "src/deepstack"):
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


_ROOT = _ensure_on_path()

import mosaic  # noqa: E402

if not Path(mosaic.__file__).resolve().is_relative_to(_ROOT):
    raise ImportError(
        f"`mosaic` was imported from {mosaic.__file__}, not from DeepStack at {_ROOT}; "
        "another package named mosaic is installed or already imported"
    )

from mosaic.collectives import all_gather_wrapper, all_reduce_wrapper, ep_all_to_all_wrapper  # noqa: E402
from mosaic.collectives.all_reduce_wrapper import all_reduce_ring  # noqa: E402
from mosaic.noc.custom_profile import make_custom_profile  # noqa: E402
from mosaic.noc.energy_config import NocEnergyConfig  # noqa: E402
from mosaic.noc.noc_topo import Hierarchy, get_extend_max_routes_with_traffic  # noqa: E402
from mosaic.noc.route_stats import build_route_stats_from_extended  # noqa: E402
from mosaic.noc.traffic_matrix import TrafficMatrix  # noqa: E402
from mosaic.parallelism import ParallelScheme  # noqa: E402
from mosaic.utils import (  # noqa: E402
    Modeling_Granularity,
    count_expert_frequency_flatten_wrapper,
    estimate_experts_activated,
    estimate_moe_routing_imbalance_overhead,
)

_loaded = sorted(name for name in _REFERENCE_BINARIES if name in sys.modules)
if _loaded:
    raise ImportError(f"DeepStack reference binaries were loaded unexpectedly: {', '.join(_loaded)}")


def load_routing_trace(name: str) -> tuple[object, object]:
    """``(prefill, decode)`` expert-id arrays packaged with DeepStack (``mosaic.data.routing``)."""

    from mosaic.data.routing import load_routing

    return load_routing(name)


ROUTING_TRACES = ("qwen3_235b", "deepseek_v3")

__all__ = [
    "ROUTING_TRACES",
    "Hierarchy",
    "Modeling_Granularity",
    "NocEnergyConfig",
    "ParallelScheme",
    "TrafficMatrix",
    "all_gather_wrapper",
    "all_reduce_ring",
    "all_reduce_wrapper",
    "build_route_stats_from_extended",
    "count_expert_frequency_flatten_wrapper",
    "deepstack_root",
    "ep_all_to_all_wrapper",
    "estimate_experts_activated",
    "estimate_moe_routing_imbalance_overhead",
    "get_extend_max_routes_with_traffic",
    "load_routing_trace",
    "make_custom_profile",
]
