"""Collective and pipeline transfer costs, priced by DeepStack's network model.

Every cost is DeepStack's two-stage estimate on the NoC hierarchy: the longest
summed hop latency of any route plus the busiest link's bytes over its
bandwidth (``mosaic.noc.noc_topo.get_extend_max_routes_with_traffic``), for
the algorithms DeepStack's wrappers choose between
(``mosaic.collectives``; arXiv:2604.04750, tile-ai/DeepStack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._deepstack import (
    Modeling_Granularity,
    ParallelScheme,
    TrafficMatrix,
    all_gather_wrapper,
    all_reduce_ring,
    all_reduce_wrapper,
    build_route_stats_from_extended,
    ep_all_to_all_wrapper,
    get_extend_max_routes_with_traffic,
)
from .noc import NocProfile


@dataclass(frozen=True)
class CommCost:
    """Time of one collective (or one round of pipeline transfers) and its link traffic.

    ``traffic`` is DeepStack's extended traffic matrix of the whole system: every
    group of the parallel dimension takes part at once.
    """

    hop_seconds: float
    link_seconds: float
    traffic: np.ndarray | None

    @property
    def seconds(self) -> float:
        return self.hop_seconds + self.link_seconds


NO_COMM = CommCost(0.0, 0.0, None)


def _key(scheme: ParallelScheme) -> tuple[int, ...]:
    return (scheme.tp, scheme.ep, scheme.sp, scheme.cp, scheme.dp, scheme.pp)


def _group_size(scheme: ParallelScheme, dim: str) -> int:
    return int(getattr(scheme, dim))


class Network:
    """DeepStack collectives on one NoC profile, memoised by their arguments."""

    def __init__(self, noc: NocProfile) -> None:
        self.noc = noc
        # The wrappers take a granularity for interface symmetry; overlap is applied by the caller.
        self._granularity = Modeling_Granularity("coarse", False, False)
        self._cache: dict[tuple[Any, ...], CommCost] = {}
        self._energy: dict[int, tuple[np.ndarray, float]] = {}

    def _cached(self, key: tuple[Any, ...], compute) -> CommCost:
        if key not in self._cache:
            hop, link, traffic = compute()
            self._cache[key] = CommCost(float(hop), float(link), traffic)
        return self._cache[key]

    def all_reduce(self, scheme: ParallelScheme, dim: str, nbytes: float) -> CommCost:
        size = _group_size(scheme, dim)
        if size == 1 or nbytes <= 0:
            return NO_COMM
        # DeepStack's recursive-doubling, Rabenseifner and double-tree variants need a power-of-two group.
        wrapper = all_reduce_wrapper if size & (size - 1) == 0 else all_reduce_ring
        return self._cached(
            ("all_reduce", _key(scheme), dim, int(nbytes)),
            lambda: wrapper(None, scheme, self.noc.hierarchy, self._granularity, dim, int(nbytes)),
        )

    def all_gather(self, scheme: ParallelScheme, dim: str, nbytes: float) -> CommCost:
        if _group_size(scheme, dim) == 1 or nbytes <= 0:
            return NO_COMM
        return self._cached(
            ("all_gather", _key(scheme), dim, int(nbytes)),
            lambda: all_gather_wrapper(None, scheme, self.noc.hierarchy, self._granularity, dim, int(nbytes)),
        )

    def ep_all_to_all(
        self,
        scheme: ParallelScheme,
        *,
        bytes_each_token: float,
        routing: np.ndarray | None,
        micro_batch: int,
        seq: int,
        num_experts: int,
        top_k: int,
        imbalance: float,
        tag: str,
    ) -> CommCost:
        """Token dispatch (or combine) across each expert-parallel group."""

        if scheme.ep == 1:
            return NO_COMM
        return self._cached(
            (
                "ep_all_to_all",
                _key(scheme),
                int(bytes_each_token),
                micro_batch,
                seq,
                num_experts,
                top_k,
                imbalance,
                tag,
            ),
            lambda: ep_all_to_all_wrapper(
                scheme,
                self.noc.hierarchy,
                self._granularity,
                int(bytes_each_token),
                routing,
                micro_batch,
                seq,
                num_experts,
                top_k,
                imbalance,
            ),
        )

    def pipeline_p2p(self, scheme: ParallelScheme, nbytes: float) -> CommCost:
        """Activation hand-off between pipeline stages.

        Every rank of stage ``i`` sends ``nbytes`` to its counterpart in stage
        ``i + 1``. As in DeepStack's decode driver (``get_pipeline_time``) each
        stage boundary is routed on its own and the time is the slowest
        boundary's; the traffic is summed over all boundaries, which all
        transfer once per pipeline period.
        """

        if scheme.pp == 1 or nbytes <= 0:
            return NO_COMM

        def compute():
            slowest = (0.0, 0.0)
            total = None
            for stage in range(scheme.pp - 1):
                tm = TrafficMatrix(scheme.world_size())
                tm.add_intra_group_traffic_pair_bulk(
                    "pp",
                    [[int(nbytes), stage, stage + 1]],
                    tp=scheme.tp,
                    ep=scheme.ep,
                    sp=scheme.sp,
                    cp=scheme.cp,
                    dp=scheme.dp,
                    pp=scheme.pp,
                )
                hop, link, _, traffic = get_extend_max_routes_with_traffic(tm, self.noc.hierarchy)
                if hop + link > sum(slowest):
                    slowest = (hop, link)
                total = traffic.copy() if total is None else total + traffic
            return slowest[0], slowest[1], total

        return self._cached(("pipeline_p2p", _key(scheme), int(nbytes)), compute)

    def energy_pj(self, cost: CommCost) -> float | None:
        """NoC energy of one collective's traffic; ``None`` without energy coefficients."""

        if self.noc.energy is None:
            return None
        if cost.traffic is None:
            return 0.0
        cached = self._energy.get(id(cost.traffic))
        if cached is None or cached[0] is not cost.traffic:
            stats = build_route_stats_from_extended(
                [cost.traffic], self.noc.hierarchy, max(cost.seconds, 1e-30), self.noc.energy
            )
            cached = (cost.traffic, float(stats.total_noc_energy_pj))
            self._energy[id(cost.traffic)] = cached
        return cached[1]
