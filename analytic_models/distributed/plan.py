"""Parallelism of a multi-chip PLENA system, mapped onto DeepStack ``ParallelScheme``.

The conventions follow DeepStack's decode driver (arXiv:2604.04750,
tile-ai/DeepStack, ``mosaic/dse_space/dse_framework_multi_process_v4_decode_dump_stats.py``):

* attention, dense FFN, norms and pipeline transfers use ``tp``, ``cp`` and
  ``dp * ep`` data-parallel ranks (EP ranks act as data parallel outside MoE);
* MoE layers use ``ep`` expert-parallel ranks; with ``moe_tp_mode="replace"``
  the tensor-parallel ranks become expert-parallel ranks too
  (``_get_effective_moe_parallel`` with ``"replace_only"``), with ``"keep"``
  the experts stay tensor parallel inside each EP rank (``"none"``);
* ranks are grouped TP, EP, SP, CP, DP, PP from the innermost NoC level out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..stacked_dram._validate import positive_int
from ._deepstack import ParallelScheme

MOE_TP_MODES = ("replace", "keep")


@dataclass(frozen=True)
class ParallelPlan:
    tp: int = 1
    ep: int = 1
    dp: int = 1
    pp: int = 1
    cp: int = 1
    moe_tp_mode: str = "replace"

    def __post_init__(self) -> None:
        for name in ("tp", "ep", "dp", "pp", "cp"):
            positive_int(getattr(self, name), name)
        if self.moe_tp_mode not in MOE_TP_MODES:
            raise ValueError(f"moe_tp_mode must be one of {', '.join(MOE_TP_MODES)}")

    @property
    def world_size(self) -> int:
        return self.tp * self.ep * self.dp * self.pp * self.cp

    def dense_scheme(self) -> ParallelScheme:
        """Scheme of attention, dense FFN, norms and pipeline transfers."""

        return ParallelScheme(tp=self.tp, ep=1, cp=self.cp, dp=self.dp * self.ep, pp=self.pp)

    def moe_scheme(self, num_experts: int) -> ParallelScheme:
        """Scheme of the routed experts of an MoE layer."""

        tp, ep = self.tp, self.ep
        if self.moe_tp_mode == "replace":
            combined = tp * ep
            if combined > num_experts:
                if combined % num_experts:
                    raise ValueError(f"tp*ep={combined} is not a multiple of the {num_experts} experts")
                ep, tp = num_experts, combined // num_experts
            else:
                ep, tp = combined, 1
        return ParallelScheme(tp=tp, ep=ep, cp=self.cp, dp=self.dp, pp=self.pp, ep1=ep, ep2=1)

    def describe(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "ep": self.ep,
            "dp": self.dp,
            "pp": self.pp,
            "cp": self.cp,
            "moe_tp_mode": self.moe_tp_mode,
            "world_size": self.world_size,
        }
