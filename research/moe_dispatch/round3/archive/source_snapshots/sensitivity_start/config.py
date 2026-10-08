"""The single source of operating points for every round-three stage.

An operating point is an analytical service ceiling, not measured HBM speed.
The credit count is passed in one Parameters object through costs, execution,
lower bounds, assignment relaxation, and replay. No consumer derives a second
independent HBM bandwidth setting.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import MappingProxyType
from ..round2.model import Parameters as FrozenParameters

SEED = 20261007
MAIN_CREDITS = 520
CLOCK_NS = 1.0
OPERATING_POINTS = MappingProxyType({126.03: 256, 192.0: 390, 256.0: MAIN_CREDITS})
MODES = ("pipelined", "port_tight")
BATCHES = (2, 4, 8, 16, 64, 96, 128)
FAMILIES = ("B1", "B2", "H51", "H42", "H33")


@dataclass(frozen=True)
class Parameters(FrozenParameters):
    credits: int = MAIN_CREDITS


def parameters(onchip_mode: str = "pipelined", *, credits: int | None = None,
               bw_GBps: float | None = None, **overrides) -> Parameters:
    if credits is not None and bw_GBps is not None:
        raise ValueError("Specify the credit count or an operating point, not both")
    if bw_GBps is not None:
        credits = OPERATING_POINTS[float(bw_GBps)]
    return Parameters(onchip_mode=onchip_mode,
                      credits=MAIN_CREDITS if credits is None else int(credits), **overrides)


def with_credits(p: Parameters, credits: int) -> Parameters:
    return replace(p, credits=int(credits))
