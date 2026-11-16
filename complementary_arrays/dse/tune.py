"""Start environments, capacity repair, dispatch policies and coordinate tuning of a search.

Start environments are tuned engine environments of earlier searches (templates.json, one per model, BF16 or INT8,
multiplier budget and organisation); they also fix every engine field that is not tuned here. A template is adapted to
the cell's HBM, SRAM, weight format and converter; the shared weight port of W bytes per cycle (port_mode 1) carries
both first passes and on-chip re-deliveries.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from ..sim.engine import Environment
from .evaluate import INF, Evaluator, sram_mib
from .formats import TIMING, with_format
from .organisations import Organisation

TEMPLATES = json.loads((Path(__file__).with_name("templates.json")).read_text())
TEMPLATE_BUDGETS = (98304, 196608)
PROBE_BATCHES = (32, 128)
SCREEN_BATCHES = (32, 64, 96, 128)
# Extra tuning axes (field, values) appended to every organisation's tuning space.
PACKED = (("redeliv_packed", (0, 1)),)
PACKED_PLACE = (("redeliv_packed", (0, 1)), ("shared_place", (0, 1, 2)))


def _pow2(x: float) -> int:
    return 1 << max(0, math.ceil(math.log2(x)))


def template(model: str, fmt: str, P: int, org: str) -> tuple[Environment, int]:
    """Template of the nearest budget; Qwen3-Next uses the Qwen3.5 templates, the other formats the INT8 ones."""
    Pt = min(TEMPLATE_BUDGETS, key=lambda q: abs(math.log(q / P)))
    key = f"{'qwen' if model == 'qwen3next' else model}_{fmt if fmt in ('bf16', 'w8g128') else 'w8g128'}_{Pt}/{org}"
    return Environment.from_dict(TEMPLATES[key]), Pt


def adapt(e: Environment, ev: Evaluator, Pt: int, hbm_tbps: int, org: Organisation, fields: dict) -> Environment:
    """Template of budget Pt adapted to the cell; fields: further engine settings (e.g. redeliv_packed)."""
    P = ev.P
    e = replace(
        e,
        H=ev.hbm,
        capacity=sram_mib(P) * 2**20,
        W=max(e.W, _pow2(e.W * hbm_tbps / 8)),
        X=max(e.X, _pow2(e.X * P / Pt)),
        Y=max(e.Y, _pow2(e.Y * P / Pt)),
        kappa_reread=0.0,
        port_mode=1,
        shared_array=org.shared_array,
        **TIMING,
        **fields,
    )
    return with_format(e, ev.fmt, ev.hbm)


def starts(ev: Evaluator, org: Organisation, hbm_tbps: int = 8, fields: dict | None = None) -> list[Environment]:
    """Start environments from the organisation's template; a single array also starts with both streams given SRAM
    (share 1000 leaves the second stream none)."""
    e, Pt = template(ev.model, ev.fmt, ev.P, org.name)
    e = adapt(e, ev, Pt, hbm_tbps, org, fields or {})
    return [e, replace(e, share=500)] if org.kind == "single" and e.share >= 1000 else [e]


def fit(ev: Evaluator, org: Organisation, shapes, e: Environment, kind: str, batches) -> tuple[float, Environment, str]:
    """Score a start environment, repairing it when the chip does not fit the SRAM: another SRAM split, then halved
    weight and staging slots down to 4, then a minimal footprint."""
    v, tag = ev.score(shapes, e, kind, batches), "tpl"
    for s in (500, 375, 625, 250, 750, 125, 875):
        if v < INF:
            break
        if s != e.share:
            v2 = ev.score(shapes, replace(e, share=s), kind, batches)
            if v2 < INF:
                e, v, tag = replace(e, share=s), v2, tag + f"_share{s}"
                break
    while v == INF and (e.slots > 4 or e.stage_slots > 4):
        e = replace(e, slots=max(4, e.slots // 2), stage_slots=max(4, e.stage_slots // 2))
        tag += "_half"
        v = ev.score(shapes, e, kind, batches)
    if v == INF:
        e = replace(e, slots=4, stage_slots=4, vslots=1, ybanks=1, lookahead=16)
        tag += "_minfoot"
        v = ev.score(shapes, e, kind, batches)
    return v, e, tag


def policies(org: Organisation) -> list[tuple[int, int | None]]:
    """(dispatch policy, steal rule): 2 per-stream queues, 13 shared queue from both ends, 14 band dispatch with the
    taller array at the hot end (steal rules 0, 1, 2), 15 supply-aware two-ended dispatch, 16 band dispatch with
    re-planning (steal rule 2; one array: no steal rule)."""
    out = [(2, None), (13, None)]
    if org.arrays == 2:
        out += [(14, s) for s in (0, 1, 2)]
    return out + [(15, None), (16, 2) if org.arrays == 2 else (16, None)]


def with_policy(e: Environment, policy) -> Environment:
    p, steal = policy
    e = replace(e, policy=p)
    return replace(e, p14_steal=steal) if steal is not None else e


def tune_space(extra=()) -> list[tuple[str, tuple]]:
    return [
        ("share", (125, 250, 375, 500, 625, 750, 875)),
        ("lookahead", (16, 64, 256, 1024)),
        ("slots", (4, 8, 16, 32, 48, 64, 96)),
        ("stage_slots", (4, 8, 12, 16, 24, 32, 48)),
        ("vslots", (4, 8, 16, 32)),
        *extra,
    ]


def coordinate_pass(
    ev: Evaluator, shapes, e: Environment, v: float, kind: str, batches, extra=(), passes: int = 1
) -> tuple[Environment, float]:
    """Passes over the tuning space; a value is kept when it lowers the score; stops after a pass without gain."""
    for _ in range(passes):
        gain = False
        for name, values in tune_space(extra):
            for x in values:
                if getattr(e, name) == x:
                    continue
                v2 = ev.score(shapes, replace(e, **{name: x}), kind, batches)
                if v2 < v:
                    v, e, gain = v2, replace(e, **{name: x}), True
        if not gain:
            break
    return e, v


def valid(org: Organisation, select: dict) -> bool:
    """Every array works in every batch."""
    return org.arrays == 1 or all(min(x["busy"]) > 0 for x in select.values())
