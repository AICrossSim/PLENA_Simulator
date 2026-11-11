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

from ..sim import workload as wl
from ..sim.engine import Environment
from .evaluate import INF, Evaluator, sram_mib
from .organisations import FAMILIES, Organisation

TEMPLATES = json.loads((Path(__file__).with_name("templates.json")).read_text())
TEMPLATE_BUDGETS = (98304, 196608)
SCREEN_BATCHES = (32, 128)


def _pow2(x: float) -> int:
    return 1 << max(0, math.ceil(math.log2(x)))


def template(model: str, fmt: str, P: int, org: str) -> tuple[Environment, int]:
    """Template of the nearest budget; Qwen3-Next uses the Qwen3.5 templates, INT4 the INT8 ones."""
    Pt = min(TEMPLATE_BUDGETS, key=lambda q: abs(math.log(q / P)))
    key = f"{'qwen' if model == 'qwen3next' else model}_{fmt if fmt in ('bf16', 'w8g128') else 'w8g128'}_{Pt}/{org}"
    return Environment.from_dict(TEMPLATES[key]), Pt


def adapt(e: Environment, ev: Evaluator, Pt: int, hbm_tbps: int, org: Organisation, fields: dict) -> Environment:
    """Template of budget Pt adapted to the cell; fields: further engine settings (e.g. the systolic timing mode)."""
    f, P = wl.FORMATS[ev.fmt], ev.P
    e = replace(
        e,
        H=ev.hbm,
        capacity=sram_mib(P) * 2**20,
        bits=f["bits"],
        group=f["group"],
        converter_rate=ev.converter,
        W=max(e.W, _pow2(e.W * hbm_tbps / 8)),
        X=max(e.X, _pow2(e.X * P / Pt)),
        Y=max(e.Y, _pow2(e.Y * P / Pt)),
        kappa_reread=0.0,
        port_mode=1,
        shared_array=org.shared_array,
        **fields,
    )
    if org.streams == 4:
        e = replace(e, core_caps=1, ctx_per_core=org.streams // org.arrays, share2=500)
    return e


def _with_unshared_variant(org: Organisation, e: Environment) -> list[Environment]:
    """A single array also starts with every stream given SRAM (share 1000 leaves the second stream none)."""
    return [e, replace(e, share=500)] if org.kind == "single" and e.share >= 1000 else [e]


def starts(
    ev: Evaluator, org: Organisation, hbm_tbps: int = 8, fields: dict | None = None
) -> tuple[list[Environment], list[Environment]]:
    """Start environments from the organisation's template; four streams also from all three four-stream templates."""

    def adapted(name):
        e, Pt = template(ev.model, ev.fmt, ev.P, name)
        return _with_unshared_variant(org, adapt(e, ev, Pt, hbm_tbps, org, fields or {}))

    own = adapted(org.name)
    every = [e for name in FAMILIES[4] for e in adapted(name)] if org.streams == 4 else []
    return own, every


def fit(ev: Evaluator, org: Organisation, shapes, e: Environment, kind: str, batches) -> tuple[float, Environment, str]:
    """Score a start environment, repairing it when the chip does not fit the SRAM: another SRAM split, then halved
    weight and staging slots (two streams: down to 4 and a minimal footprint; four streams: down to 2)."""
    v, tag = ev.score(shapes, e, kind, batches), "tpl"
    two = org.streams == 2
    for s in (500, 375, 625, 250, 750, 125, 875) if two else (500, 375, 625, 250, 750):
        if v < INF:
            break
        if s != e.share:
            v2 = ev.score(shapes, replace(e, share=s), kind, batches)
            if v2 < INF:
                e, v, tag = replace(e, share=s), v2, tag + f"_share{s}"
                break
    floor = 4 if two else 2
    while v == INF and (e.slots > floor or e.stage_slots > floor):
        e = replace(e, slots=max(floor, e.slots // 2), stage_slots=max(floor, e.stage_slots // 2))
        tag += "_half"
        v = ev.score(shapes, e, kind, batches)
    if v == INF and two:
        e = replace(e, slots=4, stage_slots=4, vslots=1, ybanks=1, lookahead=16)
        tag += "_minfoot"
        v = ev.score(shapes, e, kind, batches)
    return v, e, tag


def policies(org: Organisation) -> list[tuple[int, int | None]]:
    """(dispatch policy, policy-14 steal rule): 2 = per-stream queues (two streams only), 13 = shared queue from both
    ends, 14 = band dispatch with the taller array at the hot end."""
    out = [(2, None), (13, None)] if org.streams == 2 else [(13, None)]
    if org.streams == 4 and org.kind == "single":
        out.append((14, None))
    if org.arrays == 2:
        out += [(14, s) for s in (0, 1, 2)]
    return out


def with_policy(e: Environment, policy) -> Environment:
    p, steal = policy
    e = replace(e, policy=p)
    return replace(e, p14_steal=steal) if steal is not None else e


def tune_space(org: Organisation) -> list[tuple[str, tuple]]:
    space = [("share", (125, 250, 375, 500, 625, 750, 875))]
    if org.streams == 4:
        space.append(("share2", (250, 375, 500, 625, 750)))
    return [
        *space,
        ("lookahead", (16, 64, 256, 1024)),
        ("slots", (4, 8, 16, 32, 48, 64, 96)),
        ("stage_slots", (4, 8, 12, 16, 24, 32, 48)),
        ("vslots", (4, 8, 16, 32)),
    ]


def coordinate_pass(
    ev: Evaluator, org: Organisation, shapes, e: Environment, v: float, kind: str, batches
) -> tuple[Environment, float]:
    """One pass over the tuning space; a value is kept when it lowers the score."""
    for name, values in tune_space(org):
        for x in values:
            if getattr(e, name) == x:
                continue
            v2 = ev.score(shapes, replace(e, **{name: x}), kind, batches)
            if v2 < v:
                v, e = v2, replace(e, **{name: x})
    return e, v


def valid(org: Organisation, select: dict) -> bool:
    """Every array works in every batch."""
    return org.arrays == 1 or all(min(x["busy"]) > 0 for x in select.values())
