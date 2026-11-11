"""Design-space search of one organisation in one cell.

Stages: top10 (rule candidates), hetopt (rule candidates plus every-height pairs, het only), fullhom (hom candidates
using >= 97% of the budget), hetcap (het candidates using at most the multipliers of the cell's top10 hom chip).
Flow: probe every legal candidate on the screen windows of B32 and B128 from the template starts (four streams: the
best 30 again from all four-stream templates) -> the best 10 under every dispatch policy on the screen windows of
B32..B128 -> the best 10 distinct bases get one coordinate pass and are measured on the dev select windows -> the
chip with the lowest geometric mean over B32..B128 (best_main) and B2..B128 (best_all) among those whose arrays all
work in every batch -> heldout report windows and stall counters of the chosen chips.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..sim import workload as wl
from ..sim.engine import Environment, drain_latency
from . import results as rs
from .candidates import Ranking, base_macs, extend, extension_pairs, rule_candidates
from .evaluate import INF, Evaluator, sram_mib
from .shapes import AnalyticModel
from .organisations import ORGS, expand
from .tune import SCREEN_BATCHES, coordinate_pass, fit, policies, starts, valid, with_policy

STAGES = ("top10", "hetopt", "fullhom", "hetcap")
TOPN = 10
# Analytic re-delivery rate in units of the HBM weight rate: what a 16 KB/cycle port leaves after first passes.
KAPPA = {"bf16": 1.0, "w8g128": 0.54, "w4g32": 0.33}
# Exposed drains per expert task of the systolic timing modes (two matrix phases feed a vector phase in modes 1 and 2).
EXPERT_DRAINS = {1: 2, 2: 2, 3: 1}


def _key(base) -> tuple:
    return tuple(tuple(x) for x in base)


def stage_candidates(
    root: Path,
    model: str,
    fmt: str,
    P: int,
    org_name: str,
    stage: str,
    out: Path,
    hbm_tbps: int = 8,
    fields: dict | None = None,
) -> tuple[list[dict], dict]:
    """Candidates of a search stage and the counts recorded with its result."""
    org, kappa, fields = ORGS[org_name], KAPPA[fmt], fields or {}
    rank = Ranking(AnalyticModel(root, model, fmt, hbm=hbm_tbps * 1000.0), kappa)
    rules = rule_candidates(rank, P)
    cands, info = rules[org.kind], dict(rule_candidates=rules["C"], mac_cap=None)
    if stage == "hetopt":
        if org.kind != "het":
            raise ValueError("hetopt needs a het organisation")
        systolic = fields.get("systolic", 0)
        ext_rank = rank
        if systolic:
            env = replace(Environment(), **fields)
            ext_rank = Ranking(
                AnalyticModel(
                    root,
                    model,
                    fmt,
                    hbm=hbm_tbps * 1000.0,
                    drain=lambda n: drain_latency(n, env),
                    expert_drains=EXPERT_DRAINS[systolic],
                ),
                kappa,
            )
        pairs, ranked = extension_pairs(ext_rank, P, keep_small_n=bool(systolic))
        info.update(
            n_pairs_generated=len(pairs),
            n_extended=extend(cands, ext_rank, ranked),
            heights=sorted({x[0] for x in rank.shapes}),
        )
    elif stage == "fullhom":
        cands = [c for c in cands if base_macs(c["base"], "hom") >= 0.97 * P]
    elif stage == "hetcap":
        reference = rs.read_json(
            rs.search_path(out, "top10", rs.cell_name(model, fmt, P, hbm_tbps), org.name.replace("het", "hom"))
        )
        info["mac_cap"] = base_macs(reference["best_main"]["base"], "het")
        cands = [c for c in cands if base_macs(c["base"], "het") <= info["mac_cap"]]
    return cands, info


def search(
    root: Path,
    model: str,
    fmt: str,
    P: int,
    org_name: str,
    stage: str,
    out: Path,
    hbm_tbps: int = 8,
    threads: int = 3,
    engine_fields: dict | None = None,
) -> dict:
    t0 = time.time()
    org, fields = ORGS[org_name], engine_fields or {}
    cands, info = stage_candidates(root, model, fmt, P, org_name, stage, out, hbm_tbps, fields)

    ev = Evaluator(root, model, fmt, P, hbm_tbps, threads)
    own, every = starts(ev, org, hbm_tbps, fields)
    first_policy = (14, None) if org.kind == "het" else (13, None)

    probe = []
    for c in cands:
        shapes = expand(org, c["base"])
        if not ev.legal([list(x) for x in c["base"]]):
            continue
        v, e, tag = min(
            (fit(ev, org, shapes, with_policy(s, first_policy), "screen", SCREEN_BATCHES) for s in own),
            key=lambda z: z[0],
        )
        probe.append(dict(base=c["base"], rule=c["rule"], shapes=shapes, probe=v, start=tag, env=asdict(e)))
    probe.sort(key=lambda r: r["probe"])
    if every:
        for r in probe[:30]:
            v, e, tag = min(
                (fit(ev, org, r["shapes"], with_policy(s, first_policy), "screen", SCREEN_BATCHES) for s in every),
                key=lambda z: z[0],
            )
            if v < r["probe"]:
                r.update(probe=v, start="alltpl_" + tag, env=asdict(e))
        probe.sort(key=lambda r: r["probe"])

    stage2 = []
    for r in [r for r in probe if r["probe"] < INF][: max(6, TOPN)]:
        for pol in policies(org):
            e = with_policy(Environment(**r["env"]), pol)
            stage2.append(
                dict(
                    base=r["base"],
                    rule=r["rule"],
                    shapes=r["shapes"],
                    policy=list(pol),
                    screen=ev.score(r["shapes"], e, "screen", wl.MAIN_BATCHES),
                    env=asdict(e),
                )
            )
    stage2.sort(key=lambda r: r["screen"])
    top, seen = [], set()
    for r in stage2:
        if r["screen"] < INF and _key(r["base"]) not in seen:
            seen.add(_key(r["base"]))
            top.append(r)
        if len(top) == TOPN:
            break

    tuned = []

    def tune(r, rescued=False):
        e, v = coordinate_pass(ev, org, r["shapes"], Environment(**r["env"]), r["screen"], "screen", wl.MAIN_BATCHES)
        sel = ev.per_window(r["shapes"], e, "select", wl.BATCHES)
        tuned.append(
            dict(
                base=r["base"],
                rule=r["rule"],
                shapes=r["shapes"],
                policy=r["policy"],
                screen=r["screen"],
                tuned=v,
                env=asdict(e),
                select={k: x["cycles"] for k, x in sel.items()},
                select_busy={k: x["busy"] for k, x in sel.items()},
                valid=valid(org, sel),
                select_main=wl.geomean([sel[f"B{B}"]["cycles"] for B in wl.MAIN_BATCHES]),
                select_all=wl.geomean([sel[f"B{B}"]["cycles"] for B in wl.BATCHES]),
            )
        )
        if rescued:
            tuned[-1]["rescued"] = True

    def rescue(r):
        """Retune from a balanced SRAM split, halving the slots until the chip fits."""
        for share in (500, 375, 625):
            e = replace(Environment(**r["env"]), share=share)
            v = ev.score(r["shapes"], e, "screen", wl.MAIN_BATCHES)
            while v == INF and (e.slots > 2 or e.stage_slots > 2):
                e = replace(e, slots=max(2, e.slots // 2), stage_slots=max(2, e.stage_slots // 2))
                v = ev.score(r["shapes"], e, "screen", wl.MAIN_BATCHES)
            if v < INF:
                tune(dict(r, env=asdict(e), screen=v), rescued=True)
                return

    for r in top:
        tune(r)
    if org.arrays == 2 and not any(t["valid"] for t in tuned):
        for r in top:
            rescue(r)
    if not any(t["valid"] for t in tuned):
        more = [x for x in stage2 if x["screen"] < INF] + [
            dict(r, policy=list(first_policy), screen=r["probe"]) for r in probe if r["probe"] < INF
        ]
        for r in more:
            if _key(r["base"]) in seen:
                continue
            seen.add(_key(r["base"]))
            tune(r)
            if org.arrays == 2 and not tuned[-1]["valid"]:
                rescue(r)
            if any(t["valid"] for t in tuned) or len(tuned) >= 20:
                break

    best = {}
    for metric in ("select_main", "select_all"):
        best[metric] = (
            sorted([t for t in tuned if t["valid"]], key=lambda t: t[metric]) or sorted(tuned, key=lambda t: t[metric])
        )[0]
    reports = {}
    for b in best.values():
        e = Environment(**b["env"])
        if (_key(b["base"]), e) not in reports:
            reports[_key(b["base"]), e] = (
                ev.per_window(b["shapes"], e, "report", wl.BATCHES),
                ev.per_window(b["shapes"], e, "select", SCREEN_BATCHES, mechanism=True),
            )
        report, mech = reports[_key(b["base"]), e]
        b["report"] = report
        b["mechanism"] = {k: x["mechanism"] for k, x in mech.items()}
        b["report_main"] = wl.geomean([report[f"B{B}"]["cycles"] for B in wl.MAIN_BATCHES])
        b["report_all"] = wl.geomean([report[f"B{B}"]["cycles"] for B in wl.BATCHES])
    return dict(
        stage=stage,
        model=model,
        fmt=fmt,
        hbm_tbps=hbm_tbps,
        P=P,
        sram_mib=sram_mib(P),
        org=org.name,
        streams=org.streams,
        topn=TOPN,
        n_candidates=len(cands),
        **info,
        probe=probe,
        stage2=stage2,
        tuned=tuned,
        best_main=best["select_main"],
        best_all=best["select_all"],
        rejects=ev.rejects[:50],
        engine_fields=fields,
        seconds=time.time() - t0,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("org", choices=sorted(ORGS))
    ap.add_argument("--stage", choices=STAGES, default="top10")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--systolic", type=int, default=0, help="systolic timing mode of the engine (0 = none)")
    ap.add_argument("--drain-form", default=None)
    ap.add_argument("--drain-segment", type=int, default=None)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    fields = {
        k: v
        for k, v in (("systolic", a.systolic), ("drain_form", a.drain_form), ("drain_segment", a.drain_segment))
        if v
    }
    res = search(a.data, a.model, a.fmt, a.P, a.org, a.stage, a.out, a.hbm, a.threads, fields)
    rs.write_json(rs.search_path(a.out, a.stage, rs.cell_name(a.model, a.fmt, a.P, a.hbm), a.org), res)
    b = res["best_main"]
    print(
        a.stage,
        a.model,
        a.fmt,
        a.P,
        a.org,
        "candidates",
        res["n_candidates"],
        "best",
        b["base"],
        b["policy"],
        round(b["report_main"], 1),
        "valid",
        b["valid"],
        f"{res['seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
