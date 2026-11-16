"""Design of one organisation for one batch size: the search with the objective replaced by the MoE time at that batch
on the dev snapshots, with the same budget for every organisation.

Candidates: the organisation's rule candidates ranked on the batch's load histograms, and seeds (given chips with their
settings, and the tuned lists of given search results), converted to the cell's weight format and, under the
equal-multiplier rule, promoted to the whole budget. Probe: every candidate from the template start and every seed with
its own settings, on the screen windows of the batch. Policy menu: the best 10 under every dispatch policy, a seed also
under its own. Tuning: the best 10 distinct shapes get one coordinate pass, then the select windows of the batch
(every array must work); rescue as in the search when none is valid. Chips given with --as-is join the tuned list as
they are. The design is the fastest valid tuned chip. Output perbatch/<cell>/B<B>/<org>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..sim import workload as wl
from ..sim.engine import Environment
from . import results as rs
from .candidates import base_macs, promote, rule_candidates
from .evaluate import INF, Evaluator, chip_setup
from .formats import with_format
from .organisations import ORGS, expand
from .search import Tuner, best, parse_tune_extra, ranking
from .tune import fit, policies, starts, with_policy

TOPN = 10


def seeds_of(ev: Evaluator, org, specs, packed_variants=(), full_budget=False) -> list[dict]:
    """Seeds from name=winner[,tuned] specs: the best_main chip of `winner` and every tuned chip of the search result
    `tuned` (either may be missing)."""
    out, seen = [], set()
    for spec in specs:
        name, _, paths = spec.partition("=")
        winner, _, tuned = paths.partition(",")
        items = []
        for path, label in ((winner, "winner"), (tuned, "tuned")):
            p = Path(path) if path else None
            if p is not None and p.is_dir():
                p = p / f"{org.name}.json"
            if p is None or not p.exists():
                continue
            x = rs.read_json(p)
            items += [(f"{name}:winner", x["best_main"])] if label == "winner" else [
                (f"{name}:tuned:{i}", t) for i, t in enumerate(x["tuned"])
            ]
        for tag, b in items:
            _, e = chip_setup(dict(b, shapes=expand(org, b["base"])), org.name)
            e = with_format(e, ev.fmt, ev.hbm)
            variants = [(tag, e)]
            if any(name.startswith(p) for p in packed_variants) and e.redeliv_packed == 0:
                variants.append((tag + "-P", replace(e, redeliv_packed=1)))
            if json.dumps(b["base"]) + json.dumps(asdict(e), sort_keys=True) in seen:
                continue
            for label, e in variants:
                key = json.dumps(b["base"]) + json.dumps(asdict(e), sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.append(dict(seed=label, base=b["base"], shapes=expand(org, b["base"]), env=asdict(e)))
    if full_budget:
        promoted = []
        for s in out:
            for nb in promote(org.kind, s["base"], ev.P):
                key = json.dumps(nb) + json.dumps(s["env"], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    promoted.append(dict(s, seed=s["seed"] + "-up", base=nb, shapes=expand(org, nb)))
        out = [s for s in out if base_macs(s["base"], org.kind) == ev.P] + promoted
    return out


def perbatch(a) -> dict:
    t0, org, B = time.time(), ORGS[a.org], a.batch
    ev = Evaluator(a.data, a.model, a.fmt, a.P, a.hbm, a.threads, a.route)
    floor, extra = (1.0 if a.full_budget else 0.85), parse_tune_extra(a.tune_extra)
    cands = rule_candidates(ranking(ev, batches=(B,)), a.P, floor)[org.kind]
    seeds = seeds_of(ev, org, a.seeds, a.packed_variants, a.full_budget)
    first = (14, None) if org.kind == "het" else (13, None)
    own, probe = starts(ev, org, a.hbm), []
    for c in cands:
        if not ev.legal([list(x) for x in c["base"]]):
            continue
        shapes = expand(org, c["base"])
        v, e, tag = min((fit(ev, org, shapes, with_policy(s, first), "screen", (B,)) for s in own), key=lambda z: z[0])
        probe.append(dict(base=c["base"], rule=c["rule"], shapes=shapes, probe=v, start=tag, env=asdict(e)))
    for s in seeds:
        v = ev.score(s["shapes"], Environment(**s["env"]), "screen", (B,))
        probe.append(dict(base=s["base"], rule="seed", seed=s["seed"], shapes=s["shapes"], probe=v, start="own", env=s["env"]))
    probe.sort(key=lambda r: r["probe"])

    stage2 = []
    for r in [r for r in probe if r["probe"] < INF][:TOPN]:
        for pol in policies(org):
            e = with_policy(Environment(**r["env"]), pol)
            stage2.append(dict(base=r["base"], rule=r["rule"], seed=r.get("seed"), shapes=r["shapes"], policy=list(pol),
                               screen=ev.score(r["shapes"], e, "screen", (B,)), env=asdict(e)))
        if r["rule"] == "seed":
            e = Environment(**r["env"])
            stage2.append(dict(base=r["base"], rule="seed", seed=r["seed"], shapes=r["shapes"],
                               policy=[e.policy, e.p14_steal], screen=r["probe"], env=r["env"]))
    stage2.sort(key=lambda r: r["screen"])
    top, seen = [], set()
    for r in stage2:
        if r["screen"] < INF and json.dumps(r["base"]) not in seen:
            seen.add(json.dumps(r["base"]))
            top.append(r)
        if len(top) == TOPN:
            break

    tuner = Tuner(ev, org, extra, screen=(B,), select=(B,), main=(B,))
    for r in top:
        tuner.tune(r)
    if org.arrays == 2 and not any(t["valid"] for t in tuner.tuned):
        for r in top:
            tuner.rescue(r)
    for name, path in (x.partition("=")[::2] for x in a.as_is):
        p = Path(path) / f"{org.name}.json" if Path(path).is_dir() else Path(path)
        if p.exists():
            b = rs.read_json(p)["best_main"]
            if not a.full_budget or base_macs(b["base"], org.kind) == a.P:
                tuner.tuned.append(dict(b, rule="as-is", seed=name, source=str(p)))
    return dict(model=a.model, fmt=a.fmt, route=ev.route, P=a.P, org=org.name, batch=B, full_budget=a.full_budget,
                tune_extra=[[k, list(v)] for k, v in extra], n_candidates=len(cands), n_seeds=len(seeds), probe=probe,
                stage2=stage2, tuned=tuner.tuned, best_main=best(tuner.tuned), seconds=time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("org", choices=sorted(ORGS))
    ap.add_argument("batch", type=int)
    ap.add_argument("--full-budget", action="store_true", help="every chip uses exactly P multipliers")
    ap.add_argument("--tune-extra", nargs="*", default=(), help="extra tuning axes, e.g. redeliv_packed=0,1")
    ap.add_argument("--seeds", nargs="*", default=(), help="name=winner[,tuned]: chip record and search result")
    ap.add_argument("--packed-variants", nargs="*", default=(), help="seed name prefixes also run packed")
    ap.add_argument("--as-is", nargs="*", default=(), help="name=chip record joining the tuned list as it is")
    ap.add_argument("--route", default=None, help="routing capture (default: by format)")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    res = perbatch(a)
    rs.write_json(rs.perbatch_path(a.out, rs.cell_name(a.model, a.fmt, a.P, a.hbm), a.batch, a.org), res)
    b = res["best_main"]
    print("perbatch", a.model, a.fmt, a.P, a.org, f"B{a.batch}", b["base"], b["policy"], round(b["select_main"]),
          "valid", b["valid"], f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
