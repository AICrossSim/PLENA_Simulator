"""Design-space search of one organisation in one cell.

Stages (output search/<stage>/<cell>/<org>.json):
  top10   the organisation's rule candidates (candidates.py)
  hetopt  pair only: rule candidates plus the 200 best pairs of every height (analytic model with the drain latency)
  hetx<i> pair only: shard i of the wide pair pool (rule candidates, the 600 best pairs with six cores and partners per
          pair of heights and split, the three best pairs of every pair of heights, and the shapes probed by given
          stages); probe and policy menu start from packed re-reads
  prior   given chips (earlier designs) converted to the cell's weight format and, under the equal-multiplier rule,
          promoted to the whole budget: one coordinate pass each from their own settings, then the select windows
  retune<i> part i of the best K distinct valid shapes of given stages, tuned again from their own settings and from
          those of a reference chip, passes until one gives no gain
Funnel of top10, hetopt and hetx: probe every legal candidate on the screen windows of B32 and B128 from the template
start -> the best 10 under every dispatch policy on the screen windows of B32..B128 -> the best 10 distinct shapes get
one coordinate pass and are measured on the select windows (B2..B128) -> while no tuned chip is valid, or the best valid
one is more than 3% slower than the best tuned one, the next shapes of the ranked lists are tuned too (at most 30).
The chip of a stage is the valid one (every array works in every batch) with the lowest geometric mean over B32..B128
(best_main) or B2..B128 (best_all) on the select windows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..sim import workload as wl
from ..sim.engine import Environment, drain_latency
from . import results as rs
from . import snapshots as ss
from .candidates import Ranking, base_macs, extend, extension_pairs, promote, rule_candidates, wide_pairs
from .evaluate import INF, Evaluator, chip_setup, sram_mib
from .formats import TIMING, kappa, with_format
from .organisations import ORGS, Organisation, expand
from .shapes import AnalyticModel
from .tune import PROBE_BATCHES, SCREEN_BATCHES, coordinate_pass, fit, policies, starts, valid, with_policy

TOPN = 10
WALK_TOLERANCE, WALK_LIMIT = 1.03, 30
RETUNE_PASSES = 3


def _key(base) -> str:
    return json.dumps(base)


def ranking(ev: Evaluator, drain: bool = False, batches=(32, 128)) -> Ranking:
    """Analytic pre-ranking on the dev snapshots of the batches; drain adds the exposed drain latency of every task."""
    extra = {}
    if drain:
        env = Environment(**TIMING)
        extra = dict(drain=lambda n: drain_latency(n, env), expert_drains=1)
    model = AnalyticModel(
        lambda B: ss.histograms(ev.root, ev.model, ev.route, B), ev.model, ev.fmt, hbm=float(ev.hbm), **extra
    )
    return Ranking(model, kappa(ev.fmt, ev.hbm), batches)


def candidates(ev: Evaluator, org: Organisation, stage: str, floor: float, out: Path, probes=()) -> tuple[list, dict]:
    """Candidates of a funnel stage and the counts recorded with its result."""
    rank = ranking(ev)
    rules = rule_candidates(rank, ev.P, floor)
    cands, info = list(rules[org.kind]), dict(rule_candidates=rules["C"])
    if stage == "top10":
        return cands, info
    if org.kind != "het":
        raise ValueError(f"{stage} needs the pair organisation")
    lat = ranking(ev, drain=True)
    if stage == "hetopt":
        pairs, ranked = extension_pairs(lat, ev.P, keep_small_n=True, floor=floor)
        info.update(n_pairs_generated=len(pairs), n_extended=extend(cands, lat, ranked))
        return cands, info
    n_pairs, ranked, per_heights = wide_pairs(lat, ev.P, floor)
    info.update(n_pairs_generated=n_pairs, n_ranked=extend(cands, lat, ranked, "wide"))
    info["n_heights"] = extend(cands, lat, per_heights, "heights")
    have = {_key(c["base"]) for c in cands}
    for stage_name in probes:
        for r in rs.read_json(rs.search_path(out, stage_name, rs.cell_name(ev.model, ev.fmt, ev.P), org.name))["probe"]:
            if _key(r["base"]) not in have:
                have.add(_key(r["base"]))
                cands.append(dict(base=r["base"], rule=f"probed-{stage_name}"))
    info["n_pool"] = len(cands)
    return cands, info


class Tuner:
    """Coordinate tuning of chips on the screen windows and their measurement on the select windows."""

    def __init__(self, ev: Evaluator, org: Organisation, extra=(), screen=SCREEN_BATCHES, select=wl.BATCHES,
                 main=wl.MAIN_BATCHES):
        self.ev, self.org, self.extra, self.tuned = ev, org, tuple(extra), []
        self.screen, self.select, self.main = screen, select, main

    def score(self, shapes, e: Environment) -> float:
        return self.ev.score(shapes, e, "screen", self.screen)

    def coordinate(self, shapes, e: Environment, v: float, passes: int = 1) -> tuple[Environment, float]:
        return coordinate_pass(self.ev, shapes, e, v, "screen", self.screen, self.extra, passes)

    def measure(self, r: dict, e: Environment, v: float, **more) -> dict:
        sel = self.ev.per_window(r["shapes"], e, "select", self.select)
        ok = {k: x for k, x in sel.items() if not x.get("infeasible")}
        feasible = len(ok) == len(sel)
        rec = dict(
            r,
            tuned=v,
            env=asdict(e),
            select={k: x["cycles"] for k, x in ok.items()},
            select_busy={k: x["busy"] for k, x in ok.items()},
            select_infeasible=[k for k in sel if k not in ok],
            valid=feasible and valid(self.org, sel),
            select_main=wl.geomean([sel[f"B{B}"]["cycles"] for B in self.main]) if feasible else INF,
            select_all=wl.geomean([sel[f"B{B}"]["cycles"] for B in self.select]) if feasible else INF,
            **more,
        )
        self.tuned.append(rec)
        return rec

    def tune(self, r: dict, passes: int = 1, **more) -> dict:
        e, v = self.coordinate(r["shapes"], Environment(**r["env"]), r["screen"], passes)
        return self.measure(r, e, v, **more)

    def rescue(self, r: dict) -> None:
        """Retune from a balanced SRAM split, halving the slots until the chip fits."""
        for share in (500, 375, 625):
            e = replace(Environment(**r["env"]), share=share)
            v = self.score(r["shapes"], e)
            while v == INF and (e.slots > 2 or e.stage_slots > 2):
                e = replace(e, slots=max(2, e.slots // 2), stage_slots=max(2, e.stage_slots // 2))
                v = self.score(r["shapes"], e)
            if v < INF:
                self.tune(dict(r, env=asdict(e), screen=v), rescued=True)
                return


def best(tuned: list[dict], metric: str = "select_main") -> dict | None:
    """The fastest valid chip, else the fastest one."""
    ranked = sorted([t for t in tuned if t["valid"]], key=lambda t: t[metric]) or sorted(tuned, key=lambda t: t[metric])
    return ranked[0] if ranked else None


def funnel(ev: Evaluator, org: Organisation, cands: list, hbm_tbps: int, start_fields: dict, extra=(), topn=TOPN) -> dict:
    first = (14, None) if org.kind == "het" else (13, None)
    own = starts(ev, org, hbm_tbps, start_fields)
    probe = []
    for c in cands:
        if not ev.legal([list(x) for x in c["base"]]):
            continue
        shapes = expand(org, c["base"])
        v, e, tag = min(
            (fit(ev, org, shapes, with_policy(s, first), "screen", PROBE_BATCHES) for s in own), key=lambda z: z[0]
        )
        probe.append(dict(base=c["base"], rule=c["rule"], shapes=shapes, probe=v, start=tag, env=asdict(e)))
    probe.sort(key=lambda r: r["probe"])

    stage2 = []
    for r in [r for r in probe if r["probe"] < INF][: max(6, topn)]:
        for pol in policies(org):
            e = with_policy(Environment(**r["env"]), pol)
            screen = ev.score(r["shapes"], e, "screen", SCREEN_BATCHES)
            stage2.append(dict(base=r["base"], rule=r["rule"], shapes=r["shapes"], policy=list(pol), screen=screen, env=asdict(e)))
    stage2.sort(key=lambda r: r["screen"])
    top, seen = [], set()
    for r in stage2:
        if r["screen"] < INF and _key(r["base"]) not in seen:
            seen.add(_key(r["base"]))
            top.append(r)
        if len(top) == topn:
            break

    tuner = Tuner(ev, org, extra)
    for r in top:
        tuner.tune(r)
    if org.arrays == 2 and not any(t["valid"] for t in tuner.tuned):
        for r in top:
            tuner.rescue(r)

    def need_more():
        ok = [t["select_main"] for t in tuner.tuned if t["valid"]]
        return not ok or min(ok) > WALK_TOLERANCE * min(t["select_main"] for t in tuner.tuned)

    if need_more() and len(tuner.tuned) < WALK_LIMIT:
        more = [x for x in stage2 if x["screen"] < INF] + [
            dict(r, policy=list(first), screen=r["probe"]) for r in probe if r["probe"] < INF
        ]
        for r in more:
            if _key(r["base"]) in seen:
                continue
            seen.add(_key(r["base"]))
            tuner.tune(r)
            if org.arrays == 2 and not tuner.tuned[-1]["valid"]:
                tuner.rescue(r)
            if not need_more() or len(tuner.tuned) >= WALK_LIMIT:
                break
    return dict(probe=probe, stage2=stage2, tuned=tuner.tuned)


def read_seeds(specs, org: str) -> list[tuple[str, Path]]:
    """name=path or path; a directory stands for its <org>.json."""
    out = []
    for spec in specs:
        name, _, path = spec.rpartition("=")
        path = Path(path)
        path = path / f"{org}.json" if path.is_dir() else path
        out.append((name or str(path.parent.name), path))
    return out


def prior_candidates(ev: Evaluator, org: Organisation, seeds, full_budget: bool, packed_variants=()) -> tuple[list, int]:
    """Seed chips in the cell's format (seeds named with a prefix in packed_variants also with packed re-reads), deduplicated,
    and under the equal-multiplier rule those using P plus the promoted ones. Returns the candidates and the seed count."""
    cands, seen = [], set()
    for name, path in seeds:
        if not path.exists():
            print("missing seed", path, flush=True)
            continue
        b = rs.read_json(path)["best_main"]
        _, e = chip_setup(b, org.name)
        e = with_format(e, ev.fmt, ev.hbm)
        variants = [(name, e)]
        if any(name.startswith(p) for p in packed_variants) and e.redeliv_packed == 0:
            variants.append((name + "-P", replace(e, redeliv_packed=1)))
        if _key(b["base"]) + json.dumps(asdict(e), sort_keys=True) in seen:
            continue
        for label, e in variants:
            key = _key(b["base"]) + json.dumps(asdict(e), sort_keys=True)
            if key not in seen:
                seen.add(key)
                cands.append(dict(prior=label, base=b["base"], shapes=expand(org, b["base"]), policy=b["policy"],
                                  env=asdict(e), source=str(path)))
    n_seeds, promoted = len(cands), []
    if full_budget:
        for r in cands:
            for nb in promote(org.kind, r["base"], ev.P):
                key = _key(nb) + json.dumps(r["env"], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    promoted.append(dict(r, prior=r["prior"] + "-up", base=nb, shapes=expand(org, nb), promoted_from=r["base"]))
        cands = [r for r in cands if base_macs(r["base"], org.kind) == ev.P] + promoted
    return cands, n_seeds


def prior(ev: Evaluator, org: Organisation, seeds, full_budget: bool, packed_variants=(), extra=()) -> dict:
    cands, n_seeds = prior_candidates(ev, org, seeds, full_budget, packed_variants)
    tuner = Tuner(ev, org, extra)
    for r in cands:
        e = Environment(**r["env"])
        r = dict(r, screen=tuner.score(r["shapes"], e))
        e, v = tuner.coordinate(r["shapes"], e, r["screen"])
        if v == INF:
            tuner.tuned.append(dict(r, tuned=v, env=asdict(e), select={}, select_busy={}, valid=False, select_main=INF, select_all=INF))
        else:
            tuner.measure(r, e, v)
    return dict(tuned=tuner.tuned, n_seeds=n_seeds, n_candidates=len(cands))


def retune(ev: Evaluator, org: Organisation, out: Path, stages, reference: Path, K: int, part: int, nparts: int, extra=()) -> dict:
    cell = rs.cell_name(ev.model, ev.fmt, ev.P)
    pool = []
    for stage in stages:
        path = rs.search_path(out, stage, cell, org.name)
        pool += [dict(t, src=stage) for t in rs.read_json(path)["tuned"] if t["valid"]]
    pool.sort(key=lambda t: t["select_main"])
    top, seen = [], set()
    for t in pool:
        if _key(t["base"]) not in seen:
            seen.add(_key(t["base"]))
            top.append(t)
        if len(top) == K:
            break
    ref = rs.read_json(reference)["best_main"]
    _, eref = chip_setup(ref, org.name)
    tuner = Tuner(ev, org, extra)
    for t in top[part::nparts]:
        shapes, own = expand(org, t["base"]), Environment(**t["env"])
        ref_policy = (eref.policy, eref.p14_steal if eref.policy in (14, 16) else None)
        options = [("own", own)] + [
            (f"ref-pol{p[0]}", with_policy(eref, p))
            for p in {tuple(t["policy"]) if t["policy"][1] is not None else (t["policy"][0], None), ref_policy}
        ]
        found = None
        for name, e0 in options:
            e, v = tuner.coordinate(shapes, e0, tuner.score(shapes, e0), RETUNE_PASSES)
            if found is None or v < found[0]:
                found = (v, e, name)
        if found is None or found[0] == INF:
            continue
        v, e, name = found
        r = dict(base=t["base"], shapes=shapes, policy=[e.policy, e.p14_steal if e.policy in (14, 16) else None],
                 start=name, from_stage=t["src"], stage_select_main=t["select_main"])
        tuner.measure(r, e, v)
    return dict(tuned=tuner.tuned, K=K, part=part, nparts=nparts, stages=list(stages), reference=str(reference))


def hetx_pool(ev: Evaluator, org: Organisation, floor: float, out: Path, probes) -> tuple[list, dict]:
    """The wide pair pool, built once per cell and kept under search/hetx_pool/."""
    path = Path(out) / "search" / "hetx_pool" / f"{rs.cell_name(ev.model, ev.fmt, ev.P)}.json"
    if path.exists():
        x = rs.read_json(path)
        return x["cands"], x["info"]
    cands, info = candidates(ev, org, "hetx", floor, out, probes)
    rs.write_json(path, dict(cands=cands, info=info))
    return cands, info


def parse_tune_extra(items) -> tuple:
    return tuple((k, tuple(int(x) for x in v.split(","))) for k, v in (item.split("=", 1) for item in items or ()))


def search(a) -> dict:
    t0, org = time.time(), ORGS[a.org]
    ev = Evaluator(a.data, a.model, a.fmt, a.P, a.hbm, a.threads, a.route)
    floor, extra = (1.0 if a.full_budget else 0.85), parse_tune_extra(a.tune_extra)
    info = dict(stage=a.stage, model=a.model, fmt=a.fmt, route=ev.route, hbm_tbps=a.hbm, P=a.P, sram_mib=sram_mib(a.P),
                org=org.name, full_budget=a.full_budget, tune_extra=[[k, list(v)] for k, v in extra],
                policies=[list(p) for p in policies(org)])
    if a.stage == "prior":
        res = prior(ev, org, read_seeds(a.seeds, org.name), a.full_budget, a.packed_variants, extra)
    elif a.stage.startswith("retune"):
        res = retune(ev, org, a.out, a.from_stages, a.reference, a.K, int(a.stage[6:] or 0), a.parts, extra)
    elif a.stage.startswith("hetx"):
        cands, pool = hetx_pool(ev, org, floor, a.out, a.probes)
        shard = int(a.stage[4:] or 0)
        mine = cands[shard :: a.parts]
        res = funnel(ev, org, mine, a.hbm, dict(redeliv_packed=1), extra)
        res.update(pool=pool, shard=shard, nshards=a.parts, n_candidates=len(mine))
    else:
        cands, more = candidates(ev, org, a.stage, floor, a.out)
        res = funnel(ev, org, cands, a.hbm, {}, extra)
        res.update(more, n_candidates=len(cands))
    if a.full_budget:
        assert all(base_macs(t["base"], org.kind) == a.P for t in res["tuned"]), "a candidate is short of the budget"
    return dict(info, **res, best_main=best(res["tuned"]), best_all=best(res["tuned"], "select_all"),
                rejects=ev.rejects[:50], seconds=time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("org", choices=sorted(ORGS))
    ap.add_argument("--stage", default="top10", help="top10, hetopt, hetx<i>, prior or retune<i>")
    ap.add_argument("--full-budget", action="store_true", help="every chip uses exactly P multipliers")
    ap.add_argument("--tune-extra", nargs="*", default=(), help="extra tuning axes, e.g. redeliv_packed=0,1")
    ap.add_argument("--parts", type=int, default=1, help="hetx shards / retune parts")
    ap.add_argument("--probes", nargs="*", default=(), help="hetx: stages whose probed shapes join the pool")
    ap.add_argument("--seeds", nargs="*", default=(), help="prior: [name=]chip file or directory of <org>.json")
    ap.add_argument("--packed-variants", nargs="*", default=(), help="prior: seed name prefixes also run packed")
    ap.add_argument("--from-stages", nargs="*", default=(), help="retune: stages whose valid tuned chips are re-tuned")
    ap.add_argument("--reference", type=Path, help="retune: chip record whose settings are the second start")
    ap.add_argument("-K", type=int, default=6, help="retune: distinct shapes re-tuned")
    ap.add_argument("--route", default=None, help="routing capture (default: by format)")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    res = search(a)
    rs.write_json(rs.search_path(a.out, a.stage, rs.cell_name(a.model, a.fmt, a.P, a.hbm), a.org), res)
    b = res["best_main"] or {}
    print(a.stage, a.model, a.fmt, a.P, a.org, "tuned", len(res["tuned"]), "best", b.get("base"), b.get("policy"),
          round(b.get("select_main", INF), 1), "valid", b.get("valid"), f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
