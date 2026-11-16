"""Two-stage bootstrap of the speed-up of the pair over one array and over two identical arrays.

One replicate: (1) within each source the eval requests are resampled with replacement (same count; copies count as
separate requests); (2) new snapshots (16 per batch) are drawn from the resampled pool with the snapshot rule; (3) the
three chips run them, one engine call per batch. A 16-snapshot replicate carries the request variance in full and four
times the snapshot variance of the 64-snapshot estimate, so the interval is conservative. Statistic: MoE time per decode
step per batch; over several batches the geometric mean, with the faster rival chosen on it. 95% percentile intervals.
A chip that does not hold a batch is recorded as infeasible there.
usage: bootstrap.py run ... --first i --last j (replicates i..j), then bootstrap.py merge ...
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import CapacityError
from . import results as rs
from . import snapshots as ss
from .evaluate import Evaluator, chip_setup
from .formats import ROUTES
from .organisations import FAMILY

SNAPSHOTS = 16
RIVALS = {"vs_hom": ("hom11",), "vs_single": ("single2",), "vs_best": ("single2", "hom11")}


def resample(pool: ss.Pool, label: str, model: str, route: str, rep: int, batches, n_snap: int):
    """Snapshot windows [n_snap, layers, B, topk] per batch of one replicate, and its distinct request count."""
    rng = np.random.default_rng(ss.seed_of(label, model, route, rep))
    idx = pool.drawable("eval")
    src = pool.sources[idx]
    ent = np.concatenate(
        [rng.choice(idx[src == s], int((src == s).sum()), replace=True) for s in sorted(set(src.tolist()))]
    )
    w = ss.s1_weights(pool.sources[ent], pool.lengths[ent])
    wins = {}
    for B in batches:
        req, pos = ss.draw_snapshots(w, pool.lengths[ent], B, n_snap, rng)
        wins[B] = ss.materialize(pool, ent[req], pos)
    return wins, int(len(set(ent.tolist())))


def run(a, chips: dict, cell: str, route: str) -> None:
    pool = ss.load_pool(a.captures, a.model, route)
    ev = Evaluator(a.data, a.model, a.fmt, a.P, a.hbm, a.threads, route)
    folder = Path(a.out) / "bootstrap" / a.name / cell / f"n{a.snapshots}"
    for rep in range(a.first, a.last + 1):
        path = folder / f"rep_{rep:04d}.json"
        if path.exists():
            continue
        t0 = time.time()
        wins, distinct = resample(pool, a.seed_label, a.model, route, rep, a.batches, a.snapshots)
        res = dict(rep=rep, cell=cell, nsnap=a.snapshots, distinct_requests=distinct, batches={})
        for org, chip in chips.items():
            shapes, env = chip_setup(chip, org)
            for B in a.batches:
                rows = [dict(ids=x, live=B) for x in wins[B].reshape(-1, B, pool.topk)]
                try:
                    cyc, _ = ev.run(rows, shapes, env)
                except (CapacityError, RuntimeError, ValueError) as ex:
                    res["batches"].setdefault(f"B{B}", {})[org] = dict(infeasible=True, why=str(ex)[:200])
                    continue
                per = cyc.reshape(a.snapshots, -1).sum(1)
                res["batches"].setdefault(f"B{B}", {})[org] = dict(step_cycles=float(per.mean()), per_snapshot=per.tolist())
        res["seconds"] = time.time() - t0
        rs.write_json(path, res, indent=None)
        print("bootstrap", cell, a.name, "replicate", rep, f"{res['seconds']:.0f}s", flush=True)


def interval(x) -> dict | None:
    x = np.asarray(x, float)
    return dict(ci95=[float(v) for v in np.percentile(x, [2.5, 97.5])], replicates=int(len(x)),
                replicate_median=float(np.median(x)), replicate_sd_log=float(np.std(np.log(x)))) if len(x) else None


def merge(reps: list[dict], batches) -> dict:
    """Per batch and over all batches (geometric mean; replicates where every chip holds every batch)."""
    T = {B: {o: np.array([r["batches"][f"B{B}"][o].get("step_cycles", np.nan) for r in reps]) for o in FAMILY}
         for B in batches}
    out = dict(replicates=len(reps), batches={})
    for B in batches:
        t = T[B]
        out["batches"][f"B{B}"] = {}
        for name, rivals in RIVALS.items():
            x = np.min([t[o] for o in rivals], axis=0) / t["het11"]
            ok = np.isfinite(x)
            out["batches"][f"B{B}"][name] = dict(interval(x[ok]) or {}, infeasible=int((~ok).sum()))
    if len(batches) > 1:
        G = {o: np.exp(np.mean([np.log(T[B][o]) for B in batches], axis=0)) for o in FAMILY}
        out["geomean"] = {}
        for name, rivals in RIVALS.items():
            x = np.min([G[o] for o in rivals], axis=0) / G["het11"]
            out["geomean"][name] = interval(x[np.isfinite(x)])
    return out


def chip_records(specs, default: Path | None) -> dict:
    """org=directory (or file) per organisation, the others from the default directory."""
    given = dict(s.split("=", 1) for s in specs or ())
    out = {}
    for org in FAMILY:
        p = Path(given.get(org, default))
        out[org] = rs.read_json(p / f"{org}.json" if p.is_dir() else p)["best_main"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("run", "merge"))
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("--name", required=True, help="name of the interval set, e.g. final")
    ap.add_argument("--chips", type=Path, help="directory with the <org>.json chip records")
    ap.add_argument("--chip", nargs="*", default=(), help="org=chip record or directory, overriding --chips")
    ap.add_argument("--batches", type=int, nargs="+", default=list(wl.MAIN_BATCHES))
    ap.add_argument("--seed-label", default="boot2", help="label of the replicate seeds")
    ap.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=999)
    ap.add_argument("--captures", type=Path, help="directory of the routing capture chunks (run)")
    ap.add_argument("--route", default=None, help="routing capture (default: by format)")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or COMPLEMENTARY_ARRAYS_OUT) is required")
    cell, route = rs.cell_name(a.model, a.fmt, a.P, a.hbm), a.route or ROUTES[a.fmt]
    if a.cmd == "run":
        if a.captures is None or a.data is None:
            ap.error("run needs --captures and --data")
        run(a, chip_records(a.chip, a.chips), cell, route)
        return
    folder = Path(a.out) / "bootstrap" / a.name / cell / f"n{a.snapshots}"
    reps = [rs.read_json(p) for p in sorted(folder.glob("rep_*.json"))]
    res = dict(cell=cell, name=a.name, nsnap=a.snapshots, batches_used=list(a.batches), **merge(reps, a.batches))
    rs.write_json(Path(a.out) / "bootstrap" / a.name / f"{cell}.json", res)
    print(cell, a.name, len(reps), "replicates", {k: v["ci95"] for k, v in (res.get("geomean") or {}).items() if v})


if __name__ == "__main__":
    main()
