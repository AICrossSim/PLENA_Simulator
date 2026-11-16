"""Final chip of one organisation: the fastest valid chip on the dev select windows (geometric mean over B32..B128) among
the best chips of the given search stages, chips taken as they are with their recorded select values, and chips
measured as they are (converted to the cell's weight format) on the select windows. A tie keeps the earlier candidate.
With --budget every candidate must use exactly P multipliers: stage chips are checked, other chips short of P are left
out."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from ..sim import workload as wl
from . import results as rs
from .candidates import base_macs
from .evaluate import Evaluator, chip_setup
from .formats import with_format
from .organisations import ORGS
from .search import read_seeds
from .tune import valid

FIELDS = ("base", "shapes", "policy", "env", "select", "select_busy", "valid", "select_main")


def candidate(b: dict) -> dict:
    return dict({k: b[k] for k in FIELDS}, **({"prior": b["prior"]} if "prior" in b else {}))


def measured(ev: Evaluator, org: str, chip: dict) -> dict:
    """A chip as it is on the select windows."""
    shapes, e = chip_setup(chip, org)
    e = with_format(e, ev.fmt, ev.hbm)
    sel = ev.per_window(shapes, e, "select", wl.BATCHES)
    if any(x.get("infeasible") for x in sel.values()):
        return dict(base=chip["base"], shapes=shapes, policy=chip["policy"], env=asdict(e), select={}, select_busy={},
                    valid=False, select_main=float("inf"))
    return dict(base=chip["base"], shapes=shapes, policy=chip["policy"], env=asdict(e),
                select={k: v["cycles"] for k, v in sel.items()}, select_busy={k: v["busy"] for k, v in sel.items()},
                valid=valid(ORGS[org], sel), select_main=wl.geomean([sel[f"B{B}"]["cycles"] for B in wl.MAIN_BATCHES]))


def select(org: str, stages, as_is=(), measure=(), ev: Evaluator | None = None, budget: int | None = None) -> dict:
    """stages, as_is, measure: (name, chip file) pairs; stage files hold a search result, the others a chip record."""
    kind, X = ORGS[org].kind, []
    for name, path in stages:
        b = rs.read_json(path).get("best_main")
        if b is None:
            continue
        if budget is not None and base_macs(b["base"], kind) != budget:
            raise ValueError(f"{name}: {b['base']} does not use {budget} multipliers")
        X.append((name, candidate(b), str(path)))
    for name, path in list(as_is) + list(measure):
        if not path.exists():
            continue
        b = rs.read_json(path)["best_main"]
        if budget is not None and base_macs(b["base"], kind) != budget:
            print(f"{name}: {b['base']} does not use {budget} multipliers, left out", flush=True)
            continue
        X.append((name, candidate(b) if (name, path) in as_is else measured(ev, org, b), str(path)))
    name, b, path = min(X, key=lambda z: (not z[1]["valid"], z[1]["select_main"]))
    return dict(
        best_main=dict(b, selected_from=name),
        selected_from=name,
        source=path,
        candidates_select_main={k: x["select_main"] for k, x, _ in X},
        candidates_valid={k: x["valid"] for k, x, _ in X},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("org", choices=sorted(ORGS))
    ap.add_argument("--stages", nargs="+", required=True,
                    help="stage names under search/ or name=file (a directory stands for its <org>.json)")
    ap.add_argument("--as-is", nargs="*", default=(), help="[name=]chip record taken with its select values")
    ap.add_argument("--measure", nargs="*", default=(), help="[name=]chip record measured on the select windows")
    ap.add_argument("--budget", action="store_true", help="every candidate uses exactly P multipliers")
    ap.add_argument("--name", default="final", help="output folder of the chosen chips")
    ap.add_argument("--route", default=None, help="routing capture (default: by format)")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.out is None or (a.measure and a.data is None):
        ap.error("--out (and --data with --measure) are required (or COMPLEMENTARY_ARRAYS_OUT / _DATA)")
    cell = rs.cell_name(a.model, a.fmt, a.P, a.hbm)
    stages = [(s, rs.search_path(a.out, s, cell, a.org)) if "=" not in s else read_seeds([s], a.org)[0] for s in a.stages]
    ev = Evaluator(a.data, a.model, a.fmt, a.P, a.hbm, a.threads, a.route) if a.measure else None
    res = select(a.org, stages, read_seeds(a.as_is, a.org), read_seeds(a.measure, a.org), ev, a.P if a.budget else None)
    rs.write_json(rs.final_path(a.out, cell, a.org, a.name), res)
    b = res["best_main"]
    print(a.name, cell, a.org, res["selected_from"], b["base"], b["policy"], round(b["select_main"]),
          {k: round(v) for k, v in res["candidates_select_main"].items()}, flush=True)


if __name__ == "__main__":
    main()
