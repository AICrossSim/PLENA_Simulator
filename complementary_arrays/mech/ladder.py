"""Weight-port ladder: layer cycles of the final two-stream chips of a cell when only the weight-delivery model changes.

Levels: A shared port unlimited (W = 1 MiB/cycle), B shared port 32 KB/cycle, C separate 16 KB/cycle first-pass port
with re-delivery limited to kappa times the HBM weight rate (port_mode 0), D the reported chip (shared 16 KB/cycle
port). Fixed: shapes, policy and every tuned setting of the reported chip. Retuned (levels A and B): dispatch policy
sweep and one coordinate pass of the search's tuning space on the dev screen windows. Heldout report windows,
B32..B128; the ratios compare het with hom, single and the faster of the two.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from ..dse import results as rs
from ..dse.evaluate import Evaluator
from ..dse.search import KAPPA
from ..dse.organisations import FAMILIES, ORGS, expand
from ..dse.tune import coordinate_pass, policies, with_policy
from ..sim import workload as wl
from .stall import chip_environment, final_chip

FIXED_BATCHES = (32, 64, 128)
RETUNED_FIELDS = ("policy", "p14_steal", "share", "share2", "lookahead", "slots", "stage_slots", "vslots")


def levels(fmt: str) -> dict:
    return dict(A=dict(W=1048576), B=dict(W=32768), C=dict(port_mode=0, kappa_reread=KAPPA[fmt], W=16384))


def chip_ladder(root: Path, out: Path, model: str, fmt: str, P: int, org: str, threads: int = 3) -> dict:
    """Fixed levels A..D at B32..B128 (D at B96 is the result table) and the retuned levels A and B of one chip."""
    cell = rs.cell_name(model, fmt, P)
    meta, chip = final_chip(out, cell, org)
    ev = Evaluator(root, model, fmt, P, threads=threads)
    o, shapes, env = ORGS[org], expand(ORGS[org], chip["base"]), chip_environment(org, chip)
    if env.port_mode != 1 or env.W != 16384:
        raise ValueError("the ladder starts from a shared 16 KB/cycle port")
    res = dict(meta, cell=cell, org=org, base=chip["base"], policy=chip["policy"], fixed={}, retuned={})
    for name, change in dict(levels(fmt), D={}).items():
        rep = ev.per_window(shapes, replace(env, **change), "report", FIXED_BATCHES)
        res["fixed"][name] = {f"B{B}": rep[f"B{B}"]["cycles"] for B in FIXED_BATCHES}
        res["fixed"][name]["B96"] = (
            ev.per_window(shapes, replace(env, **change), "report", (96,))["B96"]["cycles"]
            if change
            else chip["report"]["B96"]["cycles"]
        )
    res["D_matches_main_table"] = all(
        abs(res["fixed"]["D"][f"B{B}"] - chip["report"][f"B{B}"]["cycles"]) <= 1e-6 * chip["report"][f"B{B}"]["cycles"]
        for B in FIXED_BATCHES
    )
    for name in ("A", "B"):
        start = replace(env, **levels(fmt)[name])
        ev.evaluations = 0
        v, e = min(
            (
                (ev.score(shapes, with_policy(start, p), "screen", wl.MAIN_BATCHES), with_policy(start, p))
                for p in policies(o)
            ),
            key=lambda z: z[0],
        )
        e, v = coordinate_pass(ev, o, shapes, e, v, "screen", wl.MAIN_BATCHES)
        rep = ev.per_window(shapes, e, "report", wl.MAIN_BATCHES)
        res["retuned"][name] = dict(
            {f"B{B}": rep[f"B{B}"]["cycles"] for B in wl.MAIN_BATCHES},
            B32_128=wl.geomean([rep[f"B{B}"]["cycles"] for B in wl.MAIN_BATCHES]),
            busy={f"B{B}": rep[f"B{B}"]["busy"] for B in wl.MAIN_BATCHES},
            screen=v,
            n_screen_evals=ev.evaluations,
            env_changes={
                f: [getattr(env, f), getattr(e, f)] for f in RETUNED_FIELDS if getattr(e, f) != getattr(env, f)
            },
        )
    return res


def merge(out: Path, model: str, fmt: str, P: int) -> dict:
    cell, fam = rs.cell_name(model, fmt, P), FAMILIES[2]
    parts = {o: rs.read_json(Path(out) / "mech" / "ladder" / "chips" / f"{cell}_{o}.json") for o in fam}
    res = dict(
        cell=cell,
        model=model,
        fmt=fmt,
        P=P,
        chips={},
        ratios={},
        levels=dict(
            A="shared port unlimited (W = 1 MiB/cycle)",
            B="shared port 32 KB/cycle",
            C=f"separate 16 KB/cycle first-pass port, re-delivery limited to {KAPPA[fmt]} x the HBM weight rate",
            D="shared port 16 KB/cycle (the reported chip)",
        ),
    )
    for o, p in parts.items():
        fixed = {k: dict(v, B32_128=wl.geomean([v[f"B{B}"] for B in wl.MAIN_BATCHES])) for k, v in p["fixed"].items()}
        res["chips"][o] = dict(
            base=p["base"],
            selected_from=p["selected_from"],
            fixed=fixed,
            retuned=dict(p["retuned"], C=fixed["C"], D=fixed["D"]),
        )
    for setting in ("fixed", "retuned"):
        res["ratios"][setting] = {}
        for k in "ABCD":
            g = {o: res["chips"][o][setting][k] for o in fam}
            res["ratios"][setting][k] = {
                key: dict(
                    het_over_hom=g["hom11"][key] / g["het11"][key],
                    het_over_single=g["single2"][key] / g["het11"][key],
                    het_over_best=min(g["hom11"][key], g["single2"][key]) / g["het11"][key],
                )
                for key in [f"B{B}" for B in wl.MAIN_BATCHES] + ["B32_128"]
            }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("chip", "merge"))
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int)
    ap.add_argument("org", nargs="?", choices=FAMILIES[2])
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.out is None or (a.step == "chip" and (a.data is None or a.org is None)):
        ap.error("chip needs --data, --out and an organisation; merge needs --out")
    cell = rs.cell_name(a.model, a.fmt, a.P)
    if a.step == "chip":
        res = chip_ladder(a.data, a.out, a.model, a.fmt, a.P, a.org, a.threads)
        rs.write_json(Path(a.out) / "mech" / "ladder" / "chips" / f"{cell}_{a.org}.json", res)
        print(
            cell,
            a.org,
            {k: round(res["retuned"][k]["B32_128"]) for k in res["retuned"]},
            "D matches",
            res["D_matches_main_table"],
            flush=True,
        )
    else:
        res = merge(a.out, a.model, a.fmt, a.P)
        rs.write_json(Path(a.out) / "mech" / "ladder" / f"{cell}.json", res)
        print(cell, {k: round(res["ratios"]["retuned"][k]["B32_128"]["het_over_hom"], 4) for k in "ABCD"}, flush=True)


if __name__ == "__main__":
    main()
