"""Ablations on fixed chips: every chip of a set keeps its shapes and tuned settings, one mechanism is changed (engine
settings field=value, e.g. policy=13 for two-ended dispatch on every chip, redeliv_packed=0 for re-reads stored as BF16),
and the chips run the eval snapshots of the given batches. Output per variant and organisation: MoE time per decode
step, per-snapshot step times, HBM busy share and the share of cycles with one array idle while the other works."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..dse import results as rs
from ..dse import snapshots as ss
from ..dse.evaluate import chip_setup, parse_overrides
from ..dse.formats import ROUTES
from ..dse.organisations import FAMILY
from ..sim import workload as wl
from ..sim.engine import counters, simulate


def ablate(root: Path, model: str, fmt: str, org: str, chip: dict, overrides: dict, batches, threads: int = 3) -> dict:
    shapes, env = chip_setup(chip, org, overrides)
    res = {}
    for B in batches:
        rows = ss.windows(root, model, ROUTES[fmt], B, "eval")
        c = counters(simulate(np.stack([r["ids"] for r in rows]), shapes, wl.MODELS[model], env, threads=threads))
        cyc = c["cycles"].astype(float)
        layers, snaps = len({r["layer"] for r in rows}), len({r["snapshot"] for r in rows})
        res[f"B{B}"] = dict(
            step_cycles=float(cyc.mean()) * layers,
            mean_window_cycles=float(cyc.mean()),
            per_snapshot=cyc.reshape(snaps, -1).sum(1).tolist(),
            hbm_busy_frac=float(c["hbm_service"].sum()) / (float(cyc.sum()) * env.H),
            idle_while_other_works_frac=float(c["idle_while_other_works"].sum()) / float(cyc.sum())
            if "idle_while_other_works" in c else None,
        )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("variant", help="name of the variant")
    ap.add_argument("overrides", nargs="*", help="engine settings field=value (none: the chips as they are)")
    ap.add_argument("--chips", type=Path, required=True, help="directory with <org>.json chip records")
    ap.add_argument("--version", required=True, help="name of the chip set")
    ap.add_argument("--orgs", default=",".join(FAMILY))
    ap.add_argument("--batches", type=int, nargs="+", default=list(wl.MAIN_BATCHES))
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    cell, overrides = rs.cell_name(a.model, a.fmt, a.P), parse_overrides(a.overrides)
    for org in a.orgs.split(","):
        chip = rs.read_json(a.chips / f"{org}.json")["best_main"]
        res = ablate(a.data, a.model, a.fmt, org, chip, overrides, a.batches, a.threads)
        rs.write_json(Path(a.out) / "mech" / "ablate" / a.version / cell / f"{a.variant}_{org}.json",
                      dict(cell=cell, org=org, version=a.version, variant=a.variant, overrides=overrides,
                           base=chip["base"], policy=chip["policy"], batches=res))
        print(cell, a.version, a.variant, org, {k: round(v["step_cycles"]) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
