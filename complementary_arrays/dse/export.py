"""Result table of a chip set: per cell the chips, their MoE time per decode step on a window set (microseconds at 1 GHz)
and the speed-up of the pair over two identical arrays, one array and the faster of the two, per batch and as the
geometric mean over B32..B128, with the bootstrap intervals when they exist."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..sim import workload as wl
from . import results as rs
from .organisations import FAMILY


def _gm(xs):
    return wl.geomean(xs) if xs and all(x is not None for x in xs) else None


def _ratio(a, b):
    return a / b if a is not None and b is not None else None


def table(out: Path, version: str, set_name: str = "s1_eval", bootstrap: str | None = None) -> list[dict]:
    rows = []
    root = rs.eval_dir(out, version, "")
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        paths = {o: cell_dir / f"{set_name}_{o}.json" for o in FAMILY}
        if not all(p.exists() for p in paths.values()):
            continue
        recs = {o: rs.read_json(p) for o, p in paths.items()}
        batches = sorted({int(k[1:]) for r in recs.values() for k in r["batches"]})
        step = {o: {B: recs[o]["batches"].get(f"B{B}", {}).get("step_cycles") for B in batches} for o in FAMILY}
        main = [B for B in wl.MAIN_BATCHES if B in batches]
        gm = {o: _gm([step[o][B] for B in main]) for o in FAMILY}
        best = {B: min((x for x in (step["single2"][B], step["hom11"][B]) if x is not None), default=None) for B in batches}
        row = dict(
            cell=cell_dir.name,
            chips={o: dict(base=recs[o]["base"], policy=recs[o]["policy"]) for o in FAMILY},
            step_us={o: {f"B{B}": (v / 1000 if v is not None else None) for B, v in step[o].items()} for o in FAMILY},
            vs_hom={f"B{B}": _ratio(step["hom11"][B], step["het11"][B]) for B in batches},
            vs_single={f"B{B}": _ratio(step["single2"][B], step["het11"][B]) for B in batches},
            vs_best={f"B{B}": _ratio(best[B], step["het11"][B]) for B in batches},
            main=dict(
                vs_hom=_ratio(gm["hom11"], gm["het11"]),
                vs_single=_ratio(gm["single2"], gm["het11"]),
                vs_best=_ratio(min((x for x in (gm["single2"], gm["hom11"]) if x is not None), default=None), gm["het11"]),
            ),
        )
        boot = Path(out) / "bootstrap" / bootstrap / f"{cell_dir.name}.json" if bootstrap else None
        if boot is not None and boot.exists():
            b = rs.read_json(boot)
            row["intervals"] = dict(geomean=b.get("geomean"), batches=b["batches"])
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="final", help="chip set under eval/")
    ap.add_argument("--set", default="s1_eval", help="window set")
    ap.add_argument("--bootstrap", default=None, help="interval set under bootstrap/")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or COMPLEMENTARY_ARRAYS_OUT) is required")
    rows = table(a.out, a.version, a.set, a.bootstrap)
    rs.write_json(Path(a.out) / f"table_{a.version}_{a.set}.json", rows)
    for r in rows:
        m = r["main"]
        print(r["cell"], {o: r["chips"][o]["base"] for o in FAMILY},
              "vs_hom %.3f vs_single %.3f" % (m["vs_hom"] or 0, m["vs_single"] or 0))


if __name__ == "__main__":
    main()
