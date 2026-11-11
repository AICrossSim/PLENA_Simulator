"""Final two-stream chips of a cell on the sealed request group (Qwen3.5 and DeepSeek, B32..B128)."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import Environment, counters
from . import results as rs
from .evaluate import Evaluator
from .organisations import FAMILIES, ORGS, expand


def sealed(root: Path, out: Path, model: str, fmt: str, P: int, hbm_tbps: int = 8, threads: int = 3) -> dict:
    t0, cell = time.time(), rs.cell_name(model, fmt, P, hbm_tbps)
    res = dict(model=model, fmt=fmt, P=P, batches={}, bases={}, envs={})
    ev = Evaluator(root, model, fmt, P, hbm_tbps, threads)
    rows = {B: wl.sealed_windows(root, model, B) for B in wl.MAIN_BATCHES}
    for B, rr in rows.items():
        res["batches"][B] = dict(
            weights=wl.window_weights(rr).tolist(),
            layers=[r["layer"] for r in rr],
            groups=[r["group"] for r in rr],
            sources=[r["source"] for r in rr],
            cycles={},
        )
    for o in FAMILIES[2]:
        chip = rs.read_json(rs.final_path(out, cell, o))["best_main"]
        res["bases"][o], res["envs"][o] = chip["base"], chip["env"]
        shapes = expand(ORGS[o], chip["base"])
        env = replace(Environment.from_dict(chip["env"]), shared_array=ORGS[o].shared_array)
        for B, rr in rows.items():
            cyc = np.zeros(len(rr))
            for ix, engine_out in ev.outputs(rr, shapes, env):
                cyc[ix] = counters(engine_out)["cycles"]
            res["batches"][B]["cycles"][o] = cyc.tolist()
    res["seconds"] = time.time() - t0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.SEALED_DIRS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int)
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    res = sealed(a.data, a.out, a.model, a.fmt, a.P, threads=a.threads)
    rs.write_json(rs.sealed_path(a.out, rs.cell_name(a.model, a.fmt, a.P)), res, indent=None)
    print("sealed", a.model, a.fmt, a.P, f"{res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
