"""Final chips with every setting frozen, on the heldout report windows under other engine settings (e.g. a systolic
timing mode of the arrays): per batch the mean cycles and busy shares, the per-window cycles, weights, layers and
groups, and the per-window drain counters."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import Environment, counters
from . import results as rs
from .evaluate import Evaluator, stream_busy
from .organisations import ORGS, expand

DRAIN_COUNTERS = ("drain_wait", "drain_waits", "independent_phases")


def frozen(root: Path, out: Path, model: str, fmt: str, P: int, org: str, fields: dict, threads: int = 3) -> dict:
    cell = rs.cell_name(model, fmt, P)
    rec = rs.read_json(rs.final_path(out, cell, org))
    chip = rec["best_main"]
    shapes = expand(ORGS[org], chip["base"])
    env = replace(Environment.from_dict(chip["env"]), shared_array=ORGS[org].shared_array, **fields)
    ev, S, report = Evaluator(root, model, fmt, P, threads=threads), len(shapes), {}
    for B in wl.BATCHES:
        rows = ev.windows(B, "report")
        w, n = wl.window_weights(rows), len(rows)
        cyc, busy, drain = np.zeros(n), np.zeros((n, 2)), np.zeros((n, 3))
        for ix, engine_out in ev.outputs(rows, shapes, env):
            c = counters(engine_out)
            cyc[ix], busy[ix] = c["cycles"], stream_busy(c, S)
            drain[ix] = np.stack([c[k] for k in DRAIN_COUNTERS], 1)
        report[f"B{B}"] = dict(
            cycles=wl.dot(w, cyc) / float(w.sum()),
            busy=[wl.dot(w, busy[:, j] / cyc) / float(w.sum()) for j in range(2)],
            window_cycles=cyc.tolist(),
            window_weights=w.tolist(),
            window_layers=[int(r["layer"]) for r in rows],
            window_groups=[r["group"] for r in rows],
            window_sys=drain.astype(np.int64).tolist(),
            sys_wait_share=wl.dot(w, drain[:, 0] / (S * cyc)) / float(w.sum()),
        )
    return dict(
        cell=cell,
        model=model,
        fmt=fmt,
        P=P,
        org=org,
        streams=S,
        engine_fields=fields,
        selected_from=rec["selected_from"],
        base=chip["base"],
        shapes=chip["shapes"],
        policy=chip["policy"],
        env=chip["env"],
        report=report,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int)
    ap.add_argument("org", choices=sorted(ORGS))
    ap.add_argument("--systolic", type=int, required=True, help="systolic timing mode of the engine")
    ap.add_argument("--drain-form", default="linear")
    ap.add_argument("--drain-segment", type=int, default=16)
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    fields = dict(systolic=a.systolic, drain_form=a.drain_form, drain_segment=a.drain_segment)
    res = frozen(a.data, a.out, a.model, a.fmt, a.P, a.org, fields, a.threads)
    settings = f"systolic{a.systolic}" + ("" if a.drain_form == "linear" else f"-{a.drain_form}{a.drain_segment}")
    rs.write_json(Path(a.out) / "frozen" / settings / res["cell"] / f"{a.org}.json", res, indent=None)
    print(res["cell"], a.org, settings, {k: round(v["cycles"], 1) for k, v in res["report"].items()}, flush=True)


if __name__ == "__main__":
    main()
