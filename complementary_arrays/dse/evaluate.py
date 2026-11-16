"""Cycle-level evaluation of chips on the snapshot windows of one cell (model, weight format, multiplier budget), and the
evaluation of final chips on a window set.

Sets: s1_eval (eval snapshots, B2..B128), s1_sealed (sealed snapshots, B2..B128), s1_big (eval snapshots, B256 and B512;
snapshots there reuse requests), s1_dev (dev snapshots, B2..B128), s1_b<B> (eval snapshots of one batch). Every window
of a batch is simulated in one engine call. Output per set and organisation: <set>_<org>.npz with the per-window cycles,
busy cycles of both task streams, weights, layer, snapshot and group of every batch, and <set>_<org>.json with the
per-batch mean window cycles, MoE time per decode step (mean x MoE layers) and busy shares.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import CapacityError, Environment, counters, simulate
from . import results as rs
from . import shapes as sh
from . import snapshots as ss
from .formats import ROUTES, TIMING, converter_rate
from .organisations import FAMILY, ORGS, expand

INF = float("inf")
SETS = {"s1_eval": ("eval", wl.BATCHES), "s1_sealed": ("sealed", wl.BATCHES), "s1_dev": ("select", wl.BATCHES),
        "s1_big": ("eval", (256, 512))}


def sram_mib(P: int) -> int:
    """16 MiB up to 128K multipliers, then in proportion to the multipliers."""
    return 16 * max(1, P // 131072)


def chip_setup(chip: dict, org: str, overrides: dict | None = None) -> tuple[list, Environment]:
    """Array shapes of the task streams and the engine settings of a chip record (best_main of a result file)."""
    o = ORGS[org]
    shapes = expand(o, chip["base"])
    if "shapes" in chip and [list(map(int, s)) for s in chip["shapes"]] != shapes:
        raise ValueError(f"stream shapes {chip['shapes']} do not match base {chip['base']}")
    env = Environment.from_dict({**TIMING, **chip["env"], **(overrides or {})})
    return shapes, replace(env, shared_array=o.shared_array)


class Evaluator:
    def __init__(self, root: Path, model: str, fmt: str, P: int, hbm_tbps: int = 8, threads: int = 3, route=None):
        self.root, self.model, self.fmt, self.P, self.threads = Path(root), model, fmt, P, threads
        self.route = route or ROUTES[fmt]
        self.hbm = int(hbm_tbps * 1000)
        self.converter = converter_rate(fmt, self.hbm)
        self.rejects, self.evaluations = [], 0

    def windows(self, batch: int, kind: str) -> list[dict]:
        return ss.windows(self.root, self.model, self.route, batch, kind)

    def legal(self, base) -> bool:
        return sh.legal(base, self.P)

    def outputs(self, rows: list[dict], shapes, env):
        """Engine output of the windows, one engine call per live token count."""
        live = np.array([r["live"] for r in rows])
        for b in np.unique(live):
            ix = np.flatnonzero(live == b)
            yield ix, simulate(
                np.stack([rows[i]["ids"] for i in ix]), shapes, wl.MODELS[self.model], env, threads=self.threads
            )

    def run(self, rows: list[dict], shapes, env) -> tuple[np.ndarray, np.ndarray]:
        """Cycles and busy cycles of the two task streams of every window."""
        cyc, busy = np.zeros(len(rows)), np.zeros((len(rows), 2))
        for ix, out in self.outputs(rows, shapes, env):
            c = counters(out)
            cyc[ix], busy[ix] = c["cycles"], c["ctx_enabled"][:, :2]
        return cyc, busy

    def score(self, shapes, env, kind: str, batches) -> float:
        """Geometric mean over batches of the window-weighted mean layer cycles; inf if the chip does not run."""
        self.evaluations += 1
        try:
            t = []
            for B in batches:
                rows = self.windows(B, kind)
                t.append(wl.dot(wl.window_weights(rows), self.run(rows, shapes, env)[0]))
            return wl.geomean(t)
        except (CapacityError, RuntimeError, ValueError) as ex:
            self.rejects.append(dict(shapes=shapes, env=asdict(env), reason=str(ex)[:200]))
            return INF

    def per_window(self, shapes, env, kind: str, batches) -> dict:
        """Per batch: weighted mean layer cycles and busy shares of the two task streams (infeasible: the chip does not
        run the batch)."""
        res = {}
        for B in batches:
            rows = self.windows(B, kind)
            try:
                cyc, busy = self.run(rows, shapes, env)
            except (CapacityError, RuntimeError, ValueError) as ex:
                res[f"B{B}"] = dict(infeasible=True, why=str(ex)[:200])
                continue
            w = wl.window_weights(rows)
            res[f"B{B}"] = dict(
                cycles=wl.dot(w, cyc) / float(w.sum()),
                busy=[wl.dot(w, busy[:, j] / cyc) / float(w.sum()) for j in range(2)],
            )
        return res


def set_batches(name: str) -> tuple[str, tuple]:
    if name.startswith("s1_b"):
        return "eval", (int(name[4:]),)
    return SETS[name]


def evaluate(ev: Evaluator, chip: dict, org: str, set_name: str, overrides: dict | None = None) -> tuple[dict, dict]:
    """Per-window arrays and per-batch summary of one chip on a window set."""
    shapes, env = chip_setup(chip, org, overrides)
    kind, batches = set_batches(set_name)
    arrays, summary = {}, {}
    for B in batches:
        rows = ev.windows(B, kind)
        try:
            cyc, busy = ev.run(rows, shapes, env)
        except (CapacityError, RuntimeError, ValueError) as ex:
            summary[f"B{B}"] = dict(infeasible=True, why=str(ex)[:200])
            continue
        w = wl.window_weights(rows)
        layers = len({r["layer"] for r in rows})
        mean = wl.dot(w, cyc) / float(w.sum())
        summary[f"B{B}"] = dict(
            windows=len(rows),
            layers=layers,
            mean_window_cycles=mean,
            step_cycles=mean * layers,
            busy=[wl.dot(w, busy[:, j] / cyc) / float(w.sum()) for j in range(2)],
            mean_live=wl.dot(w, np.array([r["live"] for r in rows], float)) / float(w.sum()),
        )
        arrays.update(
            {
                f"B{B}_cycles": cyc,
                f"B{B}_busy": busy.astype(np.float32),
                f"B{B}_weights": w,
                f"B{B}_layer": np.array([r["layer"] for r in rows], np.int16),
                f"B{B}_unit": np.array([r.get("snapshot", -1) for r in rows], np.int32),
                f"B{B}_group": np.array([r["group"] for r in rows]),
            }
        )
    return arrays, summary


def parse_overrides(items) -> dict:
    """field=value pairs; values are numbers, true / false, or strings."""
    out = {}
    for item in items or ():
        k, v = item.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.lstrip("-").replace(".", "", 1).isdigit():
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("set", help="s1_eval, s1_sealed, s1_big, s1_dev or s1_b<B>")
    ap.add_argument("--chips", type=Path, required=True, help="directory with <org>.json chip records")
    ap.add_argument("--version", required=True, help="name of the chip set, e.g. final")
    ap.add_argument("--orgs", default=",".join(FAMILY))
    ap.add_argument("--override", nargs="*", help="engine settings field=value applied to every chip")
    ap.add_argument("--tag", default="", help="suffix of the output version (e.g. for overrides)")
    ap.add_argument("--route", default=None, help="routing capture (default: by format)")
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--hbm", type=int, default=8, help="HBM TB/s")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    ev = Evaluator(a.data, a.model, a.fmt, a.P, a.hbm, a.threads, a.route)
    cell, overrides = rs.cell_name(a.model, a.fmt, a.P, a.hbm), parse_overrides(a.override)
    for org in a.orgs.split(","):
        t0, path = time.time(), a.chips / f"{org}.json"
        chip = rs.read_json(path)["best_main"]
        arrays, summary = evaluate(ev, chip, org, a.set, overrides)
        out = rs.eval_dir(a.out, a.version + a.tag, cell)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f".{a.set}_{org}.{os.getpid()}.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(out / f"{a.set}_{org}.npz")
        rs.write_json(
            out / f"{a.set}_{org}.json",
            dict(cell=cell, org=org, version=a.version + a.tag, route=ev.route, set=a.set, chip_file=str(path),
                 base=chip["base"], policy=chip["policy"], env=chip["env"], overrides=overrides, batches=summary,
                 seconds=time.time() - t0),
        )
        print(cell, a.version + a.tag, a.set, org,
              {k: round(v["step_cycles"]) if "step_cycles" in v else "infeasible" for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    main()
