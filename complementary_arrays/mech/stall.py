"""Where the time of a final chip goes: per task stream, the partition of the layer cycles into compute, waits for
first-pass weights from HBM, waits for the shared weight port (first pass or re-delivery), other waits and idle time,
on the heldout report windows (window-weighted means), with HBM and port utilisation."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..dse import results as rs
from ..dse.evaluate import Evaluator
from ..dse.organisations import FAMILIES, ORGS, expand
from ..sim import workload as wl
from ..sim.engine import Environment, core_map, counters

BATCHES = (32, 64, 128)
FIELDS = (
    "enabled",
    "input_stall",
    "output_stall",
    "merge_stall",
    "commit_wait",
    "idle",
    "share_stall",
    "hbm_service",
    "delivery_wait",
    "delivered_weights",
    "wait_hbm",
    "wait_wport",
    "wait_x",
    "wait_latency",
    "wait_decode",
)


def final_chip(out: Path, cell: str, org: str) -> tuple[dict, dict]:
    rec = rs.read_json(rs.final_path(out, cell, org))
    meta = dict(
        selected_from=rec["selected_from"],
        source=rec["source"],
        provisional=rec["provisional"],
        candidates=rec["candidates_select_main"],
    )
    return meta, rec["best_main"]


def chip_environment(org: str, chip: dict) -> Environment:
    return replace(Environment.from_dict(chip["env"]), shared_array=ORGS[org].shared_array)


def labels(org: str, shapes, cmap) -> list[str]:
    if org.startswith("het"):
        tallest = max(int(s[0]) for s in shapes)
        return ["tall" if int(s[0]) == tallest else "short" for s in shapes]
    if org.startswith("single"):
        return [f"A0/s{c}" for c in range(len(shapes))]
    return [f"A{cmap[c]}" for c in range(len(shapes))]


def stall(root: Path, out: Path, model: str, fmt: str, P: int, org: str, threads: int = 3) -> dict:
    cell = rs.cell_name(model, fmt, P)
    meta, chip = final_chip(out, cell, org)
    shapes, env = expand(ORGS[org], chip["base"]), chip_environment(org, chip)
    S, cmap = len(shapes), core_map(env, len(shapes))
    lab, ev = labels(org, shapes, cmap), Evaluator(root, model, fmt, P, threads=threads)
    res = dict(
        cell=cell,
        model=model,
        fmt=fmt,
        P=P,
        chip=org,
        **meta,
        base=chip["base"],
        stream_shapes=[list(map(int, s)) for s in shapes],
        policy=chip["policy"],
        windows="heldout report windows, weighted as in the result table",
        unit="cycles per layer (window-weighted mean); each stream row sums to layer_cycles",
    )
    for B in BATCHES:
        rows = wl.windows(root, model, B, "report")
        w, n = wl.window_weights(rows), len(rows)
        cyc, init = np.zeros(n), np.zeros(n)
        ctx = {f: np.zeros((n, S)) for f in FIELDS}
        first, reread, used, saturated, hbm = (np.zeros(n) for _ in range(5))
        for ix, engine_out in ev.outputs(rows, shapes, env):
            c = counters(engine_out)
            cyc[ix], init[ix] = c["cycles"], c["startup"]
            for f in FIELDS:
                ctx[f][ix] = c["ctx_" + f][:, :S]
            first[ix] = np.asarray(c["wport_bytes"]).reshape(len(ix), -1).sum(1)
            reread[ix] = np.asarray(c["reread_bytes"]).reshape(len(ix), -1).sum(1)
            used[ix], saturated[ix] = c["port_bytes"], c["port_saturated_cycles"]
            hbm[ix] = c["ctx_hbm_service"][:, :S].sum(1)
        W = float(w.sum())

        def wm(x):
            return wl.dot(w, x) / W

        streams = []
        for s in range(S):
            g = {f: ctx[f][:, s] for f in FIELDS}
            other = (
                g["wait_decode"]
                + g["wait_latency"]
                + g["wait_x"]
                + g["output_stall"]
                + g["merge_stall"]
                + g["commit_wait"]
                + init
            )
            cats = dict(
                compute=wm(g["enabled"]),
                wait_hbm_first=wm(g["wait_hbm"]),
                wait_port_first=wm(g["wait_wport"]),
                wait_port_reread=wm(g["delivery_wait"]),
                wait_other=wm(other),
                idle=wm(g["idle"]),
            )
            part = (
                g["enabled"]
                + g["input_stall"]
                + g["output_stall"]
                + g["merge_stall"]
                + g["commit_wait"]
                + g["idle"]
                + init
            )
            if not np.allclose(part, cyc):
                raise ValueError("stream partition does not sum to the layer cycles")
            streams.append(
                dict(
                    array=int(cmap[s]),
                    stream=s,
                    label=lab[s],
                    shape=list(map(int, shapes[s])),
                    cycles=cats,
                    total=wl.sequential_sum(cats.values()),
                    detail=dict(
                        wait_other=dict(
                            converter_decode=wm(g["wait_decode"]),
                            pipeline_latency=wm(g["wait_latency"]),
                            activation_x=wm(g["wait_x"]),
                            output_stall=wm(g["output_stall"]),
                            merge_stall_token_lock_or_shared_array=wm(g["merge_stall"]),
                            of_which_shared_array=wm(g["share_stall"]),
                            commit_wait=wm(g["commit_wait"]),
                            startup_activation_load=wm(init),
                        ),
                        hbm_bytes_to_stream=wm(g["hbm_service"]),
                        redelivered_weights=wm(g["delivered_weights"]),
                    ),
                )
            )
        wc = wl.dot(w, cyc)
        main = chip["report"][f"B{B}"]["cycles"]
        if abs(wc / W - main) > 1e-6 * main:
            raise ValueError(f"layer cycles {wc / W} differ from the result table {main}")
        res[f"B{B}"] = dict(
            layer_cycles=wc / W,
            main_table_cycles=main,
            arrays=streams,
            hbm_busy_frac=wl.dot(w, hbm) / (wc * env.H),
            port_bytes=dict(first=wm(first), reread=wm(reread)),
            port_bytes_per_cycle=dict(first=wl.dot(w, first) / wc, reread=wl.dot(w, reread) / wc, W=env.W),
            port_used_check=dict(used=wm(used), first_plus_reread=wm(first + reread)),
            port_saturated_frac=wl.dot(w, saturated) / wc,
            n_windows=n,
        )
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int)
    ap.add_argument("org", choices=FAMILIES[2])
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    res = stall(a.data, a.out, a.model, a.fmt, a.P, a.org, a.threads)
    rs.write_json(Path(a.out) / "mech" / "stall" / f"{res['cell']}_{a.org}.json", res)
    print(res["cell"], a.org, {B: round(res[f"B{B}"]["layer_cycles"], 1) for B in BATCHES}, flush=True)


if __name__ == "__main__":
    main()
