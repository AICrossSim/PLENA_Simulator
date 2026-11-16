"""Summary of the closed-loop HBM cross-check: per chip group and batch, the summed cycles of the sampled windows with
the ideal HBM stream and with Ramulator, their ratio, and the speedup of the pair over two identical arrays and over the
single array on the windows done for all three (geometric mean over the batches and at the largest batch); with --logs
also the achieved HBM bandwidth and transfer latency quantiles of the closed-loop windows.

usage: python -m complementary_arrays.xcheck.summary SAMPLE_DIR RUN_DIR OUT.json --group NAME=SINGLE,HOM,HET [...]
       [--batches 32,64,96,128] [--logs]
  SINGLE, HOM, HET: sample tags (without _B<batch>) of the single array, the two identical arrays and the pair; a
  group may end with @B,B,... for its own batches. Run outputs are RUN_DIR/<tag>_B<batch>_w<window>.npz (run.py).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

ORGS = ("single2", "hom11", "het11")


def geomean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def load(samples, runs, tag):
    ideal = np.load(samples / f"{tag}.npz")["out"][:, 0].astype(float)
    ram = np.full(len(ideal), np.nan)
    for w in range(len(ideal)):
        f = runs / f"{tag}_w{w}.npz"
        if f.exists():
            ram[w] = float(np.load(f)["out"][0, 0])
    return ideal, ram


def window_log(path):
    """Cycles, achieved bandwidth (bytes per cycle over all, busy and backlogged cycles) and latency quantiles."""
    j = json.loads(Path(path).read_text())
    h = np.array(j.pop("hist"), float)
    c = np.cumsum(h) / h.sum()
    span = j["cycles"] - j["init"]
    return dict(
        cycles=j["cycles"],
        bw_all=j["bytes"] / span,
        bw_busy=j["bytes"] / j["busy_cycles"],
        bw_backlog=j["backlog_bytes"] / max(1, j["backlog_cycles"]),
        backlog_share=j["backlog_cycles"] / span,
        lat_mean=j["lat_sum"] / j["lat_n"],
        rejects=j["rejects"],
        **{f"p{int(100 * x)}": int(np.searchsorted(c, x)) for x in (0.5, 0.9, 0.99)},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("samples", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--group", action="append", required=True)
    p.add_argument("--batches", default="32,64,96,128")
    p.add_argument("--logs", action="store_true")
    a = p.parse_args()
    res = []
    for g in a.group:
        name, tags = g.split("=", 1)
        tags, _, own = tags.partition("@")
        tags = dict(zip(ORGS, tags.split(","), strict=True))
        batches = [int(b) for b in (own or a.batches).split(",")]
        rec = dict(group=name, tags=tags, batches={})
        for b in batches:
            data = {o: load(a.samples, a.runs, f"{t}_B{b}") for o, t in tags.items()}
            done = np.logical_and.reduce([np.isfinite(data[o][1]) for o in ORGS])
            row = {
                o: dict(windows=int(done.sum()), ideal=float(data[o][0][done].sum()), ram=float(data[o][1][done].sum()))
                for o in ORGS
            }
            if done.any():
                for o in ORGS:
                    row[o]["ram_over_ideal"] = row[o]["ram"] / row[o]["ideal"]
                for o, key in (("hom11", "pair_over_identical"), ("single2", "pair_over_single")):
                    row[key] = {m: row[o][m] / row["het11"][m] for m in ("ideal", "ram")}
            if a.logs:
                for o, t in tags.items():
                    logs = [window_log(f) for f in sorted(a.runs.glob(f"{t}_B{b}_w*_w0.json"))]
                    if logs:
                        row[o]["closed_loop"] = {k: float(np.mean([x[k] for x in logs])) for k in logs[0]}
            rec["batches"][f"B{b}"] = row
        ok = [rec["batches"][f"B{b}"] for b in batches if "pair_over_identical" in rec["batches"][f"B{b}"]]
        rec["complete"] = all(
            rec["batches"][f"B{b}"]["het11"]["windows"] == len(np.load(a.samples / f"{tags['het11']}_B{b}.npz")["out"])
            for b in batches
        )
        if len(ok) == len(batches):
            for key in ("pair_over_identical", "pair_over_single"):
                rec[key] = {
                    m: dict(geomean=geomean([r[key][m] for r in ok]), last=ok[-1][key][m]) for m in ("ideal", "ram")
                }
            rec["ram_over_ideal_geomean"] = {o: geomean([r[o]["ram_over_ideal"] for r in ok]) for o in ORGS}
        res.append(rec)
        line = f"{name:36s} {'complete' if rec['complete'] else 'partial ':8s} "
        line += " ".join(f"B{b}:{rec['batches'][f'B{b}']['het11']['windows']}" for b in batches)
        if "pair_over_identical" in rec:
            for key, lab in (("pair_over_identical", "vs identical"), ("pair_over_single", "vs single")):
                v = rec[key]
                line += f"  {lab} ideal {v['ideal']['geomean']:.3f}/{v['ideal']['last']:.3f}"
                line += f" ram {v['ram']['geomean']:.3f}/{v['ram']['last']:.3f}"
        print(line)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
