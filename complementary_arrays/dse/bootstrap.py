"""Paired two-way bootstrap of the het speed-up over the hom, single and best other chip of each cell.

One draw resamples the layers (shared across batches) and, per batch, that batch's request groups, with replacement;
a window weighs its weight times the draw counts of its layer and group. Per batch the weighted mean cycles, per
metric (B32-128, B2-128, single batches) the geometric mean over its batches; vs_best compares with the faster of
single and hom in every draw. 2,000 draws, seed 0, 95% percentile intervals; the layer-only interval resamples layers
only. Heldout = report windows of the chosen chips; sealed = the sealed request group (final chips, two streams).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from . import results as rs
from .organisations import FAMILIES

DRAWS = 2000
VERSIONS = {"top10": "search/top10", "final": "final"}


def draws(per_batch: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    layers = sorted({int(x) for d in per_batch.values() for x in d["layers"]})
    li = {x: i for i, x in enumerate(layers)}
    CL = np.stack(
        [np.bincount(rng.integers(0, len(layers), len(layers)), minlength=len(layers)) for _ in range(DRAWS)]
    ).astype(float)
    M = {}
    for B in sorted(per_batch):
        d = per_batch[B]
        groups = sorted(set(d["groups"]))
        gi = {g: i for i, g in enumerate(groups)}
        CG = np.stack(
            [np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)) for _ in range(DRAWS)]
        ).astype(float)
        lay, grp = np.array([li[int(x)] for x in d["layers"]]), np.array([gi[g] for g in d["groups"]])
        w = np.asarray(d["weights"], float)
        M[B] = (CL[:, lay] * CG[:, grp] * w, CL[:, lay] * w, w)
    return M


def stats(per_batch: dict, fam: tuple) -> dict:
    """per_batch[B] = dict(cycles={org: per-window cycles}, weights, layers, groups)."""
    M, (single, hom, het), out = draws(per_batch), fam, {}
    lt = {}
    for B in per_batch:
        lt[B] = {}
        for o in fam:
            c = np.asarray(per_batch[B]["cycles"][o], float)
            lt[B][o] = (
                np.log((M[B][0] @ c) / M[B][0].sum(1)),
                np.log((M[B][1] @ c) / M[B][1].sum(1)),
                np.log(wl.dot(M[B][2], c) / float(M[B][2].sum())),
            )
    metrics = {
        "B32-128": [B for B in wl.MAIN_BATCHES if B in per_batch],
        "B2-128": [B for B in wl.BATCHES if B in per_batch],
    }
    metrics.update({f"B{B}": [B] for B in sorted(per_batch)})
    for name, bs in metrics.items():
        if name == "B2-128" and len(bs) < len(wl.BATCHES):
            continue
        g = {o: tuple(sum(lt[B][o][i] for B in bs) / len(bs) for i in range(3)) for o in fam}
        r = {}
        for key, o in (("vs_hom", hom), ("vs_single", single), ("vs_best", None)):
            if o is None:
                two, layer, point = (
                    np.minimum(g[single][0], g[hom][0]) - g[het][0],
                    np.minimum(g[single][1], g[hom][1]) - g[het][1],
                    min(g[single][2], g[hom][2]) - g[het][2],
                )
            else:
                two, layer, point = g[o][0] - g[het][0], g[o][1] - g[het][1], g[o][2] - g[het][2]
            r[key] = dict(
                point=float(np.exp(point)),
                ci_two_way=[float(x) for x in np.percentile(np.exp(two), [2.5, 97.5])],
                ci_layer=[float(x) for x in np.percentile(np.exp(layer), [2.5, 97.5])],
            )
        r["times"] = {o: float(np.exp(g[o][2])) for o in fam}
        out[name] = r
    return out


def heldout(root: Path, out: Path, model: str, fmt: str, P: int, streams: int, version: str) -> dict | None:
    fam, cell = FAMILIES[streams], rs.cell_name(model, fmt, P)
    paths = {o: Path(out) / VERSIONS[version] / cell / f"{o}.json" for o in fam}
    if not all(p.exists() for p in paths.values()):
        return None
    chips = {o: rs.read_json(p)["best_main"] for o, p in paths.items()}
    per_batch = {}
    for B in wl.BATCHES:
        rows = wl.windows(root, model, B, "report")
        rep = {o: chips[o]["report"][f"B{B}"] for o in fam}
        if any(len(rep[o]["window_cycles"]) != len(rows) for o in fam) or rep[fam[2]]["window_layers"] != [
            int(r["layer"]) for r in rows
        ]:
            raise ValueError(f"report windows of {cell} do not match the traces")
        per_batch[B] = dict(
            cycles={o: rep[o]["window_cycles"] for o in fam},
            weights=rep[fam[2]]["window_weights"],
            layers=rep[fam[2]]["window_layers"],
            groups=[r["group"] for r in rows],
        )
    return stats(per_batch, fam)


def sealed(out: Path, model: str, fmt: str, P: int, streams: int, version: str) -> dict | None:
    path = rs.sealed_path(out, rs.cell_name(model, fmt, P))
    if streams != 2 or version != "final" or not path.exists():
        return None
    return stats({int(B): v for B, v in rs.read_json(path)["batches"].items()}, FAMILIES[2])


def bootstrap(root: Path, out: Path) -> list[dict]:
    rows = []
    for S in FAMILIES:
        for fmt in rs.FORMAT_ORDER:
            for model in rs.MODEL_ORDER:
                for P in rs.BUDGETS:
                    for version in VERSIONS:
                        h = heldout(root, out, model, fmt, P, S, version)
                        if h is not None:
                            rows.append(
                                dict(
                                    model=model,
                                    fmt=fmt,
                                    P=P,
                                    streams=S,
                                    version=version,
                                    heldout=h,
                                    sealed=sealed(out, model, fmt, P, S, version),
                                )
                            )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    rows = bootstrap(a.data, a.out)
    rs.write_json(Path(a.out) / "bootstrap.json", rows)
    for x in rows:
        h, s = x["heldout"]["B32-128"]["vs_hom"], (x["sealed"] or {}).get("B32-128", {}).get("vs_hom")
        print(
            f"S{x['streams']} {x['model']:9s} {x['fmt']:6s} {x['P'] // 1024:3d}K {x['version']:6s} heldout vs_hom {h['point']:.3f} "
            f"[{h['ci_two_way'][0]:.3f}, {h['ci_two_way'][1]:.3f}]"
            + (f" | sealed {s['point']:.3f} [{s['ci_two_way'][0]:.3f}, {s['ci_two_way'][1]:.3f}]" if s else "")
        )


if __name__ == "__main__":
    main()
