"""Decode-step snapshots of a continuously batched server, built from per-request routing captures, and the windows that
the search and the evaluation simulate.

A pool is a list of requests of one model and routing capture; request i has a source, a split (dev, eval or sealed), L_i
decode steps and routes [L_i, MoE layers, topk]. A snapshot is one decode step of a server with B slots: B distinct
requests, each at a uniformly drawn position. Every slot holds request i with probability w_i = (1 / sources) /
(requests of its source) x L_i / sum, drawn by systematic sampling proportional to size (requests with B w_i > 1 are
capped at one slot). Every batch size has 64 snapshots; a window is one MoE layer of one snapshot.

Data root layout: <model>_<route>/s1_<dev|eval|sealed>_b<B>.npz with routes [snapshots x layers, B, topk], live, metadata
(one record per window: layer, snapshot, source = group = 's1'), the slot draws and the request keys of the pool.
Windows: screen = dev snapshots 0..15 at the MoE layers L/4, L/2, 3L/4; select = every dev window; eval, sealed = every
window of those splits. Every window weighs the same.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..sim import workload as wl

SEED = 20261113
SNAPSHOTS = 64
SCREEN_SNAPSHOTS = 16
SPLIT_BATCHES = {
    "dev": (2, 4, 8, 16, 32, 64, 96, 128),
    "sealed": (2, 4, 8, 16, 32, 64, 96, 128),
    "eval": (2, 4, 8, 16, 32, 64, 96, 128, 256, 512),
}
CAPTURE_KIND = {"qwen": "qwen35", "deepseek": "deepseek"}


def seed_of(*parts) -> list[int]:
    """A stable seed from labels (model, routing, scenario, split, batch, ...)."""
    return [SEED, zlib.crc32("|".join(str(p) for p in parts).encode())]


def s1_weights(sources, lengths) -> np.ndarray:
    """Draw weight per request: equal share per source, split evenly over its requests, times the request length."""
    sources, lengths = np.asarray(sources), np.asarray(lengths, float)
    names = sorted(set(sources.tolist()))
    n = {s: int((sources == s).sum()) for s in names}
    w = np.array([1.0 / len(names) / n[s] for s in sources]) * lengths
    return w / w.sum()


def inclusion(weights, B: int) -> tuple[np.ndarray, int]:
    """Inclusion probabilities pi_i = B w_i of a B-request snapshot, requests with B w_i > 1 capped at 1 and the rest
    rescaled until none exceeds 1. Returns pi (sum B) and the number of capped requests."""
    w = np.asarray(weights, float) / np.sum(weights)
    pi, capped = B * w, np.zeros(len(w), bool)
    if B > int((w > 0).sum()):
        raise ValueError(f"batch {B} needs more distinct requests than the pool has ({int((w > 0).sum())})")
    while np.any(pi[~capped] > 1 + 1e-12):
        capped |= pi > 1 - 1e-12
        free = w * ~capped
        pi = np.where(capped, 1.0, (B - capped.sum()) * free / free.sum())
    return pi, int(capped.sum())


def draw_snapshots(weights, lengths, B: int, n_snap: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n_snap snapshots of B distinct requests by systematic sampling proportional to size (random pool order, one random
    start, B equally spaced picks on the cumulated inclusion probabilities), slot order shuffled, then a uniform position
    in each request. Returns req [n_snap, B] and pos [n_snap, B]."""
    weights, lengths, n = np.asarray(weights, float), np.asarray(lengths, np.int64), len(weights)
    req, pos = np.zeros((n_snap, B), np.int64), np.zeros((n_snap, B), np.int64)
    pi, _ = inclusion(weights, B)
    for s in range(n_snap):
        order = rng.permutation(n)
        c = np.cumsum(pi[order])
        c *= B / c[-1]
        picks = np.minimum(np.searchsorted(c, rng.random() + np.arange(B), side="right"), n - 1)
        r = order[picks]
        assert len(set(r.tolist())) == B
        req[s] = r[rng.permutation(B)]
        pos[s] = (rng.random(B) * lengths[req[s]]).astype(np.int64)
    assert np.all(pos < lengths[req]) and np.all(pos >= 0)
    return req, pos


class Pool:
    """Requests of one model and routing capture; routes_of(i) is the [L_i, layers, topk] int16 array of request i."""

    def __init__(self, keys, sources, splits, lengths, layers, topk, experts, routes, files=()):
        self.keys, self.sources, self.splits = list(keys), np.asarray(sources), np.asarray(splits)
        self.lengths, self.layers = np.asarray(lengths, np.int64), list(layers)
        self.topk, self.experts, self._routes, self.files = int(topk), int(experts), routes, list(files)

    def routes_of(self, i: int) -> np.ndarray:
        r = self._routes[i].astype(np.int16)
        assert r.shape == (self.lengths[i], len(self.layers), self.topk), (i, r.shape)
        return r

    def subset(self, split=None, source=None) -> np.ndarray:
        m = np.ones(len(self.keys), bool)
        if split is not None:
            m &= np.isin(self.splits, [split] if isinstance(split, str) else list(split))
        if source is not None:
            m &= self.sources == source
        return np.flatnonzero(m)

    def drawable(self, split: str) -> np.ndarray:
        """Requests of a split with at least one decode step."""
        idx = self.subset(split=split)
        return idx[self.lengths[idx] > 0]


def load_pool(captures: Path, model: str, route: str, mode: str = "sample") -> Pool:
    """Pool from the capture chunks <captures>/<kind>_<route>_<mode>_chunk*.npz (request, source, split, gen_tokens,
    offsets, routes [sum of generated tokens, MoE layers, topk], moe_layers). Row 0 of a request is its last prompt token;
    rows 1..G-1 are its decode steps, so a request holds a slot for L = G - 1 steps."""
    files = sorted(glob.glob(str(Path(captures) / f"{CAPTURE_KIND[model]}_{route}_{mode}_chunk*.npz")))
    if not files:
        raise FileNotFoundError(f"no capture chunks for {model} {route} {mode} in {captures}")
    keys, sources, splits, gen, routes, layers = [], [], [], [], [], None
    for f in files:
        with np.load(f) as z:
            off, r = z["offsets"], z["routes"]
            for i in range(len(z["request"])):
                keys.append(str(z["request"][i]))
                sources.append(str(z["source"][i]))
                splits.append(str(z["split"][i]))
                gen.append(int(z["gen_tokens"][i]))
                routes.append(np.asarray(r[off[i] + 1 : off[i + 1]]))
            ml = [int(x) for x in z["moe_layers"]]
            assert layers is None or layers == ml
            layers = ml
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate request in the capture")
    w = wl.MODELS[model]
    return Pool(keys, sources, splits, np.array(gen) - 1, layers, w["topk"], w["experts"], routes, files)


def materialize(pool: Pool, req: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Windows of the snapshots: ids [n_snap, layers, B, topk] int16 in slot order."""
    n_snap, B = req.shape
    out = np.zeros((n_snap, len(pool.layers), B, pool.topk), np.int16)
    flat_req, flat_pos = req.ravel(), pos.ravel()
    for f in np.argsort(req, axis=None, kind="stable"):
        s, j = divmod(int(f), B)
        out[s, :, j, :] = pool.routes_of(int(flat_req[f]))[int(flat_pos[f])]
    return out


def save_snapshots(path: Path, model, pool, req, pos, ids, split, B, seed, extra=None) -> None:
    n_snap, L = ids.shape[:2]
    meta = [
        dict(model=model, batch=B, live=B, source="s1", group="s1", cohort=0, layer=int(pool.layers[l]), step=1000 + s,
             snapshot=s)
        for s in range(n_snap)
        for l in range(L)
    ]
    info = dict(model=model, scenario="s1", split=split, batch=B, snapshots=n_snap, layers=list(map(int, pool.layers)),
                seed=list(map(int, seed)), pool_keys=[pool.keys[i] for i in sorted(set(req.ravel().tolist()))], **(extra or {}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp, routes=ids.reshape(n_snap * L, B, pool.topk), live=np.full(len(meta), B, np.int16), metadata=json.dumps(meta),
        slot_req=req.astype(np.int32), slot_pos=pos.astype(np.int32), slot_source=np.asarray(pool.sources)[req],
        slot_len=pool.lengths[req].astype(np.int32), info=json.dumps(info),
    )
    tmp.replace(path)


def load_snapshots(path: Path) -> tuple[list[dict], dict, dict]:
    """Windows (metadata records with ids and live), slot draws and file info of a snapshot file."""
    with np.load(path) as z:
        meta, routes, live = json.loads(str(z["metadata"])), z["routes"], z["live"]
        slots = {k: z[k] for k in ("slot_req", "slot_pos", "slot_source", "slot_len") if k in z.files}
        info = json.loads(str(z["info"])) if "info" in z.files else {}
    return [dict(r, ids=routes[i, : live[i]], live=int(live[i])) for i, r in enumerate(meta)], slots, info


def digest(path: Path) -> str:
    """Content digest of a snapshot file (routes, slot requests and positions)."""
    h = hashlib.sha256()
    with np.load(path) as z:
        for k in ("routes", "slot_req", "slot_pos"):
            h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()


def snapshot_path(root: Path, model: str, route: str, split: str, B: int) -> Path:
    return Path(root) / f"{model}_{route}" / f"s1_{split}_b{B}.npz"


def build_file(pool: Pool, root: Path, model: str, route: str, split: str, B: int, mode: str = "sample") -> dict:
    idx = pool.drawable(split)
    src, L = pool.sources[idx], pool.lengths[idx]
    w = s1_weights(src, L)
    seed = seed_of(model, route, mode, "s1", split, B)
    _, capped = inclusion(w, B)
    req, pos = draw_snapshots(w, L, B, SNAPSHOTS, np.random.default_rng(seed))
    path = snapshot_path(root, model, route, split, B)
    save_snapshots(path, model, pool, idx[req], pos, materialize(pool, idx[req], pos), split, B, seed,
                   dict(capped_requests=capped, pool_requests=int(len(idx))))
    return dict(batch=B, split=split, pool_requests=int(len(idx)), capped_requests=capped,
                distinct_requests=int(len(set(idx[req].ravel().tolist()))), digest=digest(path))


def build(captures: Path, root: Path, model: str, route: str, mode: str = "sample", only=None) -> dict:
    """S1 files of one model and routing capture (or only the named ones, e.g. ['s1_eval_b32'])."""
    pool = load_pool(captures, model, route, mode)
    summary = dict(model=model, route=route, mode=mode, requests=len(pool.keys), files={})
    for split, batches in SPLIT_BATCHES.items():
        for B in batches:
            name = f"s1_{split}_b{B}"
            if only is None or name in only:
                summary["files"][name] = build_file(pool, root, model, route, split, B, mode)
                print(name, summary["files"][name], flush=True)
    if only is None:
        (Path(root) / f"{model}_{route}" / "build_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


@lru_cache(maxsize=48)
def _rows(root: str, model: str, route: str, split: str, B: int) -> list[dict]:
    return load_snapshots(snapshot_path(Path(root), model, route, split, B))[0]


def windows(root: Path, model: str, route: str, B: int, kind: str) -> list[dict]:
    """Windows of one batch size: screen, select (dev), eval or sealed."""
    if kind == "screen":
        rows = _rows(str(root), model, route, "dev", B)
        layers = sorted({r["layer"] for r in rows})
        keep = {layers[round(len(layers) * q / 4)] for q in (1, 2, 3)}
        return [r for r in rows if r["snapshot"] < SCREEN_SNAPSHOTS and r["layer"] in keep]
    split = {"select": "dev", "eval": "eval", "sealed": "sealed"}[kind]
    return _rows(str(root), model, route, split, B)


def histograms(root: Path, model: str, route: str, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Distinct expert-load histograms of the dev windows at batch B and their normalised weights (row layout as
    workload.histograms: column t = experts receiving t tokens, column 129 = live tokens)."""
    rows = windows(root, model, route, B, "select")
    experts, topk = wl.MODELS[model]["experts"], wl.MODELS[model]["topk"]
    h = np.zeros((len(rows), 130), np.int32)
    for j, r in enumerate(rows):
        tokens = np.bincount(r["ids"].ravel(), minlength=experts)
        assert tokens.sum() == r["live"] * topk
        h[j, :129] = np.bincount(tokens, minlength=129)
        h[j, 129] = r["live"]
    hh, inverse = np.unique(h, axis=0, return_inverse=True)
    wh = np.bincount(inverse.ravel(), weights=wl.window_weights(rows))
    return np.ascontiguousarray(hh, np.int32), wh / wh.sum()


def check(root: Path, listing: Path) -> list[str]:
    """Snapshot files whose content digest differs from the listing ('<digest>  <model>_<route>/<name>' lines)."""
    bad = []
    for line in Path(listing).read_text().splitlines():
        want, name = line.split()
        path = Path(root) / f"{name}.npz"
        if not path.exists() or digest(path) != want:
            bad.append(name)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="snapshot files from the routing captures")
    b.add_argument("model", choices=sorted(CAPTURE_KIND))
    b.add_argument("route", choices=("int4", "ref"))
    b.add_argument("--captures", type=Path, required=True, help="directory of the capture chunks")
    b.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    b.add_argument("--mode", default="sample")
    b.add_argument("--only", nargs="*", help="file names to build, e.g. s1_eval_b32")
    c = sub.add_parser("check", help="compare snapshot contents with a digest listing")
    c.add_argument("listing", type=Path)
    c.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    a = ap.parse_args(argv)
    if a.data is None:
        ap.error("--data (or COMPLEMENTARY_ARRAYS_DATA) is required")
    if a.cmd == "build":
        build(a.captures, a.data, a.model, a.route, a.mode, set(a.only) if a.only else None)
    else:
        bad = check(a.data, a.listing)
        print("snapshot files differ:" if bad else "snapshot files match the listing", *bad)
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
