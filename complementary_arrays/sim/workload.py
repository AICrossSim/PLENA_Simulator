"""MoE decode workload: layer dimensions, weight formats, routing traces, evaluation windows and per-task cycle tables.

Data root layout: ``traces/<model>_b<batch>_<dev|heldout>.npz`` (routing of every captured decode step and layer) and
``sealed/<qwen35_bf16|deepseek_bf16>/`` (the sealed request group: manifest and shards).
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = {
    "qwen": dict(hidden=2048, routed_width=512, shared_width=512, experts=256, topk=8, shared_gate=True),
    "deepseek": dict(hidden=2048, routed_width=1408, shared_width=2816, experts=64, topk=6, shared_gate=False),
    "qwen3next": dict(hidden=2048, routed_width=512, shared_width=512, experts=512, topk=10, shared_gate=True),
}
MODEL_NAMES = {"qwen": "Qwen3.5-35B-A3B", "deepseek": "DeepSeek-V2-Lite", "qwen3next": "Qwen3-Next-80B-A3B"}
# HBM bytes per weight include the scale beat of each 16-row record (2048 INT8 or 512 INT4 weights).
FORMATS = {
    "bf16": dict(bits=16, group=32, bytes_per_weight=2.0),
    "w8g128": dict(bits=8, group=128, bytes_per_weight=2112 / 2048),
    "w4g32": dict(bits=4, group=32, bytes_per_weight=320 / 512),
}
BATCHES = (2, 4, 8, 16, 32, 64, 96, 128)
MAIN_BATCHES = (32, 64, 96, 128)
FIRST_STEP, LAST_STEP = 16, 32
SEALED_DIRS = {"qwen": "qwen35_bf16", "deepseek": "deepseek_bf16"}
TASK_COLUMNS = 384  # routed expert at 1..128 tokens, shared expert and router at 1..128 live tokens
VECTOR_LANES = 2048


def dot(a, b) -> float:
    """Inner product; every weighted average over windows goes through this function."""
    return float(np.dot(a, b))


def geomean(xs) -> float:
    return float(np.exp(np.mean(np.log(xs))))


def sequential_sum(xs) -> float:
    """Left-to-right sum of floats (sum() of floats is compensated from Python 3.12 on)."""
    total = 0
    for x in xs:
        total += x
    return total


@lru_cache(maxsize=64)
def load_trace(root: Path, model: str, batch: int, split: str) -> list[dict]:
    with np.load(Path(root) / "traces" / f"{model}_b{batch}_{split}.npz") as z:
        meta = json.loads(str(z["metadata"]))
        routes, live = z["routes"], z["live"]
    return [dict(r, ids=routes[i, : live[i]], live=int(live[i])) for i, r in enumerate(meta)]


def _spread(layers: list[int], count: int) -> set[int]:
    return {layers[round(i * (len(layers) - 1) / (count - 1))] for i in range(count)}


def windows(root: Path, model: str, batch: int, kind: str) -> list[dict]:
    """Evaluation windows (one layer of one decode step of one request group).

    screen and select use even layers of the dev requests, report uses odd layers of the heldout requests.
    """
    split, parity = ("heldout", 1) if kind == "report" else ("dev", 0)
    rows = load_trace(root, model, batch, split)
    rank = {layer: i for i, layer in enumerate(sorted({r["layer"] for r in rows}))}
    rows = [r for r in rows if rank[r["layer"]] % 2 == parity and r["step"] >= FIRST_STEP]
    layers = sorted({r["layer"] for r in rows})
    n = len(layers)
    if kind == "screen":
        keep, steps = {layers[n // 3], layers[2 * n // 3]}, {16, 24, 32}
    elif kind == "select":
        keep, steps = {layers[i * n // 5] for i in range(1, 5)}, {16, 20, 24, 28, 32}
    elif kind == "report":
        keep, steps = _spread(layers, 10), set(range(FIRST_STEP, LAST_STEP + 1))
    else:
        raise ValueError(kind)
    return [r for r in rows if r["layer"] in keep and r["step"] in steps]


def sealed_windows(root: Path, model: str, batch: int) -> list[dict]:
    """Report-rule windows of the sealed request group."""
    base = Path(root) / "sealed" / SEALED_DIRS[model]
    manifest = json.loads((base / "manifest.json").read_text())
    plans = {g["id"]: g for g in manifest["plan"]}
    topk = manifest["workload"]["topk"]
    keep = sorted(_spread(sorted(int(x) for x in manifest["layers"]), 10))
    rows = []
    for shard in manifest["shards"]:
        group = plans[shard["id"]]
        if int(group["batch"]) != batch:
            continue
        path = base / shard["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        with np.load(path) as z:
            for step, live in enumerate(z["live_counts"], start=1):
                if not FIRST_STEP <= step <= LAST_STEP:
                    continue
                for layer in keep:
                    ids = z[f"ids_l{layer}_s{step}"].astype(np.int64)
                    if ids.shape != (int(live), topk):
                        raise ValueError(f"bad routing shape in {path}")
                    rows.append(
                        dict(source=group["source"], group=group["id"], layer=layer, step=step, live=int(live), ids=ids)
                    )
    return rows


def window_weights(rows: list[dict]) -> np.ndarray:
    """Sources weigh equally, request groups equally within a source, windows equally within a group."""
    w = np.zeros(len(rows))
    src = np.array([r["source"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    sources = sorted(set(src))
    for s in sources:
        groups = sorted(set(grp[src == s]))
        for g in groups:
            ix = (src == s) & (grp == g)
            w[ix] = 1 / (len(sources) * len(groups) * ix.sum())
    return w


def histograms(root: Path, model: str, batch: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Distinct per-window expert load histograms and their normalised window weights.

    Row layout: column t (1..128) = number of experts receiving t tokens, column 129 = live tokens.
    """
    rows = [r for r in load_trace(root, model, batch, split) if r["step"] >= FIRST_STEP]
    experts, topk = MODELS[model]["experts"], MODELS[model]["topk"]
    h = np.zeros((len(rows), 130), np.int32)
    for j, r in enumerate(rows):
        tokens = np.bincount(r["ids"].ravel(), minlength=experts)
        assert tokens.sum() == r["live"] * topk
        h[j, :129] = np.bincount(tokens, minlength=129)
        h[j, 129] = r["live"]
    hh, inverse = np.unique(h, axis=0, return_inverse=True)
    wh = np.bincount(inverse.ravel(), weights=window_weights(rows))
    return np.ascontiguousarray(hh, np.int32), wh / wh.sum()


def _ceil(a, b):
    return -(-a // b)


def _matrix_cycles(m, n, k, depth, M, N: int, K: int):
    """M x N x K product on an (m, n, k) array: ceil(M/m) * ceil(N/n) output tiles, each streaming ceil(K/k)
    blocks of m rows, plus pipeline fill and the reduction-tree depth."""
    tiles = _ceil(M, m) * _ceil(N, n)
    return tiles * _ceil(K, k) * m + tiles - 1 + 3 * m + depth


def matrix_table(model: str, shapes: np.ndarray) -> np.ndarray:
    """Matrix cycles of every task on every shape: [shape, task column] with columns 0..127 a routed expert at
    1..128 tokens, 128..255 the shared expert and 256..383 the router at 1..128 live tokens."""
    w = MODELS[model]
    H, F, S, E, gate = w["hidden"], w["routed_width"], w["shared_width"], w["experts"], int(w["shared_gate"])
    sh = np.asarray(shapes, np.int64)
    m, n, k = sh[:, 0:1], sh[:, 1:2], sh[:, 2:3]
    depth = np.array([[(int(kk) // int(mm) - 1).bit_length()] for mm, kk in zip(sh[:, 0], sh[:, 2])], np.int64)
    M = np.arange(1, 129, dtype=np.int64)[None, :]
    routed = _matrix_cycles(m, n, k, depth, M, 2 * F, H) + _matrix_cycles(m, n, k, depth, M, H, F)
    shared = _matrix_cycles(m, n, k, depth, M, 2 * S + gate, H) + _matrix_cycles(m, n, k, depth, M, H, S)
    router = _matrix_cycles(m, n, k, depth, M, E, H)
    return np.ascontiguousarray(np.concatenate([routed, shared, router], axis=1))


@lru_cache(maxsize=8)
def vector_table(model: str) -> np.ndarray:
    """Vector-unit cycles of every task column for 1..2048 vector lanes (row = lanes; row 0 unused)."""
    w = MODELS[model]
    H, F, S, E, gate, topk = (
        w["hidden"],
        w["routed_width"],
        w["shared_width"],
        w["experts"],
        int(w["shared_gate"]),
        w["topk"],
    )
    v = np.arange(1, VECTOR_LANES + 1, dtype=np.int64)[:, None]
    M = np.arange(1, 129, dtype=np.int64)[None, :]
    routed = 12 + _ceil(2 * M * F, v) + _ceil(M * H, v) + 4 + M * (_ceil(H, v) + 4)
    shared = 12 + _ceil(2 * M * S, v) + M * (_ceil(H, v) + 4)
    if gate:
        shared = shared + _ceil(M * H, v) + 4 + _ceil(M, v) + 4
    router = 12 + _ceil(M * E * (4 + topk), v)
    table = np.zeros((VECTOR_LANES + 1, TASK_COLUMNS), np.int64)
    table[1:] = np.concatenate([routed, shared, router], axis=1)
    return table
