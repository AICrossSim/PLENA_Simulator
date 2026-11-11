"""Array shape vocabulary and the analytic step-time model that pre-ranks candidate chips.

An array (m, n, k) has height m (tokens per pass), n output and k reduction lanes and n*k multipliers. The analytic
model charges every task (routed expert at t tokens, shared expert and router at the live token count) its matrix and
vector cycles on a core, and bounds a decode step from below by the HBM time of the step's weights and by the
on-chip re-delivery of weights for passes 2..ceil(t/m) at kappa times the HBM rate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..sim import workload as wl

HEIGHTS = (2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32, 48, 64)
MAX_N = 1024
MAX_MACS = 1048576
MAX_OUTPUT_LANES, MAX_INPUT_LANES, MAX_STATE_BYTES = 1024, 8192, 64 * 2**20


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def gemm_dims(model: str) -> tuple[list[int], list[int]]:
    """Output and reduction dimensions of the layer's matrix products (routed and shared experts, router)."""
    w = wl.MODELS[model]
    cols = [2 * w["routed_width"], w["hidden"], 2 * w["shared_width"] + int(w["shared_gate"]), w["experts"]]
    reds = [w["hidden"], w["routed_width"], w["shared_width"]]
    return cols, reds


def state_bytes(m: int, n: int, k: int) -> int:
    """Register and bank storage of an (m, n, k) array: accumulators and snapshots, reduction registers, input
    panels, skew registers and result banks."""
    m, n, k = int(m), int(n), int(k)
    if m < 2 or n % m or k % m:
        raise ValueError((m, n, k))
    q = k // m
    banks = (3 * m - 2 + (q - 1).bit_length() + 2 + m) // (m + 1) + 2
    return 8 * n * k + 4 * (q - 1) * m * n + 4 * (m * k + n * k) + 2 * n * k * (m - 1) // m + 4 * banks * m * n


def legal(shapes, P: int) -> bool:
    """Multipliers, output lanes, input lanes and array state of the physical arrays within the chip limits."""
    try:
        state = sum(state_bytes(*s) for s in shapes)
    except ValueError:
        return False
    return (
        sum(n * k for _, n, k in shapes) <= P
        and sum(n for _, n, _ in shapes) <= MAX_OUTPUT_LANES
        and sum(k for _, _, k in shapes) <= MAX_INPUT_LANES
        and state <= MAX_STATE_BYTES
    )


def vocabulary(model: str, fmt: str) -> np.ndarray:
    """Every height in HEIGHTS with N a multiple of lcm(m, 16) and K a multiple of lcm(m, group), both reduced to
    the smallest value with the same tile counts over the model's matrix dimensions; rows sorted by (m, n, k)."""
    cols, reds = gemm_dims(model)
    group = wl.FORMATS[fmt]["group"]
    nmax = min(MAX_N, max(cols))

    def class_min(dims, step, x):
        return max(-(-(-(-d // -(-d // x))) // step) * step for d in dims)

    rows = set()
    for m in HEIGHTS:
        an, ak = lcm(m, 16), lcm(m, group)
        ns = sorted({class_min(cols, an, n) for n in range(an, nmax + an, an)} & set(range(an, MAX_N + 1, an)))
        ks = sorted({class_min(reds, ak, k) for k in range(ak, max(reds) + ak, ak)})
        rows.update((m, n, k) for n in ns for k in ks if n * k <= MAX_MACS)
    return np.array(sorted(rows), np.int32)


class AnalyticModel:
    """Analytic per-window step time of one model in one weight format on the shape vocabulary.

    hbm: HBM bytes per cycle. drain(n), when given, adds the exposed drain latency of an n-column array to every
    task: expert_drains times per routed or shared expert, once per router task.
    """

    def __init__(self, root: Path, model: str, fmt: str, hbm: float = 8000.0, drain=None, expert_drains: int = 0):
        w = wl.MODELS[model]
        H, F, S, E, gate = w["hidden"], w["routed_width"], w["shared_width"], w["experts"], int(w["shared_gate"])
        self.root, self.model, self.fmt, self.hbm = root, model, fmt, hbm
        self.shapes = vocabulary(model, fmt)
        self.macs = self.shapes[:, 1].astype(np.int64) * self.shapes[:, 2]
        self.matrix = wl.matrix_table(model, self.shapes)
        self.vector = wl.vector_table(model)
        self.routed_weights, self.shared_weights, self.router_weights = 3 * H * F, 3 * H * S + gate * H, H * E
        self.bytes_per_weight = wl.FORMATS[fmt]["bytes_per_weight"]
        self.drain, self.expert_drains = drain, expert_drains

    def histograms(self, batches=(32, 128), split="dev") -> dict:
        return {B: wl.histograms(self.root, self.model, B, split) for B in batches}

    def items(self, idx: int, lanes: int, hh: np.ndarray):
        """Per histogram row and task group (routed at t = 1..128 tokens, shared, router): cycles per task,
        re-delivered bytes per task, task count; and the step's HBM bytes."""
        m, nh, live = int(self.shapes[idx, 0]), len(hh), hh[:, 129].astype(int)
        routed = self.matrix[idx, :128] + self.vector[lanes, :128]
        shared = self.matrix[idx, 128 + live - 1] + self.vector[lanes, 128 + live - 1]
        router = self.matrix[idx, 256 + live - 1] + self.vector[lanes, 256 + live - 1]
        cost = np.concatenate([np.broadcast_to(routed, (nh, 128)), shared[:, None], router[:, None]], 1).astype(float)
        if self.drain is not None:
            drain = self.drain(int(self.shapes[idx, 1]))
            cost[:, :129] += self.expert_drains * drain
            cost[:, 129] += drain
        passes = -(-np.arange(1, 129) // m) - 1
        bpw = self.bytes_per_weight
        redelivered = np.concatenate(
            [
                np.broadcast_to(passes * self.routed_weights * bpw, (nh, 128)),
                ((-(-live // m) - 1) * self.shared_weights * bpw)[:, None],
                ((-(-live // m) - 1) * self.router_weights * 2.0)[:, None],
            ],
            1,
        )
        count = np.concatenate([hh[:, 1:129], np.ones((nh, 2))], 1).astype(float)
        return cost, redelivered, count, self.step_bytes(hh)

    def step_bytes(self, hh: np.ndarray) -> np.ndarray:
        return (
            hh[:, 1:129].sum(1) * self.routed_weights + self.shared_weights
        ) * self.bytes_per_weight + self.router_weights * 2.0

    def floor(self, hh: np.ndarray) -> np.ndarray:
        return self.step_bytes(hh) / self.hbm

    def _redelivery_bound(self, redelivered, kappa):
        return redelivered / (kappa * self.hbm) if kappa > 0 else np.where(redelivered > 0, np.inf, 0.0)

    def single(self, i: int, hh: np.ndarray, kappa: float) -> np.ndarray:
        """One array runs every task."""
        c, rd, n, hbm = self.items(i, wl.VECTOR_LANES, hh)
        return np.maximum.reduce([(c * n).sum(1), hbm / self.hbm, self._redelivery_bound((rd * n).sum(1), kappa)])

    def homogeneous(self, i: int, hh: np.ndarray, kappa: float) -> np.ndarray:
        """Two identical arrays share the load evenly."""
        c, rd, n, hbm = self.items(i, wl.VECTOR_LANES // 2, hh)
        return np.maximum.reduce([(c * n).sum(1) / 2, hbm / self.hbm, self._redelivery_bound((rd * n).sum(1), kappa)])

    def pair(self, i: int, j: int, hh: np.ndarray, kappa: float) -> np.ndarray:
        """Shorter array i takes the cold end, taller array j the hot end of the tasks ordered by token count; the
        cut may fall inside a task group (fractional)."""
        p0, p1 = int(self.macs[i]), int(self.macs[j])
        v0 = max(1, wl.VECTOR_LANES * p0 // (p0 + p1))
        c1, r1, n, hbm = self.items(i, v0, hh)
        c2, r2, _, _ = self.items(j, wl.VECTOR_LANES - v0, hh)
        z = np.zeros((len(hh), 1))
        L1 = np.concatenate([z, np.cumsum(c1 * n, 1)], 1)
        L2 = (c2 * n).sum(1, keepdims=True) - np.concatenate([z, np.cumsum(c2 * n, 1)], 1)
        RR = (
            np.concatenate([z, np.cumsum(r1 * n, 1)], 1)
            + (r2 * n).sum(1, keepdims=True)
            - np.concatenate([z, np.cumsum(r2 * n, 1)], 1)
        )
        rb = self._redelivery_bound(RR, kappa)
        best = np.maximum.reduce([L1, L2, rb, np.broadcast_to((hbm / self.hbm)[:, None], L1.shape)]).min(1)
        g = np.clip(np.argmax(L1 - L2 >= 0, 1), 1, L1.shape[1] - 1)
        r = np.arange(len(hh))
        a0, a1, b0, b1 = L1[r, g - 1], L1[r, g], L2[r, g - 1], L2[r, g]
        x = np.clip((b0 - a0) / np.maximum((a1 - a0) + (b0 - b1), 1e-9), 0, 1)
        cut = np.maximum.reduce(
            [a0 + x * (a1 - a0), b0 + x * (b1 - b0), rb[r, g - 1] + x * (rb[r, g] - rb[r, g - 1]), hbm / self.hbm]
        )
        return np.minimum(best, cut)

    def relative_time(self, hists: dict, fn, args: tuple, kappa: float) -> float:
        """Mean over the histogram batches of the weighted step time over the weighted HBM floor."""
        kr = kappa if kappa > 0 else float("inf")
        return float(np.mean([wl.dot(w, fn(*args, h, kr)) / wl.dot(w, self.floor(h)) for h, w in hists.values()]))
