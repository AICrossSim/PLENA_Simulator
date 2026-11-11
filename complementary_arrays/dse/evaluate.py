"""Cycle-level evaluation of chips on the windows of one cell (model, weight format, multiplier budget)."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import CapacityError, counters, simulate
from . import shapes as sh

INF = float("inf")
MECHANISM = (
    "hbm_busy",
    "port_full",
    "wait_hbm",
    "wait_wport",
    "wait_x",
    "wait_latency",
    "wait_decode",
    "computing",
    "input_stall",
    "merge_stall",
    "idle",
    "delivery_wait",
    "array0_busy",
    "array1_busy",
)


def converter_rate(fmt: str, hbm: int) -> int:
    """Weights decoded per cycle: the smallest power of two that keeps up with HBM (BF16 needs no decoding)."""
    return 4096 if fmt == "bf16" else max(2048, 2 ** math.ceil(math.log2(hbm / wl.FORMATS[fmt]["bytes_per_weight"])))


def sram_mib(P: int) -> int:
    """16 MiB up to 128K multipliers, then in proportion to the multipliers."""
    return 16 * max(1, P // 131072)


def stream_busy(c: dict, streams: int) -> np.ndarray:
    """Enabled cycles of the first two task streams (up to two streams) or of the two arrays (four streams)."""
    return c["ctx_enabled"][:, :2] if streams <= 2 else c["array_busy"][:, :2]


class Evaluator:
    def __init__(self, root: Path, model: str, fmt: str, P: int, hbm_tbps: int = 8, threads: int = 3):
        self.root, self.model, self.fmt, self.P, self.threads = Path(root), model, fmt, P, threads
        self.hbm = int(hbm_tbps * 1000)
        self.converter = converter_rate(fmt, self.hbm)
        self.rejects, self.evaluations = [], 0

    def windows(self, batch: int, kind: str) -> list[dict]:
        return wl.windows(self.root, self.model, batch, kind)

    def legal(self, base) -> bool:
        return sh.legal(base, self.P)

    def outputs(self, rows: list[dict], shapes, env):
        """Engine output of the windows, one engine call per live token count."""
        live = np.array([r["live"] for r in rows])
        for b in np.unique(live):
            ix = np.flatnonzero(live == b)
            yield (
                ix,
                simulate(
                    np.stack([rows[i]["ids"] for i in ix]), shapes, wl.MODELS[self.model], env, threads=self.threads
                ),
            )

    def run(self, rows: list[dict], shapes, env) -> np.ndarray:
        """Engine output rows of the windows, as floats."""
        values = None
        for ix, out in self.outputs(rows, shapes, env):
            if values is None:
                values = np.zeros((len(rows), out.shape[1]))
            values[ix] = out
        return values

    def score(self, shapes, env, kind: str, batches) -> float:
        """Geometric mean over batches of the window-weighted mean layer cycles; inf if the chip does not fit."""
        self.evaluations += 1
        try:
            t = []
            for B in batches:
                rows = self.windows(B, kind)
                t.append(wl.dot(wl.window_weights(rows), counters(self.run(rows, shapes, env))["cycles"]))
            return wl.geomean(t)
        except CapacityError as ex:
            self.rejects.append(dict(shapes=shapes, env=asdict(env), reason=str(ex)))
            return INF

    def per_window(self, shapes, env, kind: str, batches, mechanism: bool = False) -> dict:
        """Per batch: weighted mean layer cycles and busy shares; report windows add the per-window cycles, weights
        and layers; mechanism adds the time shares of the engine's stall counters."""
        S, res = len(shapes), {}
        for B in batches:
            rows = self.windows(B, kind)
            w, n = wl.window_weights(rows), len(rows)
            cyc, busy, ext = np.zeros(n), np.zeros((n, 2)), np.zeros((n, 15))
            for ix, out in self.outputs(rows, shapes, env):
                c = counters(out)
                cyc[ix] = c["cycles"]
                busy[ix] = stream_busy(c, S)
                if mechanism:
                    ext[ix] = np.stack(
                        [
                            c["hbm_service"] / self.hbm,
                            c["delivery_limited_cycles"],
                            c["wait_hbm"],
                            c["wait_wport"],
                            c["wait_x"],
                            c["wait_latency"],
                            c["wait_decode"],
                            *(
                                c["ctx_" + f][:, :4].sum(1)
                                for f in ("enabled", "input_stall", "merge_stall", "idle", "delivery_wait")
                            ),
                            *busy[ix].T,
                            c["cycles"],
                        ],
                        1,
                    )
            d = dict(
                cycles=wl.dot(w, cyc) / float(w.sum()),
                busy=[wl.dot(w, busy[:, j] / cyc) / float(w.sum()) for j in range(2)],
            )
            if kind == "report":
                d.update(
                    window_cycles=cyc.tolist(), window_weights=w.tolist(), window_layers=[int(r["layer"]) for r in rows]
                )
            if mechanism:
                d["mechanism"] = {
                    k: wl.dot(w, ext[:, j] / ext[:, 14]) / float(w.sum()) / (S if 2 <= j <= 11 else 1)
                    for j, k in enumerate(MECHANISM)
                }
                d["mechanism"]["streams_averaged"] = S
            res[f"B{B}"] = d
        return res
