"""Effective-bandwidth model for the decode chip's HBM traffic.

Peak bandwidth (`bytes / (HBM_WIDTH/8 x freq)`) overstates what decode achieves:
its transfers are small-to-medium bursts from a single-slot load engine, so they
use only part of the channels and pay a fixed per-transfer latency.

Instead we use bandwidth measured on the emulator (Ramulator DMA in
--blocking-prefetch mode, see testbench/calibration/kernel_sweep.py), split by
traffic class:

  weights_kv  — H_PREFETCH_M   HBM -> Matrix SRAM (weights + KV tiles)
  activations — H_PREFETCH_V   HBM -> Vector SRAM (small vector loads)
  writeback   — H_STORE_V      Vector SRAM -> HBM (read-modify-write)

Bandwidth is looked up by (class, gen, channels); channel counts between
calibrated points are filled in by log-log interpolation. memory_time()
replaces the peak-bandwidth division.

CLI: python analytic_models/disagg_serve/memory.py --report [--holdout 16]
prints the calibration table and the leave-one-channel-out error.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = _HERE / "calibration_bw.csv"
DEFAULT_DMA_CALIBRATION = _HERE / "calibration_dma.csv"

# op mnemonic (measurement) -> traffic class (model)
OP_TO_CLASS = {
    "H_PREFETCH_M": "weights_kv",
    "H_PREFETCH_V": "activations",
    "H_STORE_V": "writeback",
}
CLASSES = tuple(OP_TO_CLASS.values())


@dataclass(frozen=True)
class _Point:
    kv_size: int
    gen: str
    channels: int
    op: str
    bytes: int
    dt_ps: int

    @property
    def gbps(self) -> float:
        return self.bytes / (self.dt_ps / 1e12) / 1e9 if self.dt_ps else 0.0


class TransferSizeModel:
    """Per-transfer DMA cost fitted from the dma_microbench sweep:
    time = t0 + bytes / bw_inf, one line per (gen, channels). Bandwidth is then
    bw(size) = size / (t0 + size/bw_inf), which rises with transfer size and
    saturates at bw_inf for large transfers (what large-MLEN chips issue),
    rather than being fixed at the decode testbench's transfer size."""

    def __init__(self, csv_path: str | Path = DEFAULT_DMA_CALIBRATION):
        pts: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (row["hbm_gen"], int(row["channels"]))
                pts[key].append(
                    (float(row["bytes_per_transfer"]), float(row["dt_ps_per_transfer"]) / 1e12)
                )
        # Fit a line t = t0 + slope * bytes for each (gen, channels).
        self._fit: dict[tuple[str, int], tuple[float, float]] = {}
        for key, xy in pts.items():
            n = len(xy)
            sx = sum(x for x, _ in xy)
            sy = sum(y for _, y in xy)
            sxx = sum(x * x for x, _ in xy)
            sxy = sum(x * y for x, y in xy)
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            t0 = (sy - slope * sx) / n
            self._fit[key] = (max(t0, 0.0), max(slope, 1e-15))

    def channel_counts(self, gen: str) -> list[int]:
        return sorted(ch for (g, ch) in self._fit if g == gen)

    def _params(self, gen: str, channels: int) -> tuple[float, float]:
        if (gen, channels) in self._fit:
            return self._fit[(gen, channels)]
        chs = self.channel_counts(gen)
        if not chs:
            raise KeyError(f"no DMA calibration for gen={gen}")
        if channels <= chs[0]:
            return self._fit[(gen, chs[0])]
        if channels >= chs[-1]:
            return self._fit[(gen, chs[-1])]
        lo = max(c for c in chs if c < channels)
        hi = min(c for c in chs if c > channels)
        f = (math.log(channels) - math.log(lo)) / (math.log(hi) - math.log(lo))
        t0l, sl = self._fit[(gen, lo)]
        t0h, sh = self._fit[(gen, hi)]
        interp = lambda a, b: math.exp(math.log(a) + f * (math.log(b) - math.log(a)))
        return interp(t0l, t0h), interp(sl, sh)

    def bw_gbps(self, gen: str, channels: int, transfer_bytes: float) -> float:
        t0, slope = self._params(gen, channels)
        t = t0 + slope * transfer_bytes
        return transfer_bytes / t / 1e9

    def holdout_report(self, holdout_amount_bytes: float = 32 * 1024) -> str:
        """For each (gen, ch), refit without the point nearest
        `holdout_amount_bytes` and report the error at that held-out point."""
        lines = [f"transfer-size holdout near {holdout_amount_bytes/1024:.0f} KiB"]
        errs = []
        with open(DEFAULT_DMA_CALIBRATION) as f:
            rows = list(csv.DictReader(f))
        keys = sorted({(r["hbm_gen"], int(r["channels"])) for r in rows})
        for gen, ch in keys:
            sub = [r for r in rows if r["hbm_gen"] == gen and int(r["channels"]) == ch]
            held = min(sub, key=lambda r: abs(float(r["bytes_per_transfer"]) - holdout_amount_bytes))
            rest = [r for r in sub if r is not held]
            n = len(rest)
            xs = [float(r["bytes_per_transfer"]) for r in rest]
            ys = [float(r["dt_ps_per_transfer"]) / 1e12 for r in rest]
            slope = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / (
                n * sum(x * x for x in xs) - sum(xs) ** 2)
            t0 = (sum(ys) - slope * sum(xs)) / n
            hb = float(held["bytes_per_transfer"])
            measured = hb / (float(held["dt_ps_per_transfer"]) / 1e12) / 1e9
            pred = hb / (t0 + slope * hb) / 1e9
            err = abs(pred - measured) / measured * 100
            errs.append(err)
            lines.append(f"  {gen:<6} ch={ch:<3} measured {measured:7.1f} GB/s  "
                         f"predicted {pred:7.1f}  err {err:4.1f}%")
        if errs:
            errs.sort()
            lines.append(f"  median {errs[len(errs)//2]:.1f}%   max {errs[-1]:.1f}%   n={len(errs)}")
        return "\n".join(lines)


class CalibratedBandwidth:
    """Effective-bandwidth lookup fitted from a kernel_sweep calibration CSV."""

    def __init__(self, points: list[_Point]):
        self._points = points
        self.size_model: TransferSizeModel | None = None
        # Bandwidth per (class, gen, channels), aggregated over kv sizes and
        # weighted by bytes so the large transfers that dominate decode also
        # dominate the estimate.
        acc: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
        for p in points:
            if p.op not in OP_TO_CLASS:
                continue
            key = (OP_TO_CLASS[p.op], p.gen, p.channels)
            acc[key][0] += p.bytes
            acc[key][1] += p.dt_ps
        self._bw: dict[tuple[str, str, int], float] = {
            key: (b / (t / 1e12) / 1e9) for key, (b, t) in acc.items() if t > 0
        }

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, csv_path: str | Path = DEFAULT_CALIBRATION) -> "CalibratedBandwidth":
        points = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                points.append(
                    _Point(
                        kv_size=int(row["kv_size"]),
                        gen=row["hbm_gen"],
                        channels=int(row["channels"]),
                        op=row["op"],
                        bytes=int(row["bytes"]),
                        dt_ps=int(row["dt_ps"]),
                    )
                )
        model = cls(points)
        # Use the size-aware curve for the weights/KV stream if the DMA sweep
        # exists; otherwise fall back to the per-class table.
        if DEFAULT_DMA_CALIBRATION.exists():
            model.size_model = TransferSizeModel()
        return model

    # -- queries -------------------------------------------------------------

    def channel_counts(self, cls_name: str, gen: str) -> list[int]:
        return sorted(ch for (c, g, ch) in self._bw if c == cls_name and g == gen)

    def bw_gbps(self, cls_name: str, gen: str, channels: int) -> float:
        """Effective bandwidth in GB/s; log-log interpolation across channels."""
        key = (cls_name, gen, channels)
        if key in self._bw:
            return self._bw[key]
        chs = self.channel_counts(cls_name, gen)
        if not chs:
            raise KeyError(f"no calibration for class={cls_name} gen={gen}")
        # Clamp outside the calibrated range: channel scaling saturates, so
        # extrapolating the slope upward would overstate bandwidth.
        if channels <= chs[0]:
            return self._bw[(cls_name, gen, chs[0])]
        if channels >= chs[-1]:
            return self._bw[(cls_name, gen, chs[-1])]
        lo = max(c for c in chs if c < channels)
        hi = min(c for c in chs if c > channels)
        bw_lo = self._bw[(cls_name, gen, lo)]
        bw_hi = self._bw[(cls_name, gen, hi)]
        frac = (math.log(channels) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return math.exp(math.log(bw_lo) + frac * (math.log(bw_hi) - math.log(bw_lo)))

    def memory_time(
        self,
        bytes_by_class: dict[str, float],
        gen: str,
        channels: int,
        transfer_bytes: float | None = None,
    ) -> float:
        """Seconds to move the given per-class byte counts, added up serially.

        Each engine runs one DMA at a time and decode is dominated by the
        weights_kv stream, so summing the class times is a fair first-order
        estimate. `transfer_bytes` is the weights/KV per-DMA size; if given (and
        the DMA sweep exists) that stream is priced on the size-aware curve.
        """
        total = 0.0
        for cls_name, nbytes in bytes_by_class.items():
            if nbytes <= 0:
                continue
            if cls_name == "weights_kv" and transfer_bytes and self.size_model is not None:
                bw = self.size_model.bw_gbps(gen, channels, transfer_bytes)
            else:
                bw = self.bw_gbps(cls_name, gen, channels)
            total += nbytes / (bw * 1e9)
        return total

    # -- reporting / validation ---------------------------------------------

    def table(self) -> str:
        lines = [f"{'class':<12}{'gen':<6}{'ch':>4}{'BW_eff GB/s':>14}"]
        for (cls_name, gen, ch), bw in sorted(self._bw.items()):
            lines.append(f"{cls_name:<12}{gen:<6}{ch:>4}{bw:>14.1f}")
        return "\n".join(lines)

    def holdout_report(self, holdout_channels: int) -> str:
        """Hold out one channel count and predict it from the rest by interpolation."""
        errs = []
        lines = [f"holdout: channels={holdout_channels}"]
        full = self._bw
        rest = CalibratedBandwidth(
            [p for p in self._points if p.channels != holdout_channels]
        )
        for (cls_name, gen, ch), measured in sorted(full.items()):
            if ch != holdout_channels:
                continue
            pred = rest.bw_gbps(cls_name, gen, ch)
            err = abs(pred - measured) / measured * 100
            errs.append(err)
            lines.append(
                f"  {cls_name:<12}{gen:<6} measured {measured:8.1f}  "
                f"predicted {pred:8.1f}  err {err:5.1f}%"
            )
        if errs:
            errs.sort()
            median = errs[len(errs) // 2]
            p95 = errs[min(len(errs) - 1, int(0.95 * len(errs)))]
            lines.append(f"  median {median:.1f}%   P95 {p95:.1f}%   n={len(errs)}")
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--holdout", type=int, default=16,
                    help="channel count to hold out for the fit-error report")
    args = ap.parse_args()

    model = CalibratedBandwidth.load(args.calibration)
    if args.report:
        print(model.table())
        print()
        print(model.holdout_report(args.holdout))


if __name__ == "__main__":
    main()
