"""Weight formats, the routing capture each format uses, and the engine settings that every chip shares."""

from __future__ import annotations

import math
from dataclasses import replace

from ..sim import workload as wl

# bits: stored bits per weight; group: record width along K; scale_group: quantisation group inside a record (0: one scale
# per record row, -1: block scales outside the record); bytes_per_weight: HBM bytes per weight including the scales.
FORMATS = dict(wl.FORMATS)
# NVFP4: 16 x 64 records of 512 B values and 64 B scales (one scale per 32 weights).
FORMATS.setdefault("w4g32p", dict(bits=4, group=64, scale_group=32, bytes_per_weight=576 / 1024))
# FP8 E4M3 with 128 x 128 block scales kept outside the record.
FORMATS.setdefault("fp8b128", dict(bits=8, group=128, scale_group=-1, bytes_per_weight=1.0))
FORMAT_NAMES = {"bf16": "BF16", "w8g128": "INT8", "w4g32": "INT4", "w4g32p": "NVFP4", "fp8b128": "FP8"}
# Routing traces: 4-bit cells use the captures of the 4-bit model, the others those of the reference model.
ROUTES = {"bf16": "ref", "w8g128": "ref", "fp8b128": "ref", "w4g32": "int4", "w4g32p": "int4"}
# Systolic timing of every chip: segmented activation delivery, drain L(n) = 15 + ceil(log2(ceil(n / 16))) for n > 16.
TIMING = dict(systolic=3, drain_form="tree", drain_segment=16)
# Re-delivery rates of the analytic pre-ranking at the main point (8 TB/s, 16 KB/cycle port), in HBM weight rates.
MAIN_KAPPA = {"bf16": 1.0, "w8g128": 0.54, "w4g32": 0.33}


def converter_rate(fmt: str, hbm: int) -> int:
    """Weights decoded per cycle: the smallest power of two that keeps up with HBM (BF16 needs no decoding)."""
    return 4096 if fmt == "bf16" else max(2048, 2 ** math.ceil(math.log2(hbm / FORMATS[fmt]["bytes_per_weight"])))


def kappa(fmt: str, hbm: int = 8000, W: int = 16384) -> float:
    """Port share left after a first pass at the HBM rate, in weights per cycle over HBM weights per cycle."""
    if (hbm, W) == (8000, 16384) and fmt in MAIN_KAPPA:
        return MAIN_KAPPA[fmt]
    return ((W - hbm) / 2.0) / (hbm / FORMATS[fmt]["bytes_per_weight"])


def with_format(env, fmt: str, hbm: int):
    """Settings with the storage fields of fmt (bits, record width, scale group, converter); nothing else changes."""
    f = FORMATS[fmt]
    env = replace(env, bits=f["bits"], group=f["group"], converter_rate=converter_rate(fmt, hbm))
    return replace(env, scale_group=f["scale_group"]) if "scale_group" in f else env

