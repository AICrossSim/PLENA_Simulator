#!/usr/bin/env python3
"""Calibrate MatrixMachine area from PE through complete hardware hierarchy.

This script is intentionally self-contained so calibration can run from the
PLENA_Simulator repo while using lightweight PLENA_RTL worker copies in /tmp.

MXINT and MXFP are fitted independently. Cheap PE and mini-array sweeps learn
precision/block scaling; sparse MatrixMachine anchors learn reduction,
accumulator, and top-level effects. DC hierarchy reports are retained so the
fit can be supervised by submodule area rather than total area alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count
except ModuleNotFoundError:
    from .license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count

ROOT = Path(__file__).resolve().parents[3]
RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_area_workers_structural_v4")
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"
GIB = 1024**3
TMP_FIXED_HEADROOM_GIB = 10
TMP_PER_WORKER_GIB = 6
TMP_HARD_MIN_GIB = 15

CSV_FIELDS = [
    "point_key",
    "point_id",
    "level",
    "mode",
    "module",
    "top_module",
    "status",
    "worker_id",
    "elapsed_sec",
    "area_um2",
    "dynamic_power",
    "leakage_power",
    "total_power",
    "report_dir",
    "summary_log",
    "failure_reason",
    "ACT_WIDTH",
    "KV_WIDTH",
    "WEIGHT_WIDTH",
    "T_BITS",
    "L_BITS",
    "T_EXP",
    "T_MANT",
    "L_EXP",
    "L_MANT",
    "BLOCK_DIM",
    "ACC_DEPTH",
    "ACC_WIDTH",
    "ACC_FRAC_WIDTH",
    "FP_EXP_WIDTH",
    "FP_MANT_WIDTH",
    "COMPUTE_DIM",
    "SYS_ARRAY_AMOUNT",
    "STRUCTURAL_COMPOSITE",
    "PE_AREA_UM2",
    "raw_synth_area_um2",
    "MLEN",
    "BLEN",
    "scale_width",
    "hier_total_area",
    "hier_compute_unit_area",
    "hier_array_area",
    "hier_reduce_area",
    "hier_output_accumulator_area",
    "hier_output_conversion_area",
    "hier_result_buffer_area",
    "hier_io_pipeline_area",
    "hier_control_area",
    "hier_accum_area",
    "hier_top_glue_area",
]


@dataclass(frozen=True)
class Point:
    """One immutable PE, mini-array, or MatrixMachine calibration point."""
    point_id: str
    level: str
    mode: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "level": self.level,
            "mode": self.mode,
            "module": self.module,
            "top_module": self.top_module,
            "params": self.params,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"{self.level}_{self.mode}_{digest}")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()


def mxfp_name(exp: int, mant: int) -> str:
    """Return the canonical software token for one MXFP format."""
    return f"MXFP_E{exp}M{mant}"


def mxfp_width(exp: int, mant: int) -> int:
    """Return sign + exponent + mantissa bits."""
    return 1 + exp + mant


def lookup_mxfp_pe_area(t_exp: int, t_mant: int, l_exp: int, l_mant: int) -> float:
    """Return the committed leaf area for one concrete MXFP PE capability."""

    path = CALIBRATION_DIR / "pe_mxfp.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"MXFP PE calibration is required before structural composition: {path}"
        )
    with path.open(newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "complete"
            and int(row["T_EXP"]) == t_exp
            and int(row["T_MANT"]) == t_mant
            and int(row["L_EXP"]) == l_exp
            and int(row["L_MANT"]) == l_mant
        ]
    if not matches:
        raise ValueError(
            "No MXFP PE leaf area for "
            f"T=E{t_exp}M{t_mant}, L=E{l_exp}M{l_mant}"
        )
    return float(matches[-1]["area_um2"])


def unique_mxfp_side_pairs() -> list[tuple[int, int, int, int]]:
    """Enumerate unique T/L PE capabilities induced by software profiles."""
    formats = [(1, 2), (2, 1), (4, 3), (5, 2)]
    pairs: set[tuple[int, int, int, int]] = set()
    for act_exp, act_mant in formats:
        for kv_exp, kv_mant in formats:
            for wt_exp, wt_mant in formats:
                pairs.add((max(kv_exp, wt_exp), max(kv_mant, wt_mant), act_exp, act_mant))
    return sorted(pairs)


def representative_mxfp_profiles() -> list[dict[str, Any]]:
    """Return compact same-format and asymmetric mini-array anchor profiles."""
    raw = [
        ((1, 2), (1, 2), (1, 2)),
        ((2, 1), (2, 1), (2, 1)),
        ((4, 3), (4, 3), (4, 3)),
        ((5, 2), (5, 2), (5, 2)),
        ((1, 2), (4, 3), (4, 3)),
        ((4, 3), (1, 2), (1, 2)),
        ((2, 1), (5, 2), (5, 2)),
        ((5, 2), (2, 1), (2, 1)),
    ]
    profiles = []
    for act, kv, wt in raw:
        profiles.append(
            {
                "ACT_WIDTH": mxfp_name(*act),
                "KV_WIDTH": mxfp_name(*kv),
                "WEIGHT_WIDTH": mxfp_name(*wt),
                "T_EXP": max(kv[0], wt[0]),
                "T_MANT": max(kv[1], wt[1]),
                "L_EXP": act[0],
                "L_MANT": act[1],
            }
        )
    return profiles


def build_plan(mode: str) -> list[Point]:
    """Build a named hierarchical calibration or refinement point set."""
    points: list[Point] = []
    if mode == "structural-v4-leaves":
        # Accumulator-depth anchors isolate the logarithmic widening inside the
        # integer PE without synthesizing a large MatrixMachine.
        for t_bits, l_bits, acc_depth in [
            (4, 4, 64),
            (4, 4, 256),
            (4, 4, 1024),
            (4, 2, 1024),
            (8, 8, 1024),
        ]:
            top = f"area_new_v4_mxint_pe_t{t_bits}_l{l_bits}_acc{acc_depth}"
            points.append(
                Point(
                    point_id=top,
                    level="pe",
                    mode="mxint",
                    module="mxint_default_pe",
                    top_module=top,
                    params={
                        "T_BITS": t_bits,
                        "L_BITS": l_bits,
                        "ACC_DEPTH": acc_depth,
                        "scale_width": 8,
                    },
                )
            )

        # B=32 confirms the B^2 PE census without monolithically compiling 1024
        # identical FP PEs. The real shift/valid periphery is synthesized and
        # composed with B^2 independently synthesized PE leaves.
        for t_exp, t_mant, l_exp, l_mant in [
            (1, 2, 1, 2),
            (4, 3, 4, 3),
            (5, 2, 2, 1),
        ]:
            top = (
                f"area_new_v4_mxfp_mini_te{t_exp}m{t_mant}_"
                f"le{l_exp}m{l_mant}_b32"
            )
            points.append(
                Point(
                    point_id=top,
                    level="mini_array",
                    mode="mxfp",
                    module="mx_mini_systolic_array",
                    top_module=top,
                    params={
                        "T_EXP": t_exp,
                        "T_MANT": t_mant,
                        "L_EXP": l_exp,
                        "L_MANT": l_mant,
                        "BLOCK_DIM": 32,
                        "STRUCTURAL_COMPOSITE": 1,
                        "PE_AREA_UM2": lookup_mxfp_pe_area(
                            t_exp, t_mant, l_exp, l_mant
                        ),
                        "scale_width": 8,
                    },
                )
            )

        for t_bits, l_bits, acc_width in [(4, 2, 28), (8, 8, 40)]:
            top = f"area_new_v4_mxint_acc2fp_t{t_bits}_l{l_bits}_a{acc_width}"
            points.append(
                Point(
                    point_id=top,
                    level="output_conversion",
                    mode="mxint",
                    module="mxint_acc_2_fp",
                    top_module=top,
                    params={
                        "T_BITS": t_bits,
                        "L_BITS": l_bits,
                        "ACC_WIDTH": acc_width,
                        "ACC_FRAC_WIDTH": t_bits + l_bits - 2,
                        "FP_EXP_WIDTH": 5,
                        "FP_MANT_WIDTH": 6,
                        "scale_width": 8,
                    },
                )
            )

        points.append(
            Point(
                point_id="area_new_v4_mxint_reduce_b4_s4_t4_l4",
                level="reduce_leaf",
                mode="mxint",
                module="mxint_sum_across",
                top_module="area_new_v4_mxint_reduce_b4_s4_t4_l4",
                params={
                    "T_BITS": 4,
                    "L_BITS": 4,
                    "ACC_WIDTH": 12,
                    "COMPUTE_DIM": 4,
                    "SYS_ARRAY_AMOUNT": 4,
                    "scale_width": 8,
                },
            )
        )
        points.append(
            Point(
                point_id="area_new_v4_mxfp_reduce_b4_s4_e8m7",
                level="reduce_leaf",
                mode="mxfp",
                module="mx_sum_across_sa",
                top_module="area_new_v4_mxfp_reduce_b4_s4_e8m7",
                params={
                    "T_EXP": 4,
                    "T_MANT": 3,
                    "L_EXP": 4,
                    "L_MANT": 3,
                    "FP_EXP_WIDTH": 8,
                    "FP_MANT_WIDTH": 7,
                    "COMPUTE_DIM": 4,
                    "SYS_ARRAY_AMOUNT": 4,
                    "scale_width": 8,
                },
            )
        )
        return points
    if mode == "structural-v4-composite-smoke":
        for block_dim in (8, 16):
            top = (
                "area_new_v4_mxfp_mini_composite_"
                f"te1m2_le1m2_b{block_dim}"
            )
            points.append(
                Point(
                    point_id=top,
                    level="mini_array",
                    mode="mxfp",
                    module="mx_mini_systolic_array",
                    top_module=top,
                    params={
                        "T_EXP": 1,
                        "T_MANT": 2,
                        "L_EXP": 1,
                        "L_MANT": 2,
                        "BLOCK_DIM": block_dim,
                        "STRUCTURAL_COMPOSITE": 1,
                        "PE_AREA_UM2": lookup_mxfp_pe_area(1, 2, 1, 2),
                        "scale_width": 8,
                    },
                )
            )
        return points
    if mode == "matrix-small":
        mxint_profiles = [
            ("mxint_t4_l2_m16_b4", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 2}, (16, 4)),
            ("mxint_t4_l8_m16_b4", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 8}, (16, 4)),
            ("mxint_t8_l4_m16_b4", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 4}, (16, 4)),
            ("mxint_t4_l4_m32_b8", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 4}, (32, 8)),
            ("mxint_t8_l8_m32_b8", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "T_BITS": 8, "L_BITS": 8}, (32, 8)),
            ("mxint_t8_l2_m32_b8", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 2}, (32, 8)),
        ]
        for suffix, profile, shape in mxint_profiles:
            mlen, blen = shape
            params = dict(profile, MLEN=mlen, BLEN=blen, scale_width=8)
            points.append(
                Point(
                    point_id=f"area_new_mm_{suffix}",
                    level="matrix_machine",
                    mode="mxint",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )

        mxfp_profiles = [
            ("mxfp_e2m1_m16_b4", (2, 1), (2, 1), (16, 4)),
            ("mxfp_e5m2_m16_b4", (5, 2), (5, 2), (16, 4)),
            ("mxfp_te4m3_le1m2_m16_b4", (4, 3), (1, 2), (16, 4)),
            ("mxfp_te1m2_le4m3_m16_b4", (1, 2), (4, 3), (16, 4)),
            ("mxfp_e4m3_m32_b8", (4, 3), (4, 3), (32, 8)),
            ("mxfp_te5m2_le2m1_m32_b8", (5, 2), (2, 1), (32, 8)),
        ]
        for suffix, t_fmt, l_fmt, shape in mxfp_profiles:
            mlen, blen = shape
            t_exp, t_mant = t_fmt
            l_exp, l_mant = l_fmt
            params = {
                "ACT_WIDTH": mxfp_name(l_exp, l_mant),
                "KV_WIDTH": mxfp_name(t_exp, t_mant),
                "WEIGHT_WIDTH": mxfp_name(t_exp, t_mant),
                "T_EXP": t_exp,
                "T_MANT": t_mant,
                "L_EXP": l_exp,
                "L_MANT": l_mant,
                "MLEN": mlen,
                "BLEN": blen,
                "scale_width": 8,
            }
            points.append(
                Point(
                    point_id=f"area_new_mm_{suffix}",
                    level="matrix_machine",
                    mode="mxfp",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )
        return points

    if mode == "matrix-mxint-refine-v1":
        mxint_profiles = [
            ("mxint_t8_l4_m32_b4", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 4}, (32, 4)),
            ("mxint_t4_l8_m32_b4", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 8}, (32, 4)),
            ("mxint_t8_l8_m16_b4", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "T_BITS": 8, "L_BITS": 8}, (16, 4)),
            ("mxint_t8_l8_m64_b8", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "T_BITS": 8, "L_BITS": 8}, (64, 8)),
        ]
        for suffix, profile, shape in mxint_profiles:
            mlen, blen = shape
            params = dict(profile, MLEN=mlen, BLEN=blen, scale_width=8)
            points.append(
                Point(
                    point_id=f"area_new_mm_{suffix}",
                    level="matrix_machine",
                    mode="mxint",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )
        return points

    if mode == "matrix-mxint-small-grid-v2":
        mxint_profiles = [
            # Fill the cheap missing precision grid around the current worst-error region.
            ("mxint_t8_l2_m16_b4", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 2}, (16, 4)),
            ("mxint_t4_l2_m32_b4", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 2}, (32, 4)),
            ("mxint_t4_l4_m32_b4", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 4}, (32, 4)),
            ("mxint_t8_l2_m32_b4", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 2}, (32, 4)),
            # Add small BLEN=8 anchors to decouple BLEN^2 accum/glue from precision width.
            ("mxint_t4_l2_m16_b8", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 2}, (16, 8)),
            ("mxint_t4_l4_m16_b8", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 4}, (16, 8)),
            ("mxint_t8_l4_m16_b8", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 4}, (16, 8)),
            ("mxint_t8_l8_m16_b8", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "T_BITS": 8, "L_BITS": 8}, (16, 8)),
            ("mxint_t4_l2_m32_b8", {"ACT_WIDTH": "MXINT2", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 2}, (32, 8)),
            ("mxint_t4_l8_m32_b8", {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 8}, (32, 8)),
            ("mxint_t8_l4_m32_b8", {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 8, "L_BITS": 4}, (32, 8)),
        ]
        for suffix, profile, shape in mxint_profiles:
            mlen, blen = shape
            params = dict(profile, MLEN=mlen, BLEN=blen, scale_width=8)
            points.append(
                Point(
                    point_id=f"area_new_mm_{suffix}",
                    level="matrix_machine",
                    mode="mxint",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )
        return points

    if mode in {"mxint", "both"}:
        for t_bits in [4, 8]:
            for l_bits in [2, 4, 8]:
                top = f"area_new_mxint_pe_t{t_bits}_l{l_bits}_acc16"
                points.append(
                    Point(
                        point_id=top,
                        level="pe",
                        mode="mxint",
                        module="mxint_default_pe",
                        top_module=top,
                        params={"T_BITS": t_bits, "L_BITS": l_bits, "ACC_DEPTH": 16, "scale_width": 8},
                    )
                )
        for t_bits in [4, 8]:
            for l_bits in [2, 4, 8]:
                for block_dim in [4, 8, 16, 32]:
                    top = f"area_new_mxint_mini_t{t_bits}_l{l_bits}_b{block_dim}"
                    points.append(
                        Point(
                            point_id=top,
                            level="mini_array",
                            mode="mxint",
                            module="mxint_mini_systolic_array",
                            top_module=top,
                            params={
                                "T_BITS": t_bits,
                                "L_BITS": l_bits,
                                "BLOCK_DIM": block_dim,
                                "ACC_DEPTH": 16,
                                "scale_width": 8,
                            },
                        )
                    )
        matrix_profiles = [
            {"ACT_WIDTH": "MXINT4", "KV_WIDTH": "MXINT4", "WEIGHT_WIDTH": "MXINT4", "T_BITS": 4, "L_BITS": 4},
            {"ACT_WIDTH": "MXINT8", "KV_WIDTH": "MXINT8", "WEIGHT_WIDTH": "MXINT8", "T_BITS": 8, "L_BITS": 8},
        ]
        shapes = [(16, 4), (32, 4)]
        for profile, shape in zip(matrix_profiles, shapes, strict=True):
            mlen, blen = shape
            point_id = f"area_new_mm_mxint_t{profile['T_BITS']}_l{profile['L_BITS']}_m{mlen}_b{blen}"
            params = dict(profile, MLEN=mlen, BLEN=blen, scale_width=8)
            points.append(
                Point(
                    point_id=point_id,
                    level="matrix_machine",
                    mode="mxint",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )

    if mode in {"mxfp", "both"}:
        for t_exp, t_mant, l_exp, l_mant in unique_mxfp_side_pairs():
            top = f"area_new_mxfp_pe_te{t_exp}m{t_mant}_le{l_exp}m{l_mant}"
            points.append(
                Point(
                    point_id=top,
                    level="pe",
                    mode="mxfp",
                    module="mxfp_default_pe",
                    top_module=top,
                    params={
                        "T_EXP": t_exp,
                        "T_MANT": t_mant,
                        "L_EXP": l_exp,
                        "L_MANT": l_mant,
                        "scale_width": 8,
                    },
                )
            )
        for profile in representative_mxfp_profiles():
            for block_dim in [4, 8, 16, 32]:
                top = (
                    "area_new_mxfp_mini_"
                    f"te{profile['T_EXP']}m{profile['T_MANT']}_"
                    f"le{profile['L_EXP']}m{profile['L_MANT']}_b{block_dim}"
                )
                params = dict(profile, BLOCK_DIM=block_dim, scale_width=8)
                points.append(
                    Point(
                        point_id=top,
                        level="mini_array",
                        mode="mxfp",
                        module="mx_mini_systolic_array",
                        top_module=top,
                        params=params,
                    )
                )
        for fmt in [(1, 2), (4, 3)]:
            mlen, blen = (16, 4)
            fmt_name = mxfp_name(*fmt)
            point_id = f"area_new_mm_mxfp_e{fmt[0]}m{fmt[1]}_m{mlen}_b{blen}"
            params = {
                "ACT_WIDTH": fmt_name,
                "KV_WIDTH": fmt_name,
                "WEIGHT_WIDTH": fmt_name,
                "T_EXP": fmt[0],
                "T_MANT": fmt[1],
                "L_EXP": fmt[0],
                "L_MANT": fmt[1],
                "MLEN": mlen,
                "BLEN": blen,
                "scale_width": 8,
            }
            points.append(
                Point(
                    point_id=point_id,
                    level="matrix_machine",
                    mode="mxfp",
                    module="matrix_machine",
                    top_module="matrix_machine",
                    params=params,
                )
            )
    return points


def write_plan_csv(points: list[Point], path: Path) -> None:
    """Persist the complete planned sweep before synthesis starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point_to_row(point, status="planned"))


def read_completed_keys(path: Path) -> set[str]:
    """Return successful point keys used to resume an interrupted run."""
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                completed.add(str(row["point_key"]))
    return completed


def point_to_row(
    point: Point,
    *,
    status: str,
    worker_id: int | str = "",
    elapsed_sec: float | str = "",
    area_um2: float | str = "",
    dynamic_power: float | str | None = "",
    leakage_power: float | str | None = "",
    total_power: float | str | None = "",
    report_dir: str = "",
    summary_log: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    """Serialize a point outcome into the stable Matrix calibration schema."""
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "point_key": point.point_key,
            "point_id": point.point_id,
            "level": point.level,
            "mode": point.mode,
            "module": point.module,
            "top_module": point.top_module,
            "status": status,
            "worker_id": worker_id,
            "elapsed_sec": elapsed_sec,
            "area_um2": area_um2,
            "dynamic_power": "" if dynamic_power is None else dynamic_power,
            "leakage_power": "" if leakage_power is None else leakage_power,
            "total_power": "" if total_power is None else total_power,
            "report_dir": report_dir,
            "summary_log": summary_log,
            "failure_reason": failure_reason,
        }
    )
    for key, value in point.params.items():
        if key in row:
            row[key] = value
    return row


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    """Append one completed attempt under a shared writer lock."""
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def normalize_csv_schema(path: Path) -> None:
    """Rewrite a resumable raw CSV when new diagnostic columns are added."""

    if not path.exists():
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = list(reader)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def create_worker_copy(worker_id: int, worker_root: Path, source_root: Path) -> Path:
    """Create a lightweight isolated RTL tree for one concurrent DC worker.

    Version-control metadata, environments, caches, and old build products are
    excluded to control /tmp usage. Calibration-only fixes are applied to the
    copy and never modify the source checkout.
    """
    dest = worker_root / f"worker_{worker_id}" / "PLENA_RTL"
    if dest.exists():
        shutil.rmtree(dest)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        skip = {
            ".git",
            ".venv",
            ".direnv",
            "build",
            "result",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "node_modules",
        }
        return {name for name in names if name in skip or name.endswith(".pyc")}

    shutil.copytree(source_root, dest, ignore=ignore)
    patch_worker_rtl(dest)
    return dest


def patch_worker_rtl(worker_rtl: Path) -> None:
    """Apply calibration-local RTL fixes to the worker copy.

    The source tree remains untouched. Current PLENA_RTL has an MXFP synthesis
    link mismatch in fp_cp_asym_mult: it passes IN_EXP_WIDTH_* + 1 into
    fp_asym_mult, whose exp ports already include the signed bit in their range.
    That makes the child port one bit wider than the parent signal.
    """
    path = worker_rtl / "src/basic_components/fp_operation/rtl/fp_cp_asym_mult.sv"
    if path.exists():
        text = path.read_text()
        if "p1_signed_exp_a_ext" not in text:
            text = text.replace(
            "    logic signed [IN_EXP_WIDTH_B:0]     p1_signed_exp_b;\n"
            "    logic signed [IN_FIXED_WIDTH_B - 1:0]   p1_signed_mant_b;\n",
            "    logic signed [IN_EXP_WIDTH_B:0]     p1_signed_exp_b;\n"
            "    logic signed [IN_FIXED_WIDTH_B - 1:0]   p1_signed_mant_b;\n"
            "    logic signed [IN_EXP_WIDTH_A + 1:0] p1_signed_exp_a_ext;\n"
            "    logic signed [IN_EXP_WIDTH_B + 1:0] p1_signed_exp_b_ext;\n",
            )
            text = text.replace(
            "    logic p1_mult_valid;\n",
            "    assign p1_signed_exp_a_ext = {p1_signed_exp_a[IN_EXP_WIDTH_A], p1_signed_exp_a};\n"
            "    assign p1_signed_exp_b_ext = {p1_signed_exp_b[IN_EXP_WIDTH_B], p1_signed_exp_b};\n\n"
            "    logic p1_mult_valid;\n",
            )
            text = text.replace(".exp_a      (p1_signed_exp_a),", ".exp_a      (p1_signed_exp_a_ext),")
            text = text.replace(".exp_b      (p1_signed_exp_b),", ".exp_b      (p1_signed_exp_b_ext),")
        path.write_text(text)

    path = worker_rtl / "src/basic_components/gemv/rtl/fp_adder_tree_layer.sv"
    if path.exists():
        text = path.read_text()
        text = text.replace(".IN_EXP_WIDTH(IN_EXP_WIDTH),", ".IN_EXP_WIDTH(IN_EXP_WIDTH + 1),")
        path.write_text(text)

    for path in [
        worker_rtl / "src/basic_components/systolic_gemm_mx/rtl/mxfp_default_pe.sv",
        worker_rtl / "src/basic_components/systolic_gemm_mx/rtl/mx_default_pe.sv",
    ]:
        if not path.exists():
            continue
        text = path.read_text()
        if "MULT_EXP_WIDTH" not in text:
            text = text.replace(
                "logic [MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0] block_mult_result;",
                "localparam int MULT_EXP_WIDTH = "
                "(MX_T_EXP_WIDTH > MX_L_EXP_WIDTH ? MX_T_EXP_WIDTH : MX_L_EXP_WIDTH);\n"
                "            localparam int MULT_MANT_WIDTH = "
                "(MX_T_MANT_WIDTH > MX_L_MANT_WIDTH ? MX_T_MANT_WIDTH : MX_L_MANT_WIDTH);\n"
                "            logic [MULT_EXP_WIDTH + MULT_MANT_WIDTH : 0] block_mult_result;",
            )
            text = text.replace(".MXFP_EXP_WIDTH     (MX_L_EXP_WIDTH),", ".MXFP_EXP_WIDTH     (MULT_EXP_WIDTH),")
            text = text.replace(".MXFP_MANT_WIDTH    (MX_L_MANT_WIDTH),", ".MXFP_MANT_WIDTH    (MULT_MANT_WIDTH),")
            path.write_text(text)

def cleanup_workers(worker_root: Path) -> None:
    """Remove all temporary worker trees after a run or interruption."""
    if worker_root.exists():
        shutil.rmtree(worker_root)


def _path_used_by_live_process(path: Path) -> bool:
    """Return whether one of this user's live processes references ``path``."""
    resolved = path.resolve()
    uid = os.getuid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != uid:
                continue
            links = [proc / "cwd", proc / "root"]
            links.extend((proc / "fd").iterdir())
            for link in links:
                try:
                    target = link.resolve()
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                if target == resolved or resolved in target.parents:
                    return True
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return False


def _cleanup_owned_stale_tmp() -> list[str]:
    """Remove only inactive PLENA calibration trees owned by this user."""
    removed: list[str] = []
    patterns = ("plena_rtl_area_workers*", "area_new_*")
    for pattern in patterns:
        for path in Path("/tmp").glob(pattern):
            try:
                if path.stat().st_uid != os.getuid() or _path_used_by_live_process(path):
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(str(path))
            except (FileNotFoundError, PermissionError, OSError):
                continue
    return removed


def preflight_tmp_workers(requested_workers: int) -> tuple[int, dict[str, Any]]:
    """Enforce temporary-space headroom and safely reduce concurrency.

    The check runs before worker copies are created. Cleanup is attempted only
    when the requested worker count does not fit, and never touches paths that
    are owned by another user or referenced by a live process.
    """
    if requested_workers < 1:
        raise ValueError("requested_workers must be positive")
    free_before = shutil.disk_usage("/tmp").free
    required = (TMP_FIXED_HEADROOM_GIB + TMP_PER_WORKER_GIB * requested_workers) * GIB
    removed: list[str] = []
    if free_before < required:
        removed = _cleanup_owned_stale_tmp()
    free_after = shutil.disk_usage("/tmp").free
    if free_after < TMP_HARD_MIN_GIB * GIB:
        raise RuntimeError(
            f"/tmp has only {free_after / GIB:.1f} GiB free after safe cleanup; "
            f"at least {TMP_HARD_MIN_GIB} GiB is required"
        )
    affordable = max(0, int((free_after / GIB - TMP_FIXED_HEADROOM_GIB) // TMP_PER_WORKER_GIB))
    workers = min(requested_workers, affordable)
    if workers < 1:
        raise RuntimeError(
            f"/tmp has {free_after / GIB:.1f} GiB free, insufficient for one "
            "quota-safe synthesis worker"
        )
    return workers, {
        "requested_workers": requested_workers,
        "resolved_workers": workers,
        "free_before_gib": free_before / GIB,
        "free_after_gib": free_after / GIB,
        "required_for_requested_gib": required / GIB,
        "removed_stale_paths": removed,
    }


def mxint_pe_wrapper(module: str, t_bits: int, l_bits: int, acc_depth: int) -> str:
    """Generate a concrete top wrapper for one parameterized MXINT PE."""
    out_int_width = t_bits + l_bits + max(1, (acc_depth - 1).bit_length())
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic [{t_bits - 1}:0] in_top_element,
    input logic [7:0] in_top_scale,
    input logic in_top_valid,
    input logic [{l_bits - 1}:0] in_left_element,
    input logic [7:0] in_left_scale,
    input logic in_left_valid,
    output logic [{t_bits - 1}:0] out_bottom_element,
    output logic [7:0] out_bottom_scale,
    output logic [{l_bits - 1}:0] out_right_element,
    output logic [7:0] out_right_scale,
    output logic [{out_int_width - 1}:0] out_int,
    output logic [8:0] out_scale,
    output logic out_valid
);
    mxint_default_pe #(
        .MX_T_INT_WIDTH({t_bits}),
        .MX_L_INT_WIDTH({l_bits}),
        .MXINT_SCALE_WIDTH(8),
        .ACC_DEPTH({acc_depth})
    ) dut (
        .clk(clk), .rst(rst),
        .in_top_element(in_top_element),
        .in_top_scale(in_top_scale),
        .in_top_valid(in_top_valid),
        .in_left_element(in_left_element),
        .in_left_scale(in_left_scale),
        .in_left_valid(in_left_valid),
        .out_bottom_element(out_bottom_element),
        .out_bottom_scale(out_bottom_scale),
        .out_right_element(out_right_element),
        .out_right_scale(out_right_scale),
        .out_int(out_int),
        .out_scale(out_scale),
        .out_valid(out_valid)
    );
endmodule
"""


def mxfp_pe_wrapper(module: str, t_exp: int, t_mant: int, l_exp: int, l_mant: int) -> str:
    """Generate a concrete top wrapper for one asymmetric MXFP PE."""
    t_width = mxfp_width(t_exp, t_mant)
    l_width = mxfp_width(l_exp, l_mant)
    out_width = 16
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic clear_accumulator,
    input logic [{t_width - 1}:0] in_top_element,
    input logic [7:0] in_top_scale,
    input logic system_top_valid,
    input logic [{l_width - 1}:0] in_left_element,
    input logic [7:0] in_left_scale,
    input logic system_left_valid,
    output logic [{t_width - 1}:0] out_bottom_element,
    output logic [7:0] out_bottom_scale,
    output logic [{l_width - 1}:0] out_right_element,
    output logic [7:0] out_right_scale,
    output logic [{out_width - 1}:0] out_fp,
    output logic out_result_valid
);
    mxfp_default_pe #(
        .MX_T_EXP_WIDTH({t_exp}),
        .MX_T_MANT_WIDTH({t_mant}),
        .MX_L_EXP_WIDTH({l_exp}),
        .MX_L_MANT_WIDTH({l_mant}),
        .MX_SCALE_WIDTH(8),
        .ACC_FP_EXP_WIDTH(8),
        .ACC_FP_MANT_WIDTH(7)
    ) dut (
        .clk(clk), .rst(rst), .clear_accumulator(clear_accumulator),
        .in_top_element(in_top_element), .in_top_scale(in_top_scale), .system_top_valid(system_top_valid),
        .in_left_element(in_left_element), .in_left_scale(in_left_scale), .system_left_valid(system_left_valid),
        .out_bottom_element(out_bottom_element), .out_bottom_scale(out_bottom_scale),
        .out_right_element(out_right_element), .out_right_scale(out_right_scale),
        .out_fp(out_fp), .out_result_valid(out_result_valid)
    );
endmodule
"""


def mxint_mini_wrapper(module: str, t_bits: int, l_bits: int, block_dim: int, acc_depth: int) -> str:
    """Generate a concrete MXINT mini systolic-array synthesis top."""
    out_int_width = t_bits + l_bits + max(1, (acc_depth - 1).bit_length())
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic [{acc_depth - 1}:0][{l_bits - 1}:0] load_a_row,
    input logic [{acc_depth - 1}:0][{t_bits - 1}:0] load_b_col,
    input logic [7:0] load_a_scale,
    input logic [7:0] load_b_scale,
    input logic load_valid,
    output logic [{block_dim - 1}:0][{block_dim - 1}:0][{out_int_width - 1}:0] out_int,
    output logic [{block_dim - 1}:0][{block_dim - 1}:0][8:0] out_scale,
    output logic out_valid
);
    mxint_mini_systolic_array #(
        .MX_T_INT_WIDTH({t_bits}),
        .MX_L_INT_WIDTH({l_bits}),
        .MXINT_SCALE_WIDTH(8),
        .BLOCK_DIM({block_dim}),
        .ACC_DEPTH({acc_depth})
    ) dut (
        .clk(clk), .rst(rst),
        .load_a_row(load_a_row), .load_b_col(load_b_col),
        .load_a_scale(load_a_scale), .load_b_scale(load_b_scale),
        .load_valid(load_valid),
        .out_int(out_int), .out_scale(out_scale), .out_valid(out_valid)
    );
endmodule
"""


def mxfp_mini_wrapper(module: str, t_exp: int, t_mant: int, l_exp: int, l_mant: int, block_dim: int) -> str:
    """Generate a concrete MXFP mini systolic-array synthesis top."""
    t_width = mxfp_width(t_exp, t_mant)
    l_width = mxfp_width(l_exp, l_mant)
    out_width = 16
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic clear_accumulator,
    input logic [{block_dim - 1}:0][{t_width - 1}:0] in_top_element,
    input logic [{block_dim - 1}:0][7:0] in_top_scale,
    input logic system_top_valid,
    input logic [{block_dim - 1}:0][{l_width - 1}:0] in_left_element,
    input logic [7:0] in_left_scale,
    input logic system_left_valid,
    output logic [{block_dim - 1}:0][{t_width - 1}:0] out_bottom_element,
    output logic [{block_dim - 1}:0][7:0] out_bottom_scale,
    output logic [{block_dim - 1}:0][{l_width - 1}:0] out_right_element,
    output logic [7:0] out_right_scale,
    output logic [{block_dim - 1}:0][{block_dim - 1}:0][{out_width - 1}:0] out_fp,
    output logic out_result_valid
);
    mx_mini_systolic_array #(
        .MX_T_EXP_WIDTH({t_exp}),
        .MX_T_MANT_WIDTH({t_mant}),
        .MX_L_EXP_WIDTH({l_exp}),
        .MX_L_MANT_WIDTH({l_mant}),
        .MX_SCALE_WIDTH(8),
        .BLOCK_DIM({block_dim}),
        .ACC_FP_EXP_WIDTH(8),
        .ACC_FP_MANT_WIDTH(7),
        .L_MX_INT_EN(0),
        .T_MX_INT_EN(0)
    ) dut (
        .clk(clk), .rst(rst), .clear_accumulator(clear_accumulator),
        .in_top_element(in_top_element), .in_top_scale(in_top_scale), .system_top_valid(system_top_valid),
        .in_left_element(in_left_element), .in_left_scale(in_left_scale), .system_left_valid(system_left_valid),
        .out_bottom_element(out_bottom_element), .out_bottom_scale(out_bottom_scale),
        .out_right_element(out_right_element), .out_right_scale(out_right_scale),
        .out_fp(out_fp), .out_result_valid(out_result_valid)
    );
endmodule
"""


def mxfp_mini_periphery_wrapper(module: str, block_dim: int) -> str:
    """Generate the non-PE logic of ``mx_mini_systolic_array``.

    The RTL periphery is a BLOCK_DIM-deep scale shift register plus an AND
    reduction over PE-valid signals. Data transfers between PEs are wires and
    consume no cells. The runner adds ``BLOCK_DIM**2`` mapped PE leaf areas to
    this mapped peripheral area after synthesis.
    """

    pe_count = block_dim * block_dim
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic [7:0] in_left_scale,
    input logic [{pe_count - 1}:0] pe_result_valid,
    output logic [7:0] out_right_scale,
    output logic out_result_valid
);
    logic [{block_dim}:0][7:0] first_col_scale;
    assign first_col_scale[0] = in_left_scale;
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 1; i < {block_dim + 1}; i++) begin
                first_col_scale[i] <= '0;
            end
        end else begin
            for (int i = 0; i < {block_dim}; i++) begin
                first_col_scale[i + 1] <= first_col_scale[i];
            end
        end
    end
    assign out_right_scale = first_col_scale[{block_dim}];
    assign out_result_valid = &pe_result_valid;
endmodule
"""


def mxint_acc_to_fp_wrapper(
    module: str,
    acc_width: int,
    acc_frac_width: int,
    fp_exp_width: int,
    fp_mant_width: int,
) -> str:
    """Generate a concrete integer-accumulator to FP conversion top."""
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic signed [{acc_width - 1}:0] acc_in,
    input logic [8:0] scale_in,
    output logic [{fp_exp_width + fp_mant_width}:0] fp_out
);
    mxint_acc_2_fp #(
        .ACC_WIDTH({acc_width}),
        .ACC_FRAC_WIDTH({acc_frac_width}),
        .IN_SCALE_WIDTH(9),
        .MXINT_SCALE_WIDTH(8),
        .FP_EXP_WIDTH({fp_exp_width}),
        .FP_MANT_WIDTH({fp_mant_width})
    ) dut (
        .acc_in(acc_in), .scale_in(scale_in), .fp_out(fp_out)
    );
endmodule
"""


def mxint_reduce_wrapper(
    module: str,
    int_width: int,
    compute_dim: int,
    splits: int,
) -> str:
    """Generate one concrete MXINT cross-K reduction leaf."""
    out_width = int_width + 16 + max(1, splits.bit_length())
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic [{splits - 1}:0][{compute_dim - 1}:0][{compute_dim - 1}:0][{int_width - 1}:0] m_in_int,
    input logic [{splits - 1}:0][{compute_dim - 1}:0][{compute_dim - 1}:0][8:0] m_in_scale,
    input logic in_valid,
    output logic [{compute_dim - 1}:0][{compute_dim - 1}:0][{out_width - 1}:0] m_out_int,
    output logic [{compute_dim - 1}:0][{compute_dim - 1}:0][8:0] m_out_scale,
    output logic out_valid
);
    mxint_sum_across #(
        .INT_WIDTH({int_width}), .SCALE_WIDTH(9),
        .COMPUTE_DIM({compute_dim}), .SYS_ARRAY_AMOUNT({splits}), .MAX_SHIFT(16)
    ) dut (
        .clk(clk), .rst(rst), .m_in_int(m_in_int), .m_in_scale(m_in_scale),
        .in_valid(in_valid), .m_out_int(m_out_int), .m_out_scale(m_out_scale),
        .out_valid(out_valid)
    );
endmodule
"""


def mxfp_reduce_wrapper(
    module: str,
    fp_exp_width: int,
    fp_mant_width: int,
    compute_dim: int,
    splits: int,
) -> str:
    """Generate one concrete MXFP cross-K reduction leaf."""
    fp_width = 1 + fp_exp_width + fp_mant_width
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic [{splits - 1}:0][{compute_dim - 1}:0][{compute_dim - 1}:0][{fp_width - 1}:0] m_in_data,
    input logic in_valid,
    output logic [{compute_dim - 1}:0][{compute_dim - 1}:0][{fp_width - 1}:0] m_out_data,
    output logic out_valid
);
    mx_sum_across_sa #(
        .ACC_FP_MANT_WIDTH({fp_mant_width}), .ACC_FP_EXP_WIDTH({fp_exp_width}),
        .COMPUTE_DIM({compute_dim}), .SYS_ARRAY_AMOUNT({splits})
    ) dut (
        .clk(clk), .rst(rst), .m_in_data(m_in_data), .in_valid(in_valid),
        .m_out_data(m_out_data), .out_valid(out_valid)
    );
endmodule
"""


def wrapper_text(point: Point) -> str:
    """Dispatch wrapper generation according to hierarchy level and family."""
    p = point.params
    if point.level == "pe" and point.mode == "mxint":
        return mxint_pe_wrapper(point.top_module, int(p["T_BITS"]), int(p["L_BITS"]), int(p["ACC_DEPTH"]))
    if point.level == "pe" and point.mode == "mxfp":
        return mxfp_pe_wrapper(
            point.top_module,
            int(p["T_EXP"]),
            int(p["T_MANT"]),
            int(p["L_EXP"]),
            int(p["L_MANT"]),
        )
    if point.level == "mini_array" and point.mode == "mxint":
        return mxint_mini_wrapper(
            point.top_module,
            int(p["T_BITS"]),
            int(p["L_BITS"]),
            int(p["BLOCK_DIM"]),
            int(p["ACC_DEPTH"]),
        )
    if point.level == "mini_array" and point.mode == "mxfp":
        if int(p.get("STRUCTURAL_COMPOSITE", 0)):
            return mxfp_mini_periphery_wrapper(
                point.top_module,
                int(p["BLOCK_DIM"]),
            )
        return mxfp_mini_wrapper(
            point.top_module,
            int(p["T_EXP"]),
            int(p["T_MANT"]),
            int(p["L_EXP"]),
            int(p["L_MANT"]),
            int(p["BLOCK_DIM"]),
        )
    if point.level == "output_conversion" and point.mode == "mxint":
        return mxint_acc_to_fp_wrapper(
            point.top_module,
            int(p["ACC_WIDTH"]),
            int(p["ACC_FRAC_WIDTH"]),
            int(p["FP_EXP_WIDTH"]),
            int(p["FP_MANT_WIDTH"]),
        )
    if point.level == "reduce_leaf" and point.mode == "mxint":
        return mxint_reduce_wrapper(
            point.top_module,
            int(p["ACC_WIDTH"]),
            int(p["COMPUTE_DIM"]),
            int(p["SYS_ARRAY_AMOUNT"]),
        )
    if point.level == "reduce_leaf" and point.mode == "mxfp":
        return mxfp_reduce_wrapper(
            point.top_module,
            int(p["FP_EXP_WIDTH"]),
            int(p["FP_MANT_WIDTH"]),
            int(p["COMPUTE_DIM"]),
            int(p["SYS_ARRAY_AMOUNT"]),
        )
    raise ValueError(f"no wrapper for {point.level}/{point.mode}")


def write_wrapper(point: Point, rtl_root: Path) -> Path | None:
    """Materialize a wrapper, or return ``None`` for the native Matrix top."""
    if point.level == "matrix_machine":
        return None
    if point.mode == "mxint":
        target_dir = rtl_root / "src/basic_components/systolic_gemm_mxint/rtl"
    else:
        target_dir = rtl_root / "src/basic_components/systolic_gemm_mx/rtl"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{point.top_module}.sv"
    path.write_text(wrapper_text(point))
    return path


def replace_localparam(path: Path, key: str, value: int) -> None:
    """Replace exactly one SystemVerilog integer localparam or fail loudly."""
    text = path.read_text()
    pattern = re.compile(rf"(localparam\s+{re.escape(key)}\s*=\s*)([^;]+)(;)")
    new, count = pattern.subn(rf"\g<1>{value}\3", text, count=1)
    if count == 0:
        raise KeyError(f"localparam {key} not found in {path}")
    path.write_text(new)


def patch_matrix_machine_config(point: Point, rtl_root: Path) -> None:
    """Patch MatrixMachine dimensions and precision into one worker copy."""
    p = point.params
    precision = rtl_root / "src/definitions/precision.svh"
    configuration = rtl_root / "src/definitions/configuration.svh"
    mlen = int(p["MLEN"])
    blen = int(p["BLEN"])
    replace_localparam(configuration, "MLEN", mlen)
    replace_localparam(configuration, "VLEN", mlen)
    replace_localparam(configuration, "BLEN", blen)
    replace_localparam(configuration, "MATRIX_SRAM_DEPTH", max(2 * mlen, 32))
    replace_localparam(configuration, "VECTOR_SRAM_DEPTH", 128)
    replace_localparam(precision, "BLOCK_DIM", blen)
    if point.mode == "mxint":
        replace_localparam(precision, "ACT_MX_INT_ENABLE", 1)
        replace_localparam(precision, "KV_MX_INT_ENABLE", 1)
        replace_localparam(precision, "WT_MX_INT_ENABLE", 1)
        replace_localparam(precision, "ACT_MX_INT_WIDTH", int(p["L_BITS"]))
        replace_localparam(precision, "KV_MX_INT_WIDTH", int(p["T_BITS"]))
        replace_localparam(precision, "WT_MX_INT_WIDTH", int(p["T_BITS"]))
    else:
        replace_localparam(precision, "ACT_MX_INT_ENABLE", 0)
        replace_localparam(precision, "KV_MX_INT_ENABLE", 0)
        replace_localparam(precision, "WT_MX_INT_ENABLE", 0)
        replace_localparam(precision, "ACT_MXFP_EXP_WIDTH", int(p["L_EXP"]))
        replace_localparam(precision, "ACT_MXFP_MANT_WIDTH", int(p["L_MANT"]))
        replace_localparam(precision, "KV_MX_EXP_WIDTH", int(p["T_EXP"]))
        replace_localparam(precision, "KV_MX_MANT_WIDTH", int(p["T_MANT"]))
        replace_localparam(precision, "WT_MX_EXP_WIDTH", int(p["T_EXP"]))
        replace_localparam(precision, "WT_MX_MANT_WIDTH", int(p["T_MANT"]))


def parse_area(report: Path) -> float:
    """Parse ``Total cell area`` in um^2 from a DC report."""
    if not report.exists():
        raise FileNotFoundError(report)
    match = re.search(r"Total cell area:\s*([0-9.]+)", report.read_text(errors="ignore"))
    if not match:
        raise ValueError(f"Total cell area not found in {report}")
    return float(match.group(1))


def parse_area_from_text(text: str) -> float | None:
    """Recover total cell area from command output if a report is missing."""
    match = re.search(r"Total cell area:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_hierarchy_area(report: Path) -> dict[str, float]:
    """Extract stable MatrixMachine hierarchy buckets from a DC area report.

    Only direct hierarchy nodes are counted. Descendants are deliberately
    excluded because every parent row already contains its complete subtree.
    This makes the buckets mutually exclusive and suitable as supervised
    fitting targets.
    """
    if not report.exists():
        return {}
    rows: list[tuple[str, float]] = []
    pending_name: str | None = None
    numeric_line = re.compile(r"^\s+([0-9.]+)\s+([0-9.]+)\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+")
    inline_line = re.compile(
        r"^(.{1,34}?)(\s+[0-9.]+\s+[0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s+\S+)"
    )
    for line in report.read_text(errors="ignore").splitlines():
        if (
            not line.strip()
            or line.startswith("-")
            or line.startswith("Hierarchical")
            or line.startswith("Global")
            or line.startswith("Total")
            or line.startswith("Design")
        ):
            continue
        inline = inline_line.match(line)
        if inline and inline.group(1).strip():
            match = numeric_line.match(inline.group(2))
            if match:
                rows.append((inline.group(1).strip(), float(match.group(1))))
                pending_name = None
                continue
        match = numeric_line.match(line)
        if match and pending_name:
            rows.append((pending_name, float(match.group(1))))
            pending_name = None
            continue
        if not re.search(r"[0-9]+\.[0-9]+", line):
            pending_name = line.strip()

    total = 0.0
    compute = 0.0
    arrays = 0.0
    reduce = 0.0
    output_accumulator = 0.0
    output_conversion = 0.0
    result_buffer = 0.0
    io_pipeline = 0.0
    for name, area in rows:
        if name == "matrix_machine":
            total = area
        elif "systolic_mcu_matrix_compute_unit" in name and "/" not in name:
            compute = area
        elif re.fullmatch(
            r"gen_mxint_systolic_mcu_matrix_compute_unit/g_mini_\d+__mini",
            name,
        ):
            arrays += area
        elif re.fullmatch(
            r"gen_mx_systolic_mcu_matrix_compute_unit/genblk1_\d+__"
            r"(?:systolic_array_inst|left_streamer|top_streamer)",
            name,
        ):
            arrays += area
        elif (name.endswith("/cross_k_reduce") or name.endswith("/sa_sum_across_inst")) and name.count("/") == 1:
            reduce += area
        elif re.search(r"/g_acc_row_\d+__g_acc_col_\d+__acc$", name):
            output_accumulator += area
        elif re.search(r"/g_fp_row_\d+__g_fp_col_\d+__acc_to_fp$", name):
            output_conversion += area
        elif re.fullmatch(
            r"gen_mx_systolic_mcu_matrix_compute_unit/"
            r"genblk2_gen_quantize_\d+__cast_inst",
            name,
        ):
            output_conversion += area
        elif name in {
            "gen_mx_systolic_mcu_matrix_compute_unit/hold_and_unroll_for_gemm",
            "gen_mx_systolic_mcu_matrix_compute_unit/quantized_result_buffer",
            "gen_mx_systolic_mcu_matrix_compute_unit/result_buffer",
            "result_buffer",
        }:
            result_buffer += area
        elif name in {
            "matrix_element_buffer",
            "matrix_scale_buffer",
            "vector_element_buffer",
            "vector_scale_buffer",
        }:
            io_pipeline += area
    classified = (
        arrays
        + reduce
        + output_accumulator
        + output_conversion
        + result_buffer
        + io_pipeline
    )
    control = max(0.0, total - classified) if total else 0.0
    return {
        "hier_total_area": total,
        "hier_compute_unit_area": compute,
        "hier_array_area": arrays,
        "hier_reduce_area": reduce,
        "hier_output_accumulator_area": output_accumulator,
        "hier_output_conversion_area": output_conversion,
        "hier_result_buffer_area": result_buffer,
        "hier_io_pipeline_area": io_pipeline,
        "hier_control_area": control,
        # Legacy aliases remain populated so historical diagnostics continue
        # to work while structural-v4 consumes the precise fields above.
        "hier_accum_area": output_accumulator,
        "hier_top_glue_area": max(0.0, total - compute) if total and compute else 0.0,
    }


def backfill_hierarchy_fields(csv_path: Path) -> None:
    """Parse retained reports to enrich historical MatrixMachine CSV rows."""
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    changed = False
    for row in rows:
        if row.get("status") != "complete" or row.get("level") != "matrix_machine":
            continue
        report_dir = row.get("report_dir")
        if not report_dir:
            continue
        report = Path(report_dir) / "area.rpt"
        if not report.exists():
            report = ROOT / report
        hierarchy = parse_hierarchy_area(report)
        if not hierarchy:
            continue
        for key, value in hierarchy.items():
            row[key] = value
        changed = True
    if changed:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_power(report: Path) -> dict[str, float | None]:
    """Parse optional dynamic, leakage, and total power report values."""
    if not report.exists():
        return {"dynamic_power": None, "leakage_power": None, "total_power": None}
    text = report.read_text(errors="ignore")

    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "dynamic_power": grab(r"Total Dynamic Power\s*=\s*([0-9.eE+-]+)"),
        "leakage_power": grab(r"Cell Leakage Power\s*=\s*([0-9.eE+-]+)"),
        "total_power": grab(r"Total Power\s*=\s*([0-9.eE+-]+)"),
    }


def json_safe(value: Any) -> Any:
    """Convert NaN/Inf and nested values into strict JSON-compatible data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def copy_reports(worker_rtl: Path, point: Point, run_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """Retain compact area/power/QoR/reference reports before cleanup."""
    latest = worker_rtl / f"build/synth/{point.top_module}/latest"
    dest = run_dir / "reports" / point.point_key
    dest.mkdir(parents=True, exist_ok=True)
    area_src = latest / f"reports/{point.top_module}_area.rpt"
    power_src = latest / f"reports/{point.top_module}_power.rpt"
    reference_src = latest / f"reports/{point.top_module}_reference.rpt"
    qor_src = latest / f"reports/{point.top_module}_qor.rpt"
    summary_src = latest / "logs/summary.log"
    area_dst = dest / "area.rpt"
    power_dst = dest / "power.rpt"
    summary_dst = dest / "summary.log"
    if area_src.exists():
        shutil.copy2(area_src, area_dst)
    if power_src.exists():
        shutil.copy2(power_src, power_dst)
    if reference_src.exists():
        shutil.copy2(reference_src, dest / "reference.rpt")
    if qor_src.exists():
        shutil.copy2(qor_src, dest / "qor.rpt")
    if summary_src.exists():
        shutil.copy2(summary_src, summary_dst)
    return area_dst, power_dst if power_dst.exists() else None, summary_dst if summary_dst.exists() else None


def summarize_synth_failure(result: subprocess.CompletedProcess[str]) -> str:
    """Extract a concise CSV-safe reason from failed DC stdout/stderr."""
    text = f"{result.stdout}\n{result.stderr}"
    if is_dc_license_unavailable_text(text):
        return "design compiler license unavailable (SEC-50)"
    if "No justfile found" in text:
        return "worker copy missing justfile; possible shared worker-root cleanup"
    return f"synth failed with exit code {result.returncode}"


def cleanup_worker_build(worker_rtl: Path, point: Point) -> None:
    """Delete point-local synthesis output after reports are retained."""
    path = worker_rtl / f"build/synth/{point.top_module}"
    if path.exists():
        shutil.rmtree(path)


def run_point(
    point: Point,
    worker_id: int,
    worker_rtl: Path,
    rtl_root: Path,
    run_dir: Path,
    cleanup_builds: bool,
    license_retry_wait_sec: float,
    license_max_retries: int,
) -> dict[str, Any]:
    """Execute one point with license retry, report archival, and cleanup."""
    start = time.time()
    try:
        if point.level == "matrix_machine":
            patch_matrix_machine_config(point, worker_rtl)
        else:
            write_wrapper(point, worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth {point.top_module} 1000 area"
        cmd = ["nix", "develop", "-c", "bash", "-lc", synth_cmd]
        log_dir = run_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            attempt += 1
            result = subprocess.run(cmd, cwd=rtl_root, text=True, capture_output=True, check=False)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stderr.log").write_text(result.stderr)
            (log_dir / f"{point.point_key}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.stderr.log").write_text(result.stderr)

            if result.returncode == 0:
                break

            text = f"{result.stdout}\n{result.stderr}"
            if not is_dc_license_unavailable_text(text):
                break
            if license_max_retries > 0 and attempt > license_max_retries:
                break
            print(
                f"[license-busy] worker={worker_id} point={point.point_id} "
                f"attempt={attempt}; retrying in {license_retry_wait_sec:g}s",
                flush=True,
            )
            if cleanup_builds:
                cleanup_worker_build(worker_rtl, point)
            time.sleep(license_retry_wait_sec)
        if result.returncode != 0:
            stdout_area = parse_area_from_text(result.stdout)
            if stdout_area is not None and "Status: SUCCESS" in result.stdout:
                return point_to_row(
                    point,
                    status="complete",
                    worker_id=worker_id,
                    elapsed_sec=round(time.time() - start, 3),
                    area_um2=stdout_area,
                    failure_reason="synth reported success but report copy/summary failed",
                )
            return point_to_row(
                point,
                status="failed",
                worker_id=worker_id,
                elapsed_sec=round(time.time() - start, 3),
                failure_reason=summarize_synth_failure(result),
            )
        area_report, power_report, summary_log = copy_reports(worker_rtl, point, run_dir)
        raw_area = parse_area(area_report)
        area = raw_area
        if int(point.params.get("STRUCTURAL_COMPOSITE", 0)):
            block_dim = int(point.params["BLOCK_DIM"])
            area += block_dim * block_dim * float(point.params["PE_AREA_UM2"])
        power = parse_power(power_report) if power_report else parse_power(Path("__missing__"))
        row = point_to_row(
            point,
            status="complete",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            area_um2=area,
            dynamic_power=power["dynamic_power"],
            leakage_power=power["leakage_power"],
            total_power=power["total_power"],
            report_dir=str(area_report.parent),
            summary_log=str(summary_log or ""),
        )
        row["raw_synth_area_um2"] = raw_area
        return row
    except Exception as exc:  # noqa: BLE001 - calibration should record failures and continue
        return point_to_row(
            point,
            status="failed",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            failure_reason=repr(exc),
        )
    finally:
        if cleanup_builds:
            cleanup_worker_build(worker_rtl, point)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    aug = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for idx in range(col, n + 1):
            aug[col][idx] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for idx in range(col, n + 1):
                aug[row][idx] -= factor * aug[col][idx]
    return [aug[row][n] for row in range(n)]


def _least_squares(features: list[list[float]], targets: list[float]) -> list[float] | None:
    if not features:
        return None
    cols = len(features[0])
    normal = [
        [sum(row[left] * row[right] for row in features) for right in range(cols)]
        for left in range(cols)
    ]
    rhs = [sum(row[col] * target for row, target in zip(features, targets)) for col in range(cols)]
    return _solve_linear_system(normal, rhs)


def _fit_nonnegative(features: list[list[float]], targets: list[float]) -> list[float]:
    """Tiny active-set NNLS by enumerating feature subsets.

    The calibration formulas have at most four features, so exhaustive subset
    search is simpler than adding a scipy dependency and keeps coefficients
    physically monotonic for DSE extrapolation.
    """
    if not features:
        return []
    cols = len(features[0])
    best: tuple[float, list[float]] | None = None
    for mask in range(1, 1 << cols):
        active = [idx for idx in range(cols) if (mask >> idx) & 1]
        sub_features = [[row[idx] for idx in active] for row in features]
        sub_coeffs = _least_squares(sub_features, targets)
        if sub_coeffs is None or any(coeff < -1e-9 for coeff in sub_coeffs):
            continue
        coeffs = [0.0] * cols
        for idx, coeff in zip(active, sub_coeffs):
            coeffs[idx] = max(0.0, coeff)
        rss = sum((sum(row[idx] * coeffs[idx] for idx in range(cols)) - target) ** 2 for row, target in zip(features, targets))
        if best is None or rss < best[0]:
            best = (rss, coeffs)
    return best[1] if best else [0.0] * cols


def fit_nonnegative(
    rows: list[dict[str, Any]],
    features: list[str],
    target: str = "area_um2",
) -> tuple[list[float], float]:
    """Fit nonnegative coefficients and return them with training MAPE."""
    if not rows:
        return [1.0] * len(features), float("nan")
    x = [[float(row[name]) for name in features] for row in rows]
    y = [float(row[target]) for row in rows]
    coeffs = _fit_nonnegative(x, y)
    pred = [sum(row[idx] * coeffs[idx] for idx in range(len(coeffs))) for row in x]
    mape = sum(abs((item - target_value) / max(abs(target_value), 1e-9)) for item, target_value in zip(pred, y)) / len(y) * 100
    return [float(c) for c in coeffs], float(mape)


def matrix_total_mape(rows: list[dict[str, Any]], beta_width: float, gamma_mlen: float) -> float:
    """Evaluate the legacy MatrixMachine total-area residual equation."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        predicted = (
            float(row["matrix_base_area"])
            + beta_width * float(row["feat_width"])
            + gamma_mlen * float(row["feat_mlen"])
        )
        errors.append(abs((predicted - actual) / max(abs(actual), 1e-9)))
    return float(sum(errors) / len(errors) * 100.0)


def matrix_residual_mape(rows: list[dict[str, Any]], reduce_c: float, accum_c: float, top_c: float) -> float:
    """Evaluate fixed mini-stack plus fitted residual total-area MAPE."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        predicted = (
            float(row["matrix_base_area"])
            + reduce_c * float(row["feat_reduce"])
            + accum_c * float(row["feat_accum"])
            + top_c
        )
        errors.append(abs((predicted - actual) / max(abs(actual), 1e-9)))
    return float(sum(errors) / len(errors) * 100.0)


def matrix_hierarchy_mape(rows: list[dict[str, Any]], stack_c: float, reduce_c: float, accum_c: float, top_c: float) -> float:
    """Evaluate composed hierarchy-supervised MatrixMachine total MAPE."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        predicted = (
            stack_c * float(row["matrix_base_area"])
            + reduce_c * float(row["feat_reduce"])
            + accum_c * float(row["feat_accum"])
            + top_c
        )
        errors.append(abs((predicted - actual) / max(abs(actual), 1e-9)))
    return float(sum(errors) / len(errors) * 100.0)


MXINT_DIRECT_FEATURES = [
    "feat_direct_tl",
    "feat_direct_sum",
    "feat_direct_scale",
    "feat_direct_tile",
    "feat_direct_b2w",
    "feat_direct_b2",
    "feat_direct_mw",
    "feat_direct_m",
    "feat_const",
]


def matrix_direct_mape(rows: list[dict[str, Any]], coeffs: list[float]) -> float:
    """Evaluate the direct precision/shape feature equation."""
    if not rows:
        return float("nan")
    errors = []
    for row in rows:
        actual = float(row["area_um2"])
        predicted = sum(float(row[name]) * coeff for name, coeff in zip(MXINT_DIRECT_FEATURES, coeffs))
        errors.append(abs((predicted - actual) / max(abs(actual), 1e-9)))
    return float(sum(errors) / len(errors) * 100.0)


def hierarchy_supervised_matrix_coeffs(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, float]]:
    """Fit hierarchy buckets and select against a fixed-stack residual model.

    Submodule targets improve interpretability, but the residual candidate is
    selected when it predicts total area more accurately on available anchors.
    """
    residual_vals, _ = fit_nonnegative(rows, ["feat_reduce", "feat_accum", "feat_const"], target="matrix_residual_area")
    residual_candidate = [1.0, residual_vals[0], residual_vals[1], residual_vals[2]]
    residual_mape = matrix_hierarchy_mape(rows, *residual_candidate)
    with_hierarchy = [
        row
        for row in rows
        if row.get("hier_reduce_area") not in {"", None}
        and row.get("hier_accum_area") not in {"", None}
        and row.get("hier_top_glue_area") not in {"", None}
    ]
    if not with_hierarchy:
        return residual_candidate, {
            "hier_reduce_mape_pct": float("nan"),
            "hier_accum_mape_pct": float("nan"),
            "hier_top_mape_pct": float("nan"),
            "matrix_residual_fixed_stack_mape_pct": residual_mape,
            "matrix_fit_target": "total_residual_fallback",
        }

    for row in with_hierarchy:
        compute = float(row.get("hier_compute_unit_area") or 0.0)
        reduce_area = float(row.get("hier_reduce_area") or 0.0)
        accum_area = float(row.get("hier_accum_area") or 0.0)
        row["hier_stack_area"] = max(0.0, compute - reduce_area - accum_area)
    stack_vals, stack_mape = fit_nonnegative(with_hierarchy, ["matrix_base_area"], target="hier_stack_area")
    reduce_vals, reduce_mape = fit_nonnegative(with_hierarchy, ["feat_reduce"], target="hier_reduce_area")
    accum_vals, accum_mape = fit_nonnegative(with_hierarchy, ["feat_accum"], target="hier_accum_area")
    top_vals, top_mape = fit_nonnegative(with_hierarchy, ["feat_const"], target="hier_top_glue_area")
    vals = [stack_vals[0], reduce_vals[0], accum_vals[0], top_vals[0]]
    supervised_mape = matrix_hierarchy_mape(rows, *vals)
    selected = vals
    selected_target = "hierarchy_submodules"
    if residual_mape < supervised_mape:
        selected = residual_candidate
        selected_target = "total_residual_fixed_stack_selected"
    return selected, {
        "hier_stack_mape_pct": stack_mape,
        "hier_reduce_mape_pct": reduce_mape,
        "hier_accum_mape_pct": accum_mape,
        "hier_top_mape_pct": top_mape,
        "matrix_hierarchy_supervised_mape_pct": supervised_mape,
        "matrix_residual_fixed_stack_mape_pct": residual_mape,
        "matrix_fit_target": selected_target,
    }


def add_mxint_direct_features(row: dict[str, Any]) -> None:
    """Attach precision-grid, BLEN, MLEN, scale, and control features in place."""
    mlen = float(row["MLEN"])
    blen = float(row["BLEN"])
    t_bits = float(row["T_BITS"])
    l_bits = float(row["L_BITS"])
    scale_width = float(row.get("scale_width") or 8)
    width = t_bits + l_bits + scale_width
    row["feat_direct_tl"] = mlen * blen * t_bits * l_bits
    row["feat_direct_sum"] = mlen * blen * (t_bits + l_bits)
    row["feat_direct_scale"] = mlen * blen * scale_width
    row["feat_direct_tile"] = mlen * blen
    row["feat_direct_b2w"] = blen * blen * width
    row["feat_direct_b2"] = blen * blen
    row["feat_direct_mw"] = mlen * width
    row["feat_direct_m"] = mlen
    row["feat_const"] = 1.0


def write_matrix_diagnostics(run_dir: Path, mode: str, rows: list[dict[str, Any]], coeffs: dict[str, float]) -> None:
    """Write per-point total and hierarchy residual diagnostics."""
    if not rows:
        return
    path = run_dir / f"matrix_machine_{mode}_diagnostics.csv"
    fields = [
        "point_id",
        "actual_area_um2",
        "predicted_area_um2",
        "error_pct",
        "base_stack_area",
        "fitted_stack_area",
        "fitted_reduce_tree_area",
        "fitted_accumulator_grid_area",
        "fitted_top_glue_area",
        "hier_reduce_area",
        "hier_accum_area",
        "hier_top_glue_area",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            actual = float(row["area_um2"])
            base = float(row.get("matrix_base_area") or 0.0)
            if "mm_direct_tl" in coeffs:
                add_mxint_direct_features(row)
                stack_area = (
                    coeffs.get("mm_direct_tl", 0.0) * float(row["feat_direct_tl"])
                    + coeffs.get("mm_direct_sum", 0.0) * float(row["feat_direct_sum"])
                )
                reduce_area = coeffs.get("mm_direct_scale", 0.0) * float(row["feat_direct_scale"])
                accum_area = (
                    coeffs.get("mm_direct_tile", 0.0) * float(row["feat_direct_tile"])
                    + coeffs.get("mm_direct_b2w", 0.0) * float(row["feat_direct_b2w"])
                    + coeffs.get("mm_direct_b2", 0.0) * float(row["feat_direct_b2"])
                    + coeffs.get("mm_direct_mw", 0.0) * float(row["feat_direct_mw"])
                    + coeffs.get("mm_direct_m", 0.0) * float(row["feat_direct_m"])
                )
                top_area = coeffs.get("mm_direct_const", 0.0)
            else:
                stack_area = coeffs.get("mm_stack_c", 1.0) * base
                reduce_area = coeffs["mm_reduce_c"] * float(row["feat_reduce"])
                accum_area = coeffs.get("mm_accum_c", 0.0) * float(row["feat_accum"])
                top_area = coeffs.get("mm_top_c", 0.0)
            predicted = stack_area + reduce_area + accum_area + top_area
            writer.writerow(
                {
                    "point_id": row["point_id"],
                    "actual_area_um2": actual,
                    "predicted_area_um2": predicted,
                    "error_pct": (predicted - actual) / max(abs(actual), 1e-9) * 100.0,
                    "base_stack_area": base,
                    "fitted_stack_area": stack_area,
                    "fitted_reduce_tree_area": reduce_area,
                    "fitted_accumulator_grid_area": accum_area,
                    "fitted_top_glue_area": top_area,
                    "hier_reduce_area": row.get("hier_reduce_area", ""),
                    "hier_accum_area": row.get("hier_accum_area", ""),
                    "hier_top_glue_area": row.get("hier_top_glue_area", ""),
                }
            )


def load_completed_rows(csv_path: Path, mode: str, level: str) -> list[dict[str, Any]]:
    """Load successful rows for one numeric family and hierarchy level."""
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete" and row.get("mode") == mode and row.get("level") == level:
                rows.append(row)
    return rows


def fit_and_write_coefficients(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Fit PE, mini-array, and MatrixMachine equations in hierarchy order."""
    backfill_hierarchy_fields(csv_path)
    for mode in ["mxint", "mxfp"]:
        pe_rows = load_completed_rows(csv_path, mode, "pe")
        mini_rows = load_completed_rows(csv_path, mode, "mini_array")
        mm_rows = load_completed_rows(csv_path, mode, "matrix_machine")
        coeffs: dict[str, float] = {}
        metadata: dict[str, Any] = {
            "source_csv": str(csv_path),
            "mode": mode,
            "pe_rows": len(pe_rows),
            "mini_array_rows": len(mini_rows),
            "matrix_machine_rows": len(mm_rows),
        }
        if mode == "mxint":
            for row in pe_rows:
                row["feat_1"] = 1.0
                row["feat_tl"] = float(row["T_BITS"]) * float(row["L_BITS"])
                row["feat_sum"] = float(row["T_BITS"]) + float(row["L_BITS"])
            vals, pe_mape = fit_nonnegative(pe_rows, ["feat_1", "feat_tl", "feat_sum"])
            coeffs.update({"pe_c0": vals[0], "pe_c_tl": vals[1], "pe_c_sum": vals[2]})
            metadata["pe_mape_pct"] = pe_mape
            for row in mini_rows:
                b = float(row["BLOCK_DIM"])
                pe = coeffs["pe_c0"] + coeffs["pe_c_tl"] * float(row["T_BITS"]) * float(row["L_BITS"]) + coeffs[
                    "pe_c_sum"
                ] * (float(row["T_BITS"]) + float(row["L_BITS"]))
                row["feat_pe_grid"] = b * b * pe
                row["feat_scale"] = b * float(row.get("scale_width") or 8)
                row["feat_grid"] = b * b
                row["feat_1"] = 1.0
            vals, mini_mape = fit_nonnegative(mini_rows, ["feat_pe_grid", "feat_scale", "feat_grid", "feat_1"])
            coeffs.update({"mini_pe_scale": vals[0], "mini_a_scale": vals[1], "mini_a_grid": vals[2], "mini_a0": vals[3]})
            metadata["mini_mape_pct"] = mini_mape
            for row in mm_rows:
                mlen = float(row["MLEN"])
                blen = float(row["BLEN"])
                t = float(row["T_BITS"])
                l = float(row["L_BITS"])
                scale = float(row.get("scale_width") or 8)
                pe = coeffs["pe_c0"] + coeffs["pe_c_tl"] * t * l + coeffs["pe_c_sum"] * (t + l)
                mini = coeffs["mini_pe_scale"] * blen * blen * pe + coeffs["mini_a_scale"] * blen * scale + coeffs[
                    "mini_a_grid"
                ] * blen * blen + coeffs["mini_a0"]
                width = t + l + scale
                row["matrix_base_area"] = (mlen / blen) * mini
                row["matrix_residual_area"] = max(0.0, float(row["area_um2"]) - row["matrix_base_area"])
                row["feat_reduce"] = mlen * blen * width
                row["feat_accum"] = blen * blen
                row["feat_const"] = 1.0
            vals, hierarchy_metadata = hierarchy_supervised_matrix_coeffs(mm_rows)
            hierarchy_mape = matrix_hierarchy_mape(mm_rows, vals[0], vals[1], vals[2], vals[3])
            for row in mm_rows:
                add_mxint_direct_features(row)
            direct_vals, direct_mape = fit_nonnegative(mm_rows, MXINT_DIRECT_FEATURES)
            metadata["matrix_hierarchy_candidate_mape_pct"] = hierarchy_mape
            metadata["matrix_direct_width_candidate_mape_pct"] = direct_mape
            if direct_mape < hierarchy_mape:
                coeffs.update(
                    {
                        "mm_direct_tl": direct_vals[0],
                        "mm_direct_sum": direct_vals[1],
                        "mm_direct_scale": direct_vals[2],
                        "mm_direct_tile": direct_vals[3],
                        "mm_direct_b2w": direct_vals[4],
                        "mm_direct_b2": direct_vals[5],
                        "mm_direct_mw": direct_vals[6],
                        "mm_direct_m": direct_vals[7],
                        "mm_direct_const": direct_vals[8],
                    }
                )
                mm_mape = direct_mape
                metadata["model_version"] = "matrix_machine_direct_width_v3"
                metadata["matrix_fit_target"] = "total_area_direct_width_selected"
                metadata.update(hierarchy_metadata)
            else:
                coeffs.update(
                    {
                        "mm_stack_c": vals[0],
                        "mm_reduce_c": vals[1],
                        "mm_accum_c": vals[2],
                        "mm_top_c": vals[3],
                    }
                )
                mm_mape = hierarchy_mape
                metadata["model_version"] = "matrix_machine_hierarchy_residual_v2"
                metadata.update(hierarchy_metadata)
            metadata["matrix_machine_mape_pct"] = mm_mape
            write_matrix_diagnostics(run_dir, mode, mm_rows, coeffs)
        else:
            for row in pe_rows:
                tw = 1 + float(row["T_EXP"]) + float(row["T_MANT"])
                lw = 1 + float(row["L_EXP"]) + float(row["L_MANT"])
                row["feat_1"] = 1.0
                row["feat_tl"] = tw * lw
                row["feat_sum"] = tw + lw
                row["feat_exp"] = float(row["T_EXP"]) + float(row["L_EXP"])
            vals, pe_mape = fit_nonnegative(pe_rows, ["feat_1", "feat_tl", "feat_sum", "feat_exp"])
            coeffs.update({"pe_c0": vals[0], "pe_c_tl": vals[1], "pe_c_sum": vals[2], "pe_c_exp": vals[3]})
            metadata["pe_mape_pct"] = pe_mape
            for row in mini_rows:
                tw = 1 + float(row["T_EXP"]) + float(row["T_MANT"])
                lw = 1 + float(row["L_EXP"]) + float(row["L_MANT"])
                b = float(row["BLOCK_DIM"])
                pe = coeffs["pe_c0"] + coeffs["pe_c_tl"] * tw * lw + coeffs["pe_c_sum"] * (tw + lw) + coeffs[
                    "pe_c_exp"
                ] * (float(row["T_EXP"]) + float(row["L_EXP"]))
                row["feat_pe_grid"] = b * b * pe
                row["feat_scale"] = b * float(row.get("scale_width") or 8)
                row["feat_grid"] = b * b
                row["feat_1"] = 1.0
            vals, mini_mape = fit_nonnegative(mini_rows, ["feat_pe_grid", "feat_scale", "feat_grid", "feat_1"])
            coeffs.update({"mini_pe_scale": vals[0], "mini_a_scale": vals[1], "mini_a_grid": vals[2], "mini_a0": vals[3]})
            metadata["mini_mape_pct"] = mini_mape
            for row in mm_rows:
                mlen = float(row["MLEN"])
                blen = float(row["BLEN"])
                tw = 1 + float(row["T_EXP"]) + float(row["T_MANT"])
                lw = 1 + float(row["L_EXP"]) + float(row["L_MANT"])
                scale = float(row.get("scale_width") or 8)
                pe = coeffs["pe_c0"] + coeffs["pe_c_tl"] * tw * lw + coeffs["pe_c_sum"] * (tw + lw) + coeffs[
                    "pe_c_exp"
                ] * (float(row["T_EXP"]) + float(row["L_EXP"]))
                mini = coeffs["mini_pe_scale"] * blen * blen * pe + coeffs["mini_a_scale"] * blen * scale + coeffs[
                    "mini_a_grid"
                ] * blen * blen + coeffs["mini_a0"]
                reduce_width = float(row.get("ACC_FP_WIDTH") or row.get("acc_fp_width") or 16)
                row["matrix_base_area"] = (mlen / blen) * mini
                row["matrix_residual_area"] = max(0.0, float(row["area_um2"]) - row["matrix_base_area"])
                row["feat_reduce"] = mlen * blen * reduce_width
                row["feat_accum"] = blen * blen
                row["feat_const"] = 1.0
            vals, hierarchy_metadata = hierarchy_supervised_matrix_coeffs(mm_rows)
            coeffs.update(
                {
                    "mm_stack_c": vals[0],
                    "mm_reduce_c": vals[1],
                    "mm_accum_c": vals[2],
                    "mm_top_c": vals[3],
                    "mm_acc_fp_width": 16.0,
                }
            )
            mm_mape = matrix_hierarchy_mape(mm_rows, vals[0], vals[1], vals[2], vals[3])
            metadata["matrix_machine_mape_pct"] = mm_mape
            metadata["model_version"] = "matrix_machine_hierarchy_residual_v2"
            metadata.update(hierarchy_metadata)
            write_matrix_diagnostics(run_dir, mode, mm_rows, coeffs)
        out = {
            "metadata": {
                **metadata,
                "status": "fitted_from_local_plena_rtl_synth" if pe_rows else "bootstrap_insufficient_data",
            },
            "coefficients": coeffs,
        }
        path = run_dir / f"{mode}_model_coefficients.json"
        path.write_text(json.dumps(json_safe(out), indent=2, sort_keys=True))
        if copy_to_calibration:
            CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, CALIBRATION_DIR / path.name)


def write_split_csvs(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Export successful rows into family/level compact CSV artifacts."""
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    for mode in ["mxint", "mxfp"]:
        for level, name in [("pe", "pe"), ("mini_array", "mini_array"), ("matrix_machine", "matrix_machine")]:
            selected = [row for row in rows if row.get("status") == "complete" and row.get("mode") == mode and row.get("level") == level]
            if not selected:
                continue
            out = run_dir / f"{name}_{mode}.csv"
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(selected)
            if copy_to_calibration:
                CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, CALIBRATION_DIR / out.name)


def write_structural_leaf_csv(
    csv_path: Path,
    run_dir: Path,
    copy_to_calibration: bool,
) -> None:
    """Export v4 leaf anchors without overwriting the historical level CSVs."""
    if not csv_path.exists():
        return
    rows = [
        row
        for row in csv.DictReader(csv_path.open(newline=""))
        if row.get("status") == "complete"
    ]
    if not rows:
        return
    out = run_dir / "matrix_structural_leaf_points.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, CALIBRATION_DIR / out.name)


def run_dry_run(points: list[Point], run_dir: Path) -> None:
    """Generate plan and wrappers without modifying source RTL or invoking DC."""
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    wrappers = run_dir / "wrappers"
    wrappers.mkdir(parents=True, exist_ok=True)
    for point in points:
        if point.level != "matrix_machine":
            (wrappers / f"{point.top_module}.sv").write_text(wrapper_text(point))
    print(f"Dry run wrote {len(points)} planned points to {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse hierarchy mode, worker, resume, retry, and cleanup options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "mxint",
            "mxfp",
            "both",
            "matrix-small",
            "matrix-mxint-refine-v1",
            "matrix-mxint-small-grid-v2",
            "structural-v4-leaves",
            "structural-v4-composite-smoke",
        ],
        default="both",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=RTL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--point-id-regex", help="only run planned points whose point_id matches this regex")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", default="4", help="worker count or 'auto' to use available DC licenses")
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--cleanup-worker-builds", action="store_true", default=True)
    parser.add_argument("--no-cleanup-worker-builds", dest="cleanup_worker_builds", action="store_false")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--no-copy-to-calibration", action="store_true")
    parser.add_argument(
        "--license-retry-wait-sec",
        type=float,
        default=float(os.environ.get("PLENA_DC_LICENSE_RETRY_WAIT_SEC", "60")),
        help="seconds to wait before retrying a point when DC licenses are busy",
    )
    parser.add_argument(
        "--license-max-retries",
        type=int,
        default=int(os.environ.get("PLENA_DC_LICENSE_MAX_RETRIES", "0")),
        help="max license-busy retries per point; 0 means retry indefinitely",
    )
    return parser.parse_args()


def main() -> int:
    """Run or dry-run the standalone MatrixMachine calibration workflow."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"matrix_{safe_run_name}_{os.getpid()}"
    points = build_plan(args.mode)
    if args.point_id_regex:
        pattern = re.compile(args.point_id_regex)
        points = [point for point in points if pattern.search(point.point_id)]
    if args.limit is not None:
        points = points[: args.limit]
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    if args.dry_run:
        run_dry_run(points, run_dir)
        return 0

    csv_path = run_dir / "calibration_points.csv"
    normalize_csv_schema(csv_path)
    completed = read_completed_keys(csv_path) if args.resume else set()
    pending = [point for point in points if point.point_key not in completed]
    if not pending:
        print("No pending points.")
        if args.mode == "structural-v4-leaves":
            write_structural_leaf_csv(csv_path, run_dir, not args.no_copy_to_calibration)
        else:
            fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
            write_split_csvs(csv_path, run_dir, not args.no_copy_to_calibration)
        return 0

    if args.worker_root.exists() and not args.keep_workers:
        shutil.rmtree(args.worker_root)
    args.worker_root.mkdir(parents=True, exist_ok=True)
    requested_workers = resolve_dc_worker_count(args.workers, repo_root=ROOT)
    worker_count, tmp_preflight = preflight_tmp_workers(requested_workers)
    (run_dir / "tmp_preflight.json").write_text(
        json.dumps(tmp_preflight, indent=2, sort_keys=True)
    )
    print(
        f"/tmp preflight: {tmp_preflight['free_after_gib']:.1f} GiB free; "
        f"using {worker_count}/{requested_workers} workers",
        flush=True,
    )
    worker_paths = [create_worker_copy(i, args.worker_root, args.rtl_root) for i in range(worker_count)]
    worker_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in enumerate(worker_paths):
        worker_queue.put(item)
    csv_lock = threading.Lock()

    def wrapped(point: Point) -> dict[str, Any]:
        worker_id, worker_path = worker_queue.get()
        try:
            return run_point(
                point,
                worker_id,
                worker_path,
                args.rtl_root,
                run_dir,
                args.cleanup_worker_builds,
                args.license_retry_wait_sec,
                args.license_max_retries,
            )
        finally:
            worker_queue.put((worker_id, worker_path))

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(wrapped, point): point for point in pending}
            for future in as_completed(futures):
                row = future.result()
                append_row(csv_path, row, csv_lock)
                print(f"[{row['status']}] {row['point_id']} area={row['area_um2']} reason={row['failure_reason']}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; compact rows already written remain resumable.", file=sys.stderr)
    finally:
        if args.mode == "structural-v4-leaves":
            write_structural_leaf_csv(csv_path, run_dir, not args.no_copy_to_calibration)
        else:
            write_split_csvs(csv_path, run_dir, not args.no_copy_to_calibration)
            fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
        if not args.keep_workers:
            cleanup_workers(args.worker_root)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
