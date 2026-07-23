"""On-chip SRAM macro area for the PLENA decode chip (um^2).

Behavioural SRAMs are black boxes in the full-chip DC flow, so their bitcell
area is estimated by tiling real single-port ASAP7 macros (OpenROAD collateral)
and taking the minimum-area legal tiling per logical memory. Precision enters
through the Matrix-SRAM row width (T-side element + shared scale) and the
Vector-SRAM payload (FP activation + block-scaled ACT/KV). A DSE floorplanning
proxy, not a foundry SRAM compiler result.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .precision import derive_compute_sides, parse_precision

MACRO_TABLE = Path(__file__).with_name("calibration") / "asap7_sram_macro_table.csv"


def _macros(path: str | Path | None = None) -> list[dict[str, Any]]:
    p = Path(path or os.environ.get("PLENA_AREA_SRAM_MACRO_TABLE") or MACRO_TABLE)
    if not p.exists():
        return []
    out = []
    for r in csv.DictReader(p.open(newline="")):
        out.append({"macro": r["macro"], "depth": int(r["depth"]),
                    "width": int(r["width"]), "area_um2": float(r["area_um2"])})
    return out


def _fp_width(config: Mapping[str, Any]) -> int:
    token = str(config.get("FP_SETTING", "FP_E5M6")).upper().replace("FP_", "")
    if token.startswith("E") and "M" in token:
        exp, mant = token[1:].split("M", 1)
        return 1 + int(exp) + int(mant)
    return 1 + int(config.get("FP_EXP_WIDTH", 5)) + int(config.get("FP_MANT_WIDTH", 6))


def _tile_area(depth: int, width: int, ports: int, macros: list[dict]) -> float:
    """Minimum-area tiling of single-port macros; ports = full macro copies."""
    if not macros:
        raise ValueError("empty ASAP7 SRAM macro table")
    ports = max(1, int(ports))
    return min(
        math.ceil(depth / m["depth"]) * math.ceil(width / m["width"]) * ports * m["area_um2"]
        for m in macros
    )


def _depth(config: Mapping[str, Any], *names: str, default: int) -> int:
    for n in names:
        if n in config:
            return int(config[n])
    return default


def estimate_sram_area(config: Mapping[str, Any], *, macro_table_path=None) -> dict[str, Any]:
    """Precision-aware on-chip SRAM area (um^2): Matrix + Vector + scalar SRAMs."""
    macros = _macros(macro_table_path)
    sides = derive_compute_sides(
        config["ACT_WIDTH"], config["KV_WIDTH"], config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)))
    mlen = int(config["MLEN"])
    vlen = int(config.get("VLEN", mlen))
    scale_width = int(sides["scale_width"])

    # Matrix SRAM: MLEN-wide rows of T-side element + shared scale; 2 ports.
    m_depth = _depth(config, "MATRIX_SRAM_DEPTH", "MATRIX_SRAM_SIZE", default=max(32, 2 * mlen))
    m_width = mlen * (int(sides["t_width"]) + scale_width)
    # Vector SRAM: VLEN-wide FP activation + block-scaled ACT/KV payloads; 2 ports.
    act = parse_precision(config["ACT_WIDTH"], default_scale_width=scale_width)
    kv = parse_precision(config["KV_WIDTH"], default_scale_width=scale_width)
    v_depth = _depth(config, "VECTOR_SRAM_DEPTH", "VECTOR_SRAM_SIZE", default=max(32, 256 + math.ceil(mlen / vlen)))
    v_width = vlen * (_fp_width(config) + act.element_width + kv.element_width) + 2 * scale_width * max(1, vlen // max(int(config.get("BLOCK_DIM", config.get("BLEN", 4))), 1))
    # Scalar SRAMs (small, fixed): integer + FP register files, 1 port each.
    si = _tile_area(_depth(config, "INT_SRAM_DEPTH", default=32), int(config.get("INT_DATA_WIDTH", 32)), 1, macros)
    sf = _tile_area(_depth(config, "FP_SRAM_DEPTH", default=512), _fp_width(config), 1, macros)

    breakdown = {
        "MatrixSRAM": _tile_area(m_depth, m_width, 2, macros),
        "VectorSRAM": _tile_area(v_depth, v_width, 2, macros),
        "ScalarIntSRAM": si,
        "ScalarFPSRAM": sf,
    }
    return {"area": sum(breakdown.values()), "breakdown": breakdown, "model": "asap7_sram_macro_tiling"}
