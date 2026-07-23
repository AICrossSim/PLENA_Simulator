"""Chip-area estimates for the decode chip: multiplier proxy or the
precision-aware structural `area` package.

Two models, selected by name:

  proxy       — mm^2 per multiplier, scaled from the reference array
                (0.237 mm^2 / 4096 multipliers at 7 nm). Precision-blind but
                monotone in MLEN*BLEN; the default for DSE ranking.
  calibrated  — analytic_models/area: precision-aware MatrixMachine structural
                census (exact RTL counts x DC-fitted unit areas) + ASAP7 SRAM
                macros. The census extrapolates past the calibrated shapes
                (MLEN<=64) to large arrays; one PDK constant rescales the DC
                corner so 4x1024 MXINT4 lands on the known 0.237 mm^2.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Reference flattened array: 4x1024 MXINT4 multipliers at 7 nm.
REF_MULTIPLIERS = 4096
REF_MM2 = 0.237
MM2_PER_MULTIPLIER = REF_MM2 / REF_MULTIPLIERS


def _width_token(elem_bits: int, label: str) -> str:
    """Turn a precision element into the area model's width token, e.g.
    'MXINT_4' or 'MXFP_E4M3'. A label with an exponent marker is a minifloat."""
    if "E" in str(label).upper() and "M" in str(label).upper():
        lab = str(label).upper()
        e = lab[lab.index("E"):]
        return f"MXFP_{e}" if not e.startswith("MXFP") else e
    return f"MXINT_{int(elem_bits)}"


def proxy_mm2(mlen: int, blen: int) -> float:
    return mlen * blen * MM2_PER_MULTIPLIER


def calibrated_mm2(
    mlen: int,
    blen: int,
    vlen: int,
    prec: dict,
    *,
    fp_setting: str = "FP_E5M6",
    matrix_sram_size: int = 4096,
    vector_sram_size: int = 4096,
) -> float:
    """Precision-aware mm^2 (MatrixMachine + SRAM) for one hardware/precision point.

    `prec` is the disagg_decode precision dict (attn/ffn/kv element bits + labels).
    The array must support the widest streamed operand; the area model reads the
    compute widths from the width tokens.
    """
    sys.path.insert(0, str(_HERE.parent))
    from area import estimate_area

    w_elem = max(int(prec["attn_elem"]), int(prec["ffn_elem"]))
    w_label = prec["attn_label"] if prec["attn_elem"] >= prec["ffn_elem"] else prec["ffn_label"]
    cfg = {
        "MLEN": int(mlen),
        "BLEN": int(blen),
        "VLEN": int(vlen),
        "WEIGHT_WIDTH": _width_token(w_elem, w_label),
        "KV_WIDTH": _width_token(int(prec["kv_elem"]), prec["kv_label"]),
        "ACT_WIDTH": _width_token(min(8, max(int(prec["m_bits"]), 2)), "int"),
        "FP_SETTING": fp_setting,
        "INT_DATA_WIDTH": 32,
        "MATRIX_SRAM_SIZE": int(matrix_sram_size),
        "VECTOR_SRAM_SIZE": int(vector_sram_size),
    }
    out = estimate_area(cfg)
    # Report MatrixMachine area only: it is the precision co-design lever, it
    # matches the 0.237 mm^2 reference point, and it is what the multiplier
    # proxy measures — so the two models stay directly comparable.
    return float(out["matrix_machine_area"]) / 1e6  # um^2 -> mm^2


def area_mm2(model: str, hw_cfg, prec: dict) -> float:
    """Dispatch on model name; hw_cfg is the analytic HardwareConfig."""
    if model == "calibrated":
        return calibrated_mm2(
            hw_cfg.MLEN, hw_cfg.BLEN, hw_cfg.VLEN, prec,
            matrix_sram_size=getattr(hw_cfg, "MATRIX_SRAM_SIZE", 4096),
            vector_sram_size=getattr(hw_cfg, "VECTOR_SRAM_SIZE", 4096),
        )
    return proxy_mm2(hw_cfg.MLEN, hw_cfg.BLEN)
