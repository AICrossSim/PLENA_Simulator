"""Structural-census area model for the PLENA MatrixMachine (um^2).

Area is composed from exact RTL replication counts times fitted per-unit areas,
rather than regressing total area against shape polynomials. A logical
MatrixMachine holds ``MLEN/BLEN`` systolic slices; each slice is a ``BLEN x BLEN``
PE grid, and cross-K reduction happens only between slices. Because every
MLEN/BLEN-scaling term is an exact count and only the small per-PE/per-node unit
areas are fitted, the model extrapolates from the calibrated shapes (MLEN<=64) to
large arrays (MLEN=1024, the reference decode chip), where a shape polynomial
fitted on small MLEN diverges.

Precision enters through the asymmetric PE operand sides (T = max(KV, weight),
L = activation); see ``precision.derive_compute_sides``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .precision import derive_compute_sides

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_COEFFS = CALIBRATION_DIR / "matrix_structural_coefficients.json"

# Feature -> closure over (mlen, blen, t_bits, l_bits, scale_width). Each feature
# counts real hardware, so the fit only has to set small per-unit areas.
FEATURES: dict[str, Any] = {
    "pe_tl":  lambda m, b, t, l, s: m * b * t * l,       # multiplier cells: T*L per PE
    "pe_sum": lambda m, b, t, l, s: m * b * (t + l),     # operand registers/forwarding per PE
    "pe_0":   lambda m, b, t, l, s: m * b,               # fixed logic per PE (MLEN*BLEN PEs)
    "reduce": lambda m, b, t, l, s: b * (m - b),         # cross-K reduce edges = BLEN^2*(slices-1)
    "scale":  lambda m, b, t, l, s: m * s,               # per-lane scale distribution
    "out":    lambda m, b, t, l, s: b * b,               # output accumulate/convert cells
    "fixed":  lambda m, b, t, l, s: m // b,              # per-slice fixed control
    "const":  lambda m, b, t, l, s: 1.0,                 # shape-independent offset
}
FEATURE_NAMES = list(FEATURES)


def structural_counts(mlen: int, blen: int) -> dict[str, int]:
    """Exact RTL replication counts for one legal (MLEN, BLEN) shape."""
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError(f"need MLEN>0, BLEN>0, MLEN%BLEN==0; got MLEN={mlen}, BLEN={blen}")
    splits = mlen // blen
    return {
        "slices": splits,
        "pe_count": mlen * blen,
        "reduce_edges": blen * blen * max(splits - 1, 0),
        "output_cells": blen * blen,
    }


def feature_row(mlen: int, blen: int, t_bits: int, l_bits: int, scale_width: int) -> dict[str, float]:
    """Evaluate every structural feature for one shape+precision point."""
    return {name: float(fn(mlen, blen, t_bits, l_bits, scale_width)) for name, fn in FEATURES.items()}


def _artifact(path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(path or os.environ.get("PLENA_AREA_MATRIX_COEFFICIENTS") or DEFAULT_COEFFS)
    return json.loads(selected.read_text()) if selected.exists() else {}


def load_coefficients(mode: str, path: str | Path | None = None) -> dict[str, float] | None:
    """Load fitted per-mode (mxint/mxfp) structural coefficients, or None."""
    raw = _artifact(path)
    coeffs = raw.get(mode, raw.get("coefficients")) if raw else None
    return {str(k): float(v) for k, v in coeffs.items()} if coeffs else None


def load_pdk_scale(path: str | Path | None = None) -> float:
    """DC-synthesis-corner -> reference 7 nm OpenROAD PDK conversion factor.

    The DC calibration corner runs a uniform ~1.67x above the reference PDK
    across the whole compute hierarchy. This single constant, set by the fitter
    so 4x1024 MXINT4 lands on 0.237 mm^2, rescales absolute areas onto the
    reference corner and leaves every relative precision/shape trade-off
    untouched. Defaults to 1.0 (raw DC corner) if unfitted.
    """
    return float(_artifact(path).get("pdk_scale_reference", 1.0))


def matrix_area_from_sides(
    mlen: int, blen: int, sides: Mapping[str, Any], coeffs: Mapping[str, float]
) -> float:
    """MatrixMachine area (um^2) from resolved compute sides and fitted coeffs."""
    t_bits = int(sides["t_width"])
    l_bits = int(sides["l_width"])
    scale_width = int(sides.get("scale_width", 8))
    feats = feature_row(mlen, blen, t_bits, l_bits, scale_width)
    return sum(float(coeffs.get(name, 0.0)) * feats[name] for name in FEATURE_NAMES)


def estimate_matrix_machine_area(
    config: Mapping[str, Any], *, coefficients_path=None, corner: str = "reference"
) -> dict[str, Any]:
    """Precision-aware MatrixMachine area (um^2) for one (hardware, precision) point.

    Required keys: ACT_WIDTH, KV_WIDTH, MLEN, BLEN. WEIGHT_WIDTH defaults MXINT4.
    ``corner="reference"`` (default) returns PDK-scaled area; ``corner="dc"``
    returns the raw DC-synthesis-corner area (used by holdout checks).
    """
    sides = derive_compute_sides(
        config["ACT_WIDTH"], config["KV_WIDTH"], config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    mode = str(sides["mode"])
    coeffs = load_coefficients(mode, coefficients_path)
    if coeffs is None:
        raise FileNotFoundError(
            f"no fitted matrix coefficients for mode={mode}; run `python -m area.fit`")
    mlen, blen = int(config["MLEN"]), int(config["BLEN"])
    dc_area = matrix_area_from_sides(mlen, blen, sides, coeffs)
    scale = load_pdk_scale(coefficients_path) if corner == "reference" else 1.0
    return {
        "area": dc_area * scale,
        "area_dc_corner": dc_area,
        "pdk_scale": scale,
        "area_model": f"matrix_structural_census_{mode}_{corner}",
        "mode": mode,
        "counts": structural_counts(mlen, blen),
        "sides": {k: sides[k] for k in ("t_width", "l_width", "scale_width")},
    }
