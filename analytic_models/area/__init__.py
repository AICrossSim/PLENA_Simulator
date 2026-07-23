"""Precision-aware structural area model for the PLENA decode chip.

Estimates synthesized MatrixMachine logic and on-chip SRAM macro area in square
micrometres (um^2); callers divide by 1e6 for mm^2. The MatrixMachine term is a
structural census (exact RTL counts x fitted unit areas) that extrapolates from
the calibrated shapes (MLEN<=64) to large arrays.
"""

from __future__ import annotations

from typing import Any, Mapping

from .precision import PrecisionError, derive_compute_sides, parse_precision
from .matrix import estimate_matrix_machine_area, matrix_area_from_sides, structural_counts

__all__ = [
    "PrecisionError",
    "derive_compute_sides",
    "parse_precision",
    "estimate_matrix_machine_area",
    "structural_counts",
    "estimate_area",
]


def estimate_area(config: Mapping[str, Any], **kwargs) -> dict[str, Any]:
    """Total decode-chip proxy area (um^2) = MatrixMachine + on-chip SRAM macros.

    Vector/scalar/HBM-interface logic is deliberately excluded: it is a roughly
    precision-invariant fixed overhead and not the co-design lever, whereas the
    MatrixMachine (precision-sensitive, anchor-validated) and the SRAM macros
    (KV/weight capacity) are what the decode DSE trades off.
    """
    matrix = estimate_matrix_machine_area(config)
    try:
        from .sram import estimate_sram_area
        sram = estimate_sram_area(config)
        sram_area = float(sram["area"])
        sram_breakdown = sram.get("breakdown", {})
    except Exception:
        sram_area, sram_breakdown = 0.0, {}
    total = float(matrix["area"]) + sram_area
    return {
        "area": total,
        "area_model": "plena_decode_matrix_structural_plus_sram_v1",
        "matrix_machine_area": float(matrix["area"]),
        "sram_macro_area": sram_area,
        "breakdown": {"MatrixMachine": float(matrix["area"]), **sram_breakdown},
        "matrix_machine": matrix,
    }
