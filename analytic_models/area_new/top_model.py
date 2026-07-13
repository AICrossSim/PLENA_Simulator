"""Model the area not represented by independently fitted logic modules.

Full-chip DC area exceeds the sum of Matrix, Vector, Scalar, and HBM interface
models because the top adds interconnect, control, wrappers, and integration
logic. A held-out full-chip calibration fits this aggregate residual as a
fraction of modeled logic area. It is not a physical standalone module.

SRAM macro area must be added after this correction and is never scaled by the
residual coefficient. All values use um^2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_COEFFICIENTS = CALIBRATION_DIR / "full_chip_top_residual_coefficients.json"


def _load_coefficients(path: str | Path | None = None) -> tuple[dict[str, float], Path | None, dict[str, Any]]:
    selected = Path(path or os.environ.get("PLENA_AREA_NEW_TOP_COEFFICIENTS", DEFAULT_COEFFICIENTS))
    if not selected.exists():
        return {}, None, {}
    raw = json.loads(selected.read_text())
    coefficients = raw.get("coefficients", raw)
    return {str(key): float(value) for key, value in coefficients.items()}, selected, raw.get("metadata", {})


def estimate_full_chip_top_residual(
    logic_area_um2: float,
    *,
    coefficients_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate aggregate full-chip integration residual in um^2.

    Missing coefficients intentionally disable the correction by selecting a
    zero fraction. Metadata is returned so reports can distinguish a fitted
    residual from the disabled fallback.
    """
    coefficients, source, metadata = _load_coefficients(coefficients_path)
    fraction = max(0.0, float(coefficients.get("logic_fraction", 0.0)))
    area = float(logic_area_um2) * fraction
    return {
        "area": area,
        "area_model": metadata.get("model_version", "full_chip_top_residual_disabled"),
        "inputs": {"logic_area_um2": float(logic_area_um2)},
        "coefficients": {"logic_fraction": fraction},
        "coefficients_source": str(source) if source else None,
        "metadata": metadata,
    }
