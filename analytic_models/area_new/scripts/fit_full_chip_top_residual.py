#!/usr/bin/env python3
"""Refit the proportional full-chip integration residual.

The MatrixMachine model is part of the module-logic subtotal used by this
calibration.  Whenever that model changes, the old top-level residual is no
longer calibrated against the same inputs.  This script therefore recomputes
the single nonnegative residual coefficient using only the committed training
anchors; IDs in ``full_chip_validation_split.json`` remain strict holdouts.

SRAM macro area is intentionally absent from both sides of this fit.  Full-chip
DC reports contain standard-cell logic and SRAM wrappers, but not physical SRAM
bitcell macros.  The public area proxy adds macro-table SRAM after applying this
logic residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytic_models.area_new import estimate_area


CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"
DEFAULT_ANCHORS = CALIBRATION_DIR / "full_chip_anchors.csv"
DEFAULT_SPLIT = CALIBRATION_DIR / "full_chip_validation_split.json"
DEFAULT_OUTPUT = CALIBRATION_DIR / "full_chip_top_residual_coefficients.json"

INTEGER_FIELDS = {
    "MLEN",
    "VLEN",
    "BLEN",
    "HLEN",
    "MATRIX_SRAM_DEPTH",
    "VECTOR_SRAM_DEPTH",
    "INT_SRAM_DEPTH",
    "FP_SRAM_DEPTH",
    "INT_DATA_WIDTH",
    "MX_SCALE_WIDTH",
    "HBM_M_Prefetch_Amount",
    "HBM_V_Prefetch_Amount",
    "HBM_V_Writeback_Amount",
}


def _config(row: dict[str, str]) -> dict[str, Any]:
    config: dict[str, Any] = {key: value for key, value in row.items() if value != ""}
    for key in INTEGER_FIELDS:
        if key in config:
            config[key] = int(config[key])
    return config


def _metrics(rows: list[dict[str, float]], coefficient: float) -> dict[str, float]:
    errors = [
        abs(row["logic"] * (1.0 + coefficient) - row["actual"]) / row["actual"] * 100.0
        for row in rows
    ]
    return {
        "mape_pct": sum(errors) / len(errors),
        "maximum_absolute_error_pct": max(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    validation_ids = set(json.loads(args.split.read_text())["validation_point_ids"])
    training: list[dict[str, float]] = []
    with args.anchors.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["point_id"] in validation_ids:
                continue
            raw = estimate_area(_config(row), apply_top_residual=False)
            training.append(
                {
                    "logic": float(raw["logic_area_before_top_residual"]),
                    "actual": float(row["area_um2"]),
                }
            )
    if not training:
        raise ValueError("No full-chip training anchors remain after holdout filtering")

    numerator = sum(item["logic"] * (item["actual"] - item["logic"]) for item in training)
    denominator = sum(item["logic"] ** 2 for item in training)
    coefficient = max(0.0, numerator / denominator)
    metrics = _metrics(training, coefficient)
    artifact = {
        "coefficients": {"logic_fraction": coefficient},
        "metadata": {
            "model_version": "full_chip_proportional_logic_residual_v1",
            "status": f"fitted_from_{len(training)}_full_chip_training_anchors",
            "fit_formula": "top_residual_um2 = logic_fraction * predicted_module_logic_um2",
            "source_csv": str(args.anchors.relative_to(ROOT)),
            "validation_point_ids_source": str(args.split.relative_to(ROOT)),
            "training_points": len(training),
            "training_logic_mape_pct": metrics["mape_pct"],
            "training_logic_maximum_absolute_error_pct": metrics[
                "maximum_absolute_error_pct"
            ],
            "matrix_model": "matrix_machine_structural_census_v4",
            "dc_area_scope": "logic plus SRAM wrappers; no SRAM bitcell macros",
            "note": (
                "Residual captures top-level control/interconnect, SRAM wrapper logic, "
                "and aggregate module-model bias. SRAM macro area is added afterward."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
