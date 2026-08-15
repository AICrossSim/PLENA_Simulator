"""Independent gate-level cross-validation of the MatrixMachine area census.

A separate Design Compiler campaign synthesised ``matrix_machine`` on its own -
eight timing-closed points, ASAP7 RVT_TT at ``PVT_0P7V_25C`` with a 1000 ps
clock, MLEN 16-64 and BLEN 4-8, MXFP operands only. Those points were never
part of the fit behind ``matrix_structural_coefficients.json``, so they are a
holdout for the shipped census in the strict sense: no coefficient was tuned
against them.

This module derives the comparison rather than restating it, so the recorded
figures cannot drift away from the measurements they came from. It reads the
two vendored campaign tables and returns one record:

``precision_law``
    a two-term least-squares fit of area against exponent and mantissa bits at
    the 16x4 geometry, its worst residual, and the mantissa/exponent cost ratio;
``geometry_law``
    area per processing element across the measured shapes and the retained
    64x8 extrapolation check;
``leakage_density``
    leakage per unit area, which the campaign reports directly and which needs
    no switching activity;
``census_cross_validation``
    the shipped structural census evaluated at every measured point, reported
    both raw and after removing a single uniform corner offset;
``compute_energy_envelope``
    the declared-activity dynamic envelope and the toggle rate it implies for
    the analytic ``0.203 pJ/MAC`` anchor.

**Scope.** Every number here is ``matrix_machine`` alone, in um^2, at MLEN
16-64, at 25 C. The decode study's headline areas are full-chip mm^2 at MLEN
128-1024 and above. These figures are not an error bar on those, and the record
carries that boundary in ``scope`` so it travels with any consumer.

Refresh with::

    cd analytic_models
    python -m area.gate_level_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .matrix import estimate_matrix_machine_area

SCHEMA = "plena-matrix-gate-level-validation"

CALIBRATION_DIR = Path(__file__).with_name("calibration")
AREA_TABLE = CALIBRATION_DIR / "matrix_machine_gate_level_pvt0p7v25c.csv"
ENERGY_TABLE = CALIBRATION_DIR / "matrix_machine_gate_level_activity_envelope.csv"
ARTIFACT = CALIBRATION_DIR / "matrix_gate_level_validation.json"

#: Reference geometry of the per-precision ladder: four MXFP formats at one
#: shape, which is what the precision law is fitted on.
LADDER_MLEN = 16
LADDER_BLEN = 4

#: The analytic decode model's compute anchor, cross-checked but not refitted.
ANALYTIC_PJ_PER_MAC = 0.203

#: Independent leakage-density fit quoted by the campaign, retained only as an
#: agreement check on the density derived here.
REFERENCE_LEAKAGE_MW_PER_UM2 = 9.257572890628394e-07

SCOPE: dict[str, Any] = {
    "block": "matrix_machine",
    "block_share_of_mapped_area_pct": 98.3,
    "unit": "um^2",
    "library": "ASAP7 RVT_TT",
    "operating_condition": "PVT_0P7V_25C",
    "clock_ps": 1000.0,
    "measured_mlen": [16, 32, 64],
    "measured_blen": [4, 8],
    "families": ["mxfp"],
    "not_comparable_to": (
        "full-chip mm^2 estimates at MLEN 128-1024; this campaign measures one "
        "block, in um^2, over MLEN 16-64, at 25 C"
    ),
    "timing_closed": True,
}


def _split_precision(signature: str) -> tuple[str, int, int]:
    """Return (format token, exponent bits, mantissa bits) for one signature."""

    token = signature.split(":", 1)[-1].split("x", 1)[0]
    exponent, mantissa = token.upper().split("_E", 1)[1].split("M", 1)
    return token, int(exponent), int(mantissa)


def load_area_points(path: Path = AREA_TABLE) -> list[dict[str, Any]]:
    """Read the measured area/leakage table into normalised records."""

    points: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            precision, exponent, mantissa = _split_precision(record["signature"])
            mlen, blen = int(record["MLEN"]), int(record["BLEN"])
            points.append(
                {
                    "point_id": record["point_id"],
                    "precision": precision,
                    "exp_bits": exponent,
                    "mant_bits": mantissa,
                    "mlen": mlen,
                    "blen": blen,
                    "pes": mlen * blen,
                    "area_um2": float(record["area_um2"]),
                    "leakage_uw": float(record["leakage_uW"]),
                    "slack_ps": float(record["slack_ps"]),
                    "critical_path_ps": float(record["path_ps"]),
                    "split": record["split"],
                }
            )
    if not points:
        raise ValueError(f"{path} carries no measured points")
    return sorted(points, key=lambda point: (point["pes"], point["precision"]))


def load_energy_points(path: Path = ENERGY_TABLE) -> list[dict[str, Any]]:
    """Read the declared-activity dynamic sweep into normalised records."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                {
                    "point_id": record["point_id"],
                    "precision": record["precision"].split("x", 1)[0],
                    "mlen": int(record["MLEN"]),
                    "blen": int(record["BLEN"]),
                    "toggle_rate": float(record["toggle_rate"]),
                    "pj_per_mac": float(record["pj_per_mac"]),
                }
            )
    if not rows:
        raise ValueError(f"{path} carries no activity points")
    return rows


def _least_squares(matrix: Sequence[Sequence[float]], target: Sequence[float]):
    """Solve a small over-determined system by normal equations.

    The design matrix here is 4x3 and well conditioned, so a dependency-free
    Gaussian elimination on ``AtA x = Atb`` is sufficient and keeps this module
    importable without NumPy.
    """

    columns = len(matrix[0])
    normal = [
        [
            sum(row[i] * row[j] for row in matrix)
            for j in range(columns)
        ]
        + [sum(row[i] * value for row, value in zip(matrix, target))]
        for i in range(columns)
    ]
    for pivot in range(columns):
        best = max(range(pivot, columns), key=lambda r: abs(normal[r][pivot]))
        normal[pivot], normal[best] = normal[best], normal[pivot]
        divisor = normal[pivot][pivot]
        if divisor == 0.0:
            raise ValueError("singular design matrix in the precision fit")
        normal[pivot] = [value / divisor for value in normal[pivot]]
        for row_index in range(columns):
            if row_index == pivot:
                continue
            factor = normal[row_index][pivot]
            normal[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(normal[row_index], normal[pivot])
            ]
    return [normal[i][columns] for i in range(columns)]


def precision_law(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit ``area = const + a*exp_bits + b*mant_bits`` at the ladder geometry."""

    ladder = [
        point
        for point in points
        if point["mlen"] == LADDER_MLEN and point["blen"] == LADDER_BLEN
    ]
    if len(ladder) < 3:
        raise ValueError("the precision ladder needs at least three formats")
    design = [[1.0, float(p["exp_bits"]), float(p["mant_bits"])] for p in ladder]
    constant, per_exp, per_mant = _least_squares(
        design, [p["area_um2"] for p in ladder]
    )
    residuals = {
        p["precision"]: (
            constant + per_exp * p["exp_bits"] + per_mant * p["mant_bits"]
        )
        / p["area_um2"]
        - 1.0
        for p in ladder
    }
    areas = [p["area_um2"] for p in ladder]
    return {
        "geometry": {"MLEN": LADDER_MLEN, "BLEN": LADDER_BLEN, "PEs": ladder[0]["pes"]},
        "formats": sorted(p["precision"] for p in ladder),
        "constant_um2": constant,
        "um2_per_exponent_bit": per_exp,
        "um2_per_mantissa_bit": per_mant,
        "mantissa_to_exponent_cost_ratio": per_mant / per_exp,
        "worst_abs_residual_pct": max(abs(value) for value in residuals.values()) * 100.0,
        "residual_pct_by_format": {
            name: value * 100.0 for name, value in sorted(residuals.items())
        },
        # The precision-independent share depends on which format it is taken
        # against, so both ends of the ladder are reported rather than one.
        "precision_independent_fraction_pct": {
            "vs_widest_format": constant / max(areas) * 100.0,
            "vs_narrowest_format": constant / min(areas) * 100.0,
        },
        "interpretation": (
            "a mantissa bit costs more silicon than an exponent bit, and the "
            "majority of the array does not shrink with operand width at all"
        ),
    }


def geometry_law(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report area per PE and the retained-geometry extrapolation check."""

    per_pe = {
        f"{p['precision']}_{p['mlen']}x{p['blen']}": p["area_um2"] / p["pes"]
        for p in points
    }
    holdout = [p for p in points if p["split"] == "holdout"]
    check: dict[str, Any] | None = None
    if holdout:
        retained = max(holdout, key=lambda p: p["pes"])
        anchors = [
            p
            for p in points
            if p["split"] == "train"
            and p["precision"] == retained["precision"]
            and p["mlen"] == LADDER_MLEN
            and p["blen"] == LADDER_BLEN
        ]
        if anchors:
            anchor = anchors[0]
            predicted = anchor["area_um2"] / anchor["pes"] * retained["pes"]
            check = {
                "anchor": f"{anchor['mlen']}x{anchor['blen']}",
                "retained": f"{retained['mlen']}x{retained['blen']}",
                "pe_ratio": retained["pes"] / anchor["pes"],
                "measured_um2": retained["area_um2"],
                "predicted_um2": predicted,
                "error_pct": (predicted - retained["area_um2"])
                / retained["area_um2"]
                * 100.0,
                "note": (
                    "constant area per PE, extrapolated from the smallest "
                    "measured shape to the retained largest one"
                ),
            }
    values = list(per_pe.values())
    return {
        "um2_per_pe": per_pe,
        "um2_per_pe_min": min(values),
        "um2_per_pe_max": max(values),
        "holdout_extrapolation": check,
        "shape_sensitivity_note": (
            "at an identical PE count the wide array is measurably smaller than "
            "the tall one, but the gap closes as the array grows; treat shape as "
            "a small-geometry correction, not a scaling law"
        ),
    }


def leakage_density(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive leakage per unit area, which the campaign measures directly."""

    densities = [p["leakage_uw"] * 1e-3 / p["area_um2"] for p in points]
    mean = statistics.fmean(densities)
    ladder = [
        p["leakage_uw"] * 1e-3 / p["area_um2"]
        for p in points
        if p["mlen"] == LADDER_MLEN and p["blen"] == LADDER_BLEN
    ]
    return {
        "mw_per_um2": mean,
        "w_per_mm2": mean * 1e3,
        "n_points": len(densities),
        "spread_pct": (max(densities) - min(densities)) / mean * 100.0,
        "min_mw_per_um2": min(densities),
        "max_mw_per_um2": max(densities),
        "precision_ladder_mw_per_um2": statistics.fmean(ladder) if ladder else None,
        "independent_fit_mw_per_um2": REFERENCE_LEAKAGE_MW_PER_UM2,
        "independent_fit_agreement_pct": abs(mean - REFERENCE_LEAKAGE_MW_PER_UM2)
        / REFERENCE_LEAKAGE_MW_PER_UM2
        * 100.0,
        "temperature_c": 25,
        "corner": "ASAP7 RVT_TT / PVT_0P7V_25C",
        "scope": "matrix_machine only; not full-chip logic",
        "caveat": (
            "subthreshold leakage is strongly temperature dependent, so a 25 C "
            "density is a floor rather than an operating-temperature figure; no "
            "hot-corner point was synthesised in this campaign"
        ),
    }


def census_cross_validation(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the shipped structural census against every measured point."""

    comparisons = []
    for point in points:
        config = {
            "ACT_WIDTH": point["precision"],
            "KV_WIDTH": point["precision"],
            "WEIGHT_WIDTH": point["precision"],
            "MLEN": point["mlen"],
            "BLEN": point["blen"],
        }
        predicted = float(
            estimate_matrix_machine_area(config, corner="dc")["area"]
        )
        comparisons.append(
            {
                "precision": point["precision"],
                "geometry": f"{point['mlen']}x{point['blen']}",
                "measured_um2": point["area_um2"],
                "census_dc_corner_um2": predicted,
                "ratio": predicted / point["area_um2"],
            }
        )
    ratios = [entry["ratio"] for entry in comparisons]
    offset = statistics.fmean(ratios)
    for entry in comparisons:
        entry["error_after_uniform_offset_pct"] = (
            entry["ratio"] / offset - 1.0
        ) * 100.0
    residuals = [
        abs(entry["error_after_uniform_offset_pct"]) for entry in comparisons
    ]
    return {
        "points": comparisons,
        "uniform_offset_census_over_campaign": offset,
        "raw_error_pct": {
            "min": (min(ratios) - 1.0) * 100.0,
            "max": (max(ratios) - 1.0) * 100.0,
        },
        "shape_and_precision_error_after_offset_pct": {
            "median": statistics.median(residuals),
            "mean": statistics.fmean(residuals),
            "max": max(residuals),
        },
        "verdict": (
            "the census reproduces every measured shape and precision trade-off "
            "to within a few percent once a single uniform corner offset is "
            "removed; the offset itself is a level difference between two "
            "Design Compiler campaigns, not a shape or precision error"
        ),
        "known_model_limit": (
            "the census features depend on total operand width only, so it "
            "cannot separate two MXFP formats of equal width; the campaign "
            "measures E1M2 and E2M1 apart by a few percent and that difference "
            "is unrepresentable in the current feature set"
        ),
    }


def compute_energy_envelope(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_pj_per_mac: float = ANALYTIC_PJ_PER_MAC,
) -> dict[str, Any]:
    """Bracket the analytic compute anchor inside the declared-activity sweep."""

    implied: dict[str, float] = {}
    for point_id in sorted({row["point_id"] for row in rows}):
        sweep = sorted(
            (row for row in rows if row["point_id"] == point_id),
            key=lambda row: row["toggle_rate"],
        )
        for low, high in zip(sweep, sweep[1:]):
            if low["pj_per_mac"] <= anchor_pj_per_mac <= high["pj_per_mac"]:
                span = high["pj_per_mac"] - low["pj_per_mac"]
                fraction = (anchor_pj_per_mac - low["pj_per_mac"]) / span
                key = f"{sweep[0]['precision']}_{sweep[0]['mlen']}x{sweep[0]['blen']}"
                implied[key] = low["toggle_rate"] + fraction * (
                    high["toggle_rate"] - low["toggle_rate"]
                )
                break
    if not implied:
        raise ValueError("no measured sweep brackets the analytic anchor")
    energies = [row["pj_per_mac"] for row in rows]
    toggles = [row["toggle_rate"] for row in rows]
    return {
        "anchor_pj_per_mac": anchor_pj_per_mac,
        "envelope_pj_per_mac": {"min": min(energies), "max": max(energies)},
        "declared_toggle_rates": sorted(set(toggles)),
        "implied_toggle_rate_by_geometry": dict(sorted(implied.items())),
        "implied_toggle_rate_range": {
            "min": min(implied.values()),
            "max": max(implied.values()),
        },
        "geometries_bracketing_the_anchor": len(implied),
        "verdict": (
            "the analytic anchor sits inside the measured envelope at a toggle "
            "rate that is consistent across geometries and physically plausible"
        ),
        "evidence_scope": (
            "declared-activity vectorless estimate: the toggle rate is assumed "
            "and propagated by the synthesis tool, not measured from decode "
            "switching; this corroborates the anchor, it does not calibrate it"
        ),
    }


def build_record() -> dict[str, Any]:
    """Derive the complete gate-level validation record from the campaign."""

    points = load_area_points()
    energy_rows = load_energy_points()
    return {
        "schema": SCHEMA,
        "validates": "analytic_models/area/matrix.py structural census (mxfp)",
        "sources": {
            "areas": AREA_TABLE.name,
            "activity_envelope": ENERGY_TABLE.name,
        },
        "independent_of_the_fit": True,
        "n_points": len(points),
        "scope": dict(SCOPE),
        "precision_law": precision_law(points),
        "geometry_law": geometry_law(points),
        "leakage_density": leakage_density(points),
        "census_cross_validation": census_cross_validation(points),
        "compute_energy_envelope": compute_energy_envelope(energy_rows),
        "coefficients_changed": False,
        "decision": (
            "recorded as validation only. The shipped census coefficients are "
            "unchanged: the disagreement is a uniform level offset between two "
            "synthesis campaigns, and refitting on eight small-geometry MXFP "
            "points would trade a 71-row fit spanning both families for a "
            "narrower one without improving any relative trade-off"
        ),
    }


def write_record(path: Path = ARTIFACT) -> dict[str, Any]:
    """Refresh the stored artifact from the vendored campaign tables."""

    record = build_record()
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT,
        help="artifact path to write (defaults to the calibration directory)",
    )
    arguments = parser.parse_args(argv)
    record = write_record(arguments.output)
    census = record["census_cross_validation"]
    print(f"wrote {arguments.output} ({record['n_points']} measured points)")
    print(
        "  census offset "
        f"{census['uniform_offset_census_over_campaign']:.4f}x, "
        "residual after offset "
        f"{census['shape_and_precision_error_after_offset_pct']['max']:.2f}% max"
    )
    print(
        "  leakage "
        f"{record['leakage_density']['w_per_mm2']:.6f} W/mm^2 at 25 C "
        "(matrix machine only)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual refresh entry point
    raise SystemExit(main())
