"""Fit PLENA structural area coefficients from retained DC aggregate tables.

Reads per-shape 7 nm Synopsys DC areas from ``calibration/matrix_machine_*.csv``,
plus hierarchy totals from ``calibration/full_chip_anchors.csv``. It solves for
non-negative per-unit areas and writes
``calibration/matrix_structural_coefficients.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import hbm_interface, scalar, top, vector
from .matrix import FEATURE_NAMES, feature_row, matrix_area_from_sides
from .precision import derive_compute_sides

CAL = Path(__file__).with_name("calibration")
OUT = CAL / "matrix_structural_coefficients.json"
REFERENCE_ANCHOR_UM2 = 237_000.0  # 0.237 mm^2: known area of the 4x1024 MXINT4 array
FULL_CHIP_TABLE = CAL / "full_chip_anchors.csv"
HOLDOUT_STRIDE = 4


def _load(mode: str) -> list[dict]:
    """Load completed rows with resolved (MLEN, BLEN, t_width, l_width, scale, area)."""
    path = CAL / f"matrix_machine_{mode}.csv"
    rows: list[dict] = []
    for r in csv.DictReader(path.open()):
        if r.get("status") != "complete":
            continue
        try:
            mlen, blen, scale = int(r["MLEN"]), int(r["BLEN"]), int(r["scale_width"])
            if mode == "mxint":
                t_w, l_w = int(r["T_BITS"]), int(r["L_BITS"])
            else:
                t_w = 1 + int(r["T_EXP"]) + int(r["T_MANT"])
                l_w = 1 + int(r["L_EXP"]) + int(r["L_MANT"])
            rows.append(
                dict(
                    MLEN=mlen,
                    BLEN=blen,
                    t=t_w,
                    l=l_w,
                    s=scale,
                    area=float(r["area_um2"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [
                feature_row(d["MLEN"], d["BLEN"], d["t"], d["l"], d["s"])[n]
                for n in FEATURE_NAMES
            ]
            for d in rows
        ],
        float,
    )
    y = np.array([d["area"] for d in rows], float)
    return X, y


def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve non-negative least squares with a NumPy-only active set.

    Clipping an unconstrained least-squares solution is not NNLS and materially
    changes the calibration when optional scientific packages are unavailable.
    The active set keeps every fitted structural unit area non-negative while
    matching the constrained optimum.
    """
    rows, columns = X.shape
    coefficients = np.zeros(columns, dtype=float)
    passive = np.zeros(columns, dtype=bool)
    gradient = X.T @ y
    tolerance = 10.0 * max(rows, columns) * np.spacing(1.0) * np.linalg.norm(X, 1)
    max_iterations = 30 * columns
    iterations = 0

    while np.any((~passive) & (gradient > tolerance)):
        candidate = int(np.argmax(np.where(~passive, gradient, -np.inf)))
        passive[candidate] = True

        trial = np.zeros(columns, dtype=float)
        trial[passive], *_ = np.linalg.lstsq(X[:, passive], y, rcond=None)
        while np.any(passive & (trial <= tolerance)):
            infeasible = passive & (trial <= tolerance)
            denominator = coefficients[infeasible] - trial[infeasible]
            can_step = denominator > 0.0
            if not np.any(can_step):
                passive[infeasible] = False
            else:
                alpha = np.min(
                    coefficients[infeasible][can_step] / denominator[can_step]
                )
                coefficients += alpha * (trial - coefficients)
                remove = passive & (coefficients <= tolerance)
                coefficients[remove] = 0.0
                passive[remove] = False

            trial.fill(0.0)
            if np.any(passive):
                trial[passive], *_ = np.linalg.lstsq(X[:, passive], y, rcond=None)
            iterations += 1
            if iterations >= max_iterations:
                raise RuntimeError("non-negative least-squares fit did not converge")

        coefficients = trial
        gradient = X.T @ (y - X @ coefficients)
        iterations += 1
        if iterations >= max_iterations:
            raise RuntimeError("non-negative least-squares fit did not converge")

    return np.maximum(coefficients, 0.0)


def _mape(coef: np.ndarray, rows: list[dict]) -> float:
    X, y = _design(rows)
    pred = X @ coef
    return float(np.mean(np.abs(pred - y) / y) * 100)


def _error_summary(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    errors = np.abs(predicted - observed) / observed * 100.0
    return {
        "median_abs_error_pct": round(float(np.median(errors)), 3),
        "p95_abs_error_pct": round(float(np.percentile(errors, 95)), 3),
        "max_abs_error_pct": round(float(np.max(errors)), 3),
        "mean_abs_error_pct": round(float(np.mean(errors)), 3),
    }


def _holdout_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Reserve every fourth retained row, giving an independent >=25% holdout."""

    training = [row for index, row in enumerate(rows) if index % HOLDOUT_STRIDE]
    holdout = [row for index, row in enumerate(rows) if not index % HOLDOUT_STRIDE]
    if len(holdout) / len(rows) < 0.25:
        raise ValueError("area holdout must contain at least 25% of retained rows")
    return training, holdout


def _predict(coef: np.ndarray, m: int, b: int, t: int, l: int, s: int = 8) -> float:
    feats = feature_row(m, b, t, l, s)
    return float(sum(coef[i] * feats[n] for i, n in enumerate(FEATURE_NAMES)))


def fit_mode(mode: str) -> dict:
    rows = _load(mode)
    if not rows:
        return {}
    X, y = _design(rows)
    coef = _solve(X, y)
    coeffs = {n: float(c) for n, c in zip(FEATURE_NAMES, coef)}

    in_mape = _mape(coef, rows)
    training, holdout = _holdout_split(rows)
    Xtr, ytr = _design(training)
    holdout_coefficients = _solve(Xtr, ytr)
    Xho, yho = _design(holdout)
    holdout_prediction = Xho @ holdout_coefficients
    holdout_errors = _error_summary(holdout_prediction, yho)
    anchor_ho = (
        _predict(holdout_coefficients, 1024, 4, 4, 4) if mode == "mxint" else None
    )

    report = {
        "in_sample_mape_pct": round(in_mape, 3),
        "holdout_strategy": "every_fourth_retained_row",
        "holdout_rows": len(holdout),
        "holdout_fraction": round(len(holdout) / len(rows), 6),
        "holdout_mape_pct": holdout_errors["mean_abs_error_pct"],
        **holdout_errors,
        "n_rows": len(rows),
    }
    dc_anchor = None
    if mode == "mxint":
        dc_anchor = _predict(coef, 1024, 4, 4, 4)
        report["dc_anchor_4x1024_mxint4_um2"] = round(dc_anchor, 1)
        report["implied_pdk_factor_dc_over_reference"] = round(
            dc_anchor / REFERENCE_ANCHOR_UM2, 3
        )
        if anchor_ho is not None:
            report["dc_anchor_from_holdout_um2"] = round(anchor_ho, 1)
            report["anchor_holdout_shift_pct"] = round(
                (anchor_ho - dc_anchor) / dc_anchor * 100, 2
            )
    return {"coefficients": coeffs, "report": report, "dc_anchor": dc_anchor}


def _load_full_chip() -> list[dict[str, str]]:
    with FULL_CHIP_TABLE.open(newline="") as handle:
        return [
            row for row in csv.DictReader(handle) if row.get("status") == "complete"
        ]


def _block_design(
    rows: list[dict[str, str]],
    names: tuple[str, ...],
    row_function,
    target_function,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.array(
        [[row_function(row)[name] for name in names] for row in rows], dtype=float
    )
    target = np.array([target_function(row) for row in rows], dtype=float)
    return design, target


def _top_target(row: dict[str, str]) -> float:
    hierarchy = sum(
        float(row[name])
        for name in (
            "hier_matrix_machine_area",
            "hier_vector_machine_area",
            "hier_scalar_machine_area",
            "hier_hbm_system_area",
        )
    )
    return float(row["area_um2"]) - hierarchy


_FULL_CHIP_BLOCKS = {
    "vector": (
        vector.FEATURE_NAMES,
        vector.feature_row,
        lambda row: float(row["hier_vector_machine_area"]),
    ),
    "scalar": (
        scalar.FEATURE_NAMES,
        scalar.feature_row,
        lambda row: float(row["hier_scalar_machine_area"]),
    ),
    "hbm_interface": (
        hbm_interface.FEATURE_NAMES,
        hbm_interface.feature_row,
        lambda row: float(row["hier_hbm_system_area"]),
    ),
    "top": (top.FEATURE_NAMES, top.feature_row, _top_target),
}


def fit_full_chip(matrix_fits: dict[str, dict]) -> dict[str, dict]:
    """Fit hierarchy blocks and report a five-of-seventeen full-chip holdout."""

    rows = _load_full_chip()
    training, holdout = _holdout_split(rows)
    coefficients: dict[str, dict[str, float]] = {}
    holdout_coefficients: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}

    for block, (names, row_function, target_function) in _FULL_CHIP_BLOCKS.items():
        X, y = _block_design(rows, names, row_function, target_function)
        fitted = _solve(X, y)
        coefficients[block] = {name: float(value) for name, value in zip(names, fitted)}

        Xtr, ytr = _block_design(training, names, row_function, target_function)
        held_fit = _solve(Xtr, ytr)
        holdout_coefficients[block] = held_fit
        Xho, yho = _block_design(holdout, names, row_function, target_function)
        report[block] = {
            "n_rows": len(rows),
            "holdout_rows": len(holdout),
            "holdout_fraction": round(len(holdout) / len(rows), 6),
            "holdout_strategy": "every_fourth_retained_row",
            **_error_summary(Xho @ held_fit, yho),
        }

    full_predictions: list[float] = []
    full_observations: list[float] = []
    for row in holdout:
        sides = derive_compute_sides(
            row["ACT_WIDTH"],
            row["KV_WIDTH"],
            row["WEIGHT_WIDTH"],
            default_scale_width=int(row["MX_SCALE_WIDTH"]),
        )
        matrix_coefficients = matrix_fits[str(sides["mode"])]["coefficients"]
        predicted = matrix_area_from_sides(
            int(row["MLEN"]),
            int(row["BLEN"]),
            sides,
            matrix_coefficients,
        )
        for block, (names, row_function, _) in _FULL_CHIP_BLOCKS.items():
            features = row_function(row)
            predicted += sum(
                holdout_coefficients[block][index] * features[name]
                for index, name in enumerate(names)
            )
        full_predictions.append(float(predicted))
        full_observations.append(float(row["area_um2"]))

    report["full_chip"] = {
        "n_rows": len(rows),
        "holdout_rows": len(holdout),
        "holdout_fraction": round(len(holdout) / len(rows), 6),
        "holdout_strategy": "every_fourth_retained_row",
        "includes_sram_bitcell_macros": False,
        **_error_summary(np.asarray(full_predictions), np.asarray(full_observations)),
    }
    return {"coefficients": coefficients, "report": report}


def build_artifact() -> dict:
    """Build the complete deterministic coefficient artifact in memory."""

    artifact: dict = {
        "model_version": "full_chip_structural_census",
        "reference_anchor_um2": REFERENCE_ANCHOR_UM2,
        "source": "calibration/matrix_machine_{mode}.csv",
        "full_chip_source": "calibration/full_chip_anchors.csv",
    }
    matrix_fits = {mode: fit_mode(mode) for mode in ("mxint", "mxfp")}
    for mode, fitted in matrix_fits.items():
        artifact[mode] = fitted["coefficients"]
        artifact.setdefault("report", {})[mode] = fitted["report"]
    dc_anchor = matrix_fits["mxint"]["dc_anchor"]
    artifact["pdk_scale_reference"] = REFERENCE_ANCHOR_UM2 / dc_anchor
    full_chip = fit_full_chip(matrix_fits)
    artifact["full_chip"] = full_chip["coefficients"]
    artifact["report"]["full_chip_blocks"] = full_chip["report"]
    return artifact


def main() -> None:
    artifact = build_artifact()
    print("=== PLENA full-chip structural-census fit ===")
    for mode in ("mxint", "mxfp"):
        report = artifact["report"][mode]
        print(
            f"[{mode}] n={report['n_rows']} holdout={report['holdout_rows']} "
            f"({report['holdout_fraction']:.1%}) median={report['median_abs_error_pct']}% "
            f"P95={report['p95_abs_error_pct']}%"
        )
    matrix_report = artifact["report"]["mxint"]
    print(
        "[anchor] reference 4x1024 MXINT4 = "
        f"{REFERENCE_ANCHOR_UM2 / 1e6:.3f} mm^2; "
        f"DC/reference={matrix_report['implied_pdk_factor_dc_over_reference']}x"
    )
    for block, report in artifact["report"]["full_chip_blocks"].items():
        print(
            f"[{block}] holdout={report['holdout_rows']}/{report['n_rows']} "
            f"median={report['median_abs_error_pct']}% "
            f"P95={report['p95_abs_error_pct']}%"
        )
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
