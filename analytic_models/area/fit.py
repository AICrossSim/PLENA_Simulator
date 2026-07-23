"""Fit the MatrixMachine structural area coefficients from DC synthesis data.

Reads per-shape 7 nm Synopsys DC areas from ``calibration/matrix_machine_*.csv``,
solves for the per-unit areas with non-negative least squares (every structural
term is an additive positive area), and writes
``calibration/matrix_structural_coefficients.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .matrix import FEATURE_NAMES, feature_row

CAL = Path(__file__).with_name("calibration")
OUT = CAL / "matrix_structural_coefficients.json"
REFERENCE_ANCHOR_UM2 = 237_000.0  # 0.237 mm^2: known area of the 4x1024 MXINT4 array


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
            rows.append(dict(MLEN=mlen, BLEN=blen, t=t_w, l=l_w, s=scale, area=float(r["area_um2"])))
        except (KeyError, ValueError):
            continue
    return rows


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[feature_row(d["MLEN"], d["BLEN"], d["t"], d["l"], d["s"])[n]
                   for n in FEATURE_NAMES] for d in rows], float)
    y = np.array([d["area"] for d in rows], float)
    return X, y


def _solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares (physical, positive unit areas)."""
    try:
        from scipy.optimize import nnls
        coef, _ = nnls(X, y, maxiter=10000)
        return coef
    except Exception:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.clip(coef, 0.0, None)


def _mape(coef: np.ndarray, rows: list[dict]) -> float:
    X, y = _design(rows)
    pred = X @ coef
    return float(np.mean(np.abs(pred - y) / y) * 100)


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
    # Holdout check: refit without the largest MLEN, then score against it.
    mlens = sorted({d["MLEN"] for d in rows})
    tr = [d for d in rows if d["MLEN"] < mlens[-1]]
    te = [d for d in rows if d["MLEN"] == mlens[-1]]
    ho_mape = anchor_ho = None
    if tr and te:
        Xtr, ytr = _design(tr)
        cf = _solve(Xtr, ytr)
        ho_mape = _mape(cf, te)
        anchor_ho = _predict(cf, 1024, 4, 4, 4) if mode == "mxint" else None

    report = {
        "in_sample_mape_pct": round(in_mape, 3),
        "holdout_fit_mlen_lt": mlens[-1] if te else None,
        "holdout_mape_pct": round(ho_mape, 3) if ho_mape is not None else None,
        "n_rows": len(rows),
    }
    dc_anchor = None
    if mode == "mxint":
        dc_anchor = _predict(coef, 1024, 4, 4, 4)
        report["dc_anchor_4x1024_mxint4_um2"] = round(dc_anchor, 1)
        report["implied_pdk_factor_dc_over_reference"] = round(dc_anchor / REFERENCE_ANCHOR_UM2, 3)
        if anchor_ho is not None:  # how far the anchor moves when refitted without large MLEN
            report["dc_anchor_from_holdout_um2"] = round(anchor_ho, 1)
            report["anchor_holdout_shift_pct"] = round((anchor_ho - dc_anchor) / dc_anchor * 100, 2)
    return {"coefficients": coeffs, "report": report, "dc_anchor": dc_anchor}


def main() -> None:
    artifact = {"model_version": "matrix_structural_census_v1",
                "reference_anchor_um2": REFERENCE_ANCHOR_UM2,
                "source": "calibration/matrix_machine_{mode}.csv"}
    print("=== MatrixMachine structural-census fit ===")
    dc_anchor = None
    for mode in ("mxint", "mxfp"):
        fitted = fit_mode(mode)
        if not fitted:
            print(f"[{mode}] no data — skipped")
            continue
        artifact[mode] = fitted["coefficients"]
        artifact.setdefault("report", {})[mode] = fitted["report"]
        if fitted.get("dc_anchor"):
            dc_anchor = fitted["dc_anchor"]
        r = fitted["report"]
        print(f"\n[{mode}] n={r['n_rows']}  in-sample MAPE={r['in_sample_mape_pct']}%  "
              f"holdout(fit MLEN<{r['holdout_fit_mlen_lt']}) MAPE={r['holdout_mape_pct']}%")
        if mode == "mxint":
            print(f"       DC-corner anchor 4x1024 MXINT4: {r['dc_anchor_4x1024_mxint4_um2']} um^2 "
                  f"= {r['dc_anchor_4x1024_mxint4_um2']/1e6:.4f} mm^2")
            print(f"       vs reference 0.237 mm^2 (OpenROAD PDK): DC/reference = "
                  f"{r['implied_pdk_factor_dc_over_reference']}x (corner factor, not error); "
                  f"anchor shifts {r.get('anchor_holdout_shift_pct')}% under holdout")
    # One conversion factor moves DC-corner areas onto the reference PDK.
    if dc_anchor:
        artifact["pdk_scale_reference"] = REFERENCE_ANCHOR_UM2 / dc_anchor
        print(f"\n  pdk_scale_reference = {artifact['pdk_scale_reference']:.4f}  "
              f"(DC corner -> reference PDK; pins 4x1024 MXINT4 to 0.237 mm^2)")
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
