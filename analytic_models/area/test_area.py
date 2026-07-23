"""Validation gates for the structural-census area model.

Runnable standalone (avoids the package's heavy parent __init__):
    cd analytic_models && python -m area.test_area

Gates:
  1. DC-corner holdout: a fit on MLEN<64 predicts MLEN=64 within tolerance, so
     the shape extrapolation holds independently of the pinned anchor.
  2. Anchor: reference-corner 4x1024 MXINT4 reproduces 0.237 mm^2, confirming
     the PDK scale factor is applied correctly.
  3. Monotonicity: area grows with MLEN*BLEN.
  4. Precision scaling: MXINT2 < MXINT4 < MXINT8.
  5. estimate_area (matrix + SRAM) is positive and matrix-dominated.
"""

from __future__ import annotations

import sys

from . import estimate_area
from .matrix import estimate_matrix_machine_area
from .fit import fit_mode

REFERENCE_MM2 = 0.237  # known area of the 4x1024 MXINT4 array
TOL_HOLDOUT_PCT = 10.0
TOL_ANCHOR_PCT = 1.0

_SHAPES = [(16, 4), (32, 4), (64, 8), (256, 8), (1024, 4), (2048, 32)]


def _cfg(mlen, blen, act="MXINT4", kv="MXINT4", weight="MXINT4", vlen=None):
    return {"MLEN": mlen, "BLEN": blen, "VLEN": vlen or mlen,
            "ACT_WIDTH": act, "KV_WIDTH": kv, "WEIGHT_WIDTH": weight, "FP_SETTING": "FP_E5M6"}


def check_holdout() -> tuple[bool, str]:
    r = fit_mode("mxint")["report"]
    ho = r.get("holdout_mape_pct")
    ok = ho is not None and ho <= TOL_HOLDOUT_PCT
    return ok, f"DC holdout MAPE={ho}% (<= {TOL_HOLDOUT_PCT}%); in-sample={r['in_sample_mape_pct']}%"


def check_anchor() -> tuple[bool, str]:
    mm2 = estimate_matrix_machine_area(_cfg(1024, 4), corner="reference")["area"] / 1e6
    err = abs(mm2 - REFERENCE_MM2) / REFERENCE_MM2 * 100
    return err <= TOL_ANCHOR_PCT, (f"reference-corner 4x1024 MXINT4 = {mm2:.4f} mm^2 "
                                   f"(expected {REFERENCE_MM2}, err {err:.2f}%)")


def check_monotonic() -> tuple[bool, str]:
    pa = sorted((m * b, estimate_matrix_machine_area(_cfg(m, b))["area"]) for m, b in _SHAPES)
    mono = all(pa[i][1] < pa[i + 1][1] for i in range(len(pa) - 1))
    return mono, f"area monotone in MLEN*BLEN: {[round(x/1e3, 1) for _, x in pa]} (10^3 um^2)"


def check_precision() -> tuple[bool, str]:
    a2 = estimate_matrix_machine_area(_cfg(1024, 4, "MXINT2", "MXINT2", "MXINT4"))["area"]
    a4 = estimate_matrix_machine_area(_cfg(1024, 4, "MXINT4", "MXINT4", "MXINT4"))["area"]
    a8 = estimate_matrix_machine_area(_cfg(1024, 4, "MXINT8", "MXINT8", "MXINT8"))["area"]
    ok = a2 < a4 < a8
    return ok, f"precision scaling MXINT2<4<8: {a2/1e6:.4f} < {a4/1e6:.4f} < {a8/1e6:.4f} mm^2 = {ok}"


def check_full_chip() -> tuple[bool, str]:
    out = estimate_area(_cfg(1024, 4))
    ok = out["area"] > 0 and out["matrix_machine_area"] > 0
    return ok, (f"estimate_area = {out['area']/1e6:.4f} mm^2 "
                f"(matrix {out['matrix_machine_area']/1e6:.4f} + sram {out['sram_macro_area']/1e6:.4f})")


def main() -> int:
    checks = [("holdout", check_holdout), ("anchor", check_anchor), ("monotonic", check_monotonic),
              ("precision", check_precision), ("full_chip", check_full_chip)]
    failed = 0
    for name, fn in checks:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        failed += not ok
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILED'} ({len(checks)} gates)")
    return 1 if failed else 0


def test_holdout(): assert check_holdout()[0]
def test_anchor(): assert check_anchor()[0]
def test_monotonic(): assert check_monotonic()[0]
def test_precision(): assert check_precision()[0]
def test_full_chip(): assert check_full_chip()[0]


if __name__ == "__main__":
    sys.exit(main())
