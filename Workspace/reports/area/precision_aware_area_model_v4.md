# Precision-Aware PLENA Area Model v4

## Scope

This report documents the current early-stage 7 nm area proxy. The goal is to
rank hardware and software precision choices while retaining a physically
interpretable extrapolation path beyond the MatrixMachine sizes affordable for
Synopsys DC calibration.

The proxy reports square micrometres internally and square millimetres after
division by `1e6`. It is not a placed-and-routed signoff estimate.

## Why the Previous Matrix Regression Was Replaced

The previous direct-width regression achieved low training MAPE by fitting
several highly correlated shape terms. In particular, a `k_splits * mini-array`
term and free `MLEN * BLEN * width` terms could exchange responsibility. The
fit could therefore assign nonzero reduction cost to `MLEN/BLEN=1` and produce
unphysical family crossovers at DSE sizes.

Structural v4 instead mirrors the RTL generate-time census. For `M=MLEN`,
`B=BLEN`, `S=M/B`, and output FP width `F`:

```text
array slices       = S
PEs                = M * B
cross-K nodes      = B^2 * max(S - 1, 0)
output cells       = B^2
result buffer bits = M * B * F
```

The total is the non-negative sum of array stack, cross-K reduction, output
accumulator/conversion, result buffer, IO pipeline and control. When `S=1`,
the cross-K term is exactly zero. MXINT accumulator width includes operand
widths, `ceil(log2(B))`, shift range and split growth. MXFP uses the calibrated
FP accumulator width. Precision capability follows the actual asymmetric PE
mapping: L is ACT; T is the capability union of KV and Weight.

## Calibration Strategy

The model combines three kinds of evidence:

1. Leaf synthesis lookup for supported MXINT and MXFP PE precision tuples.
2. Hierarchy-supervised fits for array, reduction, output, buffer, IO and
   control components from MatrixMachine area reports.
3. Grouped `(MLEN, BLEN)` holdout refits, used both for model selection and for
   P10/P50/P90 regression uncertainty.

All fitted coefficients are non-negative. A complete `(MLEN, BLEN)` group is
assigned to train or holdout together, preventing precision variants of the
same shape from leaking across the split. The legacy free regression remains
available only for diagnostics.

## MatrixMachine Validation

| Family | Train rows | Train MAPE | Grouped holdout rows | Holdout median | Holdout P95 |
|---|---:|---:|---:|---:|---:|
| MXINT | 48 | 3.83% | 48 | 2.92% | 9.78% |
| MXFP | 23 | 0.59% | 23 | 0.79% | 2.36% |

Median hierarchy-component errors are:

| Component | MXINT | MXFP |
|---|---:|---:|
| Array stack | 3.19% | 0.53% |
| Cross-K reduction | 3.67% | 0.46% |
| Output conversion | 7.49% | 0.01% |
| Result buffer | 0.03% | 0.09% |
| IO pipeline | 5.80% | 5.35% |
| Control | 11.40% | 9.81% |

All physical-invariant tests pass: no negative area, exact zero reduction for
one split, monotonic shape growth, monotonic same-family precision growth, and
no calibrated-shape family-order violations (169 MXINT and 61 MXFP comparisons).

## Remaining Logic Models

- VectorMachine uses hierarchy-supervised fitted logic from `vector_machine`
  synthesis across VLEN and FP formats.
- ScalarMachine logic is fitted separately from scalar SRAM. Its absolute area
  is small relative to Matrix/Vector logic.
- HBMSystem estimates only on-chip controller, packing and conversion logic;
  it excludes PHY, channels and HBM stacks.
- A proportional top residual captures top-level interconnect/control, SRAM
  wrapper logic and aggregate module-model bias:

```text
top residual = 0.1009015285 * predicted module logic
```

The residual was fit on 12 full-chip training anchors. Its training logic MAPE
is 2.31% with 4.95% maximum error.

## SRAM Model

RTL memory arrays remain synthesis black boxes. Physical SRAM is added after
logic correction by tiling the open ASAP7 SRAM macro table. Width affects the
number and type of macros, while DSE depth parameters determine capacity.
Matrix, Vector, Scalar-INT and Scalar-FP SRAM are reported separately.

The library preserves replicated single-port macros as its compatibility
default. Current DSE runs explicitly select `ideal-dual-port`, where logical
multi-port SRAM uses one macro copy. This removes replicated area but excludes
physical dual-port decoder, bitline, arbitration, routing, and timing overhead.
It is an architectural assumption, not a new ASAP7 calibration result.

This distinction is essential:

```text
full-chip DC area = synthesized logic + SRAM wrappers, no bitcell macros
composite proxy   = corrected logic + ASAP7 SRAM macro area
```

Comparing the macro proxy directly against a DC wrapper hierarchy is a scope
mismatch, not an SRAM model error.

## Full-Chip Holdout

Five latest full-chip points are held out from the top-residual fit:

| Metric | Result |
|---|---:|
| Raw logic MAPE | 8.86% |
| Corrected logic MAPE | 2.15% |
| Corrected logic maximum error | 5.76% |
| Raw composite MAPE | 6.85% |
| Corrected composite MAPE | 1.77% |

Comparable module-level holdout errors are:

| Module | MAPE | Maximum |
|---|---:|---:|
| MatrixMachine | 4.81% | 8.44% |
| VectorMachine | 4.19% | 4.26% |
| ScalarMachine logic | 0.20% | 0.99% |
| HBMSystem logic | 5.54% | 11.24% |

The fitted top-residual component itself has 35.79% MAPE because it is an
aggregate correction rather than a separately identifiable RTL block. The
useful validation is the corrected total logic error, not per-point residual
attribution.

## DSE Result and Precision Interpretation

The complete 13,905-point Qwen3-32B run uses structural v4. At
`MLEN=VLEN=2048, BLEN=1024`, the fastest all-MXINT point is 646.277 mm2; the
minimum-area tied all-MXFP point is 782.966 mm2. Thus the current RTL and leaf
calibration support an MXINT area advantage. The model does not hard-code that
ordering: it follows PE, accumulator, conversion and buffer implementation.

The selected highest-accuracy minimum-latency profile is 846.531 mm2 nominal
and 854.467 mm2 at P90. It is far outside the DC MatrixMachine calibration
domain (`MLEN=16-64`, `BLEN=4-16`) and is explicitly marked
`structural_extrapolation`.

## Claim Boundary

Supported claims:

- Precision changes compute and SRAM area in the estimator.
- The structural MatrixMachine fit predicts held-out small shapes with the
  errors above and enforces RTL replication invariants at large shapes.
- Small full-chip holdouts validate corrected logic/composite totals.

Unsupported claims:

- `2048/1024` has been synthesized or placed and routed.
- Proxy area equals final die area.
- The ASAP7 macro tiling matches a proprietary memory compiler exactly.
- HBM stack, PHY, package, clock tree, power grid, pads or routing congestion
  are included.

## Reproduction

```bash
python analytic_models/area_new/scripts/fit_matrix_structural_v4.py
python analytic_models/area_new/scripts/fit_full_chip_top_residual.py
pytest analytic_models/area_new -q
```

Canonical artifacts are
`analytic_models/area_new/calibration/matrix_structural_v4_coefficients.json`,
`matrix_structural_v4_validation.json`, and
`full_chip_top_residual_coefficients.json`. Full-chip reports are under
`Workspace/area_new_validation/matrix_structural_v4_20260718/`.
