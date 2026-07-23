# area calibration artifacts

Calibration data for the precision-aware structural area model
(`analytic_models/area/`). Only the files the model reads or refits from are
kept here; Synopsys build dirs, logs, and DC reports stay under `Workspace/`.

## Files

- `matrix_machine_mxint.csv`, `matrix_machine_mxfp.csv` — 7 nm Synopsys DC areas
  per synthesized MatrixMachine shape (MLEN, BLEN, precision). The fit source
  for the structural census (`matrix.py`). Coverage is MLEN <= 64; the census
  extrapolates past it by construction.
- `matrix_structural_coefficients.json` — the fitted per-unit areas
  (`fit.py` output) plus `pdk_scale_reference`; this is the runtime artifact
  `matrix.py` loads.
- `asap7_sram_macro_table.csv` (+ `ASAP7_SRAM_LICENSE`) — ASAP7 single-port SRAM
  macros used by `sram.py`; from `The-OpenROAD-Project/asap7_sram_0p0`, BSD
  3-Clause (license retained).
- `full_chip_anchors.csv` — full-chip DC reference points (MLEN <= 32); kept for
  cross-checking the matrix + SRAM composition, not read at runtime.

## Refit / validate

```bash
cd analytic_models
python -m area.fit          # refit from the CSVs; writes matrix_structural_coefficients.json
python -m area.test_area    # 5 gates: DC holdout, anchor, monotonic, precision, full-chip
```

## Anchor and PDK note

The structural census is fitted in the DC synthesis corner, which runs a uniform
~1.67x above the reference 7 nm OpenROAD predictive PDK. This is a real corner
difference, confirmed across the compute hierarchy, not a fit error. `fit.py`
records one `pdk_scale_reference` constant so the rescaled model reproduces the
known 0.237 mm^2 for the 4x1024 MXINT4 array. All relative precision/shape
trade-offs come from the DC fit and the constant does not affect them.
