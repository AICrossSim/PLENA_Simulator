# Area calibration inputs

Calibration data for the precision-aware structural area model in
`analytic_models/area/`. The model predicts chip area from a hardware
configuration; the files here are the measurements it is fitted to.

Only aggregate model inputs are kept. The Synopsys build directories, tool logs
and per-run Design Compiler reports that produced them are not part of this
workspace, which limits how the results may be cited — see *Evidence grade*
below.

## Files

- `matrix_machine_mxint.csv`, `matrix_machine_mxfp.csv` — 7 nm Synopsys Design
  Compiler areas, one row per synthesised MatrixMachine shape (MLEN, BLEN,
  precision). These are the fit source for the structural census in `matrix.py`.
  Coverage stops at MLEN = 64; beyond that the census extrapolates by
  construction.
- `matrix_structural_coefficients.json` — the fitted per-unit areas for the
  matrix, vector and scalar blocks, the digital HBM interface and top-level
  integration logic, together with the reference-corner scale and the holdout
  report. This is the runtime artifact the block models load.
- `asap7_sram_macro_table.csv` (with `ASAP7_SRAM_LICENSE`) — ASAP7 single-port
  SRAM macros used by `sram.py`, taken from `The-OpenROAD-Project/asap7_sram_0p0`
  under the BSD 3-Clause licence, which is retained alongside the table.
- `matrix_machine_gate_level_pvt0p7v25c.csv`,
  `matrix_machine_gate_level_activity_envelope.csv` — a **separate, later**
  Design Compiler campaign on `matrix_machine` alone: 8 timing-closed points,
  ASAP7 RVT_TT at `PVT_0P7V_25C`, 1000 ps, MLEN 16–64, BLEN 4–8, MXFP only, plus
  a declared-activity dynamic sweep over the same netlists. **No coefficient is
  fitted to these files.** They are a holdout, read by `gate_level_validation.py`
  and reported in `matrix_gate_level_validation.json`. Figures are one block, in
  µm², at 25 °C — not comparable to full-chip mm² at MLEN 128–1024.
- `matrix_gate_level_validation.json` — the derived cross-validation record for
  the campaign above, including its scope boundary and the decision not to
  refit. Regenerate with `PYTHONPATH=.. python -m area.gate_level_validation`.
- `full_chip_anchors.csv` — 17 full-chip Design Compiler aggregate points up to
  MLEN = 64, including the hierarchy totals used to fit vector, scalar, digital
  HBM-interface and top-integration unit areas. SRAM bitcells are black boxes in
  these totals, so their area is added separately from the ASAP7 macro table.

## Refitting and validating

```bash
cd analytic_models
python -m area.fit
PYTHONPATH=.. python -m pytest area/test_area.py -q
python -m area.calibration_provenance \
  --output results/calibration/area-calibration-audit.json
```

Every fitted family holds back at least 25% of its retained rows before fitting,
and the model is scored on that holdout. The coefficient artifact reports median
and P95 absolute error for each MatrixMachine family, for every full-chip logic
block, and for the whole-chip Design Compiler total.

## Evidence grade

The artifact records its grade as `aggregate_area_tables_without_raw_dc_reports`:
the fitted numbers come from aggregate tables whose underlying per-run synthesis
reports are not retained here. For the same reason the artifact sets
`publication_receipt_complete` to `false`. Both stay as they are until the
original report trees are recovered or synthesis is rerun with
content-addressed receipts. The fit and its holdout results are valid analyses;
what is missing is the ability to replay the individual synthesis runs behind
them.

## Corner and PDK scaling

The structural census is fitted in the Design Compiler synthesis corner, which
runs a uniform factor of about 1.67 above the reference 7 nm OpenROAD predictive
PDK. This is a genuine corner difference, confirmed across the whole compute
hierarchy, not an artefact of the fit. `fit.py` therefore records a single
`pdk_scale_reference` constant so the rescaled model reproduces the known
0.237 mm² figure for the 4×1024 MXINT4 array. Because the constant is a uniform
factor, every relative precision and shape trade-off comes from the Design
Compiler fit and is unaffected by it.
