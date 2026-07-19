# area_new calibration artifacts

This directory stores compact calibration CSV/JSON outputs for the
precision-aware area proxy. Synopsys build directories, external repository
checkouts, and worker copies are intentionally not stored here.

Expected generated artifacts:

- `pe_mxint.csv`
- `pe_mxfp.csv`
- `mini_array_mxint.csv`
- `mini_array_mxfp.csv`
- `matrix_machine_mxint.csv`
- `matrix_machine_mxfp.csv`
- `matrix_structural_leaf_points.csv`
- `matrix_structural_v4_coefficients.json`
- `matrix_structural_v4_diagnostics.csv`
- `matrix_structural_v4_grouped_holdout.csv`
- `matrix_structural_v4_validation.json`
- `mxint_model_coefficients.json`
- `mxfp_model_coefficients.json`
- `vector_machine.csv`
- `vector_model_coefficients.json`
- `scalar_machine.csv`
- `scalar_model_coefficients.json`
- `hbm_system.csv`
- `hbm_model_coefficients.json`
- `full_chip_anchors.csv`
- `full_chip_validation_split.json`
- `full_chip_top_residual_coefficients.json`

Raw `calibration_points.csv` files, failed attempts, command logs, and DC
reports stay under `Workspace/`. The files here contain only the latest
successful point for each complete semantic configuration. Scalar rows without
an explicit `MLEN/VLEN` are excluded because their physical shape is ambiguous.

The five IDs in `full_chip_validation_split.json` are held out from the
chip-level residual fit. Reproduce the validation report with:

```bash
python analytic_models/area_new/scripts/validate_full_chip_area_proxy.py
```

The full-chip DC target contains logic and SRAM wrapper/conversion logic, but
not SRAM bitcell macros. The fitted top residual therefore scales only the
predicted logic subtotal. ASAP7 SRAM macro area is added afterwards.

SRAM area is estimated from ASAP7 SRAM macro collateral rather than by
synthesizing behavioral SRAM arrays. The compact macro table is generated from
`The-OpenROAD-Project/asap7_sram_0p0` LIB/LEF files:

- `asap7_sram_macro_table.csv`

The table currently records upstream revision
`9f5af0939e8dd3cc1a9693a50b23441691dd7d25`, distributed under the BSD
3-Clause license retained in `ASAP7_SRAM_LICENSE`. It is calibration
collateral, not a foundry signoff macro model.

Regenerate it with:

```bash
python analytic_models/area_new/scripts/build_asap7_sram_macro_table.py \
  --source Workspace/external/asap7_sram_0p0 \
  --out analytic_models/area_new/calibration/asap7_sram_macro_table.csv
```

`run_sram_calibration.py` is retained only as an explicit debug fallback for
register-array experiments and refuses to launch DC unless
`--allow-register-array-synth` is passed.

## MatrixMachine structural-v4

The default MatrixMachine model is a physical RTL census rather than a direct
polynomial fit. For `M=MLEN`, `B=BLEN`, and `S=M/B`, it fixes the following
instance counts before fitting any area coefficient:

```text
PEs              = M * B
array slices     = S
cross-K nodes    = B^2 * max(S - 1, 0)
output cells     = B^2
nominal FP bits  = M * B * FP_WIDTH
```

Consequently, a single-split `BLEN=MLEN` design has exactly zero cross-K
reduction area. MXINT and MXFP keep separate PE lookups and separate result
buffer terms because their current RTL hierarchies are not identical.

The additional leaf anchors can be reproduced with:

```bash
nix develop --command bash -lc 'source .venv/bin/activate; \
  python analytic_models/area_new/scripts/run_matrix_machine_calibration.py \
    --mode structural-v4-leaves --workers 4 \
    --run-dir Workspace/area_new_matrix_calibration/runs/structural_v4_leaves \
    --resume --cleanup-worker-builds'
```

The three requested MXFP `BLOCK_DIM=32` array anchors are not synthesized as
monolithic 1024-PE tops.  DC spends hours duplicating the same parameterized
PE hierarchy even though the mini-array periphery contains only scale shift
registers, valid reduction, and wiring.  The runner therefore synthesizes the
real periphery wrapper and composes its area with `BLOCK_DIM^2` copies of the
matching synthesized PE leaf.  This is an exact RTL census, not an inferred
large-MatrixMachine regression.  It was checked against monolithic RTL:

```text
BLOCK_DIM=8:  actual 16,738.831 um2, composed 16,745.290 um2, error +0.0386%
BLOCK_DIM=16: actual 66,919.167 um2, composed 66,935.452 um2, error +0.0243%
```

The resulting `BLOCK_DIM=32` anchors complete in about 15 seconds each while
retaining the same physical composition.  Their raw periphery area and PE
lookup area are both retained in `matrix_structural_leaf_points.csv`.

The runner requires `10 GiB + 6 GiB * workers` free in `/tmp`. If needed it
removes only inactive, user-owned PLENA calibration trees, then lowers the
worker count; it refuses to start below 15 GiB. Refit the model with:

```bash
python analytic_models/area_new/scripts/fit_matrix_structural_v4.py
python analytic_models/area_new/scripts/fit_full_chip_top_residual.py
```

Holdouts are grouped by `(MLEN, BLEN)`. P10/P50/P90 combine the grouped-fit
coefficient ensemble with the held-out actual/predicted residual distribution;
the latter prevents a numerically stable coefficient fit from reporting an
unrealistically narrow interval during large-shape extrapolation. These are
model uncertainty bounds, not process corners. Large DSE designs remain
structural extrapolations, not synthesized anchors.
