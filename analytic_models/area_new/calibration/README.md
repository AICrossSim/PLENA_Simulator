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
