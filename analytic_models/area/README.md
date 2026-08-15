# Precision-aware structural area model

This package predicts PLENA silicon area from a hardware configuration. It is a
*structural census*: each block counts the RTL structures its configuration
actually replicates — systolic slices, PE cells, reduction and prefix edges,
lane endpoints, adapter bits, counters — and multiplies those counts by per-unit
areas fitted to 7 nm Synopsys Design Compiler results. It is deliberately not a
polynomial regression of total area against shape, because the calibration grid
stops at MLEN = 64 and the decode study evaluates geometries up to MLEN = 4096;
a shape polynomial fitted on small arrays diverges there, while a census with
small fitted per-unit areas does not.

The fit inputs, their evidence grade, and the refit procedure are documented in
[`calibration/README.md`](calibration/README.md). This document describes the
model those inputs drive, and what each of its numbers may and may not be used
to claim.

## What the two entry points return

```python
from analytic_models.area import estimate_area, estimate_system_area
```

- `estimate_area(config, *, corner="reference", coefficients_path=None,
  macro_table_path=None)` returns one **die**: the compute hierarchy plus its
  on-die SRAM. Its breakdown is exactly nine blocks — `MatrixMachine`,
  `VectorMachine`, `ScalarMachine`, `HBMInterface`, `TopOverhead`, `MatrixSRAM`,
  `VectorSRAM`, `ScalarIntSRAM`, `ScalarFPSRAM` — and `test_area.py` pins that
  the nine sum to the reported total. No missing block is silently replaced with
  zero; an unsupported option raises rather than defaulting.
- `estimate_system_area(config, *, chip_count, ports_per_chip=0,
  link_bandwidth_gbps=7200.0, hbm_interface_units_per_chip=0,
  target_node_nm=7.0, **area_options)` returns a **system**: `chip_count` dies,
  each charged its own chip-side HBM PHY and `ports_per_chip` chip-to-chip link
  PHYs. Its breakdown is `DecodeChips`, `HBMPhys`, `LinkPHYs`.

`chip_area` in the system result is the per-die total *including* that die's HBM
PHY, so the system total minus `chip_count × chip_area` is exactly the link
silicon.

Note that `top.py` is the top-level *integration residual* block model, not the
system-level entry point; `estimate_system_area` lives in `__init__.py`.

## Blocks and what each counts

| Module | Block | Census features |
| --- | --- | --- |
| `matrix.py` | `MatrixMachine` | `MLEN/BLEN` systolic slices of `BLEN×BLEN` PEs: multiplier cells `MLEN·BLEN·t·l`, adder terms `MLEN·BLEN·(t+l)`, PE count, cross-K reduce edges, scale distribution, output cells |
| `vector.py` | `VectorMachine` | lanes, reduction edges (`VLEN−1`), prefix edges (`VLEN·⌈log₂VLEN⌉`), mask heads (`VLEN/HLEN`), fixed control |
| `scalar.py` | `ScalarMachine` | FP vector-buffer bits, vector counter bits, fixed control |
| `hbm_interface.py` | `HBMInterface` | *digital* controller logic per `memory/HBM/rtl/hbm_sys.sv`: lane endpoints, precision-width adapter bits, optional MXFP converter lanes, block-scale buffer bits, transfer counters |
| `top.py` | `TopOverhead` | routing-tree edges, wide routing bit-edges, MXFP routing lanes, slice controls, fixed control — fitted as the full-chip total minus the four hierarchy blocks |
| `sram.py` | four SRAM blocks | minimum-area legal tiling of published ASAP7 single-port macros |
| `link.py` | `LinkPHYs` | literature density, node-scaled |
| `hbm_phy.py` | `HBMPhys` | declared shoreline × beachfront depth per interface unit |

`precision.py` normalises the software precision knobs into RTL operand sides
and fails early — unsupported widths and mixed MXINT/MXFP profiles raise
`PrecisionError` rather than silently selecting an uncalibrated model.
`geometry.py` inverts the model: given an area budget and candidate MLEN/BLEN/
VLEN sets it returns the maximum-compute legal geometry that fits.

Two block-level responses are counter-intuitive and are locked in by tests, so
they are model statements rather than accidents:

- `HBMInterface` area *increases* as precision narrows, because the
  narrow-to-native adapters widen (`test_hbm_interface_has_explicit_precision_response`).
- A two-port SRAM is modelled as two full copies of a single-port ASAP7 macro,
  a 2× area charge, not as a dual-port bitcell.

## Calibration domain and the extrapolation flag

Every calibrated block declares a `CALIBRATION_DOMAIN`:

| Block | Domain |
| --- | --- |
| matrix | MLEN ∈ {16, 32, 64}, BLEN ∈ {4, 8, 16}, families {mxint, mxfp} |
| vector | VLEN ∈ {16, 32, 64}, HLEN ∈ {8, 32, 64}, `FP_SETTING` = FP_E5M6 |
| scalar | MLEN/VLEN ∈ {16, 32, 64}, `INT_DATA_WIDTH` = 32, `FP_SETTING` = FP_E5M6 |
| hbm_interface | as scalar, plus uniform precision and prefetch amount 16 |
| top | as scalar, plus BLEN ∈ {4, 8, 16} and uniform precision |

A configuration outside its block's domain does not fail; it is **flagged**. The
block evidence record carries `structural_extrapolation: true` and its tier
degrades from `dc_synthesized_aggregate_fit` to
`dc_synthesized_aggregate_structural_extrapolation`. `weakest_tier` then
collapses the block tiers into one chip tier, so a single extrapolated block
degrades the whole estimate.

**This matters for how the results are read.** The decode study's geometries
(MLEN = 1024 and above) are all outside the calibrated grid, so every headline
chip area carries the extrapolation tier. The holdout errors below are
in-domain figures — they characterise the census method at MLEN ≤ 64 and are not
an error bar on a 1024-wide array.

Opt-in RTL structures that no retained DC point contains — the segmented vector
reduction (`REDUCTION_SEGMENTS`), the scalar FP issue pipeline
(`SCALAR_FP_ISSUE_PIPELINE`), the loop address generator
(`ENABLE_LOOP_ADDRESS_GENERATOR`) — are priced by a structural census against
the closest retained per-bit coefficient, tagged
`extrapolated_from_closest_retained_per_bit_fit`, and they demote the enclosing
block to `declared_structural_estimate`. They are never free: `test_area.py`
asserts the chip total rises by exactly the extension area when each is enabled.

## Holdout report

Every fitted family reserves every fourth retained row before fitting and is
scored on that holdout. From `calibration/matrix_structural_coefficients.json`:

| Family | rows | holdout | median % | P95 % |
| --- | --- | --- | --- | --- |
| matrix (MXINT) | 48 | 12 | 0.85 | 4.61 |
| matrix (MXFP) | 23 | 6 | 2.64 | 3.38 |
| vector | 17 | 5 | 0.003 | 0.006 |
| scalar | 17 | 5 | 0.0 | 0.078 |
| hbm_interface | 17 | 5 | 0.064 | 0.284 |
| top | 17 | 5 | 4.33 | 5.84 |
| full chip | 17 | 5 | 0.565 | 0.968 |

Two limits of this table should be stated wherever it is quoted. The full-chip
row records `includes_sram_bitcell_macros: false`, so the ASAP7 tiling — the
majority of the reference chip's area — is outside every holdout figure and has
no error gate beyond additivity. And the reported errors come from a
training-only refit, while the shipped coefficients are solved over all rows:
the report scores the method, not the exact artifact.

The 4×1024 MXINT4 array reproducing 0.237 mm² is likewise not independent
evidence. `pdk_scale_reference` is the single uniform constant chosen so that it
does; what the Design Compiler fit supplies is every *relative* precision and
shape trade-off, which the constant leaves untouched.

`calibration_provenance.py` pins the SHA-256 and row count of all six
calibration files and runs five gates (holdout coverage and P95, the 0.237 mm²
anchor, shape monotonicity, precision monotonicity, full-chip additivity). Any
byte-level change to a calibration file — including a refit — trips the hash
check, so `python -m area.fit` and the audit are coupled by design. The audit
records the grade `aggregate_area_tables_without_raw_dc_reports` and explicitly
lists `raw DC report provenance is complete` among its unsupported claims.

## Independent gate-level cross-validation

A later Design Compiler campaign synthesised `matrix_machine` on its own and was
never used to fit anything here, so it is a true holdout for the MXFP census.
Eight timing-closed points, ASAP7 RVT_TT at `PVT_0P7V_25C` with a 1000 ps clock,
MLEN ∈ {16, 32, 64} and BLEN ∈ {4, 8}, four MXFP formats. The measured tables are
vendored as `calibration/matrix_machine_gate_level_pvt0p7v25c.csv` and
`calibration/matrix_machine_gate_level_activity_envelope.csv`;
`gate_level_validation.py` derives the comparison and writes
`calibration/matrix_gate_level_validation.json`.

**Scope, before any number is read.** Every figure below is *one block*, in
**µm²**, over **MLEN 16–64**, at **25 °C**. The decode study's headline areas are
*full-chip*, in **mm²**, at **MLEN 128–1024** and above. These are not the same
quantity and this campaign is not an error bar on those estimates; the artifact
carries the boundary in its `scope` record so it travels with any consumer.

What the campaign establishes on its own:

| Law | Measured | Residual |
| --- | --- | --- |
| Precision | `area = 18448 + 557·exp_bits + 978·mant_bits` µm² at 16×4 | 0.30% worst |
| Geometry | area per PE constant: 328.48 µm² at 16×4, 328.02 µm² at 64×8 | +0.14% at 8× extrapolation |
| Leakage | `area × 9.2097e-07` mW/µm² over all eight points | 1.44% spread |

A mantissa bit costs **1.76×** an exponent bit. The precision-independent
constant is 78.3% of the widest measured format's area and 90.1% of the
narrowest — the accumulation and normalisation path does not shrink when
operands do. Both ends are reported because the single figure depends entirely
on which format it is quoted against.

What it says about the shipped census, evaluated at `corner="dc"` on all eight
points:

- The census **over-predicts by a uniform 1.1242×**, spanning 1.108–1.158.
- Removing that single offset leaves **0.41% median and 2.97% worst** error
  across every measured shape and precision.
- The offset is *not* a model defect. At the six geometries the two campaigns
  share, the shipped calibration CSV is itself **1.113×** the new campaign
  (1.098–1.126). The disagreement is a level difference between two synthesis
  campaigns — different setup, corner and RTL revision — and the census already
  carries a uniform level constant (`pdk_scale_reference`) that absorbs exactly
  this kind of difference without touching a relative trade-off.

One genuine model limit surfaces, and it is recorded rather than papered over:
the census features depend on **total operand width only**, so `MXFP_E1M2` and
`MXFP_E2M1` — both 4 bits wide — are predicted identically. The campaign
measures them **2.66% apart**, which is where the 2.97% worst residual comes
from. Representing it would need a feature that separates exponent from mantissa
bits, which the current census does not have.

**The shipped coefficients are unchanged, deliberately.** Refitting on eight
small-geometry MXFP points would trade a 71-row fit spanning both numeric
families for a narrower one, would import a level offset of unclear provenance,
and would not improve a single relative trade-off — the campaign confirms those
to within 3%. The right upgrade is a campaign that re-measures the *full chip*
at the *decode geometries*, not a refit onto a smaller block.

The same campaign priced two mapped netlists over a declared toggle envelope of
0.05–0.50, giving 0.113–1.126 pJ/MAC. The analytic decode model's
`REFERENCE_MAC_ENERGY_PJ = 0.203` falls inside that envelope at an implied
toggle of 0.0797 (32×4) and 0.0835 (16×4), and 0.073–0.093 across all six swept
geometries. That is an independent corroboration of the model's most
load-bearing energy coefficient — but it is a **declared-activity estimate**:
the toggle rate is assumed and propagated by the synthesis tool, not measured
from decode switching. It brackets the anchor; it does not calibrate it, and the
energy tier is unchanged. `analytic_models/disagg_serve/decode_power.py` records
it as `COMPUTE_ENERGY_CROSS_CHECK`.

Refresh the artifact with:

```bash
cd analytic_models
PYTHONPATH=.. python -m area.gate_level_validation
```

## ASAP7 SRAM macro tiling

`sram.py` reads `calibration/asap7_sram_macro_table.csv` — 36 single-port
`srambank_*_6t122` macros from `The-OpenROAD-Project/asap7_sram_0p0` (BSD
3-Clause, licence vendored alongside), spanning depths {256, 512, 1024} and
widths from 16 to 80 bits, with per-bit areas between 0.0352 and 0.0608 µm².

For each required memory the model takes the minimum over macros of
`⌈depth/macro_depth⌉ × ⌈width/macro_width⌉ × ports × macro_area`. Four memories
are built from the configuration: `MatrixSRAM` and `VectorSRAM` (two ports each),
`ScalarIntSRAM` and `ScalarFPSRAM` (one port each). The evidence tier is
`published_sram_macro_geometry` with `foundry_compiler_result: false` — this is
a DSE floorplanning proxy, not an SRAM compiler run.

The table can be overridden by `macro_table_path=` or the environment variable
`PLENA_AREA_SRAM_MACRO_TABLE`; the reader is `lru_cache`d, so an edited table is
not picked up inside a live process.

## Chip-to-chip link PHY

`link.py` projects the C2C link PHY from a published 5 nm density of
552 Gb/s/mm² (Gangasani et al., ISSCC 2022, doi:10.1109/ISSCC42614.2022.9731636),
scaled to the target node by the square of feature size. At 7 nm a bidirectional
900 GB/s-class port is 25.565 mm². The evidence tier is
`published_density_node_scaled` with `synthesized_for_plena: false`: a declared
analytic projection, not PLENA synthesis data.

## Chip-side HBM PHY

`hbm_phy.py` charges the analog HBM PHY, I/O and beachfront silicon that
`hbm_interface.py` deliberately excludes. The anchor is 11.0 mm² of chip-side
silicon per 1024-bit stack interface at 7 nm, formed from a published die-edge
(shoreline) occupancy of roughly 11 mm per stack interface (Chen et al., EMCSI
2022, doi:10.1109/EMCSI39492.2022.10050234) and a **declared** 1.0 mm beachfront
macro depth. That figure is divided evenly over the sixteen 64-bit interface
units of a stack, giving `AREA_MM2_PER_INTERFACE_UNIT = 0.6875` mm², and area is
strictly linear in the attached interface-unit count. Node scaling is applied to
the whole area with an exponent of 2.0
(`node_area_scaling_exponent` in the evidence record).

The evidence tier is `declared_structural_estimate` with
`synthesized_for_plena: false`. The beachfront depth is the declared component;
replacing it with a synthesized or vendor PHY floorplan upgrades the tier with
no caller change.

This block is what makes attached memory bandwidth a genuine area trade-off in
the decode study rather than a free configuration choice: HBM channels are a
searched axis, and each additional interface unit is charged 0.6875 mm² of
silicon on every chip. `test_area.py` pins the per-unit charge, its appearance
in `chip_area` and in the `HBMPhys` breakdown entry, and that the system total
increases strictly with interface-unit count.

**`hbm_interface_units_per_chip` defaults to 0.** A caller that omits it gets a
system estimate with no HBM PHY charged at all, and — because the HBM PHY
evidence record is only appended when the area is non-zero — no record saying so.
Callers pricing a real attached-memory configuration must pass the interface-unit
count explicitly. The decode study threads it through
`analytic_models/disagg_serve/area.py` and
`analytic_models/performance/disagg_decode.py::system_area`.

## Consumers

`analytic_models/disagg_serve/area.py` is the calibrated-area bridge used by the
decode study; `analytic_models/performance/disagg_decode.py::system_area` adds
the architecture-option blocks (KV head reuse control, drain-overlap accumulator
bank) on top of the system result and propagates the weakest tier. `test_area.py`
carries bridge tests over both, so a change here that breaks the decode study's
area accounting fails in this package's own suite.
