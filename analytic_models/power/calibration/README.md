# Power calibration inputs

This directory holds no measurements. Until it does, `power/model.py` raises on
any event signature it is asked to price, so the power model reports structure
rather than calibrated watts.

Populating it is a four-stage process, described in order below: declare the
standard-cell libraries, prepare an immutable schedule of synthesis runs, run
those externally under Design Compiler, then ingest and fit the results. Every
stage is fail-closed — an incomplete or unbound input leaves the artifact
non-rankable rather than producing an optimistic number.

Set these before running any command. Paths are examples; substitute your own.

```bash
export RESULTS=/path/to/results          # where the workflow writes
export RTL_ROOT=/path/to/PLENA_RTL       # sibling checkout
export SOFTWARE_ROOT=/path/to/PLENA_Software
```

## Reproducible DC/SAIF workflow

Create a manifest for the exact characterized `.db` files used by DC:

```bash
PYTHONPATH=analytic_models python -m power.calibration_workflow \
  library-manifest \
  --output $RESULTS/power/library.json \
  --library-id asap7-rvt-tt \
  --process-corner RVT_TT \
  --operating-condition typical \
  --file /pdk/asap7/SIMPLE_RVT_TT.db \
  --file /pdk/asap7/AO_RVT_TT.db \
  --file /pdk/asap7/SEQ_RVT_TT.db \
  --file /pdk/asap7/INVBUF_RVT_TT.db
```

Prepare the complete schedule without invoking DC:

```bash
PYTHONPATH=analytic_models python -m power.calibration_workflow \
  prepare-model \
  --output $RESULTS/power/model \
  --rtl-root $RTL_ROOT \
  --library-manifest $RESULTS/power/library.json \
  --synopsys-environment /tools/synopsys/setup.sh \
  --synopsys-setup /pdk/asap7/calibration_dc.setup \
  --dc-tool-version U-2022.12-SP7
```

The output has 502 immutable point directories: 498 isolated DC runs and
four cycle/latency trace gates. Preparation creates no `DC_SUCCESS` marker and
never labels a row measured. Each dynamic run must receive the real
`inputs/activity.saif` and `inputs/decode_trace.json` named by its
`activity_request.json`. Selector-off/on requests at one geometry list the
same two compatible point IDs and must use byte-identical SAIF and trace
artifacts. Populate the four non-DC inputs exactly as specified by their
`trace_request.json`.

Each `run.sh` verifies the requested DC version, starts DC with `-no_init`,
sources the explicit setup and generated target-library list, uses a private
design library, applies the common 1 ns constraints, and writes the success
marker only after a non-empty mapped netlist is present. The canonical
relative commands are in `workflow_manifest.json`.

After the external DC and trace jobs finish, revalidate and ingest them:

```bash
PYTHONPATH=analytic_models python -m power.calibration_workflow \
  ingest-model \
  --workflow $RESULTS/power/model \
  --measurements $RESULTS/power/model/measurements.csv \
  --artifact-catalog $RESULTS/power/model/artifacts.json \
  --require-complete
```

Ingestion rehashes the RTL census, generated definitions, file lists,
constraints, setup, characterized libraries, SAIF, decode traces, synthesis
logs, and mapped netlists. Missing runs remain `scheduled`; malformed,
unbound, non-positive, non-finite, timing-violating, or context-mismatched
outputs fail closed.

For the final selected candidate, first create an exact request from the
hardware candidate, workload, and compiler precision binding:

```bash
PYTHONPATH=analytic_models python -m power.calibration_workflow exact-request \
  --output $RESULTS/power/exact-request.json \
  --model-name Qwen/Qwen3-32B \
  --model-revision MODEL_COMMIT \
  --candidate-json candidate.json \
  --workload-json cached-q1-workload.json \
  --timing-evidence-id TIMING_ID \
  --layout-id LAYOUT_ID \
  --traffic-ledger-id TRAFFIC_ID \
  --compiler-binding compiler_precision_binding.json

PYTHONPATH=analytic_models python -m power.calibration_workflow prepare-exact \
  --output $RESULTS/power/exact \
  --request $RESULTS/power/exact-request.json \
  --rtl-root $RTL_ROOT \
  --library-manifest $RESULTS/power/library.json \
  --synopsys-environment /tools/synopsys/setup.sh \
  --synopsys-setup /pdk/asap7/calibration_dc.setup \
  --dc-tool-version U-2022.12-SP7
```

Place the full-chip decode SAIF and exact trace in `exact/run/inputs`, run
`exact/run/run.sh`, then build the independently checked anchor index:

```bash
PYTHONPATH=analytic_models python -m power.calibration_workflow ingest-exact \
  --workflow $RESULTS/power/exact \
  --output $RESULTS/power/exact/anchors.json \
  --software-root $SOFTWARE_ROOT
```

The exact path accepts only a compiler-bound, PackedKV-enabled,
steady-state cached-`q_len=1` request. It does not synthesize or infer a
measurement.

## Calibration contract

The immutable schedule can also be generated directly with:

```bash
PYTHONPATH=analytic_models python -m power.calibration_manifest \
  analytic_models/power/calibration/manifest.json \
  --csv-template analytic_models/power/calibration/measurements.csv
```

Populate a CSV with one row per manifest point.

What must be measured separately:

- Matrix activity is recorded independently for `LINEAR`, `QK` and `PV`.
  Combining QK and PV into one row is rejected.
- Vector calibration covers all six searched FP formats plus the BF16
  attribution control.
- The paired selector-off and selector-on rows use identical stimulus, so the
  difference between them yields both selector area and incremental switching
  energy.

Columns required on every row:

`status`, `point_id`, `split`, `component`, `signature`, `MLEN`, `BLEN`,
`events`, `cycles`, `clock_ns`, `dynamic_power_w`, `leakage_power_w`,
`area_mm2`, `dc_tool_version`, `library_id`, `process_corner`, `MX_BLOCK_SIZE`,
`hardware_fp_binding`.

Additional columns required on array, vector and selector rows:

`activity_class`, `saif_sha256`, `decode_trace_sha256`, `saif_source_id`,
`activity_generator`. These bind the power figure to decode-derived `q_len=1`
switching activity, rather than to a vectorless estimate that would ignore what
the circuit actually does during decode.

The current (version 3) schedule contains exactly 502 rows and 80 fitted event
signatures. Artifacts produced against the earlier 508-row vectorless schedule
are not rankable.

The complete synthesized matrix, vector, scalar, SRAM, and leakage
implementation is bound to `FP_E6M5`. `V_FP`, `M_FP`, and `S_FP` change matrix
conversion, accumulation, storage, and control widths, so the six vector rows
alone do not calibrate another FP setting. Another setting needs either a
candidate-exact full-chip anchor or a revised model that separates the
operand-only array core from all FP-dependent periphery and passes full-chip
interaction holdouts.
The two `cycle` rows carry `rtl_cycles` and `emulator_cycles`. The two
`latency` rows carry `measured_latency_s` and `analytical_latency_s`.

Only `status=complete` rows are consumed. Complete rows must match the
generated schedule one-to-one, use a 1 ns constraint, and name one common DC
version, library, process corner, activity generator, and native MX block size
8. The `(64,8)` and `(64,16)` points are
holdouts and cannot be moved into the training split. Duplicate, unexpected,
or missing points leave the fitted artifact non-rankable.

The six `CHIP_LEAKAGE` rows are complete-chip measurements, not leaf leakage.
The four small geometries fit a non-negative
`constant + cells·MLEN·BLEN + perimeter·(MLEN+BLEN)` model. The two larger
geometries are holdouts. Missing anchors or leakage median/max error above
15%/25% blocks energy and every downstream ranking.

Fit the artifact with an independently sourced HBM transfer-energy value:

```bash
PYTHONPATH=analytic_models python -m power.fit \
  measurements.csv calibration.json --artifact-catalog artifacts.json \
  --hbm-energy-pj-per-byte VALUE \
  --hbm-energy-source SOURCE_ID
```

The catalog supplies confined relative paths, byte counts, and SHA-256 hashes
for every raw DC report, SAIF, decode trace, synthesis log, cycle/latency trace,
constraints file, library manifest, RTL source manifest, and tool log. The
fitter rehashes every file and reconciles SAIF/trace hashes with the CSV.
Aggregate values without raw artifacts remain non-rankable.

The fitter validates array, vector, fixed logic, selector deltas,
emulator/RTL cycles, analytical latency, and every required operation
signature before setting `validation.passed`. The source identifier must bind
the HBM energy coefficient to a measurement or cited technology source.

Area ranking reuses the content-addressed structural-census matrix coefficients
and SRAM macro table. The fitter verifies their existing MXINT/MXFP holdouts,
reference anchor, monotonicity, and precision ordering instead of replacing
them with a small-geometry polynomial. Vector and selector area retain fitted
geometry terms. Each estimate carries the full area configuration, physical
HBM bytes, and event counts so the software bridge can independently recompute
area and energy before admitting a point to ranking. Event energy, vector and
selector area, and complete-chip leakage are rankable only inside the validated
`16 <= MLEN <= 64` and `4 <= BLEN <= 16` interpolation envelope. `MLEN=1024`
therefore needs a candidate-exact full-chip DC/SAIF anchor for deployment.

## Execution notes from the first DC campaign

Synthesis runs only where Design Compiler is installed. Two settings there are
not defaults and silently invalidate a run rather than stopping it:

- The ASAP7 operating condition is **`PVT_0P7V_25C`**. There is no condition
  named `typical`; asking for one leaves the run with no corner at all
  (`UID-62`), and the reported numbers are then meaningless.
- The characterized `.db` files use a **picosecond** time unit. Constraints
  written in nanoseconds are rejected (`IFS-001`) and the tool then optimises
  against an unintended target. The one nanosecond calibration clock is written
  as `create_clock -period 1000.0`, with uncertainty 100, input delay 80,
  output delay 50 and max transition 500, all in picoseconds. `report_units`
  confirms ps / fF / 1 V, dynamic power in mW and leakage in pW.

Design Compiler also has no `set_black_box` command; discarding the elaborated
design with `remove_design` leaves each instance an unresolved shell, which is
what a separately measured component needs. Elaboration renames a module built
with more than one parameter set, so a lookup by design name has to match
`name` and `name_<n>` alike. On any error the shell drops to an interactive
prompt and holds the licence, so generated runs read stdin from `/dev/null`.

### Measured so far

`matrix_machine`, ASAP7 RVT_TT at `PVT_0P7V_25C`, 1000 ps, MLEN 16 / BLEN 4.
The compute array is 98.3% of the mapped area, so these figures characterise
the array rather than its wrapper.

| Precision (weight x activation) | Area um2 | um2/PE | Relative | Leakage uW | Critical path ps |
| --- | --- | --- | --- | --- | --- |
| MXFP_E2M1 x MXFP_E2M1 | 20478.3 | 320.0 | 1.000 | 18.7 | 878.3 |
| MXFP_E1M2 x MXFP_E1M2 | 21022.5 | 328.5 | 1.027 | 19.4 | 878.5 |
| MXFP_E4M3 x MXFP_E4M3 | 23548.1 | 367.9 | 1.150 | 21.5 | 878.4 |

Every point closes timing with positive slack, and the critical path is flat
across precisions: the accumulation path, bound to `FP_E6M5`, sets it, not the
operand width. Fitting the four precisions gives

```
area = 18448 + 557 * exp_bits + 978 * mant_bits   um2   (64 PEs)
```

with a worst residual of 0.30%. A mantissa bit costs 1.76 times an exponent
bit, and 78.3% of the array is precision-independent - the accumulation,
alignment and normalisation path that does not shrink when operands do.
Reducing operand precision therefore buys bandwidth first and array area
second, which is what the analytic model should reflect.

Leakage follows area directly at **9.1844e-07 mW/um2** (four points, 1.28%
spread), so leakage at any geometry is `area * density` and needs no switching
activity. An independently fitted value of 9.2576e-07 mW/um2 agrees to 0.79%.

### Geometry scaling, and what the holdout settled

Five geometries were measured for `MXFP_E1M2 x MXFP_E1M2`. The largest,
`64 x 8`, was retained as a holdout: no model was fitted to it, so it tests
extrapolation from the small configurations rather than describing them.

| Geometry | PEs | Area um2 | um2/PE | Split | pJ/MAC at toggle 0.10 |
| --- | --- | --- | --- | --- | --- |
| 16 x 4 | 64 | 21022.5 | 328.48 | train | 0.2417 |
| 32 x 4 | 128 | 43535.9 | 340.12 | train | 0.2526 |
| 16 x 8 | 128 | 38842.9 | 303.46 | train | 0.2189 |
| 32 x 8 | 256 | 81940.8 | 320.08 | train | - |
| 64 x 8 | 512 | 167944.4 | 328.02 | holdout | - |

At an identical 128 processing elements the wide array is 10.78% smaller and
13.34% lower energy per MAC than the tall one. That difference is measured and
reproducible, and it matters when comparing configurations at small `MLEN`.

It does not, however, extrapolate. Predicting the retained `64 x 8` point from
the small configurations alone:

| Model | Predicted um2 | Error |
| --- | --- | --- |
| Constant area per PE, taken from `16 x 4` | 168180.1 | **+0.14%** |
| Three-term fit in `MLEN` and `BLEN` | 164317.1 | -2.16% |
| Linear in PE count through `16 x 4` and `32 x 4` | 178616.5 | +6.35% |

The constant-area-per-PE model wins, and area per PE returns to 328.02 um2 at
`64 x 8` against 328.48 um2 at `16 x 4`. The shape sensitivity is therefore a
small-geometry effect that decays as the array grows, not a scaling law. Treat
`MLEN * BLEN` as the area driver, and treat the shape term as a correction that
applies within the small-geometry envelope only.

Refitting a three-term model over all five geometries reaches a 1.04% worst
residual, but that fit includes the holdout and so measures description rather
than prediction. The validated claim is the simpler one: **area per PE is
constant to within 0.14% across an eightfold change in array size.**

### Dynamic energy

Dynamic figures come from vectorless analysis over a declared activity
envelope, re-opening each mapped netlist rather than resynthesising. Energy per
MAC across the precision ladder at 16 x 4 and toggle 0.10: E2M1 0.2347, E1M2
0.2417, E5M2 0.2684, E4M3 0.2740 pJ. Relative to area, energy tracks the
precision axis to within 1.2% and the shape axis to within 2.6%, so the
precision-energy relationship follows from the measured area law rather than
needing its own calibration.

The envelope spans 0.12 to 1.03 pJ per MAC over toggle rates 0.05 to 0.50. The
analytic decode model's 0.203 pJ per MAC corresponds to a toggle rate of 0.080
at 32 x 4 and 0.083 at 16 x 4 - consistent across geometries and physically
plausible, which is a consistency check on that constant rather than an
independent measurement of decode switching.

### Revised plan for dynamic energy

Annotating switching activity per scheduled point is not the only route, and
not the cheapest. Per-action energy coefficients extracted from a small set of
synthesised configurations, priced against action counts the emulator already
produces, cover the same ground: dynamic energy becomes a sum over actions
rather than a per-point SAIF annotation. Report it with a low/nominal/high
activity envelope and a grouped holdout residual rather than a point estimate,
and state plainly that gate-level validation has not been run.

Until those coefficients exist, dynamic power from these runs is the tool's own
zero-delay toggle propagation (`PWR-414`/`PWR-415` appear in every log) and is
not decode activity. Area, timing and leakage are measured; dynamic power is
not, and the two must not be reported at the same tier.
