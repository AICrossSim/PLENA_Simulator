# Power calibration artifacts

This directory separates the model used by the DSE from calibration evidence.

- `logic_energy_v2.json` is the active DSE shadow artifact. It passed every v2
  RTL-activity/mapped-DC promotion gate and is labelled
  `rtl_activity_calibrated_candidate_v2`.
- `logic_energy_v1.json` is the preserved compatibility bootstrap. It is no
  longer loaded by default.
- The rejected v1 fitted candidate and its verbose validation output are not
  release artifacts. Their conclusion is retained in the power report; the
  runtime keeps only the v1 bootstrap for compatibility.
- `power_calibration_points.csv` contains the latest complete version of all
  144 semantic RTL-activity replay points. The append-only attempt history is
  retained in the corresponding Workspace run directory.
- `sram_energy_asap7_v1.json` is the separately extracted ASAP7 SRAM macro
  dynamic-energy table.
- `sram_background_memexplorer_v1.json` supplies the selected 10 W/GB lower
  endpoint from MemExplorer Table 1. It is applied to allocated macro capacity
  after tiling and is explicitly a literature background-power proxy, not an
  ASAP7 Liberty leakage characterization.
- `external_memory_hbm3e_v1.json` is a literature-parameterized external-memory
  artifact. It uses 50/75/100 mW/GB background power and 3.0/3.6 pJ/bit
  read/write energy. Its default comparison configuration is the abstract
  80 GB, 2039 GB/s A100-aligned interface; this is not an integer HBM3E stack
  topology and is not a PLENA calibration result.

The v2 flow writes a run-local candidate before promotion. The release keeps
the promoted `logic_energy_v2.json`, `power_validation_v2.json`, compact
Markdown validation, and activity envelope; it does not duplicate the
byte-identical candidate. Future failed refits cannot overwrite the promoted
artifact. Nominal dynamic energy is fitted only from Qwen-like activity.
Low-toggle and random activity define a per-family empirical envelope rather
than contaminating the P50 fit. Idle clock energy is normalized by the mapped
component area supplied by `area_new`.

The accepted dataset contains 31 mapped configurations and 395 successful
activity replays. Grouped holdout median/P95 errors are 7.47%/20.88%; idle
clock median/P95 errors are 3.83%/8.34%. Component median/P95 errors are:

```text
Matrix  6.10% / 11.69%
Vector  7.89% / 20.93%
Scalar  0.90% /  8.69%
HBM     7.82% / 21.65%
```

The HBM points identify per-opcode energy per accepted logical lane. They do
not independently identify physical-line and useful-byte coefficients because
the available points use fixed transfer amounts; the runtime report preserves
this limitation as a warning.

V2 also preserves DC power-group decomposition. Its action target is the
active-minus-idle non-clock residual, while the continuously clocked baseline
comes from matched idle total dynamic energy. This avoids treating
activity-dependent variation in DC's pre-CTS `clock_network` category as
negative datapath energy. Raw total energy and the excluded clock-group delta
remain calibration evidence.

The on-chip evidence is RTL VCD activity replayed on mapped DC netlists at the
declared ASAP7 corner. Gate-level simulation, CTS, routed parasitics, package
power, and intrinsic SRAM leakage characterization are outside this calibration
scope. External HBM3E
energy is a separate system-level literature estimate driven by V4 physical
line traffic; it must not be described as measured or signoff HBM power.

## Loop AGU

`agu_energy_v1.json` contains the dedicated six-stream loop-AGU action-energy
fit. Its 13 source rows are retained in
`agu_power_calibration_points.csv`. The scenarios isolate setup, 1/3/6-stream
boundaries, affine-offset reads, and low/Qwen/random activity. Every replay
achieved 100% sequential SAIF coverage, and the six-stream 32/128/512 action
slope has R-squared greater than 0.9999999.

The artifact is valid for the 32-bit, six-stream, four-level AGU at ASAP7 TT,
0.7 V, 25 C, and a 1 ns reporting window. Like the main v2 model, it uses RTL
activity on a mapped netlist and remains a calibrated candidate rather than
gate-level or post-route evidence.
