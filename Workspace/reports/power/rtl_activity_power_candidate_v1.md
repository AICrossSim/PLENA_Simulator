# PLENA RTL-Activity On-Chip Power Candidate v1

> **Rejected calibration candidate, audited 2026-07-26.** This report is
> retained to document the first RTL-activity experiment and its failure
> modes. It is not loaded by the current DSE. The accepted on-chip component is
> documented in
> [`rtl_activity_power_candidate_v2.md`](rtl_activity_power_candidate_v2.md).

## Purpose and claim boundary

This work adds a fixed-corner, pre-layout on-chip power candidate for PLENA.
The model estimates action-dependent logic energy, ASAP7 SRAM macro dynamic
energy, and mapped-logic leakage at ASAP7 TT, 0.7 V, 25 C, with a 1000 ps
clock assumption. It excludes external HBM, HBM PHY, package, a KV handoff
link, CTS, routed parasitics, and SRAM leakage.

The candidate is intentionally a DSE shadow metric. RTL activity is replayed
on mapped Design Compiler netlists, but gate-level simulation is not part of
this calibration scope. A successful run is therefore labelled
`rtl_activity_calibrated_candidate`, not signoff or gate-level validated.

## Why the model uses hardware actions

Fitting one power value per opcode would confound opcode count, utilization,
precision, vector width, and idle clocking. Instead, compiler lowering emits
structured `EnergyAction` records. The model combines reusable physical
families:

- Matrix active MAC bit-products, output conversion, and PE clock work.
- Vector ALU, multiply, SFU, reduction, movement, and lane clock work.
- Scalar FP/INT families, register activity, and ScalarMachine clock work.
- Frontend issue activity and HBM-controller DMA/line/byte activity.
- SRAM macro reads and writes from public ASAP7 Liberty internal-power tables.

For each mapped configuration, active energy is estimated from a matched idle
window:

```text
incremental_energy(action) = active_window_energy - matched_idle_window_energy
```

The random 32/128/512 action windows expose the action slope, while idle
windows expose continuously clocked baseline energy. Nonnegative regression
preserves physical sign constraints.

## Calibration pipeline

The runner separates three resource pools:

```text
normal mapped synthesis -> Verilator RTL activity -> DC SAIF power replay
```

Sixteen small configurations are mapped once. Each configuration compiles one
Verilator binary, which is reused for nine scenarios: matched idle/random
windows at 32, 128, and 512 actions; low-toggle; representative Qwen; and a
mixed-kernel holdout. This produces 144 independent power replay jobs without
repeating synthesis.

DC creates an RTL-to-mapped SAIF name map during synthesis. Verilator represents
packed SystemVerilog members as nested scopes, for example
`decode_stage_op/m_op[3]`, while DC records names such as
`decode_stage_op[m_op][3]`. The replay flow mechanically translates only the
source side of DC's exact map and reloads it with `saif_map -read_map`. It does
not invent switching activity. Quantitative mapped sequential-cell coverage
must be at least 90%; PWR-414 remains fatal.

## Resource and storage safety

Before calibration, a manifest-based cleanup removed only known, inactive
PLENA `/tmp` artifacts. Unknown paths, other users' files,
`/tmp/moe_models_e13`, and `/tmp/bc625` were preserved. The cleanup removed
44 paths totalling 18.39 GiB.

The full run has independent `map`, `activity`, and `power` worker limits, but
a live admission gate additionally reserves 24 GiB of system memory and 15 GiB
of `/tmp` space. Vector/HBM activity compilation is serialized. Mapping uses
at most four DC cores per job; mapping and power replay share the DC-license
semaphore and reserve one license for other users.

Training VCDs are deleted immediately after successful replay. Representative
Qwen, mixed, mapped holdout, and failed VCDs are retained, with successful
validation VCDs compressed. Mapped artifacts and compact reports are copied
before worker-local builds are removed.

## Pre-full-run evidence

All component classes completed at least one real mapped-DC replay before the
full run:

| Component | Evidence |
|---|---|
| Matrix | Nine-scenario smoke; random action slope 3.851 pJ/action, R2 0.9999986 |
| Scalar | Nine-scenario smoke; random action slope 1.761 pJ/action, R2 0.9999696 |
| Vector | Mapped sequential coverage 12,530/12,530 (100%) |
| Control | Mapped sequential coverage 1,133/1,133 (100%) after packed-name translation |
| HBM controller | Mapped sequential coverage 48,120/48,120 (100%) after packed-name translation |

The full-run first Matrix point independently reproduced 100% sequential-cell
coverage for all nine scenarios and the same 3.851 pJ/action slope with
R2 0.9999986. Active dynamic energy exceeded matched idle energy at all three
repeat counts.

Two activity-generation defects were found and corrected before the final fit:

1. DC did not attach nested packed-struct SAIF names to flattened Control input
   bits. The replay flow now derives exact per-bit static probability and toggle
   rate from the source SAIF and applies 268 explicit packed-port overrides.
   Control random-128 power consequently changed from the clock-only 1.3824 mW
   to 1.9191 mW. Its final holdout error is 0.128%.
2. The original random FP generator included large positive exponents. EXP-heavy
   Vector scenarios saturated and could switch less than the zero-valued idle
   case. Random finite values are now constrained below exponent bias. All four
   Vector configurations have positive random-32/128/512 incremental energy.

These are measurement-flow corrections, not coefficient tuning. The append-only
CSV retains the superseded attempts for auditability.

## Full-run fit and validation

The canonical full run is:

```text
Workspace/area_new_power_calibration/runs/full_rtl_activity_v1_20260722_193606
```

The raw database contains 172 attempt rows. Latest-row reduction by
`point_key + scenario` gives 144/144 complete power points, and the mapping
database gives 16/16 complete mapped configurations. The canonical compact CSV
contains exactly these 144 latest complete points:

```text
analytic_models/power/calibration/power_calibration_points.csv
```

All 144 replays achieved 100% mapped sequential-cell SAIF coverage. Median DC
replay time was 19.0 s, P95 was 112.6 s, and peak replay RSS was 2.40 GiB. The
activity simulations used a median 0.37 GiB RSS and a maximum 1.35 GiB RSS.

The nonnegative structural fit produced the following validation result:

| Metric | Result | Gate |
|---|---:|---:|
| Training median APE | 7.01% | diagnostic |
| Mapped holdout median APE | 38.08% | <=15% |
| Mapped holdout P95 APE | 1109.45% | <=30% |
| Clock holdout median/P95 APE | 21.73% / 137.88% | <=15% / <=30% |
| Qwen/mixed median error | 21.09% | diagnostic |
| Qwen/mixed maximum error | 150.04% | <=20% |
| All random-window R2 | >=0.95 | pass |
| Missing action families | none | pass |

Component holdout errors expose where the model is and is not useful:

| Component | Median APE | P95 APE |
|---|---:|---:|
| Control | 0.13% | 0.13% |
| HBM controller | 8.89% | 37.54% |
| Scalar | 6.72% | 20.07% |
| Vector | 23.39% | 92.63% |
| Matrix | 116.23% | 1155.98% |

Control and Scalar validate well, and HBM is close to the intended range.
Vector remains underdetermined: four operation-family coefficients collapse to
zero under NNLS. Matrix is the dominant failure. Its dynamic design matrix has
rank 13/16, and identical structural action counts produce materially different
energy under low-toggle, random, Qwen, and mixed data. One activity-independent
coefficient vector therefore cannot explain both switching class and structural
scaling, especially the B=8 mapped holdout.

The candidate consequently **failed promotion**. The formal
`logic_energy_v1.json` remains the conservative bootstrap model. The failed
candidate and its full per-point diagnostics are retained separately as
`logic_energy_v1_candidate.json` and `power_validation_v1.json`; no partial
coefficient update is used by DSE.

Representative WNS also demonstrates the claim boundary. Several small mapped
points are timing-unclosed at 1 ns, including Vector VLEN=32 E6M5 (-261.06 ns),
Vector VLEN=64 E6M5 (-282.37 ns), and Scalar INT64/E8M7 (-858.28 ns). Their
power reports are valid RTL-activity-on-mapped-netlist evidence, but they are
not evidence of a physically closed 1 GHz implementation.

### Required model revision

The next candidate should preserve the completed replay data but separate:

- structural capacitance, including Matrix active PE count, output conversion,
  and clock area scaling;
- data-activity class or toggle-factor normalization;
- Vector operation mix with additional independent microkernels.

No additional large synthesis is justified until those features are identifiable
on the existing small points. The present failed fit is more informative than a
forced low-error regression because it prevents an unphysical model from being
promoted into architecture DSE.

## Interpretation

The current result is a completed calibration experiment, but not an accepted
power model. It establishes reliable replay infrastructure and useful
Control/Scalar/HBM evidence while falsifying the current universal action-energy
fit for Matrix and Vector. Even a future promoted candidate would support only
architecture-level relative comparison, not TDP or signoff claims. Gate-level
activity, CTS, routed parasitics, SRAM leakage, external HBM, and
package/interconnect energy still require separate validation.
