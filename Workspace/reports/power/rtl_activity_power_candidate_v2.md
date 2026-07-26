# PLENA RTL-Activity Power Candidate v2

## Objective

Power v2 corrects the semantic and identifiability failures found by the v1
experiment. It is the current fixed-corner, pre-layout on-chip component of
the formal system-energy objective at ASAP7 TT, 0.7 V, 25 C, and a nominal
1000 ps clock. It does not include gate-level simulation, CTS, routed
parasitics, SRAM leakage, external HBM/PHY, package, or the KV handoff link.

The active bootstrap artifact is not overwritten during collection or fitting.
V2 passed the grouped-holdout gates described below and was promoted to the
default on-chip action-energy component. External HBM3E and interconnect
energy are added by the system estimator.

## Why v1 failed

The 144-point v1 replay dataset was internally valid, with 100% mapped
sequential-cell SAIF coverage, but its regression semantics were too coarse:

- The same Matrix structure could consume roughly ten times different dynamic
  energy under low-toggle, Qwen-like, and random data, yet v1 forced one action
  coefficient to explain all three.
- Vector had rank 4 for five free features; Scalar had rank 2 for five. Mixed
  kernels could not identify independent hardware-family energy.
- Idle work scaled by PE/lane counts instead of mapped component area.
- Vector/Scalar format points did not cover the principal DSE formats.

This caused a 38.08% mapped-holdout median error, 1109.45% P95 error, and
150.04% worst Qwen/mixed error. The rejected candidate remains archived; the
bootstrap artifact remained active.

## V2 model semantics

V2 defines:

```text
P50 nominal = Qwen-like per-family action slope
P10/P90     = low-toggle/random per-family activity envelope
idle energy = makespan cycles * component mapped area * clock density
```

The uncertainty range is an empirical activity envelope, not a statistical
confidence interval. Structured CostEmitter actions now preserve Matrix array,
cross-K reduction, output conversion, Vector operand/reduction variants,
Scalar functional-unit families, segment count, active lanes, and active bits.
The v1 loader folds these variants back into legacy families for compatibility.

Matrix energy follows exact structural counts. In particular, cross-K reduction
is strictly zero when `MLEN / BLEN == 1`. Vector and Scalar use independent
hardware-family slopes and precision lookups for `FP_E5M6`, `FP_E6M5`, and
`FP_E8M5`.

## Calibration plan

The plan contains 31 mapped configurations and 395 independent RTL-activity to
DC replay jobs. All 16 v1 DDCs are reused read-only; only 15 configurations are
newly mapped. Each compiled activity executable is reused across its scenarios.

The new points add eight Matrix shapes/precisions, two Vector formats, and five
Scalar width/format combinations. Family-specific microkernels separately
exercise Matrix array/reduce/conversion, Vector VV/VF/VSEG and SFU/reduction/
movement/lane access, Scalar FP/INT/SFU/register families, each DMA direction,
and frontend issue.

Training uses Qwen-like N=32/128/512 slopes. Low-toggle and random N=128 define
the activity envelope. Qwen-mix and grouped configuration holdouts do not train
the nominal coefficients.

## Resource policy

The full pipeline uses separate mapping, activity, and power queues:

```text
8 mapping workers
6 activity workers, up to 4 heavy Verilator builds
up to 11 DC replay jobs, reserving one license
60 weighted CPU tokens
16 GiB memory reserve and 8 GiB /tmp reserve
```

Mapping and replay share one DC semaphore. Verilator does not consume a DC
license. Admission uses current free memory and `/tmp` plus learned peak RSS,
inflated by 1.35. SEC-50 releases the local token and retries only the affected
job. Successful training VCDs and worker build products are removed promptly.

Before launch, manifest-based cleanup deleted two inactive PLENA directories
(208.4 MiB). It did not touch unknown files, other users, `/tmp/moe_models_e13`,
or `/tmp/bc625`. The dry run observed about 111 GiB free memory and 41.5 GiB
free `/tmp`.

## Pre-full-run evidence

A real VectorMachine smoke reused the archived V16/E6M5 mapped DDC and replayed
idle, lane-load, and lane-store activity:

| Scenario | Accepted/completed | Sequential SAIF | Dynamic energy |
|---|---:|---:|---:|
| idle-128 | 0 / 0 | 12,530 / 12,530 | 62,640.128 pJ |
| lane-load-128 | 128 / 128 | 12,530 / 12,530 | 63,437.210 pJ |
| lane-store-128 | 128 / 128 | 12,530 / 12,530 | 62,906.368 pJ |

Both lane-access active windows exceed the matched idle window. The first Verilator build
used approximately 0.58 GiB peak RSS and subsequent cached scenarios about
0.38-0.58 GiB. Each DC replay used approximately 1.03 GiB peak RSS. These
measurements support the aggressive run limits while the live resource gate
continues to protect the host.

An additional CostTrace-derived mixed Vector smoke exposed an important DC
reporting effect. Its raw active total was 170.8 pJ below idle, even though the
register/combinational non-clock residual was +2,027.7 pJ. The difference was
an activity-dependent -2,191.4 pJ change in DC's pre-CTS `clock_network`
group. A single-segment reduction showed the same direction: raw -9.4 pJ,
non-clock +44.9 pJ, clock delta -54.1 pJ. Therefore v2 records all three terms
and fits action energy from matched active-minus-idle non-clock power. The
runtime model adds this to an area-scaled idle baseline. The raw clock delta is
kept for audit and is never clipped into a nonnegative coefficient.

The mixed workload itself is no longer a handwritten average. It is derived
from the current Qwen3-32B one-layer CostTrace (`seq=482`, `batch=16`,
`MLEN=VLEN=2048`, `BLEN=1024`) and identified by a stable semantic hash.

## Promotion gates

V2 is accepted only if all of the following pass:

```text
component Qwen holdout median error <= 15%
component Qwen holdout P95          <= 30%
Qwen-mix total dynamic error        <= 20%
idle holdout median/P95             <= 15% / 25%
every family slope R2               >= 0.95
no missing family, negative coefficient, or structural anomaly
cached evaluation                   < 10 ms/trial
```

Matrix must additionally pass zero single-split reduction energy, monotonic
action/size scaling, B16 grouped holdout, and asymmetric T/L checks. Promotion
is atomic: a failed future refit leaves the prior active artifact untouched and
diagnostics identify the family or structural assumption requiring revision.

## Full-run correction and fitting decisions

The first full replay exposed a stimulus-generation defect rather than a DC
failure. Packed random values repeated over power-of-two Vector lengths and
some scenarios toggled only the low portion of the input bus. The resulting
energy did not scale with active lanes. The generator was corrected to vary
every packed element and the complete 395-scenario replay was repeated. The
corrected evidence shows the expected scaling: E6M5 Vector ADD/MUL energy from
VLEN 16 to 32 approximately doubles, and MXINT T4/L4 Matrix array energy grows
from 1.35 pJ/action at B2 to 32.23 pJ/action at B16.

The final model uses only structures identifiable from the collected data:

- MXINT array actions use nonnegative fixed-launch, B-wide feed, and B-cubed
  PE-MAC terms. This predicts the B16 grouped holdout without assigning
  cross-K work to a single-split array.
- MXFP uses B-cubed PE scaling. All available MXFP anchors have B4, so separate
  launch/feed terms would not be identifiable.
- Full and single-segment Vector reductions scale with
  `VLEN * log2(VLEN)`; compact multi-segment reduction scales with `VLEN`.
- Vector lane loads and stores are distinct families and scale with VLEN.
- Missing FP formats use nonnegative exponent/mantissa interpolation before a
  total-width fallback.
- HBM controller energy is fitted per opcode and accepted logical lane. The
  fixed-amount points cannot separately identify line and useful-byte energy,
  so both coefficients remain explicitly unavailable and runtime reports warn
  about this limitation.

## Final validation

The corrected run completed all 31 mapped configurations and 395 latest-state
activity replays with no failed scenario. All fitted coefficients are
nonnegative; across the 33 multi-N family slope checks, the minimum measured R2
is 0.999526. Every promotion gate passed.

| Evidence | Median error | P95 error | Maximum error |
|---|---:|---:|---:|
| Matrix grouped holdout | 6.10% | 11.69% | 12.61% |
| Vector grouped holdout | 7.89% | 20.93% | 28.29% |
| Scalar grouped holdout | 0.90% | 8.69% | 9.77% |
| HBM controller holdout | 7.82% | 21.65% | 23.18% |
| All action holdouts | 7.47% | 20.88% | 28.29% |
| Idle clock holdouts | 3.83% | 8.34% | below 25% gate |

The worst Qwen-mix total dynamic-energy error is 11.41%, below the 20% gate.
Structural checks pass: single-K-split reduction energy is zero, Matrix energy
is monotonic through B1024, all precision/size estimates are nonnegative, and
all CostEmitter action families are covered. Cached power evaluation has a
3.21 ms median latency, below the 10 ms requirement.

The accepted candidate was promoted to:

```text
analytic_models/power/calibration/logic_energy_v2.json
calibration_status = rtl_activity_calibrated_candidate_v2
```

It is now the default on-chip component of the DSE system-energy objective.
The v1 bootstrap and rejected v1 candidate remain historical audit evidence,
not runtime alternatives in the formal profile.

## DSE integration smoke

Three one-trial Qwen3-32B smokes at `MLEN=VLEN=512, BLEN=64` exercised MXINT4,
an MXINT8-weight profile, and MXFP. Every trial loaded
`onchip_action_energy_v2`, emitted P10/P50/P90, component breakdowns, scope,
exclusions, and the HBM identifiability warning. Representative P50 results
were 9.85 W, 11.01 W, and 22.89 W respectively. A matched MXINT4 run with
the energy estimator disabled produced identical latency, area, and accuracy
values, confirming that power evaluation does not perturb the other three
objectives. Nominal system energy is nevertheless the formal fourth minimized
quantity alongside maximized accuracy.

The compact smoke evidence is retained in
`Workspace/area_new_power_calibration/runs/full_rtl_activity_v2_20260723_000207/dse_smoke_v2.json`.

## Claim boundary

V2 is calibrated to RTL activity replayed on mapped DC netlists at the declared
ASAP7 corner. It is not gate-level activity validation and does not include
CTS, routing parasitics, SRAM leakage, external HBM/PHY, package, or a KV link.
The P10/P50/P90 range is an empirical operand-activity envelope, not a
statistical confidence interval. The nominal estimate participates in the
formal energy objective, but it must be labelled an architecture-level model
until deferred gate-level and physical-design evidence is available; it is not
a signoff power constraint.
