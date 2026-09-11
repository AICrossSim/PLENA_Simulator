# MoE bottleneck diagnosis, 2026-09-11

The current numerical runner is primarily limited by supply, finite ports and
control/feedback for the tested shapes. Faster memory helps both organizations;
heterogeneous cores do not beat the correspondingly optimized single core in
this bounded experiment. This is not a claim about all shapes or schedulers.

Validation: 440 fixed-assignment runs (220 points, two repeats), 48 adaptive
work-conserving runs, 46 MoE tests, four HBM diagnostic tests and 540 independent
port-service checks passed. Numerical BF16/FP32/pre-round FP32 outputs match the
independent reference exactly; repetitions are identical and native requests
drain. Default timings reproduce the previous frozen experiment exactly.

| Window | Baseline single / best heterogeneous (us) | Ideal supply, Q32 single / heterogeneous (us) |
|---|---:|---:|
| B8 | 582.155 / 745.106 | 53.734 / 55.352 |
| B32 | 1003.224 / 1319.428 | 207.761 / 258.297 |

Ideal supply is an explicit oracle, not a hardware configuration: remove native
memory timing and accelerate non-MAC service 64x while retaining dependencies
and capacities. Single Q32 reaches approximately 91.5% / 94.6% of nominal useful
MAC throughput. Further accelerating MAC service gives 1.884x / 1.914x, providing
a near-compute-bound control. Both organizations receive the same intervention.

The adaptive follow-up exposes a concrete placement/control issue: in B32/Q32
with faster HBM and DMA, the small core steals a 30-row expert after its short
jobs. Capacity permits the job, but repeated context scans are expensive. The
window grows from 1.139ms (frozen placement) to 3.091ms (adaptive placement).
The next mechanism should combine completion-time-aware placement with bounded,
charged ready-state selection, alongside supply and accumulator improvements.

Run definitions and exact intervention semantics are in
[scripts/moe_bottleneck/README.md](../scripts/moe_bottleneck/README.md).
The full evidence is archived at
`/scratch/shared/mcl123/plena/outputs/moe_bottleneck_20260911/`:
`RESULT_ZH.md`, `manifest.json`, `sensitivity.csv`, `core_services.csv`,
`organization_comparison.csv`, `adaptive_dispatch/`, and `repro/`.
The archive includes original inputs by hash, per-point architecture/results,
logs, actual binary/native library, source patch, scripts and figures.

Scope: Compiler-exported fixed-route MoE operators, archived routes and synthetic
full-bank block8 weights; analytical core timing plus native Ramulator or clearly
marked timing oracles. No full-model inference or physical hardware measurement.
