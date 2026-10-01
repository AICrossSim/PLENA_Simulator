# BF16 fixed-budget IPD A/B results, 2026-10-01

The [specification](../../IPD_BF16_AB_SPEC.md) defines the hardware budget,
policy, timing scope, and replay commands. The corresponding new branches are
`research/moe-ipd-bf16-ab` in PLENA_Compiler and PLENA_Simulator. `receipt.json`
and `provenance_manifest.json` record the exact binary, source, and archived
input hashes. `points.csv` contains every paired point; `summary.csv` and
`summary.json` contain the aggregate calculations. Each of 180 points ran
twice with an identical complete Rust report, for 360 Rust executions.

## A: archived layer-13 route reanalysis

All 12 archived DeepSeek-V2-Lite decode windows use BF16, the same route order,
12,288 multipliers, 48 KiB W/return storage, 12 KiB X, a 2 MiB arena, a
256 B/ns shared HBM link at 64 ns, and 256 shared 32 B credits. These inputs
were examined in previous work and are **not independent held-out evidence**.
The Rust model times full FFN execution and ordered combine; Router, Attention,
native memory simulation, and silicon are outside its scope.

| Organization / policy | Geometric mean FFN latency | Against same-shape joint | Against single 6 IPD | Against homogeneous 3+3 IPD | Mean HBM configured-byte utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single 6 / IPD | 2.9693 ms | 1.0000× | 1.0000× | 0.9908× | 46.16% |
| Homogeneous 3+3 / IPD | 2.9420 ms | 1.0285× | 1.0093× | 1.0000× | 46.57% |
| Heterogeneous 4+2 / joint | 2.9729 ms | 1.0000× | — | — | 46.11% |
| Heterogeneous 4+2 / IPD without quota | 2.9373 ms | 1.0121× | — | — | 46.65% |
| Heterogeneous 4+2 / full IPD | 2.8781 ms | 1.0330× | 1.0317× | 1.0222× | 47.57% |

Thus, on this reused set, 4+2 plus IPD is 3.17% faster than single 6 plus
IPD, 2.22% faster than 3+3 plus IPD, and 3.30% faster than 4+2 plus joint.
The fixed-budget quota rule adds 2.06% over 4+2 IPD without it. These are
paired geometric-mean speedups, not a claim of 3.17 percentage points less
latency or a measured full-model improvement. A policy-only win on 4+2 would
not establish the necessity of heterogeneous cores; the 6 and 3+3 columns
provide that comparison here, subject to independent replication.

The 4+2 advantage over 3+3 is essentially absent at B2 (0.9998×), then
1.0043× at B4, 1.0369× at B8, and 1.0487× at B16. Two B2 windows are
slower than 3+3 IPD, by 0.04% and 0.03%; the worst 4+2 IPD regression
against same-shape joint is 0.13%.
Every policy and organization preserves the same useful MAC count and weight
byte count per workload. All capacity checks, DMA drain checks, and repeated
report equality checks passed.

For 4+2 IPD, mean HBM weight-read utilization is 47.57% of the configured
256 B/ns and 95.13% of the 128 B/ns steady-state ceiling from 256 × 32 B /
64 ns credits.
The latter is an analytical credit-bandwidth bound, not a measured HBM device
limit. Mean arithmetic-active observer fractions are 59.63% and 54.81% on the
two cores; useful MACs divided by nominal multiplier-cycle capacity are 1.16%
for the whole organization. Arithmetic-active can overlap feed or front-end
waits and must not be called useful MAC utilization. The per-point file also
records MAC issue/HBM-accept intersections, weight-wait fractions, spatial efficiency,
credit peaks, and control service. Simultaneous arithmetic and HBM requests
occur in the model; this does not by itself prove that future cross-layer
prefetch will improve latency, especially near the credit roof.

Review of diagnostic semantics: quota demand follows Current when present and
uses Next only when Current is absent. A newly bound Next therefore does not
immediately change the quota while Current runs. The Compiler protocol text
uses the shorter phrase "Current/Next task identity changes"; the Rust rule
above is the measured one. `ipd_quota_block_cycles` counts cycles in which at
least one candidate was filtered at the first DMA grant opportunity. It is
not a count of cycles with an idle HBM link. Neither diagnostic enters the
latency or HBM utilization calculations above.

## B: sequential no-prefetch baseline

The Rust multi-layer entry point and Python capture/prepare/replay driver are
implemented. The self-contained two-layer protocol fixture yielded 14,618
cycles, 1,572,864 total weight bytes, 42.03% configured-byte HBM utilization,
and identical repeated JSON reports. `toy_multilayer_receipt.json` records its
hashes. These numbers are **only a simulator protocol check**: the second layer
is synthetic and the zero gap omits Attention/Router time and traffic.

The original adjacent-layer route `.npz` capture is absent from this Git
archive and local workspace. Therefore this branch has no real multi-layer
DeepSeek B profile, no measured cross-layer prefetch headroom, and no
cross-layer prefetch speedup claim. Once the capture is supplied, run
`multilayer_profile.py prepare --capture PATH --layer-ids 12,13,14` and then
`multilayer_profile.py run` on the same output directory, repeating for 6,
3+3, and 4+2 under the fixed budget. Any nonzero inter-layer gap and HBM
traffic must come from a separately justified Attention/Router measurement.

## Reproduction boundary

The committed tables are a compact audit of the raw `/tmp/plena-ipd-ab-reviewed`
execution, while the repository runner regenerates full per-point JSON,
provenance snapshots, and paired summaries. The runner is deterministic, but
these windows were already inspected. A paper should freeze new request-disjoint
adjacent-layer captures before tuning and report both per-window differences
and resource-normalized baselines.
