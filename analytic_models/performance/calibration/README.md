# Compiler Cost Memory Calibration

## RTL-v1 Compute Timing

CostEmitter and the transactional emulator share one opcode timing source:

```text
transactional_emulator/calibration/rtl_opcode_timing_v1.json
```

Python loads that artifact directly and evaluates the same MXINT/MXFP formula
records as Rust.  The production compute objective is intentionally defined as

```text
serial_resource_work_cycles = sum(opcode_count * backend_resource_cycles)
compute_latency_ns = serial_resource_work_cycles * CLOCK_PERIOD_PS / 1000
```

This quantity is calibrated compute work, not a scheduled makespan.  The
ordered schema-v4 schedule can additionally be replayed with the Rust-parity
scoreboard by explicitly enabling `scheduled_shadow`.  Shadow replay reports
stalls, mutually exclusive critical-path ownership, and overlap, but remains
opt-in because production Qwen replay is currently too slow for every DSE
trial.  Post-hoc V3 service is marked `post_hoc_v3`; exact validation can use a
compact transactional `--dma-event-trace` and is marked
`ramulator_observed`.

The measured schema-v4 post-hoc path takes 9.72 seconds for one batch-1 layer,
37.19 seconds for one batch-16 layer, and 47.01 seconds for the compressed
batch-16 64-layer workload.  The last case represents 5.584 billion dynamic
instructions with 559,209 literal scheduler instructions.  This does not meet
the one-second per-trial target, so scheduled shadow is intentionally not a
default DSE objective.

The compact Qwen3-32B one-layer regression reference is:

```text
analytic_models/performance/calibration/
  rtl_v1_qwen3_32b_resource_work_reference.json
```

For its equal-E4M3, `MLEN=VLEN=512`, `BLEN=64` configuration, CostEmitter and
the transactional emulator match all 35 non-HBM opcode counts and the exact
`27,776,907` non-HBM resource-work cycles.  Replaying the 2,317 observed DMA
events also matches transactional resource work, stall categories, critical
path, and total makespan cycle-for-cycle.  The run is still labeled
`unsupported_opcodes` and out of the measured production-shape domain; this
agreement validates implementation parity, not the unsupported RTL behavior
or a 1 GHz timing-closure claim.

`evaluate_compiler_cost()` defaults to `compute_timing_mode="rtl-v1"` and
`scheduled_shadow=False`.  Use `compute_timing_mode="legacy"` only for an
explicit historical comparison.

## Production-DMA V4 Memory Shadow

`hbm_dma_service_v4.json` is the accepted schema-v4 memory-shadow artifact.
Unlike V3, its target is the completion time of the production
`gather/scatter` DMA implementation.  Store manifests therefore distinguish
full-line writes from partial-line read-modify-write operations and merge
overlapping element/scale patches onto one physical 64-byte line.

The checked dataset contains 2,592 generic production-DMA points and 36
targeted row-state anchors.  It covers HBM channel counts 8, 32, and 128;
MXINT4/8 and MXFP4/8; block sizes 64 and 8; dimensions 64 through 2048; all
three HBM opcodes; aligned, 32-byte-offset, contiguous, 2x-stride, and large
stride layouts.  Rust and Python must agree on read/write line counts and the
canonical request-manifest hash before a point can enter the fit.

Generic holdout results in `hbm_dma_service_v4_validation.json` are:

```text
request-manifest mismatches:       0
median absolute error:          3.48%
P95 absolute error:            18.10%
maximum absolute error:        42.18%
weighted MAPE:                  2.80%
H_STORE_V P95 absolute error:  19.92%
```

Promotion additionally requires fixed-point arrival-aware replay of four
production traces.  The final results recorded in
`hbm_dma_service_v4_system_validation.json` are:

| Workload | Total HBM work error | Scheduled makespan error |
| --- | ---: | ---: |
| Qwen3-32B, seq482, E4M3 block8, c128 | 3.38% | 0.09% |
| Qwen3-8B, seq64, MXINT4 block64, c128 | 6.07% | 0.16% |
| Qwen3-8B, seq128, MXINT8 block64, c32 | 11.70% | 0.57% |
| Qwen3-32B, seq128, E1M2 block64, c8 | 0.80% | 0.21% |

Every replay passes exact manifest/line/byte parity.  The old transactional
trace is used only as the first arrival-time seed; Rust then re-executes the
complete production DMA sequence through Ramulator until the bounded arrival,
duration, physical-delay, and makespan deltas converge.  The correctness gate
and numerical execution are unchanged.

V4 has two deliberately distinct runtime fidelities:

- `full-global-stateful` predicts every dynamic DMA occurrence with one global
  MOP4CLXOR row state.  This is the system-validation path and the only mode
  allowed to drive scheduled replay.
- `one-layer-stateful-scaled` predicts one complete decoder layer in issue
  order, scales only `layer/*` stages by the model layer count, and keeps
  global prologue/epilogue work single-counted.  This is the DSE memory shadow
  and is reported explicitly as `per_occurrence_prediction=false`.

For Qwen3-32B batch16, seq482, 64 layers, the scaled path takes 24.80 seconds
and peaks at about 944 MiB in a cold command-line run.  A full-global-stateful
diagnostic was still incomplete after 6 minutes 25 seconds and had reached
about 13.4 GiB RSS, so it is intentionally not the per-trial DSE path.  These
timings include compiler trace construction and cold Python/Nix startup.

A two-layer Qwen3-8B, seq64, MXINT4 block64, c128 differential check compares
the scaled mode directly with `full-global-stateful`: occurrence count,
read/write bytes, and read/write requests match exactly, while aggregate HBM
work differs by 1.82%.  The remaining difference is the disclosed omission of
the second layer's inherited global row history, not missing DMA traffic.  The
compact evidence is stored in
`hbm_dma_service_v4_scaling_validation.json`.

V4 remains a post-hoc service model: it does not run online Ramulator state
across simultaneously outstanding DMA queues.  It is therefore the default
memory shadow, not the formal DSE objective.  V3 artifacts are retained below
for historical reproduction.

## V4 Calibration Flow

```bash
python -m analytic_models.performance.compiler_cost_model \
  generate-service-v4-plan \
  --row-state-anchors \
  --output analytic_models/performance/calibration/hbm_dma_v4_plan.json

cargo run --release --bin hbm_dma_calibration -- \
  --input analytic_models/performance/calibration/hbm_dma_v4_plan.json \
  --output analytic_models/performance/calibration/hbm_dma_v4_results.json

python -m analytic_models.performance.compiler_cost_model \
  fit-service-v4 \
  --plan analytic_models/performance/calibration/hbm_dma_v4_plan.json \
  --results analytic_models/performance/calibration/hbm_dma_v4_results.json \
  --output /tmp/hbm_dma_service_v4_candidate.json \
  --validation-output /tmp/hbm_dma_service_v4_validation.json
```

The candidate is promoted only by
`aggregate_hbm_v4_system_validation.py` after the generic holdout and all four
required system cases pass.  Promotion records a portable evidence path and
SHA-256 in the model metadata.

## Historical V3 Global Model

`hbm_service_global_v3.json` is the preserved schema-v3 global HBM service
artifact. It is precision-independent at the compiler trace
boundary: the trace records logical objects and offsets, then the V3 request
builder packs them for MXINT4/8, MXFP4/8, or the existing MXFP8 block-8
transactional format.

The checked calibration contains 1536 deterministic patterns with seed
`20260711`. It covers:

- HBM channels 8, 32, and 128.
- DMA dimensions 64 through 4096.
- MXINT4/MXFP4 block-64, MXINT8/MXFP8 block-64, and MXFP8 block-8.
- Read, write, store RMW, aligned/misaligned, contiguous/2x/model-column
  stride, and single/reuse/affine/nested streams.
- At most 2048 physical requests per calibration geometry.

All formats use the same nonnegative-ridge coefficients. There is no exact
stride/residue lookup table and no Qwen workload-specific correction. All
1536 format-generic patterns supply raw Ramulator service-time samples. The
supported MXFP8 block-8 subset also runs through production `dma.rs`; all 363
of those samples match the raw request path exactly in bytes and latency.

Current grouped holdout results are recorded in
`hbm_service_global_v3_validation.json`:

```text
physical request byte mismatches: 0
median absolute error:           12.22%
P95 absolute error:              39.03%
maximum absolute error:         103.87%
```

The median target is met, but P95 is above the 25% promotion threshold. V3
therefore remains in DSE shadow mode and the historical bandwidth prune stays
enabled by default. The existing Qwen3-32B target remains a holdout: total
latency is within 10%, while its memory category is still underpredicted. Do
not promote V3 to the default objective based only on the total-latency result.
The end-to-end acceptance state, including the target holdout and active DSE
coverage, is recorded in `hbm_service_global_v3_system_validation.json`.

## V3 Calibration Flow

```bash
python -m analytic_models.performance.compiler_cost_model \
  generate-service-calibration-plan \
  --output analytic_models/performance/calibration/hbm_service_v3_plan.json

cargo run --release --bin hbm_dma_calibration -- \
  --input analytic_models/performance/calibration/hbm_service_v3_plan.json \
  --output analytic_models/performance/calibration/hbm_service_v3_results.json \
  >/dev/null

python -m analytic_models.performance.compiler_cost_model \
  fit-service-model \
  --plan analytic_models/performance/calibration/hbm_service_v3_plan.json \
  --results analytic_models/performance/calibration/hbm_service_v3_results.json \
  --output analytic_models/performance/calibration/hbm_service_global_v3.json \
  --validation-output \
    analytic_models/performance/calibration/hbm_service_global_v3_validation.json
```

The Rust driver creates fresh Ramulator state for every sample, applies the
fixed mixed-row conditioner, reports raw and optional production-DMA timing,
and independently audits read/write bytes.

## DSE Rollout

The active dense-Qwen3 DSE fixes `VLEN=MLEN`, rejects MXINT3, and loads the 103
accuracy-qualified software precision profiles.  Its default configuration is:

```bash
--compiler-cost-mode compute-objective \
--compiler-compute-timing rtl-v1 \
--compiler-cost-calibration \
  analytic_models/performance/calibration/hbm_dma_service_v4.json \
--compiler-v4-memory-evaluation one-layer-stateful-scaled \
--legacy-bandwidth-prune
```

This mode uses `compute_latency_ns` as the Optuna latency objective, retains
the historical bandwidth constraint as a guard, and records V4 memory, serial,
stage roofline, domain, and fidelity metadata for every accepted point.  The
memory result does not enter the formal objective.  `--compiler-cost-mode
objective` remains opt-in; out-of-domain memory points are rejected there.
Capacity, ISA legality, precision compatibility, and physical-port constraints
remain hard constraints.

## Legacy V2

`hbm_surrogate_transactional_dma_v2.json` and the `*_v1.json` artifacts remain
available only for explicit diagnostics and historical reproduction. V2 uses
MXFP8 block-8 exact stream cells and is not the DSE default or a V3 fallback.
Its optimized Qwen3 target result remains:

```text
one-layer latency: 492.993 ms predicted / 495.666 ms actual (-0.54%)
memory latency:     78.856 ms predicted / 81.530 ms actual (-3.28%)
```
