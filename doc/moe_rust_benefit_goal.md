# MoE Rust benefit validation goal

## Question and completion criterion

Under the same total 4096 matrix multipliers and one shared HBM configuration,
does a heterogeneous two-core normal-buffer design improve complete fixed-route
MoE FFN latency and useful MAC utilization over competitive grouped single-core
and homogeneous two-core designs? Report conditions and losses as well as wins.
A positive result is not required to complete this goal. A failed numerical or
resource check cannot be counted as a performance observation.

## Required implementation

1. Actual numerical Gate/Up, SwiGLU, Down, weighted combine in Rust, using encoded
   PLENA MX element/scale bytes read through one shared Ramulator instance.
2. Independent normal buffers and FP32 accumulators; finite double-buffered
   weights, DMA credits, shared vector and combine storage.
3. A bounded, work-conserving ready-job dispatcher in addition to the original
   fixed M threshold. Both homogeneous and heterogeneous pairs can use it.
4. A finite per-core 64-byte read cache with explicitly charged data/tags,
   access ports and latency, available to every baseline as well as the pair.
5. Full-dimension fixture export and an independent numerically checked oracle
   without materializing enormous Python scalar lists. Ready inputs and weights
   may be synthetic, but the source model's D/F dimensions and archived routing
   are retained and labeled. This is not a full model or device router claim.

## Predeclared comparison

- Single: B32/K128, B16/K256, B8/K512, and the existing B4/K1024 geometry
  (4096 multipliers each). The last was added before comparative runs so the
  existing PLENA geometry is not omitted from the strongest-single reference.
- Homogeneous pair: two B16/K128 cores (2048 + 2048).
- Heterogeneous pair: B16/K192 + B8/K128 (3072 + 1024).
- The original fixed-threshold pair remains an ablation; ready-job scheduling
  and caching are not credited to heterogeneity when the baseline can use them.
- Hold total configured activation, accumulator, weight and cache SRAM budgets,
  HBM channels, credits, clock and vector throughput fixed within a comparison.
  List per-core ports and geometry; equal multipliers/SRAM do not imply equal
  synthesized area. Compare against the fastest tested single, not a weak one.

## Workloads and measurements

- Preserve original Qwen archived D=2048/F=512 and DeepSeek D=2048/F=1408.
- Include low-M decode slices and a larger-M grouped workload. Synthetic route
  stressors remain separately labeled; never present replicated traces as a
  captured prefill trace. Preserve archived token/slot/expert/weight tuples.
- For each run report elapsed simulated time, useful and issued MACs, global
  and per-core utilization, real HBM bytes, cache requests/hits, capacity peaks,
  job assignment/completion times, and core/loader waits.
- Repeat configurations at least twice; check exact input/binary identity,
  numerical output, capacity bounds, counters and deterministic repeats.
- Predeclared sensitivities: DeepSeek batch 32 with serialized matrix service
  and with cache disabled. After primary Qwen results exposed cache-port
  domination, extend cache bypass to all three remaining windows as an
  explicitly adaptive diagnostic. Report this extension and every outcome.
- Check the matrix timing assumptions against the existing MatrixMachine and
  RTL structure. Document differences rather than claim RTL calibration. Use
  sensitivity/ablation to distinguish compute padding, HBM traffic and queue
  imbalance; no isolated optimization is a universal MoE speedup.

## Deliverables and boundaries

Deliver committed code, the complete configuration/workload manifest, tests,
raw run reports, comparison gates, and a Chinese results report explaining
whether and where the tested pair is useful. Preserve the previous V0 report.
The existing standalone Rust experiment remains the entry point for this goal;
legacy ISA integration is tracked separately and must not be claimed complete.
Transpose, many-small-core configurations, sharing, attention and RTL/PPA are
follow-up work after this normal-buffer benefit question is measured.

## Completed first measurement, 2026-09-05

Nine comparison groups / 126 valid Rust runs passed numerical, capacity,
identity and deterministic-repeat checks. With cache policy optimized per
architecture, the tested 3072+1024 pair lost to the strongest tested single in
all four windows, and also lost to the tested homogeneous pair. Do not freeze
this split or cite the cache-constrained primary speedup as proof of benefit.
This does not reject all heterogeneous designs. Calibrate issue/SRAM-port timing
before broader geometry search and legacy ISA integration.

See [Chinese result snapshot](moe_full_shape_result_20260905.md). Raw reports,
binary, route identities and reproduction files are archived under
`/scratch/shared/mcl123/plena/outputs/moe_full_shape_benefit_20260905/`.
