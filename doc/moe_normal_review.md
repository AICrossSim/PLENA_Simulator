# MoE normal-buffer cores and HBM DMA: review guide

This review implements the meeting's first experiment: independently configured
normal-buffer cores with private accumulators, shared HBM, and a fair single-core
comparison. It executes actual nonzero MoE arithmetic. The best tested large/small
pair is **slower than the optimized single core**; this is not a speedup claim.

## Read the implementation in this order

1. `transactional_emulator/src/moe_normal/types.rs`: workload, per-core resources
   and report contract. `testbench/moe_normal_review/{single,large_small}.json`
   are the actual fixed configurations selected by the completed search.
2. `src/moe_normal/engine.rs`: validate finite resources, group/dispatch expert
   jobs, gather activations, run gate/up → SwiGLU → down, and combine routes in
   deterministic token/slot order. Each core owns its normal weight SRAM and
   accumulator; the existing ISA runner is not the entry point for this path.
3. `engine.rs` DMA path plus `lib/ramulator/src/model.rs`: bounded prefetch,
   in-flight 32-byte sector merging and per-channel request submission to one
   Ramulator HBM2 instance. Returned elements/scales are copied and decoded into
   reserved BF16 slots before compute. No unbounded cache or free extra bandwidth.
4. `testbench/moe_timing/replay/compare_moe_normal.py`: execute the same workload
   on both configurations, enforce fair budgets, compare against the Compiler
   oracle, and reject resource/counter/hash/repeat mismatches before reporting time.

Paths in items 2–4 are relative to `transactional_emulator/`.

## Current dimensions

Physical tiles have M=N=BLEN and K=MLEN. A tile is serviced over multiple cycles;
multiplier count is BLEN×MLEN, not M×K×N. Whole-model D/F dimensions are separate.

| Resource | Single baseline | Large core | Small core |
|---|---:|---:|---:|
| Tile (M,K,N) | (8,512,8) | (4,768,4) | (2,512,2) |
| Multipliers | 4096 | 3072 | 1024 |
| Weight SRAM | 64 KiB | 48 KiB | 16 KiB |
| Weight slots | 4 | 4 | 4 |
| Activation/intermediate SRAM | 4 MiB | 2 MiB | 2 MiB |
| Accumulator storage | 1 MiB | 512 KiB | 512 KiB |

Both systems share the same 8-channel HBM2 preset, clock, total activation supply,
DMA budget and vector unit. The calibrated native transfer is 32 bytes; the
previous Rust wrapper's 16-byte assumption was corrected, not used as a speedup.
The DRAM command scheduler and HBM timing preset are retained. Logical DMA credits
are capped at 128, native trackers at 256; DMA frontend SRAM is 44 KiB including
staging. The two cores do not receive two independent HBM bandwidth budgets.

HBM weights are output-major `[N,K]` rows with separate element/scale ranges.
The local E4M3/E8M0 codec uses 8 elements per scale. An aligned MLEN-long row
fragment therefore needs MLEN elements and MLEN/8 scales. SRAM slots reserve both
packed ingress and decoded BF16 data. Both buffers are normal buffers in this PR.

## How time and correctness are measured

- Input activations and fixed routes are ready at time zero. Stop when final
  combined BF16 outputs are ready. Router execution and input/output HBM DMA are
  outside this boundary; weight DMA, decode, compute and combine are inside it.
- HBM delay comes from native Ramulator requests/callbacks. Core, SRAM-port,
  lookup/copy and vector service use explicit analytical timing. Time is simulated
  elapsed time, not the wall-clock duration of the Rust process. Same-output K
  accumulation dependencies are retained; cross-core waits cannot be summed as
  total latency. Those dependency/port assumptions remain subjects for review.
- The independent Compiler oracle decodes actual exported weight bytes, performs
  ascending-K FP32 arithmetic and applies the same BF16 boundaries. The comparison
  checks exact BF16 outputs as well as HBM completed bytes, finite SRAM/queue
  limits, task accounting, binary/input identity and repeat determinism.

## Essential measured result

The completed bounded search used archived decode routes, synthetic nonzero
weights/activations, D=2048 and F=512 (Qwen) or F=1408 (DeepSeek), with 8/32-token
windows. It selected a fixed winner per architecture class; those fixed choices
were then evaluated on separate route windows. This is a numerical MoE operator
experiment, not full-model inference with trained weights.

| Large/small pair versus optimized single | Relative elapsed time |
|---|---:|
| Four search workloads, geometric mean | +8.88% (slower) |
| Four held-out workloads, geometric mean | +8.53% (slower) |

`testbench/moe_normal_review/measured_latency.csv` contains only the eight pairs
needed to audit these percentages: `exp(mean(log(dual_ps/single_ps)))-1`.
These are extracted from the completed, hash-audited campaign, not a newly run
search on the review branch. Its generated weight images, route archives, raw
reports and historical DSE/ablation drivers are deliberately not in this PR.
The standalone runner can validate supplied full-size fixtures with the same
checked-in configurations; the smoke test below does not reproduce this table.

## Run the paired Compiler → Simulator smoke test

After checking out this branch, initialize the submodules (the Compiler pin is
the paired review commit). Install Python dependencies with the existing uv setup:

```sh
git submodule update --init --recursive
uv sync
nix develop --command bash -c \
  'cd transactional_emulator && cargo build --bin moe_dual_normal'
nix develop --command bash -c \
  'PYTHONPATH="$PWD/PLENA_Tools" uv run python \
    transactional_emulator/testbench/moe_normal_review/smoke.py \
    --compiler "$PWD/PLENA_Compiler" \
    --binary "$PWD/transactional_emulator/target/debug/moe_dual_normal" \
    --output-dir /tmp/plena-moe-review-smoke'
```

Use a new output directory for each invocation. If CARGO_TARGET_DIR is set, pass
the binary in that directory instead. The existing Nix derivation builds the
updated native calibration C API from its pinned Ramulator source.

The smoke generates D=31/F=47, 12 tokens, uneven expert groups and a shared expert,
then runs each checked-in architecture twice with zero numerical tolerance. It
fails on any correctness/resource/repeat mismatch and prints `all_gates_passed`.
The review checkout passed this end-to-end smoke, exporter tests, Rust library
and MoE tests, Python evidence-gate tests, formatting and strict clippy checks.

For a supplied full-size fixture, run `compare_moe_normal.py --help`; pass its
`workload.json`, `golden.json`, the two checked-in architecture files (single
first), `--atol 0 --rtol 0`, and an output directory. A full-size Qwen B8 check on
the review checkout passed exact output/resource/repeat checks and reproduced
the archived 0.524575 ms single / 0.564819 ms dual times. This one-case validation
is separate from the archived search result.

## Review boundary

No transpose buffer, Attention execution, runtime router or complete dual-core
instruction lowering is claimed. Private buffer allocation does not establish
equal physical area. The immediate review question is whether the data path,
resource accounting and timing assumptions form a sound comparison before any
further scheduling change; extra prefetch helps both the single and dual systems.
