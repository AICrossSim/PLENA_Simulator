# Live native HBM profiling

The `native-hbm` Cargo feature connects the existing event-level Current/Next
runtime to the pinned Ramulator2 CAPI v2. Each accepted 32-byte weight sector
gets a native callback; the callback enters the existing landing-write path.
The global DMA credit is released after the SRAM write acknowledges, not at the
memory callback. Rejection retries the same owner/address/slot/offset without
advancing the AGU. No separate bandwidth estimate schedules native returns.

Core, SRAM-bank, accumulator, vector and copy timing remain the Rust event
model. This is not RTL timing. The benchmark begins with routed X resident and
ends after the ordered MoE output combine. Router, attention, preceding/following
layers and model generation are outside this measurement.

## Build and reproduce

Use Ramulator2 revision `b3efdc5019a312874961a8c226097eb0581f2b5f`, fmt 10.2.1,
yaml-cpp 0.9.0, CMake >=3.14 and a C++20 compiler. `build.py` takes explicit local
dependency sources and writes a native-library receipt. The wrapper files here
match the previously validated CAPI v2; the runtime checks ABI version, 32-byte
sectors and the 1-ns native clock. A five-symbol legacy .so is incompatible.

```bash
python research/moe_dispatch/native_support/build.py \
  --source "$RAMULATOR_SOURCE" --fmt "$FMT_SOURCE" --yaml "$YAML_SOURCE" \
  --out "$NATIVE_BUILD" --cmake "$CMAKE_BINARY" --cxx "$CXX_BINARY"

RUSTFLAGS="-L native=$NATIVE_BUILD/ramulator_src -C link-arg=-Wl,-rpath,$NATIVE_BUILD/ramulator_src" \
  cargo build --release --features native-hbm \
  --manifest-path research/moe_dispatch/rust/Cargo.toml

python -m research.moe_dispatch.native_profile \
  --inputs "$FROZEN_INPUTS" --out "$CAMPAIGN_OUTPUT" \
  --binary "$RUNTIME_BINARY" --jobs 32

python -m research.moe_dispatch.native_profile_report \
  --campaign "$CAMPAIGN_OUTPUT" --native-build "$NATIVE_BUILD/native_build.json"
```

`PLENA_DISPATCH_COMPILER` must point to the matched compiler's
`research/moe_dispatch` directory. `PLENA_DISPATCH_TEST_BINARY` selects the
feature-enabled binary for `test_native_profile.py`. A separate system Python
can orchestrate Rust subprocesses; the ctypes memory-only probe needs a Python
whose libc is compatible with the native library.

## Frozen campaign

BF16; 1 GHz core; PM organizations `[6]`, `[3,3]`, `[4,2]`, PN=4, PK=512.
Equal totals: 12,288 multipliers; 10 weight slots/40 KiB plus 8 KiB returns;
12 KiB X double buffers; 2 MiB accumulator/workspace (including resident X/Z/Y
and combine storage in this runtime); W/X/workspace bank counts 64/24/12;
4 KiB total control budget. Equal gross capacities/bank counts are not an
iso-area claim. Both cores share one eight-channel HBM2_2000 configuration,
256 credits and the eight-sector/cycle DMA request frontend. The complete
resolved timing, FRFCFS scheduler, open-row policy, mapper and addresses are
saved with every run.

The compiler lays out private storage; dynamic dispatch assigns whole experts
using a bounded eight-entry window and capacity checks. N groups have four
tiles. Next can reserve one existing W slot; stock-based arbitration chooses
individual DMA sectors. These policies are identical in all organizations;
this campaign does not retune them to make a particular organization win.

All 108 frozen heldout decode windows are used: BFCL/GPQA/SWE, 27 per B2/B4/B8/B16.
Each of 324 points runs twice in separate processes; full reports must match.
Counts must conserve accepted requests, callbacks, landing writes, useful MACs
and weight bytes, and all queues/credits/slots must drain. Synthetic numeric
tests replay the actual native DMA/issue/commit schedule with BF16 payloads;
the large captured-route timing runs themselves do not execute model tensors.

## Four measured columns

- Fetch span: first accepted weight request to final native return callback.
  Includes supply gaps; final SRAM landing is counted in total completion time.
- Fetch bandwidth: weight bytes / fetch span, in decimal GB/s.
- Compute active time: union of operand-ready to Dot-completion intervals,
  across all tiles and cores. Arithmetic tail is the configured 20-cycle model.
  This is measured from the live schedule, not a compute-only counterfactual.
- Equivalent compute rate: 2 × useful MACs / compute active time, in GFLOP/s.

Pipeline occupancy including operand feeding is recorded separately. Frontend
state counters form an exclusive per-core partition but can overlap backend
MAC activity. Outstanding-read cycles are not HBM bus-busy cycles. Fetch and
compute spans overlap and must not be added into an end-to-end breakdown.
Rates aggregated across windows use summed work / summed measured time;
latencies use arithmetic means. Raw per-window and per-core outputs are retained.

The memory-only request-order probe holds addresses, byte counts and native
settings fixed while varying DMA burst order. It is a diagnostic for stream
interference, not a replacement FFN benchmark or an optimized design result.
