# Software projection and bounded recurrent access — 2026-10-01

The main candidate removes the historical dedicated projection replay,
input-slice, M_MM.P and S4 extensions. The Compiler emits existing Matrix
opcodes with static-transposed weights and legal SRAM allocations. The
recurrent access/update extension remains. This archive contains **100
analytical cases**, **27 machine-executed Matrix checks**, and **two real-weight
B1 one-token sublayer checks**.

Times are the entire batch's one-token recurrent sublayer, from input
normalization through output projection, at a modeled 1 GHz. Weights and
persistent state are BF16. Outer residual, FFN/MoE and full-model execution
are excluded. These are analytical predictions checked against representative
Rust programs, not GPU or silicon measurements.

## Controlled comparison

- Previous mapping: resident M_MV, K256, full-batch request group and 58 Vector
  SRAM rows; ordinary Vector recurrence.
- Software projection: static N32-by-K1024 weights, M_TMV/M_MV_WO and the same
  ordinary Vector recurrence.
- Proposed recurrence: the same software projection, native L_TILE supply,
  L=256, II=2, update latency=6, FP32 update intermediates and BF16 tree reduction.

The first change is a Compiler benefit on the common bounded Matrix/view
substrate. The second changes recurrence hardware and its declared arithmetic.
Neither is labeled the best unmodified original PLENA. A separate packed/native
L_TILE comparison keeps recurrence arithmetic and the FSM fixed.

| Model | B | Previous mapping + Vector, ms | Software projection + Vector, ms | Software projection + L_TILE, ms | Compiler ratio, Vector fixed | Recurrence ratio, projection fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mamba | 1 | 3.211801 | 1.958903 | 1.814291 | 1.6396× | 1.0797× |
| mamba | 2 | 4.788285 | 2.655792 | 2.362590 | 1.8030× | 1.1241× |
| mamba | 4 | 8.277965 | 3.890506 | 3.305475 | 2.1277× | 1.1770× |
| mamba | 8 | 16.869521 | 6.321879 | 5.138199 | 2.6684× | 1.2304× |
| mamba | 16 | 33.377727 | 11.674227 | 9.339455 | 2.8591× | 1.2500× |
| kda | 1 | 37.178647 | 23.706121 | 20.796339 | 1.5683× | 1.1399× |
| kda | 2 | 56.500550 | 31.037064 | 25.212365 | 1.8204× | 1.2310× |
| kda | 4 | 105.513435 | 45.768100 | 34.118793 | 2.3054× | 1.3414× |
| kda | 8 | 204.551970 | 77.450970 | 54.144592 | 2.6411× | 1.4304× |
| kda | 16 | 403.079034 | 149.628337 | 102.985753 | 2.6939× | 1.4529× |

All four controlled arms, HBM bytes, combined ratios and matched-arithmetic
packed/native ratios are in [sublayers.csv](sublayers.csv). Do not attribute
whole-sublayer ratios entirely to bank-conflict removal.

## Stage distribution

The table below is the new software projection + native L_TILE arm, in ms.
Each stage already includes its DMA; do not add DMA again.

| Model | B | Input/gate projections | Output projection | Norm | Conv | Gate | Layout | Recurrence | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mamba | 1 | 1.255595 | 0.496449 | 0.002519 | 0.004543 | 0.002594 | 0.011598 | 0.040993 | 1.814291 |
| mamba | 2 | 1.627615 | 0.608220 | 0.005854 | 0.010063 | 0.005997 | 0.024324 | 0.080517 | 2.362590 |
| mamba | 4 | 2.228768 | 0.821463 | 0.011936 | 0.018076 | 0.012751 | 0.048595 | 0.163886 | 3.305475 |
| mamba | 8 | 3.415463 | 1.219244 | 0.025063 | 0.033509 | 0.024681 | 0.095515 | 0.324724 | 5.138199 |
| mamba | 16 | 6.091666 | 2.237728 | 0.050758 | 0.067756 | 0.049303 | 0.191308 | 0.650936 | 9.339455 |
| kda | 1 | 16.512284 | 4.041009 | 0.051732 | 0.023979 | 0.010638 | 0.008853 | 0.147844 | 20.796339 |
| kda | 2 | 19.876952 | 4.844500 | 0.106980 | 0.047904 | 0.022858 | 0.018972 | 0.294199 | 25.212365 |
| kda | 4 | 26.660804 | 6.472850 | 0.213407 | 0.095260 | 0.044991 | 0.035794 | 0.595687 | 34.118793 |
| kda | 8 | 41.668561 | 10.502753 | 0.434324 | 0.189884 | 0.090038 | 0.070534 | 1.188498 | 54.144592 |
| kda | 16 | 79.387193 | 19.661412 | 0.863442 | 0.381364 | 0.179538 | 0.141426 | 2.371378 | 102.985753 |

All comparison arms and their issue/scalar/SRAM/arithmetic/dependency/DMA and
HBM read/write bytes are in [stages.csv](stages.csv). Individual operators are
in [operators.csv](operators.csv). Projection and HBM costs still limit the
whole-sublayer improvement.

## What changed

`lower_software_projection` searches request groups and 58/64 software-owned
Vector SRAM rows, explicitly reloading weights between groups.
`lower_transposed_projection` stores each N32 weight packet as output rows
with contiguous K, then uses M_TMV. K1024 fills the modeled reduction width
instead of invoking the full tree four times on K256 slices. Four output rows
are read serially into common Matrix operand registers. No replay buffer,
arbitrary slice network, extra SRAM port, S4 tree or concurrent engine is assumed.

For a complete K1024×N32 example, the earlier mapping takes four packets of
256 arithmetic + 40 SRAM cycles; the transposed packet takes 256 + 40. This is
a 4× local service difference, not an 8× complete projection speedup. Full
sublayers include issue, input supply, output handling and Ramulator service.

Weights are transposed offline, once for static inference weights, not per
token. K1024 changes BF16 reduction grouping relative to K256. Its own numerical
reference is checked, but bit equivalence to the earlier grouping is not
claimed. Runtime NVFP4 with this layout is rejected until block/scale placement
and decoding have their own implementation and checks.

Original packed M_MM lowering also received an independent address fix:
K-slice column offset BLEN → BLEN×MLEN. Five functional tests exercise live
partial sums across two weight loads. They do not calibrate full-size M_MM
performance; the padded fixture is not offered as an efficient mapping.

## Resources and hardware boundary

- Common reference: 4×1024 Matrix multipliers, 1 MiB / 64-bank Matrix SRAM,
  256 KiB Vector SRAM and finite operand registers. Matrix views are common
  to both arms.
- Memory: 16 independent HBM2 controllers, 32 GiB capacity, configured peak
  256 GB/s, bounded 32 read / 32 write requests. These are candidate settings,
  not 16 HBM stacks or an audited original RTL DMA window.
- Added dedicated projection storage/opcodes: zero relative to that common
  reference. This does not prove the original PLENA RTL has the required BF16
  feed/hold behavior; profiled M_TMV timing is new simulator work.
- Recurrent payload: 2 KiB state staging + 4 KiB coefficient sectors + 4 KiB
  invariant/residual holding + 2 KiB result holding = 12 KiB. Descriptors,
  tags, control, pipeline registers, alignment/broadcast and arithmetic add cost.
- The 256 update lanes keep FP32 intermediates and commit BF16 state. The dot
  uses the existing BF16 Vector tree, not dedicated FP32 context SRAM.
  II=2 allows at most 128 update elements/cycle.
- Persistent state stays in HBM; the active head group uses existing Matrix
  SRAM. Weights and state follow explicit lifetimes. No free overlap or
  cross-stage zero-copy transfer is credited.

The historical 22.25 KiB projection payload and optional 128-byte S4 roots
are removed from the main candidate, not deleted from historical code.
Capacity removed is not an area/power number. The historical hardware-assisted
S4 candidate can still be faster at B16. Pure software simplifies the design;
it has not been shown to dominate every hardware alternative.

## Verification and scope

[validation.json](validation.json) contains observations and calibration;
[evidence.json](evidence.json) binds cases to source/program/runtime hashes.

| Check | Completed evidence | Limit |
| --- | --- | --- |
| Resident software projection | 10 cases; B1/2/4/8/16, full/half request groups, distinct inputs and K/N tails | Small representative Matrix programs |
| Original packed M_MM | 5 cases; 1,984 active + 320 padding values exact; two K reloads | Functional only; full-size timing remains open |
| Static-transposed M_TMV | 12 cases including held-out K512, B2/B8, K3841/N33; 4,360 values exact | Tested geometry only |
| Mamba real-weight B1 | 624,320 values exact to defined arithmetic; relative L2 vs FP32 formula 0.00574424 | One token, conservative correctness packing |
| KDA real-weight B1 | 1,968,640 values exact; output relative L2 vs captured native 0.01757243 | One captured token, different external reference from Mamba |

The 22 cycle-calibrated Matrix cases and two real sublayers match Python/Rust
total and all six exclusive components exactly, with HBM traffic checked.
Both use the same resource contract and Ramulator: this is implementation
consistency, not independent silicon validation. The 100 full-shape cases are
analytical compiled-program evaluations, not 100 numerical Rust executions.
The real sublayers use conservative correctness packing, not the native
performance programs. Checkpoints are decoded offline to BF16. No new
compressed-weight runtime or long-chain quality acceptance follows.

Code checks: 291 Python performance tests passed (9 skipped); Rust formatting,
clippy and 329 workspace tests passed; the changed Compiler packed-address
regression passed. A wider packed Compiler selection has one pre-existing
Qwen comment-string assertion failure, also reproduced without this change.
Full Compiler CI is not claimed green.

## Reproduce

Use the Simulator checkout's Nix environment and its pinned dependencies.
Both repositories publish this version on `feat/matrix-sram-lcompute`;
initialize the exact Compiler revision with `git submodule update --init --recursive`.
No GPU or model download is required for the analytical sweep or synthetic checks.

```sh
git submodule update --init --recursive
nix develop --no-write-lock-file
export PLENA_COMPILER_ROOT="$PWD/PLENA_Compiler"
export PYTHONPATH="$PWD:$PLENA_COMPILER_ROOT:$PWD/PLENA_Tools:$PYTHONPATH"
export RUN_ROOT=/tmp/plena-software-projection
export CARGO_TARGET_DIR=/tmp/plena-software-target
python -m analytic_models.performance.ltile_dma --prepare "$RUN_ROOT/memory" --controllers 16
python -m analytic_models.performance.projection_software --memory-root "$RUN_ROOT/memory" --output "$RUN_ROOT/resident" --paired --reference-pairs --workers 4
python -m analytic_models.performance.projection_software --memory-root "$RUN_ROOT/memory" --output "$RUN_ROOT/transposed" --transposed-k 1024 --paired --reference-output "$RUN_ROOT/resident" --workers 4
python -m analytic_models.performance.projection_software --memory-root "$RUN_ROOT/memory" --output "$RUN_ROOT/packed" --transposed-k 1024 --coefficient-supply packed --workers 4
LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH" cargo build --release --manifest-path transactional_emulator/Cargo.toml
python -m transactional_emulator.testbench.models.unified_service_test --runtime "$CARGO_TARGET_DIR/release/transactional_emulator" --memory-root "$RUN_ROOT/memory" --output "$RUN_ROOT/check-transposed" --only transposed
```

Use `--only software` and `--only legacy_mm` for the other machine checks.
Real-layer checks use `recurrent_layer_test --help`, `--transposed-k 1024`, and
external fixture paths/commands in validation.json. Weights, captured tensors,
raw memory traces and builds are not bundled. Case/source hashes, cycle sums,
stage coverage and traffic totals were checked at export.

The sweep is 60 resident candidates + 10 historical Vector references + 10
transposed native + 10 transposed Vector + 10 transposed packed L_TILE cases.
ZIP files contain compact result JSONs. Absolute provenance paths identify
the original worktree; hashes identify content. [SHA256SUMS](SHA256SUMS)
covers delivered artifacts.

## Research interpretation and next gates

The architecture studies direct consumption of compact coefficients and
private state through bounded shared-SRAM views, while retaining effective
software projection mapping. Generic tiling, WS/IS/OS, descriptors, diagonal
storage, hybrid support and two-engine diagrams are insufficient novelty.
The [design and primary-source review](../../../doc/l_tile_projection.md)
compares PLENA, Timeloop, Gemmini, LoopTree, HLX, MARCA, HEMERA and tensor maps.

Unfinished: best legal original-M_MM service and feed/hold verification;
transposed NVFP4 layout/decoder; final long-chain/task quality; whole-model
capacity/operator calibration; integrated recurrent auxiliary modes and
hardware cost evidence. Committed-BF16 update-to-readout forwarding is a
documented, unimplemented candidate. No new full-model, TTFT, PPA, energy or
GPU speedup is asserted.
