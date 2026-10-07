# Matrix SRAM recurrence execution

The emulator executes Compiler-generated Mamba/KDA recurrence programs using
banked Matrix SRAM, explicit DMA and `L_TILE` view descriptors. Matrix projection
can write through a view; recurrent state is read back through HBM for numerical
verification. Existing Matrix accumulators are retained.

## Code entry points

| Area | Files under `transactional_emulator/` |
|---|---|
| Instruction decoding | `src/op.rs` |
| View configuration and traversal | `src/accelerator/mview.rs`, `lstream.rs` |
| Banked storage and ownership | `lib/sram/src/matrix.rs` |
| Recurrence execution | `src/accelerator/dispatch.rs`, `src/vector_machine.rs` |
| DMA and timing | `src/dma.rs`, `src/timing.rs` |
| Compiler-to-Rust checks | `testbench/aten/matrix_view_projection_test.py`, `matrix_lcompute_recurrence_test.py`, `matrix_lcompute_execution_compare.py` |

## Numerical and timing contract

`HBM_STATE_TYPE` specifies BF16 state independently of the weight, activation
and KV formats. Prepared recurrence programs contain BF16 coefficients and
state; they do not execute a full model checkpoint's weight projections.
Ordinary Vector instructions round on SRAM writeback. L_TILE retains FP32
intermediates across a recurrence primitive before BF16 writeback. Optional
FP32-dot and pairwise controls remain disabled unless explicitly selected.

The comparison harness checks each path against its own rounding reference,
then applies a common error budget before reporting a timing ratio. KDA's
ordinary sequential BF16 controls can fail that common budget even when they
exactly implement their instruction sequence; those ratios remain blank.
Timing counters cover serial issue, arithmetic, bank service and DMA waits.
They describe the executed recurrent core, not whole-model or GPU speedup.

## Run the checks

Initialize the pinned Compiler submodule, then use the configured dev environment:

```bash
git submodule update --init --recursive
nix develop --no-write-lock-file --command just test-matrix-lcompute PLENA_Compiler
```

The gate covers Compiler contracts, Rust workspace tests, projection and
official-shape fixed/phased recurrence execution. Inputs are deterministic;
no checkpoint, GPU trace or archived campaign output is required.

## Compatibility boundaries

`M_MM_WO` view encoding reuses bit17 of the old immediate. Legacy word
`0x80000046` now selects view0 instead of ordinary Vector offset 131072.
Old-binary compatibility remains unresolved. Ordinary DMA selector2 now means
State; callers relying on its former KV alias must use selector1.

L_TILE rejects logical lines wider than the configured Vector width. Its
functional FP32 intermediates still require mapping to physical storage and
datapaths; this implementation does not establish zero extra hardware, RTL
timing or PPA. Full real Nemotron/Kimi checkpoint execution is outside this gate.
