# Experimental KDA FP32 dot execution

This opt-in control adds an 8192-byte FP32 vector accumulator plus validity at
VLEN=2048. It is outside the frozen zero-added-storage L-Compute architecture.
See [encoding, storage and timing assumptions](../PLENA_Compiler/doc/EXPERIMENTAL_FP32_DOT.md).

From the Simulator repository root:

```bash
PLENA_COMPILER_ROOT=PLENA_Compiler .venv/bin/python -m transactional_emulator.testbench.aten.matrix_lcompute_execution_compare --model kda --batches 1 2 4 8 16 --tokens 2 --experimental-fp32-dot --output-dir /path/to/new-results
```

The CLI enables `PLENA_EXPERIMENTAL_FP32_DOT=1` only within the emulator call.
Default lowering remains ordinary BF16 VV and default Rust rejects the new
RESET opcode. ACC/WRITE require a live accumulator, and WRITE invalidates it.

`--seed 17 --batches 1 --tokens 8 --snapshot-states` checks every intermediate
state. These snapshots execute extra DMA and their timing never qualifies a
performance ratio. The 1% relative-L2 and existing per-element budgets remain;
each variant must also match its own independent rounding reference exactly.

A/B are executable controls with explicit/static-reused addresses. Their
coefficient expansion adds HBM traffic relative to D's compact fields. None of
these kernels executes checkpoint weights or a complete model. Passing the
numeric budget does not establish bitwise equivalence to D or silicon PPA.

Validation for the implementation: Compiler 218 passed, Simulator Python 157
passed, Rust workspace 305 passed. Cancellation/reset and bank charges have a
Rust regression; report tests prevent snapshot/mixed-contract/stale-file ratios.
The default four Mamba/KDA A/B programs remain byte-identical to their previous
version. Large executable results and CPU/GPU archive cross-checks are stored
separately in the shared KDA_FP32_DOT_EXPERIMENTAL_20260905 artifact directory.
