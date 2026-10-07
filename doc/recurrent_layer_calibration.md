# Recurrent layer calibration: executable boundaries

This revision adds two foundations for full-layer calibration. It does not
certify a complete Mamba/KDA layer or change the formal decode acceptance gate.

| Boundary | Execution | Evidence still required |
|---|---|---|
| Mamba raw dt + bias + negative A → dt, delta | Compiler, assembler, Rust, ordinary BF16 Vector instructions | conv, b/c, packing, projection, connection and long-chain quality |
| KDA raw g + bias + A_log + raw beta → delta, beta | Same path; bounded decay formula only | q/k normalization, conv, packing, projection, connection and long-chain quality |
| Matrix split-K mini-arrays + cross-array tree | Independent Rust clock-stepped numerical probe, Python cost formula | Matrix ISA integration, physical bank schedule, codec/DMA, integrated RTL |

Gate arithmetic rounds to BF16 after each instruction. Sigmoid uses
`exp(-softplus(-x))`; delta uses the separately labelled rational candidate.
These outputs differ from ideal FP32-prepared coefficients. The test compares
both the instruction-level contract (bitwise) and captured GPU-prepared values
(absolute/relative error), and never substitutes the latter into the program.
Packing/padding is explicit in the input fixture and excluded from producer
cycles. Output regions are poisoned; every other HBM byte must remain unchanged.
The input fields use existing captures, so no GPU is needed for these tests.

## Service contracts

The unified gate path requires an explicit finite SFU profile. The experimental
point is 32 lanes, II=1, with exp/softplus/reciprocal latencies 8/16/8 cycles.
These are **candidate budgets**, not measured or synthesized latencies. A vector
takes `(ceil(VLEN/lanes)-1)*II + latency`; both SRAM accesses are charged and
the instruction retires only after the last result. Historical recurrence-only
reciprocal timing is unchanged. Environment settings are opt-in; global precision
defaults and the frozen R7 executable are not modified.
This schedule requires one held input row and one result row: 4096 bytes each
at VLEN=2048/BF16. Reuse of existing Vector latches is not yet established; the
8 KiB holding requirement must remain in the resource ledger until integration.
Softplus/exp/reciprocal numerical execution uses libtorch; an RTL approximation
must be checked against this contract before the latencies can be accepted.

Matrix topology follows the parallel square mini-arrays in
`mx_systolic_mcu.sv` and `mx_sum_across_sa.sv`, with an explicit registered
reduction tree. At edge=4, reduction width=1024 there are 4096 multipliers **and
4080 cross-array adders**. Those adders, PE partial sums, tree registers, input
latches, and result holding are reported explicitly. Capacity ratios do not
estimate area. A dependent accumulator cannot issue faster than its feedback
latency. Tail work is padded and charged; completion includes accepted output
writes. This candidate is not silently identified with the existing
`MatrixCoreProfile` label or the current `M_MM` timing.

The gate fixture uses 1 MiB Matrix SRAM, 256 KiB Vector SRAM, 1 GHz assumed clock,
and bounded 32-read/32-write DMA. Rust reads `[TRANSACTIONAL]` settings even if
the shared TOML's `[MODE]` selects Python analytic mode. Numerical coefficients
and SRAM are BF16. This experiment contains no compressed model weights and
provides no NVFP4 decode/scale-throughput evidence.

## Reproduction

Initialize the pinned Compiler submodule. Build within the repository's Nix
development environment; use Python 3.12 with the existing simulator packages.
New output directories are required, so failed or historical evidence is never
silently overwritten.

```sh
git submodule update --init PLENA_Compiler
nix develop -c cargo build --locked --manifest-path transactional_emulator/Cargo.toml --bins
python -m analytic_models.performance.calibrate_matrix_service \
  --probe "$PWD/transactional_emulator/target/debug/matrix_service_probe" \
  --output /absolute/path/to/new_matrix_reference
```

The gate test takes an existing capture directory and the memory-only Ramulator
backend directory (`ltile_memory`, `ramulator.json`, `cache`). Its result records
the exact runtime, configuration, capture and Compiler hashes, invocation,
resource profile, predicted/observed components and coverage exclusions.

```sh
python -m transactional_emulator.testbench.aten.recurrent_gate_test \
  --kind mamba --layer 46 --tokens 32 \
  --capture /absolute/path/to/captures_short/0_bfcl_v3 \
  --runtime "$PWD/transactional_emulator/target/debug/transactional_emulator" \
  --memory-root /absolute/path/to/memory \
  --output /absolute/path/to/new_gate_result
```

For KDA use `--kind kda` and its capture directory, which must contain
`static.npz` and `flags.json`. The loader rejects unsupported decay formulas.
Use `--sfu-lanes 16` or `64`, with `--sfu-ii 2`, for held-out service checks.
Predictions are written before execution. Gate DMA shares Ramulator with Rust;
agreement validates integration, not an independent DRAM timing model.

## Promotion to full-layer evidence

Do not mark `projection` or `connected_layer` passed from this reference probe.
First connect actual Matrix instructions and coefficient producers with bounded
live allocations, add packing/codec/DMA costs, and check hidden-input to
hidden-output values. KDA q/k normalization and both models' conv/gating/output
paths remain mandatory. Then validate other shapes and batches before assembling
full decode. Keep RN/SR and candidate/ideal coefficient contracts separate.
