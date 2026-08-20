# Connected Hybrid Validation

This document separates five validation levels that must not be reported as
equivalent.

1. **Compiler structure**: every model stage emits instructions.
2. **Full symbolic machine code**: every decode layer assembles to legal 32-bit
   words, while checkpoint parameter ranges remain unresolved in a manifest.
3. **Connected compiler dataflow**: each stage consumes the tensor returned by
   its producer, including residual ownership and fixed-address `X_STATE`
   handoffs.
4. **Compact Rust numerical execution**: deterministic tensors and a compact
   HBM image execute in the transactional emulator and compare with a CPU
   reference.
5. **Full-model Rust execution**: all real model weights and persistent caches
   execute for every layer. This level is not implemented.

## Numerical Gates

The compact tests use deterministic synthetic weights. KDA and Mamba retain
their real recurrent-state dimensions; the surrounding hidden width is reduced
where needed to keep the HBM image small enough for a correctness test.

| Model path | Rust cycles | Maximum absolute error | Result |
|---|---:|---:|---|
| Kimi MLA | 46,603 | 0 | pass |
| Kimi latent MoE | 24,297 | 0 | pass |
| Kimi MLA -> latent MoE | 71,097 | 0.001953125 | pass |
| Kimi AttnRes | 3,852 | 0 | pass |
| Kimi AttnRes -> MLA -> AttnRes -> MoE | 78,900 | 0.00390625 | pass |
| Kimi KDA | 72,342 | 0 | pass |
| Kimi KDA -> MoE | 94,523 | 0 | pass |
| Kimi AttnRes -> KDA -> AttnRes -> MoE | 96,980 | 0 | pass |
| Nemotron routed + shared MoE | 14,275 | 0 | pass |
| Nemotron real-size Mamba state core | 1,710,927 | 0.015625 | pass within BF16 tolerance |
| Nemotron Mamba -> MoE | 1,725,603 | 0.046875 model-level; 0 at the physical handoff | pass |

The compact Matrix loops have two additional Rust numerical gates. An MXFP8
`1x320 @ 320x384` projection traverses two K chunks and six N tiles in 93
instructions, takes 38,215 emulator cycles, and returns all 384 values exactly.
A BF16 stream-K `1x320 @ 320x128` projection traverses five K tiles in 71
instructions, takes 37,596 cycles, and returns all 128 values exactly. These
tests validate nested-loop address progression and accumulator lifetime; they
are not full-layer or full-model performance measurements.

The Nemotron two-layer test has two independent checks. The complete path is
bounded against the CPU formula, where Mamba's 128-element sequential
reduction and BF16 rounding accumulate error. The test also reads the actual
Mamba output from the Rust VRAM dump and uses it as the CPU MoE input. That
second check is exact, so stale addresses, dropped values, and a disconnected
producer-consumer edge cannot pass under the wider model-level tolerance.

The Mamba and KDA rows above include an executable `L_SCATTER_M` immediately
before `X_STATE`.  Both use a 64-value FIFO, 16 single-port banks, and a
64-value producer burst.  The real-size Mamba run reports a 64-value FIFO peak,
zero consumer-read bank stalls, and four layout-write stalls; increasing the
FIFO to 256 values produced the same 1,710,927-cycle result. That equality is
not evidence on its own: the Rust flow charges backpressure from the spill
width and the producer burst, never from the queue depth, so no capacity can
move this number. The depth is justified by the analytic
`ProjectionFifoSpillModel`, which steps a real queue and reports a measured
high-watermark.  The compact KDA
run reports zero layout read/write stalls.  These counters validate the
pre-RTL layout path; they do not include RTL mux delay or timing closure.

The compact Kimi and Nemotron MoE tests store each expert's 64x64 element tile
and its MX scale bytes in the same tile-major group used by the full compiler.
The expert id selected by Rust `V_TOPK` supplies the runtime byte offset. Exact
CPU agreement therefore validates the physical expert mapping and scale-stream
addressing, not only the arithmetic after a preselected host tensor.

## Reproduction

Run from the Simulator root after the matching Compiler revision is available
at `PLENA_Compiler/`. When testing a feature worktree, point the recipes at it;
if the `PLENA_Tools` submodule is not initialized, either initialize it or set
its existing checkout explicitly:

```bash
export PLENA_COMPILER_ROOT=/path/to/PLENA_Compiler
export PLENA_TOOLS_ROOT=/path/to/PLENA_Tools
just test-kimi3-connected --stage all
just test-kimi3-kda-connected --stage all
just test-nemotron3-mamba-connected --stage all
```

Each command creates assembly, HBM/VRAM preloads, a runtime stage profile, and
`summary.json`. It exits non-zero on a numerical or physical-handoff failure.

The common descriptor contract is checked across repositories with:

```bash
cd ../PLENA_Compiler
python tools/state_contract.py --simulator-root ../PLENA_Simulator --check
```

## Honest Boundary

Current connected programs are single-token decode programs. They reject
prefill and context lengths greater than one because persistent multi-token MLA
and GQA K/V-cache append/read are not implemented. The Compiler now emits two
complete symbolic-weight artifacts: Nemotron has 6,202,663 instructions
(23.66 MiB raw machine code) for 23 Mamba, 23 MoE, and 6 Attention layers; Kimi
has 11,502,370 instructions (43.88 MiB) for 69 KDA, 24 MLA, 92 latent-MoE, and
one dense-FFN block at the real 96-head shape. Every `.mem` line was assembled
as one legal 32-bit word, and every unresolved HBM parameter range is recorded
in a non-overlapping manifest. Neither artifact binds checkpoint bytes or has
been replayed from layer 1 through the final layer in Rust, so neither may be
used for an end-to-end PLENA cycle claim.

The formal B200 campaign does not change that boundary. It validates the real
KDA/Nemotron shapes and identifies the system bottlenecks: KDA's Matrix path is
62-74% of profiled kernel time, while Nemotron prefill MoE reads 8.9x as much
DRAM as Mamba. All six Nemotron NCU captures and the exact 127-step routing
trace are now local and hash-validated. They calibrate workload traffic,
checkpoint precision, routing, and the GPU baseline, not PLENA cycle latency.

The optional local KDA Stage 2 cross-check verifies that its source revisions,
manifest hashes, recurrent-core call counts, and DRAM reads agree with the
formal summary. It is an older raw subset and does not independently validate
the complete campaign.

The current Kimi MLA lowering still statically expands all 96 heads. Compact
Matrix N/K loops and looped Top-K make the 93-layer artifact bounded, but a
dynamic-address wide-head MLA loop remains desirable to reduce its 43.88 MiB
instruction footprint. Multi-token execution additionally needs the compressed
MLA cache; neither optimization is implied by the current artifact.
