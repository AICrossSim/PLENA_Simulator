# Connected Hybrid Validation

This document separates four validation levels that must not be reported as
equivalent.

1. **Compiler structure**: every model stage emits instructions.
2. **Connected compiler dataflow**: each stage consumes the tensor returned by
   its producer, including residual ownership and fixed-address `X_STATE`
   handoffs.
3. **Compact Rust numerical execution**: deterministic tensors and a compact
   HBM image execute in the transactional emulator and compare with a CPU
   reference.
4. **Full-model Rust execution**: all real model weights and persistent caches
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
and GQA K/V-cache append/read are not implemented. The full 52-layer Nemotron
compiler program uses symbolic weight addresses and is a machine-code
validation artifact, not a full-weight Rust execution. Kimi has a complete
93-layer structural trace and compact connected numerical programs, but no
full-size 93-layer machine-code artifact. Routed Top-K now uses one dynamic
expert body in a hardware loop and passes the compact Rust numerical test; the
remaining full-size guard covers both Matrix output-column/K-tile expansion and
MLA's 24 x 96 = 2,304 statically expanded head bodies. A post-Top-K `heads=1`
diagnostic still emitted 100,221,916 instructions and required 7m10s/24.1 GiB
RSS. Neither model may be used for an end-to-end PLENA cycle claim.

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

The current Kimi MLA lowering also statically expands all 96 heads. PLENA has a
hardware loop for packed GQA, but its 64-wide head slots cannot represent MLA's
192-wide Q/K and 128-wide V. A deployable Kimi binary therefore still needs a
dynamic-address wide-head MHA loop and the compressed MLA cache. Looped Top-K
expert dispatch is implemented for the current single-token decode path.
