# L_TILE decode model (R3)

The supported recurrent model is `analytic_models.performance.ltile_cost`, with
the Compiler adapter in `ltile_program` and the memory-only backend in
`ltile_dma` / `ltile_memory.cc`. It uses L=256, update II=2, latency=6, BF16 SRAM
tree reduction, bounded operand credits, finite Matrix/Vector ports and serial
instruction retirement. L=128/512 and FP32 dot context are explicit sensitivity
settings. Clock is the R3 assumption of 1 GHz, not a measured chip frequency.

There are three interfaces: optimized old ISA; row instructions with the new
arithmetic/access path; and L_TILE FSM with the same new arithmetic. Old ISA and
v2 do not share rounding boundaries. Only row/FSM isolates control scheduling
under identical arithmetic. Both comparisons share DMA and surrounding work.

## Memory and calibration

An aggregate bytes/startup DMA fit failed held-out validation. The accepted
implementation therefore composes analytical compute/port scheduling with the
pinned Ramulator memory backend. It emits addresses and non-memory delays from
Compiler assembly; it neither executes tensor arithmetic nor reads measured
cycles as predictor inputs. The C++ helper uses the same DRAM library as Rust.
Agreement checks timing integration, not independent DRAM or RTL correctness.

The original writer serializes a read-modify-write per 64-byte block. It is not
the optimized aligned writer with window=1. Windows 1/64 are sensitivities that
also remove redundant reads. The primary model retains the original service.

`ltile_calibration` preserves the preregistered B1/L256 split and reports held-out
L128/512, private B2–B16, DMA windows and port/feedback changes. The final model
fits zero observed cycle parameters. It compares issue, scalar control, SRAM,
arithmetic, dependency, DMA and total; speedup error has a separate gate. The old
`test_kda_stage_calibration.py` remains an instruction-count regression only.
Development did inspect failures in the historical validation set. Those results
are configuration validation, not a one-shot blind test. The completed campaign
also records four new B3/T3 combinations with predictions saved before executing
Rust, without changing or refitting the predictor afterward.

## Reproduction

Use the project Python environment. `EVIDENCE` is the archived v2 directory,
which contains the frozen E Compiler, E runtime and E/E2 measurements. `OUT`
is a result directory outside the source tree. `RAMULATOR_LIB` must be the same
pinned library used by that runtime. The exact paths, hashes, captured DRAM JSON
and commands for the completed campaign are supplied in its result manifest.

```sh
c++ -std=c++17 -O2 analytic_models/performance/ltile_memory.cc \
  -L"$RAMULATOR_LIB" -Wl,-rpath,"$RAMULATOR_LIB" -lramulator \
  -o "$OUT/ltile_memory"

python -m analytic_models.performance.ltile_calibration \
  --evidence "$EVIDENCE" --output "$OUT/calibration" \
  --memory-binary "$OUT/ltile_memory" --memory-config "$OUT/ramulator.json" \
  --memory-cache "$OUT/memory_cache"

python -m analytic_models.performance.ltile_decode \
  --evidence "$EVIDENCE" --output "$OUT/decode" \
  --memory-binary "$OUT/ltile_memory" --memory-config "$OUT/ramulator.json" \
  --memory-cache "$OUT/memory_cache" \
  --campaign "$AGENTIC_ARCHIVE" --gate "$OUT/calibration/validation.json"
```

`ltile_decode` refuses to run unless the recurrent calibration gate passed.
It covers the 52/93-layer decode workload at B1/2/4/8/16 and contexts
4K/32K/128K, reporting per-step time, aggregate TPS, exclusive stage and movement
breakdowns, and both speedup comparisons. Recurrent pricing compiles each batch
with private addresses; it does not multiply a B1 timing by B.

## Interpretation limits

Surrounding operators are shared tiled analytical estimates. They are not
validated complete Compiler/Rust schedules. The 4096-PE Matrix profile logged by
R3 is metadata; current MatrixMachine cycle charges are not derived from that
geometry. This is explicitly a remaining Matrix/attention/MoE calibration gap.

Prepared coefficients are produced and materialized explicitly in the full
composition. KDA key re-reads are deduplicated when charging their producer.
Coefficient preparation/lowering itself remains an analytical mapping.

Nemotron uses the 48-input, 93-group archived routing replay, with active-expert
unions and Matrix row occupancy. Routing was not newly collected at the swept
contexts. Kimi reports minimum/maximum unique-expert bounds. Both use the
requested NVFP4 weight storage study with BF16 exclusions. Kimi's routed-expert
NVFP4 policy is a storage hypothesis; its source checkpoint policy is MXFP4.

Capacity counts all experts, the full embedding table, persistent state and KV,
not just active weights per step. The R3 16-GiB point fails the full-model weight
capacity check. Such rows are conditional performance estimates, not a valid
no-offload system configuration. No external weight streaming is silently added.

No TTFT/prefill, GPU comparison, power, energy, complete RTL or task-quality
claim is produced. BF16 state quality and auxiliary product/residual RTL
integration retain their separate outstanding verification requirements.
