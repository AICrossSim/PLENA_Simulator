# Round A: equal-MAC geometry screening

This is a new, separate Python experiment. It does not change the frozen supply-first v3 engine, defaults, input captures or evaluation gates. Its result is an architectural screening signal, not full HBM end-to-end latency or a synthesized hardware configuration.

## Search and selection

The physical dimension order is **M × N × K**. The fixed main compute budget is 12,288 MACs, equal to `6×4×512`. K is spatial with `P_K=512`; it is not a serial K traversal through a conventional 2-D systolic array. We enumerate integer `P_M=1…8`, `P_N=1…24` factorizations of 24 dot-product outputs: six single-core geometries and 90 unique two-core geometries, of which five have identical core shapes and 85 have different shapes. Mirror pairs are one design.

The current Rust/compiler datapath only implements `P_N=4`, `P_K=512`. Other N widths in this experiment are **prospective hardware**. The search does not assert that current Rust can run them.

The geometry search uses one common earliest-estimated-finish whole-expert scheduling rule. We choose the best single, best identical dual-core and best asymmetric dual-core using only the sum of development layer makespans. Hardware is then frozen. A separate runtime sweep chooses a threshold from `{1,2,3,4,6,8,12,16}` on each selected geometry, still using development data only. The complete geometry × runtime table is retained, but its jointly optimized winner does not replace the geometry selected under common EFT.

Shared experts follow the same task eligibility rule as routed experts. EFT can use any core. Under the threshold policy, “stream” means the core with the smaller physical M dimension; if M is equal, the core with fewer MACs wins this tie. It does not necessarily have fewer MACs when M differs. Only physically identical core shapes bypass a threshold and use EFT among both cores. Every architecture uses whole-expert ownership, the input descriptor order, FIFO per core, and no expert row packing, cross-core splitting or stealing.

## Compute timing and dependency model

The defaults are taken from the existing v3 compute contract: initiation interval one cycle, dot latency 20 cycles, followed by one cycle of result commitment. The original v3 sources are `rust/src/v3/mod.rs` (dot scheduling/default vector length) and `rust/src/v3/IMPLEMENTATION_CONTRACT.md` (per-output K dependency); the earlier projection oracle in `rust/src/compute.rs` uses the same 21-cycle issue-to-result completion.

An output group contains up to eight N tiles and all its M blocks. K segments visit these independent outputs in the same order. The next K segment of one output waits for its previous result to commit. All outputs of a group complete before the next group. For `q = ceil(M/P_M) × group_N_tiles` records and `s = ceil(K/512)` K segments, the group's final result commits after

```
(s−1) × max(q × II, dot_latency + commit)
  + (q−1) × II + dot_latency + commit
```

This expression is verified against an independent issue-event implementation in the tests. It allows independent dot products to overlap; it does not allow dependent K segments to overlap incorrectly. `waves × pipeline_depth` is not used. The output group bounds the independent records to `8 × ceil(Me/P_M)`; there is no additional eight-in-flight-result cap. The older `compute.rs` oracle does have that extra eight-entry restriction. This model therefore inherits timing constants, not the exact old Rust oracle timing or its complete context microarchitecture. `SOURCE_PINS.json`, written before selection freeze, identifies the exact source files and this distinction.

For an expert with hidden dimension H and intermediate width F, the actual modeled work is:

```
Gate/Up: X[M,H] × concatenate(W_gate, W_up)[H,2F]
  → one shared vector service for SiLU/product
Down: Z[M,F] × W_down[F,H]
  → one shared vector service for weighted output combine
```

This uses exactly `3 × M × H × F` useful main MACs. There is one global vector engine, 64 elements/cycle, across all organizations. Its contention is scheduled by events; adding a core does not add a second vector engine. An expert remains on its owner core throughout all stages. Stage-level barriers and the eight-tile output-group retirement are deliberate conservative assumptions, not a claim to reproduce every instruction in v3's more detailed pipeline.

At 1 GHz, one cycle is one nanosecond and one million cycles is one millisecond. Layer latency starts after all Router results are available and ends after the modeled output combine. HBM, DMA, SRAM-bank service, paid runtime control service, attention and Router timing are absent.

## Inputs and holdout

The four archived captured input files are `development.json`, `mixed_development.json`, `heldout.json`, and `mixed_heldout.json`. No random or Zipf routing is generated. We verify token indices, top-k memberships, expert populations and exact useful MAC totals, and record all source hashes. Development includes 12 captured decode windows and six captured mixed decode/prefill windows. Evaluation includes 108 decode windows and 27 mixed windows. These are correlated layer/window/prefix samples, not 153 independent complete model executions.

`FROZEN_ROUND_A.json` is written and hashed before this program first opens the heldout files. The selected hardware and runtime rules are never revised using heldout results. The same underlying v3 heldout traces were previously exposed for correctness diagnosis, so this experiment is **not a pristine blind holdout**; the new geometry selection is nevertheless development-only.

Every development point and selected heldout point runs twice, and complete serialized layer results must match. A repeat is a determinism check, not an independent statistical sample.

## What resource equality means here

All geometries have exactly 12,288 main MACs, BF16 operands, FP32 sums and a 1 GHz assumed clock. All record the same nominal 2,158,592-byte SRAM budget. The program checks a necessary minimum footprint for double-buffered weight tiles, double-buffered X tiles, an output group, control and return bytes.

That minimum ignores persistent Z/U, the full output combine, bank counts, port widths, routing, pipeline registers and timing closure. Passing it **does not certify physical SRAM feasibility, equal area, equal power or equal memory bandwidth**. New M/N splits change required buffers and ports. Round B must model and constrain them before any design is called a complete equal-resource architecture.

The frontier is latency versus spatial utilization at fixed main MAC count. It is not an area Pareto frontier; area data does not exist.

## Outputs

`development_search.csv/json`: every geometry × runtime policy and minimum-footprint check. `FROZEN_ROUND_A.json`: development-only hardware and subsequent runtime selection. `heldout_layers.csv/json`: exact per-layer measurements for the nine frozen points. `heldout_summary.csv/json`: decode, mixed and combined totals, p95 and comparisons against both the fixed `6×4×512` baseline and the best searched single/identical-core designs. `teacher_toy.csv/json`: fixed-owner `Me=4,2`, `N=128`, `K=512` projection. `input_provenance.json`, `CONCLUSIONS.json`, and `REPORT.md` retain scope and limits.

Spatial utilization is `useful / issued_MACs` and describes padding only. Wall MAC utilization is `useful / (installed_MACs × layer_cycles)` and additionally includes pipeline gaps, vector time and final core imbalance. They are not interchangeable.

## Reproduction

```bash
python -m pytest -q research/moe_dispatch/test_round_a_dse.py
python research/moe_dispatch/round_a_dse.py \
  --inputs outputs/moe_supply_first_v3/inputs \
  --output /tmp/plena-round-a-dse-20261003
```

The output directory must be new; earlier experiments are never overwritten. The task's requested 2-D systolic approximation `T_K + P_M + P_N − 2` would model different hardware and is therefore intentionally not substituted for PLENA's spatial-K dot-product pipeline.
