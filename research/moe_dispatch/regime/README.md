# Regime diagnosis and fair 3D geometry search

This package implements the 2026-10-05 evaluation contract. It is a bounded,
finite-resource **analytical screening model**, not a new Rust/native backend,
RTL measurement, whole-model generation benchmark, or claim of heterogeneous
hardware superiority. Existing `geometry3d` results remain frozen.

## Inputs and timing boundary

Use the four captured JSON files: `development.json`,
`mixed_development.json`, `heldout.json`, and `mixed_heldout.json`.
They contain 18 development and 135 historically exposed heldout windows from
DeepSeek-V2-Lite BFCL/GPQA/SWE captures. Hidden size is 2048, routed FFN size
1408, shared FFN size 2816, and routed top-k is 6. B2/B4/B8/B16 each have
27 heldout windows; mixed B64/B96/B128 each have 9. These are captured routed
token populations, not newly executed model inference or random Zipf inputs.

Time starts with activations and routing available and includes Gate/Up,
SiLU/Z, Down and combining service. Router, Attention and generation are
excluded. The clock is an assumed 1 GHz: 1 cycle = 1 ns; 1,000,000 cycles =
1 ms. The model advances phase-level continuous fluid progress; fractional
cycles are allowed. It does not check full FFN numerical outputs, per-bank
address conflicts or cycle-by-cycle native request/array activity.

The scheduling unit is an entire expert. Its projections stay on one core;
insufficient Z capacity causes explicit M chunks and weight reloads. This
domain does not include cross-core N-band splitting, shared partial sums,
elastic bank leasing, or a newly trained online dispatch predictor. A failure
in this search does not rule out those other architectures.

## Controlled resources and search domain

Main BF16 multipliers total exactly 12,288. PM is 1–16, PN 1–192 and **each
core's independent PK** is in {32,64,128,256,512,1024}. This gives 44 single,
41 identical-pair and 16,678 asymmetric-pair geometries before SRAM legality.
It is a flat reduction-tree engine model, not the serial-K systolic formula
`K + PM + PN - 2`.

Installed storage totals 2,158,592 B (2.0586 MiB): shared X 512 KiB, output
1 MiB, private Z 384 KiB, W 40 KiB, ingress 8 KiB, local X 12 KiB,
accumulator 96 KiB, control 16 KiB and routing 16 KiB. W/X/accumulator ports
total 64/24/12 banks, respectively, at 16 B/cycle per bank. Global activation
bandwidth is 384 B/cycle, vector service 64 operations/cycle, and control one
service cycle per cycle. Equal capacity and MACs **do not imply equal area**.

For two cores the W, local X, accumulator and Z quota fractions, and W/X/acc
bank fractions, are seven independent axes. Each arena and bank type sums to
its fixed total. Canonical core 0 is the first shape in the frozen geometry
string, not necessarily the larger core. No extra storage is silently given
to a preferred organization.

The primary protocol retains physical padded tile reservations and full
issued MAC slots, but transfers only valid M/N/K SRAM elements. Rust's
`research/moe_dispatch/rust/src/main.rs` uses `wspans` and `xspans` built from
`n_valid`, `m_valid`, and `k_valid`; this is the port-accounting convention.
FP32 N records round to 16 B words. Padding does not create payload traffic
or extra capacity. The default `Settings()` deliberately preserves the old
padded-port hypothesis so that 945 frozen v6 windows reproduce exactly.

There are eight paired output records, bounded WS/OS traversal programs and
prefetch caps {2,6,32}. Gate/Up and Down independently choose an N-group cap
from {auto,1,2,4,8}. Larger groups increase X reuse but occupy more current W
slots and leave fewer slots for lookahead. Every organization gets these
compiler knobs. Full weights serve all retained M blocks before retirement;
M chunks too large for the installed live state explicitly cause reloads.

## Credit / format grid and lower bounds

Grid: credits {256,512} × {BF16,W8,W4} × seven batches. At fixed 64-cycle
response and one landing cycle, the continuous cap is
`min(256, credits * 32 / 65)` B/ns: 126.030769 or 252.061538 GB/s.
These are **modeled upper service caps**, not Ramulator measurements.
512 credits use a fixed 8 KiB ingress plus modeled backpressure; extra 512 B
tag state is charged inside control. A native 512-credit implementation is
not claimed.

W8/W4 retain BF16 operand slots and multiplication. Only transport bytes
change: codes and group-128 FP16 scales, each row rounded to 32 B sectors.
Scale-sector retention/coalescing and the primary decoder are ideal
assumptions. Decoder sensitivities use 128/512 elements/cycle. The scheme is
not QERA/LQER, an implemented compressed-native path or qualified task
accuracy. `quant_probe.py` checks six actual pretrained matrices with
synthetic BF16 X; NMSE is not perplexity or accuracy.

For a result, use separate bound scopes:

* `hbm_lb_ms`: actual read bytes / cap, conditional on mapped reloads.
* `unique_weight_hbm_lb_ms`: one unique weight pass / cap.
* `compute_fixed_owner_lb_ms`: largest per-core compute/dependency demand,
  conditional on ownership and engine shape.
* `port_fixed_mapping_lb_ms`: largest actual traffic / installed port demand,
  conditional on mapping. Do not use it to forbid geometries reducing traffic.
* `architecture_search_lb_ms`: max(unique-weight HBM, useful MAC / 12,288,
  necessary minimum port service for this fixed staging protocol).

Let `HF = sum(H_e F_e)`, `A = sum(M_e(H_e+F_e))` and
`C = sum(M_e(16F_e+8H_e))`. Necessary W service is
`(packed_unique + 12HF)/1024`: three BF16 matrices filled and read at least
once. Necessary X service is `4A/384`; necessary accumulator service is
`C/192`; global activation service uses its separately derived minimum plus
output clearing. These minima do not include padded lanes or reloads.
They are conditional on the declared staging/combine protocol and cannot
exclude a future direct-decode or register-only architecture.

Headroom = geomean(window time / search bound) − 1. Only a regime with at
least one **development** batch at ≥10% proceeds to geometry search.
Below 3% is labeled below an **assumed** screening margin; no measured 3D
model error is implied. Insufficient space is a statement about that regime,
not a theorem about all heterogeneous architectures.

## Selection and gates

Architecture and runtime selection are separate stages. Rank using the
geometric mean of per-window paired latency ratios to a common reference
(equivalently geomean latency for ranking on identical windows), never total
ms. Hardware/runtime is frozen across all batches within each regime before
heldout evaluation. Different format/credit design scenarios are not a single
chip that changes geometry for each test workload.

Geometry screening evaluates all shapes with equal and proportional initial
quotas. Strong single and homogeneous baselines additionally scan both bounded
flows, three prefetch caps and 25 GU/Down cap pairs. Top eight geometries per
family plus strong-baseline seeds receive independent quota refinement:
fractions {1/8,2/8,...,7/8}, three starts, two coordinate sweeps and 32
deterministic cross-axis probes. Runtime candidates include EFT, idle,
round-robin and token thresholds {1,2,4,8,16}. Near-best 1% candidates and
development selection stability are saved. This is an exhaustive **geometry**
screen with bounded hierarchical allocation/runtime refinement, not a global
joint optimum over every compiler/dataflow configuration.

Every model point is run twice and the complete result objects must agree.
Statistics use paired, batch-stratified window bootstrap (4,000 replicates).
It estimates window-sampling variability, not analytical model error;
correlated, historically exposed windows are not independent blind trials.

The heterogeneous candidate must meet the user-approved thresholds against
**both the optimized single and optimized identical-pair baseline**:

1. Enter matching Rust/native calibration only if heldout paired geomean
   latency reduction is ≥5% against both. Freeze six concrete calibration
   points at that time; do not substitute old unrelated native timings.
2. Declare victory only after actual calibration gives ≥10% reduction
   against both, with paired bootstrap 95% lower bound ≥5%. Report every
   batch and dataset; a single favorable batch does not establish robustness.
3. The alternative ≥20% area/energy saving at no more than +1% single-core
   latency is inactive until technology-qualified synthesis/energy exists.
   Multiplier count is not an area model.

The current Rust research backend fixes PN4/PK512. An older native backend
supports common PN/PK independent GEMMs, but not this model's per-core
independent PN/PK, fused Gate/Up, bounded Z, Down and combine boundary.
Calibration is not available by renaming a CSV. Vivado is present but no
technology-qualified ASIC BF16/FP32/SRAM area ledger is present.

## Reproduction and outputs

From the simulator repository root, with NumPy, pytest, matplotlib, PyTorch
and safetensors installed:

```sh
python -m research.moe_dispatch.regime.run \
  --inputs /absolute/path/to/four/captured/inputs \
  --old-system /absolute/path/to/frozen/v6/system \
  --weights /absolute/path/to/deepseek-v2-lite-chat \
  --out /absolute/path/to/new/output --workers 32
```

The runner stops on failed checks, preserves input hashes, freezes source
and emits a final manifest. Use an empty output directory; never overwrite
frozen experiments. The primary output table is
`search/paired_gates.csv`, with per-batch `paired_gates_by_batch.csv` and
per-dataset `paired_gates_by_dataset.csv`. `REPORT_ZH.md` explains the results.
`optimized_grid/optimized_grid.csv` is the lower-bound grid; `diagnosis/`
contains historical compatibility checks, not the new optimized primary
score. `audit/` contains independent dimension/traffic/quota checks and
technology-independent structural counts, not synthesized mm²/J.

Service counters overlap and are never summed into wall-clock time. Report
the fluid limiter as a model attribution, not a measured native stall cause.
Full selected result objects, every evaluated candidate, source snapshots,
capture inputs and checksums belong in the durable artifact archive.
