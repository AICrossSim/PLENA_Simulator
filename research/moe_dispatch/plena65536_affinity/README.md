# PLENA 65,536 MACs: Shared + MoE affinity DSE

This directory implements a new pure-compute experiment. Previous frozen
`plena65536_moe` and `plena65536_dedicated` outputs are not replaced.

## Contract

- Exactly 65,536 physical multipliers, BF16 operands, FP32 accumulators,
  1 GHz. Every expert executes Gate, Up and Down. The captured router output
  is an input, not a timed operation.
- Unified OS and zero-latency, unlimited-bandwidth operand supply. Timed
  arithmetic uses the previously declared PLENA-topology analytical model,
  including the M_SUM pipeline/drain model. No SRAM bank, NoC, DMA, controller,
  nonlinear activation or routing latency is included.
- This is a BF16/topology extension of the paper geometry, not a measured
  implementation of the paper's MX/INT datapath. In particular, BLEN=1 is an
  explicit scalar-subarray extension. Native Rust/RTL calibration and PPA
  are not complete. Equal multiplier counts do not imply equal area.
- Geometry is `PM × PN × G`, with `G = MLEN / BLEN`. `G` is not logical K.
  PM is a power of two from 1 through 64, PN is aligned to 4, and MLEN is a
  power of two no greater than 2048. All legal integer budget splits in
  this finite domain are enumerated; none uses rounded 2:1 MAC counts.
- Core 0/1 are stable role labels. No rule requires Core 0 to be taller or
  have more MACs. The grid contains 118,046 individual core geometries,
  266 full-budget single cores, 250 equal half-budget geometries, and
  4,410,364 ordered nonidentical two-core pairs.

## Scheduling policies

**A / `hard_threshold`:** Shared and Routed with `Me > T` go to Core 0;
other Routed experts go to Core 1. T is chosen from `{1,2,4,8}` on development
data and then frozen. Experts cannot move after assignment.

**Fair greedy / `greedy_eft`:** Process descriptors in descending useful
MAC count, then expert ID; assign each to the core with the smallest predicted
queue finish plus task cost. Exact completion ties use the threshold affinity.

**B / `workload_aware`:** Evaluate greedy EFT and four hard-threshold plans
using the current known workload and declared analytical shape costs. Choose
the lowest makespan, preferring greedy on an exact tie, then T=1/2/4/8.
The threshold parameter affects greedy's exact ties and is development-tuned.
No future workload, observed completion time or heldout label is used.

**Threshold-only diagnostic / `adaptive_threshold`:** Evaluate only the four
hard-threshold plans on the current descriptors.

All are software reference policies with zero planning latency. B is a
five-plan analytical selection algorithm, not an implemented single-cycle
hardware dispatcher, a trained predictor, or an implemented prefetcher.
No experts are split, migrated or stolen while executing.

## Development and evaluation

The existing manifest supplies 18 development and 135 heldout windows with
DeepSeek-V2-Lite Shared (`F=2816`) and Top-6 Routed (`F=1408`), both `H=2048`.
Batches are 2, 4, 8, 16, 64, 96 and 128. Search loads development only.
Each objective is GM across windows within each batch, then an equal GM
across the seven batches. Geometry and static parameters are frozen before
loading heldout payloads. A per-window B decision may change; hardware may not.

The 135 heldout windows have been inspected in previous research iterations.
They are exploratory validation, not a pristine final paper test set. New
unseen captures are required for final generalization claims.

A evaluates every geometry/threshold point. B enumerates every pair and
uses an admissible fractional two-machine assignment bound to reject pairs
that cannot beat the incumbent. Survivors are evaluated with the actual B
algorithm and all four tie parameters. The certificate establishes the best
configuration in this finite geometry/algorithm domain, not an unrestricted
optimal scheduler or optimal physical hardware architecture.

The primary table includes paper-default geometry, DSE single, DSE homogeneous
greedy, same-policy DSE homogeneous B, heterogeneous A and heterogeneous B.
Both selected heterogeneous geometries are additionally replayed under A,
greedy, adaptive-threshold and B, using parameters selected on development.

## Current frozen result

The main-profile candidate has Core 0 `1×704×64` (45,056 MACs) and Core 1
`1×160×128` (20,480 MACs), an 11:5 split. **Both PM values are 1**; the
search does not establish that a tall Dense core is optimal.

| Frozen design | Heldout seven-batch GM compute time (ms) |
|---|---:|
| Paper-default `32×32×64` | 0.178193059 |
| DSE single `1×128×512` | 0.022383588 |
| DSE homogeneous greedy `1×128×256` ×2 | 0.021393375 |
| Same-policy homogeneous B | 0.021391022 |
| Heterogeneous A, fixed T=1 | 0.030154606 |
| Same heterogeneous geometry, B | 0.021107933 |

B's speedup is 1.060435× over the DSE single and 1.013411× over same-policy
homogeneous B. This is a small uncalibrated analytical advantage over the
strong homogeneous reference, not a demonstrated physical architecture win.
The same-geometry A/B contrast measures the scheduling change directly.
It does not prove that every possible fixed threshold or hardware split must
fail on every workload.

## Supply reverse ledger

Three boundaries are kept separate: cold unique external weights, shared
resident-pool refills into local operand buffers, and local-buffer broadcast
ingress into the arrays. Array fanout is not counted as external traffic.

`supply.py` conserves OS W/X byte counts and reconstructs each feed/drain
interval to compute the actual full-width array-ingress peak. The shared
SRAM minimum peak is deliberately unspecified: it depends on refill timing,
prefetch distance, initial occupancy, banks and transport latency.

The entire model layer's BF16 weights total 1,089 MiB (33 MiB Shared plus
64 × 16.5 MiB Routed). Residency capacity does not prove sufficient supply
bandwidth. The selected B candidate needs a seven-batch GM **123.1115 TB/s**
of masked W+X pool refill service in this declared OS loop; its full-width
array-ingress peak is **131.456 TB/s**. These are different interfaces and
statistics, not a measured SRAM specification or an end-to-end result.

## Reproduction and artifacts

Run from the repository root using Python 3.11:

```bash
python3.11 -m pip install -r research/moe_dispatch/plena65536_affinity/requirements.txt
python3.11 -m pytest research/moe_dispatch/plena65536_affinity -q
python3.11 -m research.moe_dispatch.plena65536_affinity.run --out /path/to/new/results
```

The complete runner performs two searches, verifies identical selections and
arrays, generates the report, and packages sources/results/captured inputs.
The portable archive contains `REPRODUCE.sh` and a rewritten relative input
manifest. A result directory includes:

- `FROZEN_SELECTION.json`, `SEARCH_CERTIFICATE.json`, exact search arrays,
  repeated-search receipts and input hashes;
- `table1_compute.csv`, `table2_imbalance.csv`, `table3_supply.csv` and all
  per-window results;
- same-hardware policy ablations, dynamic-boundary counts, paired bootstrap
  intervals and frozen-geometry timing sensitivity;
- full per-expert plans in `execution_details.jsonl.gz`;
- `REPORT.md`, a formatted Excel workbook and its cell-readback audit;
- `SUPPLY_CONTRACT.md`, independent numerical audits and a portable ZIP.

Every report point is deterministically replayed twice. Tests cover real-route
work conservation, affinity, queue timing, policy choice, exact budget,
zero-load imbalance handling, equal-batch weighting, and the distinction
between array peaks and shared-pool requirements.
