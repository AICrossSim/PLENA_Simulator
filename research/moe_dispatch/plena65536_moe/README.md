# PLENA 65,536-multiplier Shared + routed MoE compute DSE

This directory is the corrected **Shared + all active routed experts** study.
It does not replace the preserved Shared-only study in `plena65536_shared`.

## Input and timing boundary

The source is the byte-verified frozen DeepSeek-V2-Lite routing capture:
18 development windows and 135 heldout windows. The heldout set has 27 windows
each at B2/B4/B8/B16 and 9 each at B64/B96/B128. Each token selects six distinct
experts out of 64; every window also includes one Shared expert processing the
whole batch. H=2048, routed F=1408, Shared F=2816. Every active expert executes
Gate, Up and Down. No synthetic routing distribution or inactive weights are
substituted. These are archived routing replays, not newly executed numerical
model inference; prior studies have already accessed the historical heldout.

The experiment times only Matrix Unit computation under ideal operand supply.
It excludes HBM/SRAM port stalls, router, gather/scatter, SiLU, gating/combine,
dispatch and vector execution. The assumed precision is BF16/FP32, a precision
variant of the PLENA geometry: the paper's main MX4 system is not relabelled
BF16. One reference clock cycle is 1 ns; new shapes are not synthesized or
frequency/area calibrated.

## Physical geometry and finite search

The original geometry is BLEN=32, MLEN=2048, 64 native 32x32 subarrays and
65,536 Matrix multipliers. VLEN=2048 is a separate Vector Unit, not another
65,536-multiplier allowance. Native BLEN/MLEN timing comes from the preserved
geometry/timing modules in `plena65536_shared`.

A core is `(b,R,RM,RN)`, with physical output widths
`PM=b*RM`, `PN=b*RN`, physical reduction groups `G=R/b`, and
`multipliers=PM*PN*G`. **G is not logical GEMM K, and it is not MLEN R.**
One logical reduction frame streams for b cycles. BLEN and PM are powers of
two from 2 through 64; R is a power of two from b through 2048; PN is a multiple
of four. Output replication extends the original native core; it is an
architectural proposal, not a claim that the paper already supports every
replicated geometry. There is no workload-derived PN cap.

The full finite domain has 53,040 individual cores; 188 exact-budget singles,
176 identical half-budget pairs and 784,402 distinct heterogeneous pairs.
Every design totals exactly 65,536 multipliers. Equal multipliers do not prove
equal area, storage or frequency. Separate strict-native subsets are reported.

## Queueing contract and exact search proof

Primary ownership is a whole expert: its three GEMMs remain on one core.
All experts, including Shared, may be assigned to either core. A grouped
integer dynamic program finds the exact two-core layer makespan; queue order
is deterministic LPT and does not change ideal serial-core load.

Geometry selection uses only development windows. The objective is a geometric
mean within each batch, then an equal-weight geometric mean over the seven
batches. It is not a sum dominated by large windows, and it does not change
hardware between heldout workloads.

Every dual geometry obtains an admissible divisible-job LP dual lower bound.
The dual numerator uses checked exact integer arithmetic and its quotient is
rounded down. Designs are pruned only when this lower bound is worse than an
actually scheduled incumbent. Competitive designs obtain exact integer
assignment for every development window. The certificate proves optimality
within this finite geometry and whole-expert contract, **not** arbitrary
output-column/token splitting, preemption or cross-expert phase overlap.

Frozen hardware also receives an exact Shared-row-cut diagnostic (all cuts
and exact routed ownership) and a feasible sequential cooperative-column
diagnostic. They do not silently change the primary globally searched contract.

Three timing profiles cover a one-cycle/tree-stage pipeline, two-cycle stages,
and conservative serialized output retirement. The default is an analytical
pipeline model with explicit shadow partials and a charged reduction tree;
these sensitivities do not substitute for native/RTL calibration.

## Dataflow, bandwidth and SRAM interpretation

All three single-core operand retention loops OS/WS/IS and all nine dual-core
combinations are evaluated on the frozen hardware. Physical PE partials remain
OS. WS/IS are proposed operand-loop/context retention, not original hardware
WS/IS support. With ideal unlimited ports their compute times are equal; the
traffic and storage costs differ. Complete G-expanded context save/restore is
charged rather than saving a fictitious single final sum.

The ledger distinguishes cold external unique weights, resident-pool refills,
array delivery, and a separate no-resident-pool streaming system. Required
bandwidth is bytes divided by compute makespan, a necessary average rate,
not a guarantee of sufficient banks/peak bandwidth. `unique W / 256 GB/s` is
only a cold-load service reference and is not measured end-to-end time.

Capacity outputs are conservative envelopes for active weights, intermediates,
scheduled scratch, PE/shadow/tree state and operand buffers. A large shared
resident pool is explicitly priced; it is not free SRAM. OS/WS/IS context and
local bandwidth must be considered before a real hardware design is claimed.

## Reproduction

From the repository root, use a Python 3.11+ environment with NumPy,
Matplotlib, openpyxl and pytest:

```sh
python -m research.moe_dispatch.plena65536_moe.run --out /path/to/new/output
```

The runner loads the routing manifest, repeats the complete geometry search,
evaluates frozen windows and all retention loops twice, runs split diagnostics,
creates the workbook/figures and verifies deterministic results. The supplied
archive contains the relevant source, raw routing input files and a portable
manifest; run from the extracted archive directory.

Core deliverables: `FROZEN_SELECTION.json`, `SEARCH_CERTIFICATE.json`,
`INDEPENDENT_SEARCH_AUDIT.json`, `compute_summary.csv`, `cross_dataflow.csv`,
`cross_dataflow_details.jsonl.gz`, timing/dispatch/split tables, the report,
spreadsheet and standalone scientific figures.

Paper reference: https://arxiv.org/html/2509.09505v3
