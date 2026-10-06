# Variable M/N/K geometry study

## New compute-only robust search (2026-10-06)

`robust_compute.py` is separate from the older full-system `study.py` below.
It enumerates 114,034 equal-12,288-multiplier designs: 88 single, 72 identical
dual, and 113,874 different dual designs. Both dual PKs vary independently
over `{16,32,64,128,256,512,1024,2048}`. PM and PN enumerate every positive
integer allowed by the multiplier budget, without the older 16/192 caps.

```bash
python -m research.moe_dispatch.geometry3d.robust_compute --inputs INPUTS --out OUTPUT
python -m pytest research/moe_dispatch/geometry3d/test_robust_compute.py research/moe_dispatch/geometry3d/test_compute.py -q
```

The primary scope is **pure GEMM compute with ideal operand/output storage**:
no memory-capacity eligibility, HBM, SRAM ports, vector, or control timing.
Gate and Up are independent outputs in one K-major stream, with separate
tail padding; Down waits for both. The same output's next K segment waits
for its previous FP32 commit. For R independent physical output records,
S K segments, latency L (dot + commit), and initiation interval 1:

`cycles = (S-1)*max(R,L) + R-1 + L`.

All ready experts use common logical-work LPT ordering and whole-expert EFT
assignment. No cross-expert GEMM packing, N splitting, learned prediction,
or hardware retuning by test window is allowed. Ideal output storage is a
compute ceiling, **not a claim that every candidate fits installed SRAM**;
the per-task required FP32 space is reported. Separate finite-record
diagnostics use the same 8/16/32 output records *per engine*, not a shared
fixed record count that halves each dual engine's pipeline capability.

Each batch gets equal weight in the development latency-ratio geometric
mean. Three explicit, unvalidated reduction-latency hypotheses each rerun
the complete search twice. Minimax within-family regret freezes one geometry
per family before loading held-out windows. Held-out results repeat twice.
Times in ms assume 1 GHz and are analytical estimates, not native simulator
or model inference measurements. Historical held-out captures are not blind.

Compact new results: `research/moe_dispatch/validation_robust_compute/20261006`.
Full enumeration, detailed assignments, input/source hashes, and exact
source are archived in `final_artifacts/moe_robust_compute_20261006_v1`.

## Older system study (2026-10-04)

This is the three-dimensional analytical study requested after the earlier
Round A search fixed physical `PK=512`. It does not replace the frozen Rust
v3 simulator or its results. Physical dimensions are always `PM × PN × PK`.

Read [REPORT_ZH.md](REPORT_ZH.md) for the measured execution of this analytical
study and [METHODS.md](METHODS.md) for the resource and timing assumptions.
“Complete search” means the declared 16,763-geometry primary domain. The
subsequent allocation/dataflow search refines the top four candidates per
family; it is not a global joint optimum or a physically validated design.

The implementation provides:

- Exact geometry enumeration, useful/issued MAC counts and finite output
  contexts, with an independent issue/commit scoreboard for the teacher toy.
- A closed installed-storage ledger, native BF16 32-byte addresses, finite
  weight slots, activation buffers and Z chunks.
- A phase-level shared HBM/activation/vector/control service model. Small
  finite-request cases separately check credit retention and landing leases.
- Development-only selection, frozen hardware/runtime settings, two identical
  executions per point, heldout evaluation, timing sensitivities and resource
  oracles that retain charged expert ownership.
- BF16/FP32 numerical reference checks. Changing PK can change rounding;
  cross-geometry bit-exactness and trained-model quality are not claimed.

The current timing results are **prospective analytical estimates at a
hypothetical common 1 GHz**, not native Rust/Ramulator, RTL, chip measurements,
or full-model generation latency. The post-Router timing boundary includes
Gate/Up, SiLU/product, Down and combine, but excludes Router and attention.

## Run

From the simulator repository root, with the archived captured input JSONs:

```bash
python -m pytest research/moe_dispatch/geometry3d -q
python -m research.moe_dispatch.geometry3d.study --inputs INPUTS --out OUTPUT --stage search --workers 8
python -m research.moe_dispatch.geometry3d.study --inputs INPUTS --out OUTPUT --stage final
# Older compute diagnostic still retains memory eligibility and phase/control
# structure. Use robust_compute above for the new pure-GEMM-only scope.
python -m research.moe_dispatch.geometry3d.toy --out OUTPUT
python -m research.moe_dispatch.geometry3d.figures --input OUTPUT --out OUTPUT/figures
```

Use a fresh output directory when changing timing sources. Development
captures choose the configuration; do not use heldout results to change it.
Historical heldout captures were exposed in previous studies, so this is not
a pristine blind evaluation.

## Results and provenance

Compact versioned results are in
`research/moe_dispatch/validation_geometry3d/20261004/{system,compute}`.
`PREREGISTERED.json`, `FROZEN_SELECTION.json` and
`SOURCE_FROZEN_BEFORE_HELDOUT.json` identify the selection contract.
`POSTPROCESSING_RECEIPT.json` records presentation-only source updates without
rewriting the original completion receipt. The four timing sources remained
unchanged between selection and heldout evaluation.

Full captured inputs, 945 per-layer detailed objects per study, source
snapshots, independent audits and figures are archived at:

`/scratch/shared/mcl123/plena/final_artifacts/moe_geometry3d_20261004_v6`

No new compiler ISA or Rust geometry execution path is claimed by this
directory. The compiler worktree stays pinned to
`480d558c72f8fce19572b2408eb02a1ae52c40eb`.
