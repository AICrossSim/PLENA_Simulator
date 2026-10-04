# Variable M/N/K geometry study

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
# Independent geometry-only diagnostic: add --compute-only to both commands.
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
