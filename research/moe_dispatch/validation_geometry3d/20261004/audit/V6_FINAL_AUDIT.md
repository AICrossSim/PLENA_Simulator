# Final v6 read-only audit

Both completed studies passed the independent artifact checks. No timing or
search implementation was edited by this audit. Heldout input files were not
opened; configuration checks used development artifacts, and final metric
checks used frozen generated results.

## Search and selection

The independently enumerated domain is PM=1..16, PN=1..192 and
PK={32,64,128,256,512,1024}, with exactly 12,288 main multipliers in one or two
canonical cores. Each study contains all 16,763 unique designs. There are
44 single, 41 homogeneous dual, and 16,678 heterogeneous dual designs;
12,016 dual designs have unequal PK. Under the primary fixed storage contract,
15,107 designs are legal and 1,656 are excluded with explicit capacity reasons.

For both studies, top-four development geometries per family fed the stated
allocation/dataflow refinement. Every selected family point matches its
refinement development argmin. Seven frozen points each have all 24 declared
development policy/prefetch trials, and every frozen runtime setting matches
the trial argmin. The extra unequal-PK point matches its fullspace development
selection. These checks establish the stated selection procedure, not a joint
exhaustive mapping/runtime optimum.

## Source freeze and result integrity

For compute.py, memory.py, model.py and study.py, the before-heldout source
receipt hashes match current source and both completion receipts. The source
receipt predates each selection freeze; completion pins the selection JSON
hash. Later report/figure postprocessing is outside these four source pins
and should carry its own receipt rather than rewriting completion.json.

Each study has 7 frozen points evaluated on the same 135 captured layer
invocations: 945 layer records, 56 batch/all summaries, 35 owner-frozen resource
conditions and 28 timing-profile rows. Summary totals, nearest-rank p95,
aggregate spatial/wall utilization, HBM/X-fetch bytes and reload ratios were
recomputed independently from detailed records. Every charged resource oracle
matches its original charged result. Useful MACs match between all points;
phase useful/issued/HBM totals match layer totals. Every detailed installed
ledger is exactly 2,158,592 bytes, bank totals are W64/X24/acc12, and declared
accumulator/X/W peaks fit their private quotas.

## Audited primary heldout totals

All values below are analytical milliseconds at the declared hypothetical
1 GHz clock. They are the sum of the 135 captured invocations, not full-model
generation latency or independent trial averages.

| Study | Development-selected single | Homogeneous dual | Heterogeneous dual | Unequal-PK point | Original fixed single |
|---|---:|---:|---:|---:|---:|
| Compute/resource-disable diagnostic | 48.358120 | 52.199002 | 49.824023 | 49.647390 | 83.963800 |
| Charged finite-resource model | 617.272275 | 667.204351 | 638.794639 | 638.794639 | 812.413907 |

Charged selected single is 6x16x128. Charged selected heterogeneous/unequal-PK
is 2x23x32 + 13x13x64. Compute diagnostic selected single is 3x128x32;
heterogeneous is 4x2x32 + 8x47x32, with the separately selected unequal-PK
point 1x4x64 + 8x47x32. Compute diagnostic removes HBM, SRAM-port and control
constraints but retains finite capacities, bounded groups and shared vector
service.

Both studies favor the selected single on aggregate heldout totals. In the
charged study, the selected heterogeneous point beats that single by only
about 0.027%–0.029% for T=2/4/8 and falls behind for T=16/64/96/128. The
selected single remains best among these frozen points under all four declared
timing profiles. This does not establish a universal best architecture or a
guaranteed heterogeneity benefit.

## Validation and scope

The frozen source passed 51 current geometry3d tests. Additional independent
v6 regressions passed 9 no-spare native-tail occupancy cases, 4 spare native
wire-window cases, 48 single-owner estimate/actual equalities and 5 identical
dual threshold/EFT equalities. Earlier independent checks covered all geometry
ledgers, 58,800 bounded phases, 840 native-sector slices, 500 max-min resource
cases and 50 finite SharedHBM event cases; their receipts identify their source
versions and should not be presented as cycle-exact fullspace validation.

The result remains a prospective phase-fluid comparison. Consumer and SRAM
services are averaged across phases; native per-bank conflicts, exact packet
scheduling, RTL clock closure, PPA and pretrained-model accuracy are outside
the model. The 20-cycle PK512 anchor is inherited, with explicit slope/flat
sensitivities. PK changes FP32 grouping; equal precision is not bitwise
equivalence. The heldout capture was historically exposed in earlier work,
so the present development-only selection is respected without claiming a
pristine blind dataset.

## Evidence

- audit_artifacts.py: independent read-only artifact validator.
- compute_v6_artifacts_receipt.json: complete compute study validation.
- supply_v6_development_receipt.json: supply development-only validation.
- supply_v6_artifacts_receipt.json: complete supply study validation.
- final_model_regressions.py and final_model_regressions_receipt.json:
  independent current v6 timing regressions.

No unresolved contract violations were found in the frozen v6 source or
completed artifacts within the declared analytical scope.
