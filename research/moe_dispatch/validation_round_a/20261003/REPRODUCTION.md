# Round A reproduction and scope

Run from the simulator worktree with the pinned Python environment:

```bash
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python -B -m pytest -q research/moe_dispatch/test_round_a_dse.py
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python -B research/moe_dispatch/round_a_dse.py --inputs outputs/moe_supply_first_v3/inputs --output /tmp/plena-round-a-new-reproduction
```

Use a new output directory. All geometries have exactly12,288 main MACs. Architecture is selected on18 development captures under common EFT, hardware is frozen, then runtime thresholds are selected from development only;135 captured heldout windows evaluate the fixed points. These captures contain top-k6 routes and a Shared expert. The nominal1GHz assumption converts one million cycles to one millisecond. The sum of135 layer-window times is not one full-model inference.

This is a prospective ideal-operand compute/vector model. It excludes HBM, SRAM banks/ports, runtime decision service, Router and attention. New N widths are not implemented in the frozen Rust/compiler. Minimum footprint is necessary-only; passing it does not certify iso-SRAM/iso-area physical feasibility. Round B and Round C have not been performed for these new shapes. Historically exposed v3 evaluation captures are not pristine blind data.

The preliminary v1 threshold accidentally bypassed unequal-N cores when their M widths matched. v2 corrects that shared rule for every organization and reruns the entire search. Preliminary summaries/freeze are retained underpreliminary_v1_history; the complete original v1 directory remains untouched at/tmp/plena-round-a-dse-20261003. Current reported conclusions use onlyv2. Source pins and the new freeze precede current heldout access. A complete independent reproduction has14 byte-identical core artifacts; the separate reviewer recomputed selections, totals and metrics and passed9 tests.

The supplied TypeError repair belongs only to the previous v3 report: missing optional CSV values remained unavailable in memory; no simulator, CSV or acceptance gate changed. Its full110,024-run evidence was already published separately.
