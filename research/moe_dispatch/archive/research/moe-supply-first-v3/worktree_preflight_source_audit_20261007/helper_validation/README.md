# E0 BF16 historical compatibility

All 945 v6 plus 405 previous BF16/256 design records must be exact.
Each full result is simulated twice and compared as a complete object.

Commit: `e061ea68e98f9c3341e1fea11e0eb341bef293d6`

```sh
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python /scratch/shared/mcl123/plena/outputs/round2_preflight_20261007/reproduce_e0.py --repo /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator --workspace /scratch/shared/mcl123/plena --out /absolute/new/empty/output
```

Input and source hashes: reproduction_receipt.json and frozen_inputs.json.
