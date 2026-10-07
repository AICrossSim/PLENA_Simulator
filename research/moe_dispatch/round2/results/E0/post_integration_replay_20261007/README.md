# E0 BF16 historical compatibility

All 945 v6 plus 405 previous BF16/256 design records must be exact.
Each full result is simulated twice and compared as a complete object.

Commit: `6130f924c3417ffda19cd0bf4c98d6f4c10b3cae`

```sh
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/reproduce_e0.py --repo /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator --workspace /scratch/shared/mcl123/plena --out /absolute/new/empty/output
```

Input and source hashes: reproduction_receipt.json and frozen_inputs.json.
