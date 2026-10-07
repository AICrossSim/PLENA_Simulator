# Unit-check attempt

Each suite retains its actual exit status, JUnit cases where available, and unedited log. Missing dependencies or historical artifacts are not reported as passes. No checkpoints or GPU benchmarks are launched.

Commit at start: `6130f924c3417ffda19cd0bf4c98d6f4c10b3cae`

```sh
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/round2/run_unit_checks.py --suites research --compiler /scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-compiler --binary /scratch/shared/mcl123/plena/outputs/moe_dispatch_hardware_bound_20260928/layout_async_simulator --out research/moe_dispatch/round2/results/E0/tests
```

Use a fresh output directory or inspect the timestamped new attempt. Exact compiler override, supplied binary hash and package versions are recorded in UNIT_CHECKS.json.
