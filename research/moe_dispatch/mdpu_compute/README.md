# M-DPU pure-compute experiment

This directory contains a finite-controller experiment using the archived
Shared + Top-6 Routed DeepSeek-V2-Lite workload, 65,536 physical multipliers,
BF16/FP32, OS, ideal operand supply and an assumed 1 GHz execution clock.
It is not an HBM experiment or a calibrated physical PLENA matrix datapath.

The requested compact-family domain is compared separately from controls with
the same available scalar/narrow-row capabilities. BLEN>=16 is an experiment
restriction, not a rule in PLENA Table III. Replicated geometries and BF16
arithmetic are declared extensions. Equal multiplier counts do not establish
equal silicon area.

`domain.py` defines the complete finite grid. `run.py` enumerates 38,593
geometry/policy/threshold points on 18 development windows, persists the
selection, then evaluates 135 archived heldout windows. Each result is run
twice. Every batch has equal weight in the seven-batch geometric mean.

`runtime.py` models bounded dual queues, ingress backpressure, shape service
estimation, unstarted-head bypass and a current/shadow context per core.
All organizations receive the same context preparation capability. The
default context setup cost is a designed four cycles, with zero/sixteen-cycle
sensitivity checks; it is not a measured PLENA task-switch cost.

The first experiment exposed a forced-spill failure: full preferred queues
caused tasks to be sent to much slower cores, and early context binding made
that choice irreversible. `--guarded` selects the best predicted completion
before considering queue capacity and can wait for that queue. Both complete
experiments are retained. The second mechanism was revised after inspecting
the first heldout failure: its reused heldout set is regression evidence,
not a new independent generalization test.

The RTL under `rtl/` implements the control path. Verilator tests compare
every expert binding/start/end and the final makespan with the event model.
The matrix service timings remain analytical. Neither mm²/W nor 1 GHz
technology timing closure has been measured. The static threshold is an
affinity tie preference; resource-aware decisions are generally not expressible
by a single token-count threshold.

Outputs are stored outside scratch because that filesystem is nearly full:

- `/home/mcl123/plena-outputs/mdpu_compute_20261013/CURRENT_RESULTS.md`
- `/home/mcl123/plena-outputs/mdpu_compute_20261013/guarded/SUMMARY.md`
- `/home/mcl123/plena-outputs/mdpu_compute_20261013/guarded/M_DPU_完整数据表.xlsx`
- `/home/mcl123/plena-outputs/mdpu_compute_20261013/rtl/validation.json`

Run `REPRODUCE.sh` with `MDPU_OUTPUT` pointing at a new output directory.
The input loader verifies historical trace hashes. Python 3.11, NumPy,
openpyxl, Verilator and a C++ compiler are required.

Do not interpret the restricted-family ladder as proof of a general
heterogeneous advantage. Inspect the equal-capability controls and each
core's useful-work share before drawing an architectural conclusion.
