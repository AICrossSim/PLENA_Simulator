# Nemotron Agentic Matrix L-Compute DSE

This directory is derived from the externally archived real-checkpoint B200
campaign. `group_results.csv` preserves every length-sorted workload group;
`summary.csv` reports N, medians and descriptive P95 values by benchmark and
batch size. P95 rows with fewer than 20 groups are explicitly exploratory.

GPU timing/energy columns are measurements. PLENA cycle/TPOT columns are
pre-RTL Compiler/Simulator estimates with symbolic weights. They are shown
side by side but must not be presented as a measured GPU speedup. Routing uses
the eager run's self-consistent token trace; the optimized timing run remains
baseline-only. Full continuations match for 3/48
samples; the 32-step replay window matches for
20/48. Exactly 35328 fully validated decode
events enter DSE and 104489 later fully validated events are excluded.

Every group includes strict-serial and ideal-resource-overlap endpoints plus
checkpoint-mixed-NVFP4, uniform-MX8/MXFP8 and uniform-BF16 weight-traffic
sensitivity. D' is materialized only as the fair packet-level bank control; no
whole-model D' timing is claimed. The source archive contains prompt token IDs
and is intentionally not part of this directory.
