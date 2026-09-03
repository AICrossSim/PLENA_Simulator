# Nemotron Agentic Matrix L-Compute DSE

This directory is derived from the externally archived real-checkpoint B200
campaign. `group_results.csv` preserves every length-sorted workload group;
`summary.csv` reports medians and P95 values by benchmark and batch size.

GPU timing/energy columns are measurements. PLENA cycle/TPOT columns are
pre-RTL Compiler/Simulator estimates with symbolic weights. They are shown
side by side but must not be presented as a measured GPU speedup. Routing uses
the eager run's self-consistent token trace; the optimized timing run remains
baseline-only because its generated continuation differs for 45/48 samples.
