# Historical Analytic Baselines

This namespace contains reproducibility-only models and benchmarks that are
not imported by the formal PLENA DSE.

- `memory/` is the pre-V4 traffic and capacity model restored from historical
  simulator revisions.
- `roofline/` and `utilisation/` implement the superseded closed-form latency
  assumptions.
- `benchmarks/` contains explicit A/B entry points for those models and for
  retired compiler transitions.

The canonical DSE uses CostEmitter lineage, ideal-II1 compute timing, HBM DMA
V4, and tile-aware TP-CP-EP scaling. Historical baselines must be launched
directly from this namespace and must not be described as current predictions.
