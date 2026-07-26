# Qwen3 Prefill Four-Objective Multi-Chip DSE v1

> **Historical integration milestone, audited 2026-07-26.** The four
> objectives remain current, but this report's original multi-chip partition
> predates final-lineage tile-aware TP x CP x EP reconstruction. Preserve its
> objective and selector design; use
> [`tile_aware_multichip_lineage_audit_v4.md`](tile_aware_multichip_lineage_audit_v4.md)
> for current distributed latency/energy evidence.

> **SRAM-search update (2026-07-25):** the formal search now uses a
> partial-resident K/V policy instead of raw Matrix SRAM tiles and assumes
> ideal dual-port SRAM area. Access energy remains action-based and is not
> divided by two. See
> `../compiler/partial_resident_kv_ideal_dual_port_dse.md`.
>
> **Compute-timing update (2026-07-24):** the four-objective infrastructure now
> defaults to `ideal-ii1` compute timing. Matrix timing remains structural;
> Vector/Scalar/control instructions cost one cycle and hazards are disabled.
> Existing rtl-v1 study journals are intentionally schema-incompatible.

## Purpose

The formal Qwen3 prefill study now optimizes:

```text
minimize latency_ms
minimize aggregate_total_silicon_area_mm2
minimize system_energy_nominal_mj
maximize accuracy_score
```

Area closeness to the 826 mm2 A100 reference is deliberately not an
objective. The nominal area constraint is 908.6 mm2. A smaller design remains
Pareto-optimal when it improves latency, energy, or accuracy.

## Integrated Models

Latency is the existing stage model:

```text
sum(max(per-chip RTL-v3 compute, per-chip R-aware HBM V4)
    + internal inter-chip communication)
```

Area is aggregate physical silicon:

```text
chip_count * (area_new core area + 10% endpoint area)
```

Energy is recomputed after the latency work partition. It is not obtained by
dividing or multiplying the final single-chip power:

- `TP+SP`: compute and local-memory actions are divided across chips.
- `TP-only`: Matrix work is divided, while Vector, Scalar, control, and their
  local SRAM work are conservatively replicated.
- Logic dynamic, SRAM dynamic, logic leakage, and ideal hierarchical
  ClockWork are evaluated per chip and then aggregated.
- External HBM dynamic energy uses aggregate physical 64-B traffic.
- HBM background uses one fixed aggregate 80 GB capacity over the multi-chip
  makespan, rather than 80 GB per PLENA chip.
- Internal communication uses 8 pJ/bit as the nominal literature proxy.

The link report also exposes 1.3 pJ/bit and 70.9 pJ/bit as named,
non-statistical sensitivity cases. The values have different generations and
measurement boundaries:

- [NVLink 6 bandwidth](https://www.nvidia.com/en-gb/data-center/nvlink/):
  3.6 TB/s bidirectional.
- [NVLink-C2C](https://developer.nvidia.com/blog/inside-nvidia-grace-cpu-nvidia-amps-up-superchip-engineering-for-hpc-and-ai/):
  1.3 pJ/bit.
- [LBNL interconnect study](https://escholarship.org/uc/item/67z0d2wn):
  8 pJ/bit early-NVLink baseline and 70.9 pJ/bit measured A100 active path.

Static link maintenance, switches, package, cooling, board regulators, SRAM
leakage, and the final FP16 KV handoff are excluded.

## Search Method

The default study uses multi-objective TPE with 2,048 trials and at most 24
worker processes. Integer hardware dimensions are sampled as ordered log2
variables. Matrix SRAM uses log-distance, while the 103 accuracy profiles use
a distance based on precision family, operand widths, internal FP width, and
accuracy.

The startup queue contains one anchor per precision profile and stratifies
`MLEN/BLEN` over 1, 2, 4, 8, and 16. SRAM anchors never exceed compiler useful
saturation. NSGA-II and exhaustive grid modes remain available.

Study metadata includes objective/search schema names and SHA-256 hashes for
the accuracy and interconnect artifacts. A three-objective study cannot be
resumed as a four-objective study.

## Validation

The focused regression suite covers:

- `N=1` multi-chip energy reproduces the existing system estimator.
- Fixed aggregate traffic has unchanged external-HBM read/write energy as
  chip count changes.
- HBM background is charged once for aggregate capacity.
- TP-only replicated on-chip work is not below TP+SP.
- Interconnect energy is monotonic for 1.3, 8, and 70.9 pJ/bit.
- Component energies sum to the aggregate system result.
- Area/energy selectors use deterministic tie-breaking.

The focused performance, power, and DSE suite passed 70 tests. The final real
`12 trials / 4 workers` smoke completed all 12 trials with no failures or
prunes. It produced formal latency, area, energy, and accuracy values, exact
role-aware HBM traffic partitioning, and all three interconnect sensitivity
results.

The smoke is an infrastructure check, not a converged DSE. Its fastest
area-feasible point used 8 chips, 829.04 mm2, 1,826.09 ms, and 267,112.68 mJ.
Its lowest-energy area-feasible point used 2 chips, 245.80 mm2, 7,531.63 ms,
and 151,351.92 mJ. This demonstrates that latency and energy select different
designs as intended.

An earlier diagnostic smoke intentionally covered small-MLEN hardware anchors
and exposed existing RTL-v3 compressed-scoreboard fixed-point failures. These
remain `FAIL`; capacity-dominated SRAM points remain `PRUNED`. Neither is
silently assigned an approximate energy objective. The final startup policy
therefore uses a stable 512/512 hardware baseline for all precision anchors
and keeps difficult small shapes in the separate hardware exploration set.

## Outputs

Each run writes the four-objective Pareto table and the following selectors:

```text
fastest_under_area_budget
lowest_energy_under_area_budget
highest_accuracy_under_area_budget
closest_area_below_826
fastest_within_5pct_of_826
lowest_energy_within_5pct_of_826
best_energy_delay_product_under_budget
smallest_design_beating_a100_area_candidate
```

It also generates:

```text
latency_vs_aggregate_area.png
energy_vs_aggregate_area.png
latency_vs_energy.png
```

## Claim Boundary

Energy combines RTL-activity-calibrated on-chip actions, ideal clock-gating,
literature HBM3E parameters, and literature interconnect proxies. It is an
architecture-level DSE metric, not signoff power. TP+SP is an optimistic
stage-level partition model; TP-only is the conservative sensitivity. The
area model is pre-layout logic plus SRAM macros and excludes package and HBM
stacks.
