# Recurrent sublayer results, 2026-09-28

This is a compact export of the completed projection campaign. Numeric CSV
fields are unchanged; local result paths are made relative. `manifest.json`
records original and exported hashes. Raw HBM images, weights, build products,
source snapshots and large logs are not included.

## Projection comparison

Both sides use native recurrence with the same arithmetic. Before uses the
historical resident M_MV schedule; after adds finite replay, compact input
slices and up to four requests per M_MM.P. Before is an analytical prediction;
after has both Rust execution and matching analytical accounting. These are
not results against the best original PLENA implementation.

Time is the entire batch's one-token recurrent sublayer, input normalization
through output projection, at a modeled 1 GHz. Outer residual and FFN/MoE are
excluded. All weights in these executions and traffic tables are BF16.

| Model | Batch | Before ms | After ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Mamba | 1 | 3.063999 | 2.909800 | 1.0530 |
| Mamba | 2 | 4.494757 | 3.047989 | 1.4747 |
| Mamba | 4 | 7.689085 | 3.378375 | 2.2760 |
| Mamba | 8 | 15.765859 | 5.297095 | 2.9763 |
| Mamba | 16 | 31.023700 | 9.155280 | 3.3886 |
| KDA | 1 | 34.266319 | 32.034199 | 1.0697 |
| KDA | 2 | 50.700886 | 33.324016 | 1.5215 |
| KDA | 4 | 93.852496 | 35.976288 | 2.6087 |
| KDA | 8 | 181.979956 | 55.509874 | 3.2783 |
| KDA | 16 | 356.450631 | 101.932709 | 3.4969 |

## Files and comparison boundaries

| File | Contents |
| --- | --- |
| `BEFORE_AFTER.csv` | The projection comparison above |
| `PARTS_BEFORE_AFTER.csv` | Input/output projections, coefficient layout, convolution, gates, norm and recurrence |
| `stage_comparison.csv` | Resident → replay → compact → batch ablation |
| `fair_recurrence_comparison.csv` | Shared optimized projection; ordinary Vector vs native recurrence; arithmetic differs |
| `UNIFIED_SUMMARY.csv`, `UNIFIED_PARTS.csv` | Combined projection/recurrence comparison under two memory profiles; every point has `original_plena_eligible=False` |
| `machine_validation.json` | Ten archived numerical observations, separate seven-component timing and read/write-byte calibration |
| `hardware.json`, `numerical_scope.json` | Explicit resource costs, precision, inputs and exclusions |

The `original_memory8` label changes memory policy only. Its other services
still include extended interfaces; it must not be relabeled original PLENA.
The common16 results use a candidate 16-controller memory system and 32/32
bounded DMA credits. These choices are shared within each comparison.

Machine results exactly match the declared arithmetic for every active HBM
allocation, including unchanged weights. The comparison count is therefore
not a count of independent output predictions. Batch members repeat the
captured first-layer B1 input in private allocations. Distinct-input Matrix
and private-state tests provide separate ownership checks. No new 2048-token
quality acceptance follows from this one-token campaign.

The original machine result's `calibration` string was written before the
subsequent calibration pass; the adjacent `calibration` object records that
pass. Rust and Python share the resource contract and Ramulator, so exact
agreement does not independently prove RTL feasibility or DRAM accuracy.

## Reproduce and extend

Use [the checkout instructions](../../doc/l_tile_projection.md). The portable
`analytic_models.performance.projection_campaign` entry regenerates the four
projection stages and operator components from the current source and an
explicit prepared memory profile. Full numerical replay additionally requires
the external captured HBM fixtures; those are not bundled here.

No whole-model speedup, GPU comparison, power, area, integrated overlap or
globally optimal mapping is established by this archive. The best legal
original M_MM baseline and proposed joint SRAM residency remain open work.
