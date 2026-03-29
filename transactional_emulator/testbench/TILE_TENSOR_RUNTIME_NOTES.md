# TileTensor Runtime Notes

This note reflects the current runtime architecture in `tile_tensor_program.py`.

## Main layers

- `TensorManager`
  Owns logical tensors, tiles, slices, and `mapt` grouping.

- `ValueManager`
  Owns `tile -> ValueTile` bindings, `ValueTileView` resolution, residency, and
  write preparation.

- `ComputeManager`
  Owns last-mile ensure-at-use, operand validation, and ISA emission.

- `ThreadManager`
  Owns the symbolic `parallel_region3d` flow: region capture, expression
  validation, graph finalization, cache planning, and execution-plan lowering.

## Core concepts

- `ValueTile`
  Persistent backing object. Residency and addresses live here.

- `ValueTileView`
  Ephemeral logical window over one `ValueTile`. Views are computed from the
  current tile binding and metadata; they are not stored as long-lived state.

- `PreparedWrite`
  Explicit write-preparation result returned by
  `prepare_updated_view_value(...)`.

- `ParallelRegionGraph`
  Captured representation of one symbolic 3D parallel region. It stores axes,
  assignments, cache metadata, and the derived execution plan.

- `ParallelExecutionPlan`
  Cycle-structured lowering plan produced from one `ParallelRegionGraph`.

- `ParallelAccess` / `ParallelExpr`
  Symbolic load and expression nodes used while building a parallel graph.

## Core functions

- `resolve_value_tile(tile)`
  Input: `TensorTile | InputTile`
  Output: `ValueTile`
  Meaning: return the current backing value for one logical tile.

- `resolve_value_tile_view(tile)`
  Input: `TensorTile | InputTile`
  Output: `ValueTileView`
  Meaning: return the logical window that this tile currently sees on its
  backing value.

- `prepare_updated_view_value(tile, view, ...)`
  Input: one destination tile plus the view being updated
  Output: `PreparedWrite`
  Meaning: main tensor write-preparation API. Decides:
  - whether the write may reuse the old backing
  - whether it must switch to a fresh backing
  - whether a partial-update preserve copy is still required

- `prepare_vram_backing_value(value, ...)`
  Lower-level helper that prepares a VRAM-backed `ValueTile`. It does not by
  itself define write semantics.

- `parallel_region3d((S, H, D), name=...)`
  Enter a symbolic 3-axis parallel capture scope.

- `where(...)`, `if_then_else(...)`
  Build symbolic selection expressions for masked parallel compute.

- `pair(axis)`, `half_index(axis)`
  Parallel indexing helpers used by RoPE-like lane pairing and coefficient
  addressing.

- `parallel_execution_plans()`
  Inspect finalized parallel execution plans.

- `lower_parallel_execution_plans()`
  Force emission of deferred parallel plans if they were not lowered on region
  exit.

## Preferred tensor write path

For tensor destinations, the preferred internal flow is:

1. resolve the target view
2. call `prepare_updated_view_value(...)`
3. run compute
4. bind/write back the result

## Parallel Execution Flow

The `parallel` feature is now a major part of the runtime, not a side helper.
The intended flow is:

1. enter `parallel_region3d((S, H, D))`
2. use symbolic axes to describe loads and assignments
3. finalize the region into `ParallelAssignment` records
4. derive cache and cycle plans
5. lower each cycle into load / compute / writeback steps
6. bind the region outputs back to tensor tiles

This gives kernel authors a higher-level way to describe structured SIMD-style
tile work while still preserving explicit lowering behavior.

## Current Parallel Contract

The current implementation is intentionally narrower than a fully general tensor
compiler, and the docs should state that clearly:

- parallel destinations must resolve to tensor-backed writes
- destination selectors must use exactly the active 3D axes
- expression lowering currently supports binary `add`, `sub`, and `mul`
- predicate lowering currently supports binary comparisons
- present lowering expects one full-width contiguous row per cycle

Within those limits, `parallel_region3d` is already powerful enough to express
important kernels such as lane-wise elementwise flows, RoPE remapping, and
parallel attention-style data movement.

## FP domain

The FP domain is intentionally separate from the tensor value/view path.

- `FPVar`
  Scalar FP storage

- `FPFragment`
  Small structured FP storage

- `pure_fp_compute(...)`, `fp_kernel(...)`
  FP-domain execution helpers

The FP domain often interacts with tensor ops through `row_op(...)`, reductions,
and scalar broadcast cases, but it is not modeled as `ValueTileView`.
