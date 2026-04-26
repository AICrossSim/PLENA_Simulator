# TileTensor Compiler Principles

This note gives a short conceptual introduction to the current
`tile_tensor_program.py` compiler/runtime structure.

It is intentionally high-level. The goal is to answer:

- what the current compiler/runtime is
- what layers it is made of
- what the main units are
- what design principles it follows

Related docs:

- `TILE_TENSOR_PROGRAM_USAGE.md`
- `TILE_TENSOR_RUNTIME_NOTES.md`
- `TILE_TENSOR_KERNEL_PROGRAMS.md`

## 1. Overall Positioning

The current `TileTensorProgram` system is not a fully general tensor compiler.
It is better understood as:

- a program-building API for TileTensor testbench kernels
- a runtime that maps logical tensor objects onto backing values
- a lowering pipeline that emits emulator-oriented ISA

At a very high level, the flow is:

1. the user describes logical tensors and logical compute
2. the runtime decides which backing values those logical objects currently see
3. the compute layer lowers the result into emulator instructions

## 2. Main Layers

The current structure can be understood as three core layers, two supporting
layers, and one user-facing facade.

### 2.1 `TensorManager`

`TensorManager` is the logical layer.

It owns:

- logical `Input`, `Tensor`, and `Vector` objects
- tile creation
- slice resolution
- `mapt` grouping

Its job is to answer questions like:

- what logical tensor exists
- how it is tiled
- which logical tile or slice an operation refers to

It does not decide backing residency or physical storage allocation.

### 2.2 `ValueManager`

`ValueManager` is the backing-value and residency layer.

It owns:

- `tile -> ValueTile` bindings
- `ValueTileView`
- residency transitions across `VRAM`, `MRAM`, `HBM`, and `FPRAM`
- write preparation and rebinding

Its job is to answer questions like:

- which real backing value a logical tile currently points to
- which window of that value the tile currently sees
- whether a write may reuse the old backing or must create a new one

### 2.3 `ComputeManager`

`ComputeManager` is the last-mile lowering layer.

It owns:

- operand validation
- ensure-at-use placement checks
- ISA emission for compute operations

Its job is to turn prepared operands into actual emulator-side compute
instructions such as:

- matmul-like operations
- tile binary operations
- FP kernel operations
- row operations

### 2.4 `ThreadManager`

`ThreadManager` is the symbolic parallel layer.

It owns:

- `parallel_region3d(...)`
- `parallel_region2d(...)`
- parallel expression capture
- `ParallelRegionGraph`
- execution-plan derivation and lowering

Its job is to support a "describe first, lower later" programming model for
parallel regions.

### 2.5 `HardwareManager`

`HardwareManager` is the hardware-object registry.

It owns metadata for:

- HBM objects
- VRAM objects
- MRAM objects

It is not responsible for tensor semantics. It acts more like a hardware-side
registry of visible objects and addresses.

### 2.6 `TileTensorProgram`

`TileTensorProgram` is the user-facing facade.

This is the API layer users work with directly:

- `input(...)`
- `tensor(...)`
- `copy(...)`
- `matmul(...)`
- `row_op(...)`
- `parallel_region3d(...)`
- `compile()`

It orchestrates the lower layers rather than replacing them.

## 3. Main Units

The current system is organized around a few important units.

### 3.1 Logical objects

- `Input`
- `Tensor`
- `Vector`

These are the logical objects users author against.

### 3.2 Logical tiles

- `InputTile`
- `TensorTile`
- `VectorTile`

These are the main execution-side logical units in the tensor path.

For the current runtime, tile is the main tensor-world unit, not individual
element IR.

### 3.3 Backing values

- `ValueTile`

This represents one concrete backing value version.

The key idea is that a logical tile and its backing value are not the same
thing. One logical tile points to one current backing value, but that binding
may change over time.

### 3.4 Views

- `ValueTileView`

This represents the logical window a tile currently sees on a backing value.

This is central to the current write model, because many writes are not simply
"replace one whole tensor", but rather "update one logical view of a backing
value".

### 3.5 FP-domain units

- `FPVar`
- `FPFragment`

These belong to the FP domain and are intentionally separate from the main
tensor value/view path.

### 3.6 Parallel symbolic units

- `ParallelAccess`
- `ParallelExpr`
- `ParallelRegionGraph`
- `ParallelExecutionPlan`

These are used by the symbolic parallel path to represent access patterns,
expressions, captured regions, and derived execution plans.

## 4. Core Runtime Law

The most important runtime law in the current system is:

`logical tile -> ValueTileView -> compute -> bind/writeback`

This means:

1. start from the logical tile the user refers to
2. resolve the view that tile currently sees
3. run compute on the appropriate backing value(s)
4. bind the result back or write it out

This is more accurate than thinking in terms of "the tensor directly owns the
physical data".

## 5. Main Design Principles

### 5.1 Logical objects and physical backing are separated

The system intentionally separates:

- logical tensor identity
- physical backing value identity

This allows rebinding, alias-safe updates, partial views, and residency control
without pretending that one logical tensor always corresponds to one immutable
piece of storage.

### 5.2 Tile is the main tensor execution unit

The current compiler/runtime is built around tile-level execution.

That means:

- tensor lowering is primarily organized around tiles
- placement and residency are tracked per value/tile relationship
- many compute paths assume tile-granular movement and tile-granular writes

### 5.3 Writes are view-aware

A destination write is not treated as a blind overwrite by default.

Instead, the runtime asks:

- what view is being updated
- whether the old backing can be safely reused
- whether a new backing must be created
- whether old contents must be preserved

This is why `ValueTileView` and `PreparedWrite` exist.

### 5.4 Alias safety is more important than naive in-place update

If the destination aliases a live source, the runtime does not assume that
overwriting in place is safe.

It prefers to preserve correct read/write semantics first, then optimize the
physical path second.

### 5.5 Preserve-copy is the last resort

For partial updates, full physical copy in VRAM is intentionally treated as the
slow fallback path.

The preferred order is:

1. reuse old backing in place when safe
2. replace whole logical tile without preserve copy when possible
3. create a partial-update successor without physical copy when possible
4. use physical preserve copy only as a last resort

### 5.6 FP domain is separate from the tensor value/view domain

The system intentionally keeps:

- tensor path
- FP-var / FP-fragment path

as two related but distinct worlds.

This is important because FP-oriented scalar/vector logic often has different
requirements from tile-backed tensor writes.

### 5.7 Parallel regions are symbolic first, executable second

`parallel_region3d(...)` and `parallel_region2d(...)` are not immediate
execution blocks.

They work in two stages:

1. capture symbolic accesses and expressions
2. finalize and lower them into execution steps later

So the parallel path is fundamentally a symbolic programming model, not just a
Python loop shortcut.

### 5.8 Prefer structured layouts over ad hoc 2D authoring

Although rank-2 shapes exist in the current runtime, the most mature and
recommended authoring path is still BSHD-style structured layouts:

- `(B, S, H, D)`
- `(B, S, 1, hidden)`

In practice, new kernels should usually prefer these layouts over building new
flows around plain 2D matrices.

## 6. How To Think About The Current Compiler

One useful mental model is:

- `TensorManager` decides what the user meant logically
- `ValueManager` decides what backing values exist and where they live
- `ComputeManager` decides how to execute the operation
- `ThreadManager` handles symbolic parallel capture and lowering
- `TileTensorProgram` ties the whole flow together

Another useful summary is:

"The current compiler/runtime is a tile-centric, view-aware lowering system
that separates logical tensors from physical backing values and lowers both
normal tensor compute and symbolic parallel regions into emulator ISA."

## 7. Short Summary

If this document needs to be reduced to one paragraph, the most accurate short
description is:

The current `TileTensorProgram` compiler/runtime is a TileTensor testbench
program builder organized around logical tensors, tile-based execution,
backing-value rebinding, and explicit lowering. Its core idea is that logical
tensors do not directly own physical storage; instead, logical tiles resolve to
views on backing values, compute runs on those values with alias-safe update
rules, and the final result is lowered into emulator-oriented ISA. Parallel
regions add a symbolic capture layer on top of that model.
