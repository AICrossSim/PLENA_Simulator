# Tile Tensor Kernel Programs

This directory contains `TileTensorProgram` rewrites of the kernels under
`/home/a13247568123124/project/tilelang_kernels`.

Status:

- `linear.py`: implemented
- `layernorm.py`: implemented
- `rmsnorm.py`: implemented
- `attention.py`: implemented from the existing multi-head FlashAttention testbench flow
- `elementwise.py`: implemented for `modulate` and `residual_gate`
- `rope.py`: implemented with explicit even/odd-split inputs and outputs
- `activations.py`: documented as unsupported with the current `TileTensorProgram` primitive set

Notes:

- These builders favor clarity and direct correspondence to the TileLang kernels.
- The runtime now has a substantial `parallel`/`parallel_region3d` execution
  model. Kernels in this directory should prefer expressing lane-wise work with
  symbolic parallel regions instead of manually spelling out per-lane scalar
  loops whenever the computation is naturally data-parallel.
- Broadcasted operands such as bias, scale, shift, and RoPE coefficients are
  modeled as already-expanded input tensors when that keeps the program simple.
- The activation kernels are intentionally not faked here: the current runtime
  exposes `exp/reci/sqrt` but does not expose tensor-domain `sigmoid` or
  `tanh`, which the TileLang GELU/SiLU kernels require.

## Parallel Feature

`TileTensorProgram` now exposes a first-class symbolic parallel programming
model:

- `parallel_region3d((S, H, D), name=...)`
  Captures a 3-axis parallel region and returns symbolic axes `(s, h, d)`.
- Tensor and input indexing inside the region builds a graph instead of
  executing eagerly.
- Assignments such as `dst[s, h, d] = ...` are validated, planned, and lowered
  into cache-aware execution cycles automatically.
- `where(...)` / `if_then_else(...)`, `pair(...)`, and `half_index(...)`
  support masked elementwise flows and RoPE-style lane remapping.

This matters because the runtime is no longer limited to "one handwritten loop
equals one emitted sequence". A kernel author can describe a whole lane-parallel
computation declaratively, and the thread manager derives:

- symbolic assignment graphs
- input/output cache slots
- per-cycle load / compute / writeback plans
- final ISA-oriented lowering

In practice, this is the feature that makes the new attention, RoPE, and
elementwise rewrites scalable instead of remaining one-off demos.

## Current Lowering Contract

The current implementation is already useful, but it is intentionally precise
about what it lowers:

- Parallel destinations must be tensor-backed assignments indexed by the active
  3D axes.
- Arithmetic expressions currently support `add`, `sub`, and `mul`.
- Predicates currently support binary comparisons such as `lt`, `le`, `gt`,
  `ge`, and `eq`.
- Lowering currently expects one full-width contiguous row per cycle, with
  the innermost lane width aligned to the runtime `mlen` contract.

So the parallel system is powerful, but it is not "arbitrary Python inside a
context manager". Kernel builders should think of it as a compact DSL for
describing structured SIMD-style tile programs.
