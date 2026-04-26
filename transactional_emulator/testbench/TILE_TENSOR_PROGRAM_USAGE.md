# TileTensorProgram Usage Guide

This document is a user-facing guide for
`transactional_emulator/testbench/tile_tensor_program.py`.

It focuses on how to author programs with `TileTensorProgram`, which public
APIs are available, how the common workflows fit together, and what the
current implementation constraints are.

For runtime internals and design notes, see:

- `TILE_TENSOR_RUNTIME_NOTES.md`
- `TILE_TENSOR_KERNEL_PROGRAMS.md`

## 1. What This File Is

`TileTensorProgram` is the main authoring API for building TileTensor testbench
programs.

At a high level it lets you:

- declare logical inputs and working tensors
- express tile-level tensor movement and compute
- express FP-domain scalar / fragment compute
- describe symbolic parallel regions
- lower all of that into emulator-oriented ISA text through `compile()`

The most common workflow is:

1. create `TileTensorProgram`
2. declare `input(...)` and `tensor(...)`
3. move data with `copy(...)`
4. run compute with `matmul(...)`, `atomic_*`, `row_op(...)`, `pure_fp_compute(...)`, or parallel regions
5. copy the final tensor back to an output buffer
6. call `compile()`

## 2. Construction

Typical construction:

```python
from tile_tensor_program import TileTensorProgram

prog = TileTensorProgram(
    mlen=64,
    blen=4,
    btmm_hlen=16,
    real_data_ratio=1.125,
    vram_tile_capacity=16,
    mram_tile_capacity=4,
    fpram_capacity=1024,
)
```

Main constructor parameters:

- `mlen`
  Tile width / height in logical elements. Many vectorized operations assume
  rows of width `mlen`.
- `blen`
  Block width used by the underlying matmul lowering.
- `btmm_hlen`
  Head width for BTMM-style paths. Must be a positive divisor of `mlen`.
- `real_data_ratio`
  Scaling factor used when allocating HBM addresses.
- `vram_tile_capacity`, `mram_tile_capacity`, `fpram_capacity`
  Resource sizing hints for the emulator/runtime.
- `hbm_base_addr`
  Initial HBM allocation base.

## 3. Logical Shapes

The runtime currently works with logical shapes of rank 2, 3, or 4:

- 2D: `(rows, cols)`
- 3D: `(x, y, z)` and internally treated as `rows=x`, `cols=y*z`
- 4D: `(B, S, H, D)` and internally treated as `rows=B*S`, `cols=H*D`

Common patterns:

- plain matrix: `(rows, cols)`
- sequence-hidden tensor: `(batch, seq, 1, hidden)`
- attention layout: `(batch, seq, heads, head_dim)`

Important recommendation:

- Although rank-2 `(rows, cols)` plain matrices are supported in several basic
  paths, the 2D path is not yet the most mature authoring path in the current
  runtime.
- For new kernels and new program authoring, prefer writing tensors in BSHD
  form, even when the computation could be expressed as a plain 2D matrix.
- In practice, the most stable and best-covered authoring style today is to
  use `(B, S, H, D)` or `(B, S, 1, hidden)` layouts rather than building new
  flows around rank-2 tensors.

## 4. Main Public APIs

The most important user-facing methods are:

- declaration
  - `input(name, logical_shape, hbm_addr=None)`
  - `tensor(name, logical_shape)`
  - `vector(name, logical_shape)`
  - `alloc_fragment(name, logical_shape, init_zero=False, dtype="fp32")`
  - `fp_var(name, value=0.0, size=1)`
  - `fp_fragment(name, shape, init=0.0)`
  - `constant(name, value, size=1)`

- tensor movement / compute
  - `copy(src, dst)`
  - `matmul(src1, src2, dst)`
  - `atomic_add(src1, src2, dst)`
  - `atomic_sub(src1, src2, dst)`
  - `atomic_mul(src1, src2, dst)`
  - `row_op(src, rhs=None, op=..., out=None, dim=-1)`
  - `clear(tensor)`
  - `free_tensor_tile(operand, weak=None)`

- FP-domain compute
  - `fp_copy`, `fp_fill`, `fp_add`, `fp_sub`, `fp_mul`, `fp_max`
  - `fp_exp`, `fp_reci`, `fp_sqrt`
  - `fill(dst, src)` for FP-domain destinations

- symbolic parallel programming
  - `parallel_region3d((S, H, D), name=None)`
  - `parallel_region2d((X, Y), name=None)`
  - `where(predicate, on_true, on_false)`
  - `if_then_else(predicate, on_true, on_false)`
  - `max(lhs, rhs)`, `exp(x)`, `reci(x)`, `sqrt(x)`
  - `pair(axis)`, `half_index(axis)`
  - `parallel_execution_plans()`
  - `lower_parallel_execution_plans()`

- loop hints and planning helpers
  - `parallel(extent)`
  - `pipelined(extent, num_stages=1)`

- reporting / output
  - `write_operation_report(output_path)`
  - `build_fp_preload(min_size=0)`
  - `compile()`

- advanced / low-level APIs
  - `pure_fp_compute(src1, dst, src2=None, control=...)`
  - `fp_kernel(src1, dst, src2=None, control=...)`
  - `mapf(...)`, `mapf_t(...)`
  - `btmm(...)`, `btmm_write(...)`
  - `alloc_hbm_addr(...)`, `add_hbm_object(...)`
  - `emit_*` family for direct ISA emission

## 5. Minimal End-To-End Example

This is the simplest practical pattern: declare input and output buffers, copy
through a working tensor, then compile.

```python
from tile_tensor_program import TileTensorProgram

prog = TileTensorProgram(
    mlen=64,
    blen=4,
    btmm_hlen=16,
    vram_tile_capacity=16,
    mram_tile_capacity=4,
    fpram_capacity=1024,
)

x_in = prog.input("X_IN", (1, 64))
out_buf = prog.input("OUT", (1, 64))
x = prog.tensor("X", (1, 64))

prog.copy(x_in, x)
prog.copy(x, out_buf)

asm = prog.compile()
print(asm)
```

Typical real kernels insert additional compute between the two `copy(...)`
operations.

## 6. Declaring Operands

### 6.1 `input(...)`

Use `input(...)` for logical tensors backed by HBM input/output objects.

```python
x_in = prog.input("X_IN", (batch, seq, 1, hidden))
out_buf = prog.input("OUT", (batch, seq, 1, hidden))
```

Notes:

- Inputs are usually sources, but an `Input` may also be used as a final
  writeback target.
- You may provide `hbm_addr=...` if the buffer must live at a fixed HBM
  address.

### 6.2 `tensor(...)`

Use `tensor(...)` for normal working tensors managed by the runtime.

```python
x = prog.tensor("X", (batch, seq, 1, hidden))
y = prog.tensor("Y", (batch, seq, 1, hidden))
```

These are the standard temporary / internal compute operands.

### 6.3 `vector(...)`

Use `vector(...)` when you want an FP-backed vector-style object. Vector tiles
are associated with FP fragments rather than the normal tensor value/view path.

This is mainly relevant for:

- explicit FP-domain authoring
- `parallel_region2d`, which currently lowers only FP-backed `Vector`
  destinations

### 6.4 `alloc_fragment(...)`

Use `alloc_fragment(...)` for scratch temporaries. Depending on the shape, the
runtime may return a normal `Tensor` or a `Vector`.

Typical examples:

```python
centered = prog.alloc_fragment("CENTERED", (1, seq_len, 1, hidden_size))
mean = prog.alloc_fragment("MEAN", (1, 1, seq_len))
```

This is the most common way to allocate internal working buffers.

### 6.5 FP scalar / fragment declarations

```python
scale = prog.fp_var("scale", value=0.125)
eps = prog.constant("eps", 1.0e-6, size=seq_len)
frag = prog.fp_fragment("tmp_fp", (seq_len,), init=0.0)
```

Use these for scalar values and small FP-domain arrays that should live in
FP_MEM / FP fragments.

## 7. Indexing, Slicing, and Element Access

The API supports Python indexing and slicing:

```python
x[batch_index, :, :, :]
x[:, :, 0:1, :]
x[0, 0, :]
```

Common patterns:

- whole tensor: `x`
- slice view: `x[:, :, :, :]`, `x[0, :, :, :]`, `x[:, :, 0:1, :]`
- element-like FP access: `scores_max[0, h, s]`

Important distinction:

- tensor / input slices participate in the tile/value runtime
- element-style accesses are used heavily by FP-domain and parallel-expression
  APIs

## 8. Data Movement

### 8.1 `copy(src, dst)`

`copy(...)` is the basic logical movement operator.

```python
prog.copy(x_in, x)
prog.copy(y, out_buf)
prog.copy(x[0, :, :, :], tmp[0, :, :, :])
```

Behavior:

- tensor/input to tensor: rebinds or prepares backing values as needed
- tensor to input: performs logical writeback
- FP-domain operands: routes to `fp_copy`

## 9. Tensor Compute APIs

### 9.1 `matmul(src1, src2, dst)`

`matmul(...)` is the main matrix multiply entrypoint. Internally it may choose
one of several paths:

- default tilewise matmul
- view-based matmul for grouped narrow-head layouts
- BTMM/QKT path when `src2` is explicitly transposed and shapes match

Example:

```python
prog.matmul(x, w, y)
prog.matmul(q_group, k_group.T, score_group)
```

Notes:

- explicit transpose syntax on the RHS is currently reserved for the BTMM/QKT
  route
- not every transposed case is supported

### 9.2 `atomic_add`, `atomic_sub`, `atomic_mul`

These implement elementwise tile ops with alias-safe destination updates.

```python
prog.atomic_add(a, b, out)
prog.atomic_mul(centered, centered, sq)
prog.atomic_add(score_head, mask_head, score_head)
```

Use these when:

- the operation is tilewise / elementwise
- the destination may alias one input
- you want runtime-managed preservation and rebinding behavior

### 9.3 `clear(tensor)`

Zeroes all current value tiles of a tensor in VRAM.

```python
prog.clear(accumulator)
```

### 9.4 `free_tensor_tile(...)`

Releases runtime bindings for one tensor, slice, or tile.

```python
prog.free_tensor_tile(tmp)
prog.free_tensor_tile(score_group)
```

This is commonly used for scratch fragments once their values are no longer
needed.

## 10. Row Operations

`row_op(...)` is the main API for row-wise vector math and reductions along the
last logical dimension.

Supported operations:

- unary row ops
  - `exp`
  - `reci`
- binary row ops
  - `mul`
  - `add`
  - `sub`
- reductions
  - `reduce_sum`
  - `reduce_max`

Examples:

```python
prog.row_op(x_head, op="reduce_sum", out=mean[0, 0, :], dim=-1)
prog.row_op(x_head, mean[0, 0, :], "sub", dim=-1)
prog.row_op(work_head, inv_rms[0, 0, :], "mul", dim=-1)
prog.row_op(score_head, op="exp", dim=-1)
```

Current contract:

- `dim` must be `-1`
- reductions require `out=...`
- binary row ops require `rhs`
- the current lowering is most natural when each logical row has width `mlen`

## 11. FP-Domain APIs

The FP domain is intentionally separate from the tensor value/view pipeline.
Use it for FP-variable / FP-fragment compute, including:

- scalar FP values
- short FP vectors
- fragment-backed row data
- elementwise FP math over one mapped FP-var list
- reduction-associated post-processing

Important recommendation:

- `pure_fp_compute(...)` and related FP mapping APIs are still supported, but
  they are better treated as lower-level runtime-facing interfaces
- for new FP-heavy authoring, the preferred direction is usually
  `parallel_region2d(...)` when the computation naturally fits lane-wise FP
  vector work
- in other words, new user-facing FP logic should generally prefer symbolic
  `parallel_region2d(...)` over building more code around `pure_fp_compute(...)`

### 11.1 Recommended direction for new FP logic: `parallel_region2d(...)`

For new FP-oriented logic, the preferred high-level style is usually
`parallel_region2d(...)`.

Examples:

```python
with prog.parallel_region2d((group_heads, mlen)) as (h, s):
    scores_max[0, h, s] = prog.max(scores_max[0, h, s], scores_max_prev[0, h, s])

with prog.parallel_region2d((1, mlen)) as (_, s):
    scores_scale[0, 0, s] = prog.exp(scores_scale[0, 0, s])
```

Why this is the preferred direction:

- it reads more like the intended lane-wise FP computation
- it avoids exposing as much FP-var plumbing in user-facing kernels
- it is closer to the current symbolic parallel authoring direction

Current limitation:

- `parallel_region2d(...)` is still narrower than a fully general FP compiler
- today it is mainly for FP-backed `Vector`-style destinations and supported
  FP expression forms

### 11.2 Low-level FP compute APIs: `pure_fp_compute(...)` and `fp_kernel(...)`

These are the generic FP compute entrypoints.

```python
prog.pure_fp_compute(mean[0, 0, :], mean[0, 0, :], src2=recip_hidden, control="mul")
prog.pure_fp_compute(var[0, 0, :], var[0, 0, :], src2=eps_vec, control="add")
```

Important clarification:

- these are not limited to one scalar element
- in normal use, they operate over the full FP-var list returned by `mapf(...)`
- that means calls like `mean[0, 0, :]` or `var[0, 0, :]` are vector-style
  elementwise FP operations over the whole slice, not one single-element op
- the runtime applies the requested FP operation across the mapped destination
  FP vars

How to think about them:

- these APIs are still useful
- but they are better considered low-level or transitional interfaces
- they expose more of the FP-mapping model than we usually want in the main
  user-facing programming style

Supported `control` values include:

- `copy`
- `add`
- `sub`
- `mul`
- `max`
- `exp`
- `reci`
- `sqrt`

In existing code they can be convenient, but for new authoring they should not
be treated as the primary recommended FP style.

### 11.3 Convenience wrappers

These all dispatch into the FP kernel path:

- `fp_copy(src, dst)`
- `fp_fill(dst, src)`
- `fp_fill_from_addr(dst, src_fpram_addr)`
- `fp_add(src1, src2, dst)`
- `fp_sub(src1, src2, dst)`
- `fp_mul(src1, src2, dst)`
- `fp_max(src1, src2, dst)`
- `fp_exp(src, dst)`
- `fp_reci(src, dst)`
- `fp_sqrt(src, dst)`

### 11.4 `fill(dst, src)`

`fill(...)` currently supports FP-domain destinations only.

```python
prog.fill(mean[0, 0, :], 0.0)
prog.fill(var[0, 0, :], 0.0)
```

### 11.5 Low-level mapping helpers: `mapf(...)` and `mapf_t(...)`

These are lower-level APIs for mapping operands into FP variables.

- `mapf(operand)`
  Returns the FP-var list for an operand.
- `mapf_t(tensor_operand, fp_operand, control="mixed")`
  Mixed tensor-to-FP mapping helper.

Most kernel authors do not need to call these directly unless they are doing
custom FP-domain orchestration.

### 11.6 `build_fp_preload(...)`

Returns the FP_MEM initialization array in address order.

```python
fp_init = prog.build_fp_preload(min_size=32)
```

This is typically passed into a testbench artifact writer.

## 12. Symbolic Parallel Programming

The runtime supports symbolic parallel authoring through `parallel_region3d`
and `parallel_region2d`.

Inside these scopes:

- indexing with region axes produces symbolic loads
- assignments register symbolic compute instead of running immediately
- region finalization builds an execution plan and lowers it later

### 12.1 `parallel_region3d((S, H, D))`

This is the main parallel API for tensor-backed parallel compute.

Example from RoPE-style code:

```python
with prog.parallel_region3d((seq_len, head_count, full_dim), name="rope_q") as (s, h, d):
    q_out[0, s, h, d] = prog.if_then_else(
        d % 2 == 0,
        xq[0, s, h, d] * cos_t[0, s, h, d]
        + xq[0, s, h, prog.pair(d)] * neg_sin_t[0, s, h, d],
        xq[0, s, h, prog.pair(d)] * sin_t[0, s, h, d]
        + xq[0, s, h, d] * cos_t[0, s, h, d],
    )
```

Useful helpers:

- `if_then_else(...)` / `where(...)`
- arithmetic operators on symbolic expressions
- comparisons like `<`, `<=`, `==`, `>=`, `>`
- `pair(d)` for even/odd lane partner selection
- `half_index(d)` for half-width grouping logic
- unary helpers `exp(...)`, `reci(...)`, `sqrt(...)`
- binary helper `max(...)`

### 12.2 `parallel_region2d((X, Y))`

This is a narrower parallel path used for FP-backed vector destinations.

Example:

```python
with prog.parallel_region2d((group_heads, mlen)) as (h, s):
    scores_max[0, h, s] = prog.max(scores_max[0, h, s], scores_max_prev[0, h, s])
```

Another example:

```python
with prog.parallel_region2d((1, mlen)) as (_, s):
    scores_scale[0, 0, s] = prog.exp(scores_scale[0, 0, s])
```

Current 2D contract is much narrower than the 3D path:

- destinations must be FP-backed `Vector`-style objects
- lowering supports FP expression kernels, not the full tensor write path

### 12.3 Plan inspection

You can inspect or force lowering of captured parallel regions:

```python
plans = prog.parallel_execution_plans()
prog.lower_parallel_execution_plans()
```

`compile()` automatically lowers deferred parallel plans if needed.

## 13. Loop Hints

### 13.1 `parallel(extent)`

Returns a range-like object and records a parallel loop hint.

```python
for local_head in prog.parallel(group_heads):
    ...
```

This is commonly used in kernel authoring to express per-head or per-lane
structure around other runtime ops.

### 13.2 `pipelined(extent, num_stages=1)`

Returns a range-like object and records a pipelining hint.

```python
for i in prog.pipelined(tile_count, num_stages=2):
    ...
```

This is mainly a planning hint for future/lower layers.

## 14. Reporting and Debugging

### 14.1 `write_operation_report(...)`

Writes a human-readable trace of recorded operations and a delta report.

```python
prog.write_operation_report("build/my_operation_report.txt")
```

The report includes:

- operation kind and details
- VRAM / MRAM / HBM-resident value tiles
- active FP fragments
- value-tile-to-slice references
- FP-fragment-to-value references

### 14.2 `compile()`

`compile()` lowers any remaining parallel execution plans, normalizes large
immediates, and returns the generated ISA text.

```python
asm = prog.compile()
```

## 15. HBM and Direct ISA Helpers

These are advanced APIs for users who need explicit control over memory
objects or direct instruction emission.

### 15.1 HBM helpers

- `alloc_hbm_addr(elems)`
- `add_hbm_object(name, shape, hbm_addr=None)`

Example:

```python
base = prog.alloc_hbm_addr(64 * 64)
prog.add_hbm_object("X_BUF", (64, 64), hbm_addr=base)
```

### 15.2 Direct emit helpers

Available direct emit APIs include:

- `emit_hbm_tile_to_mram(...)`
- `emit_load_tile_from_hbm(...)`
- `emit_store_tile_to_hbm(...)`
- `emit_zero_vram_tile(...)`
- `emit_map_v_fp_tile(...)`
- `emit_map_fp_v_tile(...)`
- `emit_btmm(...)`
- `emit_btmm_wo(...)`
- `emit_matmul(...)`
- `emit_slot_matmul(...)`
- `emit_tile_binary(...)`
- `emit_tile_add(...)`
- `emit_fp_kernel(...)`
- `emit_row_operation(...)`

These bypass much of the higher-level logical runtime. Use them only when:

- building custom lowering paths
- debugging ISA generation
- prototyping a new runtime feature

Most kernel authors should prefer the higher-level APIs first.

## 16. Advanced BTMM APIs

`matmul(...)` already routes into BTMM when the pattern matches, but the lower
level entrypoints also exist:

- `btmm(lhs_packed_value=..., rhs_value=..., task_id="btmm")`
- `btmm_write(btmm_state=..., tile_count=..., ...)`

These are advanced runtime hooks for explicit BTMM orchestration and are
normally used by the internal specialized matmul path.

## 17. Common Authoring Patterns

### 17.1 Input -> work tensor -> compute -> output

```python
prog.copy(x_in, x)
prog.matmul(x, w, y)
prog.copy(y, out_buf)
```

### 17.2 LayerNorm-style flow

```python
prog.copy(x_in, x)
prog.copy(x, centered)
prog.fill(mean[0, 0, :], 0.0)
prog.row_op(centered[0, :, 0:1, :], op="reduce_sum", out=mean[0, 0, :], dim=-1)
prog.pure_fp_compute(mean[0, 0, :], mean[0, 0, :], src2=recip_hidden, control="mul")
prog.row_op(centered[0, :, 0:1, :], mean[0, 0, :], "sub", dim=-1)
```

### 17.3 Parallel symbolic transform

```python
with prog.parallel_region3d((seq_len, heads, dim), name="transform") as (s, h, d):
    y[0, s, h, d] = prog.if_then_else(
        d % 2 == 0,
        x[0, s, h, d],
        x[0, s, h, prog.pair(d)],
    )
```

## 18. Current Constraints

The current implementation is intentionally narrower than a general tensor
compiler. The main constraints to document clearly are:

- logical shapes are currently rank 2, 3, or 4
- rank-2 plain matrix support exists, but it is not yet the most mature or
  recommended primary authoring path
- for new development, prefer expressing tensors in BSHD-style layouts
- many high-level tensor ops currently require each BSHD operation to address
  exactly one batch at a time
- `row_op(...)` currently supports `dim=-1` only
- `fill(...)` currently supports FP-domain destinations only
- explicit transposed RHS matmul is currently reserved for the BTMM/QKT route
- `parallel_region3d` lowering supports a focused subset of symbolic tensor
  expressions
- `parallel_region2d` currently supports FP-backed vector destinations only
- current parallel lowering assumes structured, row-oriented execution and is
  not a general arbitrary-index compiler
- several low-level emit helpers assume widths tied to `mlen`

## 19. Practical Advice

- Prefer `copy`, `matmul`, `atomic_*`, and `row_op` first.
- Prefer BSHD-style logical layouts for new code, even if a problem looks like
  a plain 2D matrix at first glance.
- Use `alloc_fragment(...)` for scratch temporaries instead of manually
  managing low-level storage.
- Use `parallel_region3d(...)` when the computation is naturally lane-wise over
  `(S, H, D)`.
- Prefer `parallel_region2d(...)` over `pure_fp_compute(...)` when writing new
  FP-vector-style logic.
- Use `write_operation_report(...)` when debugging residency, rebinding, or
  unexpected tile reuse.
- Use direct `emit_*` APIs only when you intentionally want ISA-level control.

## 20. Where To Look For Examples

Good in-repo examples:

- `tile_tensor_kernel_programs/linear.py`
  Basic input -> matmul -> output flow
- `tile_tensor_kernel_programs/layernorm.py`
  `row_op`, `pure_fp_compute`, fragments, and FP helpers
- `tile_tensor_kernel_programs/rmsnorm.py`
  Reduction-heavy FP/tensor mixed flow
- `tile_tensor_kernel_programs/rope.py`
  `parallel_region3d`, `if_then_else`, `pair`
- `tile_tensor_kernel_programs/attention.py`
  mixed `matmul`, `row_op`, `parallel_region2d`, and scratch-fragment usage

## 21. Quick Reference

If you only remember one short checklist, use this:

1. declare `input(...)` and `tensor(...)`
2. `copy(...)` data into working tensors
3. use `matmul`, `atomic_*`, `row_op`, or parallel regions for compute
4. use `pure_fp_compute` / `fp_*` for scalar or FP-fragment math
5. `copy(...)` final tensors into output buffers
6. call `compile()`
