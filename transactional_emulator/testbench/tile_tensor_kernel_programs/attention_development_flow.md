# Attention Kernel Development Flow

This document explains how [`attention.py`](./attention.py) is written with
`TileTensorProgram`, and how to use the same workflow to build or modify
similar kernels.

The target reader is someone who will actually write kernels in this style, so
this guide focuses on:

- how to think about the algorithm before writing code
- how to map that algorithm onto the current `TileTensorProgram` primitives
- why each buffer and loop exists
- what assumptions the current implementation relies on
- what to change safely and what to treat carefully

## 1. What This Kernel Computes

`attention.py` implements one tiled causal multi-head attention kernel in the
following form:

1. Load `Q`, `K`, and `V`
2. Compute attention scores `Q @ K^T`
3. Apply scale `1 / sqrt(head_dim)`
4. Optionally apply the causal mask
5. Run a numerically stable softmax over the score rows
6. Compute `softmax(scores) @ V`
7. Write the result to output

Before looking at the DSL version, it helps to keep the intended algorithm in
plain pseudocode.

### MHA pseudocode

```text
for b in range(batch_size):
    for q_block in query_blocks:
        for head_group in head_groups:
            Q_blk = load Q[b, q_block, head_group, :]
            O_blk = 0
            M_blk = -inf   # running row max
            L_blk = 0      # running row sum / denominator

            for kv_block in kv_blocks:
                if causal and kv_block is fully in the future of q_block:
                    continue

                K_blk = load K[b, kv_block, head_group, :]
                V_blk = load V[b, kv_block, head_group, :]

                S_blk = Q_blk @ K_blk^T
                S_blk = S_blk * (1 / sqrt(head_dim))

                if causal and kv_block overlaps the diagonal of q_block:
                    S_blk = S_blk + causal_mask

                M_prev = M_blk
                M_tile = rowwise_max(S_blk)
                M_blk = max(M_prev, M_tile)

                P_scale = exp(M_prev - M_blk)
                P_blk = exp(S_blk - M_blk)
                L_blk = L_blk * P_scale + rowwise_sum(P_blk)
                O_blk = O_blk * P_scale

                PV_blk = P_blk @ V_blk
                O_blk = O_blk + PV_blk

            O_blk = O_blk / L_blk
            store O_blk to OUT[b, q_block, head_group, :]
```

This is the real structure implemented by the kernel. The rest of this
document explains how each line above maps onto `TileTensorProgram`.

The reference implementation in the same file is:

```python
scores = torch.einsum("bshd,bthd->bsht", q, k) * scale
scores = scores.masked_fill(mask, -1.0e4)   # when causal=True
probs = torch.softmax(scores, dim=-1)
out = torch.einsum("bsht,bthd->bshd", probs, v)
```

The DSL version does the same computation, but it does not call one
high-level `softmax` operator. Instead, it explicitly builds the running-max
and running-sum flow needed for tiled FlashAttention-style accumulation.

## 2. The Authoring Mindset

When writing a kernel in this framework, it helps to follow this order:

1. Fix the logical tensor shapes.
2. Decide the tile shape implied by `mlen`.
3. Decide which loops are batch loops, which are sequence-block loops, and
   which are head-group loops.
4. Decide which intermediates must survive across `kv_block` iterations.
5. Decide which operations are best expressed as:
   - tensor-domain ops like `copy`, `matmul`, `atomic_add`, `clear`
   - row-wise ops like `row_op(..., op="reduce_max")`
   - symbolic lane-wise ops inside `parallel_region2d/3d`
6. Only then start writing the program.

That is exactly the flow used in `attention.py`.

## 3. Program Setup

The builder entry point is:

```python
build_flashattention_program(
    *,
    mlen,
    blen,
    hlen,
    seq_len,
    head_count,
    causal=True,
)
```

Important parameters:

- `mlen`: the lane width and tile width used by the runtime
- `blen`: hardware/runtime block parameter used by the simulator
- `hlen`: per-head hidden dimension
- `seq_len`: sequence length
- `head_count`: number of attention heads
- `causal`: whether to apply an upper-triangular causal mask

The code fixes:

- `group_heads = 4`
- `batch_size = 2`

and validates:

- `mlen % hlen == 0`
- `head_count % group_heads == 0`

These are not decorative checks. They reflect layout assumptions used by the
kernel and the current runtime lowering.

## 4. Inputs, Outputs, and Persistent Tensors

The first structural step is to declare external tensors:

```python
q_in = prog.input("Q_IN", (batch_size, seq_len, head_count, hlen))
k_in = prog.input("K_IN", (batch_size, seq_len, head_count, hlen))
v_in = prog.input("V_IN", (batch_size, seq_len, head_count, hlen))
out_buf = prog.input("OUT", (batch_size, seq_len, head_count, hlen))
```

In this DSL, `input(...)` means "externally backed tensor object", not
"read-only semantic input". `OUT` is declared as an input-shaped object
because the testbench expects a pre-existing output buffer to write into.

This version writes results directly into `out_buf` once one query-block /
head-group result is finalized. There is no separate persistent output tensor
between the local accumulation buffer and the external output buffer.

## 5. Scalars and Scratch Fragments

The next step is to declare constants and temporary buffers:

```python
scale_scalar = prog.constant(...)
neg_inf_scalar = prog.constant(...)
zero_scalar = prog.constant(...)
scores_max = prog.alloc_fragment(...)
logsum = prog.alloc_fragment(...)
scores_max_prev = prog.alloc_fragment(...)
scores_scale = prog.alloc_fragment(...)
scores_sum = prog.alloc_fragment(...)
mask_head = prog.alloc_fragment(...)
```

### Constants

These are FP-domain scalars used repeatedly:

- `scale_scalar = 1 / sqrt(hlen)`
- `neg_inf_scalar = -1e4`
- `zero_scalar = 0`

### Scratch buffers

These buffers are the heart of the softmax implementation:

- `scores_max`: running row-wise max for the current query block
- `logsum`: running row-wise denominator for the current query block
- `scores_max_prev`: previous max before incorporating the next KV block
- `scores_scale`: `exp(old_max - new_max)` used to rescale previous partial sums
- `scores_sum`: row-wise sum of the current exponentiated score tile
- `mask_head`: one per-head causal mask tile

### Practical interpretation of `alloc_fragment` here

In this codebase, `alloc_fragment` is best read as "allocate kernel scratch
storage with fragment-style intent", not "allocate a special user-visible
shared-memory type". For tensor-shaped temporaries like these, use it as your
default scratch-buffer constructor.

## 6. Why the Shapes Look the Way They Do

Several shapes in this file are easy to read past and hard to redesign later,
so they are worth understanding explicitly.

### `q_group`, `k_group`, `v_group`, `out_group`, `pv_group`

These all use:

```python
(1, mlen, group_heads, hlen)
```

Interpretation:

- batch dimension is fixed to one block at a time
- sequence dimension is one `mlen`-sized block
- head dimension is one `group_heads`-sized block
- feature dimension is the head hidden size `hlen`

### `score_group`

This uses:

```python
(1, mlen, group_heads, mlen)
```

Interpretation:

- each query block has `mlen` query positions
- each KV block contributes `mlen` key positions
- for each local head in the group, we need one `mlen x mlen` score tile

### `scores_max` and `logsum`

These use:

```python
(1, group_heads, mlen)
```

Interpretation:

- one set per local head
- one scalar per query row

This is the exact shape needed for row-wise stable softmax bookkeeping.

### `mask_head`

This uses:

```python
(1, mlen, 1, mlen)
```

Interpretation:

- one query tile by one key tile
- no per-head variation inside the tile
- reusable across all heads in the group

That last point is why the head axis is `1` rather than `group_heads`.

## 7. Building the Causal Mask

The causal mask is constructed once, ahead of the main attention loops:

```python
with prog.parallel_region3d((mlen, 1, mlen), name="causal_mask") as (q_local, head_local, k_local):
    mask_head[0, q_local, head_local, k_local] = prog.if_then_else(
        q_local < k_local,
        neg_inf_scalar,
        zero_scalar,
    )
```

### Why this is written as a symbolic parallel region

Because the mask is naturally lane-parallel:

- each `(q_local, k_local)` position is independent
- each element is just a predicate plus a select
- the runtime can lower this declaratively

This is a good example of when `parallel_region3d` is better than a Python
nested loop. You are describing a tile-wide data-parallel relation, not
manually stepping through scalar work.

### Why the comparison is `q_local < k_local`

For causal attention, positions may only attend to current and previous keys.
So entries above the diagonal should be suppressed with `-inf`-like values.

## 8. The Outer Loop Nest

After setup, the kernel enters the real execution structure:

```python
for batch_index in range(batch_size):
    for q_block in prog.pipelined(q_block_count, num_stages=2):
        for group_block in prog.parallel(group_block_count):
            ...
            for kv_block in prog.pipelined(q_block_count, num_stages=2):
                ...
```

Each loop has a different role.

### `batch_index`

This is plain Python control over batches. Nothing special here.

### `q_block`

This selects one query tile of length `mlen`:

- `q_start = q_block * mlen`
- `q_end = q_start + mlen`

All scratch state inside this scope is tied to the current query tile.

### `group_block`

This selects a group of attention heads:

- `group_start = group_block * group_heads`

The kernel processes heads in fixed-size groups rather than all heads at once.
That keeps the working set bounded and lines up with the chosen layout.

### `kv_block`

This iterates over all key/value tiles that contribute to the current query
tile. This is where streaming accumulation happens.

## 9. Query-Block Working Set

Inside one `(batch_index, q_block, group_block)` scope, the kernel allocates:

```python
q_group
score_group
out_group
pv_group
```

These are the main per-query-block working buffers.

### `q_group`

Loaded once for the current query tile and reused across all `kv_block`
iterations.

### `score_group`

Holds `Q @ K^T` for the current query tile against the current key tile.

### `out_group`

Holds the running output accumulation for the current query tile and head
group. It starts at zero and is updated across KV tiles.

### `pv_group`

Temporary buffer for `(softmax tile) @ V tile` before accumulation into
`out_group`.

## 10. Loading the Query Tile

The query load is:

```python
prog.copy(
    q_in[batch_index, q_start:q_end, group_start : group_start + group_heads, :],
    q_group,
)
```

This is a very common pattern in this DSL:

1. slice a logical input tensor
2. copy it into a scratch tile/group buffer
3. do all local compute from that buffer

For reusable operands like `Q`, this is almost always the right move.

## 11. Initializing the Running Softmax State

Before visiting any KV block, the kernel initializes:

```python
prog.clear(out_group)
for local_head in prog.parallel(group_heads):
    prog.fill(logsum[0, local_head, :], 0.0)
    prog.fill(scores_max[0, local_head, :], neg_inf_scalar)
```

This corresponds to the standard online softmax initialization:

- output accumulator starts at zero
- denominator accumulator starts at zero
- row-wise max starts at negative infinity

### Why `fill` is used here

These destinations are fragment/FP-style scratch destinations, so `fill` is a
natural fit for row-wise scalar initialization.

## 12. Skipping Future KV Tiles in Causal Mode

At the top of the KV loop:

```python
if causal and kv_start >= q_end:
    continue
```

This is an important coarse-grain optimization.

If the key tile starts entirely after the end of the query tile, then every
score in that tile would be masked out. There is no reason to:

- load `K`
- load `V`
- compute `QK^T`
- apply softmax machinery

So the kernel skips the whole block.

This is one of the first things to preserve if you refactor the loop nest.

## 13. Loading K and V for One KV Block

For each contributing KV tile:

```python
k_group = prog.alloc_fragment(...)
v_group = prog.alloc_fragment(...)
prog.copy(k_in[...], k_group)
prog.copy(v_in[...], v_group)
```

These are loaded fresh inside the `kv_block` loop because they are only needed
for the current streamed step.

Immediately after `matmul(q_group, k_group.T, score_group)`, the code frees
`k_group`:

```python
prog.free_tensor_tile(k_group)
```

This is a good pattern for short-lived temporaries. Free them as soon as the
last use is finished.

## 14. Computing the Score Tile

The score computation is:

```python
prog.matmul(q_group, k_group.T, score_group)
```

Conceptually this is:

```python
score_group = Q_block @ K_block^T
```

The transpose on `k_group.T` matters because the score matrix is query-by-key.

## 15. Per-Head Softmax Processing

After the score tile is produced, the kernel iterates over local heads:

```python
for local_head in prog.parallel(group_heads):
    score_head = score_group[0, :, local_head : local_head + 1, :]
    out_head = out_group[0, :, local_head : local_head + 1, :]
```

This isolates one head's `mlen x mlen` score tile and one head's output tile.

### Step 1: scale the scores

```python
prog.row_op(score_head, scale_scalar, "mul", dim=-1)
```

This multiplies each row by `1 / sqrt(hlen)`.

### Step 2: apply the causal mask on overlapping diagonal tiles

```python
if causal and kv_start < q_end and q_start < kv_end:
    prog.atomic_add(score_head, mask_head, score_head)
```

This is a finer-grain condition than the earlier whole-block skip.

- the earlier check skips tiles fully in the future
- this check handles tiles that overlap the query tile's causal boundary

In those overlapping tiles, only part of the tile is invalid, so the mask is
added elementwise.

### Step 3: preserve the previous running max

```python
prog.copy(scores_max[0, local_head, :], scores_max_prev[0, local_head, :])
prog.fill(scores_max[0, local_head, :], neg_inf_scalar)
prog.row_op(score_head, op="reduce_max", out=scores_max[0, local_head, :], dim=-1)
```

This computes the new tile-local row max while retaining the previous running
max for stable rescaling.

## 16. Merging the Old and New Row Max

After per-head local reduction, the kernel merges old and new maxima:

```python
with prog.parallel_region2d((group_heads, mlen)) as (h, s):
    scores_max[0, h, s] = prog.max(scores_max[0, h, s], scores_max_prev[0, h, s])
```

This is a clean example of what `parallel_region2d` is good at:

- same operation across all heads
- same operation across all rows
- scalar expression over fragment elements

Conceptually:

```python
new_running_max[h, s] = max(tile_max[h, s], old_running_max[h, s])
```

## 17. Stable Online Softmax Update

This is the most important part of the kernel.

For each local head, the code performs:

```python
scores_scale = exp(scores_max_prev - scores_max)
score_head = exp(score_head - scores_max)
scores_sum = reduce_sum(score_head)
logsum = logsum * scores_scale + scores_sum
out_head = out_head * scores_scale
```

This is the tiled online-softmax recurrence.

### Why this works

Suppose previous KV blocks already contributed:

- a row-wise max `m_prev`
- a row-wise denominator `l_prev`
- a row-wise accumulated output `o_prev`

When a new score tile arrives with a larger max, the old accumulated values
must be rescaled into the new exponent frame. That is what:

```python
exp(m_prev - m_new)
```

does.

### How the code expresses it

The rescale factor is built with symbolic parallel regions:

```python
with prog.parallel_region2d((1, mlen)) as (_, s):
    scores_scale[0, 0, s] = scores_max_prev[0, local_head, s] - scores_max[0, local_head, s]
with prog.parallel_region2d((1, mlen)) as (_, s):
    scores_scale[0, 0, s] = prog.exp(scores_scale[0, 0, s])
```

Then row-wise tensor ops update the score tile:

```python
prog.row_op(score_head, scores_max[0, local_head, :], "sub", dim=-1)
prog.row_op(score_head, op="exp", dim=-1)
prog.fill(scores_sum[0, 0, :], 0.0)
prog.row_op(score_head, op="reduce_sum", out=scores_sum[0, 0, :], dim=-1)
```

And finally the running denominator and output are updated:

```python
with prog.parallel_region2d((1, mlen)) as (_, s):
    logsum[0, local_head, s] = logsum[0, local_head, s] * scores_scale[0, 0, s] + scores_sum[0, 0, s]

prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)
```

Notice the split:

- scalar-per-row recurrences use `parallel_region2d`
- row-wise tile transforms use `row_op`

That is a very good style to follow in new kernels.

## 18. Weighted Value Accumulation

Once `score_head` has been exponentiated and normalized relative to the
running softmax frame, the kernel computes:

```python
prog.matmul(score_group, v_group, pv_group)
prog.free_tensor_tile(score_group)
prog.free_tensor_tile(v_group)
prog.atomic_add(out_group, pv_group, out_group)
prog.free_tensor_tile(pv_group)
```

Conceptually:

```python
pv_group = exp_shifted_scores @ V_block
out_group += pv_group
```

The accumulation into `out_group` is delayed until after the per-KV-tile
matrix multiply, which keeps the structure simple and explicit.

### Why free buffers aggressively here

By the time `pv_group` has been accumulated:

- `score_group` is no longer needed
- `v_group` is no longer needed
- `pv_group` is no longer needed

Releasing them early reduces pressure on the runtime's tile storage.

## 19. Final Normalization After All KV Blocks

After the `kv_block` loop finishes, `out_group` still holds the numerator of
softmax-weighted accumulation, while `logsum` holds the denominator.

So the kernel performs:

```python
for local_head in prog.parallel(group_heads):
    out_head = out_group[0, :, local_head : local_head + 1, :]
    with prog.parallel_region2d((1, mlen)) as (_, s):
        scores_scale[0, 0, s] = prog.reci(logsum[0, local_head, s])
    prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)
```

Conceptually:

```python
out_head = out_head / logsum
```

The reciprocal is computed first, then applied as a row-wise multiply.

This is a common pattern in the current DSL because `reci` exists as a scalar
parallel expression and `row_op(..., "mul")` handles the row broadcast.

## 20. Writing Back the Result

Once one query/head-group block is finalized:

```python
prog.copy(
    out_group,
    out_buf[batch_index, q_start:q_end, group_start : group_start + group_heads, :],
)
prog.free_tensor_tile(q_group)
prog.free_tensor_tile(out_group)
```

This version uses direct final writeback. That keeps the dataflow shorter and
avoids maintaining one extra full output tensor when the block result is
already complete at this point.

## 21. Why This File Is a Good Template

This kernel is a strong example for new authors because it demonstrates all of
the most important authoring patterns in one place:

- declaring logical external tensors and internal tensors
- using `alloc_fragment` for scratch
- loading reusable tiles once and streamed tiles per iteration
- mixing `matmul`, `row_op`, `fill`, `clear`, and `atomic_add`
- using `parallel_region3d` for tile-wide symbolic construction
- using `parallel_region2d` for per-row scalar recurrences
- explicitly freeing short-lived buffers

If someone understands this file well, they can usually move on to writing
layernorm-like, RMSNorm-like, or other block-streaming attention variants.

## 22. A Practical Recipe for Writing Similar Kernels

When implementing a new kernel, use this sequence.

### Step A: write the math in plain tensor form

Before touching the DSL, write the intended computation in PyTorch-like math.

Examples:

- `scores = q @ k.T`
- `probs = softmax(scores)`
- `out = probs @ v`

If the math is not clear yet, the DSL code will get messy quickly.

### Step B: pick the block that is reused the longest

In attention, that block is `q_group`.

Ask:

- what should be loaded once per outer loop?
- what should be loaded once per streamed inner loop?
- what must survive across streamed tiles?

### Step C: identify online recurrence state

For softmax-style streaming kernels, this usually means:

- running max
- running denominator
- running output numerator

If you miss this step, you will often allocate the wrong scratch buffers.

### Step D: choose the right primitive for each part

Use these rules of thumb.

- Use `copy` for staging external tensor slices into working buffers.
- Use `matmul` for block GEMM-shaped work.
- Use `row_op` for row-wise reductions or broadcasts along the last dimension.
- Use `parallel_region2d/3d` for scalar lane-parallel recurrences, predicates,
  and elementwise expressions that are clearer as symbolic assignment graphs.
- Use `atomic_add` for explicit tensor accumulation.
- Use `fill` and `clear` for initialization.

### Step E: free short-lived temporaries immediately after their last use

This matters in this runtime. Do not wait until the end of the whole kernel if
the buffer only lives for one inner-loop iteration.

## 23. Common Pitfalls

### Pitfall 1: forgetting which buffers are per-query-block and which are per-KV-block

If a buffer must persist across `kv_block`, do not allocate it inside the
wrong scope.

Good examples of persistent buffers:

- `out_group`
- `logsum`
- `scores_max`

Short-lived examples:

- `k_group`
- `v_group`
- `pv_group`

### Pitfall 2: using a whole-tile mask when only a diagonal overlap needs masking

There are two separate causal checks for a reason:

- whole future tiles are skipped entirely
- overlapping diagonal tiles still need elementwise masking

Do not collapse them into one condition unless you re-derive the logic.

### Pitfall 3: losing numerical stability during streaming softmax

If you remove `scores_max_prev`, `scores_scale`, or the rescaling of
`out_head`, the kernel may still run but produce incorrect results.

The online recurrence is not optional bookkeeping. It is the algorithm.

### Pitfall 4: trying to express everything as `parallel_region`

Not every operation should become a symbolic region.

In this DSL:

- dense row transforms are often cleaner with `row_op`
- dense block products are clearer with `matmul`
- symbolic regions are best for per-lane scalar relations and predicates

### Pitfall 5: assuming `alloc_fragment` means "special vector-only fragment"

In this codebase, it is used as a general scratch allocation tool for
tensor-shaped intermediates too. Read it as scratch-with-fragment-intent, not
as a promise about one specific storage abstraction.

## 24. What You Can Change Safely

These changes are usually straightforward if you keep shapes consistent.

- changing `causal=True/False`
- changing constant values like the mask sentinel
- changing `seq_len` as long as the block logic still matches
- changing `head_count` as long as it remains divisible by `group_heads`
- changing `group_heads` if the layout and capacity assumptions still hold
- adding extra instrumentation or comments

## 25. What You Should Treat Carefully

These changes can easily break correctness or lowering assumptions.

- changing the `mlen` / `hlen` relationship
- changing scratch shapes without re-deriving the recurrence
- moving `free_tensor_tile(...)` calls earlier
- replacing symbolic `parallel_region2d/3d` blocks with ad hoc loops
- changing head grouping without checking matmul and row-shape expectations
- modifying the online softmax recurrence

## 26. How to Read This File Top to Bottom

A good reading strategy for new authors is:

1. Read the PyTorch golden implementation first.
2. Read only the declarations and scratch allocations.
3. Identify the outer loop nest and annotate what each loop indexes.
4. Follow one `kv_block` iteration end to end.
5. Ignore the final testbench section until the kernel body is clear.

If someone can explain one complete `kv_block` update in their own words, they
usually understand the kernel well enough to start editing it.

## 27. Minimal Mental Model

If you want one compact summary to keep in mind while writing similar kernels,
use this:

- `Q` is loaded once per query block.
- `K` and `V` are streamed tile by tile.
- `score_group` is the current `QK^T` tile.
- `scores_max` and `logsum` carry the online softmax state across KV tiles.
- `out_group` carries the running output numerator across KV tiles.
- `parallel_region` handles scalar lane-parallel glue.
- `row_op` handles row-wise transforms and reductions.
- `matmul` handles the block products.
- final output normalization happens only after all KV tiles have contributed.

That is the development flow behind this kernel.
