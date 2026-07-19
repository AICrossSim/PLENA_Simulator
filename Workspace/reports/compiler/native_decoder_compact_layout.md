# Native Decoder Compact Batch/Head Packing

**Status (2026-07-19):** Integrated and retained as the default native layout.
The 21.706 s best latency in this report is an intermediate A/B result. After
the packed-attention and vector/scalar passes documented in the adjacent
reports, the same exhaustive workload reaches 16.405 s. See
`../system_validation_status.md` for the current end-to-end result.

## Scope

This change removes two compiler-created utilization losses in the native
Qwen3 decoder path without changing the ISA, RTL, numerical comparison gate,
or latency calibration:

1. short sequences from different logical batches can share one `MLEN` row
   slab; and
2. multiple packed-GQA logical groups can share one `MLEN` Q/O storage block.

The same planner is used by the executable compiler and CostEmitter. Therefore
the modeled work and emitted ISA use the same physical row and column layout.
The legacy layout remains available through `native_layout_mode="legacy"` for
A/B tests.

## Sequence Layout

For `seq_len <= MLEN`, compact mode uses

```text
batch_pack_factor = min(batch_size, floor(MLEN / seq_len))
attention_group_count = ceil(batch_size / batch_pack_factor)
physical_rows = attention_group_count * MLEN
```

Logical batches are placed consecutively in a slab. The attention mask is
block-diagonal causal: a token can attend only to earlier tokens in its own
logical batch. The final incomplete group is padded with zero dummy batches.
For `seq_len > MLEN`, the previous one-batch multi-tile path is retained.

For Qwen3-32B prefill (`seq=482`, `batch=16`):

| MLEN | Batches/slab | Slabs | Legacy rows | Compact rows | Row utilization |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 16 | 8,192 | 8,192 | 94.14% |
| 1,024 | 2 | 8 | 16,384 | 8,192 | 94.14% |
| 2,048 | 4 | 4 | 32,768 | 8,192 | 94.14% |

## Packed-GQA Storage Layout

One logical attention group occupies

```text
attention_group_width = physical_broadcast * HLEN
groups_per_storage_block = floor(MLEN / attention_group_width)
physical_q_width = ceil(logical_group_count / groups_per_storage_block) * MLEN
```

Qwen3-32B has 64 Q heads, not 32, so its true logical Q/O width is
`64 * 128 = 8192`. Compact Q/O width remains 8,192 at MLEN 512, 1,024 and
2,048. The legacy MLEN=2,048 layout used 16,384 columns and therefore only 50%
of its head lanes.

Vector SRAM addresses must be VLEN-aligned. A transactional experiment that
attempted to read the second slot directly failed with
`address 8 not multiple of 16` in the tiny MLEN=16 case. Consequently, the
current ISA cannot select a later slot through an offset Q address. For a
shared block, the executable implementation broadcasts the current single KV
head across all stored lanes and retains only the score/output lanes assigned
to the target group. It never mixes different KV heads. At MLEN=2,048 this
means an eight-head logical group uses a 16-lane physical broadcast while the
other eight score lanes are discarded. This is the cost of Q/O storage
compaction on the current aligned Vector SRAM interface and is included in
CostEmitter work. Metadata reports storage lane utilization and execution lane
utilization separately; the MLEN=2,048 case is 100% storage-utilized but 50%
execution-lane-utilized during each attention group.

## Correctness Evidence

The direct packed-GQA transactional test was extended with
`--compact-layout`. It runs the same shared planner and exercises batch
co-packing, block-diagonal masking, two logical KV groups in one storage block,
and unchanged memory comparison.

| Test | Configuration | Result |
|---|---|---|
| Packed batches | `batch=4, seq=7, MLEN=VLEN=16, BLEN=4, HLEN=4, Hq/Hkv=4/2` | PASS, 100% all-close, max absolute error 0.003906 |
| Dummy tail | same, `batch=3` | PASS, real batches all-close; dummy output zero apart from BF16 subnormal noise (~1e-39) |

The existing comparison tolerance and PASS threshold were not modified.
`packed_active_row_indices()` is independently unit-tested for the exact
logical-to-physical mapping.

An additional whole-decoder diagnostic was not used as acceptance evidence:
both compact and legacy layouts produced an all-zero final VRAM output on the
same tiny learned-norm fixture. Since the failure is layout-independent, it is
tracked as an existing full native decoder/emulator integration issue rather
than being hidden or attributed to compact packing.

For the same realistic tiny Qwen3 fixture (including input/post-attention/final
RMSNorm and Q/K norm), CostEmitter's complete dynamic opcode histogram exactly
matches the expanded native ASM histogram. The summed multiplicity of each
`H_PREFETCH_M`, `H_PREFETCH_V`, and `H_STORE_V` DMA event also exactly matches
its dynamic opcode count.

Focused Python regression status after the final compiler, CostEmitter,
comparison, and DSE integration changes:

```text
56 compiler/CostEmitter/comparison tests passed
13 DSE integration tests passed
69 focused tests passed in total
```

The transactional packed-attention tests above are additional end-to-end runs;
their numerical comparison uses the existing gate and is not included in the
69-test Python count.

## CostEmitter A/B Evidence

The A/B experiment fixes model, precision, V4 HBM calibration, rtl-v1 opcode
timing, sequence length, batch size, and layer count. Only the native layout
mode changes. Source data is in
`Workspace/qwen3_32b_dense_analytic/native_layout_ab_large.json`.

| MLEN/BLEN | Layout | Rows | Q width | One-layer resource work | One-layer roofline | 64-layer roofline |
|---:|---|---:|---:|---:|---:|---:|
| 512/512 | legacy/compact | 8,192 | 8,192 | 497,517,280 cycles | unchanged | 31.062 s |
| 1,024/1,024 | legacy | 16,384 | 8,192 | 570,956,614 cycles | 573.055 ms | 35.454 s |
| 1,024/1,024 | compact | 8,192 | 8,192 | 371,552,462 cycles | 372.698 ms | 23.318 s |
| 2,048/1,024 | legacy | 32,768 | 16,384 | 889,814,509 cycles | 896.000 ms | 54.373 s |
| 2,048/1,024 | compact | 8,192 | 8,192 | 344,567,649 cycles | 345.718 ms | 21.706 s |

Compact-minus-legacy deltas are:

```text
MLEN=1024: -199,404,152 compute cycles/layer, -200.357 ms/layer
MLEN=2048: -545,246,860 compute cycles/layer, -550.282 ms/layer
```

The resource-work reduction is attributable to emitted instruction classes,
not a changed latency coefficient:

| Shape | Matrix cycles | Vector cycles | Scalar cycles | Control cycles |
|---|---:|---:|---:|---:|
| 1,024/1,024 | -13,020,184 | -133,586,944 | -50,647,944 | -2,149,080 |
| 2,048/1,024 | -30,257,296 | -330,600,448 | -179,045,376 | -5,343,740 |

The largest dynamic-count reductions are loop/address bookkeeping
(`S_ADDI_INT`, `C_LOOP_END`) and row-wise vector work
(`V_ADD_VV`, `V_MUL_VV/VF`, reductions). This is the expected signature of
removing padded physical rows; no rtl-v1 opcode timing or V4 coefficient was
changed in the A/B experiment.

For the representative points, compact MLEN=2,048 is now faster than MLEN=512.
This demonstrates that the earlier MLEN=512 optimum was primarily caused by
compiler padding, not an inherent large-MLEN latency penalty. Compact attention
work remains close between MLEN=1,024 and 2,048 because the latter executes the
aligned 16-lane broadcast described above; FFN work continues to decrease with
the larger machine.

## Exhaustive DSE Result

The corrected compiler layout was evaluated over the full categorical grid:

```text
103 precision profiles
5 MLEN values x 9 BLEN values x 3 INT widths
13,905 unique tuples
12,051 complete, 1,854 pruned only because BLEN > MLEN
64 worker processes, at most 4 trials per worker lifetime
```

The final canonical CSV contains exactly 13,905 unique tuples and no duplicate
keys. All 12,051 complete rows report layout schema v2 and compact mode. For
every MLEN, physical token rows and Q/O width are both 8,192.

The same highest-accuracy latency-optimal precision profile wins at every
MLEN: `W=E2M1, ACT=E1M2, KV=E4M3, FP=E6M5, INT=16`, with accuracy 0.98.

| MLEN/BLEN | Batch pack | HW broadcast | Attention lane utilization | Area P50/P90 (mm2) | Full-decoder roofline |
|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 1 | 100% | 6.08 / 6.13 | 186.219 s |
| 256/256 | 1 | 2 | 100% | 23.68 / 23.90 | 64.752 s |
| 512/512 | 1 | 4 | 100% | 93.49 / 94.36 | 31.062 s |
| 1,024/1,024 | 2 | 8 | 100% | 371.59 / 375.09 | 23.318 s |
| 2,048/1,024 | 4 | 16 | 50% | 846.53 / 854.47 | **21.706 s** |

Therefore the lowest modeled latency is now MLEN=2,048 rather than 512. It is
30.1% below the MLEN=512 optimum and 6.9% below MLEN=1,024. The smaller final
gain is explainable: MLEN=2,048 removes row and Q/O padding but executes two
eight-head groups through an aligned 16-lane operation, so half of the
attention lanes are discarded for each KV head.

Against the pre-compaction exhaustive run, the matched-tuple latency changes
are:

| MLEN | Median change over all matched tuples | Best-point change |
|---:|---:|---:|
| 128 | 0.0% | 0.0% |
| 256 | 0.0% | 0.0% |
| 512 | 0.0% | 0.0% |
| 1,024 | -44.6% | -34.2% |
| 2,048 | -77.7% | -60.1% |

This boundary is expected: sequences cannot be co-packed below MLEN=964, and
the MLEN=512 Q/O layout was already full. The MLEN=1,024 and 2,048 reductions
are therefore direct evidence of the two compiler layout changes rather than a
timing-model coefficient change.

The fastest all-MXINT candidate is only 0.33% slower than the global MXFP
candidate, but is substantially smaller:

```text
MXINT: 21.778 s, 646.48 mm2 P50, 672.44 mm2 P90, accuracy 0.98
MXFP:  21.706 s, 846.53 mm2 P50, 854.47 mm2 P90, accuracy 0.98
```

The point closest to the 826 mm2 A100 reference is 834.65 mm2 P50 (842.50 mm2
P90) at 21.706 s. The fastest point remains below the 908.6 mm2 budget under
both P50 and P90.

At the fastest point, one-layer compute resource work is 344,567,649 cycles:
66.6% vector, 29.3% scalar, 2.9% matrix and 1.2% control. Across the 64-layer
roofline, attention contributes 20.146 s and FFN 1.554 s. The remaining
MLEN=1,024 to 2,048 gap is consequently dominated by attention/vector behavior,
not sequence padding.

Artifacts:

- exhaustive run: `Workspace/qwen3_32b_dense_analytic/runs/roofline_v4_grid_compact_layout_20260718/`
- canonical data: `grid_trials.csv`
- per-MLEN legacy/compact comparison: `native_layout_dse_comparison.csv`
- plot: `qwen3_32b_latency_area_scatter_compact.png`
- A/B opcode evidence: `Workspace/qwen3_32b_dense_analytic/native_layout_ab_large.json`

All large candidates remain exploratory rather than signoff results. The
fastest point is outside the DC area calibration domain, and its rtl-v1 compute
work is 19.2% full-machine measured, 77.4% structural extrapolation and 3.4%
unsupported RTL opcode work. V4 HBM reports in-domain, but the result is a
stage-wise roofline estimate rather than cycle-exact scheduled RTL latency.

## Reporting and Limitations

Every CostTrace records the layout schema, logical/physical rows, row
utilization, batch packing factor, mask type, logical/physical Q width, storage
block count, and head-lane utilization. DSE trial records flatten these fields
into CSV/JSON columns.

The result is not a claim that latency must monotonically decrease with MLEN.
Vector reductions, hardware broadcast work, BLEN, memory traffic, and area can
still produce a smaller optimum. The acceptance criterion is narrower: no
remaining large-MLEN regression may be caused by per-batch row padding or an
empty Q/O storage half.
