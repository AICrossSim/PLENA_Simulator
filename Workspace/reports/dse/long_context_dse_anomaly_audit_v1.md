# Long-Context DSE Anomaly Audit v1

> **Superseded anomaly snapshot, audited 2026-07-26.** The FFN discontinuities
> identified here were addressed by `affine-loop-v2`; later multi-chip
> lineage defects were addressed separately. Current distributed evidence is
> in [`tile_aware_multichip_lineage_audit_v4.md`](tile_aware_multichip_lineage_audit_v4.md).

## Scope

This audit checks the completed Qwen3-32B long-context studies at:

```text
sequence length = 32768
batch size      = 1
```

The first study contains 2,048 points over `MLEN=256..4096` and
`chip_count=1..8`. The second contains 2,048 points after enabling
`ffn_address_schedule=live-stride-v1` and extending the domain to
`MLEN=8192` and 16 chips.

The audit uses exact matched comparisons: precision profile, INT width,
chip count, SRAM policy, and the non-swept hardware fields are held constant
while one hardware axis changes. This avoids interpreting sampler mix as a
hardware trend.

## Resolution Update: Unified Affine FFN v2

The projection-full SRAM reversal documented below has now been fixed by
`ffn_projection_schedule=affine-loop-v2`. Matrix SRAM capacity continues to
select the same K-chunk boundaries, but all capacities now use one explicit
affine loop IR instead of switching to `_ffn_asm_with_loops`.

Controlled 64-layer results are:

| Configuration | Legacy template | Affine v2 | Compute reduction | FFN `S_ADDI_INT` reduction |
|---|---:|---:|---:|---:|
| `M4096/B64/N4`, 7 tiles | 75.835B cycles | 44.419B cycles | 41.43% | 99.76% |
| `M8192/B32/N4`, 4 tiles | 149.955B cycles | 85.345B cycles | 43.09% | 99.52% |

At `M4096/B64/N4`, the new per-chip compute share is `11.105 s`, below
the latest 2-tile streaming reference of `11.616 s`. Matrix opcode counts,
HBM opcode counts, DMA occurrences, and the complete DMA manifest hash are
identical between compatibility and affine-v2 arms. A short-context
`M2048/B1024/N1` point improves by another `0.71%`, so the fix does not trade
the long-context recovery for a short-context regression.

Tiny transactional execution passes the unchanged correctness gate on both
paths and produces a bitwise-identical complete VRAM image. Full
implementation and evidence are in
[`../compiler/unified_affine_ffn_loop_lowering_v2.md`](../compiler/unified_affine_ffn_loop_lowering_v2.md).

The complete 2,048-point affine-v2 study strengthens the controlled result:

```text
COMPLETE / PRUNED / FAIL              2048 / 0 / 0
affine structural guard passes        2048 / 2048
compatibility fallbacks                         0
matched adjacent SRAM comparisons             123
latency regressions >2%                          0
latency improvements >2%                      108
```

The pre-fix live-stride study had 59 regressions among 121 matched pairs.
Under the new schema, no matched SRAM increase makes latency worse, so
projection-full capacity is no longer a source of compiler-induced
performance reversal.

The remainder of this document preserves the pre-fix audit as the historical
root-cause record. Its warnings about projection-full results apply to the old
study schema, not to new `affine-loop-v2` studies.

## Resolved Large-MLEN Anomaly

The original `M8192/B32/N4` evaluation reported:

```text
latency             64.855 s
S_ADDI_INT work    173.947B cycles
S_ADDI fraction      67.1%
```

`live-stride-v1` removes dead post-compute updates and replaces live
large-immediate updates with invariant register-register strides:

```text
latency             21.796 s
S_ADDI_INT work      1.543B cycles
S_ADDI reduction      99.11%
Matrix work delta      0
HBM work delta         0
```

This was a compiler-lowering defect, not an intrinsic penalty of an
8,192-wide MatrixMachine.

## Historical Systematic Anomaly: SRAM Reversal

Increasing Matrix SRAM should not make an otherwise identical compiler
schedule slower: the compiler can always leave excess capacity unused.
The completed studies violate this invariant.

Exact adjacent SRAM comparisons in the original study show:

```text
total matched comparisons        139
latency improvement >2%           27
approximately neutral             49
latency regression >2%            63
latency regression >10%           29
maximum regression               52.0%
```

The live-stride study still shows 59 regressions among 121 matched
comparisons. Its maximum sampled regression is 73.2%. Almost every regression
enters `projection-full`.

### Representative point

The following pair is identical except for Matrix SRAM capacity:

```text
MLEN=4096, BLEN=64, chip_count=4
precision=w_mxint4__act_mxint8__kv_mxint8__fp_e6m5
INT width=16
```

| Metric | 2-tile streaming | 7-tile projection-full |
|---|---:|---:|
| DSE latency | 13.437 s | 20.424 s |
| Aggregate compute | 53.636B cycles | 81.587B cycles |
| FFN stage | 18.009B cycles | 45.960B cycles |
| Attention stage | 35.626B cycles | 35.626B cycles |
| `S_ADDI_INT` | 3.390B | 32.443B |
| V4 memory work | 717.9 ms | 593.2 ms |

The larger SRAM correctly reduces HBM work, but FFN address arithmetic grows
by much more. Re-evaluating the 7-tile point with `live-stride-v1` reproduces
the same 20.424 s result, proving that this is separate from the dead-update
fix.

The current lowering sets the projection K-chunk width from available Matrix
SRAM. A larger capacity therefore changes the FFN projection from several
small chunks to one large K loop. In the representative pair,
`agu_residual_s_addi` grows from 53.091M to 507.052M per layer. The current
AGU repeat/refolding pass therefore eliminates far fewer address updates in
the large-chunk schedule. Residual large-stride updates remain in the hot loop
and are expanded into many `S_ADDI_INT` instructions.

This audit does not attribute that loss of coverage solely to the six-stream
descriptor limit. The large K loop contains only a subset of all surrounding
affine address patterns, and the interaction among nested repeats, repeat
refolding, and candidate legality needs a dedicated schedule-level trace
before choosing between an AGU-capacity change and a compiler-only rewrite.

The schedule-level trace resolves this question. The stream histograms are:

```text
2 tiles: ... 4-stream loops=1801, 6-stream loops=1
7 tiles: ... 4-stream loops=1801, 6-stream loops=1
```

The 6-stream limit is reached by exactly one loop in both schedules. Increasing
the descriptor count therefore cannot explain or recover the regression.
Instead, the number of instructions recovered by exact repeat refolding falls
from 2,632,175 to 207,360, while residual `S_ADDI_INT` rises by 9.6x.

`loop-agu-v2` was also evaluated on the 7-tile point. It produced exactly the
same 20.424416 s latency as `loop-agu-v1`; the post-increment extension did not
expose or remove the missing address work. This establishes compiler schedule
formation/refolding, rather than AGU parallelism, as the first-order fix.

This affects both latency and system energy. The current selectors that use
`projection-full`, especially lowest-energy and closest-area selectors, must
be treated as provisional until this issue is fixed and the study is rerun.

## Checks That Did Not Find Another Structural Reversal

In the completed live-stride study, exact matched adjacent comparisons found:

```text
chip-count comparisons     93, regressions >2% = 0
BLEN comparisons          205, regressions >2% = 0
MLEN comparisons          158, regressions >2% = 0
area monotonicity violations across MLEN/BLEN/chips/SRAM = 0
```

Thus no second unknown large discontinuity was found in BLEN, chip scaling, or
the structural area equations.

The near-linear chip-count speedup is not RTL evidence. It follows from the
explicit optimistic TP+SP analytical partition and should retain that fidelity
label.

## Why M8192 Is Not Globally Best

Under the aggregate 908.6 mm2 constraint, the minimum sampled feasible
latencies are:

```text
M2048:  2.883 s, up to 16 chips feasible
M4096:  3.972 s, up to  8 chips feasible
M8192: 10.785 s, up to  4 chips feasible
```

This is mainly an aggregate-area tradeoff. A larger per-chip array consumes
the budget with fewer chips. In the controlled streaming A/B at the same
four-chip configuration:

```text
M2048/B32: 28.927 s
M4096/B32: 21.512 s
M8192/B32: 21.796 s
```

The useful MLEN scaling saturates around 4096 for this compiler/model, but the
former 64.855 s collapse is gone. M8192 remains a structural extrapolation and
does not establish timing closure or RTL feasibility.

## Expected but Potentially Confusing Results

### Infeasible COMPLETE trials

The latest study contains 446 trials above 908.6 mm2. Optuna records these as
`COMPLETE` because objective evaluation succeeded; feasibility is carried
separately as a constraint. None of the 110 reported Pareto trials exceeds the
area budget. A `COMPLETE` state must therefore not be read as
area-feasible.

### Weak SRAM latency benefit

For points without the FFN lowering reversal, partial KV residency reduces V4
traffic, but most of the workload remains compute-bound. SRAM can lower energy
or memory work without materially lowering stage-roofline latency. This is
expected and is distinct from the projection-full regression.

### Precision-dependent latency

Identical hardware can vary by roughly 20-28% across precision profiles because
Matrix structural timing and physical DMA packing depend on the actual
weight/activation/KV families. Precision is not merely an accuracy label.

## Follow-Up Status

The original follow-up requested:

1. Enumerate legal K-chunk widths up to physical capacity.
2. Use the shared compiler/CostEmitter lowering to select the minimum-work
   width. Since total Matrix work is unchanged across chunk partitions, the
   selection can minimize exact non-Matrix instruction work without depending
   on a new timing artifact.
3. Preserve remaining SRAM for partial K/V residency.
4. Add an invariant test that increasing SRAM cannot increase generated
   ideal-II1 work.
5. Rerun the long-context DSE before using projection-full selectors.

A temporary safe policy is to cap projection chunk width at the largest width
that does not increase exact compiler work. This avoids discarding the SRAM
capacity or changing ISA/RTL while the remaining affine-address lowering is
improved.

The longer-term compiler cleanup should emit explicit affine nested-loop IR
for FFN output blocks, activation columns, and K tiles. Base addresses and
strides should be materialized once, with the inner K loop exposing its two
pointer streams directly to AGU-v1. This removes dependence on reconstructing
large loops by pattern matching already-expanded instruction sequences.

Increasing AGU streams should only be reconsidered after that rewrite and only
if the resulting trace contains a material number of profitable loops with
more than six simultaneously live affine streams.

`affine-loop-v2` completes the explicit nested-loop IR portion of this work and
removes the capacity-triggered template switch. The implemented guard compares
K chunks and Matrix/HBM/partial-sum census before selecting the candidate, and
retains a per-projection compatibility fallback. AGU-v1 remains the default;
AGU-v2 and a wider descriptor file remain unjustified by the measured traces.

The remaining operational action is to rerun the 2,048-point long-context
study under the new DSE schema before replacing published selectors. The old
study remains useful only as a before-fix baseline.
