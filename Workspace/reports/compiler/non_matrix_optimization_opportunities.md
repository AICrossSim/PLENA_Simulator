# Non-Matrix Optimization Opportunities

## Scope

This analysis uses the current Qwen3-32B one-layer trace:

```text
seq_len=482, batch_size=16
MLEN=VLEN=2048, BLEN=1024, HLEN=128
streamed-v2 softmax state
broadcast-k-major-v1 QK traversal
loop-agu-v1 address generation
ideal-II1 compute timing at 1 GHz
```

AGU v1 is the production default. AGU v2 was retired from the implementation:
the fixed Qwen traces selected zero legal post-increment streams, while the
mapped v2 sidecar added 2,846.249 um2 and left negligible 1 ns timing margin.
Its report remains as negative-result evidence.

## Current Stage Breakdown

| Stage | Total cycles | Matrix | Vector | Scalar | Control |
|---|---:|---:|---:|---:|---:|
| Attention | 19,278,518 | 4,974,656 | 5,617,408 | 8,417,860 | 268,594 |
| FFN | 5,741,159 | 4,331,664 | 1,167,136 | 239,173 | 3,186 |
| Final norm | 275,780 | 0 | 93,984 | 181,730 | 66 |
| Global loads | 24,797 | 0 | 24,576 | 98 | 123 |
| **One layer** | **25,320,254** | **9,306,320** | **6,903,104** | **8,838,861** | **271,969** |

Non-Matrix work is 16,013,934 cycles, or 63.25% of ideal-II1 compute.
Attention contributes 89.32% of that non-Matrix work. Optimizing FFN control
or generic frontend issue cannot materially change the result before the
attention path is addressed.

## Opportunity 1: Loop-Invariant Selector Hoisting

The attention stage executes 2,778,750 `S_ADDI_INT` instructions. Trace
variants expose two dominant classes that are not affine addresses:

```text
gp5 <- constant batch/softmax segment selector: about 987k instructions
gp3 <- constant head/segment selector:          about 987k instructions
```

The selected value is invariant for the surrounding row loop, but the current
lowering materializes it inside the loop. Moving it before the loop is a
compiler-only, bitwise-preserving transformation and does not require AGU v2.

```text
ideal upper-bound removal: about 1.97M cycles/layer
fraction of total compute: about 7.8%
```

The actual saving will be slightly smaller because one setup instruction per
outer segment/head remains. This should be the first implementation target.

## Opportunity 2: Compact Segment-Statistics Normalization

Q/K RMSNorm already uses `V_RED_SUM_SEGS`, reducing 555,264 independent
single-segment reductions to 38,560 multi-segment reductions. However, each
of the 555,264 resulting segment statistics is still normalized through:

```text
S_LD_VLANE_FP
S_MUL_FP
S_ADD_FP
S_RSQRT_FP
S_ST_VLANE_FP
```

This is 2,776,320 Scalar instructions per layer. A compact statistics unit
operating on at most 16 valid lanes could apply multiply, epsilon addition,
and reciprocal-square-root to the packed reduction result. It should not
instantiate an RSQRT unit across all 2,048 Vector lanes.

Replacing the current sequence with three compact-vector operations per
multi-segment result gives the counterfactual:

```text
current work:      2,776,320 cycles
replacement work:   115,680 cycles
upper-bound saving: 2,660,640 cycles/layer
fraction of total:  10.5%
```

This requires a new compact-stats datapath/opcode and timing/area calibration,
but offers the largest isolated non-Matrix opportunity with clear structural
counts.

## Opportunity 3: Packed Multi-Row Softmax

The short-sequence attention path still handles one score row at a time:

```text
V_RED_MAX_SEG
V_SUB_VF
V_EXP_V
V_RED_SUM_SEG
S_RECI_FP
V_MUL_VF
```

Each operation occurs 493,568 times per layer. At `seq_len=482` and
`VLEN=2048`, four independent score rows fit in four 512-lane segments.
Packing four rows into one Vector SRAM word would allow segmented MAX/SUM,
segment broadcast, one vector EXP, compact reciprocal, and segment scaling.

The six-operation core has a four-way upper-bound saving of:

```text
about 2.22M cycles/layer
about 8.8% of total compute
```

This is not a compiler-only layout change under the current Matrix writeout:
the BMM result is emitted as one full-width score row. A useful implementation
needs packed/active-width Matrix writeout or an equivalent low-cost scatter.
Repacking with ordinary `V_SHIFT_V` instructions is likely to consume much of
the saving. The benefit also falls for contexts at or above `MLEN`.

## Opportunity 4: Reduction Overwrite Mode

First-block softmax loads neutral values before every reduction:

```text
S_LD_FP(-inf) -> V_RED_MAX_SEG
S_LD_FP(0)    -> V_RED_SUM_SEG
```

Each load occurs 493,568 times. A reduction overwrite bit that ignores the
old scalar destination would preserve the exact result while removing:

```text
987,136 Scalar loads/layer
3.9% of total compute
```

This is a small and low-risk RTL/ISA extension compared with multi-row
packing. It also reduces Scalar FP SRAM traffic. It overlaps with the
multi-row softmax opportunity and the savings must not be added twice.

## Opportunity 5: Single-Block Softmax State Elision

For the `seq_len=482` case there is only one K block. The normalization sum is
currently stored to Scalar FP SRAM before PV and loaded again for final output
normalization. The compiler could compute `1/l` immediately, normalize P, and
then issue PV:

```text
remove S_ST_FP(l): about 493,568
remove S_LD_FP(l): about 493,568
```

The direct upper bound is another 3.9%, plus some address work. This changes
the quantization point because P is scaled before Matrix PV rather than O
after PV. It therefore requires accuracy validation and cannot be described
as bitwise-preserving. It does not generalize directly to recurrent
multi-K-block online softmax.

## Smaller Opportunities

### Shift-add packed output

Direct packed-O accumulation executes about 493,568 avoidable
`V_SHIFT_V` operations. A destination-lane Matrix writeout or
shift-and-add Vector operation could save roughly 1.9% before setup effects.

### FFN activation

The FFN has 106,496 occurrences each of `V_SUB_VF`, `V_EXP_V`, `V_ADD_VF`,
and `V_RECI_V`. A SiLU macro-operation could save at most about 319k
instructions, or 1.3% of total compute. This is lower priority than attention
and introduces a fused opcode.

### Control and HBM

All control work is only 1.07% of compute. HBM V4 work is about 1.151 ms per
layer and is almost entirely hidden by the current stage roofline in the
short-context reference. Neither is the next latency bottleneck, although HBM
remains important for long-context and energy studies.

## Recommended Order

1. Implement selector loop-invariant code motion and remeasure the exact
   opcode delta. This is compiler-only and bitwise-preserving.
2. Add reduction overwrite mode. It is a small hardware change with a
   cycle-exact expected count.
3. Prototype compact segment-stat normalization and synthesize only the
   compact 16-lane leaf to estimate area/timing.
4. Evaluate packed multi-row softmax together with a packed Matrix writeout
   interface; do not implement expensive Vector repacking first.
5. Consider single-block P pre-normalization only as an accuracy-qualified
   compiler mode.

The counterfactual percentages above overlap. They are isolated upper bounds,
not an additive prediction of final latency.
