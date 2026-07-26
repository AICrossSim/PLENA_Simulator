# Packed-GQA Pipeline-Aware Compiler Optimization v1

> **Historical rtl-v3 scheduling study.** Row interleaving remains available
> to the hazard-aware validation path, but formal DSE uses ideal-II1 timing and
> the current RTL-v4 lowering. The cycle values below are not current DSE
> objectives.

## Executive Summary

This change reschedules independent packed-GQA rows to use the RTL-v3 Scalar
ROB and pipelined Vector/Scalar units more effectively. It does not add an RTL
unit, ISA operation, approximation, or arithmetic shortcut. Within each query
row, the floating-point operation order is unchanged.

For Qwen3-32B prefill (`seq_len=482`, `batch=16`), the compute-only RTL-v3
scoreboard predicts the following one-layer improvements:

| MLEN / BLEN | Row-serial cycles | Interleaved cycles | Reduction |
|---:|---:|---:|---:|
| 512 / 64 | 523,713,105 | 489,590,353 | 6.52% |
| 1024 / 1024 | 181,700,007 | 147,114,919 | 19.03% |
| 2048 / 1024 | 155,274,688 | 120,350,912 | 22.49% |

At the required `2048/1024` point, total GQA-related stall cycles fall from
128,741,272 to 95,410,840 (25.89%). The formal stage-roofline latency falls
from 155.286 ms to 120.362 ms per layer at the reporting assumption of 1 GHz.
The repeated 64-layer estimate is therefore 9.938 s versus 7.703 s; this is a
layer-scaling estimate, not a full scheduled decoder replay.

## What Changed

### Timing-aware row scheduling

The new public option is:

```python
gqa_pipeline_schedule="row-interleaved-v1"
```

`row-serial` remains available for A/B and compatibility. The optimized mode
requires `vector_scalar_schedule="rtl-v3"` and reads the single timing source:

```text
transactional_emulator/calibration/rtl_opcode_timing_v3.json
SHA-256: 83270726dfbccf8b21346ba785a696354441db2abd32eea02a70edcea13cb447
```

Missing or non-v3 timing data is a hard error. Independent row chains are
ordered with measured ready latency and initiation interval while preserving
every row-local dependency. The fixed microkernel widths are:

| Path | Rows in flight |
|---|---:|
| First-block online softmax | 3 |
| Recurrent online softmax | 2 |
| Packed-O rescale/final reciprocal | 8 |
| Shift/add scratch ring | up to 16 |

Widths are checked against the calibrated 8-entry Scalar ROB, 16 FP registers,
and available Vector SRAM scratch. Tail rows use a smaller instance of the same
plan and are executed exactly once.

### Packed-O scheduling

Reciprocal and scalar-to-vector operations rotate across eight FP registers.
`V_SHIFT_V` producers use a Vector SRAM scratch ring before masked `V_ADD_VV`
consumers. Each scratch row has one producer and one consumer, and writes to
the packed output retain their original per-lane order.

### Nonresident K/V double buffering

When at least two Matrix SRAM tile slots exist and K/V are not resident, the
compiler emits this order:

```text
K[k] load -> QK[k] -> V[k] load -> softmax[k] -> PV[k]
          -> K[k+1] load -> O update[k]
```

K uses slot 0 and V uses slot 1. A slot is not overwritten before its consumer
finishes. Causal future blocks are skipped, resident K/V retain the existing
preload/reuse path, and insufficient capacity records
`mram_tile_capacity_lt_2` rather than silently claiming overlap.

## Why the Result Is a Scheduling Gain

The A/B harness rejects a result unless both arms have identical:

- Matrix opcode counts, including QK and PV;
- Vector and Scalar arithmetic opcode counts;
- HBM opcode counts;
- HBM read/write bytes; and
- memory-event counts.

All tested short- and long-context cases pass these invariants. The small
resource-work differences come only from control/address loop construction.
The `2048/1024` resource-work delta is -1,593,344 cycles, while its scheduled
makespan delta is -34,923,776 cycles. Most of the improvement therefore comes
from hiding dependency latency, not deleting arithmetic.

At `2048/1024`, notable stall changes are:

| Stall reason | Row-serial | Interleaved |
|---|---:|---:|
| Pipeline recovery | 7,025,227 | 4,776,267 |
| Scalar broadcast operand not retired | 3,547,520 | 98,688 |
| Scalar operand not ready | 52,750,080 | 49,955,584 |
| Vector SRAM port-A write | 44,780,925 | 16,166,013 |
| Mixed-latency vector in-order | 6,050,456 | 9,827,224 |

The increase in mixed-latency Vector stalls is disclosed: the scheduler trades
some Vector retirement pressure for much larger reductions in SRAM write and
Scalar dependency stalls.

## Long-Context Evidence

| Case | Row-serial cycles | Interleaved cycles | Reduction | Double buffered |
|---|---:|---:|---:|:---:|
| seq=4097, batch=1, M=512, B=64 | 543,315,254 | 468,208,823 | 13.82% | yes |
| seq=8192, batch=1, M=2048, B=64 | 653,521,843 | 565,794,419 | 13.42% | yes |

The `seq=8192` exact compressed compute-scoreboard evaluation completed in
296.82 s wall time with 1,036,864 KiB peak RSS after raising the explicit
schedule expansion limit from four to eight million instructions. It did not
fall back to serial work or an approximate II=1 calculation.

CostEmitter's compute pipeline deliberately removes HBM operations before
scoreboarding; V4 memory is combined later by the stage roofline. Therefore,
these production-shape cycle reductions prove compute scheduling efficiency,
while the emitted double-buffer ordering itself is validated by the
transactional test below. They are not evidence of cycle-exact production-size
DMA overlap.

## Transactional Validation

Two generated-ISA tests were run in both modes with the unchanged comparison
gate:

| Case | Gate | Output parity | Makespan |
|---|---|---|---:|
| seq=7, batch=4, M=16, B=4 | PASS in both modes | bitwise equal | smoke only |
| seq=39, batch=1, M=16, B=4 | PASS in both modes | bitwise equal | 60,214 -> 43,098 cycles (-28.42%) |

The second case has `seq_len > 2*MLEN`, exercises recurrent online softmax,
tail handling, nonresident K/V slots, and the emulator's asynchronous resource
scheduler. HBM reads remain 17,408 bytes in both modes. The output-region hashes
are recorded in `Workspace/gqa_pipeline_validation/transactional_ab_summary.json`.

The transactional RTL-validation status still reports unsupported or
out-of-domain opcodes inherited from the wider machine model. Consequently,
this is strong compiler/emulator parity evidence, but not a claim of complete
cycle-exact RTL validation at production shape.

## Integration

The option is propagated through native compilation, CostEmitter, persistent
trace and pipeline cache keys, the Qwen3-32B DSE CLI, and transactional GQA
testbench CLI. Trial records include the timing artifact hash, selected widths,
double-buffer status, eligible DMA occurrences, fallback reason, and GQA stall
breakdown. The validated default is `row-interleaved-v1` whenever RTL-v3 is
selected; `row-serial` remains explicit.

The formal DSE latency remains:

```text
sum(stage max(compute_pipeline_makespan, V4_memory_work))
```

The A/B benchmark now labels this separately from the conservative
`compute + memory` serial upper bound.

## Reproduction

```bash
nix develop --command bash -lc '
  source .venv/bin/activate
  python Workspace/qwen3_32b_dense_analytic/benchmark_gqa_pipeline_ab.py \
    --seq-len 482 --batch-size 16 \
    --output-dir Workspace/gqa_pipeline_validation/short_context
'
```

Machine-readable evidence is under `Workspace/gqa_pipeline_validation/`.

## Claim Boundaries

- Arithmetic order is preserved per row and tested bitwise at small shape.
- The 1 GHz conversion is an assumption, not post-synthesis timing closure.
- V4 remains a post-hoc per-DMA service model, not online Ramulator state.
- Stage roofline can miss overlap across compiler stage boundaries.
- The long-context scoreboard measures compute scheduling; exact production
  DMA overlap still requires full scheduled replay or transactional execution.
- This work changes compiler scheduling only; area and power are unchanged.
