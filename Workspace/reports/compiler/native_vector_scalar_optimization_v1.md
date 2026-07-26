# Native Vector/Scalar Compiler Optimization v1

> **Superseded as the default execution path.** This report remains the
> compiler-only A/B baseline. The current RTL-backed schedule is documented in
> [`../rtl/vector_scalar_rtl_v3_segment_parallel.md`](../rtl/vector_scalar_rtl_v3_segment_parallel.md).
> The subsequent experimental compact-statistics RTL-v4 path is documented in
> [`compact_stats_selector_overwrite_v1.md`](compact_stats_selector_overwrite_v1.md).

**Report date:** 2026-07-19. This was the final compiler-only optimization
report before the RTL-v3/v4 architecture passes; all three reports remain
staged A/B evidence rather than current final latency claims.

## Scope

This change reduces VectorMachine and ScalarMachine instruction work emitted
for the native Qwen decoder. It does not change RTL, ISA encodings, numerical
comparison thresholds, `rtl-v1` opcode timing, HBM V4 coefficients, the area
model, or the packed-attention QK/PV algorithm.

The validated default is:

```python
vector_scalar_schedule="compiler-v1"
```

The previous lowering remains available as `legacy` for A/B regression. The
ASM compiler and CostEmitter consume the same structured normalization plans;
there is no independent analytical opcode formula for the optimized path.

## Optimizations

### Grouped segmented Q/K RMSNorm

The legacy packed-Q normalization rebuilt a full-width scratch vector for
every `(head lane, row)`. Compiler-v1 groups work by `(storage block, row)`,
shares the source copy across active lanes, hoists scalar constants, and skips
inactive physical rows.

An initial implementation tried to share the squared vector across every
lane. Stage-by-stage comparison found that this was not bitwise equivalent:
the current masked `V_RED_SUM` implementation carries unselected lanes, while
the legacy path recomputed scratch after each normalized lane. The final
implementation therefore emits one direct square refresh between lanes. This
preserves legacy numerical semantics while still replacing the repeated
zero/copy/square sequence with a single refresh.

### Active-row ordinary RMSNorm

Input, post-attention, final, Q and K normalization use the common sequence
packing plan's `active_row_ranges()`. Dummy rows are never normalized. Address
generation uses loop-carried increments, and independent address preparation
occupies existing reciprocal hazard slots where possible. Hidden-column
strides continue to use the complete physical row count.

### Redundant valid-column masks

For a partial causal tile, the explicit valid-column mask is omitted only when
the packed block-diagonal causal mask already proves that padding and
cross-batch columns are invalid. Non-causal and partial-past tiles retain the
mask. Reusable masks are program-global CostEmitter stages and are not scaled
by decoder-layer count.

## Numerical Evidence

### Full decoder stage checkpoints

The Qwen3-MoE tiny native test was run in both modes with stage checkpoint
capture enabled. All 18 observable decoder regions were byte-identical,
including input/attention RMSNorm, Q/K/V projection and RoPE, packed attention
output, residual paths, MoE router/combine, and final RMSNorm. Both runs passed
the unchanged comparison gate.

This checkpoint test was also the evidence used to reject the first
square-sharing implementation and add the between-lane refresh described
above. Comparing only the final output would not have localized that mismatch.

### Transactional packed-GQA tests

| Case | Coverage | Result |
|---|---|---|
| `batch=4, seq=7, MLEN=16, BLEN=4, HLEN=4` | packed batches, shared storage block, causal-mask elimination | PASS, 100% allclose, max abs error 0.003906 |
| `batch=2, seq=39, MLEN=16, BLEN=4` | multiple sequence/K tiles, retained partial-past valid mask | PASS, 100% allclose, max abs error 0.053711 |

The correctness gate and numerical reference were unchanged.

## CostEmitter Parity

Unit tests compare the dynamic opcode histogram generated from the structured
normalization plan with the CostEmitter schedule tree. The complete trial-402
A/B comparison also shows exact invariance for the work that this optimization
must not alter:

```text
Matrix opcodes:
  M_BMM_WO=256, M_BTMM=256, M_MM=3728, M_MM_WO=2608
HBM opcodes:
  H_PREFETCH_M=502, H_PREFETCH_V=234, H_STORE_V=128
Packed attention:
  QK=256, PV=256, KV tile loads=64
HBM bytes:
  read=1,491,861,504, write=272,629,760
```

These values are identical in `legacy` and `compiler-v1`.

At `MLEN=2048, BLEN=1024`, the largest dynamic opcode changes are:

| Opcode | Legacy | Compiler-v1 | Delta |
|---|---:|---:|---:|
| `S_ADDI_INT` | 12,529,434 | 7,537,969 | -4,991,465 |
| `S_LUI_INT` | 3,012,670 | 1,337,468 | -1,675,202 |
| `S_LD_FP` | 2,167,048 | 987,417 | -1,179,631 |
| `V_ADD_VV` | 2,838,656 | 1,847,808 | -990,848 |
| `V_MUL_VF` | 2,322,432 | 1,786,272 | -536,160 |
| `S_MAP_V_FP` | 2,048 | 0 | -2,048 |
| `C_SET_V_MASK_REG` | 584 | 555,776 | +555,192 |

The mask and loop-control increase is intentional: it is the control cost of
performing less full-vector arithmetic and preserving only active lanes.

## Resource-Work Results

All values below are one-layer `rtl-v1` calibrated resource-work cycles for
the same Qwen3-32B workload (`seq=482`, `batch=16`) and software precision
profile. The source data is in
`native_vector_scalar_optimization_v1.csv`.

| MLEN/BLEN | Total reduction | Vector reduction | Scalar reduction | Matrix delta | HBM-byte delta |
|---|---:|---:|---:|---:|---:|
| 512/512 | 6.43% | 6.19% | 11.06% | 0 | 0 |
| 1024/1024 | 9.44% | 8.91% | 14.13% | 0 | 0 |
| 2048/1024 | 11.73% | 10.17% | 19.37% | 0 | 0 |

For the target `2048/1024` point:

```text
Legacy total       294,530,913 cycles
Compiler-v1 total  259,968,410 cycles
Delta              -34,562,503 cycles (-11.73%)

Vector              212,772,864 -> 191,135,808 (-10.17%)
Scalar               69,678,521 ->  56,182,495 (-19.37%)
Matrix                9,998,256 ->   9,998,256 (unchanged)
Control               2,081,272 ->   2,651,851 (+27.41%)
```

The target acceptance thresholds (at least 8% Vector, 8% Scalar and 9% total
reduction) are met. The total saving is smaller at MLEN 512 because fewer
dummy rows and fewer packed head lanes exist for compiler-v1 to eliminate.

## Optimization Counters

The target point records:

```text
segmented norm constant loads elided: 1,179,630
segmented norm copy operations elided:   497,280
segmented norm square operations elided:  34,560
inactive normalization rows elided:       36,000
redundant valid masks elided:                 256
RMSNorm address loads elided:             98,286
RMSNorm NOPs elided:                      69,408
valid-mask builds:                             0
```

These counters describe emitted work avoided relative to the legacy lowering;
they are not hardware-cycle estimates by themselves.

## Test Status

The focused compiler suite passes:

```text
72 passed
```

It covers native layout, structured normalization plans, CostEmitter parity,
compiler regression, packed attention, and Qwen3-MoE lowering. A real DSE
smoke trial also completed with `vector_scalar_schedule=compiler-v1` and the
optimization counters present in `trial_record.json`.

After the Qwen3-MoE/reference integration was finalized, the complete
`PLENA_Compiler` suite also passed:

```text
98 passed, 1 PytestUnknownMarkWarning in 160.69 s
```

The warning is an unregistered test marker and does not indicate a failed
compiler or numerical test.

## Exhaustive DSE Result

The final Qwen3-32B sweep uses compact layout,
`direct-first-block-v1`, compiler-v1, `rtl-v1` compute timing, V4 one-layer
cached occurrence scaling, and the precision-aware area proxy. The A100
reference is 826 mm2 and the feasibility budget is 908.6 mm2 (110%).

Coverage was checked using the complete design key
`(precision_profile, MLEN, BLEN, INT_DATA_WIDTH)`:

```text
Expected unique grid points  13,905
Complete design points       12,051
Constraint-pruned points      1,854
Failed points                     0
```

The fastest feasible point is the same hardware and software configuration as
the pre-optimization sweep:

```text
Weight / ACT / KV / FP   MXFP_E2M1 / MXFP_E1M2 / MXFP_E4M3 / FP_E6M5
MLEN / VLEN / BLEN       2048 / 2048 / 1024
INT_DATA_WIDTH           16
Accuracy                 0.98
Estimated area           846.531 mm2

Previous latency         18,503.496 ms
Compiler-v1 latency      16,405.355 ms
Latency reduction         2,098.141 ms (11.34%)
```

The minimum latency is shared by 72 configurations. The profile above is the
highest-accuracy representative among those ties. Across all tied profiles,
nominal area ranges from 782.966 to 861.449 mm2.

For an arithmetic-family comparison, the fastest all-MXINT point uses
MXINT4 weight/ACT/KV and FP_E6M5 at the same `2048/1024` hardware. It predicts
16,477.759 ms and 646.277 mm2. The 72.404 ms latency gap is small relative to
the 136.689 mm2 area reduction, so the current model supports an MXINT density
advantage but not a large compute-latency advantage.

The identical selected configuration and identical modeled area isolate this
change to compiler-generated Vector/Scalar work. The area model, V4 memory
coefficients, precision profile, and hardware search point did not change.
The P90-conservative area comparison selects the same point.

The closest point to the 826 mm2 A100 reference is 834.651 mm2, with the same
2048/2048/1024 hardware dimensions and a modeled latency of 16,405.365 ms. It
uses `MXFP_E1M2 / MXFP_E5M2 / MXFP_E1M2 / FP_E6M5` and has accuracy 0.92.

### Reproducible Parallel Execution

A single shared SQLite study caused severe writer contention: 64 workers
processed only about 38 points/minute despite exact compiler-trace and V4 cache
hits. The exhaustive runner therefore uses 103 independent Optuna studies,
one per precision profile, and merges them only after each 135-point hardware
grid passes its local completeness check.

An isolated 135-point profile took 29.13 s and 289 MiB peak RSS. With 64
profile workers, all 103 shards completed in a 50.14 s execution span. The
merge then verifies all 13,905 unique design keys before publishing the normal
`trials.jsonl`, CSV, per-trial reports, A100 comparison, and run summary.

As an orchestration parity check, 1,888 completed designs from the interrupted
single-SQLite run were joined to the sharded result by the complete four-field
design key. `latency_ms`, `area_mm2`, and `accuracy_score` had zero mismatches
at an absolute tolerance of `1e-12`.

Artifacts:

```text
Workspace/qwen3_32b_dense_analytic/runs/
  roofline_v4_grid_vector_scalar_v1_sharded_20260719/
    run_summary.json
    a100_comparison.json
    all_trials.csv
    pareto_trials.csv
    qwen3_32b_latency_area_vector_scalar_v1.png
```

The runner is `run_sharded_bruteforce_dse.py`. It does not replace the DSE
objective or evaluate a simplified model; every shard invokes
`run_optuna_dse.py` with the same formal settings. Sharding only removes
cross-profile database contention.

## Claim Boundary

The reported improvements are reductions in calibrated serial resource work,
which is the practical Stage-1 DSE objective. They are not cycle-exact
scheduled makespan or post-layout performance. HBM V4 remains a post-hoc DMA
service model, and the 1 ns period is a reporting assumption rather than a
timing-closure result.

At the selected production-size point, the timing artifact classifies 17.50%
of resource cycles as directly full-Machine measured, 79.90% as structural
extrapolation and 2.60% as unsupported-RTL fallback. Those fractions are a
fidelity disclosure: they do not invalidate the compiler A/B opcode reduction,
but they limit absolute latency claims.
