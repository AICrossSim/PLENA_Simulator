# Unified Affine FFN Loop Lowering v2

## Scope

This change removes a compiler performance discontinuity in native Qwen FFN
lowering. It does not change RTL, ISA, Matrix timing, HBM V4, numerical
rounding, or the correctness gate.

The previous compiler selected different implementations from Matrix SRAM
capacity:

```text
streaming SRAM       -> structured K-split + live-stride-v1
projection-full SRAM -> legacy _ffn_asm_with_loops template
```

The legacy template expanded large affine pointer increments before loop
recovery. At large `MLEN`, one logical increment could become hundreds or
thousands of `S_ADDI_INT` instructions. More SRAM could therefore reduce HBM
work while increasing compute latency.

Increasing the six-stream AGU capacity is not the fix. Only one loop reached
six streams in the failing traces, and AGU v2 produced no measured benefit.

## Implementation

The new default is:

```text
ffn_projection_schedule = affine-loop-v2
ffn_address_schedule    = live-stride-v1
address_generation_mode = loop-agu-v1
```

`FfnProjectionPlan` is the shared compiler IR for up, gate, and down
projections. It represents:

```text
K chunk
  output MLEN block
    output BLEN tile
      activation column
        K accumulation tile
```

Matrix SRAM capacity changes only the existing K-chunk boundaries. It no
longer selects a different FFN template. The same plan produces native ASM and
CostEmitter instructions. The inner loops expose at most two affine streams
per axis to the existing AGU, within four hardware loop levels.

The operation order remains:

```text
prefetch -> M_MM accumulation -> M_MM_WO -> optional partial-sum V_ADD
```

The structural guard verifies identical K chunks and Matrix/HBM/partial-sum
census before selecting the new plan. A failed projection falls back to
`legacy-auto-v1` and records the reason. Explicit legacy address mode also
selects the complete legacy projection path, preserving historical tests.

The CostEmitter metadata reports the final post-AGU `layer/ffn` address census,
not the pre-rewrite estimate:

```text
ffn_loop_plan_version
ffn_explicit_loop_depth
ffn_agu_streams_by_axis
ffn_address_cycles_before/after
ffn_schedule_guard_status
ffn_schedule_fallback_reason
ffn_legacy_template_bypassed
```

The DSE latency/objective/search schemas were advanced to the affine-loop
version, so old studies cannot be resumed into the new timing semantics.

## Controlled Results

All measurements use Qwen3-32B, ideal-II1 compute, 64 decoder layers, and the
same precision, HBM V4 model, and SRAM configuration on both arms.

| Configuration | Legacy compute | Affine compute | Reduction | FFN `S_ADDI_INT` reduction |
|---|---:|---:|---:|---:|
| Short: `seq=482,batch=16,M2048/B1024,N1` | 1.0689B | 1.0613B | 0.71% | 67.93% |
| Long: `seq=32768,batch=1,M4096/B64,N4,7 tiles` | 75.8347B | 44.4190B | 41.43% | 99.76% |
| Long: `seq=32768,batch=1,M8192/B32,N4,4 tiles` | 149.9546B | 85.3449B | 43.09% | 99.52% |

For the representative `M4096/B64/N4` point:

```text
one-layer compute       1,185,723,375 ->   694,852,784 cycles
aggregate FFN stage    42,487,008,832 -> 11,071,291,008 cycles
aggregate attention    33,346,840,576 -> 33,346,840,576 cycles
N=4 compute share             18.959 s ->          11.105 s
```

The new N=4 compute share is below the previous 2-tile streaming reference
(`11.616 s` in the latest completed study and `13.437 s` in the original
audit), resolving the SRAM performance reversal at this point.

For `M8192/B32/N4`, the FFN stage falls from `83.942B` to `19.332B` cycles.
The attention stage remains `66.012B` cycles, so attention is now the real
large-MLEN limit rather than address legalization.

Machine-readable A/B results:

```text
Workspace/reports/compiler/unified_affine_ffn_loop_v2_short_m2048_b1024.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_m4096_b64.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_m8192_b32.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_smoke64.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_dse2048.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_a100_comparison.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_no_regression_audit.json
Workspace/reports/compiler/unified_affine_ffn_loop_v2_pareto.csv
```

They are reproducible with:

```text
analytic_models/performance/benchmark_affine_ffn_v2.py
```

## Correctness and Work Invariants

The tiny transactional FFN A/B produced:

```text
correctness gate       PASS 100% on both paths
active output bytes    bitwise identical
complete 128 MiB VRAM  bitwise identical
```

For all three controlled CostEmitter A/B points:

```text
M_MM count             identical
M_MM_WO count          identical
V_ADD partial sums     identical
H_PREFETCH_M count     identical
DMA occurrence count  identical
DMA manifest hash      identical
```

The focused compiler suites cover plan census, K-tile counts 1/2/4, AGU
addressing, CostEmitter/ASM parity, dense Qwen regression, packed GQA, and the
native compiler. An 18-case post-AGU domain matrix additionally spans
`M256..M8192`, flattened and square BLEN choices, and 1/2/4/7-tile projection
depths; every case preserves Matrix/HBM work and has no address/control
regression. The DSE tests cover schema isolation and CSV propagation. The
combined focused regression result is `127 passed`.

The long-context DSE smoke completed with:

```text
target COMPLETE trials      64
attempts                    64
COMPLETE / PRUNED / FAIL    64 / 0 / 0
requested / peak workers    32 / 32
peak worker RSS             1.05 GiB
minimum MemAvailable       92.11 GiB
affine guard fallbacks       0
post-AGU address regressions 0
compact output size         11.7 MB
```

Every trial records `affine-loop-v2`, loop depth 4 or less, a passed
structural guard, and post-AGU address work no greater than its compatibility
estimate.

## Complete Long-Context DSE

The smoke study was resumed to the complete long-context budget:

```text
target COMPLETE trials          2,048
attempts                        2,048
COMPLETE / PRUNED / FAIL        2,048 / 0 / 0
requested worker limit              64
maximum dynamic concurrency         58
peak worker RSS                  1.18 GiB
peak active process-tree RSS    34.07 GiB
minimum MemAvailable            76.52 GiB
compact output size            159.27 MB
```

All 2,048 trials used `affine-loop-v2`, passed the structural guard, and
avoided compatibility fallback. None had greater post-AGU FFN address work
than its compatibility estimate; the maximum observed ratio was `0.33642`.

Most importantly, exact adjacent Matrix-SRAM comparisons holding hardware,
precision, chip count, and INT width constant show:

```text
matched adjacent SRAM pairs            123
latency regressions >2%                   0
latency regressions >10%                  0
latency improvements >2%                108
maximum observed latency increase      0.0%
```

The earlier live-stride study had 59 regressions among 121 comparable pairs.
The complete new study therefore resolves the systematic projection-full
reversal, rather than only fixing the two controlled examples.

Key under-budget selectors from this run are:

| Selector | Trial | Latency | Area | Energy | Hardware |
|---|---:|---:|---:|---:|---|
| Fastest / best EDP | 982 | 2.880 s | 781.3 mm2 | 991.7 J | N16, M2048/B128, 2 tiles |
| Lowest energy | 1009 | 6.473 s | 388.3 mm2 | 766.2 J | N16, M1024/B64, 25 tiles |
| Fastest within 5% of A100 area | 543 | 4.640 s | 792.9 mm2 | 1,052.4 J | N16, M1024/B128, 25 tiles |
| Closest below 826 mm2 | 1316 | 11.546 s | 810.9 mm2 | 1,874.1 J | N8, M4096/B32, 7 tiles |

These selectors inherit the ideal-II1, optimistic TP+SP, ideal dual-port SRAM,
and large-shape extrapolation assumptions. They are DSE comparisons under the
current model, not measured silicon claims.

Generated plots:

```text
Workspace/reports/compiler/unified_affine_ffn_loop_v2_latency_vs_area.png
Workspace/reports/compiler/unified_affine_ffn_loop_v2_energy_vs_area.png
Workspace/reports/compiler/unified_affine_ffn_loop_v2_latency_vs_energy.png
```

## Claim Boundary

The performance result is exact under the current DSE's ideal-II1 timing
semantics. It is not a cycle-exact RTL claim. Matrix broadcast remains
RTL-unvalidated, HBM V4 remains a post-hoc service model, and `M4096/M8192`
remain structural extrapolation points.

The change does establish a compiler invariant: selecting more Matrix SRAM no
longer routes FFN through the known slower legacy template. Any future
projection that fails the structural/work guard retains the compatibility
lowering and reports the fallback.
