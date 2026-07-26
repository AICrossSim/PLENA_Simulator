# Legacy Analytic Model vs CostEmitter

> **Default-status update (2026-07-24):** the controlled 2.209 s
> Vector/Scalar/control one-cycle result is now implemented as the formal
> `ideal-ii1` DSE timing mode. The 7.517 s hazard-aware result is retained as a
> conservative A/B. See
> [`ideal_ii1_compute_timing_rollout.md`](ideal_ii1_compute_timing_rollout.md).
> The restored closed-form implementation now lives under
> `analytic_models.legacy` and is not imported by the formal DSE path.

## Purpose

This report compares the historical closed-form Qwen3 latency model with the
current compiler-derived CostEmitter model. It separates two effects:

1. Work emitted by the compiler but absent from the closed-form equations.
2. Vector/Scalar latency, dependencies, and resource hazards hidden by an
   ideal pipelined-latency assumption.

The controlled configuration is:

```text
Qwen3-32B dense, 64 decoder layers
sequence length = 482
batch size      = 16
MLEN/VLEN/BLEN  = 2048/2048/1024
HLEN            = 128
CLOCK_PERIOD_PS = 1000
```

The CostEmitter run uses compact layout, `direct-first-block-v1`,
Vector/Scalar `rtl-v3`, row-interleaved GQA, and HBM V4. The precision profile
comes from `m2048_b1024.toml`; the legacy model does not model
precision-dependent instruction timing.

## Results

| Model | Full decoder latency | Meaning |
|---|---:|---|
| Legacy Qwen3 closed form | **0.950 s** | Hand-written layer equations and historical pipelined constants |
| Real CostTrace, historical opcode constants | **2.489 s compute** | Real compiler counts, old per-opcode latency |
| Real CostTrace, V/S/control resource work = 1 cycle | **2.209 s compute work** | Current default `ideal-ii1`; Matrix timing unchanged |
| Real ordered CostTrace, V/S timing = 1 cycle | **2.374 s roofline** | Same schedule and scoreboard; Matrix/HBM unchanged |
| Explicit rtl-v1/rtl-v3 CostEmitter | **7.517 s roofline** | Conservative full-machine timing and compute scoreboard |
| Current serial resource work | **12.003 s** | Diagnostic work sum before legal overlap |

HBM V4 contributes only **69.6 ms** in this configuration. This comparison is
compute dominated.

The current result is **7.91x** the legacy result. Using the ordered one-cycle
counterfactual, the 6.567 s gap decomposes as:

```text
Current timing/dependency contribution:
    7.517 - 2.374 = 5.143 s  (78.3% of the gap)

Work absent from the legacy closed form:
    2.374 - 0.950 = 1.424 s  (21.7% of the gap)
```

This is a controlled counterfactual, not a statistical decomposition. It holds
the compiler trace and Matrix timing fixed and changes Vector/Scalar timing.

## Why The Old Model Is Optimistic

### Ideal pipeline constants

The historical `customISA_lib.json` assigns:

```text
V_BASIC = 1 cycle
S_BASIC = 1 cycle
S_EXP_FP/S_RECI_FP = 1 cycle
V_EXP_V = 2 cycles
V_RED_SUM/V_RED_MAX = 8/5 cycles
```

The closed-form model multiplies these values by algebraic loop counts. It does
not build an operand-dependency graph, model result-ready time, or apply SRAM
and functional-unit hazards. A `pipelined` value is consequently treated as
effective instruction cost even when a dependent chain cannot sustain that
throughput.

Current full-machine resource timing gives:

| Opcode | Dynamic count | Cycles/op | Resource work |
|---|---:|---:|---:|
| `V_RED_SUM_SEG` | 31.59 M | 75 | 2.369 B |
| `V_ADD_VV` | 118.80 M | 12 | 1.426 B |
| `V_MUL_VF` | 77.82 M | 11 | 0.856 B |
| `V_EXP_V` | 38.40 M | 21 | 0.806 B |
| `V_RED_MAX_SEG` | 31.59 M | 25 | 0.790 B |
| `S_ADD_FP` | 36.53 M | 10 | 0.365 B |
| `S_RSQRT_FP` | 36.53 M | 10 | 0.365 B |
| `S_RECI_FP` | 31.59 M | 9 | 0.284 B |

The weighted resource latency is **19.30 cycles/Vector opcode** and **2.75
cycles/Scalar opcode**. The Scalar average is reduced by 586.8 M one-cycle
integer address increments; dependent FP operations are commonly 8-10 cycles.

Collapsing all Vector/Scalar ready, done, resource, and II fields to one cycle
reduces the ordered roofline from **7.517 s to 2.374 s**. This A/B preserves
the same CostTrace, Matrix timing, HBM work, and scoreboard rules.

### Missing compiler work

The current full decoder trace contains:

```text
Matrix opcodes:       0.438 M
Vector opcodes:     434.329 M
Scalar opcodes:   1,050.975 M
Control opcodes:     84.147 M
```

One decoder layer contains 24.94 M dynamic instructions:

```text
Matrix:      6,848
Vector:  6,903,104
Scalar: 16,683,372
Control: 1,346,739
HBM:           864
```

The complete legacy layer is only 14.85 M cycles, including Matrix work. The
real Scalar instruction count alone is larger.

The historical equations summarize RMSNorm, Q/K norm, online softmax, address
generation, state loads/stores, lane movement, reduction, and output
accumulation using a few `V_BASIC`/`S_BASIC` terms. Native lowering emits these
operations explicitly. Therefore, even ideal one-cycle Vector/Scalar execution
leaves the ordered current trace **2.50x** slower than the closed form.

## Conclusion

The mismatch is mainly Vector/Scalar ready latency and dependency propagation,
but not exclusively:

```text
~78%: current Vector/Scalar timing and dependency/resource effects
~22%: compiler work absent from the old equations
```

Perfect one-cycle Vector/Scalar execution would improve this trace by about
**3.17x**, but would not recover the legacy 0.950 s prediction. It is therefore
incorrect to attribute the entire mismatch either to hazards or to instruction
count alone.

## Validation Limits

This is stronger than an equation-only comparison, but not cycle-exact
large-hardware validation:

- `2048/2048/1024` is outside the small RTL timing calibration domain.
- `M_MM` is structurally extrapolated.
- `M_MM_WO`, `M_BMM_WO`, and `M_BTMM` are marked unsupported by the timing
  policy.
- Most production-size Vector/Scalar operations are extrapolated from small
  full-machine measurements.
- 1 ns is an architecture assumption, not demonstrated timing closure.

These caveats affect the absolute 7.517 s value. They do not invalidate the
same-trace evidence that the old model omits work and assigns unrealistically
cheap timing to dependent Vector/Scalar execution.

## Restored Models From `c4d0907`

Commit `c4d0907` deleted seven analytic-model files. They were restored from
its parent revision for assessment.

### Memory

```text
analytic_models/memory/memory_model.py
analytic_models/memory/llm_memory_model.py
```

Remaining value:

- logical weight and KV-cache footprint;
- HBM capacity checks;
- logical traffic and bandwidth-utilization sanity checks;
- tensor-level precision sensitivity.

They cannot replace HBM V4: they do not construct production 64-byte DMA line
manifests, partial-line read-modify-write traffic, row-state service latency,
or per-occurrence completions.

### Roofline

```text
analytic_models/roofline/asm_profiler.py
analytic_models/roofline/decoder_roofline.py
```

`decoder_roofline.py` remains useful for arithmetic-intensity presentation.
`asm_profiler.py` is historical only: it hard-codes 64-wide hardware and
assigns simplistic costs such as `VLEN` to Vector operations and `MLEN` to
Matrix operations. It must not become the production objective.

### Utilisation

```text
analytic_models/utilisation/utilisation_model.py
```

This remains useful for tile padding, Matrix/Vector lane utilization, and
qualitative shape diagnostics. It does not observe current compiler layout,
SRAM residency, ordered hazards, or V4 DMA service, so it should be reported
beside CostEmitter rather than multiplied into its latency.

## Evidence

```text
Workspace/reports/model_latency/
  qwen3_32b_m2048_b1024_latency_comparison.json
  qwen3_32b_m2048_b1024_ideal_vs_scheduled.json

analytic_models/performance/compare_legacy_costemitter.py
```

The restored files match `c4d0907^` except for a package-relative import
compatibility fix in `llm_memory_model.py`.
