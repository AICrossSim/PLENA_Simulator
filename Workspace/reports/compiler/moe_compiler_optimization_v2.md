# Qwen3-MoE Existing-Hardware Compiler Optimization v2

## Scope

This work optimizes the Qwen3-MoE router, route-weight, dispatch, expert,
and combine lowering using the existing RTL-v4 ISA. It does not add RTL,
opcodes, timing coefficients, or a new correctness threshold.

The optimized default is:

```text
moe_lowering_schedule = compact-route-v2
```

`legacy-static-v1` remains available for controlled A/B comparisons.
Runtime arg-top-k is still host-selected and excluded from latency.

## Implementation

### Compact router and route weights

- The router uses the single-block softmax path over active physical rows.
- SUM/MAX reductions use overwrite mode, removing explicit accumulator setup.
- Padded expert columns are masked without a replicated mask tensor.
- Selected probabilities are read directly with `S_LD_VLANE_FP`.
- Up to eight route probabilities stay in rotating FP registers for
  normalization.
- Normalized weights are written to the first top-k lanes with
  `S_ST_VLANE_FP`.
- The old `MLEN x MLEN` identity tensor, full-vector multiply/reduction,
  full-row reset, and `S_MAP_V_FP` path are removed.
- `top_k > 8` explicitly falls back to `legacy-static-v1`.

### Dispatch, experts, and combine

- `MoeExpertRoutePlan` builds token, expert, rank, and affine-run indices once.
- Fixed-balanced lowering uses an algebraic histogram and materializes zero
  `StaticRoute` objects.
- A route weight is loaded once per route and reused across all hidden blocks.
- Experts with the same padded bucket shape share one FFN CostTrace template.
- Dispatch uses one template; combine uses eight rank-specific templates.
- Expert FFN continues to use `affine-loop-v2`, `live-stride-v1`, and AGU-v1.

### CostEmitter and HBM V4

- ASM and CostEmitter consume the same lowering helpers.
- Expert FFN compute, EnergyAction, ClockWork, and DMA are replayed from an
  exact template. Weight DMA remains represented by at most 128 lightweight
  expert descriptors.
- V4 groups those descriptors by exact physical feature signature.
- Exact row-conflict accounting now uses a compact row-bank census for small
  row spans and retains stable-sort fallback for large spans.
- No V4 coefficient or prediction equation changed.

## Target Result

Configuration:

```text
Qwen3-235B-A22B, one decoder layer
seq_len=482, batch_size=16
MLEN=VLEN=512, BLEN=64, HLEN=128
MXFP E4M3 block-8, 128 HBM channels
fixed-balanced routing: 61,696 routes, 128 active experts
ideal-II1 compute + HBM V4 stage roofline
```

| Metric | Legacy | Compact v2 | Change |
|---|---:|---:|---:|
| Compute work | 120.673 M cycles | 111.253 M cycles | -7.81% |
| Router + dispatch + combine | 15.518 M | 6.098 M | -60.70% |
| Router | 6.528 M | 1.172 M | -82.04% |
| Dispatch | 2.009 M | 2.009 M | unchanged |
| Combine | 6.981 M | 2.917 M | -58.21% |
| Expert FFN | 58.257 M | 58.257 M | unchanged |
| HBM read traffic | 4,620.386 MB | 4,615.373 MB | -5.014 MB |
| Roofline latency | 120.983 ms | 111.577 ms | -7.77% |

The largest removed compute-work categories are:

```text
S_ST_FP        -4.165 M
S_ADDI_INT     -1.920 M
C_LOOP_START   -0.740 M
C_LOOP_END     -0.740 M
S_LUI_INT      -0.671 M
S_LD_FP        -0.648 M
```

New direct-access work is limited mainly to:

```text
S_LD_VLANE_FP  +123,392
S_ST_VLANE_FP   +61,696
```

Matrix opcode work, attention QK/PV, and expert FFN work are unchanged.

## CostEmitter Performance

For the target point:

```text
cold CostEmitter + V4 wall time: 9.13 s
peak RSS:                         0.60 GiB
warm median over 10 runs:       15.34 ms
warm P95:                       16.04 ms
materialized route objects:         0
unique V4 feature signatures:      43
logical DMA occurrences:        23,625
occurrences elided by grouping: 20,047
```

The original 42.6 s baseline included repeated expert lowering and the older
V4 cold path. Current legacy mode also benefits from the shared-template and
V4 infrastructure, so wall-time comparisons between current legacy and
compact mode are not a measure of the compiler optimization itself.

## Validation Evidence

### Transactional correctness

The tiny static-index Qwen3-MoE end-to-end run passed the unchanged gate:

```text
match rate:          100.00%
allclose match rate: 100.00%
relative match rate:  90.53%
MSE:                0.000839
MAE:                0.021484
maximum abs error:  0.140625
gate:               atol=0.2, rtol=0.2
```

This test exposed and fixed a real integration defect: compact router softmax
wrote `l` in the streamed-v2 state layout while final normalization initially
read the legacy `3*MLEN` address. The corrected path uses the same streamed
address for production and consumption.

### Parity and regression

```text
MoE compiler tests:       15 passed
Dense CostEmitter + AGU:  39 passed
HBM V4 tests:             15 passed
Power model tests:        53 passed
```

- Fixed-balanced and explicit round-robin traces match in stage opcodes and
  DMA totals on the tested shape.
- Matrix, Vector, Scalar, and HBM opcode counts match native ASM.
- DMA opcode count equals MemoryEvent multiplicity.
- V4 grouped and scalar paths preserve traffic and the existing `1e-9 ns`
  latency acceptance.
- Tests cover top-k 1, 2, and 8 plus top-k 9 fallback.
- Dense Qwen and AGU regressions remain unchanged.

The native textual AGU pass can optimize 11 small affine-width loops that the
compressed CostTrace conservatively leaves unmodified because rendered
instruction width varies by iteration. Non-address opcodes and DMA are exact;
the CostEmitter address/control count is a small conservative overestimate.
Legacy address-generation mode gives exact full-histogram parity. This is an
existing AGU representation limitation, not a MoE arithmetic or memory-model
mismatch.

## Fidelity and Limits

- Fixed-balanced routing is a latency-only routing assumption.
- Static-index transactional execution remains input-specialized.
- Runtime arg-top-k is not represented by the current RTL/ISA and is excluded.
- Compact routing supports top-k up to eight; larger values use explicit
  legacy fallback.
- Ideal-II1 remains an architectural timing assumption, not cycle-exact RTL.
- HBM V4 remains a post-hoc service model, although its grouped evaluation is
  algebraically exact with respect to the current cold-geometry semantics.
- Area is unchanged. Power uses the new action counts but no coefficient was
  recalibrated.

Machine-readable results and the full opcode delta are in:

```text
moe_compiler_optimization_v2_results.json
moe_compiler_optimization_v2_opcode_delta.csv
```
