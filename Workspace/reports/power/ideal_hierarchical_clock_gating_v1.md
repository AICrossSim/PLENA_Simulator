# Ideal Hierarchical Clock-Gating Power Model v1

## Purpose

The RTL-activity power candidate originally charged every mapped logic block
for every cycle of the full-model roofline makespan. This is a valid pre-CTS
ungated upper bound, but it is not representative of a design with hierarchical
clock gating. For a large `MLEN=VLEN=2048, BLEN=1024` point, that assumption
alone produced about 1.11 kW of clock power.

This change keeps the old result and adds an explicitly ideal lower-bound
interpretation. It does not modify RTL and must not be reported as measured
clock-gating effectiveness.

## Model

CostEmitter now emits compressed `ClockWork`:

```text
stage, component, subcomponent
equivalent_full_area_cycles
component_active_cycles
source_opcode
active_instances / total_instances
fidelity
```

The selected clock energy is:

```text
E_clock =
    clock_density(component)
  * sum(subcomponent_area * equivalent_active_cycles)
  + fixed_clock_energy(component) * component_active_cycles
```

Each subcomponent's equivalent cycles are capped at the full makespan. The
mapping is structural:

- Matrix: array stack and I/O, cross-K reduction and accumulator, output
  conversion, result buffer, and active control.
- Vector: lane datapath, reduction tree, buffers, segment-parallel delta, and
  active control, scaled by exact active lanes/tree nodes.
- Scalar: FP, INT, lane access, rtl-v3 pipeline delta, and active control.
- HBM controller: matrix/vector/scale/address/control paths over V4 DMA service
  windows.
- SRAM: unchanged Liberty read/write action energy, with no continuous clock
  baseline.
- Ideal dual-port SRAM area does not divide access energy: each read/write is
  still charged once, and two simultaneous accesses are two actions. Physical
  dual-port overhead and SRAM leakage remain unmodelled.
- Full-chip top residual: leakage is retained; ideal clock energy is zero
  because this residual is not a calibrated clocked hierarchy.

The ideal assumptions are:

```text
idle clock energy   = 0
gate overhead       = 0
wake-up latency     = 0
inactive lanes      = perfectly gated
inactive slices     = perfectly gated
```

`estimate_onchip_power()` defaults to `ungated` for API compatibility.
`estimate_system_power()` and DSE default to `ideal_hierarchical`. Both modes
report the ungated upper bound.

## Fail-Closed Evidence

Vector mask activity is recovered by constant propagation over the compressed
schedule, including loop bodies. Full-width, single-segment, and exact
segment-mask activity are distinguished. A masked operation with an unknown
mask is not treated as full-width or one lane: ideal estimation fails with
`clock_work_unavailable`.

HBM controller occupancy requires a positive V4 service window for every DMA
opcode. A missing window also fails closed. The three production smokes below
reported:

```text
ClockWork status     = complete
unavailable records  = 0
action coverage      = 100%
```

The current shared `rtl_opcode_timing_v4.json` is the sole logic timing
source; it preserves the structural Matrix timing inherited from v3 and adds
the RTL-v4 Vector/Scalar variants.
The model does not introduce a second set of cycle constants.

## Regression Results

All values use Qwen3-32B prefill, `seq_len=482`, `batch=16`, 64 decoder layers,
1 GHz reporting, the rtl-v3 compute pipeline, V4 memory, current area model,
and the HBM3E-equivalent external-memory shadow.

| Point | Precision | Latency (s) | Area (mm2) | Ideal on-chip (W) | Ungated on-chip (W) | External HBM (W) | Ideal system (W) | Ungated system (W) | Clock saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `512/512/64` | MXINT4 | 34.758 | 6.940 | 3.849 | 9.850 | 6.072 | 9.921 | 15.922 | 62.22% |
| `1024/1024/1024` | MXFP E4M3 | 9.132 | 383.262 | 37.435 | 471.770 | 6.406 | 43.841 | 478.176 | 96.39% |
| `2048/2048/1024` | MXINT8-weight profile | 7.590 | 707.468 | 57.104 | 1109.502 | 6.474 | 63.578 | 1115.977 | 95.01% |

The large-point ungated clock regression exactly reproduces the motivating
breakdown to rounding:

| Component | Ungated clock (W) | Ideal clock (W) | Active fraction |
|---|---:|---:|---:|
| Matrix | 1005.245 | 54.725 | 5.44% |
| Vector | 1.595 | 0.526 | 32.88% |
| Scalar | 0.005 | 0.001 | 9.96% |
| HBM controller | 2.637 | 0.035 | 1.31% |
| Top/control residual | 98.203 | 0.000 | 0.00% |
| **Total clock** | **1107.685** | **55.287** | |

The difference is therefore not an unexplained fitted coefficient. It is
mostly the removal of full-makespan clock charging from the very large Matrix
array, followed by excluding unmodeled top residual from the ideal clock path.
Logic leakage is unchanged.

## Validation

The focused power suite contains 42 tests and covers:

- exact compressed mask propagation;
- idle ideal clock equal to zero;
- `ideal <= ungated` for every component;
- subcomponent area-cycle caps;
- top residual leakage/clock treatment;
- clock energy excluded from action-activity uncertainty;
- fail-closed unknown masks and missing HBM service windows;
- action-count monotonicity;
- external-HBM physical-byte accounting and system-energy summation.

CostEmitter frontend regression tests additionally verify that action fidelity
survives decoder-layer scaling. Three real DSE smokes cover MXINT4, MXINT8, and
MXFP plus `512/64`, `1024/1024`, and `2048/1024`. All completed with unchanged
latency/area/accuracy objectives and complete ClockWork.

On the 64-layer `512/64` trace:

```text
ClockWork records              311
ClockWork median runtime       6.83 ms
Full ideal on-chip median      19.63 ms
```

This meets the under-10-ms requirement for the incremental clock-work
calculation and does not expand the dynamic instruction stream.

The compact machine-readable evidence is preserved in
[`ideal_hierarchical_clock_gating_v1_validation.json`](ideal_hierarchical_clock_gating_v1_validation.json).

## Claim Boundary

The ideal result is useful for DSE sensitivity and for showing how much of the
ungated estimate is architectural clock opportunity. It is not a prediction
of the current RTL's post-layout power.

The model excludes ICG area/power, enable logic, clock-tree synthesis, skew,
routed parasitics, wake-up delay, package, board conversion, KV-link energy,
SRAM leakage, and thermal limits. HBM3E coefficients remain literature
parameters. The nominal system-energy estimate now participates in the formal
four-objective DSE, but it remains an architecture-level estimate rather than
a hard power constraint or evidence that clock gating exists in the current
RTL.
