# Ideal II=1 Compute Timing Rollout

## Decision

The default compute model is now `ideal-ii1`:

```text
Matrix opcodes        current RTL-v4 artifact's structural Matrix timing
Vector opcodes        1 cycle per dynamic instruction
Scalar opcodes        1 cycle per dynamic instruction
Control opcodes       1 cycle per dynamic instruction
HBM opcodes           excluded from compute; handled by Ramulator/V4
compute dependencies  disabled
```

This is an architectural ideal assumption chosen for alignment with the
historical PLENA pipelining model. It is not a cycle-exact claim about the
current RTL. The hazard-aware `rtl-v1` model and the historical `legacy` model
remain available as explicit A/B modes.

The current compiler uses compact batch/head layout, streamed softmax,
K-major broadcast QK reuse, RTL-v4 compact-statistics Vector/Scalar lowering,
affine-loop-v2 FFN, and loop-AGU-v1. Those transformations determine the
number of dynamic instructions. `ideal-ii1` changes only their timing cost.
The controlled numbers below predate several of those instruction-count
optimizations and remain a historical timing-policy A/B, not the current
end-to-end DSE result.

## Controlled Result

Configuration:

```text
Qwen3-32B dense, 64 decoder layers
seq_len=482, batch_size=16
MLEN=VLEN=2048, BLEN=1024, HLEN=128
CLOCK_PERIOD_PS=1000
HBM V4 production-DMA service model
```

Production CostEmitter result:

| Component | Cycles | Share |
|---|---:|---:|
| Matrix | 639,888,384 | 28.96% |
| Vector | 434,329,376 | 19.66% |
| Scalar | 1,050,974,721 | 47.57% |
| Control | 84,146,694 | 3.81% |
| **Compute total** | **2,209,339,175** | **100.00%** |

The current HBM V4 artifact gives:

```text
compute work              2.209339175 s
stage-wise V4 roofline    2.209349822 s
aggregate V4 memory work  0.070018931 s
```

The compute total exactly matches the independently generated one-cycle
counterfactual. The 0.000010647 s roofline increment comes from memory-bound
global load stages; large layer stages remain compute-bound.

## A/B Interpretation

| Model | Full decoder latency | Semantics |
|---|---:|---|
| Legacy closed form | 0.950 s | Hand-written equations with incomplete compiler work |
| **Ideal II=1** | **2.209 s** | Real compiler trace; Matrix structural; V/S/C one cycle |
| Ordered one-cycle sensitivity | 2.374 s | One-cycle V/S/C but legacy scoreboard ordering retained |
| rtl-v1/rtl-v3 conservative path | 7.517 s | Full-machine opcode latency and hazard-aware scoreboard |

The new default deliberately selects the second row. The 2.209 s result is
larger than the 0.950 s legacy estimate because the real compiler emits
1.569 billion non-HBM instructions, including RMSNorm, Q/K normalization,
online softmax, address generation, state movement, and packed-output work
that the closed-form model did not represent.

The 7.517 s result remains useful as a conservative current-RTL sensitivity.
It is no longer the formal DSE objective.

## Implementation

### Transactional emulator

- `--timing-mode {ideal-ii1,rtl-v1,legacy}` now defaults to `ideal-ii1`.
- A constant-space ideal accumulator runs after each functional opcode.
- Matrix cycles come from the current `rtl_opcode_timing_v4.json`; its Matrix
  entries preserve the structural timing inherited from v3.
- V/S/C cycles are one; observed Ramulator completion cycles are accumulated
  separately.
- The reported transactional latency remains the serial sum of ideal compute
  and observed memory for compatibility.
- Event records report `timing_provenance=architectural_ideal_ii1`,
  `dependency_model=disabled`, and zero stall cycles.
- Functional execution is unchanged, so changing timing mode cannot change
  numerical state.

### CostEmitter and DSE

- Public CostEmitter APIs and CLI now default to `ideal-ii1`.
- Stage compute is derived from the real dynamic opcode counts.
- Formal DSE latency remains:

  ```text
  sum(max(stage ideal compute, stage R-aware HBM V4))
  + inter-chip communication
  ```

- Scheduled replay and `--require-rtl-validated` are rejected in ideal mode.
- Study metadata includes the compute timing mode and a new objective schema,
  preventing accidental resume of an rtl-v1 study.

### Power

- Dynamic action energy and SRAM/HBM coefficients are unchanged.
- V/S/C ClockWork occupancy is one cycle per action in ideal mode.
- Matrix ClockWork retains structural timing; HBM-controller occupancy retains
  V4 service windows.
- Leakage and HBM background use the ideal stage-roofline makespan.
- Reports explicitly set:

  ```text
  compute_timing_status=architectural_ideal_assumption
  compute_hazards_included=false
  clock_gating_status=architectural_ideal_assumption
  ```

## Validation

- Rust `cargo check --all-targets`: pass.
- Rust focused timing tests: 16 pass, including ideal V/S/C, Matrix, and HBM
  accounting.
- Python CostEmitter, pipeline, parity, DSE, power, and emulator-contract
  tests: 117 pass across the focused suites.
- Production 2048/1024 CostEmitter evaluation: exact category and total-cycle
  match to the controlled reference.
- Three-trial Optuna DSE smoke: 3 complete, 0 failed. All trials report
  `compute_timing_mode=ideal-ii1`,
  `compute_timing_semantics=hazard_free_effective_opcode_cost`, complete V4,
  area, multi-chip, and system-energy outputs.
- Four-worker Optuna DSE smoke: 12 complete, 0 pruned, 0 failed. The Journal
  study produced all four objectives and every trial reported
  `hazards_modeled=false`, `rtl_cycle_validation_claim=false`, and power
  timing status `architectural_ideal_assumption`.

## Claim Boundary

Use the phrase:

> hazard-free effective one-cycle Vector/Scalar/control model with
> RTL-structural Matrix timing

Do not call this cycle-exact simulation, measured II=1 hardware, or timing
closure evidence. `rtl-v1` and its microbenchmarks remain the evidence for the
implemented RTL timing and hazard behavior.
