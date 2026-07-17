# Transactional Emulator `rtl-v1` Latency Validation

## Scope

This report validates timing only. Numerical golden data, tolerances, and PASS criteria are unchanged.
Cycles are primary; nanoseconds use `CLOCK_PERIOD_PS=1000` (1 GHz assumption), not a timing-closed fmax.
Ramulator service is currently placed on a post-hoc scheduler timeline rather than online cycle-coupled co-simulation.

## Full-Machine RTL Microbenchmarks

| Unit | Metrics | Max abs. error (cycles) | Holdout metrics | Holdout max error | <=1 cycle |
|---|---:|---:|---:|---:|:---:|
| matrix_machine | 24 | 0.000 | 0 | 0.000 | PASS |
| scalar_machine | 70 | 0.000 | 14 | 0.000 | PASS |
| vector_machine | 115 | 0.000 | 27 | 0.000 | PASS |

### Matrix MLEN Independence Check

| BLEN | MLEN -> measured M_MM cycles |
|---:|---|
| 4 | 16->8, 32->8, 64->8 |
| 8 | 32->12, 64->12 |
| 16 | 64->20 |

`M_MM_WO` is not claimed cycle-exact for consumer readiness: current RTL emits one write-valid pulse for a BLEN-row result.
The emulator therefore waits until backend idle for every output row.

## Hazard Scheduler Evidence

- RTL pipeline-control recovery checks: PASS
- Scheduler differential cases: 6/6
- Scheduler invariant checks: PASS
- Full-VectorMachine mixed-latency ordering: PASS

- `HAZARD_HBM_V_VECTOR`: raw stall=3 cycles, recovery=1 cycle
- `HAZARD_REDUCTION_SCALAR`: raw stall=5 cycles, recovery=1 cycle
- `HAZARD_SFU_BROADCAST`: raw stall=4 cycles, recovery=1 cycle
- `HAZARD_VECTOR_PORT_WRITE`: raw stall=2 cycles, recovery=1 cycle

## Qwen3-32B One-Layer System Result

| Mode | Cycles | Time at 1 GHz |
|---|---:|---:|
| Legacy functional executor | 71503360.0 | 71503360.0 ns |
| Previous rtl-v1 | 21893358.0 | 21893358.0 ns |
| Fixed rtl-v1 | 25452002.0 | 25452002.0 ns |

Functional comparison metrics unchanged: `True`.
Decoded BF16 output bitwise identical: `True`.
Fixed vs previous rtl-v1: `3558644.0` cycles (`16.254%`).
Fixed vs legacy-equivalent serial executor: `-64.404%`.
RTL validation status: `unsupported_opcodes`.
Validated opcode coverage: `666300/3473181`; unsupported=`76032`, out-of-domain=`2730849`.
Unsupported tail sensitivity: `0` cycles (local tail removal only; not a rescheduled counterfactual).
Latency delta reconciled by critical-path attribution: `True` (attributed sum=`3558644` cycles).

### Largest Opcode Resource-Work Deltas

| Opcode | Count | Total delta (cycles) | Delta/op (cycles) |
|---|---:|---:|---:|
| V_MUL_VF | 157184 | 628736 | 4.000 |
| V_ADD_VV | 176160 | 528480 | 3.000 |
| M_MM | 129280 | 517120 | 4.000 |
| V_MUL_VV | 90176 | 270528 | 3.000 |
| V_SUB_VF | 56448 | 225792 | 4.000 |
| S_ADD_FP | 95619 | 191238 | 2.000 |
| V_EXP_V | 56448 | 169344 | 3.000 |
| V_RED_SUM | 46208 | 138624 | 3.000 |
| V_ADD_VF | 25600 | 102400 | 4.000 |
| V_RED_MAX | 30848 | 92544 | 3.000 |
| V_RECI_V | 25600 | 76800 | 3.000 |
| S_MUL_FP | 32384 | 64768 | 2.000 |

### Resource Work and Critical Path

Resource work can overlap and therefore need not sum to makespan. Critical-path contributions are mutually exclusive.

| Resource | Fixed work | Previous critical path | Fixed critical path | Delta |
|---|---:|---:|---:|---:|
| control_frontend | 440364 | 3473181 | 3473181 | 0 |
| hbm_matrix_dma | 956893 | 893407 | 894480 | 1073 |
| hbm_vector_dma | 11288 | 10242 | 10426 | 184 |
| hbm_vector_store | 3992 | 3296 | 3424 | 128 |
| matrix_compute | 8795392 | 6672668 | 7247898 | 575230 |
| matrix_writeout | 2012352 | 2708974 | 2778984 | 70010 |
| scalar_pipeline | 4982815 | 2047622 | 2653704 | 606082 |
| vector_pipeline | 11545984 | 6083968 | 8389905 | 2305937 |

### Largest Stall Deltas

| Stall reason | Delta cycles |
|---|---:|
| vector_sram_port_a_write | 3344736 |
| vector_sram_operand_not_ready | -2204096 |
| pipeline_recovery | 785033 |
| vector_mixed_latency_in_order | 503145 |
| matrix_mcu_active | 460184 |
| scalar_fp_compute_in_progress | 320768 |
| vector_reduction_result_not_ready | 189696 |
| scalar_fp_operand_not_ready | 61696 |
| matrix_writeout_active | 55752 |
| vector_pipeline_busy | 41472 |
| scalar_fp_sram_busy | 1024 |
| matrix_sram_operand_not_ready | -632 |

### Unsupported RTL Coverage

| Opcode | Count |
|---|---:|
| S_MAX_FP | 30848 |
| V_SHIFT_V | 30848 |
| M_MM_WO | 14208 |
| M_BMM_WO | 64 |
| M_BTMM | 64 |

Unsupported resource work: `matrix_compute=4352`, `matrix_writeout=2012352`, `scalar_pipeline=308480`, `vector_pipeline=30848`

## Claim Boundary

Strong claims apply to tested full-Machine compute timings and the normalized pipeline-control hazards above.
Runs containing `S_MAX_FP`, unsupported Matrix broadcast/writeout opcodes, or out-of-domain production shapes are labeled accordingly and are not cycle-exact RTL validation.
Memory-bound accuracy remains limited by post-hoc Ramulator arrival times; online memory/compute co-simulation is deferred.
