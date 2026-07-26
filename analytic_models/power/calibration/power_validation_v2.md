# RTL-Activity Power Model v2 Validation

**Promotion status:** PASS

Qwen-like microkernel slopes define nominal P50. Low-toggle and random slopes define an empirical activity envelope; they are not training samples for the nominal coefficients.
Pre-CTS DC clock-network variation is retained for audit but excluded from action slopes. Validation uses matched-idle total plus the measured non-clock active residual.

| Metric | Result |
|---|---:|
| Qwen component holdout median APE | 7.473% |
| Qwen component holdout P95 APE | 20.885% |
| Idle clock holdout median APE | 3.828% |
| Idle clock holdout P95 APE | 8.340% |
| Minimum action slope R² | 0.999526 |
| Cached power evaluation median | 3.211 ms |
| Non-positive non-clock residual rows | 0 |

## Acceptance Gates

- `each_component_qwen_holdout_median_le_15pct`: PASS
- `each_component_qwen_holdout_p95_le_30pct`: PASS
- `idle_clock_holdout_median_le_15pct`: PASS
- `idle_clock_holdout_p95_le_25pct`: PASS
- `all_action_family_slope_r2_ge_0p95`: PASS
- `no_missing_action_family`: PASS
- `all_active_nonclock_residuals_positive`: PASS
- `no_negative_or_nonfinite_coefficient`: PASS
- `qwen_mix_uses_costtrace_counts`: PASS
- `qwen_mix_total_error_le_20pct`: PASS
- `matrix_structural_invariants`: PASS
- `cached_power_evaluation_lt_10ms`: PASS

The result is RTL VCD activity replayed on mapped DC netlists. It does not include gate-level timing activity, CTS, routed parasitics, external HBM/PHY/package power, KV links, or SRAM leakage.
