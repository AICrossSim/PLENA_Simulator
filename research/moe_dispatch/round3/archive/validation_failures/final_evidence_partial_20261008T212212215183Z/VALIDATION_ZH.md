# 第三轮最终验收

状态：**failed**。完整交付：False。

本验收检查已保存的逐窗口结果、资源约束、重复运行凭证及证明范围。性能是否跨过 5% 门槛单独报告；如实报告未胜出不等于实验验收失败。

| 检查 | 结果 |
|---|---|
| schema:E0/repro.csv | 通过 |
| schema:E2/bounds_by_window.csv | 通过 |
| schema:E2/bounds_summary.csv | 通过 |
| schema:E2/port_tight/bounds_by_window.csv | 通过 |
| schema:E2/port_tight/bounds_summary.csv | 通过 |
| schema:E3/ablation.csv | 通过 |
| schema:E4/dse_progress.csv | 通过 |
| schema:E4/union_generation_budget.csv | 通过 |
| schema:E4/proof_status.csv | 通过 |
| schema:E4/bootstrap_stability.csv | 通过 |
| schema:E4/heldout_main_table.csv | 通过 |
| schema:E4/heldout_main_table_pipelined.csv | 通过 |
| schema:E4/heldout_main_table_port_tight.csv | 通过 |
| schema:E4/breakdown.csv | 通过 |
| schema:E4/gates.csv | 通过 |
| schema:E4/selected_baseline_headroom.csv | 通过 |
| schema:E4/cross_bw.csv | 通过 |
| schema:E4/synthetic/reverse_search.csv | 通过 |
| schema:E5/dispatch/compare.csv | 通过 |
| schema:E5/dispatch/hbm_bytes.csv | 通过 |
| schema:E5/dispatch/regression.csv | 通过 |
| schema:E5/dispatch/development_per_window.csv | 通过 |
| schema:E5/predictor/predictor_table.csv | 通过 |
| schema:E5/sobol/sobol_indices.csv | 通过 |
| schema:E5/sobol/sobol_samples.csv | 通过 |
| schema:E5/sobol/flip_points.csv | 通过 |
| round2_readonly | 通过 |
| E0_exact_reproduction_and_LB | 通过 |
| E2_saved_bounds_and_reference_LB | 通过 |
| all_available_per_window_LB_and_repeat | 通过 |
| repeat_receipts_raw_hashes_and_raw_LB | 通过 |
| dispatch_development_fixed_and_EFT_reference_repeats | 通过 |
| all_available_search_witnesses_LB_and_certificate_scope | 通过 |
| bootstrap_and_campaign_counts | 通过 |
| published_latency_tables_from_window_values | 通过 |
| new_baseline_headroom_from_same_protocol_windows | 通过 |
| single_core_bitexact_regression | AssertionError: Duplicate single-core physical regression protocol |
| frozen_main_hardware_isoresource | 通过 |
| development_frozen_heldout_robust_diagnostics | 通过 |

实际逐窗口延迟与全局下界比较：1614924 条；下界违例／相关检查失败 0。
重复配置凭证 344 项，匹配摘要对 35090 对。

解析模型的片段流量和在途字节是模型观测／估计，不是 Ramulator 请求级实测；离线精确分配不能代表全部时间调度的全局最优。

尚未完成的文件：

- REPORT_ZH.md
- README.md
