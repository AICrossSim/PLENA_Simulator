# 冻结 GPQA T128 Shared 因果实例

窗口：`v3_captured_mixed_heldout_gpqa_t128_l13`。

本实例的目标是延迟与 HBM 都在同硬件 MILP/LPT 的 2% 以内（延迟比≤1.02）；全留出集几何均值的 1.01 调度差距门槛另列。

| 设计 | 模式 | 调度 | Shared 核 | Shared 开始 ms | HBM MiB | 原生唯一权重 MiB | 整层 ms | Shared 重读倍数 | Shared Z 分块 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | pipelined | eft_old | 0 | 21.1525 | 3267.00 | 1023.00 | 27.1931 | 22.0000 | 2 |
| B0 | pipelined | fixed | 0 | 21.1525 | 3267.00 | 1023.00 | 27.1931 | 22.0000 | 2 |
| B0 | pipelined | milp | 0 | 0.0057 | 3267.00 | 1023.00 | 27.1932 | 22.0000 | 2 |
| B1 | pipelined | eft_old | 0 | 8.2480 | 1056.00 | 1023.00 | 8.9866 | 2.0000 | 2 |
| B1 | pipelined | fixed | 0 | 8.2480 | 1056.00 | 1023.00 | 8.9866 | 2.0000 | 2 |
| B1 | pipelined | milp | 0 | 0.0057 | 1056.00 | 1023.00 | 8.9864 | 2.0000 | 2 |
| B2 | pipelined | eft_old | 0 | 14.6521 | 3861.00 | 1023.00 | 32.2243 | 64.0000 | 4 |
| B2 | pipelined | fixed | 1 | 0.0060 | 1204.50 | 1023.00 | 10.6717 | 4.0000 | 4 |
| B2 | pipelined | milp | 1 | 0.0058 | 1122.00 | 1023.00 | 9.9266 | 4.0000 | 4 |
| best_hetero | pipelined | eft_old | 0 | 10.1733 | 2095.50 | 1023.00 | 17.4451 | 26.0000 | 26 |
| best_hetero | pipelined | fixed | 1 | 0.0060 | 1188.00 | 1023.00 | 9.8912 | 2.0000 | 2 |
| best_hetero | pipelined | milp | 1 | 0.0058 | 1056.00 | 1023.00 | 8.8446 | 2.0000 | 2 |
| fixed_4+2 | pipelined | eft_old | 1 | 36.7805 | 6715.50 | 1023.00 | 57.5906 | 67.0000 | 6 |
| fixed_4+2 | pipelined | fixed | 0 | 0.0060 | 5973.00 | 1023.00 | 49.7017 | 33.0000 | 3 |
| fixed_4+2 | pipelined | milp | 0 | 0.0058 | 4620.00 | 1023.00 | 38.4504 | 33.0000 | 3 |
| B0 | port_tight | eft_old | 0 | 60.5345 | 3267.00 | 1023.00 | 77.8272 | 22.0000 | 2 |
| B0 | port_tight | fixed | 0 | 60.5345 | 3267.00 | 1023.00 | 77.8272 | 22.0000 | 2 |
| B0 | port_tight | milp | 0 | 0.0057 | 3267.00 | 1023.00 | 77.8270 | 22.0000 | 2 |
| B1 | port_tight | eft_old | 0 | 23.1234 | 1056.00 | 1023.00 | 24.6645 | 2.0000 | 2 |
| B1 | port_tight | fixed | 0 | 23.1234 | 1056.00 | 1023.00 | 24.6645 | 2.0000 | 2 |
| B1 | port_tight | milp | 0 | 0.0057 | 1056.00 | 1023.00 | 24.6646 | 2.0000 | 2 |
| B2 | port_tight | eft_old | 1 | 23.1219 | 1122.00 | 1023.00 | 29.2860 | 4.0000 | 4 |
| B2 | port_tight | fixed | 0 | 0.0060 | 1122.00 | 1023.00 | 26.2039 | 4.0000 | 4 |
| B2 | port_tight | milp | 0 | 0.0058 | 1122.00 | 1023.00 | 26.2042 | 4.0000 | 4 |
| best_hetero | port_tight | eft_old | 0 | 26.9755 | 2029.50 | 1023.00 | 67.0427 | 26.0000 | 26 |
| best_hetero | port_tight | fixed | 1 | 0.0060 | 1237.50 | 1023.00 | 30.0564 | 2.0000 | 2 |
| best_hetero | port_tight | milp | 1 | 0.0058 | 1105.50 | 1023.00 | 26.2037 | 2.0000 | 2 |
| fixed_4+2 | port_tight | eft_old | 1 | 99.4742 | 6402.00 | 1023.00 | 252.6778 | 67.0000 | 6 |
| fixed_4+2 | port_tight | fixed | 0 | 0.0060 | 5494.50 | 1023.00 | 131.4873 | 33.0000 | 3 |
| fixed_4+2 | port_tight | milp | 0 | 0.0058 | 5346.00 | 1023.00 | 127.5694 | 33.0000 | 3 |

pipelined/B0：fixed/MILP 延迟比=0.999997；HBM 超额=+0.000%。2% 延迟门槛（≤1.02）通过；2% HBM 门槛通过。

pipelined/B1：fixed/MILP 延迟比=1.000024；HBM 超额=+0.000%。2% 延迟门槛（≤1.02）通过；2% HBM 门槛通过。

pipelined/B2：fixed/MILP 延迟比=1.075057；HBM 超额=+7.353%。2% 延迟门槛（≤1.02）未通过；2% HBM 门槛未通过。
仍有重读的任务原因：all_cores_refetch: 1；immediate_refetch_faster_than_wait: 1。
Shared 专家 -1：绑定核 1，重读倍数 4.000；所有已安装核的最小倍数=4.000，对应核 1；决策 all_cores_refetch。实际流量 132.000 MiB，已安装核最小流量 132.000 MiB。
以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。两者相同的强制重读贡献为零，未列入差值表。

| 专家 | Shared | Me | Fixed 核 | MILP 核 | Fixed r | MILP r | 差值 MiB | Fixed 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | False | 12 | 0 | 1 | 6.000 | 1.000 | 82.500 | immediate_refetch_faster_than_wait |

pipelined/best_hetero：fixed/MILP 延迟比=1.118337；HBM 超额=+12.500%。2% 延迟门槛（≤1.02）未通过；2% HBM 门槛未通过。
仍有重读的任务原因：all_cores_refetch: 1；immediate_refetch_faster_than_wait: 8。
Shared 专家 -1：绑定核 1，重读倍数 2.000；所有已安装核的最小倍数=2.000，对应核 1；决策 all_cores_refetch。实际流量 66.000 MiB，已安装核最小流量 66.000 MiB。
以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。两者相同的强制重读贡献为零，未列入差值表。

| 专家 | Shared | Me | Fixed 核 | MILP 核 | Fixed r | MILP r | 差值 MiB | Fixed 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | False | 17 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 15 | False | 19 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 23 | False | 12 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 29 | False | 17 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 32 | False | 15 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 41 | False | 17 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 44 | False | 16 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 55 | False | 15 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |

pipelined/fixed_4+2：fixed/MILP 延迟比=1.292621；HBM 超额=+29.286%。2% 延迟门槛（≤1.02）未通过；2% HBM 门槛未通过。
仍有重读的任务原因：all_cores_refetch: 52。
Shared 专家 -1：绑定核 0，重读倍数 33.000；所有已安装核的最小倍数=33.000，对应核 0；决策 all_cores_refetch。实际流量 1089.000 MiB，已安装核最小流量 1089.000 MiB。
以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。两者相同的强制重读贡献为零，未列入差值表。

| 专家 | Shared | Me | Fixed 核 | MILP 核 | Fixed r | MILP r | 差值 MiB | Fixed 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | False | 18 | 1 | 0 | 9.000 | 5.000 | 66.000 | all_cores_refetch |
| 5 | False | 6 | 1 | 0 | 3.000 | 2.000 | 16.500 | all_cores_refetch |
| 6 | False | 21 | 1 | 0 | 11.000 | 6.000 | 82.500 | all_cores_refetch |
| 9 | False | 15 | 1 | 0 | 8.000 | 4.000 | 66.000 | all_cores_refetch |
| 11 | False | 17 | 1 | 0 | 9.000 | 5.000 | 66.000 | all_cores_refetch |
| 12 | False | 7 | 1 | 0 | 4.000 | 2.000 | 33.000 | all_cores_refetch |
| 16 | False | 11 | 1 | 0 | 6.000 | 3.000 | 49.500 | all_cores_refetch |
| 19 | False | 5 | 1 | 0 | 3.000 | 2.000 | 16.500 | all_cores_refetch |
| 26 | False | 21 | 1 | 0 | 11.000 | 6.000 | 82.500 | all_cores_refetch |
| 27 | False | 9 | 1 | 0 | 5.000 | 3.000 | 33.000 | all_cores_refetch |
| 30 | False | 14 | 1 | 0 | 7.000 | 4.000 | 49.500 | all_cores_refetch |
| 34 | False | 7 | 1 | 0 | 4.000 | 2.000 | 33.000 | all_cores_refetch |
| 35 | False | 28 | 1 | 0 | 14.000 | 7.000 | 115.500 | all_cores_refetch |
| 36 | False | 12 | 1 | 0 | 6.000 | 3.000 | 49.500 | all_cores_refetch |
| 38 | False | 18 | 1 | 0 | 9.000 | 5.000 | 66.000 | all_cores_refetch |
| 42 | False | 19 | 1 | 0 | 10.000 | 5.000 | 82.500 | all_cores_refetch |
| 45 | False | 26 | 1 | 0 | 13.000 | 7.000 | 99.000 | all_cores_refetch |
| 48 | False | 26 | 1 | 0 | 13.000 | 7.000 | 99.000 | all_cores_refetch |
| 49 | False | 16 | 1 | 0 | 8.000 | 4.000 | 66.000 | all_cores_refetch |
| 54 | False | 9 | 1 | 0 | 5.000 | 3.000 | 33.000 | all_cores_refetch |
| 63 | False | 36 | 1 | 0 | 18.000 | 9.000 | 148.500 | all_cores_refetch |

port_tight/B0：fixed/MILP 延迟比=1.000003；HBM 超额=+0.000%。2% 延迟门槛（≤1.02）通过；2% HBM 门槛通过。

port_tight/B1：fixed/MILP 延迟比=0.999998；HBM 超额=+0.000%。2% 延迟门槛（≤1.02）通过；2% HBM 门槛通过。

port_tight/B2：fixed/MILP 延迟比=0.999990；HBM 超额=+0.000%。2% 延迟门槛（≤1.02）通过；2% HBM 门槛通过。

port_tight/best_hetero：fixed/MILP 延迟比=1.147032；HBM 超额=+11.940%。2% 延迟门槛（≤1.02）未通过；2% HBM 门槛未通过。
仍有重读的任务原因：all_cores_refetch: 1；immediate_refetch_faster_than_wait: 9。
Shared 专家 -1：绑定核 1，重读倍数 2.000；所有已安装核的最小倍数=2.000，对应核 1；决策 all_cores_refetch。实际流量 66.000 MiB，已安装核最小流量 66.000 MiB。
以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。两者相同的强制重读贡献为零，未列入差值表。

| 专家 | Shared | Me | Fixed 核 | MILP 核 | Fixed r | MILP r | 差值 MiB | Fixed 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | False | 19 | 1 | 0 | 1.000 | 2.000 | -16.500 | clean |
| 23 | False | 12 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 30 | False | 14 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 33 | False | 21 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 36 | False | 12 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 41 | False | 17 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 57 | False | 21 | 0 | 1 | 2.000 | 1.000 | 16.500 | immediate_refetch_faster_than_wait |
| 63 | False | 36 | 0 | 1 | 4.000 | 1.000 | 49.500 | immediate_refetch_faster_than_wait |

port_tight/fixed_4+2：fixed/MILP 延迟比=1.030712；HBM 超额=+2.778%。2% 延迟门槛（≤1.02）未通过；2% HBM 门槛未通过。
仍有重读的任务原因：all_cores_refetch: 52。
Shared 专家 -1：绑定核 0，重读倍数 33.000；所有已安装核的最小倍数=33.000，对应核 0；决策 all_cores_refetch。实际流量 1089.000 MiB，已安装核最小流量 1089.000 MiB。
以下同一专家的字节差精确加总为 fixed−MILP 整层流量差。两者相同的强制重读贡献为零，未列入差值表。

| 专家 | Shared | Me | Fixed 核 | MILP 核 | Fixed r | MILP r | 差值 MiB | Fixed 决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | False | 15 | 1 | 0 | 8.000 | 4.000 | 66.000 | all_cores_refetch |
| 13 | False | 14 | 0 | 1 | 4.000 | 7.000 | -49.500 | all_cores_refetch |
| 19 | False | 5 | 0 | 1 | 2.000 | 3.000 | -16.500 | all_cores_refetch |
| 21 | False | 14 | 0 | 1 | 4.000 | 7.000 | -49.500 | all_cores_refetch |
| 23 | False | 12 | 1 | 0 | 6.000 | 3.000 | 49.500 | all_cores_refetch |
| 25 | False | 10 | 0 | 1 | 3.000 | 5.000 | -33.000 | all_cores_refetch |
| 27 | False | 9 | 0 | 1 | 3.000 | 5.000 | -33.000 | all_cores_refetch |
| 28 | False | 6 | 0 | 1 | 2.000 | 3.000 | -16.500 | all_cores_refetch |
| 29 | False | 17 | 0 | 1 | 5.000 | 9.000 | -66.000 | all_cores_refetch |
| 30 | False | 14 | 0 | 1 | 4.000 | 7.000 | -49.500 | all_cores_refetch |
| 33 | False | 21 | 1 | 0 | 11.000 | 6.000 | 82.500 | all_cores_refetch |
| 34 | False | 7 | 1 | 0 | 4.000 | 2.000 | 33.000 | all_cores_refetch |
| 35 | False | 28 | 1 | 0 | 14.000 | 7.000 | 115.500 | all_cores_refetch |
| 38 | False | 18 | 0 | 1 | 5.000 | 9.000 | -66.000 | all_cores_refetch |
| 43 | False | 12 | 1 | 0 | 6.000 | 3.000 | 49.500 | all_cores_refetch |
| 44 | False | 16 | 1 | 0 | 8.000 | 4.000 | 66.000 | all_cores_refetch |
| 45 | False | 26 | 1 | 0 | 13.000 | 7.000 | 99.000 | all_cores_refetch |
| 46 | False | 18 | 0 | 1 | 5.000 | 9.000 | -66.000 | all_cores_refetch |
| 54 | False | 9 | 0 | 1 | 3.000 | 5.000 | -33.000 | all_cores_refetch |
| 57 | False | 21 | 0 | 1 | 6.000 | 11.000 | -82.500 | all_cores_refetch |
| 63 | False | 36 | 1 | 0 | 18.000 | 9.000 | 148.500 | all_cores_refetch |

Shared 开始时间是模拟任务开始时间，不是原生 HBM 请求时间戳。MILP 指冻结离线专家分配后的可执行 LPT 回放，不代表证明了任意事件调度最优性。候选核的 ETA、预测完成时间与可绑定状态保留在 `task_hbm_deltas.csv` 的 candidate_comparisons 中。
