# 容量感知运行时调度修复

本次修复取得部分改善，但未通过全部验收目标。结论仅为冻结 E4 硬件上的运行时修复，不构成异构硬件胜出。整体留出集几何均值（GM）要求fixed/MILP≤1.01；指定GPQA实例要求延迟和HBM都在2%以内（延迟比≤1.02）。比值越小越快，95%改善下界越大越好，正值代表更快。

## 四个问题的直接回答

| 问题 | pipelined | port_tight |
| --- | --- | --- |
| 1. 同构/异构在线与离线GM差距 | B2: old/fixed GM=4.7512/3.9919 ms, MILP=3.8433 ms, fixed/MILP=1.038675 (未通过1.01)<br>best_hetero: old/fixed GM=3.9969/3.8236 ms, MILP=3.7869 ms, fixed/MILP=1.009680 (通过1.01)<br>fixed_4+2: old/fixed GM=5.9031/5.7314 ms, MILP=5.4836 ms, fixed/MILP=1.045177 (未通过1.01) | B2: old/fixed GM=11.0625/10.8959 ms, MILP=10.8960 ms, fixed/MILP=0.999992 (通过1.01)<br>best_hetero: old/fixed GM=11.7920/10.9175 ms, MILP=10.8254 ms, fixed/MILP=1.008503 (通过1.01)<br>fixed_4+2: old/fixed GM=17.3755/16.3690 ms, MILP=15.8366 ms, fixed/MILP=1.033618 (未通过1.01) |
| 2. 在线HBM>2%例外与GPQA目标 | B2: old 89 / fixed 75 个<br>best_hetero: old 30 / fixed 19 个<br>fixed_4+2: old 89 / fixed 83 个<br>GPQA best_hetero: 延迟比1.118337, HBM超额+12.500%; 未通过实例2%目标 | B2: old 0 / fixed 0 个<br>best_hetero: old 30 / fixed 15 个<br>fixed_4+2: old 72 / fixed 69 个<br>GPQA best_hetero: 延迟比1.147032, HBM超额+11.940%; 未通过实例2%目标 |
| 3. Fixed相对B1/B2的比值及配对95%下界 | best_hetero fixed/B1=1.007323, 95%改善下界-1.129%; fixed/B2=0.957827, 下界3.497% | best_hetero fixed/B1=1.028791, 95%改善下界-3.522%; fixed/B2=1.001980, 下界-0.574% |
| 4. 共同参数与bootstrap稳定性 | 共同t_big=3, large_first=True; 参数对胜出146/200, 排序分支胜出200/200; 只用18个开发窗口 | 共同t_big=3, large_first=True; 参数对胜出146/200, 排序分支胜出200/200; 只用18个开发窗口 |

`acceptance.json` 保留精确数值、门槛布尔值和未达标状态，所有失败项按实列出。

## 主延迟表：同硬件在线/离线

| 设计 | 模式 | 旧EFT GM ms | Fixed GM ms | MILP/LPT GM ms | Old/MILP | Fixed/MILP | GM≤1.01 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | pipelined | 4.7689 | 4.7689 | 4.7688 | 1.000020 | 1.000020 | 通过 |
| B1 | pipelined | 3.7958 | 3.7958 | 3.7958 | 0.999996 | 0.999996 | 通过 |
| B2 | pipelined | 4.7512 | 3.9919 | 3.8433 | 1.236225 | 1.038675 | 未通过 |
| best_hetero | pipelined | 3.9969 | 3.8236 | 3.7869 | 1.055433 | 1.009680 | 通过 |
| fixed_4+2 | pipelined | 5.9031 | 5.7314 | 5.4836 | 1.076502 | 1.045177 | 未通过 |
| B0 | port_tight | 13.6463 | 13.6463 | 13.6460 | 1.000022 | 1.000022 | 通过 |
| B1 | port_tight | 10.6120 | 10.6120 | 10.6118 | 1.000011 | 1.000011 | 通过 |
| B2 | port_tight | 11.0625 | 10.8959 | 10.8960 | 1.015282 | 0.999992 | 通过 |
| best_hetero | port_tight | 11.7920 | 10.9175 | 10.8254 | 1.089282 | 1.008503 | 通过 |
| fixed_4+2 | port_tight | 17.3755 | 16.3690 | 15.8366 | 1.097175 | 1.033618 | 未通过 |

## 相对B1/B2与配对置信下界

| 设计 | 模式 | Fixed/B1 | 相对B1改善95%下界 % | Fixed/B2 | 相对B2改善95%下界 % |
| --- | --- | --- | --- | --- | --- |
| B0 | pipelined | 1.256375 | -33.172 | 1.194641 | -25.817 |
| B1 | pipelined | 1.000000 | 0.000 | 0.950864 | 4.000 |
| B2 | pipelined | 1.051675 | -6.184 | 1.000000 | 0.000 |
| best_hetero | pipelined | 1.007323 | -1.129 | 0.957827 | 3.497 |
| fixed_4+2 | pipelined | 1.509925 | -66.217 | 1.435733 | -57.052 |
| B0 | port_tight | 1.285939 | -36.487 | 1.252427 | -32.741 |
| B1 | port_tight | 1.000000 | 0.000 | 0.973940 | 2.166 |
| B2 | port_tight | 1.026758 | -3.157 | 1.000000 | 0.000 |
| best_hetero | port_tight | 1.028791 | -3.522 | 1.001980 | -0.574 |
| fixed_4+2 | port_tight | 1.542505 | -68.798 | 1.502306 | -64.013 |

改善=100×(1−候选/基线)。配对百分位bootstrap为2,000次、seed=20261007；完整双侧区间、配对窗口数在`paired_improvement.csv`。B0/B1为单核、B2为同构双核、另两者为异构双核。

## 所有HBM>2%例外与未达标原因

| 设计 | 模式 | 在线调度 | >2%例外 | 窗口数 | 最大超额 % |
| --- | --- | --- | --- | --- | --- |
| B0 | pipelined | eft_old | 0 | 135 | 0.000 |
| B0 | pipelined | fixed | 0 | 135 | 0.000 |
| B1 | pipelined | eft_old | 0 | 135 | 0.000 |
| B1 | pipelined | fixed | 0 | 135 | 0.000 |
| B2 | pipelined | eft_old | 89 | 135 | 245.714 |
| B2 | pipelined | fixed | 75 | 135 | 22.222 |
| best_hetero | pipelined | eft_old | 30 | 135 | 98.438 |
| best_hetero | pipelined | fixed | 19 | 135 | 15.152 |
| fixed_4+2 | pipelined | eft_old | 89 | 135 | 46.619 |
| fixed_4+2 | pipelined | fixed | 83 | 135 | 31.317 |
| B0 | port_tight | eft_old | 0 | 135 | 0.000 |
| B0 | port_tight | fixed | 0 | 135 | 0.000 |
| B1 | port_tight | eft_old | 0 | 135 | 0.000 |
| B1 | port_tight | fixed | 0 | 135 | 0.000 |
| B2 | port_tight | eft_old | 0 | 135 | 0.000 |
| B2 | port_tight | fixed | 0 | 135 | 0.000 |
| best_hetero | port_tight | eft_old | 30 | 135 | 95.455 |
| best_hetero | port_tight | fixed | 15 | 135 | 11.940 |
| fixed_4+2 | port_tight | eft_old | 72 | 135 | 21.765 |
| fixed_4+2 | port_tight | fixed | 69 | 135 | 12.500 |

HBM超额相对同窗口/设计/模式MILP/LPT实际流量计算，原生唯一权重字节数另列。`hbm_bytes.csv`列全部主比较，`excess_hbm_windows.csv`完整列出旧EFT/Fixed所有>2%例外、原因和任务明细；控制组例外单列。

Fixed仍有261个>2%窗口。按同一专家比较fixed−MILP，其中1165个任务增加52156.500 MiB，343个任务减少9916.500 MiB；增量任务中300个有r=1替代核，865个所有核r>1。`task_hbm_deltas.csv`保留正负差值与候选核A/B的ETA/预测完成/可绑定状态，并核对每个窗口任务差值之和精确等于整层差值。两种调度相同的强制重读贡献为零，不应归因于残留。

规则1只在候选r>1且另一个已安装核r=1时触发：若等待无重读核后的预测完成时间不晚，才拒绝当前候选；ETA认为立即重读更快时允许重读。所有核r>1时保留EFT，未保证最低流量。例如Shared128大核r=2、小核r=26，只有部分重读不可避免，额外24次仍取决于归属。这些是当前未达标原因；未来ETA不会提前授予实际资源。

具体GPQA的旧/Fixed/MILP Shared归属、开始、HBM、延迟、2%验收及真正改变的专家流量见`gpqa_t128_case.md`。

## 预测器控制

`compare.csv`只含原E4 EFT、Fixed、MILP/LPT；双核Fixed使用未修改的predictor='ours'。原EFT+ours单列在`compare_predictor_control.csv`。Fixed/control使用同一预测算法与开发集预热协议，但各自调度会形成不同反馈状态，因此不是排除状态交互的纯因果减法。B0/B1保留predictor=None。

| 设计 | 模式 | Fixed/control | Control/old | Fixed/old |
| --- | --- | --- | --- | --- |
| B0 | pipelined | 1.000000 | 1.000000 | 1.000000 |
| B1 | pipelined | 1.000000 | 1.000000 | 1.000000 |
| B2 | pipelined | 0.851492 | 0.986738 | 0.840199 |
| best_hetero | pipelined | 0.952291 | 1.004576 | 0.956649 |
| fixed_4+2 | pipelined | 0.967661 | 1.003349 | 0.970901 |
| B0 | port_tight | 1.000000 | 1.000000 | 1.000000 |
| B1 | port_tight | 1.000000 | 1.000000 | 1.000000 |
| B2 | port_tight | 0.984957 | 0.999984 | 0.984941 |
| best_hetero | port_tight | 0.925555 | 1.000309 | 0.925841 |
| fixed_4+2 | port_tight | 0.940756 | 1.001400 | 0.942072 |

## 共同参数与稳定性

共同策略：t_big=3，large_first=True。只在 18 个开发窗口上选择，目标是全部五个冻结设计×两模式相对旧 EFT 的配对延迟比几何均值；留出窗口只用于评估。
Bootstrap 稳定性：所选参数对在 146/200 次重采样中胜出（73.0%）。这是开发集选参稳定性，不证明普遍最优阈值。

| t_big | large_first | 胜出次数 | 比例 |
| --- | --- | --- | --- |
| 2 | False | 0 | 0.0000 |
| 3 | False | 0 | 0.0000 |
| 4 | False | 0 | 0.0000 |
| 6 | False | 0 | 0.0000 |
| 8 | False | 0 | 0.0000 |
| 2 | True | 9 | 0.0450 |
| 3 | True | 146 | 0.7300 |
| 4 | True | 44 | 0.2200 |
| 6 | True | 1 | 0.0050 |
| 8 | True | 0 | 0.0000 |

所选 large_first 分支胜出 200/200 次（100.0%）；分支稳定性与精确阈值参数对的稳定性不同。

## 精确回归、冻结硬件与边界

B0/B1逐窗口回归540/540项cycles与完整结果哈希均bit exact；独立完整重复运行哈希一致5400/5400项，0行缺重复哈希。详见`regression.csv`；仅延迟相同不足以通过。

| 设计 | pipelined | port_tight |
| --- | --- | --- |
| B0 | 6x4x512 | 6x4x512 |
| B1 | 3x32x128 | 3x32x128 |
| B2 | 2x24x128+2x24x128 | 3x16x128+3x16x128 |
| best_hetero | 1x32x128+2x32x128 | 1x32x128+2x32x128 |
| fixed_4+2 | 4x4x512+2x4x512 | 4x4x512+2x4x512 |

几何、私有SRAM、端口、数据流全部冻结。pipelined B2为2x24x128双核且数据流OS/WS；port_tight B2为3x16x128双核且WS/WS，同构几何不保证两核成本相同。

范围：post-router BF16 Gate/Up、SiLU/Z、Down、combine的phase-fluid解析模拟。1假想cycle=1ns，ms=cycles/1e6，MiB=2^20字节。HBM是声明的模拟线传输量，不是实芯片测量；重叠计数不能相加当墙钟时间。不是RTL或全模型tokens/s，排除attention、router执行与完整服务。MILP为专家分配见证后的可执行LPT回放，分配最优不等于任意时序最优。

离线求解状态：{"OPTIMAL":1350}。

只重建报告（支持无损归档per_window.json.gz）：
```sh
python -m research.moe_dispatch.round2.dispatch_fix.report --directory PATH
```
