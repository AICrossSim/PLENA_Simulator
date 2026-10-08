# 第三轮：256 GB/s 供数上限下的完整评估

本报告使用 BF16、18 个开发窗口和 135 个既有留出窗口；1 模型周期＝1 ns。所有 ms 为路由完成后的 MoE FFN 解析估计，包含 Gate/Up、SiLU、Down 和汇合，不包含完整模型的 attention/router/norm，也不是原生 HBM 或 RTL 实测。

| 问题 | 数值答案 | 数据来源 |
| --- | --- | --- |
| Q1 最大可能收益 | 当前模型下，整体相对本轮 B1 至多 1.1264%，相对 B2 至多 2.7117%；达不到同时快 5%。B96/128 单独仍有空间。 | E4/selected_baseline_headroom.csv |
| Q2 W 槽是否主因 | H0 1.9529 → H2 1.9188 ms；W∞可回收 61.02%，未达 80% 标准。 | E3/attribution.json |
| Q3 重新搜索后 | B1 1.8785；B2 1.9091；开发集选定异构 H51 2.1333 ms。通过双基线 5% 门槛的异构 0 个。 | E4/heldout_main_table.csv; gates.csv |
| Q4 在线/离线 | B2 0.886344；H51 1.125440。1.01 验收按实际比值判断，未达标项另列原因。 | E5/dispatch/compare.csv |
| Q5 预测器收益 | ours 相对 nominal 的延迟变化：B1 0.0105%，B2 -0.0844%，H51 0.9081%。负值才是更快。 | E5/predictor/predictor_table.csv |

以下区分“126 下选出、在 256 下评估”和“256 下重新搜索”。不同控制／供数计数可以重叠，不能相加成延迟；不同消融收益也不能相加。

## 1. 工作点与前端

`min(256,credits×32/65) GB/s`：256／390／520 个额度分别对应 126.030769／192／256。三种组织共用同一前端。520 点要求平均每周期 8 个 32 B 请求、足够标签及返回落地吞吐，尚无 RTL／面积验证；详见 E1/OPERATING_POINT.md。

主设计固定 12,288 个乘法器、2,158,592 B 存储、W/X/累加 bank 总量 64/24/12、向量吞吐 64 元素/周期。可交换的私有／落地容量总量为 532 KiB，其余公共存储固定。等乘法器、容量、端口不能代替等综合面积。

## 2. 下界与能够争取的空间

| Batch | 窗口数 | 相对冻结 B1 最大收益 % | 相对冻结 B2 最大收益 % | 双基线 5% 是否可能 |
| --- | --- | --- | --- | --- |
| 2 | 27 | 0.1065 | 1.7458 | False |
| 4 | 27 | 0.1110 | 1.3345 | False |
| 8 | 27 | 0.1277 | 1.2143 | False |
| 16 | 27 | 0.1411 | 0.4546 | False |
| 64 | 9 | 6.1491 | 7.5535 | True |
| 96 | 9 | 10.1745 | 14.4945 | True |
| 128 | 9 | 12.2889 | 18.8135 | True |
| all | 135 | 2.0878 | 3.8378 | False |

下界按每窗口的唯一权重、MAC、向量、必需片上端口及乐观 532 KiB 行分块计算，取最大值。旧 Z384/W40 的区域限制已经删除。新的行分块下界同时免费授予每个任务 Z532 和 W532，故是乐观放宽；其有效性依赖当前完整行的 Gate/Up→Down 生命周期，不能用于未来的跨列融合。

整体能否胜出按全部窗口配对几何平均判断；单个大 batch 仍有局部空间。本轮全体已检查结果的下界验收见 VALIDATION.json。

E2 上限的参照是历史冻结基线及其原控制协议。开发集重新调优不保证留出集也更快，因此 E4 的开发集证明和留出集门槛分别使用最终同协议 B1/B2，不能直接套用历史上限。

下面把同一个逐窗口 LB 与第三轮冻结硬件的实际 fixed 结果重新配对；不修改 E2 历史表，也不假设开发集调优必然改善留出集。

| Batch | 窗口数 | LB GM ms | 本轮 B1 GM ms | 最多比 B1 快 % | 最多比 B2 快 % | 双基线 5% 是否可能 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 27 | 0.8464 | 0.8480 | 0.1823 | 1.7888 | False |
| 4 | 27 | 1.4248 | 1.4275 | 0.1903 | 1.3731 | False |
| 8 | 27 | 1.7855 | 1.7891 | 0.2002 | 1.2451 | False |
| 16 | 27 | 2.3801 | 2.3851 | 0.2119 | 0.4555 | False |
| 64 | 9 | 4.1230 | 4.2416 | 2.7964 | 4.5807 | False |
| 96 | 9 | 4.2931 | 4.5687 | 6.0341 | 9.0226 | True |
| 128 | 9 | 4.5325 | 4.7924 | 5.4224 | 11.6633 | True |
| all | 135 | 1.8574 | 1.8785 | 1.1264 | 2.7117 | False |

## 3. 权重缓冲消融：126 下选出、在 256 下评估

| 配置 | 等资源 | GM ms | 相对单核 | 说明 |
| --- | --- | --- | --- | --- |
| S0 | True | 1.8970 | 1.000000 | 冻结单核 |
| Sinf | False | 1.8970 | 1.000000 | W前瞻无限（诊断；非等资源） |
| H0 | True | 1.9529 | 1.029478 | 冻结异构：大核16／小核24 KiB |
| H1 | True | 1.9304 | 1.017629 | 大核24／小核16 KiB |
| H2 | True | 1.9188 | 1.011491 | 大核32／小核24 KiB；acc_bytes[1] −16 KiB |
| H3 | True | 1.9259 | 1.015267 | 共享40 KiB落地池；私有W为0 |
| Hinf | False | 1.9188 | 1.011492 | 两核W前瞻无限（诊断；非等资源） |
| M0 | True | 1.9315 | 1.018199 | 冻结同构 |
| M3 | True | 1.9325 | 1.018747 | 共享40 KiB落地池；私有W为0 |
| Minf | False | 1.9315 | 1.018206 | 两核W前瞻无限（诊断；非等资源） |

可回收比例 61.02%，按任务书标准不能写 W 槽不足是主因。余下的重读、核完成差及端口服务积分见 E3/ABLATION.md；它们只能作为诊断，未被分离成互斥的因果时间。W∞保持原行分块和阶段生命周期，只放宽有效 W 驻留／前瞻窗口，不自动引入跨阶段完整专家缓存。

曲线中的在途量是“当前流体供数率×65 ns”的服务等价代理量，冷响应等待为 0，不能据此验证逐请求槽位占用。共享池实际有限容量由独立字节预留账本检查。single_fetcher 指只有一个核获得正供数的区间；Shared 供数率是大核执行 Shared 期间全芯片 HBM 的速率，包含另一核。

## 4. 256 下重新搜索的硬件

### pipelined，C0 主约束

| 组织 | PM×PN×PK | 数据流 | 落地 | W KiB | X KiB | 累加 KiB | Z KiB | W/X/累加 banks | 向量份额 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 1×48×256 | WS | private | 64 | 32 | 64 | 372 | 64/24/12 | 64 |
| B2 | 1×48×128+1×48×128 | WS+WS | shared | 共享池 64 | 16+16 | 32+32 | 186+186 | 32+32/12+12/6+6 | 32+32 |
| H33 | 1×48×128+4×3×512 | WS+WS | private | 36+24 | 24+24 | 16+16 | 190+202 | 32+32/12+12/6+6 | 32+32 |
| H42 | 1×64×128+2×32×64 | WS+WS | private | 53+27 | 43+21 | 32+16 | 227+113 | 43+21/16+8/8+4 | 43+21 |
| H51 | 1×40×256+2×2×512 | WS+WS | private | 66+14 | 53+11 | 40+8 | 283+57 | 53+11/20+4/10+2 | 53+11 |

### port_tight，C0 主约束

| 组织 | PM×PN×PK | 数据流 | 落地 | W KiB | X KiB | 累加 KiB | Z KiB | W/X/累加 banks | 向量份额 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 3×32×128 | WS | private | 64 | 32 | 64 | 372 | 64/24/12 | 64 |
| B2 | 3×16×128+3×16×128 | WS+WS | private | 24+24 | 24+24 | 16+16 | 202+202 | 32+32/12+12/6+6 | 32+32 |
| H33 | 3×16×128+3×16×128 | WS+WS | shared | 共享池 64 | 16+16 | 32+32 | 186+186 | 32+32/12+12/6+6 | 32+32 |
| H42 | 1×32×128+2×32×128 | WS+WS | private | 24+16 | 2+10 | 16+80 | 32+352 | 32+32/4+20/4+8 | 8+56 |
| H51 | 2×20×256+8×4×64 | WS+WS | shared | 共享池 32 | 26+6 | 13+3 | 376+76 | 32+32/12+12/6+6 | 32+32 |

每核维度、存储和端口按 core0/core1 顺序列出；W 池只计算一次，共享池的各核单独最大槽数不能同时相加。H33 允许与同构空间重叠；若最终核心形状相同，不能把标签 H33 本身当作异构贡献。

H51／H42／H33 表示两核乘法器预算的无序划分族 5+1／4+2／3+3，不表示 core0/core1 必须按这个顺序排列，也不能当成 PM 行数。具体核顺序以每行 core0/core1 的形状和资源列表为准。PM 对应一次处理的 token 行数，PN 是输出列宽，PK 是物理点积宽度；本轮三个维度均有变化。

共享落地池开放的是字节容量共享；本轮仍保留各核冻结的 W bank／读端口份额，没有免费借用另一核的端口。C1 检查扣除计算块后的物理在途空间不少于 16 KiB，它不保证实际供数达到 252 GB/s：尾块有效载荷、阶段边界和片上读端口仍可能限制速率。

主目标是 18 开发窗口上 MILP 资源分配＋物理 LPT 回放延迟的几何平均。CVaR10、按 batch 最坏比值和 200 次 bootstrap 为诊断，统一使用同一 B1 参考向量。选择后硬件冻结，不按留出 batch 更换配置。

实际覆盖限制必须同时看 SEARCH_COVERAGE_ZH.md：每组 256 个实评点均来自初始点／seed，分支定界尚未解析任何单点叶子。候选生成只有六套容量总额 profile，容量份额和 bank 份额仍主要按等分／算力比例耦合；声明的独立容量、bank 和数据流大空间尚未被充分覆盖。C0/C1 联合重选对每族使用相同生成预算，不能代替全空间优化。

| 模式 | 约束 | 族 | A：双基线 5% | B：全局零差距 | 剩余差距 % | 开放区域 |
| --- | --- | --- | --- | --- | --- | --- |
| pipelined | C0 | single | True | False | 2.1068 | 1189 |
| pipelined | C0 | homogeneous | True | False | 17.7577 | 1662 |
| pipelined | C0 | 5+1 | True | False | 4.3248 | 2050 |
| pipelined | C0 | 4+2 | True | False | 7.2829 | 2050 |
| pipelined | C0 | 3+3 | True | False | 14.3109 | 2050 |
| pipelined | C1 | single | True | False | 2.1068 | 1189 |
| pipelined | C1 | homogeneous | True | False | 17.7577 | 1662 |
| pipelined | C1 | 5+1 | True | False | 4.3248 | 2050 |
| pipelined | C1 | 4+2 | True | False | 7.2829 | 2050 |
| pipelined | C1 | 3+3 | True | False | 14.3109 | 2050 |
| port_tight | C0 | single | True | False | 0.7444 | 1189 |
| port_tight | C0 | homogeneous | True | False | 4.7219 | 1662 |
| port_tight | C0 | 5+1 | True | False | 4.0985 | 2050 |
| port_tight | C0 | 4+2 | True | False | 3.2716 | 2050 |
| port_tight | C0 | 3+3 | True | False | 4.7219 | 2050 |
| port_tight | C1 | single | True | False | 0.7444 | 1189 |
| port_tight | C1 | homogeneous | True | False | 4.7219 | 1662 |
| port_tight | C1 | 5+1 | True | False | 4.0985 | 2050 |
| port_tight | C1 | 4+2 | True | False | 3.5304 | 2050 |
| port_tight | C1 | 3+3 | True | False | 4.7219 | 2050 |

A 的闭合表示可以排除同时比 B1 和 B2 快 5%，不表示该族已找到全局最优。B 未闭合时，本报告只称“等预算搜索中已评估的最好设计”，开放区域和下界保留在 certificates/。内层 CP-SAT 的 OPTIMAL 只证明资源分配松弛最优，LPT 是一个合法回放，不证明联合时序调度全局最优。

### 开发集近优候选的留出集鲁棒诊断

另对开发集距离各族最好已测点不超过 1% 的 334 个组内候选进行留出诊断，按模式和完整物理配置去重为 187 点。12 点复用已校验的双遍回放，175 点补做两遍，共新增 47250 次逐窗口物理解析回放。

以同模式本轮冻结 B1 为共同参照，比较窗口延迟比的几何平均、CVaR10 和最坏 batch；200 次 bootstrap 只重采样开发窗口。结果和后验赢家见 E4/robust_heldout/SUMMARY_ZH.md、robust_objectives.csv、winners.csv 和 selection_stability.csv。留出赢家只作诊断，未替换 selected_designs 或主表，也不是新的盲测选择。

| 模式 | 约束 | 跨族近优候选数 | GM 后验赢家形状 | 三目标是否同一完整配置 |
| --- | --- | --- | --- | --- |
| pipelined | C0 | 13 | 1x48x256 | True |
| pipelined | C1 | 12 | 1x48x256 | True |
| port_tight | C0 | 160 | 3x32x128 | True |
| port_tight | C1 | 143 | 3x32x128 | True |

### C0 主表：全部为 ms、留出集几何平均，fixed 派工

| 模式 | 组织 | B2 | B4 | B8 | B16 | B64 | B96 | B128 | 全部 GM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipelined | B0 | 0.8738 | 1.4788 | 2.0363 | 3.1102 | 8.2999 | 11.6783 | 14.8940 | 2.4730 |
| pipelined | B1 | 0.8480 | 1.4275 | 1.7891 | 2.3851 | 4.2416 | 4.5687 | 4.7924 | 1.8785 |
| pipelined | B2 | 0.8619 | 1.4447 | 1.8080 | 2.3910 | 4.3209 | 4.7188 | 5.1310 | 1.9091 |
| pipelined | H51 | 0.8926 | 1.5453 | 2.0293 | 2.6922 | 4.8931 | 6.0173 | 6.8458 | 2.1333 |
| pipelined | H42 | 0.8649 | 1.4577 | 1.8068 | 2.4102 | 4.4074 | 4.9255 | 5.4816 | 1.9332 |
| pipelined | H33 | 0.8619 | 1.4446 | 1.8282 | 2.5534 | 5.4421 | 6.4402 | 7.3808 | 2.0593 |
| pipelined | fixed_4+2 | 0.8660 | 1.4509 | 2.0619 | 3.6215 | 11.8417 | 16.9961 | 22.0880 | 2.7394 |
| port_tight | B0 | 4.9235 | 8.2877 | 11.3185 | 17.0816 | 44.3220 | 61.9261 | 78.6523 | 13.6464 |
| port_tight | B1 | 4.8259 | 8.1233 | 10.1799 | 13.5701 | 23.5090 | 25.2510 | 25.8588 | 10.6121 |
| port_tight | B2 | 4.9872 | 8.3292 | 10.4114 | 13.6669 | 24.4427 | 26.1901 | 27.3986 | 10.8959 |
| port_tight | H51 | 5.0619 | 8.4305 | 10.5283 | 13.8945 | 24.2166 | 26.4614 | 27.7657 | 11.0261 |
| port_tight | H42 | 4.9872 | 8.3291 | 10.4113 | 13.7399 | 23.7513 | 25.9330 | 28.7447 | 10.9143 |
| port_tight | H33 | 4.9872 | 8.3292 | 10.4115 | 13.6669 | 24.4427 | 26.1901 | 27.4835 | 10.8982 |
| port_tight | fixed_4+2 | 5.2496 | 8.5465 | 12.1826 | 21.1695 | 85.7681 | 126.2501 | 165.6935 | 16.9667 |

MILP 回放、C1、资源积分和双基线置信区间见同目录主表／breakdown／gates。在线/离线和预测器表采用各自注明的控制设置，不能把它们的值混为同一测量。

| 异构候选 | /B1 | /B2 | 进入校准 |
| --- | --- | --- | --- |
| H51 | 1.135629 | 1.117421 | False |
| H42 | 1.029126 | 1.012625 | False |
| H33 | 1.096240 | 1.078663 | False |

5% 是同时针对两个调优基线的进入校准门槛；10% 和置信区间下界 5% 是校准后的宣称胜出门槛。未做 Rust/native 校准及综合，不能宣称校准后的胜出或面积／能耗收益。

### 同协议交叉带宽表

| 选择带宽 | 模式 | 组织 | 126 ms | 256 ms |
| --- | --- | --- | --- | --- |
| 126 下选出 | pipelined | B1 | 3.7958 | 1.8976 |
| 126 下选出 | pipelined | B2 | 3.9003 | 2.1498 |
| 126 下选出 | pipelined | best_hetero | 3.7891 | 1.9510 |
| 126 下选出 | port_tight | B1 | 10.6120 | 10.6122 |
| 126 下选出 | port_tight | B2 | 10.8959 | 10.8960 |
| 126 下选出 | port_tight | best_hetero | 10.9198 | 10.9143 |
| 256 下重新搜索 | pipelined | B1 | 3.7843 | 1.8785 |
| 256 下重新搜索 | pipelined | B2 | 3.8129 | 1.9091 |
| 256 下重新搜索 | pipelined | H51 | 3.9711 | 2.1333 |
| 256 下重新搜索 | pipelined | H42 | 3.8147 | 1.9332 |
| 256 下重新搜索 | pipelined | H33 | 3.8250 | 2.0593 |
| 256 下重新搜索 | port_tight | B1 | 10.6118 | 10.6121 |
| 256 下重新搜索 | port_tight | B2 | 10.8958 | 10.8959 |
| 256 下重新搜索 | port_tight | H51 | 11.0261 | 11.0261 |
| 256 下重新搜索 | port_tight | H42 | 10.9198 | 10.9143 |
| 256 下重新搜索 | port_tight | H33 | 10.8981 | 10.8982 |

交叉表对新旧硬件统一使用本轮 fixed 和本轮开发集选定参数；单核关闭预测器以保持派工回归逐位一致，双核使用 ours。旧硬件保留原 E4 数据流；历史 7eb58061 的共同 WS／旧 runtime 协议值保留在 E0/E2，不与交叉表强行相等。

## 5. 派工与预测器

新 fixed 在选择时计入重读的共享总线代价；本任务自身重读已经在成本里，只追加对并发其他用户的带宽外部代价，避免重复计费。所有核都需重读时，先选重读倍数最小者，同倍数再比较完成时刻。实际执行仍按有限资源积分计时，预测不释放资源。

| 模式 | 组织 | fixed ms | 在线/离线 | 是否 ≤1.01 |
| --- | --- | --- | --- | --- |
| pipelined | B0 | 2.4730 | 1.000049 | True |
| pipelined | B1 | 1.8785 | 0.999952 | True |
| pipelined | B2 | 1.9091 | 0.886344 | True |
| pipelined | H51 | 2.1333 | 1.125440 | False |
| pipelined | H42 | 1.9332 | 0.984392 | True |
| pipelined | H33 | 2.0593 | 1.012550 | False |
| pipelined | fixed_4+2 | 2.7394 | 1.011401 | False |
| port_tight | B0 | 13.6464 | 1.000014 | True |
| port_tight | B1 | 10.6121 | 0.999995 | True |
| port_tight | B2 | 10.8959 | 0.999984 | True |
| port_tight | H51 | 11.0261 | 1.004507 | True |
| port_tight | H42 | 10.9143 | 1.008193 | True |
| port_tight | H33 | 10.8982 | 0.999988 | True |
| port_tight | fixed_4+2 | 16.9667 | 1.071355 | False |

所有在线 HBM 比离线多 2% 以上的窗口逐一列在 E5/dispatch/excess_hbm_windows.csv；单核逐位回归见 regression.csv。GPQA B128 的 Shared 归属、开始时刻、字节和延迟见 gpqa_t128_case.md。离线回放是参照，不是物理全局最优，在线比它更快也不矛盾。

H51 的具体准入缺口见 E5/dispatch/DISPATCH_DIAGNOSIS.md：GPQA B128 在 3.691454 ms 绑定一个 Me=18 的任务时，已预测大核完成于 3.766665 ms、小核于 8.632654 ms，但大核暂时不能接单，仍把任务给了小核。两核都无需重读，当前“允许等待”的规则没有覆盖这种情形。该案例不能归因于 ETA 不准；本轮保留原冻结控制逻辑及失败结果，没有用未经评测的新等待规则替换主表。

预测器在拉数前估计服务时长和完成时刻，不猜已经由 Router 确定的专家 ID。nominal 不学习，保留与其他方法相同的 25%/50%/75% 进度检查与有限 Next 准入。随机方法随机估计时长，不随机选择核。126 对照沿用旧硬件和 fixed_legacy；256 用新硬件和新 fixed，方法优劣仅在同工作点内配对比较。

| 组织 | 方法 | 全部 GM ms | /nominal | MAE % | late % | 等待空转 % |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | nominal | 1.8785 | 1.000000 | 0.2209 | 50.9684 | 0.0444 |
| B1 | random | 2.1971 | 1.169613 | 66.6409 | 59.9412 | 0.0473 |
| B1 | static | 1.9259 | 1.025237 | 8.9951 | 97.5729 | 0.0885 |
| B1 | btb | 1.8795 | 1.000537 | 0.9438 | 52.4884 | 0.0616 |
| B1 | ema | 1.8786 | 1.000027 | 0.7723 | 83.8686 | 0.0696 |
| B1 | ours | 1.8787 | 1.000105 | 0.0374 | 9.5367 | 0.0042 |
| B1 | oracle | 1.8787 | 1.000105 | 0.0000 | 9.5367 | 0.0042 |
| B2 | nominal | 1.9107 | 1.000000 | 22.3226 | 10.3195 | 0.0041 |
| B2 | random | 1.9133 | 1.001323 | 54.9640 | 54.0061 | 0.0249 |
| B2 | static | 1.9079 | 0.998530 | 9.8014 | 95.5375 | 0.0441 |
| B2 | btb | 1.9008 | 0.994778 | 3.9018 | 68.4077 | 0.0325 |
| B2 | ema | 1.9021 | 0.995464 | 3.7816 | 56.5923 | 0.0265 |
| B2 | ours | 1.9091 | 0.999156 | 2.0099 | 94.9544 | 0.0397 |
| B2 | oracle | 1.9091 | 0.999156 | 0.0000 | 94.9544 | 0.0397 |
| H51 | nominal | 2.1141 | 1.000000 | 7.3893 | 10.1420 | 0.0026 |
| H51 | random | 2.5762 | 1.218563 | 63.6282 | 53.8286 | 0.0167 |
| H51 | static | 2.2684 | 1.072983 | 24.9280 | 71.7292 | 0.0262 |
| H51 | btb | 2.1483 | 1.016194 | 3.4905 | 59.7617 | 0.0191 |
| H51 | ema | 2.1103 | 0.998198 | 4.6222 | 64.5284 | 0.0243 |
| H51 | ours | 2.1333 | 1.009081 | 3.7818 | 30.6034 | 0.0062 |
| H51 | oracle | 2.1333 | 1.009081 | 0.0000 | 30.6034 | 0.0062 |

oracle 对 ours 冻结行动计划进行独立物理回放，使用真实时长计算误差并重新验证供数、有限池和完成时刻；它不是能改变任务归属的先知调度上界。predictor_table.csv 含分 batch 延迟和整体指标；accuracy_by_batch.csv 含分 batch 的准确率、success@W、late 和 late>64/256 指标。预测更准不保证层延迟更低；慢下来的方法同时报告重读字节、等待和派工变化。

## 6. 敏感性与合成反向搜索

| 参数 | Sobol 一阶 S1 | S1 置信半宽 | 总效应 ST | ST 置信半宽 |
| --- | --- | --- | --- | --- |
| weight_tile_service_cycles | 0.1492 | 0.2042 | 1.0156 | 0.2607 |
| bank_Bpc | -0.0495 | 0.1254 | 0.5454 | 0.2478 |
| dotstagecycles | 0.0025 | 0.0038 | 0.0441 | 0.0600 |
| credits | 0.0349 | 0.0858 | 0.4894 | 0.2002 |
| vector_scale | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Sobol 使用 N=256、五参数、1,792 个样本，每点重新选几何、数据流、私有／共享容量与端口并重复完整过程。在途额度范围 256–640，其余范围沿第二轮。若搜索证明开放，指数衡量的是等预算搜索程序及其最好已测候选的敏感性，不能声称全局最优硬件的 Sobol 指数。翻转点逐一列在 flip_points.csv。

表中保留有限样本的原始估计值和置信半宽，未把负的一阶估计或超过 1 的总效应裁剪为比例。置信区间较宽，不能据此给出精确的重要性排序；总效应也不能相加成延迟归因。硬件重新选择与有限搜索候选的跳变都包含在这个响应函数里。

响应量使用同一组 18 个开发窗口：delta = 三个双核比例族（5+1、4+2、3+3）中最好已评估候选的开发集延迟几何平均 / 最好已评估单核的开发集延迟几何平均 − 1。负值仅表示该预算内已测双核候选更快；H33（3+3）允许两核计算形状相同，因此不能把双核族标签解释为严格异构。指数不使用留出窗口选硬件。

| 扫描 | 完整点数 | 资源模型回放次数（含双遍） | 全局零差距证明闭合点数 |
| --- | --- | --- | --- |
| Sobol | 1792 | 5160960 | 0 |
| 合成反向搜索 | 1440 | 230400 | 0 |

Sobol 的已测 delta 范围为 0.7152%～4.9257%，观察到的双核／单核排序反转点为 0 个。S1／ST／置信半宽从完整 delta 序列独立重算，最大逐值差为 0.0；证书、全量索引、源冻结和重复核验见 E5/sobol/COMPLETION_AUDIT.json。

获选双核中计算形状不同的样本有 1735 个、相同的有 57 个，后者属于 H33 的同构重叠空间。逐窗口下界另独立检查 1290240 项，违例 0。样本内的局部 5% incumbent 证书只限制该族已测可执行参照点的改进幅度，不是主表同时对 B1/B2 的 5% 胜出门槛。

### 合成负载：反转只用于探索，不进入主表

| 合成 Batch | 完整点数 | 双核已测领先点数 | 其中计算形状不同 | 最大已测领先 % | 全局证明 B 闭合点数 |
| --- | --- | --- | --- | --- | --- |
| B2 | 180 | 179 | 179 | 0.2109 | 0 |
| B4 | 180 | 173 | 173 | 0.2200 | 0 |
| B8 | 180 | 177 | 177 | 0.2690 | 0 |
| B16 | 180 | 171 | 171 | 0.2613 | 0 |
| B32 | 180 | 30 | 30 | 0.2604 | 0 |
| B64 | 180 | 18 | 18 | 0.2337 | 0 |
| B128 | 180 | 4 | 4 | 0.0724 | 0 |
| B256 | 180 | 3 | 3 | 0.1099 | 0 |

以上领先只是在每族相同有限搜索预算内比较已测候选，不能写成对全局最优单核的胜出。完整逐点区间、硬件、证明和重复记录保留在 E4/synthetic/reverse_search.csv、SYNTHETIC_ZH.md 与 COMPLETION_AUDIT.json。合成数据只探索可能的工作区间，不进入真实留出主表，也不能代替真实模型精度／推理验证。

## 7. 验收与局限

完整自动验收结果：

```json
{
  "checks": [
    {
      "details": 5670,
      "name": "schema:E0/repro.csv",
      "passed": true
    },
    {
      "details": 459,
      "name": "schema:E2/bounds_by_window.csv",
      "passed": true
    },
    {
      "details": 48,
      "name": "schema:E2/bounds_summary.csv",
      "passed": true
    },
    {
      "details": 459,
      "name": "schema:E2/port_tight/bounds_by_window.csv",
      "passed": true
    },
    {
      "details": 48,
      "name": "schema:E2/port_tight/bounds_summary.csv",
      "passed": true
    },
    {
      "details": 80,
      "name": "schema:E3/ablation.csv",
      "passed": true
    },
    {
      "details": 20,
      "name": "schema:E4/dse_progress.csv",
      "passed": true
    },
    {
      "details": 10,
      "name": "schema:E4/union_generation_budget.csv",
      "passed": true
    },
    {
      "details": 20,
      "name": "schema:E4/proof_status.csv",
      "passed": true
    },
    {
      "details": 17586,
      "name": "schema:E4/bootstrap_stability.csv",
      "passed": true
    },
    {
      "details": 56,
      "name": "schema:E4/heldout_main_table.csv",
      "passed": true
    },
    {
      "details": 28,
      "name": "schema:E4/heldout_main_table_pipelined.csv",
      "passed": true
    },
    {
      "details": 28,
      "name": "schema:E4/heldout_main_table_port_tight.csv",
      "passed": true
    },
    {
      "details": 448,
      "name": "schema:E4/breakdown.csv",
      "passed": true
    },
    {
      "details": 24,
      "name": "schema:E4/gates.csv",
      "passed": true
    },
    {
      "details": 32,
      "name": "schema:E4/selected_baseline_headroom.csv",
      "passed": true
    },
    {
      "details": 16,
      "name": "schema:E4/cross_bw.csv",
      "passed": true
    },
    {
      "details": 1440,
      "name": "schema:E4/synthetic/reverse_search.csv",
      "passed": true
    },
    {
      "details": 42,
      "name": "schema:E5/dispatch/compare.csv",
      "passed": true
    },
    {
      "details": 5670,
      "name": "schema:E5/dispatch/hbm_bytes.csv",
      "passed": true
    },
    {
      "details": 540,
      "name": "schema:E5/dispatch/regression.csv",
      "passed": true
    },
    {
      "details": 2520,
      "name": "schema:E5/dispatch/development_per_window.csv",
      "passed": true
    },
    {
      "details": 84,
      "name": "schema:E5/predictor/predictor_table.csv",
      "passed": true
    },
    {
      "details": 5,
      "name": "schema:E5/sobol/sobol_indices.csv",
      "passed": true
    },
    {
      "details": 1792,
      "name": "schema:E5/sobol/sobol_samples.csv",
      "passed": true
    },
    {
      "details": 0,
      "name": "schema:E5/sobol/flip_points.csv",
      "passed": true
    },
    {
      "details": {
        "tree": "c2e29a178010362d4bced173cf86217c74b0613b",
        "unchanged": true
      },
      "name": "round2_readonly",
      "passed": true
    },
    {
      "details": {
        "configurations": 42,
        "nonzero_differences": 0,
        "rows": 5670
      },
      "name": "E0_exact_reproduction_and_LB",
      "passed": true
    },
    {
      "details": 2754,
      "name": "E2_saved_bounds_and_reference_LB",
      "passed": true
    },
    {
      "details": {
        "E3": 1350,
        "E5_dispatch": 11340,
        "E5_old_runtime_setting": 1890,
        "E5_predictor": 11340
      },
      "name": "all_available_per_window_LB_and_repeat",
      "passed": true
    },
    {
      "details": {
        "archived_prediction_csv": {
          "lossless_hash_verified": true,
          "rows": 353976,
          "uncompressed_bytes": 103830785
        },
        "configuration_receipts": 344,
        "digest_pairs": 30050,
        "raw_receipts": 212
      },
      "name": "repeat_receipts_raw_hashes_and_raw_LB",
      "passed": true
    },
    {
      "details": {
        "CSV_configurations": 140,
        "raw_configurations": 140,
        "rows": 2520
      },
      "name": "dispatch_development_fixed_and_EFT_reference_repeats",
      "passed": true
    },
    {
      "details": {
        "certificates": 16180,
        "evaluated_points": 134400,
        "final_union_selection_certificates": 20,
        "main_family_certificates": 20,
        "main_union_actual_source_calls_counted_once": 367776,
        "open_regions": 582212,
        "proof_B_closed_families": 0,
        "simulator_calls_including_repeats": 5759136,
        "sobol_samples": 1792,
        "successful_points": 134388,
        "synthetic_points": 1440,
        "union_selection_certificates_no_new_calls": 20
      },
      "name": "all_available_search_witnesses_LB_and_certificate_scope",
      "passed": true
    },
    {
      "details": {
        "E4/synthetic/SYNTHETIC_PROTOCOL.json": 1440,
        "E5/sobol/SOBOL_PROTOCOL.json": 1792,
        "bootstrap_groups": 60
      },
      "name": "bootstrap_and_campaign_counts",
      "passed": true
    },
    {
      "details": {
        "recomputed_table_cells": 1536
      },
      "name": "published_latency_tables_from_window_values",
      "passed": true
    },
    {
      "details": {
        "assumed_heldout_monotonicity": false,
        "paired_rows": 32
      },
      "name": "new_baseline_headroom_from_same_protocol_windows",
      "passed": true
    },
    {
      "details": {
        "all_bitexact": true,
        "old_runtime_supplements_kept_separate": true,
        "raw_old_fixed_rechecked": true,
        "raw_protocols": 8,
        "rows": 540,
        "selected_runtime": {
          "large_first": true,
          "t_big": 4
        }
      },
      "name": "single_core_bitexact_regression",
      "passed": true
    },
    {
      "details": 20,
      "name": "frozen_main_hardware_isoresource",
      "passed": true
    },
    {
      "details": {
        "coverage": {
          "bootstrap_rows": 1986,
          "diagnostic_groups": 24,
          "group_candidate_rows": 334,
          "new_points": 175,
          "objective_rows": 662,
          "per_window_rows": 89370,
          "planned_groups": 20,
          "raw_checks": 187,
          "reused_points": 12,
          "unique_points": 187,
          "window_order_checks": 25245,
          "winner_rows": 72
        },
        "max_objective_error": 0.0,
        "selected_hardware_unchanged": true
      },
      "name": "development_frozen_heldout_robust_diagnostics",
      "passed": true
    }
  ],
  "complete_delivery": true,
  "errors": [],
  "inflight_scope": "phase-fluid rate times 65-cycle estimate, not a native discrete-credit occupancy trace",
  "lower_bound_checked_actual_values": {
    "E0": 5670,
    "E2": 2754,
    "E3": 1350,
    "E3/raw_physical": 1350,
    "E4/cross_bw_raw_physical": 4320,
    "E4_search": 91944,
    "E4_synthetic_search": 57600,
    "E4_union_selection": 105516,
    "E5/dispatch/raw_physical": 13230,
    "E5/predictor/raw_physical": 11340,
    "E5_dispatch": 11340,
    "E5_dispatch_development_raw": 5040,
    "E5_old_runtime_setting": 1890,
    "E5_predictor": 11340,
    "E5_sobol_search": 1290240
  },
  "lower_bound_checked_total": 1614924,
  "lower_bound_violation_count": 0,
  "lower_bound_violations": [],
  "missing_files": [],
  "partial_requested": false,
  "proof_scope": "Exact resource-assignment solver is not a joint temporal-scheduling optimum. Open hardware domains are reported, never closed by an optimal leaf alone.",
  "repeat_configuration_receipts": 344,
  "repeat_digest_pairs": 35090,
  "scope": "BF16 post-router phase-fluid analytical evidence; not native HBM, RTL, calibrated area, or full-model timing",
  "search_accounting": {
    "certificates": 16180,
    "evaluated_points": 134400,
    "main_union_actual_source_calls_counted_once": 367776,
    "open_regions": 582212,
    "proof_B_closed_families": 0,
    "simulator_calls_including_repeats": 5759136,
    "successful_points": 134388,
    "union_selection_certificates_no_new_calls": 20
  },
  "status": "passed"
}
```

模型是相位流体解析近似；共享池在事件边界分配有限字节窗口，并未模拟逐 DRAM 请求返回、真实 bank 地址、交叉开关或全部控制电路。端口与计算占用是重叠积分。当前生命周期和下界都不代表将来更改 compiler 融合后的架构。没有 RTL／面积／功耗综合，也没有完整模型每 token 计时。本轮留出集已被此前调试访问，不能当成全新盲测。

所有固定设计跨所有 batch 使用同一组硬件和已冻结 runtime 参数。第二轮结果只读引用，不改写；重复与执行命令、输入哈希见各阶段 receipts 和 README.md。

测试程序使用了经过逐位等价验证的运行加速：省去未输出的诊断序列化，缓存静态几何排序，以及将原整数 DFS 按相同遍历、剪枝和并列规则用 C 执行。CP-SAT 的约束和确定性工作预算、两次完整搜索及各窗口的两次物理回放均保留。这里的 C 整数遍历不是 native HBM／RTL 仿真或硬件校准；证据见 diagnostics/performance/ 和 native_enum_conformance 执行回执。
