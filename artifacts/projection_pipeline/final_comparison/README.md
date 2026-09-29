# Projection 与递推子层完整对照：2026-09-29

本轮完成按投影算子选择分块的编译调度，并重跑 Mamba/KDA × B1/2/4/8/16 的四组对照。所有时间都是 **BF16 权重、1 GHz 假设下，一个 batch 前进一步 token 的 analytical 预测**。边界为输入归一化至输出投影，包含卷积、系数生产和递推；不含外层 residual、FFN/MoE，也不是 Nemotron/Kimi 整模型时间。

## 比较对象

| 列 | 实际执行方式 |
| --- | --- |
| traditional：旧工程映射 | resident M_MV、无权重重放扩展，递推用普通 BF16 Vector 指令；已有合法系数缓存 |
| native_reference：同算术投影参考 | 旧 resident M_MV，但换成与最新方案相同的 native 递推 |
| previous：上一版优化 | 权重重放、紧凑输入、最多四请求 M_MM.P；单归约组、单 N32 panel；native 递推 |
| latest：最新候选 | 保留上述复用；四个固定归约组；每个投影选择 1/2/4/8 个 N32 panel；native 递推 |

**traditional 仍使用共同平台的 Matrix-view、softplus 等服务，不能称为未经修改或最优的原版 PLENA。** 其逐指令 BF16 更新与最新 FP32 中间值更新不同。因此 old→new 同时含映射、数据通路和算术改变；native_reference→latest 保持递推算术一致，适合隔离投影改进。

共同条件：1 MiB / 64-bank Matrix SRAM，256 KiB Vector SRAM，4096 个 Matrix 乘法器，16 个 HBM2 controller / 32 GiB / 256 GB/s，32 读与 32 写 DMA credits。投影最多用 58 个 Vector 行，另 6 行维持统一预留；本轮没有回收 BF16 情况下闲置的 codec 行。所有阶段串行退休，未免费加入投影/递推重叠。每阶段已含 DMA，不能再重复加总。

## 子层总时间

单位 ms；加速比为分子对应方案时间除以最新时间。

| 模型 | B | 旧映射 | 同算术投影参考 | 上一优化版 | 最新 | 旧/新 | 上版/新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mamba | 1 | 3.211801 | 3.063999 | 2.909800 | 2.248706 | 1.4283× | 1.2940× |
| mamba | 2 | 4.788285 | 4.494757 | 3.047989 | 2.434093 | 1.9672× | 1.2522× |
| mamba | 4 | 8.277965 | 7.689085 | 3.378375 | 2.742501 | 3.0184× | 1.2319× |
| mamba | 8 | 16.869521 | 15.765859 | 5.297095 | 4.017895 | 4.1986× | 1.3184× |
| mamba | 16 | 33.377727 | 31.023700 | 9.155280 | 6.648544 | 5.0203× | 1.3770× |
| kda | 1 | 37.178647 | 34.266319 | 32.034199 | 25.016055 | 1.4862× | 1.2805× |
| kda | 2 | 56.500550 | 50.700886 | 33.324016 | 26.335542 | 2.1454× | 1.2654× |
| kda | 4 | 105.513435 | 93.852496 | 35.976288 | 29.045908 | 3.6326× | 1.2386× |
| kda | 8 | 204.551970 | 181.979956 | 55.509874 | 41.660692 | 4.9100× | 1.3324× |
| kda | 16 | 403.079034 | 356.450631 | 101.932709 | 68.855200 | 5.8540× | 1.4804× |

## 各阶段时间

单元格为 **旧工程映射 → 最新候选**，单位 ms。细分到 q/k/v、decay 两级投影等算子的原始周期、六项独立成本及 HBM 字节见 `operators.csv`；四组聚合对照及逐阶段比值见 `stages.csv`。

### mamba

| 阶段 | B1 | B2 | B4 | B8 | B16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 输入与门控投影 | 2.158004 → 1.570232 | 3.150703 → 1.658862 | 5.301225 → 1.797833 | 10.916416 → 2.531566 | 21.376772 → 4.079853 |
| 卷积 | 0.004655 → 0.004545 | 0.008581 → 0.010064 | 0.017911 → 0.018159 | 0.034837 → 0.033037 | 0.067317 → 0.067169 |
| 门控与系数计算 | 0.002894 → 0.003330 | 0.006420 → 0.005997 | 0.012448 → 0.012142 | 0.024620 → 0.024497 | 0.049093 → 0.049303 |
| 系数排列/准备 | 0.000000 → 0.011830 | 0.000000 → 0.024276 | 0.000000 → 0.047886 | 0.000000 → 0.095985 | 0.000000 → 0.191308 |
| 递推 | 0.199085 → 0.041142 | 0.400123 → 0.080517 | 0.799166 → 0.163385 | 1.597993 → 0.323507 | 3.196478 → 0.650936 |
| 归一化 | 0.002602 → 0.002519 | 0.005872 → 0.005854 | 0.012383 → 0.011957 | 0.025101 → 0.025063 | 0.050625 → 0.050758 |
| 输出投影 | 0.844561 → 0.615108 | 1.216586 → 0.648523 | 2.134832 → 0.691139 | 4.270554 → 0.984240 | 8.637442 → 1.559217 |

### kda

| 阶段 | B1 | B2 | B4 | B8 | B16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 输入与门控投影 | 27.278851 → 19.879871 | 39.690392 → 20.734173 | 73.894013 → 22.552532 | 143.430748 → 31.805506 | 282.521259 → 52.015604 |
| 卷积 | 0.024262 → 0.023890 | 0.049309 → 0.047591 | 0.094680 → 0.095860 | 0.189380 → 0.189839 | 0.376309 → 0.382753 |
| 门控与系数计算 | 0.011699 → 0.010636 | 0.023416 → 0.023098 | 0.044312 → 0.045004 | 0.091309 → 0.090038 | 0.181091 → 0.179538 |
| 系数排列/准备 | 0.000000 → 0.009167 | 0.000000 → 0.017913 | 0.000000 → 0.036169 | 0.000000 → 0.070534 | 0.000000 → 0.141426 |
| 递推 | 3.067399 → 0.147467 | 6.142610 → 0.295686 | 12.292948 → 0.592296 | 24.580487 → 1.188498 | 49.163240 → 2.371378 |
| 归一化 | 0.051861 → 0.052766 | 0.104585 → 0.106519 | 0.210562 → 0.213482 | 0.420329 → 0.434324 | 0.846305 → 0.863442 |
| 输出投影 | 6.744575 → 4.892258 | 10.490238 → 5.110562 | 18.976920 → 5.510565 | 35.839717 → 7.881953 | 69.990830 → 12.901059 |

旧方案的部分按行系数供应包含在递推阶段，新方案仍有独立的 dt/skip/beta 准备；“旧系数排列为 0”不表示旧方案无供应成本。比较完整递推路径时应合计系数排列与递推。其他阶段算术未变时的小幅波动来自真实地址时间线、DRAM 行/刷新状态和阶段起始时间变化，不能算成额外机制收益。

## 本轮优化与剩余瓶颈

新增策略只改编译选择：先生成四套合法程序，按同名投影选择局部低成本分块，再生成并重新计费完整混合程序。若整个程序不优于最快统一分块，回退。没有直接相加各阶段最低时间，也没有新增寄存器、ISA 或运行时调度器。

| 模型 | B | 实际选择 | 相对此前最快统一分块的额外收益 |
| --- | ---: | --- | ---: |
| mamba | 1 | {"input_projection":4,"output_projection":1} | 0.2284% |
| mamba | 2 | 全部 N8 | 0.0000% |
| mamba | 4 | {"input_projection":4,"output_projection":1} | 0.2844% |
| mamba | 8 | 全部 N1 | 0.0000% |
| mamba | 16 | 全部 N1 | 0.0000% |
| kda | 1 | {"b_projection":1,"f_a_projection":1,"f_b_projection":8,"g_projection":1,"k_projection":1,"output_projection":1,"q_projection":1,"v_projection":1} | 0.0051% |
| kda | 2 | {"b_projection":1,"f_a_projection":1,"f_b_projection":4,"g_projection":1,"k_projection":1,"output_projection":1,"q_projection":1,"v_projection":1} | 0.0124% |
| kda | 4 | {"b_projection":1,"f_a_projection":1,"f_b_projection":8,"g_projection":1,"k_projection":1,"output_projection":1,"q_projection":1,"v_projection":1} | 0.0134% |
| kda | 8 | 全部 N1 | 0.0000% |
| kda | 16 | {"b_projection":4,"f_a_projection":4,"f_b_projection":1,"g_projection":8,"k_projection":8,"output_projection":8,"q_projection":8,"v_projection":8} | 0.0057% |

这是有限设计空间内的优化，额外收益很小，不是全局最优证明。此前权重重放、批量映射与固定归约分组的收益不能重复算成本轮编译选择贡献。

| 模型 | B | 最新投影时间中 DMA 占比 | 子层 HBM 读+写 MiB |
| --- | ---: | ---: | ---: |
| mamba | 1 | 62.91% | 76.391 |
| mamba | 2 | 60.88% | 78.954 |
| mamba | 4 | 55.86% | 84.080 |
| mamba | 8 | 38.95% | 94.332 |
| mamba | 16 | 24.88% | 114.836 |
| kda | 1 | 63.80% | 855.313 |
| kda | 2 | 61.30% | 864.563 |
| kda | 4 | 56.61% | 883.064 |
| kda | 8 | 40.07% | 923.550 |
| kda | 16 | 27.59% | 1020.959 |

B1 的投影仍有约 63–64% 时间用于 DMA，但这不等于已打满 HBM 峰值带宽；小请求、完成依赖和片上搬运都会限制有效速率。未来值得单独验证的是长 K 部分和驻留、合法的更粗 DMA、按生命周期回收 codec 工作区，以及投影到递推的有限缓冲交接。它们尚未进入本表，不能提前扣除周期。

## 当前架构与 novelty 的边界

1. Compiler 安排请求私有 state 和共享权重；按具体投影选择分块。权重跨请求复用、输入跨输出 panel 复用，部分和保留在明确的执行/舍入边界。这里不是三种任意 WS/IS/OS 硬件模式的切换。
2. Matrix 路径保留现有乘法器总数；K256 工作点采用四个固定归约分组，每组处理不同输出，N32 从八波变两波。增加 128 B 根结果保持，以及未综合的选通/广播/标记逻辑。原有候选投影暂存为 16 KiB 权重、4 KiB 行传输、2 KiB 有效输入、256 B 输出，合计 22.25 KiB；四分组后为 22.375 KiB 数据载荷，不等于面积。
3. Recurrent 路径复用固定对角 Matrix SRAM：view 地址/lane 对齐，紧凑系数选通广播，L_TILE 有界控制，256 更新 lane（II=2，延迟6）。state/系数为 BF16，更新中间为 FP32、RN 提交 BF16；dot 使用已有 Vector SRAM 的 BF16 树，不增加 FP32 dot-context SRAM。native 供数/结果数据暂存共 12 KiB，控制与 tags 另计。
4. Matrix 与 recurrent 共用 SRAM 容量和端口，按当前程序分时执行。已经计入有限供数和读写服务；完整仲裁/模式 RTL、同时执行和物理时序尚未验收。

适合继续验证的论文主张是：**在同一有限存储平台上，使 batch 共享的投影数据流与请求私有的递推数据流高效衔接，避免昂贵系数物化，并以可控供数/缓冲完成 state 更新。** 固定斜存本来就有；通用 tiling、权重重用、分组归约或支持 hybrid 模型各自不能单独认定为新颖贡献。原生供数访问机制、完整子层收益及明确资源代价需共同支撑主张。

## 验证、出处与复现

- 本轮五个机器码连接案例覆盖 B1/2/4/8/16、不同请求、K/N 尾块、两种连续分块和 B16 缓存压力。前一投影的实际输出进入后一投影；10,974 个有效输出值与独立声明算术参考精确一致。issue、scalar、SRAM、算术、依赖、DMA、total 全部一致。不是长链任务质量实验。
- Python 全套 282 passed / 9 skipped。为旧 KDA 大 batch 提高分析器指令保护上限后，受影响测试 75 passed / 6 skipped；硬件计费规则不变。40 个正式点均使用同一份最终源码哈希重跑。
- Rust 核心本轮没有变化；上一轮 329 workspace tests、fmt、clippy 的证据保留在 `../projection_study/`。本轮实际调用该版本的机器码运行器做上述新连接验证。
- Compiler pin：`0b2f549ac9cebeb3f328b65fc8ea6f36e1c1be49`。逐文件源码、资源 profile、机器码/地址流/内存后端 SHA 和本轮机器验证在 `evidence.json`。输出文件 SHA 在 `manifest.json`。
- Python 与 Rust 共享部分服务契约和 Ramulator，精确一致不单独证明集成电路可达 1 GHz。没有新增 GPU、全模型、NVFP4 运行时解码、PPA 或能耗结果。

使用 [工程复现环境](../../../doc/l_tile_projection.md)，保留 Nix 的原始 PYTHONPATH；大文件放 RUN_ROOT，以下四个输出目录必须不存在：

```sh
python -m analytic_models.performance.ltile_dma --prepare "$RUN_ROOT/memory16" --controllers 16
python -m analytic_models.performance.projection_campaign --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/traditional" --stages baseline --control old_isa --workers 2
python -m analytic_models.performance.projection_campaign --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/native_reference" --stages baseline --workers 2
python -m analytic_models.performance.projection_campaign --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/previous" --stages batch --workers 2
python -m analytic_models.performance.projection_campaign --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/latest" --stages batch --segments 4 --tune-panels --workers 4
python -m transactional_emulator.testbench.models.unified_service_test --only mixed --runtime "$CARGO_TARGET_DIR/release/transactional_emulator" --memory-root "$RUN_ROOT/memory16" --output "$RUN_ROOT/machine-mixed"
```

剩余门槛：最优合法原版 PLENA 对照；当前最终算术长期精度；运行时压缩权重；完整模型/容量/路由组合；有限交接数据通路集成；面积与时序验证。它们不改变已完成子层实验的边界，也不能由本表代替。
