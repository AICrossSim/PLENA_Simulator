# PLENA-MoE Supply-first v3 实施与评估报告

状态：**完整矩阵已完成**。主矩阵完成 **55012/55012** 个合法点、对应 **110024** 次已验证原始运行；已观察声明排除5777点，失败／不支持0点。矩阵是否完整以覆盖收据为准。

所有时间都是 Rust 有限资源离散事件分析模型的单个 MoE FFN 层时间，起点是路由结果已经就绪，终点是全部专家输出合并完成。Router执行、Attention和全模型生成时间未计入。1 GHz下1周期=1 ns，1,000,000周期=1 ms；主机运行秒数不等于硬件延迟。

## 实际组织及冻结比较对象

尺寸顺序统一为M×N×K。M6组织总主乘法器都是12,288，秩通道另计。共享HBM、落地池和激活存储；两个核有自己的操作数寄存器与有限累加空间。ISO固定任务指定的总供数端口，私有累加读写端口仍随核形状变化并明确计费。

| 组织 | 物理尺寸M×N×K | 数据流 |
|---|---|---|
| BL2 单核 | 6×4×512 | 可切换WS/IS，2上下文 |
| BL3 同构 | 3×4×512＋3×4×512 | 两核可切换WS/IS |
| BL4 特化大小核 | 4×4×512＋2×4×512 | 大核WS、小核IS |
| BL5 可切换大小核 | 4×4×512＋2×4×512 | 两核可切换WS/IS |

开发集12个窗口的比较对象冻结为M6=BL5、M8=M8_flex62，以后不逐测试窗口选择最快基线。下表是开发集选择证据，不能当作留出结论。

| 开发候选 | 层延迟几何平均ms | 选择 |
|---|---:|---|
| BL2 | 0.416760 | 同预算备选 |
| BL3 | 0.421445 | 同预算备选 |
| BL5 | 0.411560 | 冻结选择 |
| M8_flex53 | 0.420445 | 同预算备选 |
| M8_flex62 | 0.409020 | 冻结选择 |
| M8_homogeneous | 0.418905 | 同预算备选 |
| M8_single | 0.415046 | 同预算备选 |

## 真实留出结果：相同OP2、ISO端口

每行对同一组真实窗口求几何平均。比值小于1表示BL4更快，大于1表示冻结替代更快。只有四种组织全部完成才形成完整配对；T128属于单列压力测试，N4/N5主要比较T64/96。

| 来源 | Token数T | 已配对窗口 | 单核6 ms | 同构3+3 ms | 特化4+2 ms | 可切换4+2 ms | 特化/冻结替代 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 真实解码 | 2 | 27 | 0.252234 | 0.254428 | 0.246725 | 0.254870 | 0.9680 |
| 真实解码 | 4 | 27 | 0.422788 | 0.426092 | 0.415477 | 0.424545 | 0.9786 |
| 真实解码 | 8 | 27 | 0.536964 | 0.538100 | 0.521748 | 0.560217 | 0.9313 |
| 真实解码 | 16 | 27 | 0.767610 | 0.766189 | 0.701597 | 0.752907 | 0.9319 |
| 真实混合 | 64 | 9 | 1.781720 | 1.820799 | 1.607741 | 1.815039 | 0.8858 |
| 真实混合 | 96 | 9 | 2.185888 | 2.702498 | 2.469362 | 2.295167 | 1.0759 |
| 真实混合 | 128 | 9 | 2.598260 | 3.805055 | 2.712273 | 2.813449 | 0.9640 |

## 旧基线的有限容量扩展必须单独解释

BL0/BL1原有实现把整层专家输出保留到最后合并，真实T64/T96窗口超出其固定私有存储。新比较为这些原本无法执行的窗口加入通用、按容量选择最大合法Token块的外层串行执行器：整层X、最终Y和路由记录始终驻留原私有预算；块内运行原核函数，完全排空后复用暂存。原来能执行的B2/B4/B8/B16行为保持不变。子块X/Y是原地址的行切片，不增加隐含搬运或存储；跨块权重重新从HBM读取，实际字节和控制设置时间全部计入。因此它是明确披露的容量扩展基线，不能假装成原实现的一个无代价大批次运行。

| Token数T | 旧组织 | 窗口数 | 容量决定的块大小 | 平均块数 | 唯一权重MiB均值 | 额外重取MiB均值 | 设置ms均值 | 完整层ms几何平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | BL0 | 9 | 13–13 | 5.00 | 1008.33 | 2024.00 | 0.000870 | 28.726015 |
| 64 | BL1 | 9 | 15–15 | 5.00 | 1008.33 | 1947.00 | 0.000689 | 27.357161 |
| 96 | BL0 | 9 | 9–9 | 11.00 | 1048.67 | 4539.33 | 0.001446 | 50.904273 |
| 96 | BL1 | 9 | 11–11 | 9.00 | 1048.67 | 3963.67 | 0.001090 | 44.832080 |

该表只使用OP0真实混合留出的当前完整重复报告，保持单套256 B/周期HBM及256个信用。OP1旧基线的544信用属于额外返回区/标签的超预算诊断，另列在完整CSV；不能据此作严格等面积结论。CSV中的legacy_batch_*字段公开块大小、块数、重取、唯一字节及设置周期，每点原始报告保留完整子块排空和私有峰值核验。

## M0–M6交付证据

实现与测量完成，不代表论文门槛通过；负结果同样保留。

| 里程碑 | 当前完成状态 | 证据 |
|---|---|---|
| M0 旧架构计量 | 已有冻结诊断 | [m0_receipt.json](/tmp/plena-moe-supply-v3-numerical-artifacts-20261002/m0/m0_receipt.json) |
| M1 BF16供数 | 24735/24735合法点，均需两次原始报告 | [table_ablation_leave_one.csv](/tmp/plena-moe-supply-first-v3-output-20261002/table_ablation_leave_one.csv) |
| M2 真实权重量化 | diagnostic_non_accuracy_preserving | [frozen_format.json](/tmp/plena-moe-supply-first-v3-output-20261002/frozen_format.json) |
| M3 量化通路与补偿 | 1811/1811合法点，均需两次原始报告 | [claim_N2_compensation_evidence.csv](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/claim_N2_compensation_evidence.csv) |
| M4 数据流与组织 | 20936/20936合法点，均需两次原始报告 | [table_organization_paired.csv](/tmp/plena-moe-supply-first-v3-output-20261002/table_organization_paired.csv) |
| M5 Runtime | 1312/1312合法点，均需两次原始报告；真实LUT双重置序列已验证 | [table_policy_paired.csv](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/table_policy_paired.csv) |
| M6 留出与报告 | 完整留出完成 | [table_coverage.csv](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/table_coverage.csv) |

## N1–N5：按原阈值判定

| 主张 | 判定 | 实测值 | 原判定标准 |
|---|---|---|---|
| N1 量化节省是否兑现 | 不通过 | 单核旧供数 R 最大=0.7138；v3 解码 R 最小=0.9969；混合 R 最小=0.5020 | 旧单核≤0.40，v3各要求窗口≥0.85 |
| N2 秩通道补偿开销 | 不通过 | 630/630组配对；P1=不通过；P2=不通过 | 秩通道总核忙开销≤3%，T96层开销≤3%，乘法器≤3.2%；替代方案存在规定损失见证 |
| N3 供数机制 | 不通过 | OP0 B16 η最小=0.9934；OP1 η最小=0.8959 | η≥0.95／0.90；每个机制至少一处留一损失≥3% |
| N4 特化大小核组织 | 不通过 | 混合层时间比=0.9762；搬运比=0.9435；面积代理最大差=0.0247 | 对开发集冻结的最强替代：时间≤0.95，或搬运≤0.85且名义面积差≤5% |
| N5 运行时与动态秩 | 不通过 | IPD/joint时间比=0.9761，p95比=0.9854；数值条件=不通过 | 时间≤0.97或p95≤0.95；另须因果完整总体的等字节误差≤0.90或等误差因子字节≤0.80 |

N4名义判定保持原5%面积代理门槛；类型稳健性另报为 **不通过**。若名义通过但依赖RF/SRAM类型假设，不能声称无条件等面积硬件优势。N5调度时间判定为 **不通过**，数值资格必须另行完成，不能由时间推断。

### 供数机制的负结果也保留

| 留一机制 | 最大时间损失比例 | 最大η损失比例 | ≥3%门槛 |
|---|---:|---:|---|
| byte_pool | 1.2433 | 0.5542 | 通过 |
| credit_release | 0.0527 | 0.0500 | 通过 |
| inline_silu | 0.0365 | 0.0352 | 通过 |
| pipeline_supply | 14.6421 | 0.9361 | 通过 |
| prefetch_quota | 14.6421 | 0.9361 | 通过 |
| w_reuse | 10.1912 | 0.9106 | 通过 |
| wide_ports | 0.2290 | 0.1863 | 通过 |
| x_reuse | 1.8568 | 0.6500 | 通过 |

### 数值与补偿边界

实际物理格式：`{"factor_a": "mxint4", "factor_b": "bf16", "main_bits": 4, "rank_lanes": 8, "ranks": {"routed": [32, 32, 24], "shared": [32, 32, 48]}}`。最终资格状态：**diagnostic_non_accuracy_preserving**。P2使用低位宽MX权重和BF16 X/U/Z，factor_a是低秩矩阵A，不是主输入activation量化；不是W4A4。

硬件BF16 LUT完整Q3收据：已完成；N5数值结论：matching physical format and QERA-approx do not meet either full-population three-layer criterion。字节口径：CSV bytes are routed factors only; N5 equal-error factor-byte ratio adds fixed Shared factor bytes to both sides; main weights excluded。等字节后验子集、oracle等误差选择、未对齐实际字节的误差收益不能转为完整因果总体的通过。

N2使用相同位宽/因子格式及相同总wire字节，但padding在专家尾部真实传输，不保留每tile布局、A位置和消费次序，因此不是纯算术单变量隔离。开发集native-wire敏感性完成42/42点、84原始运行；见[native_wire_sensitivity.csv](/tmp/plena-moe-supply-first-v3-output-20261002/compensation_native/native_wire_sensitivity.csv)。B-MXINT4补充组具有独立数值条件，不能继承默认B-BF16资格。

M5真实LUT的六窗口因活动专家字节预算改变而每次重置λ0，没有跨窗口自学习收敛证据；单个真实窗口的temporal replay只验证相同预算的因果λ传递，不增加独立样本。实际预算超支、不同rank及负的时间收益均保留于causal_sequence.csv。

## 覆盖、可复现性与局限

最终binary SHA：`cb86a53a63bd15acbd027da4941f8fa496e1dfe4eb3f16e71f167ccf2aa057d5`。Timing signature包含实际编译器源码、输入、五项物理格式和runner；质量文件SHA独立。每个合法点的两次完整原始JSON必须字节相同，gzip解压后哈希仍校验。预注册提交：`073a26d31ce72954b50c72027f877295bf8c4170`。

全覆盖见[table_coverage.csv](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/table_coverage.csv)；精确原始运行和不可用组合见每点receipt/unavailable。存储迁移仅改变文件所在文件系统，不改变逻辑路径、输入、格式或阈值；[whole_output_storage_relocation_receipt.json](/tmp/plena-moe-supply-first-v3-output-20261002/whole_output_storage_relocation_receipt.json)包含文件SHA核验，[storage_resume_audit_v6.json](/tmp/plena-moe-supply-first-v3-output-20261002/storage_resume_audit_v6.json)记录中断报告保留。早期失败候选和ENOSPC尝试保留，不能混入当前通过的时序点。

真实混合留出27窗口来自只有3个独立prefill请求，并跨层/长度复用，不能报告为27个独立模型请求。混合窗口同时就绪的FFN集合不等于完整连续batching的到达过程。没有原生HBM/Ramulator、RTL频率、SRAM宏面积、功耗或整模型tokens/J实测；面积与能量只提供明确口径的相对代理。局部WOR/XOR广播端口和RF扇出没有综合证据，另作敏感性。

结论只采用上述已经完成的原阈值判定。若异构组织、Runtime或数值条件不通过，降低对应主张，保留实测数据和适用范围；不通过单独调参、更换测试配置或缩小矩阵制造胜出。

<!-- v10-output-only-protocol-disclosure:start -->
## 正确性修订与留出协议说明

原 v7 预注册提交后已经启动过留出测试；随后在合法组合中发现有限资源死锁。D032 明确记录这是留出启动后的正确性修订；原始失败、已完成点和授权文件均已保留，授权撤销后完成通用准入修复、独立回归和新的预注册修订。D033 又记录旧 BL0/BL1 的 T64/T96 大窗口原本超过有限私有存储，是原声明矩阵中的执行失败；v8b 的混合开发测试中止并保留，未获得新的留出授权。D034 记录 v9 混合开发集合法 T96/layer13/BL4/OP5/ISO/offload 点的连续 Z/U 分配死锁；完整 v9 开发流程因此失败，41 个隔离任务停止，原始失败、配置、负载、完成及在途结果均保留。v9 没有获得新的留出授权，也没有运行新的 v9 留出。修复要求取得 Current 位置前原子证明并预留真实连续 RF/Z/U 空间；不整理或搬移存活数据，不增加存储，也不提前释放存活数据。v10 使用同一组输入、硬件参数、设计空间及验收阈值，重新运行完整 55,012 个合法点，每点两次原始报告；不混用早期 binary 的时序数据，不把失败组合删除。D033 的 v9 容量扩展在 v10 保留：为原本无法执行的大窗口显式加入容量决定的串行 Token 分块：原核函数和预算不变，整层 X/Y/路由驻留，真实权重重取及设置成本均收费。D033 的历史验证中，已经可运行的小批次24个原始报告与旧版逐字节相同；该历史证据不能代替 v10 的新验证；这项容量扩展单独列在结果表，不能隐去基线执行方式的变化。原 v7 留出内容曾用于 D032 正确性排错；D034 来自开发集。这段历史仍须披露，本轮不能称为从未观察过的盲测。证据：[D032 正确性修订](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/postheldout_correctness_amendment_v8.json)、[D033 容量扩展](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/postheldout_capacity_amendment_v9.json)、[原 v7 授权](/tmp/plena-moe-supply-first-v3-output-20261002/heldout_authorization_v7_historical_before_correctness_amendment.json)、[中止与失败保留收据](/tmp/plena-moe-supply-first-v3-output-20261002/development_and_heldout_v7_aborted_receipt.json)、[D034 开发集正确性修订](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/postdevelopment_context_amendment_v10.json)、[保留的 v9 故障清单](/tmp/plena-moe-supply-first-v3-output-20261002/history/v9_mixed_legal_offload_failure_before_repair/MANIFEST.json)、[完整 v10 运行收据](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/final_pipeline_receipt.json)、[完整矩阵后的最终渲染收据](/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator/research/moe_dispatch/validation_v3/final_results_v10/generated_final_report_render_receipt.json)。

报告汇总在全部仿真完成后遇到一个类型错误：4,976 行未观测的可选 cold 指标在 CSV 中为空字符串，而冻结脚本对它进行数字比较。输出层恢复仅在加载内存中删除这个缺失字段，使原脚本使用原有缺省分支；50,036 个实际观测值和所有原始 CSV/JSON 均保留。未重跑仿真，未修改时序源码、数值、比较对象或 N1–N5 门槛。证据：[报告恢复收据](/tmp/plena-moe-supply-first-v3-output-20261002/generated_v10_reporting_continuation_receipt.json)。

图 `legacy_fit` 仅来自历史开发集 M0，不含留出数据。`wide_ports=false` 中的有效 X 服务宽度降低属于消融节流；安装的端口容量仍按原 `config.x_port` 收费，不能把有效宽度当作减少了硬件端口。

M6 与 M8 分别构成同一工作点内等主计算资源的比较组：M6 的主阵列乘法器（MAC能力）预算为 12,288，M8 为 16,384，秩通道另计。M8 相对 M6 的结果不能计为等主计算资源（iso-compute）收益。每个组织的实际供数／需求端口、RF/SRAM 物理预算和占用峰值分别报告；相同的名义存储字节容量不等于经过综合验证的等面积，面积结论仍按原有代理和类型敏感性限定。
<!-- v10-output-only-protocol-disclosure:end -->

