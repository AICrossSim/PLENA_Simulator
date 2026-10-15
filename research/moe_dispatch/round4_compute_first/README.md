# 第四轮：先计算DSE，再反推供数

分支：`research/moe-supply-first-v3`。本目录独立保存代码与结果；第二、三轮已提交文件只读。

## 物理预算、出处与边界

乘法器预算改为 **65,536**。PLENA v3 §V-E 指定 BLEN=32、MLEN=2048、VLEN=2048；§III-B/Fig.6 描述 MLEN/BLEN 个 BLEN×BLEN 子阵列，因此为 `(2048/32)×32×32=65,536`。来源：<https://arxiv.org/html/2509.09505v3>。论文主系统使用 MX 低精度配置；本实验按任务书用 BF16/FP32，相同乘法器数量不表示相同芯片面积或功耗。历史 `6×4×512=12,288` 不是论文主系统配置。

| 参数 | 统一限制 | 依据 |
|---|---|---|
| PM | 所有整数1–16 | 任务书指定的研究假设，未综合 |
| PN | 所有整数1–256；敏感性全部核放宽至512 | 研究假设，未用不同规则限制基线 |
| PK | 32/64/128/256/512 | 研究假设；沿第三轮空间归约深度展开 |
| 乘法器 | 所有组织严格合计65,536 | 论文主系统乘法器数推导 |
| 时钟 | 1 GHz | 模型假设；无BF16物理时序签核 |
| dot延迟 | `2+2*log2(PK)`，提交1周期，发射间隔1周期 | 沿用第三轮假设；无新综合验证 |
| OS输出上下文 | 每核8项 | 沿用第三轮已安装的上下文数；SRAM容量放宽不增加发射器 |
| vector | 总计64元素/周期，按MAC数整数划分 | 三种组织预算一致；不会每核复制64路 |

本轮 **PM×PN×PK 表示直接空间点积阵列**，乘积就是乘法器数，PK是每次发射的归约维宽度。它不等于此前报告的 `PM×PN×G`，G=MLEN/BLEN。只继承原论文乘法器预算，并沿用第三轮广义BF16模型；不声称所有搜索点都是原论文原生可配置硬件。原论文默认BLEN32超出本轮PM≤16的搜索限制，未作为本轮最佳单核。

严格整数比例：1:1=32768+32768、3:1=49152+16384、7:1=57344+8192。2:1和5:1分别要求总数被3和6整除，**本轮无合法精确点**；不做取整伪装。1:1异构要求形状不同。同构2/4核的形状和数据流策略相同。

## 工作负载、选择及计算契约

使用第二轮冻结的18开发+135留出窗口；首次读取由原冻结输入加载器逐文件验证SHA256，完整数据封装在本目录的inputs_snapshot.json.gz；后续读取验证INPUT_MANIFEST.json中的解压内容哈希。包括Shared（H2048、F2816、Me=Batch）和所有激活的Routed（F1408，64选6）；不是只测Shared。每个窗口总路由Me=6B。Router本身未计时，权重只用形状/地址，没有完整模型准确率评测。

选择目标为**18开发窗口的时间几何平均**；冻结后在135留出窗口评估。主表按batch及所有135窗口GM，未用7-batch等权GM替代主口径。PN≤256共28,228个、PN≤512共32,300个几何/数据流候选，所有几何×每核OS/WS/IS/PER_TASK组合穷举；每个候选在开发集执行两遍并核对。最优指给定域、成本模型和固定调度的最优，非全局任务分配最优或PPA最优。留出集为历史已有研究数据，不是首次盲测。

模式B：HBM/SRAM容量/端口无限，零供数延迟。使用第三轮 `_projection` 的归约依赖时序（OS/WS/IS有不同的循环及驻留），所有核使用同一服务模型。每个专家的Gate+Up及Down顺序执行，SiLU/融合/输出向量服务有64路总预算。任务按Shared/Me>3优先、预估时长降序，再按最早完成时间派工；不计派工控制开销，不拆分专家。Tc包含向量服务，因此不是仅FLOPs/峰值MAC的理想除法。

模式A：固定模式B的任务归属与核内顺序，仅换成真实SRAM容量和端口；HBM仍无限且零延迟。有限Z导致按token分块、OS权重重读、IS部分和溢出均计入。此对照用于隔离片上资源影响。

E3系统：有限HBM、64周期响应、真实SRAM和`fixed`在线容量感知派工全部开启，动态任务归属可能与B/A不同。每核Current+Next，8项待选描述符，t_big=3，名义成本估计；不训练新的预测器。32B请求信用数取 `ceil(BW*65/32)`，避免把256/1229 GB/s悄悄限制成126 GB/s。

`model.py`、`runtime.py` 是第三轮冻结引擎的本目录副本，扩展公共形状范围、4核和按任务切换，不修改旧结果。它们是**带有限资源和共享HBM的事件驱动phase-fluid解析系统仿真**，不是逐请求Ramulator、原生Rust硬件校准或RTL物理测量。

## 数据流、内存和带宽账本

OS在输出上下文中跨K累加；WS在M波次间复用权重；IS在N列间复用输入。PER_TASK按已知Me/H/F及冻结硬件成本选择一个整专家数据流，各组织都可用。无限SRAM选择最小计算服务时间，平局选择较少片上流量；真实SRAM选择有限资源名义服务时间，平局选择较少HBM字节。该规则不按留出窗口调整参数；切换硬件面积和额外配置延迟尚未综合。

模式B的HBM字节=每个激活专家三张BF16权重各一次+全层BF16输入/FP32输出（6BH字节）；不以padding充当有效数据。外部激活只读入一次，在全局X中共享。模式A/E3增加实际循环重读与中间结果溢出。E3显式在层边界计入全局输入/输出的传输及启动延迟。

B存储需求按实际派工时间重叠计算**任务保留缓冲上界**：每个任务在运行期间保留其各类缓冲峰值。OS/IS需要跨M重访权重时保留整投影；WS保留流式tile；X/部分和/Z分别列出，再加全局X、FP32输出及40KiB控制/路由/入口。是可实施的保留方案需求，**不是最小容量证明**；四种类别峰值可能不在同一时刻，不能把各自最大值简单相加冒充瞬时总峰值。每核类别峰值已单独输出。

端口同时报告三种量：整个窗口平均、phase-fluid峰值保留包络、未平滑的每发射接口峰值。报告中的`port_peak`是给定phase-fluid模型的充分供数包络，不能冒充真实单周期SRAM burst测量；`port_issue_peak`列另给W/X/部分和发射接口需求。有限端口值为W1024、X384、Acc192 B/周期（总bank数64/24/12，各16B/周期）。

A/E3严格保持2,158,592B：固定全局部分1588KiB，可划分W/X/Acc/Z共532KiB。每核先保留至少一个W tile、X tile、Gate/Up部分和上下文、6KiB Z；余量按共同的1/1/2/8规则分配。各类bank在核间等分，向量lane按MAC分配。所有策略一致，不按heldout或异构另调。它是冻结的建设性分配规则，**未对SRAM切分另做DSE**。选择的B设计全部能容纳最小执行tile；其理想缓冲需求按真实容量截断后回放。

带宽单位GB/s采用十进制；MiB=2^20B；1GHz时1B/周期=1GB/s。HBM2模型256、HBM3 819、HBM3E 1229GB/s按任务书作为标称假设。8×1229=9832，主扫描保留任务书9830这一近似档位。≤8堆栈只表示通过指定带宽数量判据，不表示封装、功耗和容量已证明可行。

E3 `max(Tc,bytes/BW)` 是固定计算投影下的理想重叠估计。它忽略启动/资源争用；由于E3在线派工可改变归属，不能把它声称为所有调度的数学下界。系统结果只引用E3完整事件仿真，所有超过5%的差异逐窗口列出。

## 复现

仓库根目录运行，Python环境 `/tmp/plena-round2-venv/bin/python`。所有命令建议加 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp`。

```sh
python -m research.moe_dispatch.round4_compute_first.search
python -m research.moe_dispatch.round4_compute_first.evaluate --stage E1 --jobs 4
python -m research.moe_dispatch.round4_compute_first.requirements E2
python -m research.moe_dispatch.round4_compute_first.evaluate --stage E3 --jobs 4
python -m research.moe_dispatch.round4_compute_first.requirements E3
python -m research.moe_dispatch.round4_compute_first.crossover
python -m research.moe_dispatch.round4_compute_first.report
python -m pytest research/moe_dispatch/round4_compute_first/test_round4.py -q
python -m research.moe_dispatch.round4_compute_first.validate
```

提交账本见 `COMMITS.json`（最终归档时写入各阶段真实提交号）。运行凭据/输入与源码哈希见 `PROVENANCE.json`；每点重复凭据见E1/E3的`repeat_checks.json`和DSE表。主表同构基线在开发集选择；E4门槛与系统校准门槛同时要求超过单核、同构2核、同构4核，避免只选较慢的同构作对照；未完成独立校准一律不宣布“架构胜出”。
