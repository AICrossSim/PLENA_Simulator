# 第1步证据与解释附录

本附录展开两页 `diagnosis.md` 的来源，不包含第2、3步实验。原文件均只读；完整函数路径、起始行、SHA256及可追溯提交见 [model_sources.json](results/model_sources.json)。

## 三条调用链与历史

1. **旧表**：第7轮 [legacy.py:12](../round7_64k_dualcore_dse/legacy.py) 的 `run` 导入 [scaling_g64/runtime.py:104](../scaling_g64/runtime.py) 的 `schedule`，它经 [plena65536_dedicated/model.py:16](../plena65536_dedicated/model.py) 的 `job_cycles` 调用 [plena65536_shared/timing.py:32](../plena65536_shared/timing.py) 的 `phase`。这些旧模块目前是未提交文件，Git无法证明其精确创建时间或第几轮编号。最早可定位的函数定义在未编号的 Shared-only 65,536 实验模块；随后被 Shared+MoE、专职派工、scaling_g64 阶段导入。不能凭文件修改时间伪造轮次。
2. **第7轮已有OS**：[round7/model.py:49](../round7_64k_dualcore_dse/model.py) 的 `gemm` 直接调用 [round5/model.py:28](../round5_largebatch_compute/model.py) 的 `projection`，每个 Gate、Up、Down 都设置 `paired=False`。该函数首次已提交版本为 `b095a7de`（第五轮）；公式前身是 [round4/model.py:211](../round4_compute_first/model.py) 的 `_group_cycles/_projection`，再前为 [round2/model.py:149/217](../round2/model.py)。第2轮的提交 `2a68dfdb` 明确将 `dot_latency=2+dotstagecycles*log2(PK)` 标为假设。更早的 [geometry3d/compute.py:135/169](../geometry3d/compute.py) 在提交 `1fcc764a` 中已声明树级数敏感性与8输出记录，并明确未经物理验证。
3. **ceil**：第7轮 `model.py::gemm(model="issue_formula")` 的三维向上取整乘积分支。本实现来自第7轮任务书，不调用旧 `phase` 或 R5 `projection`。更早代码存在类似发射次数计算，但不能将其认作同一整层耗时函数。

**旧重放与第7轮OS绝不是同一个函数。** R7 的 `plena_os` 名称不意味着它是 PLENA 原生模拟器路径；复用已有脚本只证明实现连续性，不证明时序正确性。

## G 的含义与逐GEMM记账

- 旧 `scaling_g64/geometry.py::Core.from_shape` 设置 `BLEN=PM`、`MLEN=PM*G`。G 是子阵列组数。每组 `PM` 个输入时钟完成一个 MLEN 宽的逻辑K段，因此 feed=`PM*ceil(K/(PM*G))`。同一 `4×256×64` 在旧模型的逻辑K段宽是256，在R7的直接点积模型里是64。
- R7 将 G 直接作为空间点积宽度PK。G=64有6层二叉归约树，**64不是64级树深度**。乘法器总量相同，并不能消除两种组织的流水线、K尾块和累计方式差别。
- 旧模型一个独立phase的归约排空记作 `log2(G)`，其余为 `PM+11` 以及K分块对齐增量；模型没有显式逐K累加依赖等待。`A=0` 表示未单独收费，并不证明硬件不存在该代价。
- 新OS模型只有合并延迟 `L=3+2log2(G)`，并没有真实的“归约等待/累加等待”两套计数器。诊断将总量严格分解为 `C + A + R + O`：`A=(nk−1)*max(0,L−q)`；末次排空 `L−1` 按公式约定拆为 `R=2log2(G)` 与 `O=2`。这是一种代数记账，不能称为实测物理归因。
- 每组q个输出按K段轮转，回到同一输出时要等上次提交。此规则能用独立事件重放复现。它刻画的是“固定小组＋固定反馈延迟”的特定实现；即便真的只有8个输出记录，也不必采用按组全部完成后再替换的策略。
- 第7轮按任务书对三个GEMM分开调用；R5原来的 `cost` 会对Gate/Up设置 `paired=True`。旧 `job_cycles` 同样是 Gate+Up 一次 `phase(factor=2)` 加 Down 一次 `phase(factor=1)`。融合少收一次排空，也改变输出独立性；不能把旧单GEMM表的三个独立phase简单相加当旧专家耗时。

## 硬件依据及其边界

- [PLENA v3 §III-B，图5/6](https://arxiv.org/html/2509.09505v3#S3.SS2) 描述小方形子阵列、PE内OS部分和、跨子阵列的M_SUM与连续GEMM tile流水化；没有规定本轮的8个输出记录或每个64宽K段必须等待15周期。不能由“存在加法树”推导“每个K段要串行等待整棵树”。加法树的完成延迟和能否每周期接收新数据也不同。
- [customISA_lib.json:2](../../../analytic_models/performance/customISA_lib.json) 给出 `M_MM alone=2*BLEN+9`、`pipelined=BLEN`。这为旧模型的feed/drain部分提供仓库来源，但文件没有完整规定旧模型采用的M_SUM、2周期提交、影子部分和及跨tile全部重叠。**这些扩展假设没有硬件校准依据。**
- [Rust matrix_machine.rs:143](../../../transactional_emulator/src/matrix_machine.rs) 原 `mm_with_view` 另收费 `SYSTOLIC_PROCESSING_OVERHEAD+MLEN`，`mm_wo:512` 还有写回服务。R7没有调用该路径，旧模型也不是这个路径的周期重放。
- [Rust matrix_service.rs:1/120](../../../transactional_emulator/src/matrix_service.rs) 的候选矩阵服务按MAC反馈、树延迟和端口分项计时，但源码开头明确是候选实现契约、未经综合。它与R7的8记录、15周期循环不是同一个模型，不能用其存在替代校准。
- 读取了独立工作树 `/scratch/shared/mcl123/plena/worktrees/pr_candidates/PLENA_RTL_fake_hbm_capacity/` 下的 `src/basic_components/systolic_gemm_fp/rtl/fp_sum_across_sa.sv`（每输出跨阵列树）与 `fp_default_pe.sv`（PE局部累加），以及 `src/basic_components/gemv/rtl/fp_adder_tree.sv`（可配置流水）。它们支持这些结构存在，**不支持R7固定8记录/两周期每级/15周期回访限制**。该工作树不是原论文冻结RTL，未在本轮进行BF16参数实例化或波形校准，不能据此声称原芯片的精确时序。

## B2为什么差两倍

全部27个B2窗口都有 `Me≤2`，在4行单核上每个F1408任务都只用一个M波次。

- ceil：Gate192 + Up192 + Down176 =560周期。
- 第7轮OS：Gate485 + Up485 + Down337 =1307周期。每专家多747周期，其中累加依赖705、归约排空36、其他6。
- 旧模型：Gate+Up phase405 + Down213 =618周期；F2816合并Shared则725+373=1098周期；单核启动/上下文暴露共5周期。
- 平均活跃路由专家数 `287/27=10.629629…`。于是R7 ceil平均 `(287/27+2)*560=7072.592593` 周期；OS平均 `(287/27+2)*1307=16506.925926` 周期；旧平均 `618*287/27+1098+5=7672.111111` 周期。旧表7.606µs取的是27个窗口时间的GM=7.605794µs，不能当AM比较。
- 旧AM到新OS AM的逐项桥接：旧总计7.672111 → 去控制7.667111 → 换新OS函数但保留F2816 Shared16.497926 → 拆为2个Shared16.506926µs。因此差异主要来自耗时函数与执行假设，不是随机波动或Shared拆分。
- 旧AM到ceil AM：7.672111 → 去控制7.667111 → ceil、合并Shared7.008593 → 拆Shared7.072593µs。全部135窗的路由集合一致核验通过。

## 数据范围与暂停点

纯decode文件 `/scratch/shared/mcl123/plena/final_artifacts/moe_regime_20261005_v1/payload/inputs/heldout.json` 有108窗，覆盖BFCL/GPQA/SWE、层2/13/26、步3/7/11、B2/4/8/16各27窗。每窗保存B个不同请求在同一解码步上的路由。相邻batch/步骤可能复用请求，后续分层随机留出不能自动视为请求级独立泛化。

同目录 `mixed_heldout.json` 有27窗，B64/96/128各9、层相同。每窗为对应层/数据集的B16 decode步7，加同一请求连续prefill前缀48/80/112。18窗的原始provenance含 `capture_prefix_smoke=true`；本轮保留标记，没有生成新路由。逐窗详情沿用R7归档，汇总见 `results/trace_inventory.csv`。

**后续口径已获用户确认。** 第2、3步以ceil为纯计算几何研究的理想主模型，以原8记录OS为未校准敏感性，并同时评估PN≤256与PN≤512；这不承诺真实硬件可达到ceil速度。完整结果见[results.md](results.md)，本文件保留第1步证据。
