# MoE 大小核 DMA：可执行实现与证据边界

日期：2026-09-05。设计依据：`moe_hbm_dma_architecture_v1_zh.md`。
目标是完成 Rust 数值执行中的权重搬运优化，并检验收益；本轮没有实现 HBM PHY、RTL 或完整模型运行时。
正式结果入口：`/scratch/shared/mcl123/plena/outputs/moe_dma_implementation_20260905/RESULT_ZH.md`。

## 已实现的数据通路

```
专家任务 → 每核 2/3/4 个已预留容量的权重槽 → element/scale 地址与复制片段
       → 有限的逻辑行额度 → 4-bank 查询 → 64B 行 MSHR / 32B sector 状态
       → 256 项原生请求跟踪池 → 每 HBM 输出通道独立提交队列
       → 同一个 Ramulator HBM2 → 8-bank 返回/复制 → packed element + scale
       → 原有共享向量单元执行真实 MX 解码 → BF16 normal SRAM → 计算核
```

1. **后端校准。** C API v2 从实际加载的库读取 `get_tx_bytes()`，验证 HBM2 原生事务为 32B。旧 wrapper 的 16B 拆分已修正。输出包括 native 库路径/SHA256、通道计数、接受/拒绝计数、原生完成统计。每次只由一个 ticker 推进模型；提交调用仍由同一个互斥锁保护，不能同时调用非线程安全后端。
2. **有界提交。** 两种策略均限制每输出通道每核心周期最多尝试一次提交，原生请求池 256 项。`global_fifo` 保留全局队首阻塞对照；`per_channel` 使某个通道的拒绝不挡住其他通道。当前 preset 是 8 个 HBM12 controller、CacheLineInterleave、32B 粒度，通道地址位为 `[7:5]`；并不把该配置自动等同于任意实体 HBM 的通道/伪通道数量。
3. **按需 sector 与在途合并。** 根据每个复制片段涉及的字节选择低/高 32B sector。MSHR 以只读权重的 64B 行地址为键；晚到的消费者可以加入已经请求的 sector，也可以补请求另一 sector。所有消费者完成复制后删除条目；不会将全局 FIFO 数据 cache 的容量/命中率偷偷引入结果。一个 executor/run 只有一个不可变权重映像，所以 run 作用域承担 epoch；本轮不支持运行中修改权重。
4. **SRAM 与端口计费。** 4 个查询 bank，各被一次查询占用 2 个周期（保守 II=2，原提案 latency=2 / II=1 的流水查询尚未建模）；8 个复制 bank，每 bank 每周期最多 32B。重复消费者各付复制代价。element 与 scale 都存入已预留的 packed 权重槽；全部需要的数据返回并完成共享向量解码后，才把 BF16 tile 交给计算核。没有单独增加免费解码器。
5. **更深的有限预取。** 初始填充整个配置窗口；每退休一个 tile 才复用一个槽。每个 tile 在全部 M 行消费完成前保持占用，维持专家权重复用与 K 累加次序。4-slot 异构配置把原有 64KiB 权重 SRAM 分成大核 48KiB、小核 16KiB；其他共享资源不增加。
6. **布局实验。** 使用现有 manifest 的分离 element/scale 地址和行 stride，把 scale 行间距变成奇数个 32B sector，避免每行都从同一输出通道开始。转换逐行验证所有元素和 scale 的字节不变；更新映像/manifest 哈希，保留原输出 oracle，并记录 padding 后的完整映像大小。没有再量化或用假压缩比例估计流量。
7. **可借用的每核保留额度。** `fair_credits` 为可消融开关。在总共 128 个逻辑额度下，每个等待的核保留最低 16 个额度；其他核空闲时可借走它的额度。已有在途请求不被抢占，释放后优先让不足最低额度的等待核恢复。等待者取消会释放保留资格。单核也用同一实现，不为双核额外增加容量。

## 资源与接口约束

- 所有架构均 4,096 个乘法器、相同频率、相同共享向量吞吐、相同 HBM preset。
- 每核向量/accumulator 私有，总量分别 4MiB / 1MiB；权重槽总 SRAM 64KiB。
- DMA 总预算统一 **44KiB**。`global_dma_staging_bytes` 是其中的 response 区域，不能与 44KiB 再相加。该预算来自旧实验 40KiB read-cache + 4KiB staging 的总额重分配；当前 read-cache 都关闭。
- 保守预留公式：`120*credits + 8192 + Σ slots*(128 + 24*BLEN*(ceil(MLEN/64)+ceil(ceil(MLEN/8)/64)+2))`。
  每逻辑额度预留 64B response、32B MSHR、24B waiter；8KiB 固定部分包括 256×16B 原生 tracker，以及查询/复制流水线、队列头和仲裁状态。每槽另预留完整复制片段描述符和 128B 状态。容量不够在进入 executor 前拒绝。
- 复制片段覆盖行尾、sector 边界和 64B 边界。尾部只参与合法元素的数值计算；不把填充数据算入有效 MAC。
- 不同 MLEN 的输入端口需求仍不同。相同乘法器/SRAM 总量只构成受控模型比较，**不是等面积、等功耗或 RTL 时序已验证的硬件结论**。
- 普通 memory trait 保留完整 64B 默认读取；只有显式声明支持 sector 的后端才能运行此 DMA 实现，避免把实际整行读取记成半行流量。

## 对照与验收

主实验包括 4 个完整 D/F 尺寸的 Qwen/DeepSeek decode 路由窗口，4 种单核形状、同构双核和异构双核；每点重复 2 次。数据仍为真实归档路由配合确定性非零合成权重与输入，不是训练权重或完整 agent trajectory。

- 四个窗口：校准后完整行/全局队列对照，完整 sector/MSHR/4-slot 候选，以及 scale 布局候选。
- DeepSeek batch-32：额外逐步测 port、sector、coalesce、128 credits、3 slots，避免把不同来源的收益混在一起。
- 保留额度补测：四窗口对每个架构测开/关；关闭开关必须逐项复现前一候选的数值、时间和原生后端统计。
- 数值输出与独立 oracle 比对；有效 MAC、任务唯一完成、SRAM 高水位、请求池、查询/复制吞吐和 native read 完成数均必须一致。对本轮只读路径，`native served reads * 32 == reported HBM bytes`。
- 必须报告最佳单核，不能只与最慢的大方阵比；每种优化也给单核/同构双核使用。

## 尚未实现的设计项

当前实现为正常矩阵 SRAM 的**权重读取**通路。专门的 scale 持久 cache、每通道每核 E/S 分级队列及 deadline/byte-DRR 仲裁、运行中取消整个 tile 与 epoch 切换、transpose 数据通路、输入/输出 DMA 与权重竞争、完整指令集/RTL 集成仍是后续设计项。共享任务 dispatch 已有实现，但并未新增任意跨核 tile 迁移。

上述项目不能由本次耗时数字宣称已经完成。是否继续做某一项，应由此次分项指标决定；若分通道排队基本无收益，就不应凭空宣称复杂仲裁器能解决当前主要瓶颈。

## 构建与复现

- Nix 的原生源码、yaml-cpp、fmt 版本仍由 `transactional_emulator/pkgs/ramulator2/default.nix` 固定。更改 C API 后必须重建该库；旧库因缺少 v2 符号不能静默运行新 Rust 二进制。
- 非 Nix 构建可使用同目录 `build_calibrated.py`，显式传入已下载的源码、yaml-cpp、fmt 路径和全新构建目录，使用本地依赖并记录产物/源码哈希。
- `run_moe_dma_campaign.py` 与 `run_moe_dma_fairness.py` 为实验入口；`compare_moe_normal.py` 执行强制验收，任意点失败不得输出整体正向收益结论。
- 输出目录 `repro/` 归档实际使用的两个 Rust 二进制、同一个原生库、构建环境与测试日志。原始 9 月 5 日旧 wrapper 结果不覆盖、不与本轮直接做加速比。
- 另外用 `run_moe_dma_bypass.py` 测试 2-slot / 4-slot 直接路径的乐观对照，查询和返回复制代价为零。该对照防止零命中查询开销人为制造收益；它不是完整端口计时的物理设计。
