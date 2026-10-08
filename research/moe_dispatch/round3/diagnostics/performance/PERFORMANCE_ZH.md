# 等价性能改进凭证

这次只减少模拟器的 Python 开销，不改变硬件、HBM、派工策略、搜索预算或实际重复次数。

- `detail=False` 不生成最终丢弃的 phase 字典；全局存储分块也不生成被丢弃的任务/phase/segment 时间偏移观察。显式 HBM spill 的字节与时间保持原样。
- 每个执行 phase 的静态瓶颈原因按原字典顺序、原除法与原 tie 规则计算一次；流体速率的浮点求和顺序不变。
- 仅 round3 隔离 CP-SAT 模块使用官方 snake_case 方法的静态兼容别名，避免每次构造模型生成弃用包装器；变量、约束、hint、求解预算及 solver 次序保持。
- 没有缓存 solver 答案、owner、物理仿真结果；所有测试两侧均实际构造新 solver、执行新回放。

## 正确性证据

| 验证 | 内容 | 结果 |
|---|---|---|
| 完整 canonical 对照 | 896 项；真实开发/留出窗口；两模式、4组参数、私有/共享权重、MILP owner、ours、nominal、detail开/关 | 全部逐位相同 |
| 全局分块/溢写路径 | 72 项；显式合成 B256/H3072，私有/共享、两模式、3种派工/预测、detail开/关 | 全部逐位相同 |
| 机制与独立回放测试 | 31 项 model/runtime、物理 oracle、search 测试 | 全部通过 |

旧源码完整保存在 `before/`，不是只保存哈希。新旧源码 SHA、逐项摘要及运行时间见 `EQUIVALENCE.json`、`CHUNK_EQUIVALENCE.json` 和两个 CSV。

主 DSE 已在改进前加载旧实现，继续运行旧源码；不能将新源码 SHA 改写成该运行的执行版本。后续 sensitivity 使用新实现时单独记录新 SHA，并引用本等价证据。Python DFS 的可选 native 等价加速由另一份独立证据说明，本凭证未启用它。

B256/H3072 小诊断的 detail=False CPU 时间从 0.551 秒到 0.462 秒；detail=True 从 0.578 秒到 0.587 秒。这里包含测量噪声，只证明未免费省略物理服务，不作为完整 DSE 加速或硬件性能结果。
