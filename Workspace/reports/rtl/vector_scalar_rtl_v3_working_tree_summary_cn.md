# PLENA RTL-v3 Vector/Scalar 优化工作区总结

> **历史工作树快照，2026-07-26 审计。** 本文记录的是提交前阶段性状态，
> 不是当前未提交改动清单。最终相关提交为：PLENA_RTL `b5feafa`，
> PLENA_Compiler `7e29ab5..1470478`，PLENA_Simulator
> `df104a0..53e1126`。当前正式 DSE 使用 RTL-v4 lowering 与 ideal-II1；
> rtl-v3 hazard-aware scoreboard 仅作 sensitivity/validation。

**日期：** 2026-07-20
**范围：** 自三个仓库最近一次提交以来的未提交改动；不包含无关的 `PLENA_Tools` 工作区修改。

## 1. 基线

| 仓库 | 基线提交 |
|---|---|
| PLENA_Simulator | `f98b009` - Document the calibrated PLENA prefill modeling stack |
| PLENA_Compiler | `1c1d77a` - Add native Qwen3 MoE and compact prefill lowering |
| PLENA_RTL | `0beb43f` - Add reproducible RTL checking and DC area workflows |

本轮目标是提高 packed-GQA 中 Vector/Scalar 路径的执行效率，同时保持 Matrix 运算量、HBM 流量和现有 correctness gate 不变。

## 2. 完成的优化

### VectorMachine：并行处理多个 head segment

- 新增 `V_RED_SUM_SEGS` / `V_RED_MAX_SEGS`：一条指令同时归约一个 Vector SRAM word 中最多 16 个 segment。
- 新增 `V_ADD_VSEG` / `V_SUB_VSEG` / `V_MUL_VSEG`：将紧凑统计量分别广播回对应 segment。
- 新增 Vector SRAM lane load/store，供 ScalarMachine读取和写回紧凑统计量。
- 复用已有 reduction tree 的中间层和 Vector ALU，没有复制完整归约树，也没有增加融合数学算子。

### ScalarMachine：8-entry ROB 流水线

- FP寄存器从 8 个扩展到 16 个，`f0`继续保持常量零。
- 增加 8-entry ROB、RAW forwarding、WAW检测、功能单元 II 和单端口顺序退休。
- 保持 in-order issue/in-order retirement，不引入寄存器重命名或乱序执行。
- 独立 Scalar FP 链可按 `II=1` 发射，从而隐藏不同算子的执行延迟。

### ISA、Compiler 与运行模型同步

- ISA/assembler/decoder加入 `0x39-0x3D` 新操作，并补齐 hazard 与 Vector SRAM端口冲突。
- Compiler新增默认 `vector_scalar_schedule="rtl-v3"`，按 storage block 生成一次 square、一次多segment归约、并行scalar normalization和一次segment broadcast。
- Transactional emulator同步新指令功能、balanced-tree数值顺序、Scalar ROB和逐周期stall原因。
- CostEmitter使用同一structured lowering和同一 timing artifact，不单独猜测opcode数量。
- DSE正式compute指标改为 rtl-v3 pipeline makespan；旧 `rtl-v2/compiler-v1/legacy` 均保留用于A/B。
- 修复集成中发现的两个 `V_SHIFT_V` 问题：shift amount来源和decoder operand route。

### Timing 与DSE运行效率

- 新增 `rtl_opcode_timing_v3.json`，包含247个full-Machine测量点及RTL/diff/raw-data hash。
- 多层DSE采用“精确单层compressed scoreboard + layer stage scaling”，不使用serial-work静默回退。
- 增加跨进程pipeline cache；key覆盖trace、硬件、precision、clock、timing artifact和scheduler源码语义。

## 3. 总体收益

固定测试：Qwen3-32B，`seq=482`、`batch=16`、单层，`MLEN=VLEN=2048`、`BLEN=1024`、`HLEN=128`。

| 指标 | rtl-v2 | rtl-v3 | 收益 |
|---|---:|---:|---:|
| Q/K segment SUM | 555,264条 | 38,560条 | **减少93.06%** |
| Compute resource work | 216.14M cycles | 183.72M cycles | **减少15.00%** |
| Compute timing | 216.14M serial cycles | 151.69M pipeline cycles | **减少29.82%** |
| 单层stage-roofline latency | 218.551 ms | 154.111 ms | **减少29.49%** |

收益由两部分组成：约15%来自删除重复的per-head Vector工作；其余来自8-entry Scalar ROB和Vector/Scalar流水并行。Scalar显式lane操作使其resource work略有增加，但这些独立链可以重叠，因此最终makespan仍明显下降。

以下工作量保持完全一致，说明收益不是通过删除模型计算或内存传输获得：

```text
Matrix opcode counts: unchanged
HBM opcode counts:    unchanged
HBM read/write bytes: unchanged
HBM request counts:   unchanged
Softmax reductions:   unchanged
```

## 4. 验证结果

- 完整RTL检查：128 modules，635个生成C++文件，PASS。
- RTL decoder聚焦测试：3 passed。
- Python Compiler/CostEmitter/timing/scheduler/cache联合回归：95 passed。
- Rust全目标测试：141 passed，1个evidence-emission测试按设计ignored。
- Tiny packed-GQA transactional：100% allclose，最大绝对误差 `0.00390625`，现有correctness gate未修改。
- Timing harness的全部数值检查通过；Python/Rust读取同一校准artifact。
- DSE固定点完整运行成功，JSON与CSV均记录pipeline fidelity和cache provenance。

缓存后的独立Optuna trial由冷路径约101秒的单层replay降至3.19秒，结果cycles和latency完全一致；缓存只减少模型计算时间，不改变预测值。

## 5. 当前边界

- rtl-v3的Vector/Scalar paired-DC面积增量已经接入area proxy；Vector总面积holdout误差为3.40%，Scalar为0.005%。但Fmax和signoff power仍未验证。
- 生产segment width 128超出实测4/8/16，latency属于结构外推。
- 64层结果是精确单层scoreboard后的stage重复，不包含跨层overlap，不能称为完整decoder cycle-exact RTL仿真。
- 当前结果支持“架构与模型预测约29.5% latency改善”，不支持尚未校准的最终PPA或signoff结论。

## 6. 结论

本轮将packed Q/K RMSNorm从逐head串行归约改为按Vector word并行归约，并使Scalar normalization链能够流水执行。在不改变Matrix/HBM工作量和correctness gate的前提下，单层预测latency降低约29.5%，同时完成了RTL、Compiler、emulator、CostEmitter和DSE的统一接入。后续paired-DC结果补齐了mapped-area overlay，但没有建立1 GHz timing closure。
