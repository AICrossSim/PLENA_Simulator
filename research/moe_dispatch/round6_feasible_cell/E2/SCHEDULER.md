# 剩余工作量配额派工器

冻结参数 δ = **0.05**。在105个开发窗口、五档算力和五种组织上统一选择；目标是各预算/组织/batch等权的在线÷离线配对GM。候选0.02/0.05/0.10/0.20，逐点记录见 `delta_development.csv.gz`。留出结果不参与选参。

Router已提供全部Me。控制器估计的是周期和两类核应承担的有效MAC工作量，不猜Expert ID。

1. 将每个专家（或Shared行块）在每类核上的成本算出。两类核的连续分配松弛按相对成本排序、平衡池内平均周期；只保留两类核的总工作量份额和层完成时刻估计，丢弃松弛中的逐任务归属。**不调用离线整数求解器，也不读取其owner列表。**
2. Shared需广播时，比较不同接收核数量的计算＋广播时间和剩余routed工作量，选择预计整层更早完成的接收核组。主实验按共同分块广播契约执行，详见METHOD.md。
3. Routed描述符按最小预计执行时间降序排列，窗口深度8；每核仅Current和Next两个上下文。保留队首优先、队满时等待，以及窗口内其他合法任务旁路。
4. 首先比较所有核的预计完成时刻。如果最快选择使某类核已接工作量超过(1+δ)×目标份额，则考虑未超额的另一类核。若它仍能在预计整层排空截止时间的(1+δ)倍内完成，就分过去；否则允许逃逸到原先最快的核。
5. 每次绑定增记该核类型的有效工作量；每个任务完成从对应剩余周期计数中扣除；已接有效工作量是层内累计值，不随任务完成减少。这里成本模型已准确知道执行成本，无另外训练的时长预测器。

原任务的“除非其他核都会更晚完成”若直接解释为“比当前任务的最小EFT更晚”，会让配额永远无效，因为其他候选本来就不会早于最小EFT。本实现把逃逸门槛明确改为**整层预测排空截止时间**，这是为使规则非空而做的语义澄清，不能隐瞒成完全逐字执行。

```text
costs = estimate_all_experts(router_counts, frozen_shapes)
targets, layer_deadline = fractional_pool_balance(costs)
plan_common_shared_broadcast_cohort()
pending = LPT(routed_tasks)
while pending:
    for task in first_8(pending):
        best = argmin_core(predicted_finish(core, task))
        if exceeds_pool_quota(best, task):
            alternate = earliest_finish_among_under_quota_pools(task)
            if alternate.finish <= (1 + delta) * forecast_layer_finish:
                best = alternate
        if best.Current_and_Next_full:
            continue  # 不强塞慢核，允许窗口内其他任务先走
        bind(task, best)
        accepted_work[type(best)] += useful_MACs(task)
        break
    else:
        wait_until_a_context_retires()
```

硬件状态：每类核目标份额、已接有效MAC工作量、预测剩余周期各64bit；每核Current/Next上下文、预计结束计数；8项前瞻描述符（Expert ID、Me、行区间、类型、预计周期）；共享广播接收掩码与块引用计数。两类核份额生成需要一次按相对成本排序/连续分配，可由顺序控制器和定点除法实现，**不是已经综合出的几个免费比较器**。每次发射计1周期，Next预配置计2周期且可隐藏。层前另计每(task,type)成本查表1周期、bitonic排序网络逐级1周期、每任务份额扫描1周期，以及Shared接收核数量逐方案扫描的周期；新旧在线共同承担基础成本查表/排序费用。此为公开的控制器服务假设，完整门级延迟/面积尚未综合，应在下一阶段校准。

旧在线列：调用第五轮冻结的LPT/EFT准入函数取得owner和每核routed顺序，再按相同Shared广播cohort重放，以避免拿无广播旧时间与收费新时间混比。原第五轮零广播周期另存 `original_r5_no_broadcast_ms`。`quota_ablation_ms` 固定新的Shared接收核规划，仅关闭份额门控，隔离配额本身的贡献。

`workload_balance.csv` 同时列出模型预测份额、在线实际份额、离线事后份额和各核空闲率。最后一项只用于诊断，绝未进入在线决策。
