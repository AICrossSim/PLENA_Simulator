# E5 在线派工与预测

oracle是同一实际调度的条件时长参考：物理重放验证完成时间与HBM守恒，E2E与固定ours计划相同，不是全局最优派工上界。每窗口的重放最大时间误差、字节守恒、源码和计划哈希见oracle_replay_validation.csv。

所有可实现预测器先按同一18开发窗口序列预热，再按同一135留出窗口计数，两次完整序列一致。Runtime为8项候选窗、每核Current+Next最多2项，实际资源检查后绑定，支持等待；估计不能替代就绪/依赖检查。

- pipelined/best_hetero: 纯阈值比回退慢-4.129%；EFT/MILP-LPT=1.05543；预测器最好ours与最差random延迟相差1.762%。
- pipelined/fixed_4+2: 纯阈值比回退慢-7.827%；EFT/MILP-LPT=1.07650；预测器最好ema与最差random延迟相差1.765%。
- port_tight/best_hetero: 纯阈值比回退慢41.045%；EFT/MILP-LPT=1.08928；预测器最好ours与最差random延迟相差1.570%。
- port_tight/fixed_4+2: 纯阈值比回退慢59.816%；EFT/MILP-LPT=1.09717；预测器最好static与最差btb延迟相差2.795%。
- fixed_issue/best_hetero: 纯阈值比回退慢-6.557%；EFT/MILP-LPT=1.16254；预测器最好static与最差random延迟相差3.272%。
- fixed_issue/fixed_4+2: 纯阈值比回退慢-7.681%；EFT/MILP-LPT=1.07470；预测器最好ema与最差random延迟相差1.763%。

