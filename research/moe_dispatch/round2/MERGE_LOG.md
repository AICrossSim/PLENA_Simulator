# Simulator 合并日志

目标：research/moe-supply-first-v3；逐次 no-ff，源分支与工作树全部保留。冲突双方先归档，再选择功能较新版本。

## research/projection-pipeline-20260928 (57d2b051aee4)

- no-ff 导入 projection/L-TILE/Mamba 最新代码与结果。
- 唯一冲突为 Compiler gitlink；双方SHA已归档。暂保留活动v3 Compiler 480d558c72f8，待Compiler单分支整合完成后指向其最终SHA；没有丢弃Compiler研究实现。
