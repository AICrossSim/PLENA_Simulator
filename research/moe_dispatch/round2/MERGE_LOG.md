# Simulator 合并日志

目标：research/moe-supply-first-v3；逐次 no-ff，源分支与工作树全部保留。冲突双方先归档，再选择功能较新版本。

## research/projection-pipeline-20260928 (57d2b051aee4)

- no-ff 导入 projection/L-TILE/Mamba 最新代码与结果。
- 唯一冲突为 Compiler gitlink；双方SHA已归档。暂保留活动v3 Compiler 480d558c72f8，待Compiler单分支整合完成后指向其最终SHA；没有丢弃Compiler研究实现。

- `research/projection-pipeline-20260928` `57d2b051aee4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/research/projection-pipeline-20260928` `6ea7434697bc` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-layer-service` `b013406fa916` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-full-layer` `60d3dec21c47` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-analytic-v3` `db3e59afac35` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/moe-bottleneck-diagnostic-20260911` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## local/moe-bottleneck-diagnostic-20260911 (da93c32a2203)

- 使用 `git merge --no-ff --no-commit`；1 个冲突，双方版本已归档。
