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

- `feat/moe-output-pool-20260909` `582c1f9e11ef` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-refinement-20260909` `5d0ce828f8c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/matrix-sram-lcompute` `3b284f93752a` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/moe-dual-normal-20260905` `fe6768aac2e4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/matrix-lcompute-20260905` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## review/matrix-lcompute-20260905 (44693edaf041)

- 使用 `git merge --no-ff --no-commit`；17 个冲突，双方版本已归档。
- 冲突 `.github/workflows/transactional_emulator.yml`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `justfile`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `plena_settings.toml`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/lib/sram/src/matrix.rs`：保留较新可配置bank映射、HashSet布局校验缓存与packet counters；旧版固定映射被后续功能取代。
- 冲突 `transactional_emulator/src/accelerator/access.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/accelerator/dispatch.rs`：保留较新mview/lstream、ScalarOperand与M_MM_P/affine-view接口；旧matrix_view/view_mask接口与当前ISA不兼容。
- 冲突 `transactional_emulator/src/accelerator/mod.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/accelerator/pipeline_tests.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/accelerator/registers.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/load_config.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/matrix_machine.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/op.rs`：保留较新lmask/stream/Matrix投影ISA，与新dispatch/compiler匹配。
- 冲突 `transactional_emulator/src/runner.rs`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/src/timing.rs`：保留较新原生HBM/Matrix-service时序接口。
- 冲突 `transactional_emulator/testbench/README.md`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。
- 冲突 `transactional_emulator/testbench/emulator_runner.py`：保留当前v3+2026-10-01 projection功能版本，旧review快照接口/配置已被较新版本取代；旧文档或结果副本保存在archive中。

- `research/projection-pipeline-20260928` `57d2b051aee4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/research/projection-pipeline-20260928` `6ea7434697bc` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-layer-service` `b013406fa916` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-full-layer` `60d3dec21c47` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `research/ltile-analytic-v3` `db3e59afac35` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/moe-bottleneck-diagnostic-20260911` `da93c32a2203` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-output-pool-20260909` `582c1f9e11ef` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-refinement-20260909` `5d0ce828f8c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/matrix-sram-lcompute` `3b284f93752a` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/moe-dual-normal-20260905` `fe6768aac2e4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/matrix-lcompute-20260905` `44693edaf041` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/review/moe-dual-normal-20260905` `ddcfae5d0f2f` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-mechanism-scope-20260905` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## archive/pr-116-before-mechanism-scope-20260905 (0e7effade1d4)

- 使用 `git merge --no-ff --no-commit`；21 个冲突，双方版本已归档。
- `.github/workflows/transactional_emulator.yml`：旧快照0e7effade1d4（168行）与当前225行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/perf_model.py`：旧快照0e7effade1d4（1565行）与当前1579行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/test_kda_stage_calibration.py`：旧快照0e7effade1d4（105行）与当前250行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `flake.nix`：旧快照0e7effade1d4（234行）与当前241行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `justfile`：旧快照0e7effade1d4（369行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/matrix.rs`：旧快照0e7effade1d4（2390行）与当前2439行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/access.rs`：旧快照0e7effade1d4（1057行）与当前1093行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照0e7effade1d4（2204行）与当前2334行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照0e7effade1d4（122行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mview.rs`：旧快照0e7effade1d4（260行）与当前268行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mview_recurrence_tests.rs`：旧快照0e7effade1d4（2390行）与当前2337行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：execute_oversized_ltile_line, ltile_dot_reduce_rejects_a_column_wider_than_vlen, ltile_outer_update_rejects_a_row_wider_than_vlen, ltile_scale_accum_rejects_a_row_wider_than_vlen；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/matrix_machine.rs`：旧快照0e7effade1d4（941行）与当前1422行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照0e7effade1d4（1640行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照0e7effade1d4（423行）与当前461行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runtime_config.rs`：旧快照0e7effade1d4（81行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/timing.rs`：旧快照0e7effade1d4（209行）与当前245行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照0e7effade1d4（1923行）与当前2249行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/matrix_lcompute_execution_compare.py`：旧快照0e7effade1d4（659行）与当前583行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/matrix_lcompute_recurrence_test.py`：旧快照0e7effade1d4（647行）与当前647行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/test_matrix_lcompute_execution_helpers.py`：旧快照0e7effade1d4（130行）与当前116行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_comparison_row, _control_results, test_common_accuracy_is_required_before_publishing_a_speedup；旧接口不接回新ISA。合并后运行全套相关单测验证。
