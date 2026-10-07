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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## archive/pr-116-before-scope-cleanup-20260905 (2cd96e29eb7c)

- 使用 `git merge --no-ff --no-commit`；28 个冲突，双方版本已归档。
- `README.md`：旧快照2cd96e29eb7c（399行）与当前185行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/matrix_lcompute_campaign.py`：旧快照2cd96e29eb7c（2759行）与当前2762行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/test_kda_stage_calibration.py`：旧快照2cd96e29eb7c（246行）与当前250行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/hybrid_lcompute_packet_v2/campaign.json`：旧快照2cd96e29eb7c（1行）与当前14446行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/hybrid_lcompute_paper2048_batch_v1/campaign.json`：旧快照2cd96e29eb7c（1行）与当前22449行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/hybrid_lcompute_paper2048_v1/campaign.json`：旧快照2cd96e29eb7c（1行）与当前14853行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/matrix_lcompute_agentic_v1/campaign.json`：旧快照2cd96e29eb7c（1行）与当前28247行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/matrix_lcompute_agentic_v2/campaign.json`：旧快照2cd96e29eb7c（1行）与当前44430行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/matrix_lcompute_e2e_v1/campaign.json`：旧快照2cd96e29eb7c（1行）与当前38559行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/matrix_lcompute_e2e_v5/campaign.json`：旧快照2cd96e29eb7c（1行）与当前260689行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `artifacts/matrix_lcompute_e2e_v6/campaign.json`：旧快照2cd96e29eb7c（1行）与当前263821行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `docs/MATRIX_LCOMPUTE_PRE_RTL_FREEZE_ZH.md`：旧快照2cd96e29eb7c（348行）与当前343行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `justfile`：旧快照2cd96e29eb7c（559行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/matrix.rs`：旧快照2cd96e29eb7c（2390行）与当前2439行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/access.rs`：旧快照2cd96e29eb7c（1057行）与当前1093行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照2cd96e29eb7c（2204行）与当前2334行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照2cd96e29eb7c（122行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mview.rs`：旧快照2cd96e29eb7c（260行）与当前268行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mview_recurrence_tests.rs`：旧快照2cd96e29eb7c（2390行）与当前2337行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：execute_oversized_ltile_line, ltile_dot_reduce_rejects_a_column_wider_than_vlen, ltile_outer_update_rejects_a_row_wider_than_vlen, ltile_scale_accum_rejects_a_row_wider_than_vlen；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/matrix_machine.rs`：旧快照2cd96e29eb7c（941行）与当前1422行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照2cd96e29eb7c（1640行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照2cd96e29eb7c（423行）与当前461行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runtime_config.rs`：旧快照2cd96e29eb7c（81行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/timing.rs`：旧快照2cd96e29eb7c（209行）与当前245行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照2cd96e29eb7c（1923行）与当前2249行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/matrix_lcompute_execution_compare.py`：旧快照2cd96e29eb7c（659行）与当前583行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/test_matrix_lcompute_execution_helpers.py`：旧快照2cd96e29eb7c（119行）与当前116行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## feat/moe-dual-normal-v0 (f74f454d2fab)

- 使用 `git merge --no-ff --no-commit`；13 个冲突，双方版本已归档。
- `.github/workflows/transactional_emulator.yml`：旧快照f74f454d2fab（153行）与当前225行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/memory/src/lib.rs`：旧快照f74f454d2fab（342行）与当前452行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/ramulator/src/model.rs`：旧快照f74f454d2fab（552行）与当前921行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照f74f454d2fab（104行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：set_coalesce_hbm_bursts；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/bin/moe_dual_normal.rs`：旧快照f74f454d2fab（181行）与当前255行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/moe_normal/engine.rs`：旧快照f74f454d2fab（1444行）与当前2253行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/moe_normal/mod.rs`：旧快照f74f454d2fab（15行）与当前21行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/moe_normal/tests.rs`：旧快照f74f454d2fab（741行）与当前935行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/moe_normal/types.rs`：旧快照f74f454d2fab（241行）与当前414行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照f74f454d2fab（650行）与当前744行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py`：旧快照f74f454d2fab（385行）与当前775行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/moe_timing/replay/test_compare_moe_normal.py`：旧快照f74f454d2fab（432行）与当前776行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## feature/mamba-kda-support (e93b832f6586)

- 使用 `git merge --no-ff --no-commit`；25 个冲突，双方版本已归档。
- `.github/workflows/transactional_emulator.yml`：旧快照e93b832f6586（187行）与当前225行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `README.md`：旧快照e93b832f6586（305行）与当前185行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/__init__.py`：旧快照e93b832f6586（3行）与当前10行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/b200_campaign_raw.py`：旧快照e93b832f6586（739行）与当前720行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/b200_formal_campaign.py`：旧快照e93b832f6586（413行）与当前405行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/kimi_k3_workload.py`：旧快照e93b832f6586（735行）与当前802行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_kda_core_stages；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/nemotron3_workload.py`：旧快照e93b832f6586（840行）与当前979行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_block_norm；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/profile_paths.py`：旧快照e93b832f6586（22行）与当前20行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/profiles/b200_kda_nemotron_campaign_complete.json`：旧快照e93b832f6586（946行）与当前941行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/profiles/nemotron3_decode_routing_trace.json`：旧快照e93b832f6586（1行）与当前1行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/test_b200_formal_campaign.py`：旧快照e93b832f6586（114行）与当前102行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `justfile`：旧快照e93b832f6586（390行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/vector.rs`：旧快照e93b832f6586（460行）与当前653行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：capacity_elements；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照e93b832f6586（1378行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：batched_matrix_vector_mirrors_the_rs1_plus_rd_addressing, batched_write_out_ops_read_nothing_but_still_provide_hiding_capacity, classify, classify_timing_access, classify_with_topk, classify_with_topk_bias, control_flow_barriers_and_scalar_ops_stay_out_of_the_model, failed_state_commands_stop_the_program, fp0_reductions_are_no_ops_and_must_not_retire_prefetches, gp_stub, prefetch_write_extents_match_the_dma_transfer_sizes, read_ranges, require_state_success, store_reads_the_region_it_drains_rather_than_acting_as_a_barrier, successful_state_commands_may_retire, timing_access_for_opcode, topk_correction_bias_is_a_real_second_vram_read, topk_escape_policy_takes_its_read_extent_from_the_control_register, topk_reads_every_row_the_expert_policy_spans, vector_write_out_ops_read_their_destination_row, write_ranges；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照e93b832f6586（115行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：fence_all_state_queues, write_state_profile；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/registers.rs`：旧快照e93b832f6586（310行）与当前331行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：set_topk_bias_vram_addr, set_topk_control, topk_bias_vram_addr, topk_control_target_selects_policy_or_bias_without_clobbering_the_other, topk_policy_preserves_kimi_sigmoid_mode_without_corrupting_shape, topk_sigmoid_normalized, topk_uses_correction_bias；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/cli.rs`：旧快照e93b832f6586（233行）与当前223行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/dma.rs`：旧快照e93b832f6586（531行）与当前610行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_layout_sub_byte_element_uses_packed_byte_lengths；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照e93b832f6586（63行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照e93b832f6586（1071行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_funct1_does_not_bleed_into_rmask, test_decode_l_scatter_m, test_decode_rejects_noncanonical_c_set_topk_reg, test_decode_rejects_noncanonical_l_scatter_m, test_decode_rejects_noncanonical_x_state, test_decode_x_state_golden_fence, test_decode_x_state_golden_step, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照e93b832f6586（342行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/stage_profile.rs`：旧快照e93b832f6586（2276行）与当前2391行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照e93b832f6586（651行）与当前744行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：compare；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/sim_env_utils.py`：旧快照e93b832f6586（837行）与当前969行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## feat/nemotron3-mamba-dse (445cd954bd53)

- 使用 `git merge --no-ff --no-commit`；37 个冲突，双方版本已归档。
- `.github/workflows/transactional_emulator.yml`：旧快照445cd954bd53（184行）与当前225行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `README.md`：旧快照445cd954bd53（386行）与当前185行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/__init__.py`：旧快照445cd954bd53（3行）与当前10行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/b200_campaign_raw.py`：旧快照445cd954bd53（729行）与当前720行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/b200_formal_campaign.py`：旧快照445cd954bd53（413行）与当前405行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/kimi_k3_workload.py`：旧快照445cd954bd53（685行）与当前802行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/layout_mode_dse.py`：旧快照445cd954bd53（210行）与当前229行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/nemotron3_routing_dse.py`：旧快照445cd954bd53（247行）与当前264行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/nemotron3_workload.py`：旧快照445cd954bd53（833行）与当前979行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_block_norm；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/profile_paths.py`：旧快照445cd954bd53（22行）与当前20行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/test_b200_formal_campaign.py`：旧快照445cd954bd53（114行）与当前102行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/performance/test_nemotron3_routing_dse.py`：旧快照445cd954bd53（59行）与当前59行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/reference/kimi_k3_kda.py`：旧快照445cd954bd53（360行）与当前367行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `analytic_models/reference/test_kimi_k3_kda.py`：旧快照445cd954bd53（216行）与当前256行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `doc/COMMON_STATE_ENGINE_DESIGN_ZH.md`：旧快照445cd954bd53（194行）与当前192行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `doc/L_COMPUTE_PRE_RTL_STATUS_ZH.md`：旧快照445cd954bd53（98行）与当前118行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `doc/connected_hybrid_validation.md`：旧快照445cd954bd53（113行）与当前221行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `justfile`：旧快照445cd954bd53（353行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/vector.rs`：旧快照445cd954bd53（460行）与当前653行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：capacity_elements；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照445cd954bd53（1378行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：batched_matrix_vector_mirrors_the_rs1_plus_rd_addressing, batched_write_out_ops_read_nothing_but_still_provide_hiding_capacity, classify, classify_timing_access, classify_with_topk, classify_with_topk_bias, control_flow_barriers_and_scalar_ops_stay_out_of_the_model, failed_state_commands_stop_the_program, fp0_reductions_are_no_ops_and_must_not_retire_prefetches, gp_stub, prefetch_write_extents_match_the_dma_transfer_sizes, read_ranges, require_state_success, store_reads_the_region_it_drains_rather_than_acting_as_a_barrier, successful_state_commands_may_retire, timing_access_for_opcode, topk_correction_bias_is_a_real_second_vram_read, topk_escape_policy_takes_its_read_extent_from_the_control_register, topk_reads_every_row_the_expert_policy_spans, vector_write_out_ops_read_their_destination_row, write_ranges；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照445cd954bd53（115行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：fence_all_state_queues, write_state_profile；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/registers.rs`：旧快照445cd954bd53（310行）与当前331行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：set_topk_bias_vram_addr, set_topk_control, topk_bias_vram_addr, topk_control_target_selects_policy_or_bias_without_clobbering_the_other, topk_policy_preserves_kimi_sigmoid_mode_without_corrupting_shape, topk_sigmoid_normalized, topk_uses_correction_bias；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/cli.rs`：旧快照445cd954bd53（233行）与当前223行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/dma.rs`：旧快照445cd954bd53（531行）与当前610行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_layout_sub_byte_element_uses_packed_byte_lengths；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照445cd954bd53（63行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照445cd954bd53（1071行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_funct1_does_not_bleed_into_rmask, test_decode_l_scatter_m, test_decode_rejects_noncanonical_c_set_topk_reg, test_decode_rejects_noncanonical_l_scatter_m, test_decode_rejects_noncanonical_x_state, test_decode_x_state_golden_fence, test_decode_x_state_golden_step, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照445cd954bd53（342行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/stage_profile.rs`：旧快照445cd954bd53（2276行）与当前2391行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/state_engine/functional/kda.rs`：旧快照445cd954bd53（240行）与当前269行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/state_engine/layout.rs`：旧快照445cd954bd53（1108行）与当前1175行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/state_engine/mod.rs`：旧快照445cd954bd53（1683行）与当前1778行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照445cd954bd53（651行）与当前744行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：compare；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/models/kimi3/connected_blocks_test.py`：旧快照445cd954bd53（786行）与当前834行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/models/kimi3/kda_connected_test.py`：旧快照445cd954bd53（664行）与当前665行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/models/nemotron3/mamba_connected_test.py`：旧快照445cd954bd53（632行）与当前648行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/sim_env_utils.py`：旧快照445cd954bd53（837行）与当前969行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testdata/l_scatter_m_v1_golden.json`：旧快照445cd954bd53（16行）与当前36行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## local/shared-route-sync-20260810 (a97eceeac7cd)

- 使用 `git merge --no-ff --no-commit`；6 个冲突，双方版本已归档。
- `transactional_emulator/lib/quantize/src/tensor.rs`：旧快照a97eceeac7cd（403行）与当前432行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_mxint8_block_decodes_and_encodes_rtl_sign_magnitude_bytes；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照a97eceeac7cd（1294行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：batched_matrix_vector_mirrors_the_rs1_plus_rd_addressing, batched_write_out_ops_read_nothing_but_still_provide_hiding_capacity, classify, classify_timing_access, classify_with_topk, control_flow_barriers_and_scalar_ops_stay_out_of_the_model, fp0_reductions_are_no_ops_and_must_not_retire_prefetches, gp_stub, prefetch_write_extents_match_the_dma_transfer_sizes, read_ranges, resolve_topk_policy, store_reads_the_region_it_drains_rather_than_acting_as_a_barrier, timing_access_for_opcode, topk_escape_policy_takes_its_read_extent_from_the_control_register, topk_reads_every_row_the_expert_policy_spans, vector_write_out_ops_read_their_destination_row, write_ranges；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/loop_state.rs`：旧快照a97eceeac7cd（161行）与当前186行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照a97eceeac7cd（94行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照a97eceeac7cd（931行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_compiler_batch4_route_fixtures, test_decode_funct1_does_not_bleed_into_rmask, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## feat/nemotron3-mamba2-system (4b4c26a90c98)

- 使用 `git merge --no-ff --no-commit`；6 个冲突，双方版本已归档。
- `justfile`：旧快照4b4c26a90c98（286行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照4b4c26a90c98（1256行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：batched_matrix_vector_mirrors_the_rs1_plus_rd_addressing, batched_write_out_ops_read_nothing_but_still_provide_hiding_capacity, classify, classify_timing_access, classify_with_topk, control_flow_barriers_and_scalar_ops_stay_out_of_the_model, fp0_reductions_are_no_ops_and_must_not_retire_prefetches, gp_stub, prefetch_write_extents_match_the_dma_transfer_sizes, read_ranges, store_reads_the_region_it_drains_rather_than_acting_as_a_barrier, timing_access_for_opcode, topk_escape_policy_takes_its_read_extent_from_the_control_register, topk_reads_every_row_the_expert_policy_spans, vector_write_out_ops_read_their_destination_row, write_ranges；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照4b4c26a90c98（98行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：write_mamba_timing_profile；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照4b4c26a90c98（58行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照4b4c26a90c98（923行）与当前1742行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_funct1_does_not_bleed_into_rmask, test_decode_x_mamba_preserves_all_validation_fields, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照4b4c26a90c98（327行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## local/simulator-fp12-expert-20260727 (0e327b871f26)

- 使用 `git merge --no-ff --no-commit`；11 个冲突，双方版本已归档。
- `transactional_emulator/lib/quantize/src/dtype.rs`：旧快照0e327b871f26（742行）与当前734行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_e6m5_bias_conversion_from_f32, test_mxint8_sign_magnitude_fraction_encoding；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照0e327b871f26（749行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：c_break_terminates_program_even_inside_active_loop, c_break_terminates_straight_line_program, run_program, test_accelerator；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/loop_state.rs`：旧快照0e327b871f26（163行）与当前186行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照0e327b871f26（92行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：load_fpsram_from_bytes；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/registers.rs`：旧快照0e327b871f26（167行）与当前331行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/load_config.rs`：旧快照0e327b871f26（741行）与当前833行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照0e327b871f26（278行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runtime_config.rs`：旧快照0e327b871f26（73行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照0e327b871f26（689行）与当前2424行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_topk_softmax_canonicalizes_zero_and_splits_positive_infinity；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照0e327b871f26（446行）与当前744行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/sim_env_utils.py`：旧快照0e327b871f26（850行）与当前969行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_resolve_compiler_root；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## local/archive-simulator-window1-p3-20260727 (eb1311db6a41)

- 使用 `git merge --no-ff --no-commit`；36 个冲突，双方版本已归档。
- `justfile`：旧快照eb1311db6a41（176行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/memory/src/chunked.rs`：旧快照eb1311db6a41（343行）与当前541行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/memory/src/lib.rs`：旧快照eb1311db6a41（281行）与当前550行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/quantize/src/tensor.rs`：旧快照eb1311db6a41（379行）与当前431行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/ramulator/src/model.rs`：旧快照eb1311db6a41（179行）与当前921行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tick, try_read, try_write_transfer；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/runtime/src/time.rs`：旧快照eb1311db6a41（225行）与当前235行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/matrix.rs`：旧快照eb1311db6a41（193行）与当前2439行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tensor_to_f32_vec, test_matrix_write_delayed_uses_tile_size_divisor；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/sram/src/vector.rs`：旧快照eb1311db6a41（480行）与当前653行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tensor_from_f32_slice, tensor_to_f32_vec；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照eb1311db6a41（813行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：timing_access_for_opcode；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照eb1311db6a41（90行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/cli.rs`：旧快照eb1311db6a41（182行）与当前237行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照eb1311db6a41（75行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：timing_golden_fixture_pins_required_workloads；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/matrix_core.rs`：旧快照eb1311db6a41（131行）与当前127行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/matrix_machine.rs`：旧快照eb1311db6a41（648行）与当前1422行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照eb1311db6a41（855行）与当前1759行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_funct1_does_not_bleed_into_rmask, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照eb1311db6a41（303行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/stage_profile.rs`：旧快照eb1311db6a41（538行）与当前2391行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：duration_to_cycles, duration_to_cycles_rounds_up_to_period, resource_json；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照eb1311db6a41（614行）与当前2424行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tensor_from_f32_slice；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/README.md`：旧快照eb1311db6a41（49行）与当前103行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/compare/isa_analysis.py`：旧快照eb1311db6a41（387行）与当前395行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/linear_test.py`：旧快照eb1311db6a41（152行）与当前152行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照eb1311db6a41（649行）与当前744行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/layout_utils.py`：旧快照eb1311db6a41（84行）与当前114行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/models/gpt_oss/attention_semantics_test.py`：旧快照eb1311db6a41（4274行）与当前4090行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_align_to_tile, _bias_parts, _comparison_params, _make_packed_rope_inputs, _make_rotate_half_matrix, _rel_rms, _resolve_sliding_window, _router_bias_block_rows, _router_margin_summary, _strict_tail_summary, _topk_match_summary, _write_json；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/models/gpt_oss/block_glue_test.py`：旧快照eb1311db6a41（311行）与当前310行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_align_to；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_activation_test.py`：旧快照eb1311db6a41（199行）与当前174行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_activation_golden, _bf16, _exact_mxfp8_tensor；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_clamp_test.py`：旧快照eb1311db6a41（158行）与当前155行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_exact_mxfp8_tensor；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_combine_test.py`：旧快照eb1311db6a41（433行）与当前382行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_activation_golden, _bf16, _exact_mxfp8_tensor, _linear_projection_golden；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_expert_test.py`：旧快照eb1311db6a41（215行）与当前305行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_activation_golden, _bf16, _exact_mxfp8_tensor；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_gate_up_test.py`：旧快照eb1311db6a41（132行）与当前129行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_exact_mxfp8_tensor；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_moe_gather_scatter_test.py`：旧快照eb1311db6a41（2140行）与当前2107行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_activation_golden, _bf16, _comparison_params_for, _decode_bf16_dump, _decode_u32_dump, _expanded_bias, _linear_projection_golden；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_real_layer0_test.py`：旧快照eb1311db6a41（838行）与当前780行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_activation_golden, _bf16, _comparison_params_for, _expanded_bias, _linear_projection_golden, _stats_dict；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_router_gemm_test.py`：旧快照eb1311db6a41（462行）与当前455行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_align_to, _bf16, _stats_dict；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/gpt_oss_topk_test.py`：旧快照eb1311db6a41（182行）与当前180行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_decode_bf16_dump, _decode_u32_dump；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/sim_env_utils.py`：旧快照eb1311db6a41（823行）与当前969行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：_resolve_compiler_root；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` `eb1311db6a41` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## local/candidate-simulator-topk-semantics-20260726 (1d5f601114c5)

- 使用 `git merge --no-ff --no-commit`；11 个冲突，双方版本已归档。
- `transactional_emulator/lib/quantize/src/dtype.rs`：旧快照1d5f601114c5（742行）与当前734行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_e6m5_bias_conversion_from_f32, test_mxint8_sign_magnitude_fraction_encoding；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照1d5f601114c5（749行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：c_break_terminates_program_even_inside_active_loop, c_break_terminates_straight_line_program, run_program, test_accelerator；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/loop_state.rs`：旧快照1d5f601114c5（163行）与当前186行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照1d5f601114c5（92行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：load_fpsram_from_bytes；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/registers.rs`：旧快照1d5f601114c5（167行）与当前331行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/scalar_sram.rs`：旧快照1d5f601114c5（186行）与当前240行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/load_config.rs`：旧快照1d5f601114c5（741行）与当前833行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照1d5f601114c5（278行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runtime_config.rs`：旧快照1d5f601114c5（73行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照1d5f601114c5（689行）与当前2424行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_topk_softmax_canonicalizes_zero_and_splits_positive_infinity；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/rtl_topk_artifact_test.py`：旧快照1d5f601114c5（169行）与当前215行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` `eb1311db6a41` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/candidate-simulator-topk-semantics-20260726` `1d5f601114c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## local/archive-e2e-simulator-fp12-expert-20260727 (f4d3c3bf0024)

- 使用 `git merge --no-ff --no-commit`；10 个冲突，双方版本已归档。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照f4d3c3bf0024（749行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：c_break_terminates_program_even_inside_active_loop, c_break_terminates_straight_line_program, run_program, test_accelerator；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/loop_state.rs`：旧快照f4d3c3bf0024（163行）与当前186行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/mod.rs`：旧快照f4d3c3bf0024（92行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：load_fpsram_from_bytes；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/registers.rs`：旧快照f4d3c3bf0024（167行）与当前331行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/scalar_sram.rs`：旧快照f4d3c3bf0024（186行）与当前240行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/load_config.rs`：旧快照f4d3c3bf0024（741行）与当前833行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照f4d3c3bf0024（278行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runtime_config.rs`：旧快照f4d3c3bf0024（73行）与当前132行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/vector_machine.rs`：旧快照f4d3c3bf0024（689行）与当前2424行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_topk_softmax_canonicalizes_zero_and_splits_positive_infinity；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/routed_moe/rtl_expert_ffn_artifact_test.py`：旧快照f4d3c3bf0024（251行）与当前293行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` `eb1311db6a41` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/candidate-simulator-topk-semantics-20260726` `1d5f601114c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-e2e-simulator-fp12-expert-20260727` `f4d3c3bf0024` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `codex/moe-e2e-sync` `6dc4e0e9e223` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p1-timing-replay` `8fb742752e04` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p2-qwen-replay` `5d9694785837` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## origin/feat/window1-p1-timing-replay (1be2b23f2a47)

- 使用 `git merge --no-ff --no-commit`；10 个冲突，双方版本已归档。
- `justfile`：旧快照1be2b23f2a47（200行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/memory/src/chunked.rs`：旧快照1be2b23f2a47（343行）与当前541行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/ramulator/src/model.rs`：旧快照1be2b23f2a47（179行）与当前921行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tick, try_read, try_write_transfer；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/runtime/src/time.rs`：旧快照1be2b23f2a47（225行）与当前235行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照1be2b23f2a47（820行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：timing_access_for_opcode；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/cli.rs`：旧快照1be2b23f2a47（182行）与当前237行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照1be2b23f2a47（54行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照1be2b23f2a47（303行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/stage_profile.rs`：旧快照1be2b23f2a47（641行）与当前2391行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：duration_to_cycles, duration_to_cycles_rounds_up_to_period；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照1be2b23f2a47（612行）与当前748行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` `eb1311db6a41` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/candidate-simulator-topk-semantics-20260726` `1d5f601114c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-e2e-simulator-fp12-expert-20260727` `f4d3c3bf0024` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `codex/moe-e2e-sync` `6dc4e0e9e223` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p1-timing-replay` `8fb742752e04` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p2-qwen-replay` `5d9694785837` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/feat/window1-p1-timing-replay` `1be2b23f2a47` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## origin/feat/window1-p2-qwen-replay (4d3fa12e7bef)

- 使用 `git merge --no-ff --no-commit`；2 个冲突，双方版本已归档。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照4d3fa12e7bef（621行）与当前748行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/window1_p2/qwen3_trace_replay_test.py`：旧快照4d3fa12e7bef（404行）与当前767行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。

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

- `archive/pr-116-before-mechanism-scope-20260905` `0e7effade1d4` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `archive/pr-116-before-scope-cleanup-20260905` `2cd96e29eb7c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/moe-dual-normal-v0` `f74f454d2fab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-official-layer` `a3100ea38b29` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/static-kda-gpu-evidence` `4ce1611270ab` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feature/mamba-kda-support` `e93b832f6586` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s3-connected` `e1c49badf0da` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s2-state-engine` `21611e3066d6` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `review/mamba-kda-s1-analytic-dse` `8a9ea5f55d6c` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba-dse` `445cd954bd53` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/shared-route-sync-20260810` `a97eceeac7cd` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/nemotron3-mamba2-system` `4b4c26a90c98` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/simulator-fp12-expert-20260727` `0e327b871f26` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-simulator-window1-p3-20260727` `eb1311db6a41` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/candidate-simulator-topk-semantics-20260726` `1d5f601114c5` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `local/archive-e2e-simulator-fp12-expert-20260727` `f4d3c3bf0024` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `codex/moe-e2e-sync` `6dc4e0e9e223` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p1-timing-replay` `8fb742752e04` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/window1-p2-qwen-replay` `5d9694785837` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/feat/window1-p1-timing-replay` `1be2b23f2a47` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `origin/feat/window1-p2-qwen-replay` `4d3fa12e7bef` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

- `feat/qwen-nonzero-functional-gate` `d942685525e3` 已是当前 HEAD 祖先，历史已完整包含，无需重复 merge。

## origin/feat/window1-p3-representative-runner (b610e1399b42)

- 使用 `git merge --no-ff --no-commit`；0 个冲突，双方版本已归档。

- `feat/routed-moe-emulator-substrate` gitlink `PLENA_Compiler`：双方指针归档，暂保留当前指针；独立Compiler整合结束后统一更新。

## feat/routed-moe-emulator-substrate (c05d4d8b2194)

- 使用 `git merge --no-ff --no-commit`；17 个冲突，双方版本已归档。
- `justfile`：旧快照c05d4d8b2194（176行）与当前558行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/memory/src/chunked.rs`：旧快照c05d4d8b2194（343行）与当前541行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/ramulator/src/model.rs`：旧快照c05d4d8b2194（179行）与当前921行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：tick, try_read, try_write_transfer；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/lib/runtime/src/time.rs`：旧快照c05d4d8b2194（225行）与当前235行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/accelerator/dispatch.rs`：旧快照c05d4d8b2194（813行）与当前2338行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：timing_access_for_opcode；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/cli.rs`：旧快照c05d4d8b2194（182行）与当前237行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/main.rs`：旧快照c05d4d8b2194（75行）与当前48行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：timing_golden_fixture_pins_required_workloads；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/op.rs`：旧快照c05d4d8b2194（855行）与当前1776行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：test_decode_funct1_does_not_bleed_into_rmask, vector_precision_from；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/runner.rs`：旧快照c05d4d8b2194（303行）与当前466行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/src/stage_profile.rs`：旧快照c05d4d8b2194（538行）与当前2391行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：duration_to_cycles, duration_to_cycles_rounds_up_to_period, resource_json；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/aten/compare/isa_analysis.py`：旧快照c05d4d8b2194（387行）与当前395行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/emulator_runner.py`：旧快照c05d4d8b2194（638行）与当前748行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/timing_goldens/run_timing_smoke.sh`：旧快照c05d4d8b2194（11行）与当前21行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/window1_p2/generate_true_routing_with_weights.py`：旧快照c05d4d8b2194（394行）与当前655行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/window1_p2/qwen3_trace_replay_test.py`：旧快照c05d4d8b2194（403行）与当前767行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
- `transactional_emulator/testbench/window1_p2/run_trace_batch.py`：旧快照c05d4d8b2194（122行）与当前145行功能冲突；保留较新projection/L-TILE/原生HBM兼容版本。旧版本双方已完整归档。旧版独有函数/测试名：无；旧接口不接回新ISA。合并后运行全套相关单测验证。
