# Simulator 分支整合记录（2026-10-20）

统一开发入口：`research/moe-supply-first-v3`。

本次以 `246f6d320d46a1151db414d34694894ed5f9066b` 为基底，整合用户推送记录或 PR 记录所关联的 13 个远端分支。12 个分支的 tip 原本已经是基底的祖先；仅 `feature/mamba-kda-support` 的两个后续提交尚未包含。完整 SHA 和验证摘要见同名 JSON。此次整合保留提交历史，不重写历史。

| 来源分支 | 核对时的远端 tip | 处理 |
|---|---|---|
| `feat/emulator-timing-core` | `2ebbd2fda6ab` | 已包含 |
| `feat/matrix-sram-lcompute` | `57d2b051aee4` | 已包含 |
| `feat/static-kda-gpu-evidence` | `4ce1611270ab` | 已包含 |
| `feat/static-kda-official-layer` | `a3100ea38b29` | 已包含 |
| `feat/window1-p1-timing-replay` | `1be2b23f2a47` | 已包含 |
| `feat/window1-p2-qwen-replay` | `4d3fa12e7bef` | 已包含 |
| `feat/window1-p3-representative-runner` | `b610e1399b42` | 已包含 |
| `feature/mamba-kda-support` | `bcc9a8942432` | 本次合并 |
| `research/moe-heterogeneous-dispatch` | `917e9d163117` | 已包含 |
| `research/moe-joint-runtime` | `3ea69acce1fb` | 已包含 |
| `research/moe-supply-first-v3` | `01e57880ae74` | 已包含 |
| `review/matrix-lcompute-20260905` | `44693edaf041` | 已包含 |
| `review/moe-dual-normal-20260905` | `fe6768aac2e4` | 已包含 |

## 实际合并与冲突处理

- 合并 `76977b40` 的格式整理，以及 `bcc9a894` 的 CI workflow 全目录检查。
- `test_nemotron3_formal_dse.py` 保留当前版本新增的 MoE combine 字节记账断言；不恢复旧断言。
- `emulator_runner.py`、`sim_env_utils.py` 保留当前分支实现，避免旧代码的格式调整复活已移除的逻辑。
- 合并后的 35 个 Python 变更文件中，34 个与基底的 AST 相同；唯一行为变更是 `test_testbench_stage_attribution.py`。
- 为适配已经存在的整合代码，将 Qwen replay 的预期 stage 调用数量从 1 更新为 3（accumulator_init、expert_activation、expert_route_weight），并将 11 个漏接的既有测试加入 CI。
- `research/moe_dispatch/` 整棵树与基底保持一致，包括第二至第八轮冻结结果。Compiler 与 Tools 子模块 pin 均保持不变。
- 原工作区的 9 个未提交文件不属于此次合并，保留原状。

## 验证结果与边界

验证使用 Python 3.11、PyTorch 2.7.1 CPU；没有重新跑 GPU 实验、RTL 或全部 Rust 系统测试。

| 检查 | 结果 |
|---|---|
| 10 个受影响分析模型测试模块 | 62 passed |
| 第七/八轮及 M-DPU、scaling 回归 | 54 passed |
| stage attribution 与 CI wiring | 6 passed |
| 新接入 CI 的测试、Kimi/Nemotron 参考计算、X_STATE 契约 | 44 passed，2 个既有失败，另有 10 个 subtests passed |
| 冲突标记与 whitespace 检查 | 通过 |
| 冻结结果和子模块 pin | 与基底相同 |

总计 166 项通过。以下两个 X_STATE 契约失败在基底原始测试代码上也已重现，不能将本次整合描述为整个项目全绿：

1. `test_l_scatter_m_golden_matches_the_compiler_spec_copy`：Simulator 与已固定 Compiler 的 L_SCATTER_M golden 不一致。
2. `test_generated_state_contract_matches_the_compiler_spec`：Compiler 契约生成器将 X_STATE 保留为 opcode `0x3D`，该位置已由 `V_SOFTPLUS_V` 使用。

本次不重分配 opcode，不修改 golden 或跳过上述检查来掩盖不一致。运行时和实验结果未因整合而改变。

复核历史覆盖，可对 JSON 中每个 `sources[].sha` 运行：

```bash
git merge-base --is-ancestor <source-sha> HEAD
```

复核冻结研究树：

```bash
git diff --exit-code 246f6d32 HEAD -- research/moe_dispatch/
```

运行关键检查：

```bash
python -m pytest -q transactional_emulator/testbench/test_testbench_stage_attribution.py
python -m pytest -q --import-mode=importlib research/moe_dispatch/round7_64k_dualcore_dse/test_round7.py research/moe_dispatch/round8_model_check_holdout_isosram/test_study.py research/moe_dispatch/round8_model_check_holdout_isosram/test_diagnosis.py research/moe_dispatch/mdpu_compute/test_runtime.py research/moe_dispatch/mdpu_compute/test_report.py research/moe_dispatch/mdpu_compute/test_domain.py research/moe_dispatch/scaling_g64/test_runtime.py
```

## 旧分支与 PR

代码与历史统一到上述入口；本次不删除其他远端分支，不关闭 PR #116/#117。两者仍保留原审阅入口。`research/moe-ipd-bf16-ab` 由其他作者创建，不在此次用户分支整合清单中。
