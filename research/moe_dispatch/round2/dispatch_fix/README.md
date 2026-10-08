# 容量感知在线派工修复

本目录保存独立实现和全部评估证据。原第二轮的代码、硬件、模型、窗口和结果
均保持原样。所有阵列尺寸统一按 **M×N×K**；单核没有修改执行路径。

入口说明见 [CONTRACT.md](CONTRACT.md)，结果和失败验收项见
[SUMMARY.md](SUMMARY.md)、[acceptance.json](acceptance.json)。
所有毫秒值来自冻结的 BF16、路由后 MoE 层解析模型，不是原生 HBM、RTL
或完整模型推理实测。MILP 分配＋LPT 回放是参照，不是全局最优时序保证。

## 复现

从本分支的仓库根目录运行；使用已有 Python 环境，依赖同第二轮
（numpy、ortools、pytest）。本次环境路径为 `/tmp/plena-round2-venv/bin/python`。

```bash
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.execute --label campaign -- /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.run --jobs 6 --stage all
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.execute --label unit_suite -- /tmp/plena-round2-venv/bin/python -m pytest research/moe_dispatch/round2 -q
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.execute --label report -- /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.report --directory research/moe_dispatch/round2/dispatch_fix
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.execute --label independent_numerical_audit -- /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.audit
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.execute --label package -- /tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.dispatch_fix.package
```

`run --stage all` 重新运行全部配置两遍并写原始 JSON，约数分钟。
`report` 和 `audit` 可直接读取提交中的 `.json.gz`，无需重新仿真。
如单独运行 `run --stage heldout`，先解压 `baseline_per_window.json.gz`；
开发集参数选择应先完成并冻结，不能用留出集重选。

## 文件与统计口径

| 文件 | 内容 |
| --- | --- |
| `compare.csv` | 5 个设计×2 模式×3 派工；分 B2/4/8/16/64/96/128 的延迟几何平均，单位 ms |
| `hbm_bytes.csv` | 4,050 个逐窗口记录；与同设计、同模式的离线流量相比 |
| `gpqa_t128_case.md` | 指定失败窗口的 Shared 核归属、开始时刻、层延迟、流量及剩余任务级原因 |
| `regression.csv` | 540 项单核旧/新完整结果逐位对照 |
| `selection.json` | 18 开发窗口的统一参数选择、200 次 bootstrap；不使用留出集选择参数 |
| `frozen_designs.json` | 两模式的全部 E4 资源账本与原文件校验值；port_tight B2 为 3×16×128 两核 |
| `e4_reproduction.csv` | 2,700 项重新生成的旧 EFT/离线记录与 E4 原记录一致性 |
| `compare_predictor_control.csv` | 旧 EFT＋同样的 ours 预测器与预热协议；学习反馈会随派工不同 |
| `paired_improvement.csv` | 新派工与 B1/B2 等参照的逐窗口配对比值、2,000 次 bootstrap 的 95% 区间 |
| `excess_hbm_windows.csv` | 全部旧/新在线流量超过离线 2% 的窗口及原因，不隐去未达标项 |
| `task_hbm_deltas.csv` | 新派工与离线的每个非零专家流量差，逐层求和校验 |
| `INDEPENDENT_AUDIT.json` | 独立审计：资源、冻结校验、有限队列、物理授予、准入比较、完整重复结果 |
| `executions/` | 实际命令、开始/结束时间、日志、源 SHA256、输入 SHA256、执行基底提交和退出码 |
| `*.json.gz` | 完整原始数值证据，无删行或舍入；原 JSON 保留在本地但不提交 |
| `RAW_ARCHIVES.json` / `PROVENANCE.csv` | 压缩往返逐字节校验及交付文件 SHA256 |

5400 个留出记录包含额外的旧 EFT＋ours 控制组；主三组为4050个。
参数选择阶段共1800个记录，每项也完整重复。阈值与排序开关对所有组织和
模式相同。GM≤1.01 是总体异构在线/离线验收，GPQA 单窗口的两项阈值均为2%。

执行时新文件尚未提交：各 receipt 的 `execution_commit` 是当时仓库基底，
实际新实现由 receipt 中的源文件 SHA256 标识，不能把基底 SHA 当成含修复代码的提交。
数值执行之后的报告文字及审计更新有独立 receipt；没有改动数值执行源代码。
