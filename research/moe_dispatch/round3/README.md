# 第三轮可复现入口

分支 `research/moe-supply-first-v3`；所有新代码／数据在本目录，第二轮树未修改。使用 Python 3.11 虚拟环境 `/tmp/plena-round2-venv/bin/python`，工作目录为 simulator worktree 根。BF16；解析估计，非 native／RTL。

实际 Python、依赖版本和宿主 C 编译器见 ENVIRONMENT.json；requirements-repro.txt 锁定本轮直接依赖。宿主 C 编译器只用于整数搜索加速，不模拟 HBM 或 RTL。

| 阶段 | 提交 | 运行命令 | 交付目录 |
| --- | --- | --- | --- |
| E0 | edc2cd7a | python -m research.moe_dispatch.round3.reproduce --jobs 6 | E0/ |
| E1 | 9f42d695 | 阅读 E1/OPERATING_POINT.md（纯规格文档） | E1/ |
| E2 | a82abc6c | python -m research.moe_dispatch.round3.bounds --jobs 4 | E2/ |
| E3 | b2e8189c | python -m research.moe_dispatch.round3.ablation --jobs 3 | E3/ |
| E4 | c2687cc0 | python -m research.moe_dispatch.round3.dse --jobs 8 --candidates 256 --nodes 2048；python -m research.moe_dispatch.round3.finalize_dse；python -m research.moe_dispatch.round3.sensitivity --stage synthetic --jobs 8 --candidates 8 --nodes 32；python -m research.moe_dispatch.round3.robust_heldout --jobs 3 | E4/ |
| E5 | 尚未入提交账本 | python -m research.moe_dispatch.round3.evaluations --stage dispatch --jobs 6；python -m research.moe_dispatch.round3.evaluations --stage cross --jobs 6；python -m research.moe_dispatch.round3.evaluations --stage predictor --jobs 6；python -m research.moe_dispatch.round3.sensitivity --stage sobol --jobs 32 --candidates 8 --nodes 32 | E5/ |

上表命令用实际 Python 路径替换 python。E4 与 E5 的主表共用冻结后的完整回放结果；跨带宽采用同一新控制协议，126 预测器历史对照采用 fixed_legacy。在线派工／预测方法从相同初始状态先运行 18 开发窗口再跑 135 留出窗口；离线 MILP＋LPT 参照没有历史预测状态，不做历史预热。

真实执行命令、时间、退出码、日志、源和输入 SHA256 保存在 executions/；被修正的报告或尝试保留原 receipt／archive，不覆盖第二轮证据。

数值输出全部完成后的初次收尾顺序：`python -m research.moe_dispatch.round3.validate --partial` → `python -m research.moe_dispatch.round3.report` → `python -m research.moe_dispatch.round3.figure_clarify` → `python -m research.moe_dispatch.round3.validate`。完整验收通过后再生成报告和绘图清单，使报告引用完整验收；已有完整交付可直接运行 validate。测试命令：`python -m pytest research/moe_dispatch/round3 -q`。

REPORT_ZH.md 给出 Q1–Q5、主表与局限。figures/ 包含五份 PDF。全局零差距证明未闭合时，必须引用 proof_status.csv 的开放区域与剩余差距，不能把最好已评估候选称全局最优。
