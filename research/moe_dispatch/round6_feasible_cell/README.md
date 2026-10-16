# 第六轮：带宽可行、最强同构、在线兑现

主条件：真实prefill Shared+Routed；完整专家矩阵+向量；普通供数理想，Shared额外副本通过共同10 TB/s片上端口流式广播。A10/A3.3是必要平均带宽筛选，不等同有限HBM系统验证。

第五轮冻结提交：`b095a7de67a0be07802df0fb1fa2c2b3063bdc75`，已推送`github/research/moe-supply-first-v3`。第五轮文件不再修改。

## 协议补充

- B4096由两个不同的第五轮2048-token请求组在Router之后拼接，开发/留出不混合。新增15窗口/划分，总计105+105。
- 12/13个相同合法核不能精确填满2的幂次预算。纳入所有不超预算的合法尺寸，并报告物理MAC数与未使用额度；其他组织沿用严格预算。
- 各组织在开发集按七Batch等权GM选择一套硬件，留出全部Batch冻结。另给按Batch开发选型参考。
- 最强同构分母：每个格子在开发集同Batch上选择H或H+，然后冻结到留出；不允许在留出挑分母。另提供比H/H+留出较快者更严格的诊断。
- 带宽主值10 TB/s，E3敏感性5/20 TB/s；A3.3仅为数值预算档，不用未经核实的产品名称指代。
- Shared按256 KiB权重块流式分发：源数据读一次，额外k−1个接收副本占广播端口。每个接收核复用当前块处理自己的所有Shared行片；Gate/Up和Down之间有cohort屏障。两块缓冲即可，不假设每核完整33 MiB权重常驻。
- 普通权重供给仍理想；有限端口只显式收费额外Shared复制。这是分析性广播实现契约，报告单列纯算术与广播延迟，不声称有限带宽端到端系统性能。
- 不能机械套用“其他核更晚就解除份额限制”：那会使EFT总能逃逸配额。实现比较替代核完成时刻与预计整层排空时刻，仍允许必要的配额逃逸；文档和消融明确说明。

## 提交台账

| 阶段 | 提交 |
|---|---|
| 第五轮完整归档 | b095a7de67a0be07802df0fb1fa2c2b3063bdc75 |
| 第六轮E1 | 07e165a18db560015de971b359b59239d0b3c922 |
| 第六轮E2 | 待完成 |
| 第六轮E3与总报告 | 待完成 |

## 运行命令

在仓库根目录执行，Python需要NumPy、SciPy、OR-Tools、Matplotlib、pytest和可用的g++/OpenMP。本次使用 `/tmp/plena-round2-venv/bin/python`。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round6_feasible_cell.run_all --stage search
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round6_feasible_cell.run_all --stage evaluate
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round6_feasible_cell.run_all --stage report
/tmp/plena-round2-venv/bin/python -m pytest -q research/moe_dispatch/round6_feasible_cell/test_round6.py research/moe_dispatch/round5_largebatch_compute/test_round5.py
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round6_feasible_cell.audit
```

`run_all --stage all`依次搜索、选δ、评价冻结硬件/纯矩阵/无广播/按batch选型附表、生成报告。五个预算作为独立进程并行；每个整数求解器固定单worker和确定性工作预算。脚本不执行git推送，阶段提交在审计后单独完成。

所有正式输出都在本目录，临时可重建矩阵缓存在 `/tmp/plena-round6-feasible-cache`。现有完整产物会跳过；要从头复算请在独立checkout中使用本目录代码与已冻结的第五轮输入，并为 `common.CACHE` 指定独立缓存路径，以免覆盖这次冻结结果。
