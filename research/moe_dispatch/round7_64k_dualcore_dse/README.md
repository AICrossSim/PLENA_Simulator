# 64K单核／同构双核／异形双核精确DSE

交付入口：[完整九张表及全部敏感性](results/results.md)、[跨模型汇总](results/sensitivity_summary.csv)、[审计记录](results/VALIDATION.json)。

本次沿用用户确认的135个原始窗口及实际分组：B2/4/8/16各27窗，B64/96/128各9窗。前四档是decode，后三档是历史混合decode/prefill窗口。所有实验在当前分支新增目录完成，没有修改第二至第六轮已冻结文件。

`plena_os`是按任务书“优先复用”要求调用第五轮的OS投影递推，三次独立Gate/Up/Down；`issue_formula`完全执行任务书的三维ceil乘积。两种模型分别穷举，不能混用其时间。每个模型都运行overhead=0/16和PN cap=256/512，共8组，每组九张CSV与Markdown表。

目录：

- `config.json`：全部搜索、频率、精度、参考带宽和输入参数。
- `trace.py`、`inputs/`：原始trace核验、HF配置、135窗逐专家计数与来源。
- `model.py`：合法几何、两种耗时函数、逐发射SRAM流量。
- `exact.cpp`、`scheduler.py`：精确Pareto DP及LPT/EFT贪心。
- `search.py`、`metrics.py`：穷举、每窗测量、先batch内算术均值再七batch GM。
- `legacy.py`：旧四形状重放与逐项差异归因。
- `report.py`：九张表及全部对照。
- `test_round7.py`、`audit.py`：手算、暴力穷举、真实路由守恒、独立CP-SAT及全输出审计。

运行（仓库根目录）：

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
/tmp/plena-round2-venv/bin/python -m pytest -q research/moe_dispatch/round7_64k_dualcore_dse/test_round7.py
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round7_64k_dualcore_dse.run_all
```

需要Python、g++（C++17）；独立审计使用OR-Tools。主精确求解器不依赖CP-SAT，且无超时、近优截断或启发式取代最优。测试使用pytest。复用第五轮已提交的`model.py`；旧结果重放还使用仓库内`scaling_g64`及其旧时序依赖，源文件哈希记在`legacy_receipt.json`。

完整穷举按cap256共126个候选、cap512共156个候选执行。不同cap共同的几何只计算一次并用于两组结果；旧异构核编号镜像另加一个参照，共157种执行配置×135窗×4种时序/overhead组合，每点两次。所有主/敏感性表共72对CSV/Markdown。

最优assignment并不唯一：DP按核0负载递增及最大核1节省量做确定性选择，不另优化SRAM字节。因此所报SRAM流量对应归档的具体最优assignment，不宣称在所有等时最优分配中也是最少流量。HBM唯一权重字节不依赖assignment。

没有执行git提交或推送；本次请求的交付是实验代码、结果与数据表。
