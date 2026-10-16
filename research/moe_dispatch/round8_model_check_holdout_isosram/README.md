# 第8轮：模型核对、留出验证、同SRAM峰值读宽DSE

已按确认完成：ceil主模型，原OS敏感性，PN≤256与512均完整执行。所有报告时间为µs、带宽为TB/s。第2～7轮不修改；未提交或推送。

- [完整报告](results.md)：诊断表、2A～2C、3A～3C，OS敏感性与旧表9定义在附录。
- [两页诊断](diagnosis.pdf)、[源码与证据](evidence.md)。
- results/study/{issue_formula,plena_os}_cap{256,512}/table{2A,2B,2C,3A,3B,3C}.{csv,md}：24组表。
- [summary.json](results/study/summary.json)：全部未舍入表格。
- [splits.json](results/study/splits.json)：种子20261016，50次划分的逐窗ID。
- [split_results.csv](results/study/split_results.csv)：1200条划分×模型×cap×组织×调度结果；[split_batches.csv](results/study/split_batches.csv)给出8400条逐batch结果。
- [全部候选](results/study/all_candidate_scores.csv)：564条模型×cap×配置。
- [带宽档选型](results/study/iso_selected.csv)：60条结果，包含无可行解。
- [逐窗分配重算](results/study/assignment_replay.json.gz)：42120点精确/贪心周期、归属和载荷。
- [表9注释](results/study/table9_annotated.csv)：112行旧表原数值，逐行注明旧G=MLEN/BLEN、本轮G=PK及旧MLEN。
- [校验记录](results/study/VALIDATION.json)、[测试结果](results/study/TESTS_FULL.txt)、[路由来源](results/trace_inventory.csv)。

## 协议

沿用R7的135窗：B2/4/8/16各27窗，B64/96/128各9个decode+prefill混合窗。每窗2个Shared任务（各F1408，M=B）与所有激活Routed。专家不拆核；BF16/FP32、1GHz、65536乘法器、PK64、PN为16倍数；PM任意正整数。异构小核8K/16K/24K/32K，允许32K+32K异形。

第2步每次71搜索窗、64评估窗，搜索集只按精确最优成本选型；冻结形状后分别最优/贪心评估。每batch窗口算术平均，再取7-batch等权GM。2C按R7全池选型给配对t区间，不是独立留出证据；未多重校正。

第3步按要求使用全部135窗重新DSE；SRAM峰值上限8.2/16.6/33.3/50.0/不限TB/s。约束是硬件读宽过滤，不是端口时序模拟，同上限不要求各设计用尽预算。Pareto采用两指标均不差且至少一项更好；相同坐标保留并列。读宽先按整数B/cycle求和再转换，避免浮点相加制造假Pareto点。

HBM Roofline逐窗取max(Compute,唯一权重/4 TB/s)再平均，假设完美重叠，是下界而非实测E2E。不含向量、路由、控制、容量、PPA。没有采集新路由。

## 复现

工作目录：/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator

~~~bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round8_model_check_holdout_isosram.diagnose
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round8_model_check_holdout_isosram.report
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round8_model_check_holdout_isosram.study
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round8_model_check_holdout_isosram.study_report
/tmp/plena-round2-venv/bin/python -m pytest -q research/moe_dispatch/round8_model_check_holdout_isosram > research/moe_dispatch/round8_model_check_holdout_isosram/results/study/TESTS_FULL.txt
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round8_model_check_holdout_isosram.audit
~~~

study重新计算156个不同几何×135窗×2模型，每点最优两遍、贪心两遍，核对R7数据/代码哈希；50次搜索评估及带宽档汇总也重复两遍，要求逐位一致。沿用无超时精确Pareto-DP，并另用小例子完全枚举交叉检查。

第1步results/VALIDATION.json是历史诊断范围记录（stage2/3=false），当前完成状态在results/study/VALIDATION.json。report只写诊断，study_report负责完整报告，诊断重生成不会覆盖完整结果。
