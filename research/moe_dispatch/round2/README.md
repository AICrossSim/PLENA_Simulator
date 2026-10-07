# 第二轮执行入口

所有脚本从 Simulator 仓库根目录运行。Python 环境 `/tmp/plena-round2-venv/bin/python`，依赖见 requirements.txt。每个指标由原始逐窗口CSV重建。旧实验不覆盖。所有真实负载点完整运行两次并比较完整结果对象；合成每个搜索点的已评估叶也重复两次。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/plena-round2-venv/bin/python -m pytest research/moe_dispatch/round2/test_model.py research/moe_dispatch/round2/test_optimizer.py research/moe_dispatch/round2/test_search.py research/moe_dispatch/round2/test_robust.py -q
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.main_search --stage search --jobs 16 --search-seconds 120
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.main_search --stage validity --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.main_search --stage gaps --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E1 --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E2 --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.regions --stage grid --jobs 16 --point-seconds 2
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.extreme --jobs 8 --max-evaluations 500 --point-seconds 2 --verify-seconds 120
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.robust --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.regions --stage sobol --jobs 16 --point-seconds 2
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.regions --stage flip --jobs 16 --point-seconds 2
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E4 --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E5 --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run --stage E6 --jobs 16
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.figures
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.report
```

E1、E2整层表、E4、E5依赖 `E3/FROZEN_SELECTION.json` 中的开发集冻结选择。选择只用开发窗口；E2单专家先运行不影响选择。E4 需要全部真实窗口下界和数据流结果后再解释。

主搜索时间上限每种模式/每个证明120秒。合成及敏感性每点搜索上限2秒，首次必须评估每族可行见证，因此总setup可超出时间上限并单独报告。完整所有采样点不等于完整最优性证明；证书保留未剪区域和最优性差距。若某族没有合法见证，则写不可行原因，不能填另一个族的数。

继续同一开放B&B证明使用 `resume.py`，严格核对工作负载、参数、engine源码哈希，不重复丢弃开放前沿：

```bash
/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.resume --certificate research/moe_dispatch/round2/results/E3/bnb_pipelined_B.json --seconds 3600
```

网格/Saltelli/翻转切片重跑原命令从逐点checkpoint继续。CMA每次保存实际搜索证书和目标轨迹；最终验证δ0未闭合要保留gap。不要把一次预热当成两次完整测量；不要把逐相位计算服务占用当成有效MAC周期。

边界和合法下界推导见 CONTRACT.md。原任务书逐字保存在 TASK.md。本轮只有分析模型，不是RTL或原生HBM实测；没有匹配的attention/router/norm计时就不填完整模型时间。
