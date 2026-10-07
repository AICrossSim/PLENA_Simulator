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

新增执行入口：`python -m research.moe_dispatch.round2.heldout_runner --jobs 16 --wait-selection` 会等待三种模式的当前源码冻结选择，再依次运行E1、E2整层、E4、E5、E6；每条实际命令和退出码保存在results/executions。主搜索和补充搜索按上面的独立命令执行。

run.py支持`--onchip-mode all/pipelined/port_tight/fixed_issue`。默认all生成正式完整三模式结果；单模式重跑写入`results/isolated_modes/<mode>/`，避免覆盖全模式表。`profile_inputs.py`输出153个真实窗口的专家数、低token专家数、唯一权重与HBM下限；Me≤2仍然是活跃专家，Shared单独统计。

每个微实验两次都清空纯函数cost缓存，真实重算后比较；整层每次运行重建完整运行时状态。连续敏感性τ表示共享W前端每4096B的服务时间，合计带宽min(64×bank_Bpc,4096/τ)，计算发射间隔仍为1；SensitivityParameters额外记录源码哈希，续跑必须一致。保守全域下界忽略该新增带宽上限时只会更松，不会失去合法性。

E5的oracle目前为同政策profile/replay参考，预取和归属改变会产生残差；不把它冒充任务书要求的同一实际调度完美oracle，此子项在最终状态中保留部分完成。所有开放搜索同样保留具体LB/gap/继续命令。
