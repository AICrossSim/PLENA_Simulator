# MoE 纯计算上限、几何 DSE 与供数反推

范围：BF16 输入/权重，FP32 部分和，12,288 个乘法器，假定 1 GHz。
继承 PLENA **研究计算合同**的 dot20、commit1、II1；不是原生 Rust/RTL
对所有新几何的时序测量。HBM、SRAM/NoC 端口、容量、DMA、router、SiLU、
combine 和派工服务不进入计算目标。供数需求是反推量，不是实测带宽。

## 两套执行契约

1. `search.py`：整专家逐个执行，Gate/Up 记录联合交织，输出全部驻留；
   用精确整数 DP 分配专家。主域含 48,599 个几何，三个时序假设。
2. `pooled_search.py`：同一核可交织不同专家的独立输出；每个输出的下一 K 段
   仍须等上段提交，Down 仍须等本专家 Gate/Up 完成。专家不跨核。
   全域 issue-load 下界配合真实事件回放；不先假定 RR 是最优指令调度。
   本次开发集的三个选定点恰好达到各自全域下界，差距为零。

第二种契约更接近无限共享上下文假设，但需要约 11 MiB FP32 部分和，
并有更多描述符、供数和控制需求。它不能免费继承第一种循环的 W/X 复用。

每套硬件均只在 18 个开发窗口中选择，冻结后评估 135 个历史留出窗口。
七个 batch 等权，先取 batch 内窗口 GM，再取七个 batch 的 GM。
历史留出已被先前实验使用，不属于新盲测。另有 Qwen 路由微批重放作为
第一种契约的冻结硬件补充检查；不声称真实新执行的批量模型推理。

## 搜索域与证明范围

- 主域：全部合法正整数 M，N 为 4 的倍数，K=16…2048、步长 16。
- 单核、同构双核、形状异构双核，总乘法器严格等于 12,288。
- 二次幂不是额外限制。异构也不强制为 4+2，或强制两核算力不同。
- 全域下界用于安全剪枝；不可能超过当前可行点的几何无需重复事件回放。
- 最优性/下界证书只覆盖声明的几何、时序和整专家归属契约。
- `global_ceiling.py` 另给不限制专家拆分的 useful-MAC 理论下界；
  它不是已经实现的调度程序。

## 运行

在仓库根目录执行，Python 需要 NumPy；绘图另需 Matplotlib，Excel 另需
openpyxl。首次运行请使用新的输出目录，避免覆盖本次已有结果。

```bash
python -m research.moe_dispatch.compute_ceiling.search --out /path/to/new-output
python -m research.moe_dispatch.compute_ceiling.ready_pool_diagnostic --out /path/to/new-output
python -m research.moe_dispatch.compute_ceiling.pooled_search --out /path/to/new-output/pooled --max-actual-per-family 8
python -m research.moe_dispatch.compute_ceiling.pooled_original_reference --out /path/to/new-output/pooled
python -m research.moe_dispatch.compute_ceiling.pooled_supply --out /path/to/new-output/pooled
python -m research.moe_dispatch.compute_ceiling.global_ceiling --out /path/to/new-output/pooled
python -m research.moe_dispatch.compute_ceiling.report --out /path/to/new-output
python -m pytest research/moe_dispatch/compute_ceiling -q
```

绘图使用 `plots.generate_plots(output_path)`。开发集事件搜索可缓存相同的
专家维度序列；留出评估的两次事件回放不使用该缓存，结果逐位比较。

## 本次交付

结果目录：`/scratch/shared/mcl123/plena/outputs/moe_compute_ceiling_20261010`。
`SUMMARY.md` 与 Excel 分开呈现两套契约；`pooled/` 是无限上下文的更强基线。
带宽区分冷加载、共享池到核、阵列交付、部分和读写。pooled 共享池到核
的具体 W/X 复用吞吐尚未由循环事件验证，明确标为未定；不填造数。

容量、带宽和物理面积不是同一个量。等乘法器不代表等面积；新几何的
时钟、bank、互连、寄存器和控制成本需要后续综合/原生校准。
