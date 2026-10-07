#!/usr/bin/env python3
"""Summarize measured spatial-M reports; no new simulation or fitted timing."""
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    root = args.results
    rows = read_csv(root / "all_points.csv")
    val = json.loads((root / "validation.json").read_text())

    def get(family, workload, shape, mode="pinned_expert", latency=25, ii=1):
        return next(r for r in rows if r["family"] == family and r["workload"] == workload
                    and r["shape"] == shape and r["ownership"] == mode
                    and int(r["latency"]) == latency and int(r["ii"]) == ii)

    def cycles(*args):
        return int(get(*args)["cycles"])

    lb_rows = []
    for window in ["b8_full", "b32_full", "b32_tokens8_31"]:
        row = get("trace", window, "1+1+1+1+1+1", "tile_stealing")
        work = int(row["useful_macs"])
        # At II=1, at most total_multipliers useful MACs can start per cycle.
        # Earliest issue is 0; final issue must finish L cycles later.
        work_bound = 25 + (work + 12287) // 12288 - 1
        # Four dependent K segments for these full-dimension trace GEMMs.
        dependency_bound = 4 * 25
        lower_bound = max(work_bound, dependency_bound)
        assert int(row["cycles"]) >= lower_bound
        lb_rows.append(dict(window=window, useful_macs=work, work_lower_bound=work_bound,
                            dependency_lower_bound=dependency_bound, lower_bound=lower_bound,
                            measured_uniform_cycles=int(row["cycles"]),
                            reaches_bound=int(row["cycles"]) == lower_bound,
                            scope="formula lower bound: fixed L25 II1, ideal interfaces, 12288 multipliers"))
    with (root / "lower_bounds.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lb_rows[0])); w.writeheader(); w.writerows(lb_rows)

    selected = []
    selection = [r for r in rows if r["family"] == "trace" and r["workload"] == "b8_full"]
    for category in ["uniform", "heterogeneous_pair"]:
        eligible = [r for r in selection if
                    (len(set(r["shape"].split("+"))) == 1 if category == "uniform" else
                     len(r["shape"].split("+")) == 2 and r["shape"] != "3+3")]
        winner = min(eligible, key=lambda r: (int(r["cycles"]), r["shape"], r["ownership"]))
        validation = get("trace", "b32_tokens8_31", winner["shape"], winner["ownership"])
        selected.append(dict(category=category, shape=winner["shape"], policy=winner["ownership"],
                             selection_cycles=int(winner["cycles"]), validation_cycles=int(validation["cycles"])))
    (root / "joint_selection_validation.json").write_text(json.dumps(selected, indent=2) + "\n")

    positive = cycles("sustained", "4_2", "3+3") / cycles("sustained", "4_2", "4+2")
    b8_gain = cycles("trace", "b8_full", "3+3") / cycles("trace", "b8_full", "4+2")
    hold_slow = cycles("trace", "b32_tokens8_31", "4+2") / cycles("trace", "b32_tokens8_31", "3+3") - 1
    b8_vsbest = cycles("trace", "b8_full", "4+2") / cycles("trace", "b8_full", "1+1+1+1+1+1", "tile_stealing") - 1
    hold_vsbest = selected[1]["validation_cycles"] / selected[0]["validation_cycles"] - 1
    toy_table = []
    for family, workload, label in [
        ("toy", "4_2", "Me=4、2，N=4/K=512"),
        ("sustained", "4_2", "Me=4、2，N=512/K=2048"),
        ("sustained", "3_3", "Me=3、3，N=512/K=2048")]:
        vals = [cycles(family, workload, s) for s in ["6", "3+3", "4+2"]]
        strong = cycles(family, workload, "1+1+1+1+1+1", "tile_stealing")
        toy_table.append(f"| {label} | {vals[0]:,} | {vals[1]:,} | {vals[2]:,} | {strong:,} |")
    trace_table = []
    for window, label in [("b8_full", "B8：选型，17 专家/64 路由行"),
                          ("b32_full", "B32：重叠参考，23 专家/256 路由行"),
                          ("b32_tokens8_31", "B32 后 24 token：验证，20 专家/192 路由行")]:
        vals = [cycles("trace", window, s) for s in ["6", "3+3", "4+2"]]
        strong = cycles("trace", window, "1+1+1+1+1+1", "tile_stealing")
        trace_table.append(f"| {label} | {vals[0]:,} | {vals[1]:,} | {vals[2]:,} | {strong:,} |")
    report = f"""# Spatial-M 机制实验：实现与结论

**结论：物理 M 并行的形状匹配机制成立，但当前证据不支持把固定大小核作为论文主收益。**
正例中 4+2 比 3+3 快 {positive:.3f}×；反例中同构获胜。允许更多同构小核与任务拆分后，
这些存档窗口都能达到当前理想计算模型的周期下界。后续应先比较供数/广播/控制的真实成本，
暂不进入大型 DSE、HBM 机制开发或论文收益定稿。

## 1. 实现了什么

新增独立 Rust `moe_spatial_m` 二进制：事件推进、有限结果流水线、逐输出 K 依赖、
实际 BF16 输入/权重的 FP32 树归约与累加。每次发射只处理一个专家；不同专家可连续进入
同一核的流水线，不强制排空。支持整专家固定归属和 tile 拆分派工。

这是**并行点积阵列的计算模型**，并非已校准的二维脉动阵列，也未接入旧 Compiler lowering。
旧路径 M 是时间批处理；新路径 M 是物理行并行。新预算 12,288 个乘法器，旧实验为 4,096，
因此这里的周期不能直接和旧 oracle 的微秒表相除。

| 配置 | 物理 M×N×K（每核） | 核数 | 总乘法器 | 完整占用时每周期权重输入元素需求* |
|---|---|---:|---:|---:|
| 单核 | 6×4×512 | 1 | 12,288 | 2,048 |
| 同构对 | 3×4×512，两核 | 2 | 12,288 | 4,096 |
| 异构对 | 4×4×512 + 2×4×512 | 2 | 12,288 | 4,096 |
| 细粒度同构 | 1×4×512，六核 | 6 | 12,288 | 12,288 |

*各核逻辑输入需求，未扣跨核广播；不是 HBM 字节。所有配置理想激活接口最多输入
6×512=3,072 个 BF16 元素/周期。L=25、II=1 时有限结果槽合计 2,400 B；
元数据槽为每核 25 个，其他流水线寄存器、SRAM、控制/互连面积尚未计价。等乘法器不等于等面积。

## 2. 机制结果（Rust 模拟周期，越小越好）

主配置 L=25 周期、II=1；这是明确假设，未经过综合标定。前三列整专家固定归属，
末列启用相同模型提供的 tile 拆分；两种策略对所有配置均已跑，完整值见 CSV。

| GEMM 输入 | 单核 6 | 同构 3+3 | 异构 4+2 | 同构 1×6 + 拆分 |
|---|---:|---:|---:|---:|
{chr(10).join(toy_table)}

单 tile 中一轮/两轮发射的时间分别是 25/26，**不能直接说 2×**。
连续独立 tile 才能隐藏填充延迟，使正例达到 {positive:.3f}×。
将 Me 换为 3、3，3+3 又更合适。因此只看正例不能主张普遍收益。
L1/II1 与 L25/II25 敏感性也完成；这是改变流水线假设，不是芯片实测。

调度会改变答案：连续 Me=4、2 时，3+3 固定/拆分为 1048/792，4+2 为 536/678。
简单 FIFO 拆分会破坏原本合适的形状匹配；“能拆”不等于“拆了更快”。
当前固定归属使用相同 LPT 启发式，拆分按就绪组 FIFO 和核编号取任务，并非最优调度证明。

## 3. 有界 trace 选型结果

读取存档路由的 Me，执行 gate GEMM 形状 N=512、K=2048；无 shared expert，
不包含 up/down、激活函数、router 或完整模型推理。规模扫描只沿物理 M 分区：
6、3+3、2+2+2、1×6，以及 1+5/2+4/4+2/5+1；每种各跑固定和拆分。

| 输入 | 单核 6 | 同构 3+3 | 异构 4+2 | 同构 1×6 + 拆分 |
|---|---:|---:|---:|---:|
{chr(10).join(trace_table)}

B8 固定派工时 4+2 相对 3+3 为 {b8_gain:.3f}×；验证窗口反而慢 {hold_slow:.2%}。
B8 上选出的异构形状+策略是 {selected[1]['shape']} / {selected[1]['policy']}，
相对在 B8 选出的最好同构慢 {b8_vsbest:.2%}，验证窗口慢 {hold_vsbest:.2%}。

**防止样本重叠：**完整 B32 的前 8 个 token 与 B8 完全相同，不能叫独立测试集。
选择在 B8 上进行；验证只取 B32 token 8–31。二者仍来自同一档案族，不能证明跨模型泛化。

**公式推导：**在固定 L=25、II=1 下，总周期至少为
`max(25 + ceil(useful_MAC / 12288) - 1, ceil(K/512) × 25)`。
六个 1-row 核加拆分在三个窗口分别为 5,486 / 21,870 / 16,408 周期，恰好达到该下界。
这说明在本次理想接口、固定流水线假设下，这些窗口的异构设计没有剩余的纯计算优势空间；
不等于物理系统中六小核一定最好。

`challenger.csv` 同时记录逐批选最优二分区的 **restricted batch oracle**。
它只枚举两核分区、沿用给定派工策略，重配置计时为零；不代表任意动态可分割阵列的最优值。

## 4. 怎么验证

{val['points']} 点 × 2 = **{val['runs']} 次**；{val['numerical_points']} 点实际执行生成的 BF16 数值，
{val['shape_only_points']} 点仅模拟存档形状时序。数值点包含两个完整 N/K 的路由形状抽检，
但操作数仍是可复现生成值，未执行原始 MX 权重数据或原模型。

每点两次完整 JSON 完全一致；所有调用逐项检查覆盖、无重复、K 顺序、发射间隔、完成时间、
有限在途容量、有效/浪费 MAC、排空。数值同时对 Rust 标量参考和独立 Python 整数参考逐位比对。
小型 dyadic BF16 fixture 的和可被 FP32 精确表达；另测非结合性反例确认实际使用树归约。
Rust workspace **297 项测试通过**（原有 289 + 新增 8）；新二进制 clippy 零警告。
继承的 271 个文件逐个比对未改，原有冻结实验未重跑或覆盖。

## 5. 决策与下一步

已测支持“空间形状与 Me 分布匹配可减少尾块”；不支持“异构必然胜过最好同构”，
也没有证明 paper novelty。旧 oracle 的负结果仍适用于旧时间 M 架构，两组证据不冲突。

下一阶段最小问题应是：**在等权重接口、广播能力、SRAM 端口和控制预算下，
少数宽核的核内权重复用，能否比很多窄核更划算？**
连续 4/2 正例里，4+2 固定归属消耗 2,097,152 个逻辑权重元素，六小核为 6,291,456 个，
后者 3×；相同权重可经缓存/广播复用，因此不能把这 3× 当成 HBM 流量。

先建有限权重端口和显式跨核广播的对照，再比较固定分区、可分割阵列、形状感知派工；
只有相同约束下仍有稳定收益，才扩大多模型独立路由 DSE。attention、量化、RTL/PPA 延后。

复现和边界见 `METHODS.md`；全部数值见 `results/` 下 CSV、请求与压缩逐事件报告。
"""
    (root.parent / "REPORT_ZH.md").write_text(report)
    decision = dict(
        mechanism_demonstrated=True, generic_heterogeneous_advantage_demonstrated=False,
        bounded_trace_validation_gain=False, uniform_reaches_ideal_compute_lower_bound=all(r["reaches_bound"] for r in lb_rows),
        proceed_large_dse=False, proceed_memory_implementation=False,
        next_research_gate="costed weight interfaces/multicast/control under equal budgets; preserve strong uniform challenger",
        old_temporal_m_results_superseded=False)
    (root.parent / "conclusions.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
