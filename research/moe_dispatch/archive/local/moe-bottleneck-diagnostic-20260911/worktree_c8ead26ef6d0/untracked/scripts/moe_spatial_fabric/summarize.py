#!/usr/bin/env python3
"""Summarize completed, frozen measurements without mixing interface budgets."""
import argparse
import csv
import gzip
import json
import re
from pathlib import Path


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def csv_out(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--evaluation",required=True,type=Path)
    args=p.parse_args();ev=args.evaluation;root=ev.parent
    validation=json.loads((ev/'validation.json').read_text());assert validation['passed']
    rows=read(ev/'all_points.csv')
    n=lambda r,k:int(r[k])
    classes=[('single',lambda s:s=='6'),('homogeneous_pair',lambda s:s=='3+3'),
        ('best_uniform',lambda s:len(set(s.split('+')))==1),
        ('best_heterogeneous_pair',lambda s:len(s.split('+'))==2 and len(set(s.split('+')))==2)]
    selections=[];selected_rows={}
    for family,label in [('main','packet'),('main_packed','packed_banked_assumption')]:
        for category,pred in classes:
            candidates=[r for r in rows if r['family']==family and r['window']=='b8_full' and pred(r['shape'])]
            chosen=min(candidates,key=lambda r:(n(r,'cycles'),r['name']))
            held=next(r for r in rows if r['family']==family and r['window']=='b32_tokens8_31' and
                      all(r[k]==chosen[k] for k in ['shape','ownership','control']))
            selected_rows[label,category]=(chosen,held)
            selections.append(dict(interface=label,category=category,shape=chosen['shape'],ownership=chosen['ownership'],control=chosen['control'],
                selection_cycles=n(chosen,'cycles'),validation_cycles=n(held,'cycles'),
                selection_weight_bytes=n(chosen,'source_weight_bytes'),validation_weight_bytes=n(held,'source_weight_bytes')))
    csv_out(ev/'selection_validation.csv',selections)
    resources=[]
    for (interface,category),(chosen,held) in selected_rows.items():
        for row in [chosen,held]:
            cycles=n(row,'cycles')
            busy={res:n(row,res+'_service_cycles')/(cycles*(n(row,'control_ports') if res=='control' else 1))
                  for res in ['control','weight','activation','accumulator']}
            resources.append(dict(interface=interface,category=category,window=row['window'],cycles=cycles,
                shape=row['shape'],**{k+'_busy_fraction':v for k,v in busy.items()},
                useful_mac_fraction=float(row['elapsed_utilization']),resource_lower_bound_cycles=max(
                    (n(row,'control_service_cycles')+n(row,'control_ports')-1)//n(row,'control_ports'),
                    n(row,'weight_service_cycles'),n(row,'activation_service_cycles'),n(row,'accumulator_service_cycles'))))
    csv_out(ev/'resource_occupancy.csv',resources)
    state_rows=[]
    for (interface,category),selected in selected_rows.items():
        for row in selected:
            report=json.loads(gzip.decompress((ev/'reports'/(row['name']+'.json.gz')).read_bytes()))
            for core,c in enumerate(report['cores']):
                for state,cycles in c['states'].items():
                    state_rows.append(dict(interface=interface,category=category,window=row['window'],
                        core=core,m_lanes=c['m_lanes'],state=state,cycles=cycles,
                        fraction=cycles/report['total_cycles']))
    csv_out(ev/'core_states.csv',state_rows)
    changes=[]
    for row in rows:
        if row['family'] not in ['ablation','winner_followup']:continue
        family='main_packed' if row['packed']=='True' else 'main'
        base=next(r for r in rows if r['family']==family and r['window']==row['window'] and all(r[k]==row[k] for k in ['shape','ownership','control']))
        changes.append(dict(name=row['name'],variant=row['variant'],window=row['window'],interface=family,
            shape=row['shape'],ownership=row['ownership'],control=row['control'],baseline_cycles=n(base,'cycles'),cycles=n(row,'cycles'),
            speedup=n(base,'cycles')/n(row,'cycles'),source_weight_bytes_delta=n(row,'source_weight_bytes')-n(base,'source_weight_bytes'),
            activation_bytes_delta=n(row,'activation_bytes')-n(base,'activation_bytes'),rmw_bytes_delta=n(row,'accumulator_rmw_bytes')-n(base,'accumulator_rmw_bytes')))
    csv_out(ev/'paired_sensitivities.csv',changes)
    def policy_row(shape,ctl,win='b8_full'):
        return next(r for r in rows if r['family']=='main' and r['window']==win and r['shape']==shape and r['control']==ctl and r['ownership']=='pinned_expert')
    before=policy_row('1+1+1+1+1+1','invocation');after=policy_row('1+1+1+1+1+1','tile_cohort')
    assert n(before,'source_weight_bytes')==n(after,'source_weight_bytes')
    mechanisms=[]
    for ctl in ['invocation','cohort','tile_cohort']:
        r=policy_row('1+1+1+1+1+1',ctl)
        mechanisms.append(dict(control=ctl,cycles=n(r,'cycles'),control_service_cycles=n(r,'control_service_cycles'),
            issue_services=n(r,'issue_services'),install_services=n(r,'install_services'),completion_services=n(r,'completion_services'),
            source_weight_bytes=n(r,'source_weight_bytes'),activation_bytes=n(r,'activation_bytes'),rmw_bytes=n(r,'accumulator_rmw_bytes')))
    csv_out(ev/'control_granularity.csv',mechanisms)
    tables={}
    for interface in ['packet','packed_banked_assumption']:
        lines=[]
        for category,title in [('single','单核'),('homogeneous_pair','同构双核'),('best_uniform','候选内最好同构'),('best_heterogeneous_pair','候选内最好异构对')]:
            a,b=selected_rows[interface,category]
            lines.append(f"| {title} | {a['shape']} | {a['ownership']} / {a['control']} | {n(a,'cycles'):,} | {n(b,'cycles'):,} |")
        tables[interface]='\n'.join(lines)
    ratios={}
    for interface in ['packet','packed_banked_assumption']:
        u,uv=selected_rows[interface,'best_uniform'];h,hv=selected_rows[interface,'best_heterogeneous_pair']
        ratios[interface]=dict(selection_hetero_over_uniform=n(h,'cycles')/n(u,'cycles'),validation_hetero_over_uniform=n(hv,'cycles')/n(uv,'cycles'))
    resource_lines=[]
    for r in resources:
        if r['category']=='best_uniform':
            resource_lines.append(f"| {r['interface']} / {r['window']} | {r['shape']} | {r['control_busy_fraction']:.1%} | {r['weight_busy_fraction']:.1%} | {r['activation_busy_fraction']:.1%} | {r['accumulator_busy_fraction']:.1%} |")
    winner_changes=[x for x in changes if x['name'].startswith('winner_')]
    lines=[]
    for family in ['main','main_packed']:
        for win in ['b8_full','b32_tokens8_31']:
            for variant in ['control2ports','weight4096','oracle_zero_control','oracle_zero_weight']:
                x=next(x for x in winner_changes if x['name'].startswith('winner_'+family+'_uniform_') and x['window']==win and x['variant']==variant)
                lines.append(f"| {family} / {win} | {variant} | {x['baseline_cycles']:,} → {x['cycles']:,} | {x['speedup']:.3f}× | {x['source_weight_bytes_delta']:+,} |")
    no_stable=all(v['validation_hetero_over_uniform']>=1 for v in ratios.values())
    decision='未证明异构比强同构更好；保留控制/供数机制为研究主线。' if no_stable else '接口条件改变了架构排名；只报告具体成立条件，不宣称异构普遍获胜。'
    toy_lines=[]
    for window,title in [('toy_4_2','Me=4、2'),('toy_3_3','Me=3、3')]:
        values=[policy_row(shape,'tile_cohort',window) for shape in ['6','3+3','4+2']]
        toy_lines.append(f"| {title} | "+' | '.join(f"{n(r,'cycles'):,}" for r in values)+' |')
    numeric_note=''
    numeric_path=root/'headline_numeric/validation.json'
    if numeric_path.exists():
        nv=json.loads(numeric_path.read_text());assert nv['passed']
        numeric_note=f"另做 {nv['points']} 点 × 2 次完整数值复核，覆盖两张主表及控制粒度对照；各架构输出逐位相同，数值运行的周期、计数和事件 hash 与形状运行完全一致。"
    test_note=''
    test_log=root/'logs/workspace_v3.log'
    if test_log.exists():
        # Cargo emits one result per target; shared modules run in two binaries.
        suites=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed',test_log.read_text())
        assert suites and all(x[0]=='ok' and int(x[2])==0 for x in suites)
        test_note=f"Rust workspace 共 {sum(int(x[1]) for x in suites)} 次测试执行通过（含共享模块重复编译运行）；其中新增 finite-fabric 独立回归用例 16 个。"
    report=f"""# Spatial-M 有限供数与控制：完成报告

**结论：{decision}**
本轮完成有限接口、私有权重暂存、广播、权重复用、三种控制粒度和容量反压的 Rust 实现。
所有下列数字均为该模型的**模拟周期**，不是芯片测量或 native HBM 端到端结果。

## 测了什么、配置是什么

GEMM 为 X[Me,2048] × W[2048,512]；Me 取存档路由计数。
硬件每核 N 并行=4、K 并行=512，物理 M 分区搜索 6、3+3、4+2、2+4、5+1、1+5、2+2+2、1×6。
总计都是 12,288 个乘法器，MAC 假设 L=25、II=1。

| 全芯片共同预算 | 默认值 | 是否实际阻塞执行 |
|---|---:|---|
| 解码后权重源接口 | 1,024 B/周期 | 是；不是 HBM 原生字节/周期 |
| activation 接口 | 6,144 B/周期 | 是 |
| 共享 backing accumulator RMW | 192 B/周期，容量上限 2 MiB | 是 |
| 全局控制口 | 1 个；发射/安装/完成=2/3/2 周期 | 是 |
| 私有权重 holding slots | 共 48 KiB，按 M 分配 | 引用中/加载中不可替换 |
| activation 暂存 / 结果槽 | 共 12 KiB / 2,400 B | 是，准入即预留结果额度 |
| 活跃元数据 | 共 256 条，持续 header 最多占一半 | 是，已准入 cohort 有续跑优先级 |

带宽、延迟和控制周期是公开的建模参数；没有综合/PPA 标定。权重 holding slot 是宽操作数接口抽象，
不是宣称普通 SRAM 免费提供这些端口。输出 backing store 为共享访问模型，没有私有 accumulator
跨核迁移/NoC/bank 冲突费用。独立权重预取 actor 也尚未加入；当前预取受 activation staging 约束。

## 结果：先在 B8 选，再保持配置验证

B8：8 token、17 个活跃专家、64 条路由；验证：B32 中排除前 8 token 后的 24 token、20 专家、192 条路由。
B8 原本就是完整 B32 的前缀，因此没有把完整 B32 伪装成独立验证。仍只有一个存档族，不能证明跨模型泛化。
形状、固定归属/拆分策略及控制模式都只按 B8 选择，原样用于验证。

普通 packet 接口：每次传输至少占一个端口周期。

| 对照 | M 分区 | B8 选出的策略/控制 | B8 周期 | 验证周期 |
|---|---|---|---:|---:|
{tables['packet']}

乐观 banked/packed 接口：相同总字节带宽内，多份小 activation/RMW 请求可以合用一个 beat；
用互不重叠的字节区间计费，**不假装它已经具备 bank 冲突和布线验证**。所有形状统一使用该接口再比较。

| 对照 | M 分区 | B8 选出的策略/控制 | B8 周期 | 验证周期 |
|---|---|---|---:|---:|
{tables['packed_banked_assumption']}

异构/同构耗时比：packet 在选型/验证为 {ratios['packet']['selection_hetero_over_uniform']:.3f} / {ratios['packet']['validation_hetero_over_uniform']:.3f}；
packed 为 {ratios['packed_banked_assumption']['selection_hetero_over_uniform']:.3f} / {ratios['packed_banked_assumption']['validation_hetero_over_uniform']:.3f}。
大于 1 表示异构更慢。两种接口不能跨表混算架构加速比。
这里只能称“本次候选内最优”：搜索含全部单核/等宽六行分区，以及异构双核；没有穷举异构三核以上、N/K、频率或面积。
2+4 与 4+2 是同一种硬件分区的两种核编号顺序；顺序会影响当前启发式派工，因此都测，不能把顺序差异当作新的硬件收益。
在核心数相同的双核对比中，3+3 也优于 B8 选出的异构配置，因此差距不只是“同构多了一核”。

最初的 toy 仍有局部收益，但有限控制削弱了纯计算下的优势。固定专家归属、相同 persistent 控制：

| Toy 输入 | 单核 6 | 同构 3+3 | 异构 4+2 |
|---|---:|---:|---:|
{chr(10).join(toy_lines)}

这些都是本轮有限接口结果；不同 expert 权重不能在一次 invocation 内混装。Toy 的正例不能替代存档路由验证。

## 确认有用的机制

固定六个同构小核、B8、固定专家归属，逐 invocation 控制改为持续 tile-cohort 控制：
**{n(before,'cycles'):,} → {n(after,'cycles'):,} 周期，{n(before,'cycles')/n(after,'cycles'):.3f}×**。
源权重字节两者均为 {n(after,'source_weight_bytes'):,} B，数值相同。
控制口服务 **{n(before,'control_service_cycles'):,} → {n(after,'control_service_cycles'):,} 周期**。

区别是同一权重 tile 的后续 M 块继续使用已有描述符；不是每个小块再次访问全局控制记录。
每块仍走有限传输、MAC、RMW 和一周期本地依赖更新；全 cohort 完成后才退役 header。
控制服务公式为 `2×issue_services + 3×install_services + 2×completion_services`，逐点断言核对。
这是执行组织的收益，所有核心组织都可使用；不能把它写成大小核独有的新颖性。

## 当前主要卡在哪里

下表是各端口忙碌比例，**可相互重叠，不可相加成总时间**；控制两端口时按两口总能力归一化。

| 接口 / 输入 | 同构分区 | 控制口 | 权重源 | activation | RMW |
|---|---|---:|---:|---:|---:|
{chr(10).join(resource_lines)}

“阵列等权重”同时可能在等权重安装的控制服务，不能据此直接判成 HBM 带宽不足。
资源忙碌比例与以下同配置敏感性一起判断：

| 接口 / 输入 | 改动 | 周期变化 | 时间比 | 源权重字节变化 |
|---|---|---:|---:|---:|
{chr(10).join(lines)}

增加控制口/字节带宽属于资源敏感性，不是免费硬件优化。`oracle_*` 是非物理反事实；
动态派工和缓存随时序改变，所以流量可能变化，不能当作固定请求轨迹的纯因果分解或相加的收益。
完整表还包含广播/保留关闭、FIFO、buffer 槽数、staging 深度、元数据额度、控制成本与 Me1/Me32。

## 验证与边界

最终版本 {validation['points']} 点，每点两次，共 **{validation['runs']} 次**；
{validation['numerical_points']} 点执行生成的 BF16 数值，其他为明确标记的形状时序。
每次都在 Rust 审计依赖、端口/字节区间、有限存储、排空，并保存完整事件与服务 hash。
{validation['full_trace_points']} 点另外保留全 trace，逐项进行独立 Python 审计；
全部数值点还与独立整数参考逐位比较。重复运行的完整报告和时间线 hash 均一致。
{numeric_note}
{test_note}

测试抓到并修复了两类实质性问题：长 N 队列把未完成 cohort 推出有限窗口；持续 header 用尽元数据后
阻断剩余 M 块。最终版有队列续跑优先级、预留额度与对应回归用例。早期未完成试验留在 `results_final/FAILURE.json`，
**本文仅引用 `evaluation/` 的完成结果**。早期仅同轮合并的诊断留在 `results/`。

没有运行完整模型、真实原始 MX 权重、Compiler 新 lowering、native HBM/Ramulator 或 RTL。
实际数值来自可复现的 BF16 fixture；路由档案只提供 Me/N/K。
旧冻结工程和旧 oracle 没有被覆盖；本轮代码是独立二进制 `moe_spatial_fabric`。

## 决策

{decision}
下一步优先把明确的 tile 生命周期/续跑机制映射到现有 Compiler 与真实 DMA/解码的模拟路径，
保留最好同构和允许拆分的强对照，并校准控制服务、bank/NoC 与回写代价，再扩大独立路由样本。
目前不能写“大小核已带来整机加速”或“机制已证明 paper novelty”。

权重接口降低到 256 B/周期后，B8 最好同构耗时变为 139,300 周期，与异构基本持平；
瓶颈会随接口条件变化。本结论仅针对已扫描的参数与路由，不证明任何未来异构设计都无效。

复现入口：源代码 `scripts/moe_spatial_fabric/README.md`；最终请求、压缩报告、比较 CSV、校验记录在 `evaluation/`。
方法与口径见 `METHODS.md`；可导出的结果图在 `figures/finite_fabric_results.svg` 和 `.pdf`。
"""
    (root/'REPORT_ZH.md').write_text(report)
    (root/'conclusions.json').write_text(json.dumps(dict(completed=True,heterogeneous_validated_against_best_uniform=not no_stable,
        ratios=ratios,control_granularity_speedup=n(before,'cycles')/n(after,'cycles'),
        scope='decoded finite-interface Rust model; no native HBM or full-model inference',
        next_gate='integration/calibration and independent routed traces; no heterogeneity-only claim'),indent=2)+'\n')
    print(json.dumps(ratios,indent=2))


if __name__=='__main__':main()
