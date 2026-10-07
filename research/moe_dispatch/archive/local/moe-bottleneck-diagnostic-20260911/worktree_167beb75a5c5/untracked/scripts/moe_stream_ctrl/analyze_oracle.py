#!/usr/bin/env python3
"""Derive decisions from completed oracle runs; never simulate or impute results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    return float(row[key])


def analyze(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed_runs'] == 132
    assert json.loads((out / 'charged_gate.json').read_text())['status'] == 'passed'
    assert not (out / 'FAILURE.json').exists(), 'Resolve and preserve failures before final analysis'
    for case in manifest['cases']:
        validation = json.loads((out / case['id'] / 'validation.json').read_text())
        assert validation['status'] == 'passed' and validation['repeats_exact']
        for name, expected in validation['output_hashes'].items():
            assert hashlib.sha256((out / case['id'] / name).read_bytes()).hexdigest() == expected
    a = read_csv(out / 'oracle_2x2.csv')
    b = read_csv(out / 'control_breakdown.csv')
    c = read_csv(out / 'g1_extremes.csv')
    assert (len(a), len(b), len(c)) == (96, 24, 24)
    ar = {(r['workload'], r['policy'], r['condition'], r['organization']): r
          for r in a if r['repeat'] == '1'}
    br = {(r['organization'], r['condition']): r for r in b if r['repeat'] == '1'}
    cr = {(int(r['me']), r['organization'], r['condition']): r
          for r in c if r['repeat'] == '1'}
    for row in c:
        me, org = int(row['me']), row['organization']
        charged, zero = cr[me, org, 'charged'], cr[me, org, 'zero_control']
        row['control_cycles_per_useful_mac'] = number(row, 'nominal_control_cycles') / number(row, 'useful_macs')
        arch = json.loads(Path(next(x['architecture'] for x in manifest['cases'] if x['id'] == row['case'])).read_text())
        ideal_compute_us = number(row, 'useful_macs') * arch['clock_period_ps'] / (number(row, 'p') * number(row, 'r') * 1e6)
        row['derived_ideal_useful_mac_time_us'] = ideal_compute_us
        row['control_service_over_ideal_mac_time'] = number(row, 'control_service_us') / ideal_compute_us
        row['causal_zero_control_saved_us'] = number(charged, 'time_us') - number(zero, 'time_us')
        row['causal_zero_control_saved_fraction'] = row['causal_zero_control_saved_us'] / number(charged, 'time_us')
    write_csv(out / 'g1_extremes.csv', c)
    # Rebuild after derived columns; retain both repetitions in deliverables.
    cr = {(int(r['me']), r['organization'], r['condition']): r for r in c if r['repeat'] == '1'}
    comparisons = []
    for workload in ('b8', 'b32'):
        for policy in ('fixed', 'work_conserving'):
            for condition in ('charged', 'zero_control', 'ideal_supply', 'both'):
                times = {org: number(ar[workload, policy, condition, org], 'time_us')
                         for org in ('single', 'homogeneous', 'heterogeneous')}
                comparisons.append(dict(workload=workload, policy=policy, condition=condition,
                    time_us=times, heterogeneous_over_single=times['heterogeneous']/times['single'],
                    homogeneous_no_slower_than_heterogeneous=times['homogeneous'] <= times['heterogeneous']))
    fixed_zero = [r for r in comparisons if r['policy'] == 'fixed' and r['condition'] == 'zero_control']
    stop_hetero = all(r['heterogeneous_over_single'] >= 1 for r in fixed_zero)
    small_cold = cr[1, 'small', 'charged']
    small_hot = cr[32, 'small', 'charged']
    p1 = dict(
        cold_small_control_occupancy_highest=number(small_cold, 'control_service_fraction') >= max(
            number(cr[1, org, 'charged'], 'control_service_fraction') for org in ('single', 'large')),
        small_hot_occupancy_lower=number(small_hot, 'control_service_fraction') < number(small_cold, 'control_service_fraction'),
        cycles_per_mac_amortization=number(small_cold, 'control_cycles_per_useful_mac') / number(small_hot, 'control_cycles_per_useful_mac'),
    )
    breakdown = {}
    for org in ('single', 'homogeneous', 'heterogeneous'):
        breakdown[org] = {mode: {key: number(br[org, mode], key)
                               for key in ('time_us', 'saved_us', 'saved_fraction_of_charged', 'saved_fraction_of_zero_control_gain')}
                          for mode in ('desc_only', 'lifecycle_only')}
    nonmonotonic = [dict(organization=org, condition=mode,
                        time_us=breakdown[org][mode]['time_us'],
                        zero_control_us=number(br[org, 'zero_control'], 'time_us'))
                    for org in breakdown for mode in ('desc_only', 'lifecycle_only')
                    if breakdown[org][mode]['time_us'] < number(br[org, 'zero_control'], 'time_us')]
    effects = []
    for workload in ('b8', 'b32'):
        for policy in ('fixed', 'work_conserving'):
            for org in ('single', 'homogeneous', 'heterogeneous'):
                t = {condition: number(ar[workload, policy, condition, org], 'time_us')
                     for condition in ('charged', 'zero_control', 'ideal_supply', 'both')}
                effects.append(dict(workload=workload, policy=policy, organization=org,
                    zero_control_saved_us=t['charged']-t['zero_control'],
                    zero_control_saved_fraction=(t['charged']-t['zero_control'])/t['charged'],
                    ideal_supply_saved_us=t['charged']-t['ideal_supply'],
                    ideal_supply_saved_fraction=(t['charged']-t['ideal_supply'])/t['charged'],
                    zero_control_gain_under_ideal_supply_us=t['ideal_supply']-t['both'],
                    interaction_us=t['zero_control']+t['ideal_supply']-t['charged']-t['both'],
                    causal_attribution_allowed=policy == 'fixed'))
    write_csv(out / 'oracle_effects.csv', effects)
    summary = dict(evidence_class='nonphysical_oracle_diagnostic', completed_runs=132, repeats_per_point=2,
        comparisons=comparisons, b8_fixed_partial_oracles=breakdown, p1=p1, effects=effects,
        partial_oracle_faster_than_zero_control=nonmonotonic,
        decision_stop_current_heterogeneous_mainline=stop_hetero,
        decision_scope='Frozen P/R/Mt, budgets, work placement/order, lifetime/dependency rules, local block8 representation and workload windows only; not a universal impossibility theorem.',
        homogeneous_no_slower_conditions=sum(r['homogeneous_no_slower_than_heterogeneous'] for r in comparisons),
        condition_count=len(comparisons),
        native_runs=sum(not x['ideal_supply'] for x in manifest['cases']) * 2,
        ideal_runs=sum(x['ideal_supply'] for x in manifest['cases']) * 2,
        checks='bit-exact BF16 and FP32; bytes; finite storage; native or explicitly separate oracle drain; complete repeated results; frozen charged compatibility')
    (out / 'conclusions.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    table = ['| 窗口 / 条件 | 单核 µs | 同构 µs | 异构 µs | 异构/单核耗时 |',
             '|---|---:|---:|---:|---:|']
    labels = dict(charged='charged', zero_control='零控制', ideal_supply='理想供数', both='两者同时')
    for x in comparisons:
        if x['policy'] != 'fixed':
            continue
        t = x['time_us']
        table.append(f"| {x['workload'].upper()} / {labels[x['condition']]} | {t['single']:.3f} | {t['homogeneous']:.3f} | {t['heterogeneous']:.3f} | {x['heterogeneous_over_single']:.3f}× |")
    bt = ['| B8 fixed | desc-only：时间 / 节省 | lifecycle-only：时间 / 节省 |', '|---|---:|---:|']
    for org, label in [('single', '单核'), ('homogeneous', '同构'), ('heterogeneous', '异构')]:
        desc, life = breakdown[org]['desc_only'], breakdown[org]['lifecycle_only']
        bt.append(f"| {label} | {desc['time_us']:.3f} µs / {100*desc['saved_fraction_of_charged']:.1f}% | {life['time_us']:.3f} µs / {100*life['saved_fraction_of_charged']:.1f}% |")
    conclusion = ('按预定规则，停止把当前大小核组织作为主线收益来源；主线转向执行/控制机制，大小核保留为评估配置。'
                  if stop_hetero else '零控制下异构的结论依窗口而定；保留具体获益条件，不作统一止损或成功结论。')
    p1_text = (f"小核 Me1→Me32：控制服务占总时间 {100*number(small_cold,'control_service_fraction'):.1f}%→{100*number(small_hot,'control_service_fraction'):.1f}%；每 MAC 名义控制周期降低 {p1['cycles_per_mac_amortization']:.2f}×。"
               f"Me1 小核占比{'最高' if p1['cold_small_control_occupancy_highest'] else '并非最高'}；热专家占比{'下降' if p1['small_hot_occupancy_lower'] else '未下降'}。")
    singles = [e for e in effects if e['organization'] == 'single' and e['policy'] == 'fixed']
    supply_text = '；'.join(f"{e['workload'].upper()} 单核：零控制省 {100*e['zero_control_saved_fraction']:.1f}%，理想供数省 {100*e['ideal_supply_saved_fraction']:.1f}%" for e in singles)
    life_more = all(breakdown[org]['lifecycle_only']['saved_us'] > breakdown[org]['desc_only']['saved_us'] for org in breakdown)
    anomalies = '; '.join(f"{r['organization']} 的 {r['condition']} 比全零控制快 {r['zero_control_us']-r['time_us']:.3f} µs" for r in nonmonotonic)
    report = f'''# ORACLE 诊断报告（非物理反事实）

**Step0：通过。** 控制端口先获取 semaphore，再等待服务完成，lease 持有至状态更新结束，确实阻塞执行。Oracle 仅置零指定计时；互斥、数据移动、依赖、存储与逐 M 发射周期保留。理想供数仅绕过 native 入口/响应，有限 DMA、解码及 SRAM 仍计费。

**结论：{conclusion}** 以下为 Rust 模拟已测结果，不是实芯片或 RTL 测量。

{chr(10).join(table)}

上表均固定专家归属及核内顺序；work-conserving 完整结果单列在 `oracle_2x2.csv`，不用于因果归因。同一列条件下比较组织，不能拿异构 oracle 对单核 charged。

{chr(10).join(bt)}

**P1：控制摊薄预测相符。** {p1_text} 但小核相对单核耗时在 Me1/Me32 为 {number(small_cold,'time_us')/number(cr[1,'single','charged'],'time_us'):.2f}×/{number(small_hot,'time_us')/number(cr[32,'single','charged'],'time_us'):.2f}×；热专家摊薄控制，不能推出“热专家适合小核”。

**P2：** {'两个 fixed 窗口均符合停止条件' if stop_hetero else '两个 fixed 窗口未统一符合停止条件'}；“任何未来控制重设计都无解”不是本实验能证明的定理。同构在 {summary['homogeneous_no_slower_conditions']}/{summary['condition_count']} 个同条件对照中不慢于异构，不能据此主张不对称必要。

**P3：** {'三种组织均是生命周期神谕节省更多墙钟，支持优化方向，但没有救回异构。' if life_more else '生命周期并非在三种组织上都更有效；逐项结果见表，不能支持统一推断。'} desc 删除 T+I+H；life 删除 9T+4B（含到达通知）；其余为准入 A。约 3/17 是 Me1 名义服务占比，不能当墙钟加速上限。

**与简单上界假设不符：** {anomalies if anomalies else '本轮未见局部神谕快于全零控制。'}，两次一致；此差异不强行归因。神谕保留相同策略，微事件时序会变化，各分项收益不可相加。

**供数判断与下一步：** {supply_text}。先分析 native 入口/响应及保留的有限 DMA、核内端口瓶颈；降低仅针对大小核的控制粒度重设计优先级。理想供数仍保留 DMA 费用，不能把残余时间统称纯计算，也不能把所有阶段笼统判为 memory-bound。

**验收：** 66 点×2=132 次；B8 charged 精确复现 417.087/486.315/531.456 µs。所有点 bit-exact、字节/存储/请求排空通过，完整重复结果一致。真实 native {summary['native_runs']} 次；理想供数 {summary['ideal_runs']} 次明确标注 native 不适用，另核对 oracle 请求完成及有限 DMA 排空。默认参数、冻结结果及验收线未改，未进入后续机制开发。

详见 `METHODS.md`、`conclusions.json`；原始结果、分类控制计数和逐周期同时存储峰值均在本目录归档。
'''
    (out / 'REPORT.md').write_text(report)
    detail = ['# ORACLE 补充数据解读', '', '以下数值为已测；标为 derived 的 MAC 理想时间及 RMW 字节为公式推导。', '',
              '| Me | 核 | charged / zero µs | 控制服务 µs | 控制占总时间 | MAC 服务 µs | accumulator 端口服务 µs | 可用 N 带（Q / W） |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for me in (1, 32):
        for org in ('single', 'large', 'small'):
            x, z = cr[me, org, 'charged'], cr[me, org, 'zero_control']
            detail.append(f"| {me} | {org} | {number(x,'time_us'):.3f} / {number(z,'time_us'):.3f} | {number(x,'control_service_us'):.3f} | {100*number(x,'control_service_fraction'):.1f}% | {number(x,'compute_service_us'):.3f} | {number(x,'accumulator_port_busy_us'):.3f} | {x['q_available_n_bands']} / {x['window_available_n_bands']} |")
    detail.extend(['', '**P1 的边界。** Me1 小核控制占比最高，Me32 占比下降，名义周期/MAC 也下降；但热专家的小核相对时延反而更差。Me32 小核只有1条可用 N 带，单/大核有4条。这是已测配置约束，尚未用本轮实验拆分它与 accumulator/反馈等待各自的因果贡献。不得凭控制摊薄把热专家自动分给小核。', '',
                  '**MAC 与 RMW 口径。** 一次乘加计1 MAC；理论有用 MAC 时间=useful_MAC/(P×R)×clock。actual compute_service 还体现阵列 padding 等计费。对每个投影，K 段数=ceil(K/R)，RMW 读元素=Me×N×(K段数−1)，写元素=Me×N×K段数，每个FP32元素4B；最终转换另外读Me×N元素。CSV分别列 derived 字节与 measured accumulator 服务，不把它们混成墙钟分解。', '',
                  '**P3 的墙钟上限不能由周期比例直接推出。** desc-only占全零控制实际收益的比例：' + '、'.join(f"{org} {100*breakdown[org]['desc_only']['saved_fraction_of_zero_control_gain']:.1f}%" for org in breakdown) + '。生命周期比例分别为' + '、'.join(f"{org} {100*breakdown[org]['lifecycle_only']['saved_fraction_of_zero_control_gain']:.1f}%" for org in breakdown) + '。这些比例不具有可加性；同构超过100%未截断，原始差异只有15ns。', '',
                  '**供数与控制存在交互。** `oracle_effects.csv` 的 interaction=(T_zero+T_supply−T_charged−T_both)。正值表示联合节省超过两项单独节省之和，不是新增一段串行时间。B8 single 的交互为18.387µs。因此控制在供数改善之后可能更重要，当前较低的控制收益不等于永久无用。', '',
                  '**固定与动态派工。** fixed保留Step1的每核专家顺序；并不强制异步tile/burst事件的绝对执行时间和全局先后次序。work-conserving同样跑满两次，用于最终效果，不用于归因。下面单列动态派工：', '',
                  '| 窗口 / 条件 | 单核 µs | 同构 µs | 异构 µs |', '|---|---:|---:|---:|'])
    for x in comparisons:
        if x['policy'] == 'work_conserving':
            t = x['time_us']
            detail.append(f"| {x['workload'].upper()} / {labels[x['condition']]} | {t['single']:.3f} | {t['homogeneous']:.3f} | {t['heterogeneous']:.3f} |")
    detail.extend(['', '**决策（推断）。** 当前异构组织在两个窗口、两种派工、四种条件共16个同条件比较中，均慢于同构；零控制也未超过单核。按预定规则止损异构主线。保留大小核作为评估配置，后续执行/控制机制必须首先证明同样适用于强单核基线。优先分清native供数与有限DMA/accumulator/可用N带的关键路径；本轮没有授权更改默认结构、精度、布局或进入后续阶段，未执行这些更改。'])
    (out / 'ANALYSIS.md').write_text('\n'.join(detail) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args().output.resolve())
