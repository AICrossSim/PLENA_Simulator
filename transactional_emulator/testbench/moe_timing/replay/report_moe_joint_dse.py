#!/usr/bin/env python3
"""Publish only after exhaustive-grid and finalist gates pass; audit artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path
from compare_moe_normal import digest,read_json,require
from run_moe_dma_campaign import save
from run_moe_joint_dse import FIXTURES,load_report,fingerprint

LABELS={'single':'单核','homogeneous':'同构双核','equal_pe_different_shape':'等 PE 异形双核','large_small':'3072+1024 大小核'}
ORDER=['single','homogeneous','equal_pe_different_shape','large_small']


def geomean(values):
    require(bool(values) and all(math.isfinite(v) and v>0 for v in values),'invalid latency')
    return math.exp(sum(math.log(v) for v in values)/len(values))


def shape_text(point):
    return ' + '.join('B{} K{}'.format(*s) for s in point['axes']['shape'])


def audit(root,plan,ranking,finalists):
    require(digest(root/'repro/moe_dual_normal')==plan['binary_sha256'],'binary drift')
    require(digest(root/'repro/libramulator.so')==plan['native_sha256'],'native drift')
    ids={p['id'] for p in plan['representatives']}
    require(len(ids)==len(plan['representatives']),'duplicate point identity')
    require({v['representative'] for v in plan['valid_variants']}==ids,'variant coverage mismatch')
    require({r['point']['id'] for r in ranking['ranked']}==ids,'ranking coverage mismatch')
    files={}
    for p in plan['representatives']:
        path=root/'architectures'/(p['id']+'.json')
        require(digest(path)==p['architecture_sha256'],'configuration drift')
        for f in FIXTURES:
            folder=root/'runs'/p['id']/f;s=read_json(folder/'status.json')
            report=folder/'report.json.gz';value=digest(report)
            require(s['status']=='passed' and s['numerical_gate']['output_bit_exact'] and value==s['report_sha256'],'invalid grid report')
            files[str(report.relative_to(root))]=value
    for entry in finalists['outcomes']:
        t=entry['task'];s=entry['outcome'];report=root/'runs'/t['label']/t['fixture']/'report.json.gz'
        value=digest(report)
        require(s['status']=='passed' and s['numerical_gate']['output_bit_exact'] and value==s['report_sha256'],'invalid finalist report')
        files[str(report.relative_to(root))]=value
    for group in [plan['fixtures'],read_json(root/'holdout_manifest.json')['fixtures']]:
        for record in group.values():
            folder=Path(record['path']);w=read_json(folder/'workload.json')
            require(digest(folder/'workload.json')==record['workload_sha256'],'workload drift')
            require(digest(folder/'golden.json')==record['golden_sha256'],'golden drift')
            require(digest(folder/w['hbm_file'])==record['hbm_sha256'],'image drift')
    save(root/'artifact_audit.json',dict(status='passed',grid_reports=plan['expected_runs'],
        finalist_reports=len(finalists['outcomes']),lossless_reports=files,
        campaign_sha256=digest(root/'campaign.json'),ranking_sha256=digest(root/'ranking.json'),
        finalist_status_sha256=digest(root/'finalists/status.json')))


def report(root):
    plan=read_json(root/'campaign.json');status=read_json(root/'status.json')
    require(status['status']=='passed' and status['passed']==status['expected']==plan['expected_runs'],'grid incomplete')
    ranking=read_json(root/'ranking.json');finalists=read_json(root/'finalists/status.json')
    require(finalists['status']=='passed' and finalists['completed']==finalists['expected'],'finalists incomplete')
    for category,w in ranking['winners'].items():
        expected=min((r for r in ranking['ranked'] if r['point']['axes']['category']==category),
                     key=lambda r:(r['geomean_ps'],r['point']['id']))
        require(w==expected,'winner is not the declared fixed-configuration minimum')
    audit(root,plan,ranking,finalists)
    primary={c:{f:ranking['winners'][c]['measurements'][f]['total_ps'] for f in FIXTURES} for c in ORDER}
    holdout={c:{} for c in ORDER};sensitivity={}
    for entry in finalists['outcomes']:
        t=entry['task'];value=entry['outcome']['total_ps'];c=t['category'];f=t['fixture']
        if t['kind']=='holdout':holdout[c][f]=value
        if t['kind']=='sensitivity':sensitivity.setdefault(t['mode'],{c:{} for c in ORDER})[c][f]=value
    scores={}
    for mode,data in [('primary',primary),('holdout',holdout)]+sorted(sensitivity.items()):
        mean={c:geomean(list(data[c].values())) for c in ORDER}
        scores[mode]=dict(geomean_ps=mean,speedup_vs_fixed_single={c:mean['single']/mean[c] for c in ORDER},timings_ps=data)
    per_fixture_oracle={f:{c:min((r for r in ranking['ranked'] if r['point']['axes']['category']==c),
        key=lambda r:(r['measurements'][f]['total_ps'],r['point']['id']))['point']['id'] for c in ORDER} for f in FIXTURES}
    save(root/'summary.json',dict(status='passed',grid_runs=status['passed'],finalist_runs=finalists['completed'],
        total_numerical_runs=status['passed']+finalists['completed'],scores=scores,
        winners={c:ranking['winners'][c]['point'] for c in ORDER},per_fixture_oracle=per_fixture_oracle,
        scope=plan['timing'],not_full_model_or_isoarea=True))
    csv_rows=[]
    for r in ranking['ranked']:
        p=r['point'];a=p['axes'];row=dict(point=p['id'],category=a['category'],shape=shape_text(p),
            slots=a['slots'],credits=a['credits'],layout=a['layout'],threshold=a['threshold'],
            representative_allocation=a['allocation'],geomean_ms=r['geomean_ps']/1e9)
        for f,m in r['measurements'].items():
            row[f+'_ms']=m['total_ps']/1e9;row[f+'_hbm_bytes']=m['hbm_read_bytes']
        csv_rows.append(row)
    with (root/'all_points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    lines=['# MoE 大小核联合 DSE 完整结果','',
        '本轮搜索和复测全部完成。以下收益仅指固定路由 MoE FFN 的 Rust 数值执行与 Ramulator 仿真；计算核/SRAM/DMA frontend 时序仍含分析假设，不是等面积 RTL 或完整模型收益。','',
        '## 覆盖与校验','',
        '- {} 种核心组合；{} 个可行参数变体，{} 个资源不可行变体；按逐任务准入等价合并为 {} 个代表。'.format(len(plan['topologies']),len(plan['valid_variants']),len(plan['excluded']),len(plan['representatives'])),
        '- 全网格 {} 次、胜者复测/holdout/敏感性 {} 次，共 {} 次完整数值执行；全部 BF16 输出逐位等于独立 oracle，HBM 字节、SRAM、队列和端口检查通过。'.format(status['passed'],finalists['completed'],status['passed']+finalists['completed']),
        '- 每类仅选一套在四个调参输入上几何平均最优的固定配置。主配置在原输入共执行三次，holdout 执行两次；完整数值、时间和 native 计数重复一致。','',
        '## 各类固定胜者','',
        '| 类别 | 核心形状 | 预取槽/credits | 布局/阈值 | 调参几何平均 ms | 相对单核加速 |',
        '|---|---|---|---|---:|---:|']
    for c in ORDER:
        p=ranking['winners'][c]['point'];a=p['axes'];s=scores['primary']
        lines.append('| {} | {} | {}/{} | {}/{} | {:.6f} | {:.4f}× |'.format(LABELS[c],shape_text(p),a['slots'],a['credits'],a['layout'],a['threshold'],s['geomean_ps'][c]/1e9,s['speedup_vs_fixed_single'][c]))
    lines+=['','加速比 = 固定单核耗时 / 对应方案耗时；大于 1 才更快。B 为共同 M/N 分块宽度，K 为 MLEN。','',
        '## 逐输入延迟（ms）','', '| 输入 | 单核 | 同构双核 | 等 PE 异形双核 | 大小核 | 大小核相对单核耗时变化 |','|---|---:|---:|---:|---:|---:|']
    for group in [primary,holdout]:
        for f in sorted(group['single']):
            times=[group[c][f]/1e9 for c in ORDER]
            lines.append('| {} | {} | {:+.2f}% |'.format(f,' | '.join('{:.6f}'.format(t) for t in times),(times[-1]/times[0]-1)*100))
    lines+=['','## 固定胜者的计时敏感性','',
        '以下没有在每种假设下重新搜索全网格，只检验已经选定配置的结论稳定性；不能称为该假设下的全局最优。乐观直通省略查询/回包计时，作为有利于简单路径的对照。','',
        '| 条件 | 同构双核/单核加速 | 等 PE 异形双核/单核加速 | 大小核/单核加速 |','|---|---:|---:|---:|']
    for mode,s in sorted(scores.items()):
        lines.append('| {} | {} |'.format(mode,' | '.join('{:.4f}×'.format(s['speedup_vs_fixed_single'][c]) for c in ORDER[1:])))
    lines+=['','## 胜者的搬运与计算证据','',
        '| 输入/类别 | HBM 读取 MB | 实际字节/总时间 GB/s | 有效 MAC/发射 MAC | 各核计算忙碌比例 |','|---|---:|---:|---:|---|']
    for c in ORDER:
        p=ranking['winners'][c]['point']
        for f in FIXTURES:
            r=load_report(root/'runs'/p['id']/f/'report.json.gz')['result']
            lines.append('| {}/{} | {:.3f} | {:.2f} | {:.3f} | {} |'.format(f,LABELS[c],r['hbm_read_bytes']/1e6,
                r['hbm_read_bytes']*1000/r['total_ps'],r['useful_macs']/r['issued_macs'],
                ', '.join('{:.1%}'.format(core['compute_busy_fraction']) for core in r['cores'])))
    lines+=['','## 核心等待分解（ms）','',
        '各核可并行，下面的时间不能跨核心相加当成总延迟。weight-ready 等待包括整个取数/解码路径，并不等于纯 HBM Controller 排队时间。', '',
        '| 输入/类别/核心 | 计算忙碌 | 等权重就绪 | 等累加依赖 | 流水排空 |', '|---|---:|---:|---:|---:|']
    for c in ORDER:
        p=ranking['winners'][c]['point']
        for f in FIXTURES:
            r=load_report(root/'runs'/p['id']/f/'report.json.gz')['result']
            for core in r['cores']:
                values=[core[k]/1e9 for k in ['compute_busy_ps','weight_ready_wait_ps','accumulator_dependency_stall_ps','pipeline_drain_ps']]
                lines.append('| {}/{}/{} | {} |'.format(f,LABELS[c],core['id'],' | '.join('{:.6f}'.format(v) for v in values)))
    lines+=['','本轮内核固定按 N tile → K tile → M block 遍历，没有搜索跨 N tile 交错发射。较短 K 的核可能更多次等待同一 accumulator；这一调度限制也应纳入后续分析，不能将其直接归为物理核心的固有限制。','']
    lines+=['','## 适用边界与证据索引','',
        '- HBM 相同，4096 PE 和各存储总容量相同。端口实现、布线、功耗和面积未由 RTL/PPA 校准，不能从 PE 相同推导面积相同。',
        '- 输入为 Qwen D=2048/F=512 与 DeepSeek D=2048/F=1408 的真实归档 decode 路由重组，B8/B32；权重和激活是非零合成值，经过实际本地 MX codec；不是实际完整模型推理、prefill 或 agent trajectory。',
        '- holdout 使用未参与调参的路由坐标，但仍来自相同模型家族/捕获数据和合成值方法，不能夸大为广泛泛化证明。',
        '- 本轮不包含 transpose、更多小核、输入/输出 DMA 竞争、完整 ISA 多核运行。不能将 normal-buffer 搜索结果替代会议整套架构验收。',
        '- `campaign.json`：冻结搜索网格、所有排除项、等价映射、资源和输入哈希。',
        '- `all_points.csv` / `ranking.json`：全部代表的四输入延迟与固定配置排名。',
        '- `finalists/execution_plan.json` / `finalists/status.json`：胜者冻结及全部复测结果。',
        '- `artifact_audit.json`：所有 gzip 原始报告哈希及输入/执行文件复核。',
        '- `summary.json`：可程序读取的结果；`runs` 中保留全部原始数值和 native 统计。',
        '- `calibration` 的冻结前试跑不计入本报告；原有 396 次 DMA 实验不混入本轮次数。','']
    (root/'RESULT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(scores,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);report(p.parse_args().root.resolve())
