#!/usr/bin/env python3
"""Read-only analysis of completed joint-runtime experiments.

This program chooses no hardware or policy. It verifies each declared point's
two full reports before producing matched comparisons. Incomplete coverage is
an error, not an invitation to summarize the completed fast points.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import joint_study as study
import run_experiments as run
from robust_study import csv_out


def gm(xs):
    xs=list(xs)
    assert xs and min(xs)>0
    return math.exp(math.fsum(map(math.log,xs))/len(xs))


def average(xs):
    xs=list(xs)
    return math.fsum(xs)/len(xs) if xs else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root,split):
    study.check_plan(root)
    policy_set='all' if split=='test' else 'pilot'
    expected=study.points(root,split,policy_set,'shared_first')
    name=f'{split}_shared_first_{policy_set}'
    receipt=json.loads((root/f'{name}_receipt.json').read_text())
    assert receipt['complete'] and receipt['points']==len(expected)
    rows=json.loads((root/f'{name}_rows.json').read_text())
    assert len(rows)==len(expected)
    indexed={r['point']:r for r in rows}
    assert len(indexed)==len(rows)
    assert set(indexed)=={p['key'] for p in expected}, 'Missing/unexpected declared points'
    binaries=set();bundles=set();reports={};weight_by_workload=defaultdict(set)
    mac_by_workload=defaultdict(set);drained=0;repeat_checks=0
    control_totals=set();joint_bytes=set();peak=[]
    for p in expected:
        row=indexed[p['key']]
        directory=Path(row['raw_directory'])
        one=(directory/'report_repeat1.json').read_bytes()
        two=(directory/'report_repeat2.json').read_bytes()
        assert one==two, f'Repeat mismatch: {p["key"]}'
        raw_sha=hashlib.sha256(one).hexdigest()
        assert row['raw_report_sha256']==raw_sha
        rep=json.loads(one);repeat_checks+=1
        rr=json.loads((directory/'repeat_receipt.json').read_text())
        manifest=json.loads((directory/'point_manifest.json').read_text())
        cfg=json.loads((directory/'config.json').read_text())
        w=json.loads((directory/'workload.json').read_text())
        assert rr['repeat_match'] and rr['repeat_count']==2 and rr['report_sha256']==raw_sha
        assert manifest['point_hash']==row['point_hash']==rr['point_hash']
        assert manifest['workload_sha256']==sha(directory/'workload.json')
        assert manifest['config_sha256']==sha(directory/'config.json')
        binaries.add(manifest['binary_sha256']);bundles.add(manifest['source_bundle_sha256'])
        assert cfg==p['config']
        assert {k:v for k,v in w.items() if k!='engine_layout'}==p['workload']
        assert w['engine_layout']['hardware']==p['resources']
        assert w['engine_layout']['m_lanes']==cfg['lanes']
        assert w['engine_layout']['group']==cfg['group']
        assert all(rep['config'].get(k)==v for k,v in cfg.items())
        assert rep['drained'] is True and rep['ownership_k_order_capacity_checks'] is True
        drained+=1
        assert rep['cycles']==row['latency_cycles_at_1ghz']
        assert row['latency_ms']==rep['cycles']/1e6
        assert rep['dma_transactions_accepted']==rep['dma_transactions_landed']
        assert rep['dma_transactions_accepted']*32==rep['weight_bytes']
        assert rep['credit_peak']<=cfg['credits']==256
        expected_weight=sum(m['physical_bytes'] for e in w['experts'] for m in e['weights'].values())
        expected_mac=sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])
        assert rep['weight_bytes']==expected_weight
        assert rep['useful_macs']==expected_mac
        weight_by_workload[w['id']].add(rep['weight_bytes'])
        mac_by_workload[w['id']].add(rep['useful_macs'])
        control_totals.add(sum(p['resources']['control_bytes']))
        joint_bytes.add(p['resources']['joint_state_bytes'])
        assert rep['pending_window_peak']<=cfg['window']==8
        for i,c in enumerate(rep['cores']):
            s=c['stats'];hw=p['resources'];m=p['config']['lanes'][i]
            assert s['weight_peak_bytes']<=4096*hw['weight_slots'][i]
            assert s['x_peak_bytes']<=2*m*512*2
            assert s['workspace_peak_bytes']<=c['capacity']
            assert c['capacity']==w['engine_layout']['cores'][i]['capacity']
            peak.append(dict(point=p['key'],core=i,workspace_peak_bytes=s['workspace_peak_bytes'],
                private_capacity_bytes=c['capacity'],weight_peak_bytes=s['weight_peak_bytes'],
                weight_capacity_bytes=4096*hw['weight_slots'][i],x_peak_bytes=s['x_peak_bytes'],
                x_capacity_bytes=2*m*512*2,contexts_peak=s['contexts_peak']))
        for a in rep['dispatch_audit']:
            assert a['actual_minus_predicted_cycles']==a['actual_finish_cycle']-a['predicted_finish_cycle']
        reports[p['key']]=rep
    assert len(binaries)==1, 'One completed formal suite must use one binary'
    assert len(bundles)==1, 'One formal suite must have one frozen source bundle'
    assert receipt['binary_sha256']==next(iter(binaries))
    assert receipt['source_bundle_sha256']==next(iter(bundles))
    assert all(len(v)==1 for v in weight_by_workload.values()), 'Architecture/policy changed required HBM bytes'
    assert all(len(v)==1 for v in mac_by_workload.values())
    assert control_totals=={4352} and joint_bytes=={256}
    return rows,reports,dict(complete=True,scope=split,declared_points=len(expected),
        full_json_identical_pairs=repeat_checks,completed_runs=2*repeat_checks,
        native_ramulator=False,drained_points=drained,all_resource_checks_passed=True,
        hbm_bytes_equal_across_all_organizations_and_policies=True,useful_macs_equal=True,
        control_reserved_bytes=4352,joint_state_reserved_bytes=256,
        binary_sha256=next(iter(binaries)),source_bundle_sha256=next(iter(bundles)),
        numerical_scope='Separate small seeded timed-trace replay tests; real-route timing is metadata-only.',
        per_core_peak_checks=peak)


def tables(rows,reports):
    by=defaultdict(list);lookup={}
    for r in rows:
        by[(r['budget_group'],r['organization'],r['mode'])].append(r)
        lookup[(r['budget_group'],r['organization'],r['mode'],r['workload'])]=r
    policies=[];architecture=[];diagnostics=[];bounds=[];shared=[]
    for key,rs in sorted(by.items()):
        budget,org,policy=key
        s=dict(budget=budget,shape=org,policy=policy,windows=len(rs),
               geomean_latency_ms=gm(r['latency_ms'] for r in rs))
        for b in (2,4,8,16):s[f'B{b}_mean_ms']=average(r['latency_ms'] for r in rs if r['batch']==b)
        for baseline in study.MAIN:
            ratios=[lookup[(budget,org,baseline,r['workload'])]['latency_ms']/r['latency_ms']
                    for r in rs if (budget,org,baseline,r['workload']) in lookup]
            if len(ratios)==len(rs):
                s[f'speedup_over_{baseline}']=gm(ratios)
                s[f'worst_slowdown_percent_vs_{baseline}']=max(0,100*(1/min(ratios)-1))
        policies.append(s)
        mono='6' if budget=='M6' else '8'
        homo='3+3' if budget=='M6' else '4+4'
        for other in (mono,homo):
            ratios=[lookup[(budget,other,policy,r['workload'])]['latency_ms']/r['latency_ms'] for r in rs]
            architecture.append(dict(budget=budget,shape=org,baseline_shape=other,policy=policy,
                windows=len(rs),speedup_same_policy=gm(ratios),
                best_speedup=max(ratios),worst_speedup=min(ratios),
                worst_slowdown_percent=max(0,100*(1/min(ratios)-1))))
        audit=[a for r in rs for a in reports[r['point']]['dispatch_audit']]
        joint=[reports[r['point']].get('joint_diagnostics',{}) for r in rs]
        candidates=[c for d in joint for a in d.get('audit',[]) for c in a.get('candidates',[])]
        errors=[a['actual_minus_predicted_cycles'] for a in audit]
        d=dict(budget=budget,shape=org,policy=policy,windows=len(rs),bindings=len(audit),
            bindings_with_both_due_cores=sum(len(a['eligible_cores'])==2 for a in audit),
            fraction_bindings_with_both_due_cores=sum(len(a['eligible_cores'])==2 for a in audit)/len(audit),
            prediction_mae_us=average(abs(e)/1000 for e in errors),
            worst_underestimate_us=max([0]+errors)/1000,
            core_finish_gap_mean_us=average(r['core_finish_gap_cycles']/1000 for r in rs),
            total_control_service_mean_us_not_wall=average(r['control_service_ns_sum_not_wall_time']/1000 for r in rs),
            next_ready_at_promotion=sum(r['next_ready_at_promotion'] for r in rs),
            next_inflight_at_promotion=sum(r['next_inflight_at_promotion'] for r in rs),
            next_weight_wait_core_cycles_sum_not_wall=sum(r['next_weight_wait_cycles'] for r in rs),
            candidate_observations=len(candidates),
            candidate_potential_both_fraction=sum(c.get('potential_mask',0)==3 for c in candidates)/len(candidates) if candidates else None,
            candidate_due_both_fraction=sum(c.get('eligible_mask',0)==3 for c in candidates)/len(candidates) if candidates else None)
        for field in ('snapshot_count','control_cycles','pair_comparisons','paired_rounds','single_rounds',
                      'reordered_bindings','aging_forced_rounds','revalidation_cancellations',
                      'deferred_placements','preferred_core_wait_cycles','late_floor_blocked_core_cycles'):
            d['joint_'+field]=sum(j.get(field,0) for j in joint)
        diagnostics.append(d)
        for r in rs:
            report=reports[r['point']]
            bytebound=math.ceil(report['weight_bytes']/r['hbm_bytes_per_ns'])
            creditbound=math.ceil(report['weight_bytes']*r['hbm_latency_ns']/(32*r['credit_limit']))
            bounds.append(dict(point=r['point'],budget=budget,shape=org,policy=policy,
                workload=r['workload'],hbm_read_bytes=report['weight_bytes'],latency_ms=r['latency_ms'],
                byte_bound_ms=bytebound/1e6,credit_occupancy_bound_ms=creditbound/1e6,
                latency_over_stronger_supply_bound=report['cycles']/max(bytebound,creditbound),
                achieved_weight_GBps=report['weight_bytes']/report['cycles'],
                note='Resource bound, not exclusive memory-stall time; credits return after finite landing.'))
            for a in report['dispatch_audit']:
                if a['expert']==-1:
                    shared.append(dict(point=r['point'],budget=budget,shape=org,policy=policy,
                        workload=r['workload'],batch=r['batch'],core=a['core'],
                        bind_us=a['cycle']/1000,actual_finish_ms=a['actual_finish_cycle']/1e6,
                        predicted_finish_ms=a['predicted_finish_cycle']/1e6,
                        eligible_cores=a['eligible_cores'],split=a.get('split',False)))
    return dict(policy_summary=policies,architecture_same_policy=architecture,
                controller_diagnostics=diagnostics,supply_bounds=bounds,shared_bindings=shared)


def order_diagnostic(root):
    # Missing original-order runs are a stated coverage gap, never inferred.
    records={}
    for path in sorted(root.glob('development_*_rows.json')):
        if '_filter_' in path.name:continue
        receipt=path.with_name(path.name.replace('_rows.json','_receipt.json'))
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        for r in json.loads(path.read_text()):
            key=(r['descriptor_order'],r['budget_group'],r['organization'],r['mode'],r['workload'])
            if key in records:assert records[key]['raw_report_sha256']==r['raw_report_sha256']
            records[key]=r
    result=[]
    for key,last in records.items():
        order,budget,org,policy,workload=key
        if order!='source_shared_last':continue
        first=records.get(('shared_first',budget,org,policy,workload))
        if first is None:continue
        binaries=[]
        for r in (first,last):
            directory=Path(r['raw_directory'])
            one=(directory/'report_repeat1.json').read_bytes()
            assert one==(directory/'report_repeat2.json').read_bytes()
            assert hashlib.sha256(one).hexdigest()==r['raw_report_sha256']
            manifest=json.loads((directory/'point_manifest.json').read_text())
            binaries.append(manifest['binary_sha256'])
        assert binaries[0]==binaries[1], 'Source-order pair must use the same engine binary'
        result.append(dict(budget=budget,shape=org,policy=policy,workload=workload,
            shared_first_ms=first['latency_ms'],shared_last_ms=last['latency_ms'],
            shared_first_speedup=last['latency_ms']/first['latency_ms'],
            first_shared_bind_us=first['shared_bind_cycle']/1000,
            last_shared_bind_us=last['shared_bind_cycle']/1000,
            both_raw_repeats_checked=True,binary_sha256=binaries[0],
            scope='Development-only descriptor-interface sensitivity; not an independent heldout result.'))
    return result


def make_conclusions(tab,checks,order):
    indexed={(s['budget'],s['shape'],s['policy']):s for s in tab['policy_summary']}
    organization=[]
    for budget,hetero,homo,mono in (('M6','4+2','3+3','6'),('M8','5+3','4+4','8')):
        if (budget,hetero,'joint') not in indexed:continue
        a={r['baseline_shape']:r for r in tab['architecture_same_policy']
           if r['budget']==budget and r['shape']==hetero and r['policy']=='joint'}
        s=indexed[budget,hetero,'joint']
        organization.append(dict(budget=budget,heterogeneous_shape=hetero,
            joint_speedup_over_same_hardware_dynamic=s['speedup_over_dynamic'],
            joint_speedup_over_same_window_lpt_ect=s['speedup_over_window_lpt_ect'],
            joint_speedup_over_prior_selected_tail_reference=s.get('speedup_over_selected_tail_reference'),
            joint_heterogeneous_speedup_over_joint_homogeneous=a[homo]['speedup_same_policy'],
            joint_heterogeneous_speedup_over_joint_monolithic=a[mono]['speedup_same_policy'],
            heterogeneous_faster_geomean_than_both_with_same_joint_policy=
                a[homo]['speedup_same_policy']>1 and a[mono]['speedup_same_policy']>1,
            interpretation='These are measured analytical comparisons, not evidence of global optimality or area/energy superiority.'))
    return dict(scope=checks['scope'],complete=checks['complete'],organization=organization,
        selection='No policy or hardware selected from these results; every declared policy reported.',
        source_order_diagnostic_points=len(order),
        limits=['One model, three captured datasets, offline request rebatching, layer13 decode-step7 for fresh test.',
                'No actual full-model inference, router timing, native HBM/Ramulator channel timing, or pretrained numerical tensors.',
                'Predictor state resets each window; no claim of cross-request training.',
                'No area/power/Fmax synthesis. All policies reserve identical extra256B control state inside the fixed2MiB arena.',
                'Front-end/control counters overlap other activity and cannot be summed into global latency.',
                'next_prefetch_tiles counts reservations; actual prefetch usefulness uses readiness/inflight at promotion.'],
        checks={k:v for k,v in checks.items() if k!='per_core_peak_checks'})


def report_zh(root,tab,conclusions,order):
    c=conclusions;scope='独立测试集' if c['scope']=='test' else '开发集诊断'
    text=[f'# 联合派工控制器：{scope}','',
        '**这次实现的是固定硬件上的有限窗口派工：观察任务形状与资源状态，联合选择两核的下一项工作，在预取需要的时间附近提交归属，并用已完成任务校正时间估计。**', '',
        '三种组织都使用相同机制、同一 Shared-first 描述符输入规则和相同新增控制预算。主对比不拆专家；selected_tail_reference 单独保留上一轮冻结的尾专家拆分能力。', '',
        '时间单位为 ms。范围是根据真实路由构造的完整 MoE FFN 分析模拟，包含 Gate/Up、激活、Down 和结果合并；不含 Router、整模型推理或原生 Ramulator。', '',
        '|预算|固定形状 M×4×512|B2|B4|B8|B16|相对同硬件 dynamic|相对同窗口 LPT/ECT|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in tab['policy_summary']:
        if s['policy']!='joint':continue
        text.append(f"|{s['budget']}|{s['shape']}|"+'|'.join(f"{s[f'B{b}_mean_ms']:.6f}" for b in (2,4,8,16))+
            f"|{s['speedup_over_dynamic']:.4f}×|{s['speedup_over_window_lpt_ect']:.4f}×|")
    text += ['', '每个 batch 是三个数据集窗口的算术平均（开发集只有一个窗口）；加速比按配对窗口取几何平均，>1 才是加速。', '',
        '|异构配置|joint 相对单核 joint|joint 相对同构 joint|是否同时超过两者|',
        '|---|---:|---:|---|']
    for s in c['organization']:
        text.append(f"|{s['budget']} / {s['heterogeneous_shape']}|{s['joint_heterogeneous_speedup_over_joint_monolithic']:.4f}×|{s['joint_heterogeneous_speedup_over_joint_homogeneous']:.4f}×|{'是，仅此测试范围' if s['heterogeneous_faster_geomean_than_both_with_same_joint_policy'] else '否'}|")
    text += ['', '**判断顺序：先看新控制器是否超过同样观察多个任务的 window_lpt_ect，再看两种双核是否超过使用相同控制器的单核。只超过 FIFO 不能证明联合机制或异构的必要性。**', '',
        '四个独立消融关闭晚绑定、配对、反馈、Next 预取；完整数据在 policy_summary.csv，未根据测试结果重新选择默认策略。', '',
        '|配置|该次决策快照两核均到期|候选能放两核|候选两核均到期|预测绝对误差 µs|平均两核结束差 µs|',
        '|---|---:|---:|---:|---:|---:|']
    for d in tab['controller_diagnostics']:
        if d['policy']!='joint':continue
        percent=lambda x:'—' if x is None else f'{100*x:.1f}%'
        text.append(f"|{d['budget']}/{d['shape']}|{percent(d['fraction_bindings_with_both_due_cores'])}|{percent(d['candidate_potential_both_fraction'])}|{percent(d['candidate_due_both_fraction'])}|{d['prediction_mae_us']:.3f}|{d['core_finish_gap_mean_us']:.3f}|")
    text += ['', '“候选”按扫描观察次数统计，第一列候选率按实际提交次数统计，使用对应决策快照中的到期掩码；不是提交瞬间再次测量。二者分母不同，不能直接等同于旧版本的绑定选择率。能放入与已经到预取时机分开统计。各核控制服务和等待时间可能重叠，不相加当作总耗时。', '',
        '供数条件仍为共享256B/ns、64ns响应、256个32B请求额度。单看额度和响应，持续供数上界为128B/ns；逐点供数下界见 supply_bounds.csv，它不是 memory-stall 的排他时间。', '',
        f"验收：{c['checks']['declared_points']}点、{c['checks']['completed_runs']}次执行，所有重复的完整JSON一致；请求排空、字节数、有效MAC和私有容量检查通过。真实性能输入没有预训练数值载荷，数值正确性由另外的小张量时序回放验证。", '',
        f'原始 Shared-last 输入顺序的开发集对照：{len(order)} 个配对点；'+('详见 source_order_diagnostic.csv。' if order else '尚无完整配对结果，不据此下结论。'), '',
        '本报告没有自动选出“最佳测试策略”，没有声称全局最优或综合后的面积、功耗、频率。']
    (root/('REPORT_ZH.md' if c['scope']=='test' else 'DEVELOPMENT_REPORT_ZH.md')).write_text('\n'.join(text)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=study.DEFAULT_ROOT)
    p.add_argument('--scope',choices=('test','development'),default='test')
    p.add_argument('--write-report',action='store_true')
    a=p.parse_args();root=a.output
    rows,reports,checks=verify(root,a.scope)
    tab=tables(rows,reports);order=order_diagnostic(root)
    conclusions=make_conclusions(tab,checks,order)
    dest=root if a.scope=='test' else root/'development_analysis'
    dest.mkdir(parents=True,exist_ok=True)
    for name,data in tab.items():csv_out(dest/(name+'.csv'),data)
    csv_out(dest/'source_order_diagnostic.csv',order)
    run.write_json(dest/'resource_and_correctness.json',checks)
    run.write_json(dest/'conclusions.json',conclusions)
    if a.write_report:report_zh(dest,tab,conclusions,order)
    print(json.dumps({k:v for k,v in conclusions.items() if k in ('scope','complete','organization')},indent=2))


if __name__=='__main__':main()
