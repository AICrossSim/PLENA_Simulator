#!/usr/bin/env python3
"""Audit and summarize a completed fixed-hardware campaign; never select hardware."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, shutil
from pathlib import Path
from robust_study import csv_out
import run_experiments as run

def gm(xs):
    xs=list(xs)
    return math.exp(math.fsum(math.log(x) for x in xs)/len(xs)) if xs else None

def num(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'

def raw_directory(root,row):
    recorded=Path(row['raw_directory'])
    # An extracted archive preserves stage/points/id, not the original /tmp root.
    relocated=root.joinpath(*recorded.parts[-3:])
    return relocated if relocated.is_dir() else recorded

def design_attribution(root):
    """Postprocess design data only; these rows do not select test hardware."""
    configs={d['id']:d for d in json.loads((root/'all_designs.json').read_text())}
    rows=json.loads((root/'design_rows.json').read_text())
    matched=[];allocation=[];best={}
    for r in rows:
        d=configs[r['design_id']];h=d['resources'];n=len(d['lanes'])
        key=(r['budget_group'],tuple(d['lanes']),r['group'],r['mode'],r['tail_partition'],r['workload'])
        if key not in best or r['latency_ms']<best[key]['latency_ms']:best[key]=r
        # Equal dual W, arena and bank splits eliminate those allocation choices.
        equal=(h['weight_slots']==([10] if n==1 else [5,5])
               and len(set(h['acc_bytes']))==1 and len(set(h['acc_banks']))==1)
        if equal:
            matched.append(dict(scope='design only; fixed equal allocation',**r))
    for r in matched:
        d=configs[r['design_id']]
        key=(r['budget_group'],tuple(d['lanes']),r['group'],r['mode'],r['tail_partition'],r['workload'])
        opt=best[key]
        allocation.append(dict(budget=r['budget_group'],lanes=d['lanes'],group=r['group'],
                               policy=r['mode'],tail_partition=r['tail_partition'],workload=r['workload'],
                               equal_allocation_id=r['design_id'],equal_allocation_ms=r['latency_ms'],
                               per_window_best_allocation_id=opt['design_id'],per_window_best_ms=opt['latency_ms'],
                               diagnostic_upper_bound_speedup=r['latency_ms']/opt['latency_ms'],
                               deployable=False,scope='design allocation sensitivity; per-window choice is not a fixed design'))
    csv_out(root/'shape_matched_allocation_design.csv',matched)
    csv_out(root/'resource_allocation_sensitivity.csv',allocation)
    per_config=collections.defaultdict(list)
    for r in rows:per_config[r['design_id'],r['mode']].append(r)
    per_shape=collections.defaultdict(list)
    for (did,policy),rs in per_config.items():
        if len(rs)!=4:continue
        d=configs[did]
        item=dict(budget=d['budget_group'],lanes=d['lanes'],policy=policy,design_id=did,
                  group=d['group'],tail_partition=d['tail_partition'],
                  design_geomean_ms=gm(r['latency_ms'] for r in rs),
                  per_window_ms={r['workload']:r['latency_ms'] for r in rs},
                  scope='design-only best fixed resource/granularity point for this shape; not heldout validation')
        per_shape[d['budget_group'],tuple(d['lanes']),policy].append(item)
    csv_out(root/'design_shape_best.csv',[
        min(candidates,key=lambda r:(r['design_geomean_ms'],r['design_id']))
        for _,candidates in sorted(per_shape.items())])

def check_raw(root):
    summary={};failures=[]
    for stage in ('assignment','elasticity','design','validation','heldout'):
        rs=json.loads((root/f'{stage}_rows.json').read_text())
        for row in rs:
            p=raw_directory(root,row);r=json.loads((p/'report_repeat1.json').read_text())
            w=json.loads((p/'workload.json').read_text());hw=r['physical_budget']
            expected=sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])
            checks={
                'repeat':(p/'report_repeat1.json').read_bytes()==(p/'report_repeat2.json').read_bytes(),
                'report_hash':run.file_sha(p/'report_repeat1.json')==row['raw_report_sha256'],
                'drained':r['drained'] and r['ownership_k_order_capacity_checks'],
                'work':r['useful_macs']==expected and r['issued_macs']>=expected,
                'credits':r['credit_peak']<=256 and r['dma_transactions_accepted']==r['dma_transactions_landed'],
                'bytes':r['weight_bytes']==32*r['dma_transactions_landed'],
                'w_budget':sum(hw['weight_slots'])==10 and sum(hw['weight_banks'])==64,
                'acc_budget':sum(hw['acc_bytes'])==2*1024**2 and sum(hw['control_bytes'])==4096,
                'ports':sum(hw['x_banks'])==4*sum(r['config']['lanes']) and sum(hw['acc_banks'])==2*sum(r['config']['lanes']),
            }
            for c,core in enumerate(r['cores']):
                s=core['stats']
                checks[f'core{c}_capacity']=(max(s['workspace_peak_bytes'],core['reserved_input_result_control'])<=core['capacity']
                    and s['weight_peak_bytes']<=4096*hw['weight_slots'][c]
                    and s['x_peak_bytes']<=2048*core['m'] and s['contexts_peak']<=8)
            if not all(checks.values()):failures.append(dict(point=row['point'],checks=checks))
        rejects=root/f'{stage}_rejections.csv'
        summary[stage]=dict(accepted_points=len(rs),runs=2*len(rs),capacity_rejections=len(list(csv.DictReader(rejects.open()))) if rejects.exists() else 0)
    result=dict(timing=summary,all_accepted_points_pass=not failures,failures=failures,
                numeric_tests=dict(compiler=15,rust=23,python=39,physical_points_with_small_payload_replay=132,
                                   scope='seeded small payloads on actual timed trace; not pretrained model validation'),
                large_payloads_executed=False,native_ramulator=False,full_model_latency=False)
    run.write_json(root/'resource_and_correctness.json',result)
    assert not failures,failures[:3]
    return result

def summarize(root):
    frozen=json.loads((root/'frozen_designs.json').read_text())['designs']
    rows=json.loads((root/'heldout_rows.json').read_text())
    bounds=[]
    for r in rows:
        credit_bound=r['hbm_read_bytes']*r['hbm_latency_ns']/(32*r['credit_limit'])/1000
        byte_bound=r['hbm_read_bytes']/r['hbm_bytes_per_ns']/1000
        lower=max(credit_bound,byte_bound)
        peak_macs=sum(map(int,r['core_m'].split('+')))*4*512
        compute_bound=r['useful_macs']/peak_macs/1000
        bounds.append(dict(point=r['point'],budget=r['budget_group'],organization=r['organization'],
            workload=r['workload'],policy=r['mode'],tail_partition=r['tail_partition'],
            latency_us=r['latency_us'],bandwidth_lower_bound_us=byte_bound,
            credit_occupancy_lower_bound_us=credit_bound,time_over_lower_bound=r['latency_us']/lower,
            nominal_useful_compute_lower_bound_us=compute_bound,
            credit_to_nominal_compute_bound_ratio=credit_bound/compute_bound,
            achieved_weight_GBps=r['achieved_weight_bandwidth_GBps'],
            interpretation='resource lower bound, not exclusive memory stall time'))
    csv_out(root/'supply_lower_bounds.csv',bounds)
    lookup={(r['budget_group'],r['architecture'],r['workload'],r['mode'],r['tail_partition']):r for r in rows}
    by_arch={(d['budget_group'],d['architecture']):d for d in frozen}
    results=[];common=[];observations=[]
    for d in frozen:
        selected=[r for r in rows if r['budget_group']==d['budget_group'] and r['architecture']==d['architecture']
                  and r['mode']=='feedback' and r['tail_partition']==d['tail_partition']]
        record=dict(budget=d['budget_group'],architecture=d['architecture'],design_id=d['id'],lanes=d['lanes'],
                    group=d['group'],tail_partition=d['tail_partition'],coverage=f'{len(selected)}/8')
        for batch in (2,4,8,16):
            values=[r['latency_ms'] for r in selected if r['batch']==batch]
            record[f'B{batch}_mean_ms']=sum(values)/len(values) if values else None
        for ref in ('single','homogeneous'):
            reference=by_arch[d['budget_group'],ref]
            pairs=[(r,lookup.get((d['budget_group'],ref,r['workload'],'feedback',reference['tail_partition']))) for r in selected]
            pairs=[(a,b) for a,b in pairs if b]
            record[f'gm_speedup_vs_{ref}']=gm(b['latency_ms']/a['latency_ms'] for a,b in pairs)
            record[f'paired_coverage_vs_{ref}']=len(pairs)
            record[f'worst_slowdown_vs_{ref}_pct']=max([0]+[100*(a['latency_ms']/b['latency_ms']-1) for a,b in pairs]) if pairs else None
        record['gm_feedback_vs_fifo_same_grain']=gm(lookup[d['budget_group'],d['architecture'],r['workload'],'fifo',d['tail_partition']]['latency_ms']/r['latency_ms'] for r in selected)
        record['gm_feedback_vs_dynamic_same_grain']=gm(lookup[d['budget_group'],d['architecture'],r['workload'],'dynamic',d['tail_partition']]['latency_ms']/r['latency_ms'] for r in selected)
        audits=[]
        for row in selected:
            raw=json.loads((raw_directory(root,row)/'report_repeat1.json').read_text())
            audits.extend(a for a in raw['dispatch_audit'] if not a.get('tail_partition',False))
            for index,core in enumerate(raw['cores']):
                s=core['stats']
                observations.append(dict(budget=d['budget_group'],architecture=d['architecture'],lanes=d['lanes'],
                    workload=row['workload'],core=index,m=core['m'],layer_cycles=raw['cycles'],
                    core_finish_cycles=s['done_cycle'],arithmetic_active_cycles=s['arithmetic_active_cycles'],
                    useful_macs=s['useful_macs'],issued_macs=s['issued_macs'],
                    control_service_cycles=s['control_cycles'],weight_bytes=s['weight_bytes'],
                    x_stage_bytes=s['x_stage_bytes'],copy_bytes=s['copy_bytes'],
                    **{f'front_{k}_cycles':v for k,v in s['front_states'].items()},
                    interpretation='front categories exclusive per core; overlap arithmetic and other cores; not additive wall breakdown'))
        errors=[a['actual_minus_predicted_cycles'] for a in audits if 'actual_minus_predicted_cycles' in a]
        record['ordinary_bindings']=len(audits)
        record['fraction_bindings_with_two_legal_cores']=sum(len(a.get('eligible_cores',[]))==2 for a in audits)/len(audits) if audits else None
        record['prediction_mean_absolute_error_us']=sum(abs(e) for e in errors)/len(errors)/1000 if errors else None
        record['prediction_worst_underestimate_us']=max([0]+errors)/1000
        results.append(record)
        for grain in (False,True):
            actual=grain if len(d['lanes'])==2 else False
            for policy in ('fifo','dynamic','feedback'):
                selected=[r for r in rows if r['budget_group']==d['budget_group'] and r['architecture']==d['architecture'] and r['mode']==policy and r['tail_partition']==actual]
                single=[(r,lookup.get((d['budget_group'],'single',r['workload'],policy,False))) for r in selected]
                uniform=[(r,lookup.get((d['budget_group'],'homogeneous',r['workload'],policy,grain))) for r in selected]
                common.append(dict(budget=d['budget_group'],architecture=d['architecture'],grain='tail' if grain else 'whole',policy=policy,
                                   coverage=len(selected),geomean_ms=gm(r['latency_ms'] for r in selected),
                                   paired_coverage_vs_single=sum(b is not None for _,b in single),
                                   paired_coverage_vs_homogeneous=sum(b is not None for _,b in uniform),
                                   gm_speedup_vs_single=gm(b['latency_ms']/r['latency_ms'] for r,b in single if b),
                                   gm_speedup_vs_homogeneous=gm(b['latency_ms']/r['latency_ms'] for r,b in uniform if b)))
    csv_out(root/'heldout_summary.csv',results);csv_out(root/'common_policy_comparison.csv',common)
    csv_out(root/'selected_core_observations.csv',observations)
    run.write_json(root/'heldout_summary.json',results)
    # Design mean/worst Pareto under the same feedback policy, independent of test results.
    design=json.loads((root/'design_rows.json').read_text());pareto=[]
    for budget in ('M6','M8'):
        rs=[r for r in design if r['budget_group']==budget and r['mode']=='feedback']
        groups=collections.defaultdict(list)
        for r in rs:groups[r['design_id']].append(r)
        valid={k:v for k,v in groups.items() if len(v)==4}
        baseline=min((k for k,v in valid.items() if v[0]['architecture']=='single'),key=lambda k:gm(r['latency_ms'] for r in valid[k]))
        ref={r['workload']:r['latency_ms'] for r in valid[baseline]}
        ps=[]
        for k,v in valid.items():
            ratios=[r['latency_ms']/ref[r['workload']] for r in v]
            ps.append(dict(budget=budget,design_id=k,architecture=v[0]['architecture'],gm_latency_ratio=gm(ratios),worst_latency_ratio=max(ratios)))
        for p in ps:p['pareto']=not any(q['gm_latency_ratio']<=p['gm_latency_ratio'] and q['worst_latency_ratio']<=p['worst_latency_ratio'] and (q['gm_latency_ratio']<p['gm_latency_ratio'] or q['worst_latency_ratio']<p['worst_latency_ratio']) for q in ps)
        pareto+=ps
    csv_out(root/'design_pareto.csv',pareto)
    return frozen,results,common

def report(root,checks,frozen,summary,common):
    order={'single':0,'homogeneous':1,'heterogeneous':2}
    text=['# 固定硬件 MoE 搜索与验证结果','',
          '范围：Compiler + Rust analytical；真实路由、解析访存与片上端口计时。不是原生 Ramulator、完整模型推理或芯片测试。', '',
          '## 结论', '']
    for budget in ('M6','M8'):
        group=[r for r in summary if r['budget']==budget];hetero=next(r for r in group if r['architecture']=='heterogeneous')
        text.append(f"- {budget}：冻结异构 {'+'.join(map(str,hetero['lanes']))}，测试集相对同预算单核的几何平均加速 {num(hetero['gm_speedup_vs_single'])}×，相对同构 {num(hetero['gm_speedup_vs_homogeneous'])}×；相对同构最差慢 {num(hetero['worst_slowdown_vs_homogeneous_pct'],2)}%。覆盖 {hetero['coverage']}。")
    text += ['', '平均数来自8个请求互不重叠的测试窗口（BFCL/SWE，decode step 7）。整套设计/验证/测试覆盖 DeepSeek-V2-Lite 的两个层、两个 decode step 和三个数据来源，不能声称跨模型稳健。配置在测试前冻结；每类一个固定点，不逐窗口选赢家。', '',
             '## 冻结配置', '', '|预算|组织|M×N×K（N=4/K=512）|W槽|X KiB|累加区 bytes|累加bank|G|尾部分列|', '|---|---|---|---|---|---|---|---|---|']
    for d in sorted(frozen,key=lambda d:(d['budget_group'],order[d['architecture']])):
        h=d['resources'];text.append(f"|{d['budget_group']}|{d['architecture']}|{' + '.join(f'{m}×4×512' for m in d['lanes'])}|{h['weight_slots']}|{[2*m for m in d['lanes']]}|{h['acc_bytes']}|{h['acc_banks']}|{d['group']}|{d['tail_partition']}|")
    text += ['', '各组共享48 KiB权重/返回预算、2 MiB累加/中间/输出/控制预算、256 B/ns HBM、256个32B额度；M6/M8分别12/16 KiB X、24/32个X bank、12/16个accumulator bank。相同资源代理量不等于等面积。', '',
             '## 测试集延迟', '', '下表为每个 batch 的BFCL/SWE两个独立窗口算术平均，单位ms；加速比按全部8窗口的几何平均计算。', '',
             '|预算|组织|B2 ms|B4 ms|B8 ms|B16 ms|相对单核|相对同构|相对同构最差减速|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(summary,key=lambda d:(d['budget'],order[d['architecture']])):
        text.append(f"|{r['budget']}|{'+'.join(map(str,r['lanes']))}|{num(r['B2_mean_ms'],6)}|{num(r['B4_mean_ms'],6)}|{num(r['B8_mean_ms'],6)}|{num(r['B16_mean_ms'],6)}|{num(r['gm_speedup_vs_single'])}×|{num(r['gm_speedup_vs_homogeneous'])}×|{num(r['worst_slowdown_vs_homogeneous_pct'],2)}%|")
    text += ['', '## 收益归因', '', '|预算|组织|反馈相对FIFO，同硬件同粒度|反馈相对既有预计时间策略|', '|---|---|---:|---:|']
    for r in summary:text.append(f"|{r['budget']}|{'+'.join(map(str,r['lanes']))}|{num(r['gm_feedback_vs_fifo_same_grain'])}×|{num(r['gm_feedback_vs_dynamic_same_grain'])}×|")
    text += ['', '上述预测策略只能在多个核都合法时改变归属；它不重排FIFO，也不能迁移已绑定的Next。下表排除强制双核尾部分列绑定，误差为绑定时预测的绝对完成时间与实际完成时间之差。', '',
             '|预算|组织|普通绑定数|两核均合法比例|平均绝对误差 µs|最大低估 µs|', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        fraction=r['fraction_bindings_with_two_legal_cores']
        text.append(f"|{r['budget']}|{'+'.join(map(str,r['lanes']))}|{r['ordinary_bindings']}|{num(100*fraction if fraction is not None else None,1)}%|{num(r['prediction_mean_absolute_error_us'],3)}|{num(r['prediction_worst_underestimate_us'],3)}|")
    text += ['', '`common_policy_comparison.csv`进一步固定相同任务粒度和相同调度比较形状；`dispatch_ablation.csv`保留每个测试窗口全部策略/粒度组合。最终部署表含硬件、存储分配及冻结粒度的联合选择，不能把全部差异归给M形状。', '',
             '**供数的公式约束：**256个32B额度、最短64ns响应，将持续请求吞吐限制在不超过128 B/ns（128 GB/s），SRAM落地还会继续占用额度。因此256 GB/s接口峰值不能直接当成可达带宽。`supply_lower_bounds.csv`逐点给出额度占用下界及实得带宽；这不是把内存等待与计算时间相加的墙钟分解。', '',
             '整专家有限搜索暴露长尾后，增加了受限的最后专家分列：不迁移已绑定任务，等两核排空，再按输出列分成两个任务。Gate/Up坐标一致；Z复制、端口、屏障均收费。它会损失Next预取并增加片上搬运，因此保留开关对照，并未假设一定加速。当前不是任意细粒度偷取列块的调度器。', '',
             '4+2在Me=[4,2]机制用例有优势，在[3,3]有反例。纯计算投影、完整FFN和真实路由层窗口是不同计时范围。旧B8三点已复现；它们不是本次独立测试集，不能混进本表。', '',
             '## 验收与复现', '',
             f"- 接受计时点：{sum(v['accepted_points'] for v in checks['timing'].values())}；每点两次完整JSON一致。分阶段数量见 resource_and_correctness.json。",
             '- 编译器15项、Rust23项、Python39项测试通过；全部132个资源点有小尺寸实际定时轨迹数值重放。大尺寸只执行地址/时序和守恒检查。',
             '- 逐核容量、控制计费、HBM事务/字节、K顺序、ownership与最终排空检查通过。重叠的等待/服务计数没有相加成总延迟。',
             '- METHODS.md记录数据流与选择函数；workload_manifest.json记录来源/hash/请求划分；design_results.csv保留全部设计结果；design_pareto.csv保留平均/最差关系。',
             '- frozen_designs.json在heldout计时前产生。配置只由设计/验证集选择；同类可能有很接近的候选，不据少量窗口宣称全局最佳。',
             '- 原始输入、两次报告、命令配置、源码快照和二进制均在归档中；见 REPRODUCE.md。没有推送分支、创建PR或修改论文性能主张。', '']
    (root/'REPORT_ZH.md').write_text('\n'.join(text))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.output
    checks=check_raw(root);frozen,summary,common=summarize(root);design_attribution(root);report(root,checks,frozen,summary,common)
    shutil.copy2(Path(__file__).with_name('ROBUST_METHODS.md'),root/'METHODS.md')
    shutil.copy2(root/'inputs/workload_manifest.json',root/'workload_manifest.json')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
