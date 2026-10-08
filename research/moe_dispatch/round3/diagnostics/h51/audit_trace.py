"""Read-only H51 attribution from saved task/phase/resource observations."""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path
from research.moe_dispatch.round3.common import ROOT, inputs, gm, write_csv, write_json, sha
from research.moe_dispatch.round3.config import parameters


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def metrics(r):
    n=len(r['core_finish_cycles']);p=parameters('pipelined');d=r['ledger']['private'];wall=r['cycles']
    classes=Counter();hbm=Counter();coreHBM=[0.0]*n;small_tail=small_tail_bytes=0.0
    for s in r['segments']:
        dt=s['end']-s['start'];active=s['active_run'];key='both' if all(active) else 'only_big' if active[0] else 'only_small' if active[1] else 'neither'
        classes[key]+=dt;hbm[key]+=dt*s['hbm_rate_Bpc']
        for c in range(n):coreHBM[c]+=dt*s['hbm_rate_Bpc_core'][c]
        if key=='only_small':
            clipped=max(0.0,s['end']-max(s['start'],r['core_finish_cycles'][0]))
            small_tail+=clipped;small_tail_bytes+=clipped*s['hbm_rate_Bpc']
    phases=r['phases'];tasks=r['tasks']
    out={'window_id':r['workload'],'batch':r['batch'],'latency_ms':r['latency_ms'],
         'HBM_MiB':r['hbm_bytes']/2**20,'effective_global_GBps':r['hbm_bytes']/wall,
         'finish_gap_ms':r['core_finish_gap']/1e6,'HBM_busy_fraction':r['hbm_busy_frac']}
    for c in range(n):
        ph=[s for s in phases if s['core']==c]
        ts=[t for t in tasks if t['core']==c]
        out.update({f'tasks_c{c}':len(ts),f'phase_HBM_MiB_c{c}':sum(s['hbm_bytes'] for s in ph)/2**20,
          f'observed_HBM_MiB_c{c}':coreHBM[c]/2**20,
          f'W_service_floor_ms_c{c}':sum(s['w_sram_bytes'] for s in ph)/(d['w_banks'][c]*p.bank_Bpc)/1e6,
          f'X_service_floor_ms_c{c}':sum(s['x_sram_bytes'] for s in ph)/(d['x_banks'][c]*p.bank_Bpc)/1e6,
          f'acc_service_floor_ms_c{c}':sum(s['acc_sram_bytes'] for s in ph)/(d['acc_banks'][c]*p.bank_Bpc)/1e6,
          f'compute_service_ms_c{c}':sum(s['compute_cycles'] for s in ph)/1e6,
          f'phase_duration_ms_c{c}':sum(s['stream_finish']-s['start'] for s in ph)/1e6,
          f'actual_finish_ms_c{c}':r['core_finish_cycles'][c]/1e6,
          f'Shared_core{c}':sum(t['expert_id']==-1 for t in ts)})
    for k in ('both','only_big','only_small','neither'):
        out[k+'_actor_ms']=classes[k]/1e6
        out[k+'_fetch_GBps']=hbm[k]/classes[k] if classes[k] else 0
        out[k+'_HBM_MiB']=hbm[k]/2**20
    out['small_tail_after_big_finish_ms']=small_tail/1e6
    out['small_tail_after_big_finish_HBM_MiB']=small_tail_bytes/2**20
    out['unobserved_end_ms']=(wall-sum(classes.values()))/1e6
    out['small_W_phase_fraction']=sum(t for k,t in r['stream_attribution'][1].items() if k=="('W', 1)")/max(1,sum(r['stream_attribution'][1].values()))
    return out


def run():
    raw=ROOT/'E5/dispatch/raw';out=ROOT/'diagnostics/h51';paths={};records={};rows=[]
    for method in ('eft_old','fixed','milp'):
        paths[method]=next(raw.glob('C0_pipelined_H51_'+method+'*520.json.gz'))
        records[method]=read(paths[method]);assert len(records[method])==135
        for r in records[method]:rows.append({'dispatch':method,**metrics(r)})
    summary=[]
    for batch in (2,4,8,16,64,96,128,'all'):
        xs={m:[r for r in rows if r['dispatch']==m and (batch=='all' or r['batch']==batch)] for m in records}
        for method,rs in xs.items():
            def avg(f):return sum(r[f] for r in rs)/len(rs)
            s=dict(batch=batch,dispatch=method,n_windows=len(rs),GM_ms=gm(r['latency_ms'] for r in rs),
                   paired_fixed_milp_ratio=gm(a['latency_ms']/b['latency_ms'] for a,b in zip(xs['fixed'],xs['milp'])),
                   mean_HBM_MiB=avg('HBM_MiB'),mean_finish_gap_ms=avg('finish_gap_ms'),
                   volume_weighted_GBps=sum(r['HBM_MiB'] for r in rs)*2**20/(sum(r['latency_ms'] for r in rs)*1e6),
                   small_HBM_share=sum(r['observed_HBM_MiB_c1'] for r in rs)/sum(r['HBM_MiB'] for r in rs))
            for field in ('small_tail_after_big_finish_ms','phase_duration_ms_c0','phase_duration_ms_c1','phase_HBM_MiB_c0','phase_HBM_MiB_c1','W_service_floor_ms_c0','W_service_floor_ms_c1',
                          'X_service_floor_ms_c0','X_service_floor_ms_c1','compute_service_ms_c0','compute_service_ms_c1',
                          'only_small_actor_ms','only_big_actor_ms','both_actor_ms','neither_actor_ms','small_W_phase_fraction'):
                s['mean_'+field]=avg(field)
            for k in ('only_small','only_big','both','neither'):
                duration=sum(r[k+'_actor_ms'] for r in rs)*1e6
                s[k+'_interval_fetch_GBps']=sum(r[k+'_HBM_MiB'] for r in rs)*2**20/duration if duration else 0
            tail=sum(r['small_tail_after_big_finish_ms'] for r in rs)*1e6
            s['small_tail_after_big_finish_fetch_GBps']=sum(r['small_tail_after_big_finish_HBM_MiB'] for r in rs)*2**20/tail if tail else 0
            summary.append(s)
    write_csv(out/'per_window.csv',rows);write_csv(out/'by_batch.csv',summary)
    ranked=sorted(zip(records['fixed'],records['milp']),key=lambda a:a[0]['latency_ms']/a[1]['latency_ms'],reverse=True)
    taskdetails=[]
    for a,b in ranked[:6]:
        for method,r in (('fixed',a),('milp',b)):
            for t in r['tasks']:
                ph=[p for p in r['phases'] if p['expert_index']==t['expert_index']]
                taskdetails.append({'window_id':r['workload'],'dispatch':method,'layer_ms':r['latency_ms'],
                    **{k:t[k] for k in ('expert_index','expert_id','Me','core','start','finish','predicted_cycles','actual_cycles','shape')},
                    'W_MiB':sum(p['w_sram_bytes'] for p in ph)/2**20,'HBM_MiB':sum(p['hbm_bytes'] for p in ph)/2**20})
    write_csv(out/'worst_task_allocations.csv',taskdetails)
    publish_diagnosis(rows,summary,records,paths)
    write_json(out/'RECEIPT.json',{'source_files':{m:{'file':str(p.relative_to(ROOT)),'sha256':sha(p)} for m,p in paths.items()},
        'scope':'read-only saved phase-fluid observations; resource work floors and actor interval partitions are separate, overlapping counters never summed as latency',
        'records':len(rows),'all_phase_HBM_matches_total':all(abs(sum(p['hbm_bytes'] for p in r['phases'])-r['hbm_bytes'])<1e-4 for rs in records.values() for r in rs)})
    print(json.dumps([r for r in summary if r['batch']=='all'],indent=2),flush=True)



def publish_diagnosis(rows,summary,records,paths):
    target=ROOT/'E5/dispatch';ws={w['id']:w for w in inputs()['heldout']}
    mixes=[]
    for method,rs in records.items():
        buckets={}
        for r in rs:
            for t in r['tasks']:
                e=ws[r['workload']]['experts'][t['expert_index']]
                role='Shared' if e.get('is_shared') else 'routed Me<=2' if e['Me']<=2 else 'routed Me>2'
                key=(role,t['core'])
                if key not in buckets:buckets[key]={'dispatch':method,'role':role,'core':t['core'],'task_count':0,'X_service_sum_ms':0.0,'HBM_sum_MiB':0.0}
                v=buckets[key];ph=[p for p in r['phases'] if p['expert_index']==t['expert_index']]
                v['task_count']+=1
                v['X_service_sum_ms']+=sum(p['x_sram_bytes'] for p in ph)/(64 if t['core']==1 else 320)/1e6
                v['HBM_sum_MiB']+=sum(p['hbm_bytes'] for p in ph)/2**20
        mixes.extend(buckets.values())
    cases=[];assignments=[];bindings=[];case_slices=[]
    for r in records['fixed']:
        for b in r['bindings']:
            qs=b.get('candidate_comparisons',[])
            if not qs:continue
            chosen=next(q for q in qs if q['core']==b['core'])
            other=min(qs,key=lambda q:(q['predicted_finish']+q['contention_cycles'],q['core']))
            if (chosen['refetch']==1 and other['refetch']==1 and not other['admissible'] and
                    other['predicted_finish']+other['contention_cycles']<chosen['predicted_finish']+chosen['contention_cycles']):
                e=ws[r['workload']]['experts'][b['expert_index']]
                bindings.append(dict(window_id=r['workload'],batch=r['batch'],expert_id=b['expert_id'],Me=e['Me'],is_shared=bool(e.get('is_shared')),
                    bind_ms=b['bind_cycle']/1e6,chosen_core=b['core'],chosen_predicted_finish_ms=chosen['predicted_finish']/1e6,
                    earlier_core=other['core'],earlier_predicted_finish_ms=other['predicted_finish']/1e6,
                    earlier_core_temporarily_admissible=other['admissible'],refetch_chosen=chosen['refetch'],refetch_earlier=other['refetch']))
    write_csv(target/'DISPATCH_DIAGNOSIS_BINDING_WAIT.csv',bindings)
    selected_ids=('v3_captured_mixed_heldout_bfcl_t128_l2','v3_captured_mixed_heldout_gpqa_t128_l13')
    key_experts={selected_ids[0]:42,selected_ids[1]:38}
    for wid in selected_ids:
        for method,rs in records.items():
            r=next(r for r in rs if r['workload']==wid);m=metrics(r)
            # Keep an exact raw record slice. These objects come directly from
            # the frozen replay output; no fields or numeric values are edited.
            raw_bind=next(b for b in r['bindings'] if b['expert_id']==key_experts[wid])
            task=next(t for t in r['tasks'] if t['expert_id']==key_experts[wid])
            at=raw_bind['bind_cycle']
            committed_before=[b for b in r['bindings'] if b['bind_cycle']<=at]
            task_by_i={t['expert_index']:t for t in r['tasks']}
            current_next=[]
            for ci in range(len(r['core_finish_cycles'])):
                outstanding=[b for b in committed_before if b['core']==ci and task_by_i[b['expert_index']]['finish']>at]
                outstanding.sort(key=lambda b:(b['bind_cycle'],b['expert_index']))
                current_next.append({'core':ci,'outstanding_expert_indices':[b['expert_index'] for b in outstanding],
                    'current':task_by_i[outstanding[0]['expert_index']] if outstanding else None,
                    'next':task_by_i[outstanding[1]['expert_index']] if len(outstanding)>1 else None})
            bound_indices={b['expert_index'] for b in committed_before}
            priority=r.get('dispatch_priority_order',list(range(len(ws[wid]['experts']))))
            pending=[i for i in priority if i not in bound_indices]
            case_slices.append({'window_id':wid,'dispatch':method,
                'source_file':str(paths[method].relative_to(ROOT)),'source_sha256':sha(paths[method]),
                'key_expert_id':key_experts[wid],'raw_binding':raw_bind,'raw_actual_task':task,
                'reconstructed_at_committed_binding_cycle':{'time_scope':'bind_cycle is after serialized controller decision, including the key expert; this is not an original FSM state log',
                    'core_current_next_from_saved_task_intervals':current_next,
                    'priority_fifo_remaining_expert_indices':pending,'finite_window_expert_indices':pending[:8]} if method!='milp' else None,
                'unrecorded_fields':{'admissibility_rejection_reason':None,'pre_decision_at_cycle':None,'raw_fifo_state':None,'raw_CurrentNext_state':None},
                'unrecorded_field_warning':'The frozen source records admissible and candidate ETA/refetch, but not the exact capacity/queue/lead-time rejection bits or pre-decision FIFO/CurrentNext snapshot. Derived post-commit states must not be called original runtime observations.',
                'raw_record':r})
            owner_pairs=[f"{t['expert_id']}:{t['Me']}→c{t['core']}" for t in r['tasks']]
            cases.append({'window_id':wid,'dispatch':method,'latency_ms':r['latency_ms'],'HBM_MiB':m['HBM_MiB'],
                'small_X_service_ms':m['X_service_floor_ms_c1'],'small_compute_service_ms':m['compute_service_ms_c1'],
                'only_small_actor_ms':m['only_small_actor_ms'],'only_small_fetch_GBps':m['only_small_fetch_GBps'],
                'small_tail_after_big_finish_ms':m['small_tail_after_big_finish_ms'],
                'task_owners':'; '.join(owner_pairs)})
            for t in r['tasks']:
                ph=[p for p in r['phases'] if p['expert_index']==t['expert_index']]
                assignments.append({'window_id':wid,'dispatch':method,**t,
                    'HBM_MiB':sum(p['hbm_bytes'] for p in ph)/2**20,
                    'X_service_ms':sum(p['x_sram_bytes'] for p in ph)/(64 if t['core']==1 else 320)/1e6})
    write_csv(target/'DISPATCH_DIAGNOSIS.csv',summary)
    write_csv(target/'DISPATCH_DIAGNOSIS_TASK_MIX.csv',mixes)
    write_csv(target/'DISPATCH_DIAGNOSIS_CASES.csv',cases)
    write_csv(target/'DISPATCH_DIAGNOSIS_CASE_TASKS.csv',assignments)
    slice_file=target/'DISPATCH_DIAGNOSIS_CASE_SLICES.json'
    write_json(slice_file,{'scope':'exact six saved per-window record slices plus explicitly marked derived post-commit state; no simulation rerun',
                         'records':case_slices})
    allrows={r['dispatch']:r for r in summary if r['batch']=='all'}
    a,b=allrows['fixed'],allrows['milp']
    md=['# H51 在线与离线派工的只读归因诊断','',
      '同一C0 pipelined冻结H51、256 GB/s上限、135个留出窗口、BF16；仅读取保存的任务、phase和HBM服务区间，不重跑或改变仿真。这里的milp是资源分配松弛加LPT合法回放，不是时序全局最优。',
      '', '## 相同HBM字节并不意味着相同完成时间','',
      '| 指标 | fixed在线 | milp离线参照 | 口径 |','|---|---:|---:|---|',
      f"| 全部窗口延迟 GM ms | {a['GM_ms']:.6f} | {b['GM_ms']:.6f} | 135窗口几何平均；在线/离线={a['paired_fixed_milp_ratio']:.6f} |",
      f"| 每窗口HBM MiB | {a['mean_HBM_MiB']:.6f} | {b['mean_HBM_MiB']:.6f} | 算术平均，且逐窗口字节相同 |",
      f"| 小核取得的HBM份额 | {100*a['small_HBM_share']:.3f}% | {100*b['small_HBM_share']:.3f}% | 体积加权，来自每核服务率积分 |",
      f"| 小核X端口服务量 ms | {a['mean_X_service_floor_ms_c1']:.6f} | {b['mean_X_service_floor_ms_c1']:.6f} | 每窗口算术平均的X字节÷64B/ns |",
      f"| 小核W端口服务量 ms | {a['mean_W_service_floor_ms_c1']:.6f} | {b['mean_W_service_floor_ms_c1']:.6f} | 每窗口算术平均的W字节÷176B/ns |",
      f"| 两核完成时间差 ms | {a['mean_finish_gap_ms']:.6f} | {b['mean_finish_gap_ms']:.6f} | 每窗口算术平均 |",
      f"| 仅小核有执行phase actor ms | {a['mean_only_small_actor_ms']:.6f} | {b['mean_only_small_actor_ms']:.6f} | 互斥运行区间，包含中途和尾部 |",
      f"| 上述区间全芯片HBM GB/s | {a['only_small_interval_fetch_GBps']:.3f} | {b['only_small_interval_fetch_GBps']:.3f} | 对区间按时间加权 |",
      f"| 大核完成后的小核执行尾部 ms | {a['mean_small_tail_after_big_finish_ms']:.6f} | {b['mean_small_tail_after_big_finish_ms']:.6f} | 精确裁剪start≥大核最终完成时刻 |",
      f"| 仅大核有执行phase actor ms | {a['mean_only_big_actor_ms']:.6f} | {b['mean_only_big_actor_ms']:.6f} | 互斥运行区间 |",
      f"| 上述区间全芯片HBM GB/s | {a['only_big_interval_fetch_GBps']:.3f} | {b['only_big_interval_fetch_GBps']:.3f} | 对区间按时间加权 |",
      '', f"小核X服务需求是离线的{a['mean_X_service_floor_ms_c1']/b['mean_X_service_floor_ms_c1']:.3f}倍。它是资源上的必需服务时间量，**不能与W服务、HBM忙碌、计算服务或尾部区间相加来拆分总延迟**。只有执行phase actor活跃也不等于所有计算乘法器一直忙。",
      '', '## 具体的任务归属和当前数据流','',
      'H51大核为1×40×256、W/X bank=53/20；小核为2×2×512、W/X bank=11/4，两核均WS。小核X容量11KiB，X端口64B/ns；大核X容量53KiB，X端口320B/ns。',
      '', '当前WS成本模型中，一个投影的N tile数为ceil(N/PN)。若整个X平面Me×K×2B能放入私有X，则外层读取一次；否则xreads=Me×K×2B×mult×Ntiles。另有每个发射的xoperand字节。投影的X端口需求为这两部分及相应激活写入之和。小核PN=2且X仅11KiB；例如K=2048时Me=3的完整X为12KiB，已经超过容量。大核PN=40、X=53KiB可支持更大的驻留平面。这里是模型中的明确规则，不是原生SRAM bank实测。',
      '', '| 派工 | 小核任务类型 | 数量 | X服务总量 ms | HBM总量 MiB |','|---|---|---:|---:|---:|']
    for r in mixes:
        if r['core']==1 and r['dispatch'] in ('fixed','milp'):
            md.append(f"| {r['dispatch']} | {r['role']} | {r['task_count']} | {r['X_service_sum_ms']:.3f} | {r['HBM_sum_MiB']:.3f} |")
    md += ['', '在线Shared全部在大核，当前慢点不是Shared仍被放到小核。在线给小核更多Me>2 routed任务，这些任务在窄N/小X资源上产生更多X搬运；小核尾部所需HBM很少，因而全芯片HBM服务率降下来。离线在B64/96/128不派任何任务给小核：完成时间差虽更大，大核仍能持续取得接近全局HBM上限的带宽，整层反而先结束。',
      '', '## 两个新fixed比eft_old慢的窗口','',
      'newfixed总体仍比eft_old更快；下面展示局部退化，不能只用总体GM掩盖。三个策略都是同一冻结H51；eft_old和fixed均用相同ours预热顺序。',
      '', '| 窗口 | 派工 | 延迟 ms | HBM MiB | 小核X服务量 ms | 仅小核执行区间 ms | 该区间HBM GB/s |','|---|---|---:|---:|---:|---:|---:|']
    for r in cases:md.append(f"| {r['window_id']} | {r['dispatch']} | {r['latency_ms']:.6f} | {r['HBM_MiB']:.3f} | {r['small_X_service_ms']:.6f} | {r['only_small_actor_ms']:.6f} | {r['only_small_fetch_GBps']:.3f} |")
    md += ['', '## 等待规则的具体缺口：并非预测器把小核算快了','',
      f'在135个fixed窗口保存的绑定记录中，有{len(bindings)}次：两个候选的refetch都为1，另一个核预计更早完成，但暂时admissible=false，于是任务仍绑定到当前合法而明显更慢的核。DISPATCH_DIAGNOSIS_BINDING_WAIT.csv完整列出这些记录。这里是保存的候选比较，不是假设用未来信息重新调度。',
      '', '| 窗口及专家 | Me | 绑定时刻 ms | 大核预计完成 ms / admissible | 小核预计完成 ms / admissible | 实际小核开始 / 完成 ms | 实际任务时长 / 预测时长 ms |',
      '|---|---:|---:|---|---|---|---|']
    for v in case_slices:
        if v['dispatch']!='fixed':continue
        b=v['raw_binding'];t=v['raw_actual_task'];qs={q['core']:q for q in b['candidate_comparisons']}
        md.append(f"| {v['window_id']} / E{v['key_expert_id']} | {t['Me']} | {b['bind_cycle']/1e6:.6f} | {qs[0]['predicted_finish']/1e6:.6f} / {qs[0]['admissible']} | {qs[1]['predicted_finish']/1e6:.6f} / {qs[1]['admissible']} | {t['start']/1e6:.6f} / {t['finish']/1e6:.6f} | {t['actual_cycles']/1e6:.6f} / {t['predicted_cycles']/1e6:.6f} |")
    md += ['', '两条案例中小核任务时长预测只比实际多64周期。当前choose_candidate只在“当前核refetch>1、另一核refetch=1”时比较等待；若两边refetch=1，admissible=false的更快候选直接被移除，未继续比较等待。这是当前允许等待的准入规则缺口，不应写成预测器失准。',
      '', '原始记录没有保存admissible=false的具体拒绝原因位，所以不能断言该次一定是队列满、容量不足或晚绑定阈值中的哪一条。DISPATCH_DIAGNOSIS_CASE_SLICES.json保留六份完整原始窗口slice、原始候选ETA/refetch/admissible和实际task，另给出明确标注“提交绑定时刻派生重建”的FIFO与Current/Next状态。拒绝原因与决策前状态未记录，字段为null，未伪造观测。原始slice和源文件SHA见receipt。']
    md += ['', '专家ID、Me、目标核、开始/完成/预测时间及X服务量逐项在DISPATCH_DIAGNOSIS_CASE_TASKS.csv。上述数据支持“X供数服务与任务归属产生小核长尾”的模型内诊断，但未做单变量反事实试验，不能把2.255倍X服务量直接宣称为12.54%总延迟的全部因果贡献，也不能据此证明未来更好的派工一定无法改善。',
      '', '原始结果和硬件不变。DISPATCH_DIAGNOSIS.csv按batch给出完整指标；逐窗口派生数据在diagnostics/h51/per_window.csv；SHA见DISPATCH_DIAGNOSIS_RECEIPT.json。']
    (target/'DISPATCH_DIAGNOSIS.md').write_text('\n'.join(md)+'\n')
    write_json(target/'DISPATCH_DIAGNOSIS_RECEIPT.json',dict(source_files={m:dict(file=str(p.relative_to(ROOT)),sha256=sha(p)) for m,p in paths.items()},
      runtime_source_sha256=sha(ROOT/'runtime.py'),model_source_sha256=sha(ROOT/'model.py'),script_sha256=sha(Path(__file__)),parameters_sha256=sha(ROOT/'config.py'),selection_sha256=sha(ROOT/'E4/selected_designs.json'),
      case_slices_sha256=sha(slice_file),case_slices_file=str(slice_file.relative_to(ROOT)),
      simulation_rerun=False,physical_sources_changed=False,scope='derived saved task/phase/fluid intervals; service amounts overlap and are not summed as wall time'))

if __name__=='__main__':run()
