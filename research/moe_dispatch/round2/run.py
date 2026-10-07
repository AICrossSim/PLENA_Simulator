"""Ordered, resumable second-round campaign. Full scopes and open proofs are explicit."""
from __future__ import annotations
import argparse, json,math,time
from pathlib import Path
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
from .common import *
from .predictors import Predictor
from .model import shape_oracle
from .optimizer import solve_assignment,evaluate_design

COMMAND='/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.run'


def _worker(job):
    w,d,p,kind,policy=job
    if kind.startswith('U1_'):
        owners=[0]*len(w['experts']) if kind.endswith('milp') else None
        a=shape_oracle(w,d,p,owners=owners);b=shape_oracle(w,d,p,owners=owners)
        assert canonical(a)==canonical(b),'U1 repeat differs'
        return a
    if kind.startswith('U2_'):
        p=replace(p,sram_weight_read_once_oracle=True)
        kind=kind.split('_',1)[1]
    if kind=='milp':
        a=evaluate_design(w,d,p)
        b=evaluate_design(w,d,p)
        assert canonical(a)==canonical(b),'two allocation/schedule runs differ'
        return a
    return evaluate_repeat(w,d,p,policy=policy)

def run_many(ws,d,p,kind='runtime',policy='eft',jobs=1):
    args=[(w,d,p,kind,policy) for w in ws]
    if jobs==1:return [_worker(a) for a in args]
    with ProcessPoolExecutor(max_workers=jobs) as pool:return list(pool.map(_worker,args,chunksize=2))

def designs_from_selection(selection,mode):
    m=selection['modes'][mode]
    def get(k,default):return decode_design(m[k]['design']) if k in m else default
    b1=get('single',Design((Core(6,16,128),),flows=('WS',)))
    b2=get('homogeneous',Design((Core(6,16,64),)*2,flows=('WS',)*2))
    old=Design((Core(1,2,64),Core(5,19,128)),flows=('WS',)*2,label='previous_asym')
    h=get('heterogeneous',old)
    ds={'B0':Design((Core(6,4,512),),label='B0'), 'B1':b1,'B2':b2,
        'fixed_3+3':Design((Core(3,4,512),)*2),
        'fixed_4+2':Design((Core(4,4,512),Core(2,4,512))),
        'previous_asym':old,'best_hetero':h}
    for k in ('5+1','4+2','2+4'):
        if k in m:ds['best_'+k]=decode_design(m[k]['design'])
    return ds

def selection_file():
    return json.loads((ROOT/'results/E3/FROZEN_SELECTION.json').read_text())

def e1(args):
    ws=inputs()['heldout'];sel=selection_file();rows=[];head=[]
    for mode in MODES:
        p=Parameters(onchip_mode=mode);ds=designs_from_selection(sel,mode)
        for name in ('B0','B1','B2','fixed_3+3','fixed_4+2','previous_asym'):
            d=ds[name];rs=run_many(ws,d,p,jobs=args.jobs)
            for w,r in zip(ws,rs):rows.append({'window_id':w['id'],'batch':w['batch'],'design':name,
                'geometry':d.geometry,'onchip_mode':mode,'latency_ms':r['latency_ms'],**bounds(w,d,p,r)})
            print('E1',mode,name,flush=True)
        for batch in BATCHES:
            group=[r for r in rows if r['onchip_mode']==mode and r['design']=='B1' and r['batch']==batch]
            vs=[r['headroom_pct'] for r in group]
            head.append({'batch':batch,'onchip_mode':mode,'windows':len(vs),
                'median_headroom_pct':float(np.median(vs)),
                'geomean_headroom_pct':100*(gmean(1+v/100 for v in vs)-1),
                'geomean_latency_ms':gmean(r['latency_ms'] for r in group),
                'median_ge_5pct':bool(np.median(vs)>=5)})
    out=ROOT/'results/E1';write_csv(out/'bounds_per_window.csv',rows);write_csv(out/'headroom_by_batch.csv',head)
    cells=[f"{r['onchip_mode']}/B{r['batch']} ({r['median_headroom_pct']:.2f}%)" for r in head if r['median_ge_5pct']]
    finalize(out,COMMAND+' --stage E1 --jobs '+str(args.jobs),
        '# E1 下限与余量\n\n逐窗口实算六种设计×三种模式，共 '+str(len(rows))+' 行；每点重复一致。\n\n'+
        'B1 中位余量≥5%的格子：'+('；'.join(cells) or '无')+'。这只是可优化余量，不能证明异构有收益。\n\n'+
        '单任务下限只取最快可行核，不因另一核较慢强制绑定。端口量为该数据流最少合法流量，各计数不可相加。')

def e2micro(args):
    rows=[];shapes=[Core(6,16,128),Core(6,4,512),Core(1,2,64),Core(5,19,128),Core(4,4,512),Core(2,4,512)]
    for shape in shapes:
      for flow in ('OS','WS','IS'):
       for typ,f in (('routed',1408),('Shared',2816)):
        for m in (1,2,3,4,6,8,12,16,32,64,128):
         for mode in MODES:
            p=Parameters(onchip_mode=mode);a=micro_cost(shape,m,f=f,flow=flow,params=p)
            b=micro_cost(shape,m,f=f,flow=flow,params=p);assert a==b
            rows.append({'shape':f'{shape.pm}x{shape.pn}x{shape.pk}','dataflow':flow,'expert_type':typ,
                'Me':m,'onchip_mode':mode,'cycles':a.isolated_cycles,'issues':a.issues,
                'w_sram_bytes':a.w_sram_bytes,'x_sram_bytes':a.x_sram_bytes,'acc_sram_bytes':a.acc_sram_bytes,
                'hbm_bytes':a.hbm_bytes,'refetch_factor':a.hbm_bytes/a.unique_hbm_bytes,
                'spatial_util':a.spatial_util})
    out=ROOT/'results/E2';write_csv(out/'micro.csv',rows)
    print('E2micro rows',len(rows),flush=True)

def e2layer(args):
    ws=inputs()['heldout'];sel=selection_file();rows=[]
    for mode in MODES:
      p=Parameters(onchip_mode=mode);ds=designs_from_selection(sel,mode)
      for name in ('B1','fixed_4+2','previous_asym','best_hetero'):
        d=ds[name];combos=[(a,b) for a in ('OS','WS','IS') for b in ('OS','WS','IS')] if len(d.cores)==2 else [(a,'') for a in ('OS','WS','IS')]
        data={}
        for a,b in combos:
            if b:
                big=max(range(2),key=lambda c:d.cores[c].macs)
                fs=[None,None];fs[big]=a;fs[1-big]=b
                dd=replace(d,flows=tuple(fs))
            else:dd=replace(d,flows=(a,))
            data[a,b]=run_many(ws,dd,p,jobs=args.jobs)
        base=data['OS','OS' if len(d.cores)==2 else '']
        for (a,b),rs in data.items():
          for batch in (*BATCHES,'all'):
            idx=[i for i,w in enumerate(ws) if batch=='all' or w['batch']==batch]
            rows.append({'design':name,'geometry':d.geometry,'df_big':a,'df_small':b,'onchip_mode':mode,'batch':batch,
                'geomean_ms':gmean(rs[i]['latency_ms'] for i in idx),'ratio_vs_OS_OS':gmean(rs[i]['cycles']/base[i]['cycles'] for i in idx),
                'w_sram_GiB':sum(rs[i]['w_sram_bytes'] for i in idx)/2**30,
                'acc_sram_GiB':sum(rs[i]['acc_sram_bytes'] for i in idx)/2**30,'hbm_GiB':sum(rs[i]['hbm_bytes'] for i in idx)/2**30})
        print('E2grid',mode,name,flush=True)
    out=ROOT/'results/E2';write_csv(out/'layer_grid.csv',rows)
    summary='# E2 OS / WS / IS\n\nOS：输出部分和跨K驻留；每M波次从SRAM重读W。WS：W驻留依次服务M，K段部分和在累加SRAM读改写。IS：X驻留遍历N，部分和读改写；组会RS在此等同IS。\n\n'
    for mode in MODES:
        subset=[r for r in rows if r['onchip_mode']==mode and r['batch']=='all']
        summary+='\n'+mode+'（fixed_issue为非等资源参考）：\n'
        for name in ('B1','fixed_4+2','previous_asym','best_hetero'):
            ss=[r for r in subset if r['design']==name];best=min(ss,key=lambda r:r['geomean_ms']);os=next(r for r in ss if r['df_big']=='OS' and r['df_small'] in ('OS',''))
            summary+=f"- {name}: 最快{best['df_big']}/{best['df_small'] or '-'}；OS/OS比最快慢{100*(os['geomean_ms']/best['geomean_ms']-1):.2f}%。\n"
    micro=read_csv(out/'micro.csv')
    # Explicitly report single-wave equality by both traffic and timing, not assumed name equivalence.
    pairs={}
    for r in micro:pairs.setdefault((r['shape'],r['expert_type'],r['Me'],r['onchip_mode']),{})[r['dataflow']]=r
    comparisons=[]
    for key,v in pairs.items():
        shape,typ,me,mode=key
        comparisons.append({'shape':shape,'expert_type':typ,'Me':int(me),'onchip_mode':mode,
            'OS_cycles':float(v['OS']['cycles']),'WS_cycles':float(v['WS']['cycles']),
            'WS_over_OS':float(v['WS']['cycles'])/float(v['OS']['cycles']),
            'multi_M_wave':int(me)>int(shape.split('x')[0])})
    write_csv(out/'micro_WS_vs_OS.csv',comparisons)
    for mode in MODES:
        relevant=[v for k,v in pairs.items() if k[3]==mode and int(k[2])<=int(k[0].split('x')[0])]
        equal=sum(float(v['OS']['cycles'])==float(v['WS']['cycles']) and all(v['OS'][x]==v['WS'][x] for x in ('w_sram_bytes','x_sram_bytes','acc_sram_bytes')) for v in relevant)
        summary+=f'\nMe≤PM在{mode}下OS与WS的时间和全部流量均相同：{equal}/{len(relevant)}；不得把一次M发射等同于全部数据流相同。\n'
        for typ in ('routed','Shared'):
            ss=[r for r in comparisons if r['onchip_mode']==mode and r['expert_type']==typ and r['multi_M_wave']]
            ratios=[r['WS_over_OS'] for r in ss]
            summary+=f"\n{mode}/{typ}多M波次微实验：WS严格快于OS {sum(x<1-1e-12 for x in ratios)}/{len(ratios)} 点；配对几何平均WS/OS={gmean(ratios):.6f}。逐Me与形状差异见micro_WS_vs_OS.csv，不能把此均值等同整层加速。\n"
        for name in ('fixed_4+2','previous_asym','best_hetero'):
            ss=[r for r in rows if r['onchip_mode']==mode and r['batch']=='all' and r['design']==name]
            best=min(ss,key=lambda r:r['geomean_ms']);os=next(r for r in ss if r['df_big']=='OS' and r['df_small']=='OS')
            ratio=os['geomean_ms']/best['geomean_ms']
            summary+=f"\n{mode}/{name}: OS/OS与最快流的延迟比={ratio:.6f}，{'在' if ratio<=1.01 else '不在'}最快值1%以内。\n"
    finalize(out,COMMAND+' --stage E2 --jobs '+str(args.jobs),summary,
        '微实验每核使用完整声明私有预算，仅用于流量/周期比较，不是等乘法器整层结论。整层只换循环顺序，其余硬件/资源/策略冻结。')


def pack_eval(x,kind):
    return x['milp_sched'] if kind=='milp' else x


def e4(args):
    ws=inputs()['heldout'];sel=selection_file();table=[];raw=[];bd=[];sensitivity=[]
    for mode in MODES:
      p=Parameters(onchip_mode=mode);ds=designs_from_selection(sel,mode);ds['U1']=ds['B1'];ds['U2']=ds['B1'];all_results={}
      for name,d in ds.items():
        if name=='previous_asym':continue
        for kind in ('milp','runtime'):
          run_kind=name+'_'+kind if name in ('U1','U2') else kind
          evs=run_many(ws,d,p,kind=run_kind,jobs=args.jobs)
          rs=[e if name=='U1' else pack_eval(e,kind) for e in evs]
          all_results[name,kind]=rs
          for w,r in zip(ws,rs):
            raw.append({'entry':name,'onchip_mode':mode,'sched_type':kind,'window_id':w['id'],'batch':w['batch'],
                'design':canonical(encode_design(d)),'latency_ms':r['latency_ms'],'cycles':r['cycles'],
                'hbm_bytes':r['hbm_bytes'],'native_unique_bytes':r['native_unique_bytes'],'spatial_util':r['spatial_utilization'],
                'core0_compute_busy':r['core_compute_busy'][0],
                'core1_compute_busy':r['core_compute_busy'][1] if len(d.cores)>1 else 0,
                'w_port_busy':r['w_port_busy'],'x_port_busy':r['x_port_busy'],'acc_port_busy':r['acc_port_busy'],
                'hbm_busy_frac':r['hbm_busy_frac'],'core_finish_gap':r['core_finish_gap'],'idle_frac':r.get('idle_frac',None),
                'binding_term':'diagnostic_oracle' if name in ('U1','U2') else bounds(w,d,p,r)['binding_term']})
          print('E4',mode,name,kind,flush=True)
      for (name,kind),rs in all_results.items():
        b1=all_results['B1',kind];b2=all_results['B2',kind]
        r1=[r['cycles']/b['cycles'] for r,b in zip(rs,b1)];r2=[r['cycles']/b['cycles'] for r,b in zip(rs,b2)]
        lo1,hi1=paired_ci(r1);lo2,hi2=paired_ci(r2)
        row={'entry':name,'onchip_mode':mode,'design':ds[name].geometry}
        for batch in BATCHES:row['B'+str(batch)]=gmean(r['latency_ms'] for w,r in zip(ws,rs) if w['batch']==batch)
        row.update(all_geomean=gmean(r['latency_ms'] for r in rs),ratio_vs_B1=gmean(r1),ratio_vs_B2=gmean(r2),
            ci95_low_vs_B1=100*lo1,ci95_low_vs_B2=100*lo2,gate_5pct_pass=name not in ('U1','U2') and gmean(r1)<=.95 and gmean(r2)<=.95,
            sched_type=kind,search_certified=sel['modes'][mode].get('all_family_optima_certified',False))
        table.append(row)
      for name in ('B1','B2','best_hetero'):
        d=ds[name];rr=run_many(ws,d,replace(p,credits=512),jobs=args.jobs);base=all_results[name,'runtime']
        for batch in (*BATCHES,'all'):
          ids=[i for i,w in enumerate(ws) if batch=='all' or w['batch']==batch]
          sensitivity.append({'entry':name,'onchip_mode':mode,'batch':batch,'design':d.geometry,'credits':512,
            'hbm_cap_GB_s':replace(p,credits=512).hbm_bandwidth,'geomean_ms':gmean(rr[i]['latency_ms'] for i in ids),
            'ratio_vs_frozen_256':gmean(rr[i]['cycles']/base[i]['cycles'] for i in ids),'hardware_researched':False})
    out=ROOT/'results/E4';write_csv(out/'heldout_main_table.csv',table);write_csv(out/'per_window.csv',raw)
    for mode in MODES:
     for kind in ('milp','runtime'):
      for name in sorted({r['entry'] for r in raw}):
       for batch in (*BATCHES,'all'):
        ss=[r for r in raw if r['onchip_mode']==mode and r['sched_type']==kind and r['entry']==name and (batch=='all' or r['batch']==batch)]
        if not ss:continue
        row={'entry':name,'onchip_mode':mode,'sched_type':kind,'batch':batch}
        for key in ('core0_compute_busy','core1_compute_busy','w_port_busy','x_port_busy','acc_port_busy','hbm_busy_frac','core_finish_gap','idle_frac'):
          vs=[r[key] for r in ss if r[key] is not None];row[key]=sum(vs)/len(vs) if vs else None
        row['binding_term']=max({r['binding_term'] for r in ss},key=lambda k:sum(r['binding_term']==k for r in ss));bd.append(row)
    write_csv(out/'breakdown.csv',bd);write_csv(out/'hbm512_sensitivity.csv',sensitivity)
    summary='# E4 留出集结果\n\n使用窗口配对延迟比几何平均，不用总ms选择或下结论。CI列为速度降低百分比的95%下界，正数表示更快。\n\n'
    for mode in MODES:
      r=next(r for r in table if r['onchip_mode']==mode and r['entry']=='best_hetero' and r['sched_type']=='runtime')
      d=designs_from_selection(sel,mode)['best_hetero'];small=min(c.macs for c in d.cores)/12288
      summary+=f"- {mode}: 最优已评估异构/B1={r['ratio_vs_B1']:.6f}，/B2={r['ratio_vs_B2']:.6f}，{'达到' if r['gate_5pct_pass'] else '未达到'}进入校准门槛；小核乘法器占{small*100:.3f}%。\n"
    for mode in MODES:
      for name in ('U1','U2'):
        r=next(r for r in table if r['onchip_mode']==mode and r['entry']==name and r['sched_type']=='runtime')
        summary+=f"- {mode}/{name}: 相对B1延迟降低{100*(1-r['ratio_vs_B1']):.3f}%。\n"
    summary+='\n尚无RTL校准，不宣布架构胜出；family全局证明未闭合时，best仅指已评估候选。U1为逐专家按独占代价选形状、零切换的诊断参考，不是整层最优时延上界；U2仅去掉重复阵列权重读的诊断参考，不混入硬件最优。各忙碌量重叠，不相加成墙钟时间。'
    finalize(out,COMMAND+' --stage E4 --jobs '+str(args.jobs),summary)


def e5(args):
    data=inputs();dev=data['development'];ws=data['heldout'];sel=selection_file();dr=[];pr=[];states=[];taskrows=[]
    for mode in MODES:
      p=Parameters(onchip_mode=mode);ds=designs_from_selection(sel,mode)
      for name in ('best_hetero','fixed_4+2'):
        d=ds[name];threshold_scores=[]
        for t in (1,2,3,4,6,8,12,16):
            rr=run_many(dev,d,p,policy=f'threshold_fallback_{t}',jobs=args.jobs)
            threshold_scores.append((gmean(r['cycles'] for r in rr),t))
        threshold=min(threshold_scores)[1]
        policies=['threshold_2',f'threshold_fallback_{threshold}','adaptive','eft','random','milp']
        dispatch={}
        for policy in policies:
            ev=run_many(ws,d,p,kind='milp' if policy=='milp' else 'runtime',policy=policy,jobs=args.jobs)
            dispatch[policy]=[pack_eval(r,'milp' if policy=='milp' else 'runtime') for r in ev]
        baseline=dispatch['milp']
        for policy,rr in dispatch.items():
          for batch in (*BATCHES,'all'):
            ids=[i for i,w in enumerate(ws) if batch=='all' or w['batch']==batch]
            dr.append({'design':name,'geometry':d.geometry,'onchip_mode':mode,'policy':policy,'batch':batch,
                'threshold_tuned_on_dev':threshold,'geomean_ms':gmean(rr[i]['latency_ms'] for i in ids),
                'ratio_vs_milp_sched':gmean(rr[i]['cycles']/baseline[i]['cycles'] for i in ids)})
        predictor_results={};stats={}
        for pname in ('random','static','btb','ema','ours','oracle'):
          # Repeat complete warmup+heldout sequence with fresh identical initial state, not each individual learned window.
          repeats=[]
          for repeat in range(2):
            predictor=Predictor(pname);warm=[];rr=[]
            for w in dev:
                warm.append(simulate(w,d,p,policy='eft',predictor=predictor))
            for w in ws:
                if pname=='oracle':
                    # First pass profiles the same policy; its ownership may change on replay. Residual is measured.
                    profile=simulate(w,d,p,policy='eft',predictor=predictor);predictor.absorb_profile(profile,w)
                rr.append(simulate(w,d,p,policy='eft',predictor=predictor))
            repeats.append(rr)
          assert canonical(repeats[0])==canonical(repeats[1]),'predictor sequence not reproducible'
          rr=repeats[0];predictor_results[pname]=rr
          errors=[];success=0;late=0;stall=0;nnext=0
          for w,r in zip(ws,rr):
           for t in r['tasks']:
            actual=t['actual_cycles'];errors.append(abs(t['predicted_cycles']-actual)/actual)
            current=t.get('current_end');ready=t.get('first_weight_ready')
            if current is not None and ready is not None:
                nnext+=1;core=d.cores[t['core']];e=w['experts'][t['expert_index']]
                W=2*((ceildiv(e['Me'],core.pm)-1)*p.issue_interval+p.dot_latency(core)+1)
                success+=current-W<=ready<=current;late+=ready>current;stall+=max(0,ready-current)
            taskrows.append({'design':name,'onchip_mode':mode,'predictor':pname,'window_id':w['id'],
                'expert_id':t['expert_id'],'core':t['core'],'predicted_cycles':t['predicted_cycles'],
                'actual_cycles':actual,'first_weight_ready':ready,'current_end':current,
                'prediction_abs_error_pct':100*errors[-1]})
          stats[pname]={'mae_pct':100*sum(errors)/len(errors),'success_pct':100*success/nnext if nnext else None,
            'late_pct':100*late/nnext if nnext else None,'stall_cycles':stall,'next_samples':nnext}
          states.append({'design':name,'onchip_mode':mode,'predictor':pname,'predictor_state_bits':predictor.state_bits(),
            'task_fifo_bits':8*512,'core_current_next_bits':2*2*512,'credit_slot_scoreboard_bits':2*(32*5+8*16),
            'synthesized_area':None,'synthesized_fmax':None,'scope':'state estimate; excludes queues common to all policies'})
          print('E5',mode,name,pname,flush=True)
        oracle=predictor_results['oracle'];ours=predictor_results['ours']
        for pname,rr in predictor_results.items():
          pr.append({'design':name,'onchip_mode':mode,'predictor':pname,**stats[pname],
            'e2e_ratio_vs_oracle':gmean(r['cycles']/b['cycles'] for r,b in zip(rr,oracle)),
            'e2e_ratio_vs_ours':gmean(r['cycles']/b['cycles'] for r,b in zip(rr,ours)),
            'geomean_ms':gmean(r['latency_ms'] for r in rr),
            'oracle_caveat':'profile-guided two-pass; changed ownership can leave residual error'})
    out=ROOT/'results/E5';write_csv(out/'dispatch_table.csv',dr);write_csv(out/'predictor_table.csv',pr)
    write_csv(out/'dispatcher_state_bits.csv',states);write_csv(out/'task_prediction_errors.csv',taskrows)
    summary='# E5 在线派工与预测\n\n所有预测器先按同一18开发窗口序列预热，再按同一135留出窗口计数，两次完整序列一致。Runtime为8项候选窗、每核Current+Next最多2项，实际资源检查后绑定，支持等待；估计不能替代就绪/依赖检查。\n\n'
    for mode in MODES:
     for name in ('best_hetero','fixed_4+2'):
      ss=[r for r in dr if r['design']==name and r['onchip_mode']==mode and r['batch']=='all'];a=next(r for r in ss if r['policy']=='threshold_2');b=next(r for r in ss if r['policy'].startswith('threshold_fallback'));eft=next(r for r in ss if r['policy']=='eft')
      ps=[r for r in pr if r['design']==name and r['onchip_mode']==mode];best=min(ps,key=lambda r:r['geomean_ms']);worst=max(ps,key=lambda r:r['geomean_ms'])
      summary+=f"- {mode}/{name}: 纯阈值比回退慢{100*(a['geomean_ms']/b['geomean_ms']-1):.3f}%；EFT/MILP-LPT={eft['ratio_vs_milp_sched']:.5f}；预测器最好{best['predictor']}与最差{worst['predictor']}延迟相差{100*(worst['geomean_ms']/best['geomean_ms']-1):.3f}%。\n"
    finalize(out,COMMAND+' --stage E5 --jobs '+str(args.jobs),summary,
        'MAE=平均|预测时长−实测时长|/实测时长。成功窗口为Current结束前两个权重块计算时间内；是近似模型内准时性。late=第一块晚于Current结束；stall为暴露权重等待，不与端口占用相加。oracle是profile-guided参考而非强制零误差。状态账本仅为可量化状态，不声称综合面积。')


def e6(args):
    rows=read_csv(ROOT/'results/E4/heldout_main_table.csv');moe=[];model=[]
    for r in rows:
      if r['entry'] not in ('B0','B1','B2','best_hetero'):continue
      for batch in BATCHES:
        baseline=next(b for b in rows if b['entry']=='B1' and b['onchip_mode']==r['onchip_mode'] and b['sched_type']==r['sched_type'])
        ms=float(r['B'+str(batch)]);ratio=ms/float(baseline['B'+str(batch)])
        moe.append({'design':r['entry'],'onchip_mode':r['onchip_mode'],'sched_type':r['sched_type'],'batch':batch,
            'moe_ms_per_layer':ms,'ratio_vs_B1':ratio,'source':'E4/heldout_main_table.csv'})
        model.append({'design':r['entry'],'batch':batch,'onchip_mode':r['onchip_mode'],'sched_type':r['sched_type'],
            'moe_ms_per_layer':ms,'non_moe_ms_per_layer':None,'layers':None,'token_ms':None,'ratio_vs_B1':None,
            'status':'missing_matching_DeepSeek_non_MoE_layer_timing'})
    out=ROOT/'results/E6';write_csv(out/'moe_layer_e2e.csv',moe);write_csv(out/'model_token_e2e.csv',model)
    finalize(out,COMMAND+' --stage E6',
        '# E6 边界\n\nMoE层结果直接来自E4，不重复计算。完整模型每token时间缺少与冻结DeepSeek捕获匹配的attention/router/norm时序以及层映射，留空并标缺失。其他模型的旧PLENA时间不能拼接成这套模型的端到端结果。GPU比较未做。整模型计时项为未完成（输入缺失）。')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['E1','E2','E2micro','E2layer','E4','E5','E6'],required=True);parser.add_argument('--jobs',type=int,default=24)
    args=parser.parse_args()
    if args.stage=='E2':e2micro(args);e2layer(args)
    else:globals()[args.stage.lower()](args)
if __name__=='__main__':main()
