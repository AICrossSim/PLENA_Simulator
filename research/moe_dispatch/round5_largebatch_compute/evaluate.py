"""Frozen-hardware holdout evaluation, causal runtime and vector controls."""
import argparse,json,math,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ROOT,BUDGETS,BATCHES,write_csv,write_json,digest,gm
from .model import Core,cost,build_tasks
from .search import costs,INDEX,matrix_for
from .schedule import optimize,online
from .inputs import load

def detail(w,tasks,cs,owner,cycles,mode,total,flow='AUTO'):
    service=[0]*len(cs);useful=[0]*len(cs);issued=[0]*len(cs);rec=[0]*len(cs);vec=[0]*len(cs)
    row_padded=[0]*len(cs);flows={};matrix=[0]*len(cs)
    for t,c in zip(tasks,owner):
        q=cost(cs[c],t['m'],t['f'],mode,total,flow)
        service[c]+=q['cycles'];matrix[c]+=q['matrix_cycles'];useful[c]+=q['useful'];issued[c]+=q['issued']
        rec[c]+=q['recurrence_cycles'];vec[c]+=q['vector_excess'];row_padded[c]+=q['useful']/q['row_eff']
        flows[q['flow']]=flows.get(q['flow'],0)+1
    total_useful=sum(useful);assert total_useful==3*2048*1408*8*w['batch']
    assert all(s<=cycles for s in service)
    weight_bytes=sum(3*2048*e['F']*2 for e in w['experts'])
    bytype={}
    for i,c in enumerate(cs):
        x=bytype.setdefault(c.name,dict(cores=0,MACs=0,useful=0,service=0,maximum=0))
        x['cores']+=1;x['MACs']+=c.macs;x['useful']+=useful[i];x['service']+=service[i];x['maximum']=max(x['maximum'],service[i])
    for x in bytype.values():
        x['work_pct']=100*x['useful']/total_useful;x['idle_pct']=100*(1-x['service']/(x['cores']*cycles))
    busy=[s for s in service if s>0]
    return dict(cycles=int(cycles),ms=cycles/1e6,GFLOPs=2*total_useful/cycles,
        utilization_pct=100*total_useful/(total*cycles),padding_pct=100*(1-total_useful/sum(issued)),
        row_fill_pct=100*total_useful/sum(row_padded),idle_MAC_pct=100*(1-sum(s*c.macs for s,c in zip(service,cs))/(total*cycles)),
        service_imbalance=max(busy)/min(busy) if busy else 1,unused_cores=sum(s==0 for s in service),
        core_cycles=json.dumps(service),core_matrix_cycles=json.dumps(matrix),core_work_pct=json.dumps([100*x/total_useful for x in useful]),
        core_idle_pct=json.dumps([100*(1-s/cycles) for s in service]),pool_metrics=json.dumps(bytype),
        recurrence_core_cycles=sum(rec),vector_excess_core_cycles=sum(vec),flows=json.dumps(flows),
        useful_MACs=total_useful,issued_MACs=sum(issued),unique_weight_MiB=weight_bytes/2**20,
        unique_weight_BW_TBs=weight_bytes/cycles/1000,
        owner=json.dumps(owner))

def evaluate_case(path,heldout,work=.08):
    target=path/'holdout.csv.gz'
    if target.exists() and (path/'holdout_receipt.json').exists():return
    selected=json.loads((path/'selected.json').read_text());first=next(iter(selected.values()))
    total=first['total_macs'];mode=first['mode'];gran=first['granularity'];chunk=64 if gran=='shared64' else 0
    data,counts,tasks=costs(heldout,mode,chunk,total)
    controls={}
    if mode=='full_scaled':controls['fixed_vector64']=costs(heldout,'full_fixed',chunk,total)[0]
    controls['uniform_OS']=costs(heldout,mode,chunk,total,force_flow='OS')[0]
    rows=[];start=time.monotonic()
    for family,s in selected.items():
        cs=tuple(Core(**c) for c in s['cores']);idx=tuple(INDEX[c] for c in cs)
        for control,table in [('primary',data)]+list(controls.items()):
            used_mode='full_fixed' if control=='fixed_vector64' else mode
            flow='OS' if control=='uniform_OS' else 'AUTO'
            def job(d):
                p=matrix_for(table,idx,d,counts);w=heldout[d];ts=tasks[d]
                effort=work if control=='primary' else min(work,.02)
                r=optimize(p,idx,work=effort);r2=optimize(p,idx,work=effort)
                # CP-SAT's diagnostic work counter can differ by one floating
                # point ULP while the bound, mapping and integer cycles are
                # identical. Check all semantic results exactly; separately
                # constrain that diagnostic-only numerical delta.
                semantic=lambda q:{k:v for k,v in q.items() if k!='deterministic_time'}
                work_delta=abs(r.get('deterministic_time',0)-r2.get('deterministic_time',0))
                assert semantic(r)==semantic(r2),'integer scheduling result repeat mismatch'
                assert work_delta<1e-10,'solver work changed materially'
                expected=max(sum(int(p[i,c]) for i in range(len(p)) if r['owner'][i]==c) for c in range(len(cs)))
                assert expected==r['upper']
                base=dict(total_macs=total,mode=mode,granularity=gran,family=family,control=control,
                    geometry=s['geometry'],window_id=w['id'],batch=w['batch'],dataset=w['dataset'],group=w['group'],layer=w['layer'],
                    lower_cycles=r['lower'],offline_gap_pct=100*(r['upper']/r['lower']-1),repeat_exact=True,solver_deterministic_work=effort,solver_work_counter_repeat_delta=work_delta)
                result=[dict(base,scheduler='offline',**detail(w,ts,cs,r['owner'],r['upper'],used_mode,total,flow))]
                if control=='primary':
                    for pre in (True,False):
                        runtime=online(p,ts,idx,preconfig=pre)
                        assert runtime==online(p,ts,idx,preconfig=pre)
                        for c in range(len(cs)):
                            spans=sorted((runtime['starts'][i],runtime['starts'][i]+p[i,c]) for i in range(len(ts)) if runtime['owner'][i]==c)
                            assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
                        result.append(dict(base,scheduler='runtime' if pre else 'runtime_no_preconfig',
                            **detail(w,ts,cs,runtime['owner'],runtime['cycles'],used_mode,total,flow)))
                return result
            with ThreadPoolExecutor(max_workers=4) as pool:
                for rr in pool.map(job,range(len(heldout))):rows.extend(rr)
        print('heldout',total,mode,gran,family,'elapsed',round(time.monotonic()-start,1),flush=True)
    write_csv(target,rows)
    write_json(path/'holdout_receipt.json',dict(windows=len(heldout),rows=len(rows),input_digest=digest(heldout),
        selection_digest=digest(selected),all_point_repeats_exact=True,integer_optimizer_deterministic_work=work,
        wall_seconds=time.monotonic()-start))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int);ap.add_argument('--mode');ap.add_argument('--granularity');ap.add_argument('--work',type=float,default=.08);a=ap.parse_args()
    held=load()['heldout']
    for total in (a.budget,) if a.budget else BUDGETS:
        for mode in (a.mode,) if a.mode else ('matrix','full_scaled'):
            for gran in (a.granularity,) if a.granularity else ('expert','shared64'):
                p=ROOT/'results'/f'{total}_{mode}_{gran}'
                evaluate_case(p,held,a.work)
    print('holdout evaluation complete',flush=True)

if __name__=='__main__':main()
