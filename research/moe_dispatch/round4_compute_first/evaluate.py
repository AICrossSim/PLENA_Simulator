"""Repeated window replay and exact numerical/byte ledgers for E1 and E3."""
import argparse, json, math, time
from collections import defaultdict
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from .common import ROOT,inputs,gm,digest,write_json,write_csv
from .config import Core,parameters,BATCHES,TOTAL_MACS
from .compute import pure,cost,quotas
from .model import Design,task_cost
from .runtime import simulate

def selected(): return json.loads((ROOT/'E1/selected_designs.json').read_text())
def decode(s):
    d=dict(s); d['cores']=tuple(Core(**c) for c in d['cores'])
    return Design(**d)
def unique(w): return sum(6*e['H']*e['F'] for e in w['experts'])
def io_bytes(w): return 6*w['batch']*w['hidden']
def flops(w): return sum(6*e['Me']*e['H']*e['F'] for e in w['experts'])

def provisioning_peak(tasks,n):
    # Each task reserves its per-category peak throughout its execution.
    # Report this explicit feasible envelope, not an alleged minimum cache.
    changes=[]
    for t in tasks:
        vals=[t[k] for k in ('W','X','acc','Z')]
        changes.extend([(t['start'],1,vals),(t['finish'],-1,vals)])
    live=[0]*4; peaks=[0]*4; total=0
    for _,sgn,vs in sorted(changes,key=lambda x:(x[0],x[1])):
        live=[a+sgn*b for a,b in zip(live,vs)]
        peaks=[max(a,b) for a,b in zip(peaks,live)]
        total=max(total,sum(live))
    return total,peaks

def ideal_row(w,s):
    cs=tuple(Core(**c) for c in s['cores']); r=pure(w,cs,tuple(s['flows']))
    T=r['cycles']; ts=r['tasks']; total,peaks=provisioning_peak(ts,len(cs))
    # Global packed input and FP32 combined output + unchanged control stores.
    fixed_live=io_bytes(w)+40*1024
    row=dict(mode='B',design=s['design'],pn_cap=s['pn_cap'],family=s['family'],
       window_id=w['id'],batch=w['batch'],Tc_ms=T/1e6,hbm_bytes=unique(w)+io_bytes(w),
       unique_bytes=unique(w),unique_with_io_bytes=unique(w)+io_bytes(w),flops=flops(w),
       sram_total_peak=total+fixed_live,sram_fixed_live=fixed_live,
       sram_peak_W=peaks[0],sram_peak_X=peaks[1],sram_peak_acc=peaks[2],sram_peak_Z=peaks[3],
       capacity_model='simultaneous task-reservation envelope',
       actual_hbm_scope='unique expert weights plus one BF16 input and FP32 output',
       mac_util_pct=100*flops(w)/(2*TOTAL_MACS*T),core_count=len(cs))
    for j,c in enumerate(cs):
        tt=[t for t in ts if t['core']==j]; service=sum(t['cycles'] for t in tt)
        row.update({f'core{j}_service_ms':service/1e6,
                    f'core{j}_matrix_ms':sum(t['matrix_cycles'] for t in tt)/1e6,
                    f'core{j}_tasks':len(tt),f'core{j}_work_pct':100*sum(t['macs'] for t in tt)/(flops(w)/2),
                    f'core{j}_idle_pct':100*(1-service/T)})
        for k in ('W','X','acc','Z'): row[f'core{j}_sram_peak_{k}']=max((t[k] for t in tt),default=0)
        for k in ('W','X','acc'):
            row[f'core{j}_port_avg_{k}']=sum(t[k+'_port'] for t in tt)/T
            row[f'core{j}_port_peak_{k}']=max((t[k+'_peak'] for t in tt),default=0)
            row[f'core{j}_port_issue_peak_{k}']=max((t[k+'_issue_peak'] for t in tt),default=0)
    for k in ('W','X','acc'):
        row['port_avg_'+k]=sum(row[f'core{j}_port_avg_{k}'] for j in range(len(cs)))
        # Sum of private peak provisioning is sufficient for all overlaps.
        row['port_peak_'+k]=sum(row[f'core{j}_port_peak_{k}'] for j in range(len(cs)))
        row['port_issue_peak_'+k]=sum(row[f'core{j}_port_issue_peak_{k}'] for j in range(len(cs)))
    return row,r

def finite_row(w,s,bw=None):
    if not s['real_capacity_feasible']:
        raise ValueError(s['design']+': '+s['capacity_reason'])
    d=decode(s['real_sram_design']); p=parameters(bw,control=bw is not None)
    # fixed is the round-three capacity-aware online protocol; all n cores
    # use the same queue/admission/ports, including the new four-core case.
    if bw is None:
        # Isolate SRAM from scheduler changes: replay the exact B ownership
        # and per-core order. No shared resource can exceed its quota here;
        # each core has fixed private ports and vector shares summing to64.
        cs=tuple(Core(**c) for c in s['cores']); reference=pure(w,cs,tuple(s['flows']))
        clear=w['batch']*w['hidden']*4/192
        ends=[clear]*len(cs);tasks=[];phases=[];hbm=0
        for t in reference['tasks']:
            i,j=t['expert_index'],t['core'];co=task_cost(w['experts'][i],d,j,p)
            start=ends[j];now=start
            for pi,q in enumerate(co.phases):
                dt=max(q.compute_cycles,q.w_sram_bytes/(d.w_banks[j]*16),
                       q.x_sram_bytes/(d.x_banks[j]*16),q.acc_sram_bytes/(d.acc_banks[j]*16),
                       q.vector_elements/d.vector_lanes[j])
                phases.append(dict(expert_index=i,core=j,phase=pi,start=now,finish=now+dt,**asdict(q)));now+=dt
            ends[j]=now;hbm+=co.hbm_bytes
            tasks.append(dict(expert_index=i,core=j,start=start,finish=now,actual_cycles=now-start))
        r=dict(cycles=max(ends),tasks=tasks,phases=phases,hbm_bytes=hbm)
    else:
        r=simulate(w,d,p,dispatch='fixed',t_big=3,large_first=True,predictor=None)
    # Third-round engine times weight streams. Add explicit global activation
    # ingress/egress at the layer boundaries; no silent omission from HBM.
    extra_time=0.0 if bw is None else 2*p.hbm_latency+io_bytes(w)/p.hbm_bandwidth
    T=r['cycles']+extra_time
    row=dict(mode='A' if bw is None else 'system',design=s['design'],pn_cap=s['pn_cap'],family=s['family'],
       window_id=w['id'],batch=w['batch'],Tc_ms=T/1e6,bw_GBps=0 if bw is None else bw,
       hbm_bytes=r['hbm_bytes']+io_bytes(w),unique_bytes=unique(w),
       unique_with_io_bytes=unique(w)+io_bytes(w),flops=flops(w),
       mac_util_pct=100*flops(w)/(2*TOTAL_MACS*T),core_count=len(d.cores),
       installed_sram_bytes=2158592,capacity_clipped=True,
       hbm_busy_pct=0.0 if bw is None else 100*(r['hbm_bytes']+io_bytes(w))/(T*bw),
       global_io_time_ms=extra_time/1e6,credits=p.credits,
       dispatch='B_ownership_replay' if bw is None else 'fixed',predictor='nominal analytical cost; no heldout training')
    for j,c in enumerate(d.cores):
        ts=[t for t in r['tasks'] if t['core']==j]
        ps=[q for q in r['phases'] if q['core']==j]
        service=sum(t['actual_cycles'] for t in ts)
        row.update({f'core{j}_service_ms':service/1e6,
            f'core{j}_matrix_ms':sum(q['compute_cycles'] for q in ps)/1e6,
            f'core{j}_tasks':len(ts),f'core{j}_work_pct':100*sum(q['useful_macs'] for q in ps)/(flops(w)/2),
            f'core{j}_idle_pct':max(0,100*(1-service/T)),
            f'core{j}_refetch_tasks':sum(task_cost(w['experts'][t['expert_index']],d,j,p).hbm_bytes>
                                      task_cost(w['experts'][t['expert_index']],d,j,p).unique_hbm_bytes for t in ts)})
        for k,name in (('W','w_bytes'),('X','x_bytes'),('acc','acc_bytes'),('Z','z_bytes')):
            row[f'core{j}_sram_peak_{k}']=getattr(d,name)[j] if ts else 0
        for k,attr in (('W','w_sram_bytes'),('X','x_sram_bytes'),('acc','acc_sram_bytes')):
            row[f'core{j}_port_avg_{k}']=sum(q[attr] for q in ps)/T
            row[f'core{j}_port_peak_{k}']=max((q[attr]/(q['finish']-q['start']) for q in ps),default=0)
    for k in ('W','X','acc','Z'):
        row['sram_peak_'+k]=sum(row[f'core{j}_sram_peak_{k}'] for j in range(len(d.cores)))
    row['sram_total_peak']=2158592
    for k in ('W','X','acc'):
        row['port_avg_'+k]=sum(row[f'core{j}_port_avg_{k}'] for j in range(len(d.cores)))
        row['port_peak_'+k]=sum(row[f'core{j}_port_peak_{k}'] for j in range(len(d.cores)))
    return row,r

def job(spec):
    name,s,bws=spec; ws=inputs()['heldout']; rows=[]; checks=[]
    for bw in bws:
        for w in ws:
            fn=ideal_row if bw=='B' else lambda w,s:finite_row(w,s,bw)
            a,raw=fn(w,s); b,raw2=fn(w,s)
            da,db=digest(raw),digest(raw2)
            assert a==b and da==db,(name,w['id'],bw)
            a['result_sha256']=da; rows.append(a)
            checks.append(dict(design=name,window=w['id'],mode=str(bw),sha256=da,repeat_exact=True))
    print(name,'complete',len(rows),flush=True)
    return rows,checks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['E1','E3'],required=True);ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args(); specs=[]
    for name,s in selected().items():
        specs.append((name,s,('B',None) if args.stage=='E1' else (256,1229)))
    rows=[];checks=[]
    with ProcessPoolExecutor(args.jobs) as pool:
        for a,b in pool.map(job,specs): rows.extend(a);checks.extend(b)
    out=ROOT/args.stage;out.mkdir(exist_ok=True)
    write_csv(out/('compute_by_window.csv' if args.stage=='E1' else 'system_by_window.csv'),rows)
    write_json(out/'repeat_checks.json',checks)
    if args.stage=='E1': summarize(rows)

def summarize(rows):
    index={(r['mode'],r['design'],r['window_id']):r for r in rows}; result=[]
    for s in selected().values():
        cap=s['pn_cap']; base=f'pn{cap}_single'
        # Stronger homo2/homo4 baseline is chosen on DEV, not heldout.
        sd=selected(); homo=min((f'pn{cap}_homo2',f'pn{cap}_homo4'),key=lambda k:sd[k]['dev_GM_cycles'])
        for mode in ('B','A'):
            for batch in (*BATCHES,'all'):
                rs=[r for r in rows if r['design']==s['design'] and r['mode']==mode and (batch=='all' or r['batch']==batch)]
                result.append(dict(mode=mode,design=s['design'],pn_cap=cap,batch=batch,
                    geometry='+'.join(f"{c['pm']}x{c['pn']}x{c['pk']}" for c in s['cores']),flows='/'.join(s['flows']),
                    windows=len(rs),Tc_ms=gm(r['Tc_ms'] for r in rs),
                    eq_GFLOPs=gm(r['flops']/(r['Tc_ms']*1e6) for r in rs),
                    mac_util_pct=gm(r['mac_util_pct'] for r in rs),
                    ratio_vs_single=gm(r['Tc_ms']/index[mode,base,r['window_id']]['Tc_ms'] for r in rs),
                    homo_baseline=homo,ratio_vs_homo=gm(r['Tc_ms']/index[mode,homo,r['window_id']]['Tc_ms'] for r in rs)))
    write_csv(ROOT/'E1/compute_main_table.csv',result)

if __name__=='__main__':main()
