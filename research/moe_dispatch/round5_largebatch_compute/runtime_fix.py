"""Causal LPT priority + finish-time admission; retain naive window baseline.

The original window policy minimized each new job's finish time across both
jobs and cores. That silently converted LPT to shortest-job-first and admitted
to a worse core when the best queue was full. This reference uses one common
rule for every organization, with no hardware/threshold fitting on holdout.
"""
import csv,fcntl,gzip,hashlib,json,os,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import ROOT,write_csv,write_json
from .model import Core
from .inputs import load

def admission(p,tasks,types,window=8,dispatch_cycles=1,preconfig=True):
    service=np.asarray(p,dtype=np.int64).tolist();nc=len(types);nt=len(tasks)
    pending=sorted(range(nt),key=lambda i:(-min(service[i]),i))
    end=[0]*nc;queues=[[] for _ in range(nc)];owners=[-1]*nt;starts=[0]*nt
    now=0;waits=0;bypasses=0
    while pending:
        for q in queues:
            while q and q[0]<=now:q.pop(0)
        choice=None
        # Preserve priority. A blocked head may be bypassed only by a task
        # whose own best predicted destination can currently accept it.
        for position,i in enumerate(pending[:window]):
            candidates=[]
            for c in range(nc):
                start=max(now+dispatch_cycles,end[c])
                setup=max(0,2-(start-now-dispatch_cycles)) if preconfig else 2
                candidates.append((start+setup+service[i][c],len(queues[c])>=2,c,start+setup))
            finish,_,c,start=min(candidates)
            if len(queues[c])<2:
                choice=(i,c,start,finish);bypasses+=int(position>0);break
        if choice is None:
            future=[q[0] for q in queues if q]
            assert future and min(future)>now
            now=min(future);waits+=1;continue
        i,c,start,finish=choice;pending.remove(i);owners[i]=c;starts[i]=start
        end[c]=finish;queues[c].append(finish);now+=dispatch_cycles
    return dict(cycles=max(end),owner=owners,starts=starts,end=end,admission_wait_events=waits,priority_bypasses=bypasses)

def retrofit(path):
    path=ROOT/'results'/path if isinstance(path,str) else path
    if not (path/'holdout_receipt.json').exists():return None
    with (path/'runtime_fix.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (path/'runtime_fix_receipt.json').exists():return path.name
        from .evaluate import detail
        from .search import costs,INDEX,matrix_for
        ws=load()['heldout'];selected=json.loads((path/'selected.json').read_text());first=next(iter(selected.values()))
        total=first['total_macs'];mode=first['mode'];gran=first['granularity'];chunk=64 if gran=='shared64' else 0
        data,counts,tasks=costs(ws,mode,chunk,total);window_index={w['id']:i for i,w in enumerate(ws)}
        file=path/'holdout.csv.gz';original_sha=hashlib.sha256(file.read_bytes()).hexdigest()
        with gzip.open(file,'rt') as f:rows=list(csv.DictReader(f))
        if any(r['scheduler']=='runtime_naive_window' for r in rows):
            # Recover an interrupted final receipt write after atomic data
            # replacement, without duplicating the corrected rows.
            assert sum(r['scheduler']=='runtime' for r in rows)==len(ws)*len(selected)
        else:
            for r in rows:
                if r['scheduler']=='runtime':r['scheduler']='runtime_naive_window'
                elif r['scheduler']=='runtime_no_preconfig':r['scheduler']='runtime_naive_no_preconfig'
            extra=[]
            for base in rows:
                if base['control']!='primary' or base['scheduler']!='offline':continue
                d=window_index[base['window_id']];w=ws[d];ts=tasks[d];s=selected[base['family']]
                cs=tuple(Core(**c) for c in s['cores']);idx=tuple(INDEX[c] for c in cs);p=matrix_for(data,idx,d,counts)
                for pre in (True,False):
                    r=admission(p,ts,idx,preconfig=pre);assert r==admission(p,ts,idx,preconfig=pre)
                    for c in range(len(cs)):
                        spans=sorted((r['starts'][i],r['starts'][i]+p[i,c]) for i in range(len(ts)) if r['owner'][i]==c)
                        assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
                    assert r['cycles']>=int(base['lower_cycles'])
                    extra.append(dict(base,scheduler='runtime' if pre else 'runtime_no_preconfig',
                        admission_wait_events=r['admission_wait_events'],priority_bypasses=r['priority_bypasses'],
                        **detail(w,ts,cs,r['owner'],r['cycles'],mode,total)))
            rows+=extra;temp=path/'holdout.fixed.csv.gz';write_csv(temp,rows);os.replace(temp,file)
        receipt=json.loads((path/'holdout_receipt.json').read_text());receipt.update(rows=len(rows),runtime_policy='LPT priority, compare all cores including temporarily-full queues; bounded8-entry bypass')
        write_json(path/'holdout_receipt.json',receipt)
        write_json(path/'runtime_fix_receipt.json',dict(rows=len(rows),original_naive_file_sha256=original_sha,
            corrected_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),all_new_runtime_points_repeated_exactly=True,
            geometry_and_offline_results_unchanged=True,policy_parameters_fitted_on_holdout=False,
            note='Post-hoc correction of generic priority/admission semantics; naive results retained for transparent comparison. No per-batch or per-organization tuning.'))
        print('RUNTIME ADMISSION RETROFIT',path.name,flush=True)
        return path.name

def retrofit_all(workers=4):
    pending=[p for p in sorted((ROOT/'results').iterdir()) if (p/'holdout_receipt.json').exists() and not (p/'runtime_fix_receipt.json').exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(retrofit,pending))

if __name__=='__main__':retrofit_all()
