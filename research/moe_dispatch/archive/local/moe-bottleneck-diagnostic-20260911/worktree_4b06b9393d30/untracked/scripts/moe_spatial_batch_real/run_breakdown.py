#!/usr/bin/env python3
"""Trace-checked issue-side partition; overlapping port service is separate.
Every interval boundary that can change the selected blocker is reconstructed.
This is a priority state classification, NOT independent causal stall removal.
"""
import csv,gzip,json,subprocess,tempfile
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import run_routes as r

DEST=r.OUT/'breakdown'
def union_length(intervals):
    end=0;total=0
    for a,b in sorted(intervals):total+=max(0,b-max(a,end));end=max(end,b)
    return total
def classify(req,rep):
    tasks=rep['trace'];reads={};installs={};fills={};end=rep['total_cycles']
    assert req['compute']['issue_interval_cycles']==1
    for s in rep['services']:
        if s['resource']=='weight':
            for q in s['requests']:reads[q]=s
        if s['resource']=='control' and s['end']-s['start']==3:
            for q in s['requests']:installs[q]=s
    def tag(t):return tuple(t[k] for k in ['core','slot','expert','n_start','k_start','weight_ready'])
    for t in tasks:
        if not t['cache_hit']:fills[tag(t)]=t['request']
    rows=[]
    for ci,core in enumerate(rep['cores']):
        ts=[t for t in tasks if t['core']==ci];adm=defaultdict(list);issue=defaultdict(list);commit=Counter()
        points={0,end};cap=(req['compute']['result_latency_cycles']+req['compute']['issue_interval_cycles']-1)//req['compute']['issue_interval_cycles']
        for t in ts:
            adm[t['admitted']].append(t);issue[t['issue_cycle']].append(t);commit[t['commit_cycle']]+=1
            points.update(t[k] for k in ['admitted','descriptor_ready','weight_ready','activation_ready','issue_cycle','mac_done','rmw_done','commit_cycle'])
            points.add(t['issue_cycle']+1)
            if tag(t) in fills:
                q=fills[tag(t)]
                for s in [reads[q],installs[q]]:points.update(s[k] for k in ['released','start','end'])
        points=sorted(p for p in points if 0<=p<=end)
        stages=[];pending=0;count=Counter()
        for now,nxt in zip(points,points[1:]):
            pending-=commit.get(now,0);stages.extend(adm.get(now,[]))
            for t in issue.get(now,[]):stages.remove(t);pending+=1
            if issue.get(now):state='issue'
            elif pending>=cap:state='result_backpressure'
            elif stages:
                t=stages[0]
                if t['descriptor_ready']>now:state='control_issue'
                elif t['weight_ready']>now:
                    q=fills[tag(t)];w=reads[q];i=installs[q]
                    assert w['end']<=i['released'] and i['end']==t['weight_ready']
                    if now<w['start']:state='source_queue'
                    elif now<w['end']:state='source_transfer'
                    elif now<i['released']:state='distribution'
                    elif now<i['start']:state='install_queue'
                    else:state='install_service'
                elif t['activation_ready']>now:state='activation'
                else:state='ready_behind_other_stage'
            else:state='no_stage'
            count[state]+=nxt-now
        s=core['states'];weight_keys=['source_queue','source_transfer','distribution','install_queue','install_service']
        assert count['issue']==s.get('issue_interval',0)
        for k in ['control_issue','result_backpressure','activation','ready_behind_other_stage']:assert count[k]==s.get(k,0),(ci,k,count,s)
        assert sum(count[k] for k in weight_keys)==s.get('weight_delivery_or_install',0)
        residual=['descriptor_credit','weight_slot','dependency_or_ready_window','finished_idle']
        assert count['no_stage']==sum(s.get(k,0) for k in residual)
        assert sum(count.values())==end
        row=dict(core=ci,m_lanes=core['m_lanes'],wall_cycles=end,last_commit=max((t['commit_cycle'] for t in ts),default=0),
            mac_pipeline_nonempty_cycles=union_length((t['issue_cycle'],t['mac_done']) for t in ts))
        row.update({k:count[k] for k in ['issue','control_issue',*weight_keys,'activation','result_backpressure','ready_behind_other_stage']})
        row.update({k:s.get(k,0) for k in residual});rows.append(row)
    return rows

def execute(w,shape,mode):
    base=f"{w['name']}__routed_gate_up__{'_'.join(map(str,shape))}__{mode}__finite"
    dest=DEST/base;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'validation.json').exists():return json.loads((dest/'validation.json').read_text())
    req=r.helper.req(base,shape,r.jobs(w,'routed_gate_up'),mode,dict(control='tile_cohort'),trace=True)
    r.save(dest/'request.json',req);hashes=[]
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix='plena-trace-',dir='/tmp') as tmp:
            p=Path(tmp)/'out.json';proc=subprocess.run([str(r.FAB),'--request',str(dest/'request.json'),'--output',str(p)],capture_output=True,timeout=1200)
            if proc.returncode:raise RuntimeError(proc.stderr.decode())
            data=p.read_bytes();rep=json.loads(data)
        hashes.append(r.digest(data))
        if repeat==0:(dest/'trace.json.gz').write_bytes(gzip.compress(data,mtime=0))
        else:assert hashes[0]==hashes[1]
    r.helper.audit(req,rep)
    receipt=r.OUT/'route_campaign/receipts'/f'{base}.json'
    if receipt.exists():
        old=json.loads(gzip.decompress((r.OUT/'route_campaign/reports'/f'{base}.json.gz').read_bytes()))
        assert old['total_cycles']==rep['total_cycles']
        assert old['invocation_sha256']==rep['invocation_sha256'] and old['service_sha256']==rep['service_sha256']
    else:raise RuntimeError('paired compact run has not finished: '+base)
    common=dict(name=base,model=w['model'],dataset=w['dataset'],batch=w['batch'],shape='+'.join(map(str,shape)),mode=mode)
    rows=[{**common,**c} for c in classify(req,rep)]
    result=dict(passed=True,repeat_sha256=hashes,cores=rows,stats=rep['stats'],wall_cycles=rep['total_cycles'],
        classification='priority issue-side partition; pipeline occupancy and port service overlap',same_compact_timeline=True)
    r.save(dest/'validation.json',result);print(base,'PASS',flush=True);return result

def main():
    DEST.mkdir(exist_ok=True);windows=json.loads((r.OUT/'inputs/route_windows.json').read_text())
    plan=[(w,s,m) for w in windows if w['dataset']=='bfcl' and w['split']=='primary' for s in r.SHAPES for m in r.MODES]
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(execute,*p):p for p in plan}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:
                r.save(DEST/'FAILURE.json',dict(error=str(e),point=fs[f]))
                for x in fs:x.cancel()
                raise
            r.save(DEST/'progress.json',dict(done=len(results),planned=len(plan)))
    rows=[c for result in results for c in result['cores']]
    with (DEST/'core_states.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    last=[max(r['cores'],key=lambda c:c['last_commit']) for r in results]
    with (DEST/'last_core.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(last)
    r.save(DEST/'validation.json',dict(passed=True,points=len(results),runs=len(results)*2,cores=len(rows),all_original_states_reconstructed=True))
if __name__=='__main__':main()
