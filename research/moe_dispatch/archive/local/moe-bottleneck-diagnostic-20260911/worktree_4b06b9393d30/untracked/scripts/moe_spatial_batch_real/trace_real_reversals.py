#!/usr/bin/env python3
"""Targeted B8/B16 actual-input timing reversals, preserving numeric event hashes."""
import csv,gzip,json,subprocess,tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from run_breakdown import classify
import run_routes as r

DEST=r.OUT/'real_breakdown'
def execute(batch,shape,phase):
    name=f"real_b{batch}__{'_'.join(map(str,shape))}__pinned_expert"
    source=r.OUT/'real_campaign'/name
    req=json.loads((source/(phase+'.request.json')).read_text())
    numeric=json.loads(gzip.decompress((source/(phase+'.report.json.gz')).read_bytes()))
    req['compute'].update(verify_values=False,record_trace=True)
    folder=DEST/(name+'__'+phase);folder.mkdir(exist_ok=True)
    r.save(folder/'request.json',req);hashes=[]
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix='plena-real-trace-',dir='/tmp') as tmp:
            p=Path(tmp)/'out.json';q=subprocess.run([str(r.FAB),'--request',str(folder/'request.json'),'--output',str(p)],capture_output=True,timeout=900)
            assert q.returncode==0,q.stderr
            data=p.read_bytes();rep=json.loads(data)
        hashes.append(r.digest(data))
        if repeat==0:(folder/'trace.json.gz').write_bytes(gzip.compress(data,mtime=0))
        else:assert hashes[0]==hashes[1]
    r.helper.audit(req,rep)
    assert rep['total_cycles']==numeric['total_cycles']
    assert rep['invocation_sha256']==numeric['invocation_sha256'] and rep['service_sha256']==numeric['service_sha256']
    if phase.endswith('gate'):
        up=json.loads(gzip.decompress((source/(phase.replace('gate','up')+'.report.json.gz')).read_bytes()))
        assert up['invocation_sha256']==numeric['invocation_sha256'] and up['service_sha256']==numeric['service_sha256']
    cores=classify(req,rep)
    last=max(cores,key=lambda x:x['last_commit'])
    row=dict(batch=batch,shape='+'.join(map(str,shape)),mode='pinned_expert',phase=phase,
        multiplicity=2 if phase.endswith('gate') else 1,**last)
    r.save(folder/'validation.json',dict(passed=True,repeat_sha256=hashes,same_numeric_timeline=True,cores=cores,last=row))
    print(name,phase,'PASS',flush=True);return row
def main():
    DEST.mkdir(exist_ok=True)
    plan=[(b,s,p) for b in [8,16] for s in [[6],[3,3],[4,2]] for p in ['routed_gate','routed_down','shared_gate','shared_down']]
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(execute,*p) for p in plan]):
            rows.append(f.result());r.save(DEST/'progress.json',dict(done=len(rows),planned=len(plan)))
    with (DEST/'last_core.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    r.save(DEST/'validation.json',dict(passed=True,points=len(rows),runs=2*len(rows),real_numeric_timeline_matched=True))
if __name__=='__main__':main()
