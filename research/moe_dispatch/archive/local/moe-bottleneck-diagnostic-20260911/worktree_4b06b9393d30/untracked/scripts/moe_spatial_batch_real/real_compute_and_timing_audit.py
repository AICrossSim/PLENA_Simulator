#!/usr/bin/env python3
"""Pure compute for the exact captured X/routes; audit value/timing independence."""
import csv,gzip,json,subprocess,tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import run_routes as r

DEST=r.OUT/'real_compute'
def execute(p):
    req=json.loads(p.read_text());req['compute']['verify_values']=False
    req['compute']['record_trace']=False
    name=p.parent.name+'__'+p.name.split('.')[0]
    q=DEST/(name+'.request.json');r.save(q,req['compute'])
    hashes=[]
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix='plena-real-pure-',dir='/tmp') as tmp:
            o=Path(tmp)/'out.json';proc=subprocess.run([str(r.PURE),'--request',str(q),'--output',str(o)],capture_output=True,timeout=300)
            assert proc.returncode==0,proc.stderr
            data=o.read_bytes();rep=json.loads(data)
        assert rep['requests_drained']
        hashes.append(r.digest(data))
        if repeat==0:(DEST/(name+'.json.gz')).write_bytes(gzip.compress(data,mtime=0))
        else:assert hashes[0]==hashes[1]
    numeric=json.loads(gzip.decompress(p.with_name(p.name.replace('.request.json','.report.json.gz')).read_bytes()))
    # Every actual-value phase also replayed once on the frozen finite engine,
    # without values. The complete event hashes must be identical.
    fq=DEST/(name+'.finite_request.json');r.save(fq,req)
    with tempfile.TemporaryDirectory(prefix='plena-real-timing-',dir='/tmp') as tmp:
        o=Path(tmp)/'out.json';proc=subprocess.run([str(r.FAB),'--request',str(fq),'--output',str(o)],capture_output=True,timeout=300)
        assert proc.returncode==0,proc.stderr
        f=json.loads(o.read_text())
    assert f['total_cycles']==numeric['total_cycles'] and f['stats']==numeric['stats']
    assert f['invocation_sha256']==numeric['invocation_sha256'] and f['service_sha256']==numeric['service_sha256']
    r.helper.audit(req,f)
    record=dict(name=name,batch=int(p.parent.name.split('__')[0].removeprefix('real_b')),
        shape=p.parent.name.split('__')[1].replace('_','+'),mode=p.parent.name.split('__')[2],phase=p.name.split('.')[0],
        pure_cycles=rep['total_cycles'],finite_cycles=f['total_cycles'],useful_macs=rep['useful_macs'],
        pure_issued_mac_slots=rep['issued_mac_slots'],pure_invocations=rep['total_invocations'],
        pure_repeat_sha256=hashes,finite_value_independent_timeline=True)
    r.save(DEST/(name+'.validation.json'),record);return record
def main():
    DEST.mkdir(exist_ok=True)
    paths=sorted((r.OUT/'real_campaign').glob('real_b*/*.request.json'));assert len(paths)==240
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(execute,p) for p in paths]):
            rows.append(f.result());r.save(DEST/'progress.json',dict(done=len(rows),planned=len(paths)))
    keys=[k for k in rows[0] if k!='pure_repeat_sha256']
    with (DEST/'all_phases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    r.save(DEST/'validation.json',dict(passed=True,points=len(rows),pure_runs=2*len(rows),frozen_finite_replays=len(rows),
        every_real_value_event_hash_matches_frozen_timing=True))
if __name__=='__main__':main()
