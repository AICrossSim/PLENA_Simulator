#!/usr/bin/env python3
"""Follow-up: allow the existing work-conserving dispatcher to react to timing."""
import concurrent.futures as cf
import copy,csv,json,os,subprocess,hashlib
from pathlib import Path
from run import OUT,sha,save,order
ROOT=OUT/'adaptive_dispatch'
PROFILES=['hbm_clock_dma2','all2','ideal_supply64']

def one(c,profile,rep,m):
    folder=ROOT/c['id'];a=folder/(profile+'.arch.json');output=folder/f'{profile}.rep{rep}.json'
    cmd=[str(OUT/'repro/moe_dual_normal'),'--workload',c['window']['workload']['path'],'--architecture',str(a),'--output',str(output),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
    with (folder/f'{profile}.rep{rep}.log').open('w') as log:
        subprocess.run(cmd,env={**os.environ,'LD_LIBRARY_PATH':str(OUT/'repro')},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    e=json.loads(output.read_text());r=e['result'];p=e['provenance'];g=json.loads(Path(c['window']['golden']['path']).read_text())
    for field in ('output_bf16','output_f32','pre_round_output_f32'):assert r[field]==g[field]
    for field,expected in [('executable_sha256',m['binary_sha256']),('native_library_sha256',m['library_sha256']),('architecture_sha256',sha(a)),('workload_sha256',c['window']['workload']['sha256']),('hbm_sha256',m['bank']['image']['sha256'])]:assert p[field]==expected
    if profile!='ideal_supply64':assert e['memory_model']['calibration']['native_pending']==0
    if rep==2:
        first=json.loads((folder/f'{profile}.rep1.json').read_text())
        assert r==first['result'];assert e['memory_model']['calibration']==first['memory_model']['calibration']
    config=json.loads(a.read_text());actual=order(r,config)
    return {'case':c['id'],'window':c['window']['name'],'organization':c['organization'],'mode':c['mode'],'profile':profile,'repeat':rep,
        'total_us':r['total_ps']/1e6,'job_order_changed':actual!=c['fixed_job_order'],'useful_macs':r['useful_macs'],
        'issued_macs':r['issued_macs'],'hbm_read_bytes':r['hbm_read_bytes'],'numerical_exact':True,'result':str(output)}

def main():
    m=json.loads((OUT/'manifest.json').read_text());cases=[c for c in m['cases'] if c['phase']=='WC']
    ROOT.mkdir(exist_ok=False)
    for c in cases:
        folder=ROOT/c['id'];folder.mkdir()
        for profile in PROFILES:
            a=json.loads((OUT/c['id']/(profile+'.arch.json')).read_text())
            del a['diagnostic']['fixed_job_order'];assert a['dispatch_policy']=='work_conserving'
            save(folder/(profile+'.arch.json'),a)
    manifest={'status':'prepared','planned_runs':len(cases)*len(PROFILES)*2,'profiles':PROFILES,'cases':cases,'workers':4,
        'binary_sha256':m['binary_sha256'],'library_sha256':m['library_sha256'],'runner_sha256':sha(__file__),
        'purpose':'Secondary check of timing-dependent placement using the existing work-conserving dispatcher; not optimal assignment search.'}
    save(ROOT/'manifest.json',manifest);rows=[];errors=[]
    for rep in (1,2):
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(one,c,p,rep,m):(c['id'],p,rep) for c in cases for p in PROFILES}
            for f in cf.as_completed(jobs):
                try:rows.append(f.result())
                except Exception as exc:errors.append({'point':jobs[f],'error':repr(exc)})
                print('ADAPTIVE',len(rows),'passed',errors,'failed',flush=True)
    with (ROOT/'measurements.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest.update(status='passed' if not errors else 'failed',completed_runs=len(rows),errors=errors)
    save(ROOT/'manifest.json',manifest)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
