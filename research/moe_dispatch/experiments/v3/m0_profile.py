#!/usr/bin/env python3
"""Frozen Joint M0 observations and declared timing interventions, two repeats."""
import argparse, concurrent.futures, copy, csv, hashlib, json, os, subprocess
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
from frontend import compiler

def run_one(job):
    root,binary,key,w,cfg=job
    target=root/key;target.mkdir(parents=True,exist_ok=True)
    (target/'workload.json').write_text(json.dumps(w,sort_keys=True))
    (target/'config.json').write_text(json.dumps(cfg,sort_keys=True))
    for repeat in range(2):
        out=target/f'repeat{repeat}.json'
        if not out.exists():
            x=subprocess.run([str(binary),str(target/'workload.json'),str(target/'config.json'),str(out)],capture_output=True,text=True)
            if x.returncode:raise RuntimeError(key+':'+x.stderr[-4000:])
    assert (target/'repeat0.json').read_bytes()==(target/'repeat1.json').read_bytes(),key
    r=json.loads((target/'repeat0.json').read_text())
    p=r.get('m0_profile',{})
    if p:assert p['mutually_exclusive'],key
    return dict(key=key,workload=w['id'],batch=w['batch'],lanes='+'.join(map(str,cfg['lanes'])),
        condition=key.split('/')[-1],cycles=r['cycles'],weight_bytes=r['weight_bytes'],
        issues=sum(c['stats']['issues'] for c in r['cores']),
        useful_macs=r['useful_macs'],effective_weight_gbps=r['weight_bytes']/r['cycles'],
        wire_bytes=r.get('compression_oracle',{}).get('scaled_wire_bytes',r['weight_bytes']),
        drained=r['drained'],duplicate_equal=True,profile_exclusive=p.get('mutually_exclusive'),
        raw_sha256=hashlib.sha256((target/'repeat0.json').read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--legacy',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    ws=json.loads(Path('/tmp/plena-joint-runtime-20260930/inputs/test_workloads.json').read_text())['workloads']
    designs=json.loads(Path('/scratch/shared/mcl123/plena/outputs/moe_robust_fixed_20260930/frozen_designs.json').read_text())['designs']
    jobs=[]
    for w0 in ws:
        if w0['batch'] not in (2,16):continue
        for lanes in ([6],[3,3],[4,2]):
            d=next(x for x in designs if x['lanes']==lanes);res=copy.deepcopy(d['resources'])
            res.update(joint_state_bytes=256,control_bytes=[4352//len(lanes)]*len(lanes))
            w=copy.deepcopy(w0);w.pop('engine_layout',None);w['experts'].sort(key=lambda e:(not e['is_shared'],e['id']))
            w['engine_layout']=compiler.engine_layout(w,lanes,d['group'],res)
            base=dict(lanes=lanes,group=d['group'],dispatch='joint',runtime_fsm=True,next_prefetch=True,
                split='none',arbiter='stock',credits=256,hbm_bytes_per_ns=256,hbm_latency_ns=64,
                control_cost=True,window=8,record_trace=False)
            opts={'frozen_old':{},'profile_off':{'arch':'joint_v1'},'profile_on':{'diagnostic_profile':True},
                'ideal_hbm':{'diagnostic_profile':True,'ideal_hbm':True},
                'ideal_onchip':{'diagnostic_profile':True,'ideal_onchip':True},
                'both_ideal':{'diagnostic_profile':True,'ideal_hbm':True,'ideal_onchip':True},
                'weight_scale':{'diagnostic_profile':True,'weight_bytes_scale':1/3.481},
                'credits544':{'diagnostic_profile':True,'credits':544,'diagnostic_credit_expansion':True}}
            for name,opt in opts.items():
                key=w['id']+'/'+'+'.join(map(str,lanes))+'/'+name
                jobs.append((a.root,a.legacy if name=='frozen_old' else a.binary,key,w,{**base,**opt}))
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(run_one,jobs):
            rows.append(row);print(row['key'],row['cycles'],flush=True)
    for w in {r['workload'] for r in rows}:
        for org in ('6','3+3','4+2'):
            rs={r['condition']:r for r in rows if r['workload']==w and r['lanes']==org}
            paths={x:a.root/rs[x]['key']/'repeat0.json' for x in ('frozen_old','profile_off','profile_on')}
            old=json.loads(paths['frozen_old'].read_text())
            for x in ('profile_off','profile_on'):
                new=json.loads(paths[x].read_text())
                # All existing result fields including every per-core counter,
                # trace and timing value; config gains selector/observer defaults.
                for k,v in old.items():
                    if k!='config':assert new[k]==v,(w,org,x,k)
    with (a.root/'m0_results.csv').open('w') as f:
        cw=csv.DictWriter(f,fieldnames=list(rows[0]));cw.writeheader();cw.writerows(rows)
    (a.root/'m0_receipt.json').write_text(json.dumps(dict(points=len(rows),runs=2*len(rows),
        frozen_timing_identical=True,all_duplicate_equal=True,all_states_exclusive=True),indent=2))
if __name__=='__main__':main()
