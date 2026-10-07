#!/usr/bin/env python3
"""Audited, resumable, repeat-twice batch campaign; never substitutes estimates."""
import argparse, csv, gzip, hashlib, importlib.util, json, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path('/scratch/shared/mcl123/plena')
OUT=ROOT/'outputs/moe_spatial_batch_real_20260921'
WORK=ROOT/'review_20260921/simulator-moe-batch-real'
spec=importlib.util.spec_from_file_location('fabric_audit',WORK/'scripts/moe_spatial_fabric/run_study.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
FAB=ROOT/'outputs/moe_spatial_fabric_20260919/ed6d56ee135b4efaac258eb195b977c6/repro/moe_spatial_fabric_v3'
PURE=Path('/tmp/plena-moe-dual-core-target/release/moe_spatial_m')
SHAPES=[[6],[3,3],[4,2],[2,2,2],[1]*6]
MODES=['pinned_expert','tile_stealing']
def digest(x):return hashlib.sha256(x).hexdigest()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def jobs(w,phase):
    shared=phase.startswith('shared')
    inter=w['shared_intermediate'] if shared else w['intermediate']
    counts={10000:w['batch']} if shared else w['expert_counts']
    n,k=(w['hidden'],inter) if phase.endswith('down') else (inter,w['hidden'])
    return [dict(expert=int(e),m=m,n=n,k=k,seed=13) for e,m in sorted(counts.items(),key=lambda x:int(x[0]))]

def tasks(windows):
    seen_shared=set()
    for w in windows:
        phases=['routed_gate_up','routed_down']
        shared_key=(w['model'],w['batch'])
        if shared_key not in seen_shared:
            phases+=['shared_gate_up','shared_down'];seen_shared.add(shared_key)
        for phase in phases:
            for shape in SHAPES:
                for mode in MODES:
                    variants=[('pure',{}),('finite',{})]
                    if w['dataset']=='swe' and w['split']=='primary' and shape in SHAPES[:3] and not phase.startswith('shared'):
                        variants += [('oracle_zero_control',{'zero_control_time':True}),
                            ('oracle_zero_weight',{'zero_weight_time':True}),
                            ('oracle_both',{'zero_control_time':True,'zero_weight_time':True})]
                    if w['dataset']=='bfcl' and w['split']=='primary' and phase=='routed_gate_up' and mode=='pinned_expert':
                        variants += [('control2',{'control_ports':2}),('weight4x',{'weight_bpc':4096}),
                            ('activation4x',{'activation_bpc':24576}),('accumulator4x',{'accumulator_bpc':768})]
                    for variant,cfg in variants:
                        name=f"{w['name']}__{phase}__{'_'.join(map(str,shape))}__{mode}__{variant}"
                        req=helper.req(name,shape,jobs(w,phase),mode,dict(control='tile_cohort',**cfg))
                        yield dict(name=name,window=w['name'],model=w['model'],dataset=w['dataset'],split=w['split'],
                            batch=w['batch'],phase=phase,shape='+'.join(map(str,shape)),mode=mode,variant=variant,request=req)

def execute(t):
    name=t['name']; dst=OUT/'route_campaign';req=t['request'];pure=t['variant']=='pure'
    b=PURE if pure else FAB; payload=req['compute'] if pure else req
    rawreq=json.dumps(payload,separators=(',',':')).encode(); rh=digest(rawreq)
    receipt=dst/'receipts'/f'{name}.json';report=dst/'reports'/f'{name}.json.gz'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['request_sha256']==rh and old['binary_sha256']==BINARY_SHA[str(b)]
        assert digest(gzip.decompress(report.read_bytes()))==old['repeat_sha256'][0]
        return old['row']
    (dst/'requests'/f'{name}.json').write_bytes(rawreq)
    hashes=[];start=time.monotonic()
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix='plena-batch-',dir='/tmp') as tmp:
            rp=Path(tmp)/'result.json'
            process=subprocess.run([str(b),'--request',str(dst/'requests'/f'{name}.json'),'--output',str(rp)],
                capture_output=True,timeout=900)
            if process.returncode:raise RuntimeError(f'{name}: {process.stderr.decode()[-3000:]}')
            data=rp.read_bytes();rep=json.loads(data)
        assert rep['useful_macs']==sum(j['m']*j['n']*j['k'] for j in req['compute']['jobs'])
        assert rep['total_multipliers']==12288 and rep['numerical_bit_exact'] is None
        if pure:
            assert rep['requests_drained']
            assert rep['total_invocations']==sum(c['invocations'] for c in rep['cores'])
        else:helper.audit(req,rep)
        hashes.append(digest(data))
        if repeat==0:report.write_bytes(gzip.compress(data,mtime=0))
        else:assert hashes[0]==hashes[1],name
    row={k:v for k,v in t.items() if k!='request'}
    row.update(cycles=rep['total_cycles'],useful_macs=rep['useful_macs'],issued_mac_slots=rep['issued_mac_slots'],
        invocations=rep.get('invocation_audit_count',rep.get('total_invocations')),
        utilization=rep['useful_macs']/(12288*rep['total_cycles']),host_wall_seconds=time.monotonic()-start,
        jobs=len(req['compute']['jobs']),token_expert_rows=sum(j['m'] for j in req['compute']['jobs']))
    if not pure:row.update(rep['stats'])
    save(receipt,dict(row=row,request_sha256=rh,binary_sha256=BINARY_SHA[str(b)],repeat_sha256=hashes,
        source='route_windows.json',numerical_scope='timing-only: true routing and full dimensions, no X/W values'))
    return row

BINARY_SHA={str(p):digest(p.read_bytes()) for p in [FAB,PURE]}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit',type=int)
    args=parser.parse_args();dst=OUT/'route_campaign'
    for d in ['requests','reports','receipts']:(dst/d).mkdir(parents=True,exist_ok=True)
    windows=json.loads((OUT/'inputs/route_windows.json').read_text());plan=list(tasks(windows))
    if args.limit:plan=plan[:args.limit]
    save(dst/'plan.json',dict(points=len(plan),repeats=2,binaries=BINARY_SHA,windows=72,
        gate_up='identical shape timing measured once; each gate and up must execute for layer accounting',
        full_model=False,native_hbm=False,selection='first 16 valid streams at step0/layer0; disjoint next16 at step8/middle layer'))
    rows=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,t):t['name'] for t in plan}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as e:
                save(dst/'FAILURE.json',dict(name=futures[future],error=str(e),completed=len(rows)))
                for f in futures:f.cancel()
                raise
            rows.append(row)
            save(dst/'progress.json',dict(completed=len(rows),planned=len(plan),elapsed_s=time.monotonic()-started,last=row['name']))
            if len(rows)%20==0:print(f'{len(rows)}/{len(plan)} PASS; {time.monotonic()-started:.1f} s',flush=True)
    rows.sort(key=lambda x:x['name']);fields=sorted(set().union(*(r.keys() for r in rows)))
    with (dst/'all_points.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save(dst/'validation.json',dict(passed=True,points=len(rows),runs=2*len(rows),all_repeats_identical=True,
        route_inventory_pairs=29495540,timing_windows=72,numerical_values_tested=False,native_hbm=False))
    print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
