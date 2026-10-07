#!/usr/bin/env python3
"""Frozen one-factor and joint sensitivity matrix; no search or route rebalancing."""
import concurrent.futures as cf
import copy, csv, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
ROOT = Path('/scratch/shared/mcl123/plena')
OLD = ROOT / 'outputs/moe_output_pool_20260909'
OUT = ROOT / 'outputs/moe_bottleneck_20260911'
SOURCE = Path(__file__).resolve().parents[2]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def save(p,j):
    p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def order(result, arch):
    jobs=sorted(result['job_completions'], key=lambda x:(x['start_ps'],x['job']))
    return [[j['job'] for j in jobs if j['core']==c['id']] for c in arch['cores']]

PROFILES = {
    'baseline':{}, 'mac2':{'mac_speedup':2}, 'activation2':{'activation_speedup':2},
    'weight_port2':{'weight_port_speedup':2}, 'accumulator2':{'accumulator_speedup':2},
    'scheduler2':{'scheduler_speedup':2}, 'dma2':{'dma_speedup':2}, 'vector2':{'vector_speedup':2},
    'hbm_column2':{'hbm_profile':'column_x2'}, 'hbm_return2':{'hbm_profile':'return_latency_x2'},
    'hbm_column_return2':{'hbm_profile':'column_and_return_x2'}, 'hbm_all_timing2':{'hbm_profile':'all_timing_x2'},
    'hbm_clock2':{'hbm_profile':'clock_x2'}, 'hbm_clock_dma2':{'hbm_profile':'clock_x2','dma_speedup':2},
    'mac_activation2':{'mac_speedup':2,'activation_speedup':2},
    'supply2':{'hbm_profile':'all_timing_x2', 'activation_speedup':2,'weight_port_speedup':2,
               'accumulator_speedup':2,'dma_speedup':2,'vector_speedup':2,'scheduler_speedup':2},
    'all2':{'hbm_profile':'all_timing_x2', 'activation_speedup':2,'weight_port_speedup':2,
               'accumulator_speedup':2,'dma_speedup':2,'vector_speedup':2,'scheduler_speedup':2,'mac_speedup':2},
    'ideal_hbm':{'ideal_hbm':True},
    'ideal_supply64':{'ideal_hbm':True,'activation_speedup':64,'weight_port_speedup':64,
               'accumulator_speedup':64,'dma_speedup':64,'vector_speedup':64,'scheduler_speedup':64},
    'ideal_supply64_mac2':{'ideal_hbm':True,'activation_speedup':64,'weight_port_speedup':64,
               'accumulator_speedup':64,'dma_speedup':64,'vector_speedup':64,'scheduler_speedup':64,'mac_speedup':2},
}

def prepare():
    OUT.mkdir(exist_ok=False)
    repro=OUT/'repro';repro.mkdir()
    shutil.copy2('/tmp/plena-moe-dual-core-target/release/moe_dual_normal',repro/'moe_dual_normal')
    shutil.copy2(OLD/'repro_01/libramulator.so',repro/'libramulator.so')
    shutil.copy2(__file__,repro/'runner.py')
    for name in ('tests','native-tests','release'):
        shutil.copy2('/tmp/plena-bottleneck-'+name+'.log',repro/(name+'.log'))
    patch=subprocess.check_output(['git','diff','--binary'],cwd=SOURCE)
    (repro/'source.patch').write_bytes(patch)
    p=json.loads((OLD/'prepared/prepared.json').read_text())
    points=list(csv.DictReader((OLD/'final_report/points.csv').open()))
    cases=[]
    for phase in ('A','WC'):
        for w in p['phases'][phase]['windows']:
            if phase=='A' and w['name'] not in ('expert_me1','expert_me8','expert_me32'):continue
            for a in p['phases'][phase]['architectures']:
                wanted=[('single','legacy_n3')] if phase=='A' else [('single','legacy_n3'),('single','pool_q32'),('heterogeneous','legacy_n2'),('heterogeneous','pool_q32')]
                if (a['organization'],a['mode']) not in wanted:continue
                assert sha(a['path'])==a['sha256']
                for k in ('workload','golden'):assert sha(w[k]['path'])==w[k]['sha256']
                arch=json.loads(Path(a['path']).read_text())
                point=next(x for x in points if x['phase']==phase and x['window']==w['name'] and x['organization']==a['organization'] and x['mode']==a['mode'])
                comparison=json.loads(Path(point['comparison']).read_text())
                prior=next(x['result'] for x in comparison['comparisons'] if x['architecture']['name']==arch['name'])
                cid=f"{w['name']}_{a['organization']}_{a['mode']}"
                folder=OUT/cid;folder.mkdir()
                case={'id':cid,'phase':phase,'window':w,'organization':a['organization'],'mode':a['mode'],
                      'original_architecture':a,'prior_total_ps':prior['total_ps'],'fixed_job_order':order(prior,arch),
                      'prior_comparison':point['comparison'],'prior_comparison_sha256':sha(point['comparison'])}
                save(folder/'prior_result.json',prior)
                for profile,knobs in PROFILES.items():
                    config=copy.deepcopy(arch)
                    config['diagnostic']={**knobs,'fixed_job_order':case['fixed_job_order']}
                    save(folder/(profile+'.arch.json'),config)
                cases.append(case)
    manifest={'status':'prepared','cases':cases,'profiles':PROFILES,'repeats':2,'workers':2,'planned_runs':len(cases)*len(PROFILES)*2,
        'scope':'fixed numerical MoE operators; archived routes, synthetic full expert bank; fixed per-core job order; counterfactual service, not equal-cost hardware speedups',
        'source':str(SOURCE),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'binary_sha256':sha(repro/'moe_dual_normal'),'library_sha256':sha(repro/'libramulator.so'),
        'source_patch_sha256':sha(repro/'source.patch'),'runner_sha256':sha(__file__),
        'bank':p['bank'],'prepared_sha256':sha(OLD/'prepared/prepared.json')}
    save(OUT/'manifest.json',manifest)
    return manifest

def run_point(case,profile,repeat,manifest):
    folder=OUT/case['id'];arch=folder/(profile+'.arch.json');out=folder/f'{profile}.rep{repeat}.json'
    log=folder/f'{profile}.rep{repeat}.log';w=case['window']
    cmd=[str(OUT/'repro/moe_dual_normal'),'--workload',w['workload']['path'],'--architecture',str(arch),
         '--output',str(out),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
    started=time.monotonic()
    if not out.exists():
        with log.open('w') as f:
            subprocess.run(cmd,env={**os.environ,'LD_LIBRARY_PATH':str(OUT/'repro')},stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=True)
    envelope=json.loads(out.read_text());r=envelope['result'];p=envelope['provenance']
    golden=json.loads(Path(w['golden']['path']).read_text())
    for field in ('output_bf16','output_f32','pre_round_output_f32'):assert r[field]==golden[field],(case['id'],profile,field)
    assert p['executable_sha256']==manifest['binary_sha256']
    assert p['native_library_sha256']==manifest['library_sha256']
    assert p['architecture_sha256']==sha(arch)
    assert p['workload_sha256']==w['workload']['sha256']
    assert p['hbm_sha256']==manifest['bank']['image']['sha256']
    native=envelope['memory_model']['calibration']
    if not PROFILES[profile].get('ideal_hbm'):assert native['native_pending']==0
    config=json.loads(arch.read_text())
    assert order(r,config)==case['fixed_job_order'],(case['id'],profile,'ownership/order')
    prior=json.loads((folder/'prior_result.json').read_text())
    for field in ('useful_macs','issued_macs'):assert r[field]==prior[field],(case['id'],profile,field)
    if profile=='baseline':assert r['total_ps']==case['prior_total_ps'],(case['id'],'default regression',r['total_ps'],case['prior_total_ps'])
    if repeat==2:
        first=json.loads((folder/f'{profile}.rep1.json').read_text())
        assert r==first['result'],(case['id'],profile,'repeat changed')
        assert native==first['memory_model']['calibration'],(case['id'],profile,'native repeat changed')
    return {'case':case['id'],'window':w['name'],'organization':case['organization'],'mode':case['mode'],'profile':profile,'repeat':repeat,
        'total_ps':r['total_ps'],'speedup':case['prior_total_ps']/r['total_ps'],'useful_macs':r['useful_macs'],'issued_macs':r['issued_macs'],
        'hbm_read_bytes':r['hbm_read_bytes'],'hbm_bytes_changed':r['hbm_read_bytes']!=prior['hbm_read_bytes'],
        'golden_exact':True,'fixed_job_order':True,'wall_seconds':round(time.monotonic()-started,2),'result':str(out)}

def main():
    if (OUT/'manifest.json').exists():
        manifest=json.loads((OUT/'manifest.json').read_text())
        assert sha(OUT/'repro/moe_dual_normal')==manifest['binary_sha256']
        assert sha(OUT/'repro/libramulator.so')==manifest['library_sha256']
        manifest.setdefault('initial_runner_sha256',manifest['runner_sha256'])
        manifest.update(workers=8,runner_sha256=sha(__file__),
            execution_amendment='Resume atomic result checkpoints with 8 workers; host has 40 CPUs and >200 GiB available. Same frozen cases, inputs and simulator binary; revalidate every existing result.')
        shutil.copy2(__file__,OUT/'repro/runner_resumed.py')
        save(OUT/'manifest.json',manifest)
    else:
        manifest=prepare()
    print(f"PREPARED {manifest['planned_runs']} runs at {OUT}",flush=True)
    rows=[];errors=[]
    # Gates: defaults must reproduce frozen timings before interventions start.
    stages=[['baseline'],[p for p in PROFILES if p!='baseline']]
    for profiles in stages:
        for repeat in (1,2):
            with cf.ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
                futures={pool.submit(run_point,c,p,repeat,manifest):(c['id'],p,repeat) for c in manifest['cases'] for p in profiles}
                for future in cf.as_completed(futures):
                    key=futures[future]
                    try:rows.append(future.result())
                    except Exception as exc:
                        errors.append({'point':key,'error':repr(exc)})
                        print('FAIL '+str(errors[-1]),flush=True)
                    if (len(rows)+len(errors))%10==0 or errors:
                        print(f"PROGRESS pass={len(rows)} fail={len(errors)} / {manifest['planned_runs']}",flush=True)
                    save(OUT/'progress.json',{'passed':len(rows),'failed':errors,'planned':manifest['planned_runs']})
            if errors and profiles==['baseline']:break
        if errors and profiles==['baseline']:break
    if rows:
        with (OUT/'measurements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(sorted(rows,key=lambda x:(x['case'],x['profile'],x['repeat'])))
    manifest.update(status='passed' if not errors and len(rows)==manifest['planned_runs'] else 'failed',completed_runs=len(rows),errors=errors)
    save(OUT/'manifest.json',manifest)
    print('FINISHED '+manifest['status'],flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
