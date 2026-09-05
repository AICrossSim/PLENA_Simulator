#!/usr/bin/env python3
"""Bounded exhaustive joint MoE DSE. Every feasible representative runs numerically.

Capacity-only variants are collapsed only when per-job/core admission masks are
identical on ALL declared search fixtures. No latency model prunes candidates.
All reports are retained losslessly compressed. Failures block final ranking.
"""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from compare_moe_normal import digest, read_json, require, validate_run
from run_moe_dma_campaign import save

BLENS = [2, 4, 8, 16, 32, 64]
MLENS = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
FIXTURES = ['qwen_full_decode_b8', 'qwen_full_decode_b32',
            'deepseek_full_decode_b8', 'deepseek_full_decode_b32']
BUDGET = dict(multipliers=4096, vector_sram_bytes=4194304, accumulator_bytes=1048576,
              weight_sram_bytes=65536, activation_elements_per_cycle=1024,
              frontend_sram_bytes=45056)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def shapes():
    possible = [(b,k) for b,k in itertools.product(BLENS, MLENS) if k % b == 0 and b*k <= 4096]
    singles = [(s,) for s in possible if s[0]*s[1] == 4096]
    pairs = [tuple(sorted((a,b), key=lambda s: (-s[0]*s[1], -s[0], s[1])))
             for a,b in itertools.combinations_with_replacement(possible, 2)
             if a[0]*a[1]+b[0]*b[1] == 4096]
    return singles + sorted(set(pairs))


def category(shape):
    if len(shape) == 1: return 'single'
    if shape[0] == shape[1]: return 'homogeneous'
    if shape[0][0]*shape[0][1] == shape[1][0]*shape[1][1]: return 'equal_pe_different_shape'
    return 'large_small'


def job_rows(w):
    rows = {}
    for route in w['routes']:
        rows[route['expert']] = rows.get(route['expert'], 0)+1
    result = sorted(rows.items())
    if w.get('shared_expert') is not None:
        result.append(('shared', len(w['inputs_bf16'])))
    return result


def admission(a, w):
    return [[2*m*(2*w['input_dim']+3*w['expert_hidden_dim']) <= c['vector_sram_bytes']
             and 4*m*max(w['input_dim'], w['expert_hidden_dim'])
             + 4*c['blen']*((c['mlen']-1).bit_length()+a['mac_pipeline_cycles']) <= c['accumulator_bytes']
             for _,m in job_rows(w)] for c in a['cores']]


def frontend_bytes(a):
    return 120*a['global_dma_credits']+8192+sum(c['weight_slots']*(128+24*c['blen']*
        ((c['mlen']+63)//64+((c['mlen']+7)//8+63)//64+2)) for c in a['cores'])


def architecture(shape, slots, credits, threshold, allocation):
    count=len(shape)
    weights = [b*k for b,k in shape]
    if allocation == 'equal': weights=[1]*count
    if allocation == 'inverse_pe': weights=list(reversed(weights))
    a=dict(schema_version=1, name='pending', clock_period_ps=1000, mac_pipeline_cycles=16,
        combine_sram_bytes=4194304, vector_elements_per_cycle=512, dispatch_queue_bytes=16384,
        dispatch_cycles=1, dispatch_policy='work_conserving', dispatch_threshold=threshold,
        global_dma_credits=credits, global_dma_staging_bytes=64*credits,
        large_core=0, small_core=count-1, cores=[],
        dma=dict(issue_policy='per_channel', sector_reads=True, coalesce=True, fair_credits=False,
                 frontend_sram_bytes=45056, lookup_ii_cycles=2))
    for i,(b,k) in enumerate(shape):
        c=dict(id='core'+str(i), blen=b, mlen=k, read_cache_bytes=0, weight_slots=slots,
               activation_elements_per_cycle=1024*b*k//4096,
               # Weights are allocated in 4KiB units proportional to PE, covering all slots.
               weight_sram_bytes=65536*b*k//4096)
        for field in ['vector_sram_bytes','accumulator_bytes']:
            c[field]=BUDGET[field]*weights[i]//sum(weights)
        a['cores'].append(c)
    return a


def resource_gate(a):
    for key in ['vector_sram_bytes','accumulator_bytes','weight_sram_bytes','activation_elements_per_cycle']:
        require(sum(c[key] for c in a['cores']) == BUDGET[key], 'unequal '+key)
    require(sum(c['blen']*c['mlen'] for c in a['cores']) == 4096, 'unequal PE')
    require(all(c['weight_slots']*c['blen']*(c['mlen']//8)*25 <= c['weight_sram_bytes']
                for c in a['cores']), 'weight slots exceed SRAM')
    require(frontend_bytes(a) <= 45056, 'DMA descriptor capacity exceeded')


def equivalence_key(a, masks, layout):
    reduced=copy.deepcopy(a)
    reduced.pop('name')
    for c in reduced['cores']:
        c.pop('vector_sram_bytes'); c.pop('accumulator_bytes')
    return fingerprint([reduced, masks, layout])


def prepare(root, source, layout_root, binary, library):
    require(not (root/'campaign.json').exists(), 'campaign already frozen; use run to resume')
    root.mkdir(parents=True, exist_ok=True)
    workloads={n:read_json(source/n/'workload.json') for n in FIXTURES}
    fixture_records={}
    for layout, folder in [('raw',source),('scale_odd32',layout_root)]:
        for name in FIXTURES:
            f=folder/name; w=read_json(f/'workload.json'); g=read_json(f/'golden.json')
            require(g['workload_sha256'] == digest(f/'workload.json'), 'golden manifest identity')
            require(g['hbm_sha256'] == digest(f/w['hbm_file']) == w['metadata']['hbm_sha256'], 'image identity')
            if layout != 'raw':
                require(w['metadata']['dma_layout_transform']['all_logical_row_bytes_equal'], 'layout equality absent')
                require(g['output_bf16'] == read_json(source/name/'golden.json')['output_bf16'], 'layout golden differs')
                # Recheck every logical row against the actual images, not just a stored claim.
                original=workloads[name]; old=(source/name/original['hbm_file']).read_bytes(); new=(f/w['hbm_file']).read_bytes()
                for oe, ne in zip(original['experts'],w['experts']):
                    for projection in ['gate','up','down']:
                        x,y=oe[projection],ne[projection]
                        for stream,width in [('element',x['cols']),('scale',(x['cols']+7)//8)]:
                            for row in range(x['rows']):
                                p=x[stream+'_base']+row*x[stream+'_row_stride'];q=y[stream+'_base']+row*y[stream+'_row_stride']
                                require(old[p:p+width] == new[q:q+width], 'layout changed logical weights')
            fixture_records[layout+'/'+name]=dict(path=str(f), workload_sha256=digest(f/'workload.json'),
                golden_sha256=digest(f/'golden.json'), hbm_sha256=g['hbm_sha256'])
    records=[]; representatives={}; excluded=[]
    for shape in shapes():
        for slots,credits,layout,threshold,allocation in itertools.product(
                [2,3,4],[64,128],['raw','scale_odd32'],[1,8,32] if len(shape)>1 else [8],
                ['equal','pe','inverse_pe'] if len(shape)>1 else ['equal']):
            a=architecture(shape,slots,credits,threshold,allocation)
            axes=dict(shape=shape, category=category(shape), slots=slots, credits=credits,
                      layout=layout, threshold=threshold, allocation=allocation)
            try:
                resource_gate(a)
                masks={n:admission(a,w) for n,w in workloads.items()}
                require(all(all(any(core[j] for core in mask) for j in range(len(mask[0])))
                            for mask in masks.values()), 'job fits no core')
            except ValueError as e:
                excluded.append(dict(axes=axes, reason=str(e))); continue
            key=equivalence_key(a,masks,layout)
            if key not in representatives:
                ident='p'+str(len(representatives)).zfill(4)
                a['name']=ident
                save(root/'architectures'/(ident+'.json'),a)
                representatives[key]=dict(id=ident, axes=axes, admission_masks=masks,
                    architecture_sha256=digest(root/'architectures'/(ident+'.json')))
            records.append(dict(axes=axes, representative=representatives[key]['id']))
    repro=root/'repro';repro.mkdir()
    shutil.copy2(str(binary),str(repro/'moe_dual_normal'))
    shutil.copy2(str(library),str(repro/'libramulator.so'))
    plan=dict(schema_version=1, budgets=BUDGET, blens=BLENS, mlens=MLENS,
        slots=[2,3,4], credits=[64,128], layouts=['raw','scale_odd32'], thresholds=[1,8,32],
        topologies=[dict(shape=s,category=category(s)) for s in shapes()],
        allocation_profiles=['equal','pe','inverse_pe'], fixtures=fixture_records,
        representatives=list(representatives.values()), valid_variants=records, excluded=excluded,
        expected_runs=len(representatives)*len(FIXTURES), numerical=True,
        binary_sha256=digest(repro/'moe_dual_normal'),native_sha256=digest(repro/'libramulator.so'),
        driver_sha256=digest(__file__),validator_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        equivalence_proof='Only vector/accumulator capacity values removed; same per-job/core admission masks across every declared search fixture. These capacities affect admission/validation only, not timing once a job fits. Equivalence is workload-specific, not universal.',
        exclusions_from_scope=['three or more cores','rectangular M/N tiles','unequal per-core prefetch depths',
            'activation port allocation other than PE-proportional','additional MX codecs','router/prefill/full model/RTL area'],
        selection='one fixed configuration per category minimizing geometric mean latency over the four search fixtures; each fixture weighted equally; independent holdout and timing sensitivity after selection',
        timing='native calibrated HBM2 8 channels; analytical compute, SRAM and DMA frontend timing; not RTL-calibrated',
        repeat_policy='one complete numerical run at every grid point; two additional executions for selected finalists')
    save(root/'campaign.json',plan)
    print(json.dumps(dict(topologies=len(shapes()),representatives=len(representatives),
        valid_variants=len(records),excluded=len(excluded),runs=plan['expected_runs'])),flush=True)


def load_report(path):
    with gzip.open(str(path),'rt') as f:return json.load(f)


def validate_identity(e, a, f, plan):
    p=e['provenance']
    require(p['executable_sha256'] == plan['binary_sha256'] and p['native_library_sha256'] == plan['native_sha256'], 'executable changed')
    require(p['workload_sha256'] == f['workload_sha256'] and p['hbm_sha256'] == f['hbm_sha256'], 'fixture changed')
    require(e['architecture_manifest'] == a, 'architecture changed')


def execute_point(root, plan, point, fixture, label=None, architecture_override=None, fixture_override=None):
    ident=label or point['id']; destination=root/'runs'/ident/fixture
    destination.mkdir(parents=True, exist_ok=True)
    a_path=root/'architectures'/(point['id']+'.json') if architecture_override is None else architecture_override
    a=read_json(a_path)
    f=fixture_override or plan['fixtures'][point['axes']['layout']+'/'+fixture]
    folder=Path(f['path']);w=read_json(folder/'workload.json');golden=read_json(folder/'golden.json')
    require(digest(a_path) == (point['architecture_sha256'] if architecture_override is None else digest(a_path)), 'architecture hash differs')
    require(digest(folder/'golden.json') == f['golden_sha256'], 'golden hash differs')
    status=destination/'status.json';report_path=destination/'report.json.gz'
    if status.exists():
        previous=read_json(status)
        if previous['status']=='passed' and report_path.exists() and digest(report_path)==previous['report_sha256']:
            e=load_report(report_path);validate_identity(e,a,f,plan)
            return previous
    start=time.monotonic()
    try:
        output=destination/'output.json'
        env=dict(os.environ, LD_LIBRARY_PATH=str(root/'repro'))
        with (destination/'process.log').open('w') as log:
            subprocess.run([str(root/'repro/moe_dual_normal'),'--workload',str(folder/'workload.json'),
                '--architecture',str(a_path),'--output',str(output),'--hbm-channels','8'],
                env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
        e=read_json(output);validate_identity(e,a,f,plan)
        gate=validate_run(e,golden,w,a,1e-6,0.01,8)
        require(gate['output_bit_exact'], 'DSE requires bit-exact BF16 golden')
        with output.open('rb') as source,gzip.open(str(report_path),'wb',compresslevel=1) as target:
            shutil.copyfileobj(source,target)
        output.unlink()
        result=dict(status='passed',point=ident,fixture=fixture,wall_seconds=time.monotonic()-start,
            total_ps=e['result']['total_ps'],hbm_read_bytes=e['result']['hbm_read_bytes'],
            numerical_gate=gate,report_sha256=digest(report_path))
    except Exception as e:
        result=dict(status='failed',point=ident,fixture=fixture,error=str(e),wall_seconds=time.monotonic()-start)
    save(status,result)
    return result


def run(root,workers):
    plan=read_json(root/'campaign.json')
    require(digest(root/'repro/moe_dual_normal')==plan['binary_sha256'], 'binary changed')
    require(digest(root/'repro/libramulator.so')==plan['native_sha256'], 'native changed')
    tasks=[(p,f) for p in plan['representatives'] for f in FIXTURES]
    outcomes=[];start=time.monotonic();last=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute_point,root,plan,p,f):(p,f) for p,f in tasks}
        for future in as_completed(futures):
            p,f=futures[future]
            try: value=future.result()
            except Exception as e:value=dict(status='failed',point=p['id'],fixture=f,error=str(e))
            outcomes.append(value)
            if time.monotonic()-last>5 or len(outcomes)==len(tasks):
                last=time.monotonic()
                status=dict(status='running',completed=len(outcomes),expected=len(tasks),
                    passed=sum(o['status']=='passed' for o in outcomes),
                    failed=[o for o in outcomes if o['status']!='passed'],wall_seconds=last-start)
                save(root/'status.json',status)
                print(json.dumps(status),flush=True)
    passed=len(outcomes)==len(tasks) and all(o['status']=='passed' for o in outcomes)
    save(root/'status.json',dict(status='passed' if passed else 'failed',completed=len(outcomes),
        expected=len(tasks),passed=sum(o['status']=='passed' for o in outcomes),
        failed=[o for o in outcomes if o['status']!='passed'],wall_seconds=time.monotonic()-start))
    require(passed,'incomplete/failed grid: do not rank')
    rows=[]
    for p in plan['representatives']:
        measurements={f:read_json(root/'runs'/p['id']/f/'status.json') for f in FIXTURES}
        times=[measurements[f]['total_ps'] for f in FIXTURES]
        rows.append(dict(point=p,measurements=measurements,geomean_ps=math.exp(sum(math.log(t) for t in times)/len(times))))
    rows.sort(key=lambda r:(r['geomean_ps'],r['point']['id']))
    winners={c:next(r for r in rows if r['point']['axes']['category']==c) for c in sorted(set(r['point']['axes']['category'] for r in rows))}
    save(root/'ranking.json',dict(status='complete_grid',winners=winners,ranked=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--layout-root',type=Path);p.add_argument('--binary',type=Path);p.add_argument('--library',type=Path)
    p.add_argument('--workers',type=int,default=32);args=p.parse_args()
    require(1<=args.workers<=40,'workers must be 1..40')
    if args.action=='prepare':prepare(args.root.resolve(),args.source.resolve(),args.layout_root.resolve(),args.binary.resolve(),args.library.resolve())
    else:run(args.root.resolve(),args.workers)
