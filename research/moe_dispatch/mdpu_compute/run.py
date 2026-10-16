"""Development-only exhaustive geometry selection and frozen heldout replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from collections import Counter, defaultdict

from .domain import domains, THRESHOLDS, label, core_from_dict, validate
from research.moe_dispatch.plena65536_moe.workloads import load_workloads, profile_window, BATCHES
from research.moe_dispatch.plena65536_moe.model import job_cycles, useful_macs, unique_weight_bytes
from research.moe_dispatch.plena65536_shared.timing import PROFILES, resource_ledger

DEFAULT_OUTPUT = Path('/home/mcl123/plena-outputs/mdpu_compute_20261013')
_WINDOWS = None


def gm(xs):
    xs = list(xs)
    return math.exp(math.fsum(math.log(float(x)) for x in xs)/len(xs))


def balanced(windows, xs):
    return gm(gm(x for w,x in zip(windows,xs) if int(w['batch']) == b) for b in BATCHES)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def init_worker(windows):
    global _WINDOWS
    _WINDOWS = windows


def simulate(cs, w, policy='mdpu', threshold=2, **kw):
    from .runtime import schedule
    if policy == 'mdpu_guarded':
        policy='mdpu'
        kw['wait_best']=True
    return schedule(cs, w, policy=policy, threshold=threshold, **kw)


def evaluate_candidate(task):
    family, index, cs, policy, threshold = task
    vals = [int(simulate(cs,w,policy,threshold,details=False)['cycles']) for w in _WINDOWS]
    return family,index,policy,threshold,vals,balanced(_WINDOWS,vals)


FAMILIES = {
    'original': ('original','greedy'),
    'compact_single': ('compact_single','greedy'),
    'compact_homo_greedy': ('compact_homo','greedy'),
    'compact_homo_mdpu': ('compact_homo','mdpu'),
    'hetero_mdpu': ('hetero','mdpu'),
    'hetero_hard': ('hetero','hard_threshold'),
    'fair_single': ('fair_single','greedy'),
    'fair_homo_greedy': ('fair_homo','greedy'),
    'fair_homo_mdpu': ('fair_homo','mdpu'),
    'fair_native_single': ('fair_native_single','greedy'),
    'fair_native_homo': ('fair_native_homo','mdpu'),
}


def search(out, workers=6):
    started=time.monotonic()
    inputs=load_workloads('development')
    windows=inputs['development']
    ds=domains()
    tasks=[]
    for family,(domain,policy) in FAMILIES.items():
        thresholds=THRESHOLDS if policy in ('mdpu','mdpu_guarded','hard_threshold') else (2,)
        for i,cs in enumerate(ds[domain]):
            validate(cs)
            for t in thresholds:
                tasks.append((family,i,cs,policy,t))
    contract={
        'budget_multipliers':65536,'frequency_Hz':10**9,'datatype':'BF16/FP32 topology extension',
        'dataflow':'OS','data_supply_cycles':0,'control_setup_cycles':4,'queue_depth_each':8,
        'allocation_block_MACs':4096,'PN_max':2048,'MLEN_max':2048,
        'thresholds':list(THRESHOLDS),'domain_counts':{k:len(v) for k,v in ds.items()},
        'wait_best_queue_admission':any(p=='mdpu_guarded' for d,p in FAMILIES.values()),
        'family_policies':FAMILIES,'candidate_parameter_points':len(tasks),
        'selection':'18 development windows ONLY; GM within each batch, equal GM over seven batches',
        'baseline_scope':'Compact family is user-imposed; fair controls allow every declared core capability',
        'physical_scope':'Replicated PLENA-inspired geometry; not verified place/route or native BF16 RTL',
        'workload_order':'Shared first then routed expert ID; no free LPT/oracle sorting',
        'memory_and_nonlinear_scope':'No memory, router, SiLU, interconnect or merge latency; matrix work and controller only',
    }
    write_json(out/'EXPERIMENT_CONTRACT.json',contract)
    write_json(out/'DEV_INPUT_MANIFEST.json',inputs['manifest'])
    print(f'SEARCH {len(tasks)} parameter points / {len(windows)} dev windows / {workers} workers',flush=True)
    best={}; allrows=[]
    with mp.Pool(workers,initializer=init_worker,initargs=(windows,)) as pool:
        for k,r in enumerate(pool.imap_unordered(evaluate_candidate,tasks,chunksize=16),1):
            family,i,policy,t,vals,score=r
            cs=ds[FAMILIES[family][0]][i]
            # Deterministic tie-breaking independent of worker completion order.
            key=(score,sum(resource_ledger(cs)[x] for x in ('full_operand_double_buffer_bytes',)),
                 tuple((c.b,c.r,c.rm,c.rn) for c in cs),t)
            if family not in best or key<best[family][0]:
                best[family]=(key,{'cores':[c.to_dict() for c in cs],'label':label(cs),
                    'policy':policy,'threshold':t,'development_GM_cycles':score,
                    'development_window_cycles':vals,'resource_ledger':resource_ledger(cs)})
            allrows.append({'family':family,'candidate_index':i,'policy':policy,'threshold':t,
                            'development_GM_cycles':score,'physical_config':label(cs)})
            if k%1000==0 or k==len(tasks):
                print(f'SEARCH {k}/{len(tasks)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    allrows.sort(key=lambda r:(r['family'],r['candidate_index'],r['threshold']))
    write_csv(out/'development_search.csv',allrows)
    selection={'families':{k:v[1] for k,v in best.items()},'contract':contract,
               'development_window_ids':[w['id'] for w in windows],
               'elapsed_seconds':time.monotonic()-started}
    write_json(out/'FROZEN_SELECTION.json',selection)
    print('FROZEN',json.dumps({k:round(v[1]['development_GM_cycles'],3) for k,v in best.items()}),flush=True)
    return selection


def evaluation_tasks(selection):
    for family,s in selection['families'].items():
        base={'policy':s['policy'],'threshold':s['threshold'],'prefetch':True,'bypass':True,
              'context_cycles':4,'profile':PROFILES[0].name,'seed':0,
              'wait_best':s['policy']=='mdpu_guarded'}
        yield family,'main',base
        if base['wait_best']:
            yield family,'unguarded_mdpu',dict(base,policy='mdpu',wait_best=False)
        for name,updates in [
            ('same_hw_greedy',{'policy':'greedy'}),
            ('same_hw_mdpu',{'policy':'mdpu'}),
            ('hard_threshold',{'policy':'hard_threshold'}),
            ('no_microop_prefetch',{'policy':'mdpu','prefetch':False}),
            ('no_bypass',{'policy':'mdpu','bypass':False}),
            ('zero_context',{'context_cycles':0}),
            ('context_16',{'context_cycles':16}),
            ('tree_two_cycles',{'profile':PROFILES[1].name}),
            ('serialized_output',{'profile':PROFILES[2].name}),
        ]:
            yield family,name,dict(base,**updates)
        for seed in (0,1,2,3,4):
            yield family,f'random_{seed}',dict(base,policy='random',seed=seed)


def summarize_result(family,variant,cs,w,kw,r):
    # Runtime-specific diagnostics stay in result.jsonl; stable columns here.
    loads=list(r['core_cycles'])
    totals=useful_macs(w)
    row={'design':family,'variant':variant,'window_id':w['id'],'batch':int(w['batch']),
         'physical_config':label(cs),'MAC_allocation':'+'.join(str(c.macs) for c in cs),
         'policy':kw['policy'],'threshold':kw['threshold'],'context_cycles':kw['context_cycles'],
         'prefetch':kw['prefetch'],'bypass':kw['bypass'],'profile':kw['profile'],
         'wait_best':kw.get('wait_best',False),
         'cycles':int(r['cycles']),'compute_time_ms':r['cycles']/1e6,
         'effective_GFLOPs':2*totals/r['cycles'],
         'chip_utilization_pct':100*totals/(65536*r['cycles']),
         'core0_compute_cycles':loads[0],'core1_compute_cycles':loads[1] if len(loads)>1 else 0,
         'useful_MACs':totals,'unique_weight_bytes':unique_weight_bytes(w),
         'required_unique_weight_TBps':unique_weight_bytes(w)/r['cycles']/1000,
         'core0_idle_pct':100*(1-loads[0]/r['cycles']),
         'core1_idle_pct':100*(1-loads[1]/r['cycles']) if len(loads)>1 else 0,
         'core_mismatch_ratio':max(loads)/min(loads) if len(loads)>1 and min(loads)>0 else None}
    return row


def evaluate(out, selection):
    # Heldout files first read only after the frozen choice has been persisted.
    inputs=load_workloads('heldout');windows=inputs['heldout']
    write_json(out/'HELDOUT_INPUT_MANIFEST.json',inputs['manifest'])
    rows=[];repeat=[]
    tasks=list(evaluation_tasks(selection))
    result_path=out/'results.jsonl'
    with result_path.open('w') as f:
        for ti,(family,variant,kw) in enumerate(tasks,1):
            cs=tuple(core_from_dict(d) for d in selection['families'][family]['cores'])
            for w in windows:
                a=simulate(cs,w,details=True,**kw)
                b=simulate(cs,w,details=True,**kw)
                canon=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False)
                sa,sb=canon(a),canon(b)
                if sa!=sb:raise AssertionError(f'Nondeterministic: {family}/{variant}/{w["id"]}')
                row=summarize_result(family,variant,cs,w,kw,a)
                rows.append(row)
                repeat.append({'design':family,'variant':variant,'window_id':w['id'],
                               'two_runs_exactly_equal':True,'result_sha256':hashlib.sha256(sa.encode()).hexdigest()})
                f.write(json.dumps({'row':row,'result':a if variant=='main' else {
                    k:v for k,v in a.items() if k not in ('jobs','events','decision_trace')}},allow_nan=False)+'\n')
            print(f'EVALUATE {ti}/{len(tasks)} {family}/{variant}',flush=True)
    write_csv(out/'window_results.csv',rows);write_csv(out/'repeat_check.csv',repeat)
    profiles=[profile_window(w) for w in windows]
    profile_rows=[]
    for b in BATCHES:
        ps=[p for p in profiles if p['batch']==b]
        profile_rows.append({'batch':b,'windows':len(ps),
            'mean_hot_experts':sum(p['active_routed_experts']-sum(int(n) for m,n in p['routed_me_histogram'].items() if 0<int(m)<=2) for p in ps)/len(ps),
            'mean_cold_experts':sum(sum(int(n) for m,n in p['routed_me_histogram'].items() if 0<int(m)<=2) for p in ps)/len(ps),
            'mean_active_routed':sum(p['active_routed_experts'] for p in ps)/len(ps),
            'shared_experts':1,'unique_weight_MiB_GM':gm(p['total_unique_weight_bytes']/2**20 for p in ps)})
    write_csv(out/'trace_profile.csv',profile_rows)
    # Paired, batch-stratified bootstrap preserves all seven batch weights.
    import numpy as np
    rng=np.random.default_rng(20261013)
    ix={(r['design'],r['variant'],r['window_id']):r for r in rows}
    ci=[]
    for ref in ('compact_single','compact_homo_greedy','compact_homo_mdpu',
                'fair_single','fair_homo_greedy','fair_homo_mdpu','fair_native_single','fair_native_homo'):
        samples=[[ix[(ref,'main',w['id'])]['cycles']/ix[('hetero_mdpu','main',w['id'])]['cycles']
                  for w in windows if int(w['batch'])==b] for b in BATCHES]
        sims=[gm(gm(rng.choice(x,len(x),replace=True)) for x in samples) for _ in range(2000)]
        ci.append({'candidate':'hetero_mdpu','baseline':ref,'speedup':gm(gm(x) for x in samples),
                   'bootstrap_95_low':float(np.quantile(sims,.025)),
                   'bootstrap_95_high':float(np.quantile(sims,.975)), 'draws':2000})
    write_csv(out/'paired_bootstrap.csv',ci)
    write_json(out/'COMPLETION.json',{'rows':len(rows),'runs_per_row':2,'exact_repeats':True,
        'heldout_windows':len(windows),'frozen_selection_sha256':hashlib.sha256((out/'FROZEN_SELECTION.json').read_bytes()).hexdigest(),
        'main_data_rows':sum(r['variant']=='main' for r in rows),'status':'simulation complete; see tests and RTL reports separately'})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--stage',choices=('search','evaluate','all'),default='all')
    parser.add_argument('--guarded',action='store_true',help='Independent development-only rerun with wait-best admission')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.guarded:
        for key,(domain,policy) in list(FAMILIES.items()):
            if policy=='mdpu':FAMILIES[key]=(domain,'mdpu_guarded')
    selection=search(args.out,args.workers) if args.stage!='evaluate' else json.loads((args.out/'FROZEN_SELECTION.json').read_text())
    if args.stage!='search':evaluate(args.out,selection)


if __name__=='__main__':main()
