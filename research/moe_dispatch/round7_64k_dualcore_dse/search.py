"""All candidates, all original windows, exact assignments; no time cutoff."""
import argparse
from collections import Counter
import hashlib
import time
from .common import CFG, ROOT, canonical, gm, read_json, save_json, save_csv, sha
from .trace import load_inputs
from .model import configurations, references, geometry, task_cost
from .scheduler import optimal, greedy, library
from .metrics import window_metrics, aggregate

def all_designs():
    out={}
    for cap in CFG['pn_caps']:
        for org,cs in configurations(cap):
            g=geometry(cs)
            if g not in out:out[g]=dict(org=org,cores=cs,caps=[])
            out[g]['caps'].append(cap)
    for label,cs in references():
        g=geometry(cs)
        if g not in out:
            # Equal-budget heterogeneous search canonicalizes the core order.
            # Preserve the user's legacy Core0/Core1 order for reference replay.
            out[g]=dict(org='reference',cores=cs,caps=[])
    return out

def run(model,overhead):
    ws,meta=load_inputs(write=False);library();start=time.monotonic();designs=all_designs()
    d=ROOT/'results'/f'{model}_overhead{overhead}';d.mkdir(parents=True,exist_ok=True)
    rows_all=[]; summaries=[]; digest=hashlib.sha256();points=0
    for j,(g,design) in enumerate(sorted(designs.items())):
        cs=design['cores'];rows=[]
        for w in ws:
            costs=[[task_cost(c,t['m'],model,overhead)['cycles'] for c in cs] for t in w['tasks']]
            solves=[optimal(costs) for _ in range(CFG['repetitions'])]
            assert all(s==solves[0] for s in solves), (g,w['id'])
            result=solves[0]
            digest.update(canonical([g,w['id'],result]).encode());points+=1
            r=window_metrics(cs,w,model,overhead,result)
            r.update(geometry=g,org=design['org'],model=model,overhead=overhead)
            rows.append(r)
        ag=aggregate(rows);score=gm([r['compute_us'] for r in ag])
        summaries.append(dict(geometry=g,org=design['org'],cores=[c.__dict__ for c in cs],
                              caps=design['caps'],gm_compute_us=score,by_batch=ag))
        rows_all.extend(rows)
        if j%20==0 or j+1==len(designs):
            print(f'{model} overhead={overhead}: {j+1}/{len(designs)} designs, {points} exact points',flush=True)
    save_json(d/'all_windows.json.gz',rows_all)
    save_json(d/'all_candidates.json',summaries)
    selected={}
    for cap in CFG['pn_caps']:
        selected[str(cap)]={}
        for org in ('single','homo','hetero'):
            candidates=[x for x in summaries if cap in x['caps'] and x['org']==org]
            best=min(candidates,key=lambda x:(x['gm_compute_us'],x['geometry']))
            selected[str(cap)][org]=dict(geometry=best['geometry'],gm_compute_us=best['gm_compute_us'])
    save_json(d/'selected.json',selected)
    save_json(d/'receipt.json',dict(model=model,overhead=overhead,candidates=len(designs),
        candidate_counts_by_cap={str(cap):dict(Counter(x['org'] for x in designs.values() if cap in x['caps'])) for cap in CFG['pn_caps']},
        windows=len(ws),points=points,executions=points*CFG['repetitions'],
        all_repetitions_identical=True,all_assignments_status='OPTIMAL',algorithm='exact_pareto_DP_no_cutoff',
        results_sha256=digest.hexdigest(),elapsed_seconds=time.monotonic()-start,
        config_sha256=sha(ROOT/'config.json'),source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
        exact_cpp_sha256=sha(ROOT/'exact.cpp')))
    print(f'COMPLETE {model} overhead={overhead} {selected}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=CFG['models'],required=True)
    ap.add_argument('--overhead',type=int,choices=CFG['overheads_per_gemm'],required=True)
    a=ap.parse_args();run(a.model,a.overhead)
