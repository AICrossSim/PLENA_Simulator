"""Audit every result, exact assignments, tables and unchanged earlier rounds."""
import csv
import hashlib
import math
import subprocess
from collections import Counter
from .common import ROOT, REPO, CFG, read_json,save_json,sha,canonical,gm
from .model import Core,task_cost,configurations
from .scheduler import optimal
from .metrics import window_metrics,aggregate

def cpsat_check(costs):
    from ortools.sat.python import cp_model
    groups=Counter(tuple(p) for p in costs)
    m=cp_model.CpModel();variables=[]
    for i,(pair,count) in enumerate(groups.items()):
        v=m.new_int_var(0,count,f'n0_{i}');variables.append((v,pair,count))
    total0=sum(p[0] for p in costs);total1=sum(p[1] for p in costs)
    t=m.new_int_var(0,max(total0,total1),'makespan')
    m.add(sum(v*p[0] for v,p,n in variables)<=t)
    m.add(sum((n-v)*p[1] for v,p,n in variables)<=t)
    m.minimize(t);solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.random_seed=20261016
    status=solver.solve(m)
    assert status==cp_model.OPTIMAL
    return solver.value(t)

def run():
    ws=read_json(ROOT/'inputs/windows.json.gz');wmap={w['id']:w for w in ws}
    source_dirs=['round2','round3','round4_compute_first','round5_largebatch_compute','round6_feasible_cell']
    diff=subprocess.check_output(['git','diff','HEAD','--',*[str(ROOT.parent.relative_to(REPO)/s) for s in source_dirs]],cwd=REPO)
    assert not diff, 'Earlier rounds must remain unchanged'
    validated=0;receipts=[];tables=0;digests={}
    for model in CFG['models']:
        for overhead in CFG['overheads_per_gemm']:
            d=ROOT/'results'/f'{model}_overhead{overhead}'
            rows=read_json(d/'all_windows.json.gz');summaries=read_json(d/'all_candidates.json')
            selections=read_json(d/'selected.json');receipt=read_json(d/'receipt.json')
            assert receipt['all_repetitions_identical'] and receipt['executions']==receipt['points']*2
            assert len(rows)==157*135==receipt['points']
            grouped={x['geometry']:[] for x in summaries}
            cores={x['geometry']:tuple(Core(**c) for c in x['cores']) for x in summaries}
            for r in rows:
                cs=cores[r['geometry']];w=wmap[r['window_id']]
                assert sum(c.macs for c in cs)==65536 and len(r['owner'])==len(w['tasks'])
                p=[[task_cost(c,t['m'],model,overhead)['cycles'] for c in cs] for t in w['tasks']]
                loads=[sum(p[i][c] for i,o in enumerate(r['owner']) if o==c) for c in range(len(cs))]
                assert loads==r['loads'] and max(loads)==r['cycles'] and r['status']=='OPTIMAL'
                if len(cs)==1:assert r['cycles']==sum(x[0] for x in p)
                assert r['weight_MB']==len(w['tasks'])*3*2048*1408*2/1e6
                schedule={k:r[k] for k in ('cycles','owner','loads','status','frontier_peak')}
                expected=window_metrics(cs,w,model,overhead,schedule)
                assert all(expected[k]==r[k] for k in expected)
                grouped[r['geometry']].append(r);validated+=1
            for x in summaries:
                a=aggregate(grouped[x['geometry']]);assert a==x['by_batch']
                assert gm([r['compute_us'] for r in a])==x['gm_compute_us']
            for cap in CFG['pn_caps']:
                expected_configs=list(configurations(cap))
                assert sum(cap in x['caps'] for x in summaries)==len(expected_configs)
                for org in ('single','homo','hetero'):
                    best=min((x for x in summaries if x['org']==org and cap in x['caps']),key=lambda x:(x['gm_compute_us'],x['geometry']))
                    assert selections[str(cap)][org]['geometry']==best['geometry']
                for i in range(1,10):
                    for suffix in ('csv','md'):
                        p=d/f'cap{cap}'/f'table{i}.{suffix}';assert p.exists() and p.stat().st_size>0
                    tables+=1
            receipts.append(receipt)
            digests[str(d.relative_to(ROOT))+'/all_windows.json.gz']=sha(d/'all_windows.json.gz')
    # Independent integer programming certificates on real large-batch inputs.
    check_shapes=[(Core(3,256),Core(1,256)),(Core(4,192),Core(1,256)),
                  (Core(7,128),Core(1,128)),(Core(1,512),Core(1,512))]
    w=next(w for w in ws if w['batch']==128);checks=[]
    for model in CFG['models']:
        for overhead in CFG['overheads_per_gemm']:
            for cs in check_shapes:
                costs=[[task_cost(c,t['m'],model,overhead)['cycles'] for c in cs] for t in w['tasks']]
                cp=cpsat_check(costs);dp=optimal(costs)['cycles'];assert cp==dp
                checks.append(dict(model=model,overhead=overhead,geometry='+'.join(c.shape for c in cs),cycles=dp))
    with (ROOT/'results/legacy_attribution.csv').open() as f:legacy=list(csv.DictReader(f))
    errors=[abs(float(r['replay_vs_published_pct'])) for r in legacy if r['replay_vs_published_pct']]
    assert max(errors)<.02, 'Published legacy values failed to reproduce at printed precision'
    result=dict(passed=True,windows=len(ws),batch_counts=dict(Counter(w['batch'] for w in ws)),
        validated_candidate_window_points=validated,paired_executions=2*validated,
        all_exact_no_timeout=True,all_repeats_identical=True,table_pairs_csv_md=tables,
        cpsat_independent_checks=checks,old_published_max_replay_error_pct=max(errors),
        earlier_rounds_unchanged=True,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
        cpp_sha256=sha(ROOT/'exact.cpp'),config_sha256=sha(ROOT/'config.json'),
        reused_model_sha256=sha(ROOT.parent/'round5_largebatch_compute/model.py'),
        result_sha256=digests)
    save_json(ROOT/'results/VALIDATION.json',result)
    print(canonical(result),flush=True)

if __name__=='__main__':run()
