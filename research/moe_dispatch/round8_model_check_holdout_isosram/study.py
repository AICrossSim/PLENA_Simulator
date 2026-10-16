"""Round 8: exhaustive DSE, repeated stratified validation, equal SRAM read width.

All assignments are recalculated twice with the frozen exact Pareto-DP and
cross-checked against Round 7. Only selection/aggregation are new; old files
are never written. There is no finite-bandwidth execution simulator here.
"""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import numpy as np
import scipy
from scipy.stats import t as student_t

from research.moe_dispatch.round7_64k_dualcore_dse.common import (
    read_json, save_json, save_csv, canonical, gm, sha)
from research.moe_dispatch.round7_64k_dualcore_dse.model import configurations, geometry, task_cost
from research.moe_dispatch.round7_64k_dualcore_dse.scheduler import optimal, greedy
from research.moe_dispatch.round7_64k_dualcore_dse.metrics import window_metrics, aggregate
from research.moe_dispatch.round7_64k_dualcore_dse.trace import load_inputs

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
R7=ROOT.parent/'round7_64k_dualcore_dse'
CFG=read_json(ROOT/'config.json')
BATCHES=(2,4,8,16,64,96,128)
ORGS=('single','homo','hetero')
MODELS=(CFG['main_model'],CFG['sensitivity_model'])

def snapshot():
    """Hash all files, including untracked frozen Round 7 results."""
    paths=[]
    for pattern in ('round2','round3*','round4*','round5*','round6*','round7*'):
        for directory in ROOT.parent.glob(pattern):
            if directory.is_dir():
                paths.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}

def make_splits(windows):
    rng=random.Random(CFG['split_seed'])
    splits=[]
    for rep in range(CFG['split_repetitions']):
        train=[];test=[]
        for b in BATCHES:
            ids=sorted(w['id'] for w in windows if w['batch']==b)
            rng.shuffle(ids);n=CFG['train_counts'][str(b)]
            train.extend(sorted(ids[:n]));test.extend(sorted(ids[n:]))
        assert not set(train)&set(test)
        assert set(train)|set(test)=={w['id'] for w in windows}
        splits.append(dict(split=rep,train_ids=train,test_ids=test))
    return splits

def score(cycles,indices):
    """AM within each batch, equal-weight GM of seven batch AMs, microseconds."""
    means=[math.fsum(int(cycles[i]) for i in ix)/len(ix)/1000 for ix in indices]
    return gm(means),means

def choose(candidates,scores):
    # Floating equality tolerance is <1e-12 us. No outcome-dependent tie break.
    return min(candidates,key=lambda g:(round(scores[g],12),g))

def interval95(diffs):
    n=len(diffs);mean=math.fsum(map(float,diffs))/n
    sd=math.sqrt(math.fsum((float(x)-mean)**2 for x in diffs)/(n-1))
    half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n)
    return mean,mean-half,mean+half

def nondominated(rows):
    """Weak Pareto dominance (<= both, strictly < at least one); retain ties."""
    out=[]
    for r in rows:
        p,t=round(r['peak'],12),round(r['gm'],12)
        if not any(round(s['peak'],12)<=p and round(s['gm'],12)<=t and
                   (round(s['peak'],12)<p or round(s['gm'],12)<t) for s in rows):
            out.append(r)
    return sorted(out,key=lambda r:(r['peak'],r['gm'],r['geometry']))

def get_catalog():
    bycap={};catalog={}
    for cap in CFG['pn_caps']:
        bycap[cap]={o:[] for o in ORGS}
        for org,cs in configurations(cap):
            g=geometry(cs)
            assert sum(c.macs for c in cs)==65536
            # Add integer bytes/cycle before conversion: equal physical widths
            # must not become distinct Pareto coordinates due to float addition.
            catalog[g]=dict(org=org,cores=cs,peak=sum(c.bytes_per_issue for c in cs)/1000)
            bycap[cap][org].append(g)
    return catalog,bycap

def calculate(windows,catalog):
    prior=read_json(R7/'results/VALIDATION.json')
    for file,digest in prior['source_sha256'].items():
        assert sha(R7/file)==digest, f'Frozen code changed: {file}'
    assert sha(R7/'exact.cpp')==prior['cpp_sha256']
    assert sha(R7.parent/'round5_largebatch_compute/model.py')==prior['reused_model_sha256']
    datasets={};validation={};raw=[]
    for model in MODELS:
        path=R7/f'results/{model}_overhead0/all_windows.json.gz'
        assert sha(path)==prior['result_sha256'][str(path.relative_to(R7))]
        cache={(r['geometry'],r['window_id']):r for r in read_json(path)}
        datasets[model]={}
        checked=0
        for gi,g in enumerate(sorted(catalog)):
            cs=catalog[g]['cores'];exact_cycles=[];greedy_cycles=[];rows=[]
            for w in windows:
                costs=[[task_cost(c,t['m'],model,0)['cycles'] for c in cs] for t in w['tasks']]
                e1=optimal(costs);e2=optimal(costs)
                g1=greedy(costs);g2=greedy(costs)
                assert e1==e2 and g1==g2
                assert e1['status']=='OPTIMAL' and e1['cycles']<=g1['cycles']
                old=cache[g,w['id']]
                assert e1['cycles']==old['cycles'] and e1['owner']==old['owner']
                metrics=window_metrics(cs,w,model,0,e1)
                for key in ('cycles','owner','loads','sram_MB','weight_MB','utilization_pct','compute_us'):
                    assert metrics[key]==old[key],(model,g,w['id'],key)
                exact_cycles.append(e1['cycles']);greedy_cycles.append(g1['cycles']);rows.append(metrics)
                raw.append(dict(model=model,geometry=g,org=catalog[g]['org'],window_id=w['id'],
                    batch=w['batch'],optimal_cycles=e1['cycles'],greedy_cycles=g1['cycles'],
                    optimal_owner=e1['owner'],greedy_owner=g1['owner'],
                    optimal_loads=e1['loads'],greedy_loads=g1['loads']))
                checked+=1
            datasets[model][g]=dict(optimal=exact_cycles,greedy=greedy_cycles,metrics=rows)
            if gi%10==0 or gi+1==len(catalog):
                print(f'{model}: {gi+1}/{len(catalog)} geometries, {checked} windows; exact+greedy repeated and verified',flush=True)
        validation[model]=dict(points=checked,exact_runs=2*checked,greedy_runs=2*checked,
                               repeated_bitwise_equal=True,round7_exact_reproduced=True)
    save_json(ROOT/'results/study/assignment_replay.json.gz',raw)
    return datasets,validation

def analyse(windows,catalog,bycap,datasets,splits):
    lookup={w['id']:i for i,w in enumerate(windows)}
    index=lambda ids:[[lookup[x] for x in ids if windows[lookup[x]]['batch']==b] for b in BATCHES]
    allix=index([w['id'] for w in windows])
    tables={};out_splits=[];out_batches=[];selected_iso=[];candidate_scores=[]
    for model in MODELS:
        data=datasets[model]
        fullscores={g:score(d['optimal'],allix)[0] for g,d in data.items()}
        for cap in CFG['pn_caps']:
            key=f'{model}_cap{cap}'
            a=[];b=[];c=[];sa=[];sb=[];sc=[]
            per_split=[]
            for split in splits:
                trainix=index(split['train_ids']);testix=index(split['test_ids'])
                trscores={g:score(d['optimal'],trainix)[0] for g,d in data.items()}
                for org in ORGS:
                    g=choose(bycap[cap][org],trscores)
                    for scheduler in ('optimal','greedy'):
                        value,bmeans=score(data[g][scheduler],testix)
                        row=dict(model=model,pn_cap=cap,split=split['split'],org=org,scheduler=scheduler,
                            geometry=g,train_gm_optimal_us=trscores[g],eval_gm_us=value)
                        per_split.append(row);out_splits.append(row)
                        for batch,mean in zip(BATCHES,bmeans):
                            out_batches.append(dict(**row,batch=batch,eval_batch_AM_us=mean))
            for scheduler in ('optimal','greedy'):
                for org in ORGS:
                    rs=[r for r in per_split if r['org']==org and r['scheduler']==scheduler]
                    counts=Counter(r['geometry'] for r in rs)
                    most=sorted(counts,key=lambda x:(-counts[x],x))[0]
                    vals=[r['eval_gm_us'] for r in rs]
                    a.append(dict(org=org,scheduler=scheduler,most_selected=most,count=counts[most],
                        median_us=float(np.median(vals)),p05_us=float(np.quantile(vals,.05)),
                        p95_us=float(np.quantile(vals,.95)),all_selection_counts=json.dumps(dict(sorted(counts.items())))))
                ratios=[]
                for split in splits:
                    rr={r['org']:r for r in per_split if r['split']==split['split'] and r['scheduler']==scheduler}
                    ratios.append(rr['homo']['eval_gm_us']/rr['hetero']['eval_gm_us'])
                b.append(dict(scheduler=scheduler,median=float(np.median(ratios)),
                    p05=float(np.quantile(ratios,.05)),p95=float(np.quantile(ratios,.95)),
                    hetero_faster_pct=100*sum(x>1+1e-12 for x in ratios)/len(ratios)))
            # 2C uses Round 7 full-set frozen winners, NOT winners chosen on any split.
            frozen=read_json(R7/f'results/{model}_overhead0/selected.json')[str(cap)]
            hg=frozen['homo']['geometry'];xg=frozen['hetero']['geometry']
            for scheduler in ('optimal','greedy'):
                for batch,ix in zip(BATCHES,allix):
                    diffs=[(data[hg][scheduler][i]-data[xg][scheduler][i])/1000 for i in ix]
                    mean,lo,hi=interval95(diffs)
                    c.append(dict(batch=batch,n=len(ix),scheduler=scheduler,homo_geometry=hg,
                        hetero_geometry=xg,mean_homo_minus_hetero_us=mean,ci95_low_us=lo,ci95_high_us=hi,
                        significant=lo>0 or hi<0,
                        direction='hetero' if lo>0 else 'homo' if hi<0 else 'inconclusive'))
            for limit in CFG['sram_peak_caps_TBps']:
                label='unlimited' if limit is None else str(limit)
                chosen={}
                for org in ORGS:
                    legal=[g for g in bycap[cap][org] if limit is None or catalog[g]['peak']<=limit+1e-12]
                    chosen[org]=choose(legal,fullscores) if legal else None
                for org in ORGS:
                    g=chosen[org]
                    row=dict(sram_cap_TBps=label,org=org,geometry=g or 'none',
                        sram_peak_TBps=catalog[g]['peak'] if g else None,gm_compute_us=fullscores[g] if g else None,
                        speedup_vs_homo=fullscores[chosen['homo']]/fullscores[g] if g and chosen['homo'] else None,
                        speedup_vs_single=fullscores[chosen['single']]/fullscores[g] if g and chosen['single'] else None,
                        feasible_candidates=sum(limit is None or catalog[x]['peak']<=limit+1e-12 for x in bycap[cap][org]))
                    sa.append(row);selected_iso.append(dict(model=model,pn_cap=cap,**row))
                    if not g: continue
                    agg=aggregate(data[g]['metrics'])
                    assert abs(gm([x['compute_us'] for x in agg])-fullscores[g])<1e-9
                    for r in agg:
                        sc.append(dict(sram_cap_TBps=label,batch=r['batch'],org=org,geometry=g,
                            compute_us=r['compute_us'],hbm_req_TBps=r['hbm_req_TBps'],hbm_fetch_us=r['hbm_fetch_us'],
                            roofline_us=r['roofline_us'],bound=r['bound'],compute_bound_windows=r['compute_bound_windows'],
                            windows=r['windows'],sram_req_TBps=r['sram_req_TBps'],
                            sram_peak_TBps=r['sram_peak_TBps'],equivalent_GFLOP=r['equivalent_GFLOP'],
                            utilization_pct=r['utilization_pct'],weight_MB=r['weight_MB']))
            for org in ORGS:
                rr=[dict(org=org,geometry=g,peak=catalog[g]['peak'],gm=fullscores[g]) for g in bycap[cap][org]]
                sb.extend(nondominated(rr))
                for r in rr:candidate_scores.append(dict(model=model,pn_cap=cap,**r))
            tables[key]={'2A':a,'2B':b,'2C':c,'3A':sa,'3B':sb,'3C':sc}
    return dict(tables=tables,split_results=out_splits,split_batches=out_batches,
                iso_selected=selected_iso,all_candidate_scores=candidate_scores)

def annotate_table9():
    rows=[]
    for model in MODELS:
        for cap in CFG['pn_caps']:
            src=R7/f'results/{model}_overhead0/cap{cap}/table9.csv'
            for r in csv.DictReader(src.open()):
                dims=[list(map(int,c.split('x'))) for c in r['形状'].split('+')]
                rows.append(dict(model=model,pn_cap=cap,**r,
                    **{'旧G定义':'MLEN/BLEN（子阵列组数），不是PK',
                       '旧BLEN':'+'.join(str(d[0]) for d in dims),
                       '旧MLEN':'+'.join(str(d[0]*d[2]) for d in dims),
                       '本轮G定义':'PK：单次发射的归约并行元素数',
                       '本轮PK':'+'.join(str(d[2]) for d in dims),
                       '可否仅按同名形状视为同硬件同周期':'否：映射、Shared融合、AM/GM及控制排空不同'}))
    save_csv(ROOT/'results/study/table9_annotated.csv',rows)

def run():
    assert CFG['approval_status'].startswith('user_confirmed')
    before=snapshot()
    windows,meta=load_inputs(write=False)
    assert len(windows)==135
    splits=make_splits(windows);assert splits==make_splits(windows)
    save_json(ROOT/'results/study/splits.json',dict(seed=CFG['split_seed'],algorithm='Python random.Random.shuffle',
        policy='per-batch AM; 7-batch equal-weight GM; 71 train / 64 eval per split',splits=splits))
    catalog,bycap=get_catalog()
    datasets,checks=calculate(windows,catalog)
    first=analyse(windows,catalog,bycap,datasets,splits)
    second=analyse(windows,catalog,bycap,datasets,splits)
    assert canonical(first)==canonical(second)
    for scenario,tables in first['tables'].items():
        for name,rows in tables.items():save_csv(ROOT/'results/study'/scenario/f'table{name}.csv',rows)
    for key in ('split_results','split_batches','iso_selected','all_candidate_scores'):
        save_csv(ROOT/'results/study'/f'{key}.csv',first[key])
    save_json(ROOT/'results/study/summary.json',first['tables'])
    annotate_table9()
    assert before==snapshot(), 'Frozen rounds changed'
    changes=subprocess.check_output(['git','diff','HEAD','--name-only','--',
        'research/moe_dispatch/round2','research/moe_dispatch/round3','research/moe_dispatch/round4_compute_first',
        'research/moe_dispatch/round5_largebatch_compute','research/moe_dispatch/round6_feasible_cell'],cwd=REPO,text=True)
    assert not changes
    save_json(ROOT/'results/study/VALIDATION.json',dict(passed=True,model_assignment_checks=checks,
        windows=135,split_count=50,seed=CFG['split_seed'],train_windows=71,evaluation_windows=64,
        models=list(MODELS),caps=CFG['pn_caps'],overhead=0,
        candidate_counts={str(cap):{o:len(gs) for o,gs in orgs.items()} for cap,orgs in bycap.items()},
        two_analyses_bitwise_identical=True,round2_to_7_unchanged=True,frozen_file_sha256=before,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        experiment_config_sha256=sha(ROOT/'config.json'),
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
        result_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'results/study').rglob('*'))
                       if p.is_file() and p.name!='VALIDATION.json'},
        stage2_executed=True,stage3_executed=True))
    print('COMPLETE: both models, both PN caps; 50 splits; 5 SRAM limits; all exact/greedy repeats identical.',flush=True)

if __name__=='__main__':run()
