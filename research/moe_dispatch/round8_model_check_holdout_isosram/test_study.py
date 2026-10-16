import itertools
import json
import math
from collections import Counter
import numpy as np
from .study import (ROOT, R7, CFG, MODELS, BATCHES, get_catalog, make_splits, score,
                    choose, interval95, nondominated, read_json)
from research.moe_dispatch.round7_64k_dualcore_dse.scheduler import optimal,greedy
from research.moe_dispatch.round7_64k_dualcore_dse.trace import load_inputs

def test_exact_scheduler_against_exhaustive_assignment():
    for costs in ([[3,7],[4,2],[6,6],[1,5]],[[10,1],[1,10],[3,3]],[[7,7]]*6):
        brute=min(max(sum(p[c] for p,c in zip(costs,owners) if c==k) for k in range(2))
                  for owners in itertools.product((0,1),repeat=len(costs)))
        assert optimal(costs)['cycles']==brute<=greedy(costs)['cycles']

def test_split_counts_disjoint_complete_reproducible():
    windows,_=load_inputs(write=False);splits=make_splits(windows)
    assert splits==make_splits(windows) and len(splits)==50
    batches={w['id']:w['batch'] for w in windows}
    for r in splits:
        assert len(r['train_ids'])==71 and len(r['test_ids'])==64
        assert set(r['train_ids']).isdisjoint(r['test_ids'])
        assert set(r['train_ids']+r['test_ids'])==set(batches)
        assert Counter(batches[i] for i in r['train_ids'])=={int(b):n for b,n in CFG['train_counts'].items()}

def test_score_is_batch_balanced_not_window_balanced():
    # Six small batches with two identical windows, last batch with one window.
    ix=[[2*j,2*j+1] for j in range(6)]+[[12]]
    cycles=[1000]*12+[128000]
    assert abs(score(cycles,ix)[0]-2)<1e-12

def test_selection_cannot_see_evaluation_values():
    train=[[i] for i in range(7)];test=[[i+7] for i in range(7)]
    a=[1000]*7+[100000]*7;b=[2000]*7+[1]*7
    assert choose(['a','b'],{'a':score(a,train)[0],'b':score(b,train)[0]})=='a'
    assert choose(['a','b'],{'a':score(a,test)[0],'b':score(b,test)[0]})=='b'

def test_student_t_interval_and_zero_variance():
    m,lo,hi=interval95([1,2,3,4,5])
    assert m==3 and abs(lo-1.0367568385)<1e-9 and abs(hi-4.9632431615)<1e-9
    assert interval95([.5]*9)==(.5,.5,.5)

def test_pareto_weak_domination_preserves_equal_points():
    r=[dict(geometry=str(i),peak=p,gm=t) for i,(p,t) in enumerate([(1,5),(2,4),(3,6),(2,5),(1,5)])]
    assert {q['geometry'] for q in nondominated(r)}=={'0','1','4'}

def test_resource_legality_and_low_budget_feasibility():
    cat,bycap=get_catalog()
    for cap in (256,512):
        assert len(bycap[cap]['single'])==(5 if cap==256 else 6)
        for org,gs in bycap[cap].items():
            for g in gs:
                cs=cat[g]['cores']
                assert sum(c.macs for c in cs)==65536
                assert all(c.g==64 and c.pn%16==0 and c.pn<=cap for c in cs)
    low=[g for g in cat if cat[g]['peak']<=8.2]
    assert low==['32x32x64']

def test_completed_result_invariants():
    summary=read_json(ROOT/'results/study/summary.json')
    for model in MODELS:
        for cap in (256,512):
            s=summary[f'{model}_cap{cap}']
            assert len(s['2A'])==6 and len(s['2B'])==2 and len(s['2C'])==14
            assert len(s['3A'])==15
            for r in s['2A']:
                assert sum(json.loads(r['all_selection_counts']).values())==50
                assert r['p05_us']<=r['median_us']<=r['p95_us']
            for r in s['3A']:
                if r['geometry']=='none':assert r['feasible_candidates']==0
                elif r['sram_cap_TBps']!='unlimited':assert r['sram_peak_TBps']<=float(r['sram_cap_TBps'])+1e-12
            for r in s['3C']:
                assert r['roofline_us']+1e-9>=max(r['compute_us'],r['hbm_fetch_us'])
                assert r['sram_req_TBps']<=r['sram_peak_TBps']+1e-12
                assert abs(r['hbm_fetch_us']-r['weight_MB']/4)<1e-9
    check=read_json(ROOT/'results/study/VALIDATION.json')
    assert check['passed'] and check['two_analyses_bitwise_identical'] and check['round2_to_7_unchanged']
    assert sum(v['points'] for v in check['model_assignment_checks'].values())==42120

def test_table9_definition_is_explicit():
    import csv
    rows=list(csv.DictReader((ROOT/'results/study/table9_annotated.csv').open()))
    assert len(rows)==112
    q=next(r for r in rows if r['参照']=='旧单核')
    assert q['旧MLEN']=='256' and q['本轮PK']=='64'
    assert 'MLEN/BLEN' in q['旧G定义']
