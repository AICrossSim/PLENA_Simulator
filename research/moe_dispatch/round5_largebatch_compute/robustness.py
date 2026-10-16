"""Group bootstrap across every enumerated core count, development only."""
import csv,gzip,json
import numpy as np
from .common import ROOT,BATCHES,write_csv
from .inputs import load

def build():
    dev=load()['development'];groups=sorted({w['group'] for w in dev});rng=np.random.default_rng(20261016)
    draws=rng.integers(0,len(groups),(200,len(groups)))
    weights=np.array([[sum(groups[k]==w['group'] for k in d)/(len(groups)*len(BATCHES)*3) for d in draws] for w in dev])
    assert np.allclose(weights.sum(axis=0),1)
    out=[]
    for path in sorted((ROOT/'results').iterdir()):
        if not (path/'selected.json').exists():continue
        selected=json.loads((path/'selected.json').read_text());first=next(iter(selected.values()))
        with gzip.open(path/'all_candidates.csv.gz','rt') as f:rs=list(csv.DictReader(f))
        with np.load(path/'development_bounds.npz') as a:scores=np.log(a['upper'])@weights
        for category in ('single','homo','hetero'):
            members=np.array([i for i,r in enumerate(rs) if r['family'].startswith(category)])
            chosen=min((v for k,v in selected.items() if k.startswith(category)),key=lambda v:v['dev_upper_cycles'])
            index=next(i for i in members if rs[i]['family']==chosen['family'] and rs[i]['geometry']==chosen['geometry'])
            best=members[np.argmin(scores[members],axis=0)];regret=np.exp(scores[index]-np.min(scores[members],axis=0))-1
            out.append(dict(total_macs=first['total_macs'],mode=first['mode'],granularity=first['granularity'],organization=category,
                chosen_family=chosen['family'],geometry=chosen['geometry'],resamples=200,
                exact_geometry_win_pct=100*float(np.mean(best==index)),within_one_percent_pct=100*float(np.mean(regret<=.01)),
                median_regret_pct=100*float(np.median(regret)),p95_regret_pct=100*float(np.quantile(regret,.95)),
                distinct_bootstrap_winners=len(set(best.tolist())),caveat='All candidate feasible-schedule upper scores; not exact scheduling optimum for every candidate.'))
        del scores,rs
    write_csv(ROOT/'report/organization_bootstrap.csv',out)
    return out

if __name__=='__main__':build()
