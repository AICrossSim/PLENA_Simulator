"""Full finite geometry enumeration, bound-aware offline DSE, frozen holdout.

No search-result claim hides the schedule gap: every candidate retains an
admissible lower bound and feasible upper bound. Search refinement is adaptive;
its reported family gap encloses the unknown exact optimum over ALL candidates.
"""
import argparse,json,math,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import ROOT,BUDGETS,BATCHES,write_json,write_csv,digest
from .model import Core,shapes,configurations,geometry,build_tasks,cost
from .schedule import grid,optimize,fast,online
from .inputs import load

ALL_SHAPES=shapes();INDEX={c:i for i,c in enumerate(ALL_SHAPES)}

def costs(windows,mode,shared_chunk=0,budget=65536,force_flow=None):
    tasks=[build_tasks(w,shared_chunk) for w in windows];counts=np.array(list(map(len,tasks)),np.int32)
    nt=max(counts);m=np.ones((len(tasks),nt),np.int64);f=np.full_like(m,1408)
    for j,ts in enumerate(tasks):
        m[j,:len(ts)]=[t['m'] for t in ts];f[j,:len(ts)]=[t['f'] for t in ts]
    out=np.zeros((len(ALL_SHAPES),len(tasks),nt),np.float64)
    def grp(q,nk,lat):return (nk-1)*np.maximum(q,lat)+(q-1)+lat
    for ci,c in enumerate(ALL_SHAPES):
        nm=(m+c.pm-1)//c.pm;lat=3+2*(c.pk.bit_length()-1)
        candidates=[]
        for flow in ('OS','WS','IS') if force_flow is None else (force_flow,):
            total=np.zeros_like(m)
            for n,k,mult,vec in ((f,np.full_like(m,2048),2,3),(np.full_like(m,2048),f,1,2)):
                nn=(n+c.pn-1)//c.pn;nk=(k+c.pk-1)//c.pk
                if flow=='OS':
                    full=nn//8;tail=nn%8
                    t=nm*(full*grp(mult*8,nk,lat)+np.where(tail>0,grp(mult*tail,nk,lat),0))
                elif flow=='WS':t=nn*grp(mult*nm,nk,lat)
                else:t=nm*grp(mult*nn,nk,lat)
                t=np.maximum(t,mult*nm*nn*nk)
                if mode!='matrix':
                    # All legal MAC counts are4096-aligned; scaled quotas are
                    # integer lanes. Fixed64 is a proportional service quota.
                    lanes=c.macs/1024 if mode=='full_scaled' else 64*c.macs/budget
                    t=np.maximum(t,np.ceil(vec*m*n/lanes).astype(np.int64))
                total+=t
            candidates.append(total)
        out[ci]=np.minimum.reduce(candidates)
    return out,counts,tasks

def weights(windows):
    return np.array([1/(len(BATCHES)*sum(q['batch']==w['batch'] for q in windows)) for w in windows])
def objective(a,wt):return np.exp(np.log(a)@wt)
def matrix_for(data,idx,d,counts):return np.ascontiguousarray(data[np.array(idx),d,:counts[d]].T)

def run_case(total,mode,granularity,dev,repeat=True,refine_limit=12):
    start=time.monotonic();target=ROOT/'results'/f'{total}_{mode}_{granularity}';target.mkdir(parents=True,exist_ok=True)
    if (target/'selected.json').exists() and (target/'development_bootstrap.csv').exists():return json.loads((target/'selected.json').read_text())
    chunk=64 if granularity=='shared64' else 0
    data,counts,tasks=costs(dev,mode,chunk,total);wt=weights(dev)
    designs=list(configurations(total));indices=[tuple(INDEX[c] for c in cs) for family,cs in designs]
    lo,hi=grid(data,counts,indices)
    if repeat:
        lo2,hi2=grid(data,counts,indices);assert np.array_equal(lo,lo2) and np.array_equal(hi,hi2)
    los=objective(lo,wt);his=objective(hi,wt)
    print(total,mode,granularity,'grid',len(designs),'seconds',round(time.monotonic()-start,2),flush=True)
    selected={};certificates=[];refined=set()
    families={f:np.array([i for i,(ff,c) in enumerate(designs) if ff==f]) for f in dict.fromkeys(f for f,c in designs)}
    def refine(pick):
        idx=indices[pick]
        def job(d):return optimize(matrix_for(data,idx,d,counts),idx,work=.03)
        with ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(job,range(len(dev))))
        for d,r in enumerate(rr):
            lo[pick,d]=max(lo[pick,d],r['lower']);hi[pick,d]=min(hi[pick,d],r['upper'])
        los[pick]=objective(lo[pick:pick+1],wt)[0];his[pick]=objective(hi[pick:pick+1],wt)[0]
        refined.add(pick)
    # Improve each core-count incumbent once, then spend additional effort
    # across the COMPLETE organization space rather than wasting it proving
    # an already-dominated core-count family optimal.
    for members in families.values():refine(int(members[np.argmin(his[members])]))
    for category in ('single','homo','hetero'):
        members=np.array([i for i,(f,c) in enumerate(designs) if f.startswith(category)]);attempt=0
        while True:
            best=int(members[np.argmin(his[members])]);global_lower=float(min(los[members]));gap=his[best]/global_lower-1
            eligible=[int(i) for i in members if i not in refined and los[i]<his[best]-1e-7]
            if best not in refined:pick=best
            elif gap<=.005 or not eligible or attempt>=refine_limit:break
            else:
                # Alternate best candidate incumbent and weakest lower bound.
                pick=min(eligible,key=lambda i:his[i] if attempt%2 else los[i])
            refine(pick);attempt+=1
    for family,members in families.items():
        best=int(members[np.argmin(his[members])]);family_lower=float(min(los[members]));family_upper=float(his[best]);cs=designs[best][1]
        selection=dict(total_macs=total,mode=mode,granularity=granularity,family=family,
            cores=[dict(pm=c.pm,pn=c.pn,pk=c.pk) for c in cs],geometry=geometry(cs),
            dev_upper_cycles=family_upper,dev_global_lower_cycles=family_lower,
            dse_gap_pct=100*(family_upper/family_lower-1),candidates=len(members),
            refined_candidates=sum(i in refined for i in members),optimality='bounded optimum of declared grid; exact only when gap=0',
            objective='6 batches equally weighted, GM across development windows within batch',
            selection_uses_heldout=False)
        selected[family]=selection;certificates.append(selection)
        print(total,mode,granularity,family,geometry(cs),'GM',round(family_upper),'gap%',round(selection['dse_gap_pct'],4),flush=True)
    rows=[dict(family=f,geometry=geometry(cs),dev_lower_cycles=float(los[i]),dev_upper_cycles=float(his[i]),
        refined=i in refined,repeat_grid_exact=True) for i,(f,cs) in enumerate(designs)]
    write_csv(target/'all_candidates.csv.gz',rows)
    write_csv(target/'within_one_percent.csv',[r for r in rows if r['dev_upper_cycles']<=1.01*selected[r['family']]['dev_upper_cycles']])
    assert np.max(hi)<2**31
    np.savez_compressed(target/'development_bounds.npz',lower=lo,upper=hi.astype(np.int32))
    write_json(target/'selected.json',selected)
    write_json(target/'receipt.json',dict(candidates=len(designs),windows=len(dev),grid_repeats=2,
        grid_bitwise_equal=True,wall_seconds=time.monotonic()-start,dev_digest=digest(dev),
        refinement_limit=refine_limit,
        note='Finite grid complete; scheduler and selection residual gaps explicitly retained.'))
    groups=sorted({w['group'] for w in dev});rng=np.random.default_rng(20261016)
    draws=rng.integers(0,len(groups),(200,len(groups)))
    bootstrap_weights=np.array([[sum(groups[k]==w['group'] for k in sample)/(len(groups)*len(BATCHES)*3) for sample in draws] for w in dev])
    assert np.allclose(bootstrap_weights.sum(axis=0),1)
    scores=np.log(hi)@bootstrap_weights
    stability=[]
    for family,s in selected.items():
        members=np.array([i for i,(f,c) in enumerate(designs) if f==family]);best=int(members[np.argmin(his[members])])
        winners=members[np.argmin(scores[members],axis=0)];regret=np.exp(scores[best]-np.min(scores[members],axis=0))-1
        stability.append(dict(family=family,resamples=200,cluster_groups=len(groups),
            same_geometry_win_pct=100*float(np.mean(winners==best)),within_one_percent_pct=100*float(np.mean(regret<=.01)),
            median_regret_pct=100*float(np.median(regret)),p95_regret_pct=100*float(np.quantile(regret,.95)),
            caveat='Bootstrap uses all-candidate feasible upper-bound scores, not exact optimum for every candidate.'))
    write_csv(target/'development_bootstrap.csv',stability)
    return selected

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int);ap.add_argument('--mode');ap.add_argument('--granularity');ap.add_argument('--refine',type=int,default=12);a=ap.parse_args()
    dev=load()['development']
    for total in (a.budget,) if a.budget else BUDGETS:
        for mode in (a.mode,) if a.mode else ('matrix','full_scaled'):
            for granularity in (a.granularity,) if a.granularity else ('expert','shared64'):
                run_case(total,mode,granularity,dev,refine_limit=a.refine)

if __name__=='__main__':main()
