"""Exhaustive finite geometry grid, certified schedule bounds, dev-only selection.

Every grid point has a feasible schedule and a strict compute lower bound.
CP-SAT refines incumbents, including per-batch incumbents. Residual DSE gaps
remain explicit; a heuristic grid minimum is never called a proven optimum.
"""
import argparse,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import *
from .inputs import load
from .model import configurations,make_costs,INDEX,geometry,matrix_for
from .solver import grid,optimize

def selection_weights(ws):
 return np.array([1/(len(BATCHES)*sum(x['batch']==w['batch'] for x in ws)) for w in ws])

def bootstrap_weights(ws,n=200):
 groups=sorted({g for w in ws for g in w['source_groups']});rng=np.random.default_rng(SEED)
 draws=rng.integers(0,len(groups),(n,len(groups)));weights=np.zeros((len(ws),n))
 for j,sample in enumerate(draws):
  mult=np.bincount(sample,minlength=len(groups))
  for b in BATCHES:
   ids=[i for i,w in enumerate(ws) if w['batch']==b]
   v=np.array([sum(mult[groups.index(g)] for g in ws[i]['source_groups'])/len(ws[i]['source_groups']) for i in ids],float)
   weights[ids,j]=v/v.sum()/len(BATCHES)
 assert np.allclose(weights.sum(axis=0),1)
 return weights

def run_case(budget,chunk,limit=24):
 path=ROOT/'E1/search'/f'{budget}_{chunk}';cache=CACHE/f'{budget}_{chunk}';cache.mkdir(parents=True,exist_ok=True)
 if (path/'selection.json').exists():return read_json(path/'selection.json')
 ws=load()['development'];data,counts,ts,ns,gu,dn=make_costs(ws,chunk,budget=budget)
 designs=list(configurations(budget,bool(chunk)));indices=[tuple(INDEX[c] for c in cs) for org,cs in designs]
 wt=selection_weights(ws);bids={b:np.array([i for i,w in enumerate(ws) if w['batch']==b]) for b in BATCHES}
 orgs=list(dict.fromkeys(f for f,cs in designs));members={f:np.array([i for i,(ff,cs) in enumerate(designs) if ff==f]) for f in orgs}
 start=time.monotonic()
 raw=cache/'grid.npz'
 if raw.exists():
  z=np.load(raw);lo=z['lo'];hi=z['hi']
 else:
  lo,hi=grid(data,counts,ns,gu,dn,indices)
  lo2,hi2=grid(data,counts,ns,gu,dn,indices);assert np.array_equal(lo,lo2) and np.array_equal(hi,hi2)
  np.savez_compressed(raw,lo=lo,hi=hi)
 print('GRID',budget,chunk,len(designs),'candidates',round(time.monotonic()-start,2),'seconds',flush=True)
 scores=lambda a:np.exp(np.log(a)@wt)
 ls=scores(lo);hs=scores(hi);refined=set()
 def refine(pick):
  cp=cache/f'refine_{pick}.json'
  if cp.exists():rr=read_json(cp)
  else:
   idx=indices[pick]
   def job(w):
    p=matrix_for(data,idx,w,counts);g=gu[list(idx),w];d=dn[list(idx),w]
    a=optimize(p,ns[w],g,d,idx,work=.08);b=optimize(p,ns[w],g,d,idx,work=.08);assert a==b
    return a
   with ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(job,range(len(ws))))
   write_json(cp,rr)
  for w,r in enumerate(rr):lo[pick,w]=max(lo[pick,w],r['lower']);hi[pick,w]=min(hi[pick,w],r['upper'])
  ls[pick]=scores(lo[pick:pick+1])[0];hs[pick]=scores(hi[pick:pick+1])[0];refined.add(pick)
 # All organization-wide and per-batch incumbents are refined, then explore
 # unrefined lower-bound contenders. Selecting a heldout result is forbidden.
 initial=set()
 for ids in members.values():
  initial.add(int(ids[np.argmin(hs[ids])]))
  for ix in bids.values():initial.add(int(ids[np.argmin(np.log(hi[ids][:,ix]).mean(axis=1))]))
 for i in sorted(initial):refine(i)
 for org,ids in members.items():
  for attempt in range(limit):
   best=int(ids[np.argmin(hs[ids])]);eligible=[int(i) for i in ids if i not in refined and ls[i]<hs[best]]
   if best not in refined:refine(best);continue
   if hs[best]/min(ls[ids])<=1.005 or not eligible:break
   pick=min(eligible,key=lambda i:hs[i] if attempt%2 else ls[i]);refine(pick)
  print('REFINE',budget,chunk,org,'GM',round(min(hs[ids]),3),'gap%',round(100*(min(hs[ids])/min(ls[ids])-1),3),flush=True)
 # Refine any new per-batch incumbent until no unrefined winner remains.
 while True:
  pending=set()
  for ids in members.values():
   pending.add(int(ids[np.argmin(hs[ids])]))
   for ix in bids.values():pending.add(int(ids[np.argmin(np.log(hi[ids][:,ix]).mean(axis=1))]))
  pending-=refined
  if not pending:break
  for i in sorted(pending):refine(i)
 selected={};perbatch=[]
 def record(org,pick,ix=None):
  cs=designs[pick][1];ids=members[org]
  upper=float(hs[pick]) if ix is None else gm(hi[pick,ix]);lower=float(min(ls[ids])) if ix is None else float(np.exp(np.log(lo[ids][:,ix]).mean(axis=1)).min())
  return dict(org=org,budget=budget,chunk=chunk,candidate=pick,cores=[dict(pm=c.pm,pn=c.pn,pk=c.pk) for c in cs],geometry=geometry(cs),
   used_macs=sum(c.macs for c in cs),unused_macs=budget-sum(c.macs for c in cs),dev_upper_cycles=upper,dev_global_lower_cycles=lower,
   dse_gap_pct=100*(upper/lower-1),dev_by_batch={str(b):gm(hi[pick,x]) for b,x in bids.items()},
   dev_by_batch_lb={str(b):gm(lo[pick,x]) for b,x in bids.items()},selection_uses_heldout=False)
 for org,ids in members.items():
  pick=int(ids[np.argmin(hs[ids])]);selected[org]=record(org,pick)
  for b,ix in bids.items():
   pick=int(ids[np.argmin(np.log(hi[ids][:,ix]).mean(axis=1))]);perbatch.append(dict(batch=b,**record(org,pick,ix)))
 rows=[]
 for i,(org,cs) in enumerate(designs):
  r=dict(candidate=i,org=org,cores=len(cs),geometry=geometry(cs),used_macs=sum(c.macs for c in cs),dev_lower_cycles=ls[i],dev_upper_cycles=hs[i],refined=i in refined)
  for b,ix in bids.items():r[f'B{b}_upper']=gm(hi[i,ix]);r[f'B{b}_lower']=gm(lo[i,ix])
  rows.append(r)
 write_csv(path/'all_candidates.csv.gz',rows)
 np.savez_compressed(cache/'development_bounds.npz',lo=lo,hi=hi)
 weights=bootstrap_weights(ws);loghi=np.log(hi);bs=loghi@weights;stability=[]
 for org,ids in members.items():
  pick=selected[org]['candidate'];winners=ids[np.argmin(bs[ids],axis=0)];regret=np.exp(bs[pick]-bs[ids].min(axis=0))-1
  stability.append(dict(org=org,resamples=200,same_hardware_pct=100*float(np.mean(winners==pick)),within_1pct_pct=100*float(np.mean(regret<=.01)),p95_regret_pct=100*float(np.quantile(regret,.95)),
   method='paired source-group bootstrap; feasible upper bounds across complete grid, not exact all-candidate optima'))
 write_csv(path/'bootstrap.csv',stability);write_json(path/'per_batch_selection.json',perbatch)
 write_json(path/'receipt.json',dict(budget=budget,chunk=chunk,candidates=len(designs),refined=len(refined),windows=len(ws),grid_repeats=2,grid_bitwise_equal=True,
  refined_repeats=2,dev_digest=digest(ws),seconds=time.monotonic()-start,broadcast_TBps=W_TBPS,lower='strict relaxation lower bound; refined CP-SAT bounds',
  grid_sha256=hashlib.sha256(lo.tobytes()+hi.tobytes()).hexdigest()))
 write_json(path/'selection.json',selected)
 return selected

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int);ap.add_argument('--chunk',type=int);ap.add_argument('--refine',type=int,default=24);a=ap.parse_args()
 for b in ([a.budget] if a.budget else BUDGETS):
  for c in ([a.chunk] if a.chunk is not None else (0,64)):run_case(b,c,a.refine)
if __name__=='__main__':main()
