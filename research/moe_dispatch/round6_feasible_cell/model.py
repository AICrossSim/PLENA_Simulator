"""Frozen R5 compute contract plus explicitly streamed Shared multicast.

Ordinary operand supply is ideal. Only ADDITIONAL Shared replicas consume the
finite broadcast port. A Shared cohort traverses256KiB weight blocks, and all
assigned row shards on each receiver reuse a block before releasing it. Two
buffers per receiver suffice; full33MiB per-core pinning is not assumed.
Gate/Up and Down have a common cohort phase barrier. This is a reproducible
implementation contract, not proof of an optimal multicast network.
"""
import math
import numpy as np
from research.moe_dispatch.round5_largebatch_compute.model import Core,shapes,configurations as r5configs,geometry,cost,projection
from research.moe_dispatch.round5_largebatch_compute.search import costs as r5costs,ALL_SHAPES,INDEX,matrix_for
from .common import BLOCK,SHARED_GU,SHARED_DOWN,W_TBPS

def configurations(total,split=False):
 seen=set()
 for family,cs in r5configs(total):
  org='S' if family=='single' else ('H+' if split else 'H') if family.startswith('homo') else ('P+' if split else 'P')
  if split and org=='S':continue
  seen.add(tuple(cs));yield org,tuple(cs)
 # Exactly filling a power-of-two budget with12 or13 identical aligned cores
 # is impossible. Search ALL fitting legal shapes and account unused budget.
 for n in (12,13):
  for c in shapes():
   if n*c.macs<=total:
    cs=(c,)*n
    if cs not in seen:yield ('H+' if split else 'H'),cs

def tasks(w,chunk=0,all_experts=False):
 out=[]
 # Routed descriptors precede Shared in storage only; execution is separate.
 for e in w['experts']:
  block=chunk if chunk and (e['is_shared'] or all_experts) else e['Me']
  for start in range(0,e['Me'],block):
   out.append(dict(expert=e['id'],m=min(block,e['Me']-start),f=e['F'],shared=e['is_shared'],row_start=start))
 return out

def phases(c,t,mode,budget):
 q=cost(c,t['m'],t['f'],mode,budget)
 ps=[projection(c,t['m'],t['f'],2048,True,q['flow']),projection(c,t['m'],2048,t['f'],False,q['flow'])]
 lanes=c.macs/1024
 v=[p['cycles'] if mode=='matrix' else max(p['cycles'],math.ceil(p['vector']/lanes)) for p in ps]
 assert sum(v)==q['cycles']
 return v

def make_costs(ws,chunk=0,mode='full_scaled',budget=65536,all_experts=False):
 ts=[tasks(w,chunk,all_experts) for w in ws];counts=np.array([len(t) for t in ts],np.int32)
 ns=np.array([sum(x['shared'] for x in t) for t in ts],np.int32)
 nt=max(counts);data=np.zeros((len(ALL_SHAPES),len(ws),nt),np.float64)
 # Vectorized R5 cost evaluator, exactly the same arithmetic contract.
 if not all_experts:
  data,counts,_=r5costs(ws,mode,chunk,budget)
 else:
  for ci,c in enumerate(ALL_SHAPES):
   for wi,tasks_ in enumerate(ts):
    data[ci,wi,:len(tasks_)]=[cost(c,t['m'],t['f'],mode,budget)['cycles'] for t in tasks_]
 gu=np.zeros((len(ALL_SHAPES),len(ws)),np.float64);down=gu.copy()
 for ci,c in enumerate(ALL_SHAPES):
  for wi,t in enumerate(ts):
   shared=[x for x in t if x['shared']];assert len({x['m'] for x in shared})==1
   gu[ci,wi],down[ci,wi]=phases(c,shared[0],mode,budget)
 return data,counts,ts,ns,gu,down

def phase_duration(cycles,blocks,transfer):
 # Exact two-buffer recurrence for blocks with compute q+1 on the first r
 # blocks and q on the rest (q,r=divmod(cycles,blocks)).
 return max(cycles+transfer,blocks*transfer+cycles//blocks)

def cohort(counts,gu,down,bw=W_TBPS):
 k=sum(n>0 for n in counts)
 if not k:return dict(cycles=0,participants=0,extra_bytes=0,transfer_cycles=0)
 d=math.ceil(BLOCK*(k-1)/(bw*1000)) if bw else 0
 cg=max(int(n*g) for n,g in zip(counts,gu));cd=max(int(n*g) for n,g in zip(counts,down))
 t=phase_duration(cg,SHARED_GU//BLOCK,d)+phase_duration(cd,SHARED_DOWN//BLOCK,d)
 return dict(cycles=t,participants=k,extra_bytes=(k-1)*(SHARED_GU+SHARED_DOWN),transfer_cycles=d,
  shared_compute_critical_cycles=cg+cd,broadcast_exposed_cycles=t-cg-cd)

def replay(p,ns,gu,down,owner,bw=W_TBPS):
 n,nc=p.shape;sc=[0]*nc;rl=[0]*nc;service=[0]*nc
 for i,c in enumerate(owner):
  service[c]+=int(p[i,c])
  if i>=n-ns:sc[c]+=1
  else:rl[c]+=int(p[i,c])
 bc=cohort(sc,gu,down,bw);ends=[rl[c]+(bc['cycles'] if sc[c] else 0) for c in range(nc)]
 assert all(service[c]<=ends[c] for c in range(nc))
 return dict(cycles=max(ends),core_finish=ends,core_service=service,shared_counts=sc,routed_load=rl,**{'broadcast':bc})
