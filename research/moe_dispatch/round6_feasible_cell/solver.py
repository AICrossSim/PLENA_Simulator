import ctypes,hashlib,math,subprocess
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
from .common import ROOT,R5,BLOCK,SEED,W_TBPS
from .model import replay
LIB=None
def library():
 global LIB
 if LIB is not None:return LIB
 src=ROOT/'kernel.cpp';h=hashlib.sha256(src.read_bytes()+(R5/'scheduler.cpp').read_bytes()).hexdigest()[:16]
 so=Path('/tmp')/f'plena-round6-{h}.so'
 if not so.exists():subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(src),'-o',str(so)],check=True)
 lib=ctypes.CDLL(str(so));D=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
 lib.schedule6.argtypes=[D,ctypes.c_int,ctypes.c_int,ctypes.c_int,D,D,I,ctypes.c_double,D,I]
 lib.grid6.argtypes=[D,D,D,ctypes.c_int,ctypes.c_int,ctypes.c_int,I,I,I,ctypes.c_int,ctypes.c_int,ctypes.c_double,D,D]
 LIB=lib;return lib
def fast(p,ns,gu,dn,types,bw=W_TBPS):
 p=np.ascontiguousarray(p,dtype=np.float64);gu=np.ascontiguousarray(gu,dtype=np.float64);dn=np.ascontiguousarray(dn,dtype=np.float64)
 typ=np.ascontiguousarray(types,dtype=np.int32);bounds=np.zeros(2);owner=np.zeros(len(p),np.int32)
 library().schedule6(p,len(p),p.shape[1],int(ns),gu,dn,typ,bw,bounds,owner)
 assert replay(p,ns,gu,dn,owner,bw)['cycles']==bounds[1]
 return max(1,math.ceil(bounds[0]-1e-7)),int(bounds[1]),owner.tolist()
def grid(data,counts,ns,gu,dn,indices,bw=W_TBPS):
 data=np.ascontiguousarray(data,np.float64);gu=np.ascontiguousarray(gu,np.float64);dn=np.ascontiguousarray(dn,np.float64)
 cfg=np.full((len(indices),16),-1,np.int32)
 for i,c in enumerate(indices):cfg[i,:len(c)]=c
 nshape,nw,nt=data.shape;lo=np.zeros((len(cfg),nw));hi=lo.copy()
 library().grid6(data,gu,dn,nw,nt,nshape,np.ascontiguousarray(counts,np.int32),np.ascontiguousarray(ns,np.int32),cfg,len(cfg),16,bw,lo,hi)
 lo=np.ceil(lo-1e-7).astype(np.int64);hi=hi.astype(np.int64)
 assert np.all(lo<=hi) and np.all(lo>0)
 return lo,hi

def optimize(p,ns,gu,dn,types,bw=W_TBPS,work=.1):
 p=np.asarray(p,dtype=np.int64);gu=np.asarray(gu,dtype=np.int64);dn=np.asarray(dn,dtype=np.int64);ns=int(ns);n,nc=p.shape
 lb,ub,owner=fast(p,ns,gu,dn,types,bw)
 if lb==ub:return dict(lower=lb,upper=ub,owner=owner,status='EXACT_BOUND')
 model=cp_model.CpModel();groups={}
 for i,row in enumerate(p[:n-ns]):groups.setdefault(tuple(map(int,row)),[]).append(i)
 x={};sc=[];active=[];rload=[];ends=[]
 z=model.NewIntVar(lb,ub,'layer_makespan')
 for gi,(cost,ids) in enumerate(groups.items()):
  for c in range(nc):
   x[gi,c]=model.NewIntVar(0,len(ids),f'x{gi}_{c}');model.AddHint(x[gi,c],sum(owner[i]==c for i in ids))
  model.Add(sum(x[gi,c] for c in range(nc))==len(ids))
 for c in range(nc):
  v=model.NewIntVar(0,ns,f'shared{c}');b=model.NewBoolVar(f'uses_shared{c}')
  model.Add(v>=1).OnlyEnforceIf(b);model.Add(v==0).OnlyEnforceIf(b.Not())
  model.AddHint(v,sum(owner[i]==c for i in range(n-ns,n)));sc.append(v);active.append(b)
  rload.append(sum(cost[c]*x[gi,c] for gi,cost in enumerate(groups)))
 model.Add(sum(sc)==ns)
 bound=max(ub,int(max(gu+dn))*ns)*4+1000000
 cg=model.NewIntVar(0,bound,'max_GU');cd=model.NewIntVar(0,bound,'max_Down')
 model.AddMaxEquality(cg,[int(gu[c])*sc[c] for c in range(nc)])
 model.AddMaxEquality(cd,[int(dn[c])*sc[c] for c in range(nc)])
 k=model.NewIntVar(1,nc,'receivers');model.Add(k==sum(active))
 # All supported bandwidth values are integer TB/s; integer rounding is exact.
 assert float(bw).is_integer()
 d=model.NewIntVar(0,1000000,'broadcast_block_cycles')
 if bw:
  rate=int(bw*1000);model.AddDivisionEquality(d,BLOCK*(k-1)+rate-1,rate)
 else:model.Add(d==0)
 phases=[]
 for cvar,blocks,label in [(cg,88,'G'),(cd,44,'D')]:
  q=model.NewIntVar(0,bound,label+'q');model.AddDivisionEquality(q,cvar,blocks)
  t=model.NewIntVar(0,bound,label+'time');model.AddMaxEquality(t,[cvar+d,blocks*d+q]);phases.append(t)
 bc=model.NewIntVar(0,bound,'cohort_end');model.Add(bc==sum(phases))
 for c in range(nc):
  extra=model.NewIntVar(0,bound,f'cohort_wait{c}')
  model.Add(extra==bc).OnlyEnforceIf(active[c]);model.Add(extra==0).OnlyEnforceIf(active[c].Not())
  end=model.NewIntVar(0,ub,f'end{c}');model.Add(end==rload[c]+extra);model.Add(end<=z);ends.append(end)
 for c in range(1,nc):
  if types[c]==types[c-1]:model.Add(ends[c-1]>=ends[c])
 model.AddMaxEquality(z,ends)
 model.Minimize(z);solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=SEED;solver.parameters.max_deterministic_time=work
 status=solver.Solve(model)
 assert status not in (cp_model.MODEL_INVALID,cp_model.INFEASIBLE),solver.StatusName(status)
 lb=max(lb,math.ceil(solver.BestObjectiveBound()-1e-7))
 if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
  new=[-1]*n
  for gi,ids in enumerate(groups.values()):
   off=0
   for c in range(nc):
    count=solver.Value(x[gi,c])
    for i in ids[off:off+count]:new[i]=c
    off+=count
  offset=n-ns
  for c in range(nc):
   count=solver.Value(sc[c]);new[offset:offset+count]=[c]*count;offset+=count
  val=replay(p,ns,gu,dn,new,bw)['cycles'];assert val==solver.Value(z)
  if val<=ub:ub=int(val);owner=new
 assert lb<=ub
 return dict(lower=int(lb),upper=int(ub),owner=owner,status=solver.StatusName(status))
