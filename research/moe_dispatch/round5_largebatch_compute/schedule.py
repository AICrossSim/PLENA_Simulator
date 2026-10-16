import ctypes, hashlib, math, subprocess
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model
from .common import ROOT

def library():
    src=ROOT/'scheduler.cpp';h=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    so=Path('/tmp')/f'plena-round5-scheduler-{h}.so'
    if not so.exists():subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(src),'-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so));d=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    lib.schedule.argtypes=[d,ctypes.c_int,ctypes.c_int,i,d,i]
    lib.grid.argtypes=[d,ctypes.c_int,ctypes.c_int,ctypes.c_int,i,i,ctypes.c_int,ctypes.c_int,d,d]
    return lib
LIB=None
def fast(p,types):
    global LIB
    if LIB is None:LIB=library()
    p=np.ascontiguousarray(p,dtype=np.float64);types=np.ascontiguousarray(types,dtype=np.int32)
    bounds=np.zeros(2);owner=np.zeros(len(p),np.int32)
    LIB.schedule(p,len(p),p.shape[1],types,bounds,owner)
    return bounds[0],bounds[1],owner

def grid(costs,counts,configs):
    global LIB
    if LIB is None:LIB=library()
    costs=np.ascontiguousarray(costs,dtype=np.float64);counts=np.ascontiguousarray(counts,dtype=np.int32)
    cfg=np.full((len(configs),16),-1,np.int32)
    for i,c in enumerate(configs):cfg[i,:len(c)]=c
    ns,nw,nt=costs.shape;lo=np.zeros((len(cfg),nw));hi=lo.copy()
    LIB.grid(costs,nw,nt,ns,counts,cfg,len(cfg),16,lo,hi)
    assert np.all(lo<=hi+1e-6)
    return lo,hi

def optimize(p,types,work=0.04):
    """Integer assignment with deterministic effort and explicit optimality gap.

    Group identical task cost vectors; per-core group counts are exact, no
    timing quantization. Identical cores get load-order symmetry breaking.
    """
    p=np.asarray(p,dtype=np.int64);lb,ub,owner=fast(p,types)
    if math.ceil(lb-1e-7)>=int(ub):return dict(lower=int(ub),upper=int(ub),owner=owner.tolist(),status='BOUND_EXACT')
    groups={}
    for i,row in enumerate(p):groups.setdefault(tuple(map(int,row)),[]).append(i)
    model=cp_model.CpModel();vs={};loads=[];z=model.NewIntVar(math.ceil(lb-1e-7),int(ub),'makespan')
    for g,(cost,ids) in enumerate(groups.items()):
        for c,t in enumerate(cost):
            v=model.NewIntVar(0,len(ids),f'x{g}_{c}');vs[g,c]=v
            model.AddHint(v,sum(owner[i]==c for i in ids))
        model.Add(sum(vs[g,c] for c in range(p.shape[1]))==len(ids))
    for c in range(p.shape[1]):
        load=sum(int(cost[c])*vs[g,c] for g,cost in enumerate(groups));model.Add(load<=z);loads.append(load)
    # The incumbent need not follow this symmetry order; the solver may drop
    # hints, but no feasible objective value is removed by permuting cores.
    for c in range(1,p.shape[1]):
        if types[c]==types[c-1]:model.Add(loads[c-1]>=loads[c])
    model.Minimize(z);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.random_seed=20261016
    s.parameters.max_deterministic_time=work
    status=s.Solve(model);lower=max(math.ceil(lb-1e-7),math.ceil(s.BestObjectiveBound()-1e-7))
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        new=np.zeros(len(p),np.int32)
        for g,ids in enumerate(groups.values()):
            off=0
            for c in range(p.shape[1]):
                count=s.Value(vs[g,c]);new[ids[off:off+count]]=c;off+=count
            assert off==len(ids)
        val=max(sum(int(p[i,c]) for i in range(len(p)) if new[i]==c) for c in range(p.shape[1]))
        if val<=ub:ub=val;owner=new
    assert lower<=ub+1e-6
    return dict(lower=int(lower),upper=int(ub),owner=owner.tolist(),status=s.StatusName(status),deterministic_time=s.ResponseProto().deterministic_time)

def online(p,tasks,types,window=8,dispatch_cycles=1,preconfig=True):
    """Causal list scheduler, known routed shapes, finite lookahead, no oracle.

    Global dispatch accepts one descriptor/cycle. Current+Next limit per core.
    At every acceptance compare predicted queue drain plus candidate service.
    Next context preconfiguration hides an assumed2-cycle context setup if
    the queued task has at least2 cycles until its core becomes available.
    """
    p=np.asarray(p)
    # Router completion reveals all task shapes; an initial longest-estimated
    # task ordering is causal and common to every organization. Only the first
    # eight pending descriptors participate in each dispatch decision.
    pending=sorted(range(len(tasks)),key=lambda i:(-int(min(p[i])),i))
    nc=p.shape[1];end=[0]*nc;queues=[[] for _ in range(nc)];owners=[-1]*len(tasks);starts=[0]*len(tasks)
    now=0
    while pending:
        for q in queues:
            while q and q[0]<=now:q.pop(0)
        eligible=[c for c in range(nc) if len(queues[c])<2]
        if not eligible:now=min(q[0] for q in queues if q);continue
        choices=[]
        for i in pending[:window]:
            for c in eligible:
                start=max(now+dispatch_cycles,end[c]);setup=max(0,2-(start-now-dispatch_cycles)) if preconfig else 2
                choices.append((start+setup+int(p[i,c]),-int(min(p[i])),i,c,start+setup))
        finish,_,i,c,start=min(choices);pending.remove(i);owners[i]=c;starts[i]=start;end[c]=finish;queues[c].append(finish);now+=dispatch_cycles
    return dict(cycles=max(end),owner=owners,starts=starts,end=end)
