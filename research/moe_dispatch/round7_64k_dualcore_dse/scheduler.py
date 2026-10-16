import ctypes
import hashlib
import subprocess
from pathlib import Path
from .common import ROOT

_lib = None
def library():
    global _lib
    if _lib is None:
        src=ROOT/'exact.cpp'; key=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        out=Path('/tmp')/('plena-r7-exact-'+key+'.so')
        if not out.exists():
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(out)],check=True)
        _lib=ctypes.CDLL(str(out));f=_lib.exact_assign;f.restype=ctypes.c_int64
        f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int64)]
    return _lib

def optimal(costs):
    n=len(costs)
    if not n or any(len(p) != len(costs[0]) for p in costs): raise ValueError('Invalid cost matrix')
    if len(costs[0])==1:
        return dict(cycles=sum(p[0] for p in costs),owner=[0]*n,
                    loads=[sum(p[0] for p in costs)],status='OPTIMAL',frontier_peak=1)
    if len(costs[0])!=2: raise ValueError('Only one or two cores')
    a=(ctypes.c_int64*n)(*(p[0] for p in costs));b=(ctypes.c_int64*n)(*(p[1] for p in costs))
    mask=(ctypes.c_uint64*2)();details=(ctypes.c_int64*3)()
    t=int(library().exact_assign(n,a,b,mask,details))
    if t<0: raise ValueError('Invalid exact assignment input')
    bits=int(mask[0]) | int(mask[1])<<64
    owner=[0 if bits & (1<<i) else 1 for i in range(n)]
    loads=[sum(p[c] for p,c in zip(costs,owner) if c==j) for j in range(2)]
    assert loads==list(details)[:2] and max(loads)==t
    return dict(cycles=t,owner=owner,loads=loads,status='OPTIMAL',frontier_peak=int(details[2]))

def greedy(costs):
    load=[0]*len(costs[0]);owner=[-1]*len(costs)
    for i in sorted(range(len(costs)),key=lambda i:(-min(costs[i]),i)):
        c=min(range(len(load)),key=lambda c:(load[c]+costs[i][c],c))
        owner[i]=c;load[c]+=costs[i][c]
    return dict(cycles=max(load),owner=owner,loads=load)
