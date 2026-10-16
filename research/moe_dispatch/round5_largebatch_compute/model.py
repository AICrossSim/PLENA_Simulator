"""Ideal-supply flattened dot arrays, integer issue and recurrence timing.

This is an analytical architectural model, not RTL-calibrated physical timing.
The loop recurrence is inherited from round4; unlike round4, PM extends to64
and vector throughput may scale with the chip MAC budget. Ideal local operand
and accumulator supply. Retains eight OS output contexts per core.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
import math

@dataclass(frozen=True,order=True)
class Core:
    pm:int
    pn:int
    pk:int
    @property
    def macs(self):return self.pm*self.pn*self.pk
    @property
    def name(self):return f'{self.pm}x{self.pn}x{self.pk}'

def shapes():
    return tuple(Core(m,n,k) for m in (1,2,4,8,16,32,64)
                 for n in range(8,257,8) for k in (32,64,128,256,512)
                 if m*n*k>=4096 and m*n*k%4096==0)

def group(q,nk,L):return (nk-1)*max(q,L)+(q-1)+L
def projection(c,m,n,k,paired,flow):
    nm=(m+c.pm-1)//c.pm;nn=(n+c.pn-1)//c.pn;nk=(k+c.pk-1)//c.pk
    mult=2 if paired else 1;lat=3+2*(c.pk.bit_length()-1)
    if flow=='OS': comp=nm*sum(group(mult*min(8,nn-j),nk,lat) for j in range(0,nn,8))
    elif flow=='WS':comp=nn*group(mult*nm,nk,lat)
    elif flow=='IS':comp=nm*group(mult*nn,nk,lat)
    else:raise ValueError(flow)
    issued=mult*nm*nn*nk*c.macs
    return dict(cycles=max(comp,mult*nm*nn*nk),issued=issued,
                useful=mult*m*n*k,vector=(3 if paired else 2)*m*n,
                row_eff=m/(nm*c.pm),col_eff=n/(nn*c.pn),k_eff=k/(nk*c.pk))

@lru_cache(None)
def cost(c,m,f,mode='matrix',budget=65536,flow='AUTO'):
    if flow=='AUTO':
        return min((cost(c,m,f,mode,budget,s) for s in ('OS','WS','IS')),
                   key=lambda x:(x['cycles'],x['flow']))
    ps=[projection(c,m,f,2048,True,flow),projection(c,m,2048,f,False,flow)]
    # A throughput entitlement, not additional memory or unconstrained vector.
    lanes=c.macs/1024 if mode=='full_scaled' else 64*c.macs/budget
    durations=[p['cycles'] if mode=='matrix' else max(p['cycles'],math.ceil(p['vector']/lanes)) for p in ps]
    return dict(cycles=sum(durations),matrix_cycles=sum(p['cycles'] for p in ps),
                issued=sum(p['issued'] for p in ps),useful=3*m*2048*f,flow=flow,
                row_eff=m/(((m+c.pm-1)//c.pm)*c.pm),
                recurrence_cycles=sum(p['cycles']-p['issued']//c.macs for p in ps),
                vector_excess=sum(durations)-sum(p['cycles'] for p in ps))

def configurations(total):
    by={}
    for c in shapes():by.setdefault(c.macs,[]).append(c)
    for n in (1,2,4,8,16):
        for c in by.get(total//n,[]):yield ('single' if n==1 else f'homo{n}',(c,)*n)
    # All two-type heterogeneous pools with2..16 cores, all exact aligned
    # budgets, including unequal pool counts. No target ratio or shape favor.
    for n in range(2,17):
        for na in range(1,n):
            nb=n-na
            for ma,aa in sorted(by.items()):
                remain=total-na*ma
                if remain<=0 or remain%nb:continue
                mb=remain//nb
                if ma<mb:continue
                for a in aa:
                    for b in by.get(mb,[]):
                        if ma==mb and a>=b:continue
                        yield (f'hetero{n}',(a,)*na+(b,)*nb)

def geometry(cs):
    from collections import Counter
    return '+'.join(f'{n}*({c.name})' for c,n in Counter(cs).items())

def build_tasks(window,shared_chunk=0):
    out=[]
    for e in window['experts']:
        m=e['Me'];chunk=shared_chunk if e['is_shared'] and shared_chunk else m
        for start in range(0,m,chunk):
            out.append(dict(expert=e['id'],m=min(chunk,m-start),f=e['F'],shared=e['is_shared'],row_start=start))
    return out
