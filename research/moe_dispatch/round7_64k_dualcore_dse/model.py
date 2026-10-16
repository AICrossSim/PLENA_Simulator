"""Reuse the frozen OS projection recurrence; also evaluate the exact task formula.

G is reduction elements per issue, NOT the older scaling_g64 MLEN/BLEN ratio.
Gate/Up/Down are three independent projections. Vector and output merge time
are excluded as requested. Operand bytes count issue slots, not stall cycles.
"""
from dataclasses import dataclass
from functools import lru_cache
from research.moe_dispatch.round5_largebatch_compute.model import Core as OldCore, projection
from .common import CFG

@dataclass(frozen=True, order=True)
class Core:
    pm: int
    pn: int
    g: int = 64
    @property
    def macs(self): return self.pm*self.pn*self.g
    @property
    def shape(self): return f'{self.pm}x{self.pn}x{self.g}'
    @property
    def bytes_per_issue(self): return (self.pn+self.pm)*self.g*CFG['element_bytes']
    @property
    def peak_feed(self): return self.bytes_per_issue*CFG['clock_hz']/1e12

def shapes(macs, cap):
    return tuple(Core(macs//(n*CFG['g']), n, CFG['g'])
                 for n in range(CFG['pn_step'], cap+1, CFG['pn_step'])
                 if macs % (n*CFG['g']) == 0)

def configurations(cap):
    for c in shapes(CFG['total_macs'], cap): yield 'single', (c,)
    for c in shapes(CFG['total_macs']//2, cap): yield 'homo', (c,c)
    for small in CFG['small_core_macs']:
        for a in shapes(CFG['total_macs']-small, cap):
            for b in shapes(small, cap):
                if small*2 == CFG['total_macs'] and a >= b: continue
                yield 'hetero', (a,b)

def references():
    return [('PLENA_32x32x64_reference', (Core(32,32),)),
            ('old_single', (Core(4,256),)),
            ('old_homo', (Core(2,256),)*2),
            ('old_hetero', (Core(4,128), Core(1,512))),
            ('old_stream_homo', (Core(1,512),)*2)]

def geometry(cs): return '+'.join(c.shape for c in cs)

def gemm(c,m,k,n,model,overhead):
    issues=((m+c.pm-1)//c.pm)*((n+c.pn-1)//c.pn)*((k+c.g-1)//c.g)
    if model == 'issue_formula': base=issues
    elif model == 'plena_os':
        q=projection(OldCore(c.pm,c.pn,c.g),m,n,k,False,'OS')
        base=q['cycles']; assert q['issued'] == issues*c.macs
    else: raise ValueError(model)
    return dict(cycles=base+overhead, issues=issues, pipeline_cycles=base-issues,
                sram_bytes=issues*c.bytes_per_issue, useful_macs=m*k*n)

@lru_cache(None)
def task_cost(c,m,model,overhead,f=1408):
    gs=[gemm(c,m,2048,f,model,overhead),gemm(c,m,2048,f,model,overhead),
        gemm(c,m,f,2048,model,overhead)]
    return {k:sum(x[k] for x in gs) for k in gs[0]}
