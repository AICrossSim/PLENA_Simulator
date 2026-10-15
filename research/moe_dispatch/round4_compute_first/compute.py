"""Pure-compute projection with explicit loop residency and traffic ledgers.

Uses round-three dot-array/FP32 recurrence timing, NOT PM*PN*G notation.
SRAM is unbounded, eight OS output records are retained. A core's vector
quota is installed independently; all quotas sum to 64 elements/cycle.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import math
from .config import Core, TOTAL_MACS, parameters
from . import model

FLOWS=('OS','WS','IS','PER_TASK')

@dataclass(frozen=True)
class IdealView:
    cores: tuple
    flows: tuple
    records: int=8
    landing_mode: str='private'
    x_bytes: tuple=(1<<40,)
    acc_bytes: tuple=(1<<40,)
    def effective_w_bytes(self,c): return 1<<40

@lru_cache(maxsize=None)
def cost(c,flow,m,h,f,lanes):
    if flow=='PER_TASK':
        return min((cost(c,x,m,h,f,lanes) for x in FLOWS[:3]),
                   key=lambda q:(q['cycles'],q['W_port']+q['X_port']+q['acc_port'],q['flow']))
    view=IdealView((c,),(flow,))
    p=parameters(control=False)
    ps=(model._projection(m,f,h,view,0,p,True),model._projection(m,h,f,view,0,p,False))
    durations=[max(s.compute_cycles,s.vector_elements/lanes) for s in ps]
    # OS/IS revisit weights across M waves, so retain a whole projection.
    # WS streams tiles with all rows sharing each tile; tile storage suffices.
    peakW=max((((2 if s.paired else 1)*model.ceildiv(s.n,c.pn)*model.ceildiv(s.k,c.pk)*s.w_tile_bytes)
               if flow!='WS' and model.ceildiv(m,c.pm)>1 else s.w_tile_bytes) for s in ps)
    peakX=max(s.peak_x_bytes for s in ps)
    peakA=max((s.peak_acc_bytes if flow!='IS' else (2 if s.paired else 1)*c.pm*s.n*4) for s in ps)
    peakZ=2*m*f
    ports={}
    for key,attr in [('W','w_sram_bytes'),('X','x_sram_bytes'),('acc','acc_sram_bytes')]:
        ports[key+'_port']=sum(getattr(s,attr) for s in ps)
        # Exact peak under the declared phase-fluid service. Physical burst
        # requirement is separately exposed; no claim of cycle-native peak.
        ports[key+'_peak']=max(getattr(s,attr)/t for s,t in zip(ps,durations))
    return dict(cycles=sum(durations),matrix_cycles=sum(s.compute_cycles for s in ps),flow=flow,
                macs=3*m*h*f,issued=sum(s.issued_macs for s in ps),
                W=peakW,X=peakX,acc=peakA,Z=peakZ,
                # Requirements at issue boundary before smoothing/banking.
                W_issue_peak=2*c.pn*c.pk,X_issue_peak=2*c.pm*c.pk,
                acc_issue_peak=(4 if flow=='OS' else 8)*c.pm*c.pn,**ports)

def quotas(total,weights):
    return model._partition(total,tuple(weights))

def pure(w,cores,flows,detail=True):
    lanes=quotas(64,[c.macs for c in cores]); n=len(cores)
    cs=[[cost(c,flows[j],e['Me'],e['H'],e['F'],lanes[j]) for j,c in enumerate(cores)] for e in w['experts']]
    # Declared deterministic compute scheduling, same for every organization.
    # Shared/Me>3 first, descending best service; no scheduling overhead in B.
    order=sorted(range(len(cs)),key=lambda i:(0,-min(x['cycles'] for x in cs[i]),i)
                 if w['experts'][i].get('is_shared') or w['experts'][i]['Me']>3 else (1,0,i))
    end=[0.0]*n; tasks=[]
    for i in order:
        j=min(range(n),key=lambda j:(end[j]+cs[i][j]['cycles'],j)); q=cs[i][j]
        start=end[j]; end[j]+=q['cycles']
        if detail: tasks.append(dict(expert_index=i,core=j,start=start,finish=end[j],**q))
    if not detail: return max(end)
    return dict(cycles=max(end),end=end,tasks=tasks)

def allocate_real(cores,flows,label=''):
    """Identical constructive allocation rule, no workload/heldout tuning.

    Minimum current W/X/paired Acc/Z per core, then distribute remaining
    1-KiB units equally among W/X/Acc/Z priorities 1/1/2/8. W and ports use
    equal shares across cores (not multiplier-proportional starvation).
    """
    n=len(cores); up=lambda x:(x+1023)//1024
    cap={
      'w_bytes':[up(c.w_slice_bytes)*1024 for c in cores],
      'x_bytes':[up(c.x_slice_bytes)*1024 for c in cores],
      'acc_bytes':[up(2*c.record_bytes)*1024 for c in cores],
      'z_bytes':[6*1024]*n}
    remain=532-sum(sum(v) for v in cap.values())//1024
    if remain<0: raise ValueError('minimum resident tiles exceed the 532-KiB capacity budget')
    # Additional W slots allow prefetch. Remaining Z reduces forced refetch.
    weights=[1]*n+[1]*n+[2]*n+[8]*n
    vals=[0]*len(weights)
    if remain>=len(weights): vals=list(quotas(remain,weights))
    else:
        for j in range(remain): vals[len(weights)-1-j]=1
    for j,key in enumerate(cap):
        cap[key]=tuple(v+1024*vals[j*n+i] for i,v in enumerate(cap[key]))
    return model.Design(tuple(cores),flows=tuple(flows),total_macs=TOTAL_MACS,label=label,
        **cap,w_banks=quotas(64,[1]*n),x_banks=quotas(24,[1]*n),
        acc_banks=quotas(12,[1]*n),vector_lanes=quotas(64,[c.macs for c in cores]))

def shape_id(cs): return '+'.join(f'{c.pm}x{c.pn}x{c.pk}' for c in cs)

def shape_dict(cs): return [asdict(c) for c in cs]
