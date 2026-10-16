"""Zero-supply-delay PLENA compute with immutable Shared/Routed ownership.

Core0 executes Shared only; Core1 executes ALL routed experts, regardless of
their token count. No migration, preemption, stealing or cross-core partials
exist. Singles execute both populations serially. BLEN1 is a declared scalar
subarray extension priced with the same conservative feed/drain contract.
"""
from __future__ import annotations
import numpy as np
from research.moe_dispatch.plena65536_shared.timing import PROFILES,phase
from research.moe_dispatch.plena65536_moe.model import (balanced_score,expert_key,
    useful_macs,unique_weight_bytes,window_histograms,cost_table)
from research.moe_dispatch.plena65536_dedicated.geometry import Core,BUDGET


def job_cycles(core,expert,profile=PROFILES[0]):
    m,h,f=expert_key(expert)
    return phase(core,m,f,h,2,profile)+phase(core,m,h,f,1,profile)


def job_issued_macs(core,expert):
    m,h,f=expert_key(expert)
    ceil=lambda a,z:(a+z-1)//z
    active_feed=(2*ceil(m,core.pm)*ceil(f,core.pn)*ceil(h,core.r)*core.b
                 +ceil(m,core.pm)*ceil(h,core.pn)*ceil(f,core.r)*core.b)
    return active_feed*core.macs


def schedule(cores,window,profile=PROFILES[0]):
    """Strict affinity schedule; dual makespan is exactly max(role loads)."""
    cores=tuple(cores)
    if len(cores) not in (1,2) or sum(c.macs for c in cores)!=BUDGET:
        raise ValueError('One/two cores must total exactly65,536 multipliers')
    experts=window['experts']
    if len({int(e['id']) for e in experts})!=len(experts):
        raise ValueError('Every expert ID must appear exactly once')
    shared=[e for e in experts if e.get('is_shared',False)]
    routed=[e for e in experts if not e.get('is_shared',False)]
    if len(shared)!=1 or int(shared[0]['Me'])!=int(window['batch']):
        raise ValueError('Exactly one Shared expert must cover the complete batch')
    queues=[shared+routed] if len(cores)==1 else [shared,routed]
    jobs=[];loads=[];total_slots=0
    for owner,queue in enumerate(queues):
        now=0
        # Shape-only serial cost is order-invariant. Use expert ID order, not
        # any dynamic transfer/queue selection that could violate affinity.
        for expert in sorted(queue,key=lambda e:(not e.get('is_shared',False),int(e['id']))):
            duration=job_cycles(cores[owner],expert,profile)
            slots=job_issued_macs(cores[owner],expert)
            jobs.append({'expert_id':int(expert['id']),'is_shared':bool(expert.get('is_shared',False)),
                'core_id':owner,'M':int(expert['Me']),'H':int(expert['H']),'F':int(expert['F']),
                'start':now,'end':now+duration,'cycles':duration,'issued_macs':slots})
            now+=duration;total_slots+=slots
        loads.append(now)
    shared_work=sum(j['cycles'] for j in jobs if j['is_shared'])
    routed_work=sum(j['cycles'] for j in jobs if not j['is_shared'])
    useful=useful_macs(window)
    assert total_slots>=useful
    if len(cores)==2:
        assert loads==[shared_work,routed_work]
        assert all(j['core_id']==(0 if j['is_shared'] else 1) for j in jobs)
    makespan=max(loads)
    assert useful<=BUDGET*makespan
    return {'cycles':makespan,'core_cycles':loads,'shared_cycles':shared_work,'routed_cycles':routed_work,
        'jobs':jobs,'issued_macs':total_slots,'useful_MACs':useful,
        'unique_weight_bytes':unique_weight_bytes(window),
        'spatial_utilization':useful/total_slots,'wall_peak_utilization':useful/(BUDGET*makespan),
        'contract':'single_serial' if len(cores)==1 else 'Shared_C0_Routed_C1_hard_affinity',
        'profile':profile.name,'migration_count':0,'steal_count':0,'supply_delay_cycles':0}


def population_costs(cores,windows,profile=PROFILES[0]):
    """Exact core-by-window Shared and Routed loads, computed separately."""
    types,hist,weights=window_histograms(windows)
    shared_hist=np.zeros_like(hist)
    index={key:i for i,key in enumerate(types)}
    for wi,window in enumerate(windows):
        for e in window['experts']:
            if e.get('is_shared',False):shared_hist[wi,index[expert_key(e)]]+=1
    routed_hist=hist-shared_hist
    costs=cost_table(cores,types,profile)
    shared=costs@shared_hist.T
    routed=costs@routed_hist.T
    return shared,routed,weights
