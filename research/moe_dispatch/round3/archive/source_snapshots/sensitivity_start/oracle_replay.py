"""Conditional timing oracle: physically replay a frozen admitted action plan.

Owner, order, bind/release and accepted prefetch times come from pass one. This
module recomputes shared-resource service and finishes, never forcing recorded
finish times. It is an accuracy reference for that plan, not an optimal or
clairvoyant dispatcher. Counterfactual unchosen-core times are not claimed.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import hashlib,heapq,json,math
from .model import (PhaseCost,_fluid_rates,_window_bandwidth,ceildiv,
                    _set_window_limits,_pool_can_reserve_next,_projection)

SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _single(plan,design,p):
    phases=plan['phases'];tasks=plan['tasks'];n=len(design.cores)
    caps={'hbm':p.hbm_bandwidth,'vector':64*p.vector_scale,'control':1.0}
    for c in range(n):
        caps.update({('compute',c):1.0,('W',c):p.w_bandwidth(design,c),
                     ('X',c):design.x_banks[c]*p.bank_Bpc,
                     ('acc',c):design.acc_banks[c]*p.bank_Bpc,
                     ('vector',c):design.vector_lanes[c]*p.vector_scale})
    bytask={t['expert_index']:t for t in tasks};phase0={ph['expert_index']:ph for ph in phases if ph['phase']==0}
    bindings={b['expert_index']:b for b in plan['bindings']}
    if set(bindings)!=set(bytask):raise AssertionError('task/binding identity mismatch')
    names={f.name for f in fields(PhaseCost)}
    for ph in phases:
        declared={k:ph[k] for k in names}
        checked=_projection(ph['m'],ph['n'],ph['k'],design,ph['core'],p,ph['paired'])
        actual={f.name:getattr(checked,f.name) for f in fields(PhaseCost)}
        if declared != actual:
            raise AssertionError('frozen phase demands do not match installed cost model')
    for t in tasks:
        b=bindings[t['expert_index']]
        if b['core']!=t['core'] or t['start']<b['bind_cycle']-1e-7:
            raise AssertionError('unbound or wrong-owner frozen task')
        request=t.get('prefetch_request')
        if request is not None:
            predecessors=[x for x in tasks if x['core']==t['core'] and x['start']<t['start'] and x['start']<=request+1e-7 and x['finish']>=request-1e-7]
            if request<b['bind_cycle']-1e-7 or request>t['start']+1e-7 or not predecessors:
                raise AssertionError('unrequested/unadmitted frozen prefetch')
            prev=max(predecessors,key=lambda x:x['start'])
            current=[ph for ph in phases if ph['expert_index']==prev['expert_index'] and ph['start']<=request+1e-7 and ph['finish']>=request-1e-7]
            if not current or max(ph['w_slots']-ph['weight_live_slots'] for ph in current)<1:
                raise AssertionError('prefetch grant lacks a legal active Current W slot')
    expected_prefetched=sum(ceildiv(phase0[t['expert_index']]['first_weight_bytes'],32)*32 for t in tasks if t.get('prefetch_request') is not None)
    if expected_prefetched!=plan.get('prefetched_bytes',expected_prefetched):raise AssertionError('unexpected frozen prefetch bytes')
    pending=[];serial=0;reserved={};active={};done={};ready={};actualbytes=0.
    pool_checks=[]
    def event(at,kind,obj):
        nonlocal serial
        serial+=1;heapq.heappush(pending,(float(at),serial,kind,obj))
    for t in tasks:
        request=t.get('prefetch_request')
        if request is None:continue
        ph=phase0[t['expert_index']];s=PhaseCost(**{k:ph[k] for k in names});b=ceildiv(s.first_weight_bytes,32)*32
        pf={'task':t['expert_index'],'core':t['core'],'bytes':b,'wbytes':b+s.w_tile_bytes,'slot':s.w_tile_bytes}
        event(request,'reserve',pf);event(request+p.hbm_latency,'prefetch',pf)
        event(ph['start'],'release',pf)
    # Frozen phase rows are emitted in completion order; chronological issue
    # order with deterministic core tie-breaking restores initial actor order.
    # That order matters for equal-deadline common-pool admission.
    for ph in sorted(phases,key=lambda q:(q['start'],q['core'],q['phase'],q['expert_index'])):
        event(ph['start'],'run',ph)
    now=0.
    while pending or active:
        # Process completions before same-cycle consumer releases.
        while pending and pending[0][0]<=now+1e-7:
            at,_,kind,obj=heapq.heappop(pending);c=obj['core'];key=(kind,c)
            if kind=='reserve':
                if c in reserved:raise AssertionError('more than one frozen Next reservation')
                if design.landing_mode=='shared' and not _pool_can_reserve_next(
                        design,{j:{'slot_bytes':v} for j,v in reserved.items()},obj['slot']):
                    raise AssertionError('frozen Next overcommits shared landing pool')
                reserved[c]=obj['slot'];continue
            if kind=='release':reserved.pop(c,None);continue
            if key in active:raise AssertionError('overlapping frozen same-core actor')
            if kind=='prefetch':
                active[key]={'task':obj['task'],'phase':None,'core':c,'cost':None,'remaining':1.,'limit':math.inf,
                             'demand':{'hbm':obj['bytes'],('W',c):obj['wbytes']}}
            else:
                s=PhaseCost(**{k:obj[k] for k in names});t=bytask[obj['expert_index']]
                subtract=ceildiv(s.first_weight_bytes,32)*32 if obj['phase']==0 and t.get('prefetch_request') is not None else 0
                demand={'hbm':s.hbm_bytes-subtract,('W',c):s.w_sram_bytes-(subtract+s.w_tile_bytes if subtract else 0),
                        'vector':s.vector_elements,('vector',c):s.vector_elements,('compute',c):s.compute_cycles,
                        ('X',c):s.x_sram_bytes,('acc',c):s.acc_sram_bytes,'control':s.control_cycles if p.charge_control else 0}
                if any(v<0 for v in demand.values()):raise AssertionError('negative replay demand')
                actor={'task':obj['expert_index'],'phase':obj['phase'],'core':c,'cost':s,'remaining':1.,'limit':math.inf,'demand':demand}
                if obj['phase']==0 and not subtract:
                    actor['prefix']=min(1.,max(s.first_weight_bytes/max(1.,demand['hbm']),
                                              (s.first_weight_bytes+s.w_tile_bytes)/max(1.,demand[('W',c)])))
                active[key]=actor
        if not active:
            if pending:now=max(now,pending[0][0]);continue
            break
        ledger=_set_window_limits(active,{c:{'slot_bytes':v} for c,v in reserved.items()},design,p,now)
        if ledger is not None:
            pool_checks.append(ledger['used_bytes'])
        rates=_fluid_rates(active,caps)
        dt=min(actor['remaining']/max(1e-300,rates[key]) for key,actor in active.items())
        for key,actor in active.items():
            if 'prefix' in actor:
                need=actor['prefix']-(1-actor['remaining'])
                if need>1e-9:dt=min(dt,need/max(1e-300,rates[key]))
        if pending:dt=min(dt,max(0.,pending[0][0]-now))
        if not math.isfinite(dt) or dt>1e20:raise AssertionError('frozen replay deadlock')
        if dt<=1e-10:dt=1e-7
        actualbytes+=sum(rates[k]*v['demand'].get('hbm',0) for k,v in active.items())*dt
        for k,v in active.items():v['remaining']-=rates[k]*dt
        now+=dt
        for k,v in list(active.items()):
            if 'prefix' in v and 1-v['remaining']>=v['prefix']-1e-9:
                ready[v['task']]=now;del v['prefix']
            if v['remaining']>1e-8:continue
            if k[0]=='prefetch':ready[v['task']]=now
            else:done[v['task'],v['phase']]=now
            del active[k]
    delta=max((abs(done[ph['expert_index'],ph['phase']]-ph['finish']) for ph in phases),default=0.)
    tolerance=max(1e-5,float(plan['cycles'])*1e-10)
    if delta>tolerance:raise AssertionError(('physical replay phase timing mismatch',delta,tolerance))
    if abs(actualbytes-plan['hbm_bytes'])>max(1.,plan['hbm_bytes']*1e-8):raise AssertionError('physical replay HBM conservation mismatch')
    result=deepcopy(plan)
    for t in result['tasks']:
        fin=max(f for (i,pi),f in done.items() if i==t['expert_index'])
        t['predicted_cycles']=bytask[t['expert_index']]['actual_cycles']
        t['actual_cycles']=fin-t['start'];t['finish']=fin;t['predicted_finish']=t['start']+t['predicted_cycles']
        t['first_weight_ready']=ready[t['expert_index']]
        delta=max(delta,abs(fin-bytask[t['expert_index']]['finish']),abs(ready[t['expert_index']]-bytask[t['expert_index']]['first_weight_ready']))
    computed=max((t['finish'] for t in result['tasks']),default=0.)
    if abs(computed-plan['cycles'])>tolerance:raise AssertionError('physical replay wall timing mismatch')
    if delta>tolerance:raise AssertionError(('physical replay task/prefix timing mismatch',delta,tolerance))
    result['cycles']=computed;result['latency_ms']=computed/1e6
    return result,delta,actualbytes, max(pool_checks,default=0)


def same_schedule_oracle(plan,design,params):
    """Second physical pass; all first-pass actions remain frozen."""
    original=deepcopy(plan);children=plan.get('chunk_results')
    if children:
        result=deepcopy(plan);offset=0.;replayed=[];deltas=[];bytecount=0.;poolmax=0
        for child in children:
            normalized=deepcopy(child)
            for t in normalized['tasks']:
                for f in ('start','finish','predicted_finish','first_weight_ready','current_end','prefetch_request'):
                    if t.get(f) is not None:t[f]-=offset
            for b in normalized['bindings']:
                for f in ('bind_cycle','predicted_finish'):
                    if b.get(f) is not None:b[f]-=offset
            for ph in normalized['phases']:
                for f in ('start','stream_finish','finish'):ph[f]-=offset
            io=child['activation_spill_bytes'];spilltime=params.hbm_latency+io/params.hbm_bandwidth
            normalized['cycles']-=spilltime;normalized['hbm_bytes']-=io
            r,delta,b,pool=_single(normalized,design,params);deltas.append(delta);bytecount+=b+io;poolmax=max(poolmax,pool)
            for t in r['tasks']:
                for f in ('start','finish','predicted_finish','first_weight_ready','current_end','prefetch_request'):
                    if t.get(f) is not None:t[f]+=offset
            replayed+=r['tasks'];offset+=r['cycles']+spilltime
        result['tasks']=replayed;result['cycles']=offset;result['latency_ms']=offset/1e6;delta=max(deltas,default=0.)
    else:result,delta,bytecount,poolmax=_single(plan,design,params)
    relative=[abs(t['predicted_cycles']-t['actual_cycles'])/t['actual_cycles'] for t in result['tasks']]
    result['oracle_replay']={'kind':'conditional_same_actual_schedule','scope':'frozen ours admitted actions; accuracy reference, not a scheduling upper bound',
        'plan_sha256':_sha(original),'source_sha256':SOURCE_SHA256,
        'max_timing_difference_cycles':delta,'hbm_replay_bytes':bytecount,'hbm_expected_bytes':original['hbm_bytes'],
        'max_hbm_difference_bytes':abs(bytecount-original['hbm_bytes']),
        'max_pool_used_bytes':poolmax,'pool_capacity_bytes':design.landing_pool_bytes,
        'mae_pct':100*sum(relative)/len(relative) if relative else 0.,'physically_replayed':True,
        'physical_scope':'independent second phase-fluid pass, not native tile/request replay',
        'observations_scope':'rate/counter observations retained from matched first-pass plan; finishes/prefix physically recomputed'}
    return result
