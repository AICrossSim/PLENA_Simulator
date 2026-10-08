"""Capacity-aware dispatch only; physical cost and phase-fluid services unchanged.

The dual-core event engine is copied from frozen round2.model. Only descriptor
ordering and binding decisions are changed. All task costs, resource checks,
phase transitions, HBM arbitration, and prefetch execution use the frozen
implementation. Single-core runs delegate directly for bit-exact regression.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import heapq, inspect, math, random
from typing import Iterable
from . import model as frozen
from .model import (Core, Design, Parameters, TaskCost, SCOPE, ceildiv,
                     task_cost, storage_chunks, _prediction, _fluid_rates,
                     _window_bandwidth, _set_window_limits, _supply_segment, _idle_segment,
                     _pool_can_reserve_next)

@dataclass(frozen=True)
class Candidate:
    core: int
    predicted_cycles: float
    earliest_start: float
    refetch: float
    admissible: bool = True
    extra_hbm_bytes: int = 0
    contention_cycles: float = 0.0

    @property
    def finish(self):
        return self.earliest_start+self.predicted_cycles

    @property
    def scored_finish(self):
        # predicted_cycles already includes this task's refetch demand.
        # Only the externality on OTHER concurrent users of the shared bus
        # is charged here. This affects selection, never the actual service.
        return self.finish+self.contention_cycles

@dataclass(frozen=True)
class Decision:
    chosen: Candidate | None
    deferred: tuple[dict, ...] = ()


def refetch_factor(cost: TaskCost) -> float:
    """Actual wire demand including spill / unique padded expert weights."""
    return cost.hbm_bytes/cost.unique_hbm_bytes


def choose_candidate(candidates: Iterable[Candidate], now: float=0.0, *, enhanced: bool=True) -> Decision:
    """Compare currently admissible execution with waiting for a no-refetch core.

    Future ETA is an estimate, never a resource grant. If all physical cores
    refetch, the exact requested rule leaves ordinary EFT in effect.
    """
    cs=list(candidates)
    if enhanced and cs and all(c.refetch>1.0 for c in cs):
        minimum=min(c.refetch for c in cs)
        # If the least-refetch core is not currently admissible, leave the
        # task pending rather than bind it permanently to a worse core.
        cs=[c for c in cs if c.refetch==minimum]
    clean=[c for c in cs if c.refetch == 1.0]
    accepted=[];deferred=[]
    for c in cs:
        if not c.admissible:
            continue
        if c.refetch>1.0 and clean:
            other=min((q for q in clean if q.core!=c.core),
                      key=lambda q:((q.scored_finish if enhanced else q.finish),q.core),default=None)
            if other is not None and (other.scored_finish if enhanced else other.finish)<=(c.scored_finish if enhanced else c.finish):
                deferred.append({"rejected_core":c.core,"wait_core":other.core,
                    "wait_finish":other.finish,"immediate_finish":c.finish,
                    "refetch_factor":c.refetch,"other_admissible":other.admissible})
                continue
        accepted.append(c)
    return Decision(min(accepted,key=lambda q:((q.scored_finish if enhanced else q.finish),q.core),default=None),tuple(deferred))


def priority_order(experts, estimated_cycles, t_big: int, large_first: bool=True):
    if t_big not in (2,3,4,6,8):
        raise ValueError("t_big must be one of 2,3,4,6,8")
    order=list(range(len(experts)))
    if large_first:
        def key(i):
            big=bool(experts[i].get("is_shared",False)) or experts[i]["Me"]>t_big
            return (0,-estimated_cycles[i],i) if big else (1,0,i)
        order.sort(key=key)
    return order


def _one_layer_fixed(workload: dict, design: Design, params: Parameters, policy: str,
               owners: tuple[int, ...] | None, predictor, detail: bool,
               cost_overrides: dict | None=None, shape_overrides: dict | None=None, *, t_big: int=2, large_first: bool=True, enhanced_dispatch: bool=True) -> dict:
    es = workload["experts"]
    if not es:
        return {"cycles":0.0,"latency_ms":0.0,"hbm_bytes":0,"useful_macs":0,"issued_macs":0,
                "tasks":[],"bindings":[],"phases":[],"core_finish_cycles":[0.0]*len(design.cores)}
    ncores = len(design.cores)
    cost_overrides=cost_overrides or {}
    shape_overrides=shape_overrides or {}
    if (cost_overrides or shape_overrides) and ncores!=1:
        raise ValueError("U1 cost/shape overrides are single-core only")
    def co_for(i,c):
        return cost_overrides[i] if i in cost_overrides else task_cost(es[i],design,c,params)
    def actual_core(i,c):
        return shape_overrides.get(i,design.cores[c])
    costs, predicted, queues = {}, {}, [[] for _ in range(ncores)]
    availability = [0.0]*ncores
    bindings = []
    binding_ready = {}
    rng = random.Random(params.seed)
    ctrl = 0.0
    order = list(range(len(es)))
    if owners is not None:
        if len(owners) != len(es):
            raise ValueError("one owner per expert required")
        # The solver's assignment is made executable in LPT order.
        order.sort(key=lambda i: (-co_for(i,int(owners[i])).isolated_cycles,i))
    if owners is None and large_first:
        estimates=[]
        for i,e in enumerate(es):
            cs=[]
            for c in range(ncores):
                try:
                    co=co_for(i,c)
                except ValueError:
                    continue
                costs[i,c]=co
                cs.append(_prediction(predictor,e,c,co.isolated_cycles,design,params))
            if not cs:
                raise ValueError(f"expert {i} has no legal installed core")
            estimates.append(min(cs))
        order=priority_order(es,estimates,t_big,True)
        # Charge descriptor estimates and an upper-bound sorting comparison
        # count in the existing 1-op/cycle controller. No operand is moved.
        big_count=sum(e.get("is_shared",False) or e["Me"]>t_big for e in es)
        if params.charge_control:
            ctrl += 2*ncores*len(es)+big_count*math.ceil(math.log2(max(1,big_count)))
    dispatch_decisions=[]
    threshold = 2
    if policy.startswith("threshold_"):
        try:
            threshold = int(policy.split("_")[-1])
        except ValueError:
            raise ValueError("threshold policy needs integer suffix")
    if policy == "adaptive":
        vals = sorted(e["Me"] for e in es if not e.get("is_shared",False))
        threshold = vals[len(vals)//2] if vals else 2
    # Fixed solver assignments use a declared LPT plan. Runtime policies bind
    # online below; no whole-layer free assignment or unbounded core FIFO.
    for i in (order if owners is not None else ()):
        options = []
        for c in range(ncores):
            try:
                co = co_for(i,c)
            except ValueError:
                continue
            costs[i,c] = co
            pred = _prediction(predictor,es[i],c,co.isolated_cycles,design,params)
            predicted[i,c] = pred
            options.append((c,pred))
        if not options:
            raise ValueError(f"expert {i} has no legal installed core")
        legal = len(options)
        if owners is not None:
            options = [o for o in options if o[0] == owners[i]]
            if not options:
                raise ValueError("fixed owner is infeasible")
        elif (policy.startswith("threshold_") or policy=="adaptive") and ncores==2:
            stream = min(range(ncores),key=lambda c:(design.cores[c].pm,design.cores[c].macs,c))
            preferred = stream if es[i]["Me"]<=threshold and not es[i].get("is_shared",False) else 1-stream
            selected = [o for o in options if o[0]==preferred]
            if "fallback" in policy and selected:
                fallback = min(options,key=lambda o:(availability[o[0]]+o[1],o[0]))
                selected = [fallback] if availability[fallback[0]]+fallback[1] < availability[preferred]+selected[0][1] else selected
            options = selected or options
        if policy == "random" and owners is None:
            chosen = options[rng.randrange(len(options))]
        elif policy in ("idle","greedy") and owners is None:
            chosen = min(options,key=lambda o:(availability[o[0]],o[0]))
        elif policy in ("round_robin","rr") and owners is None:
            chosen = min(options,key=lambda o:(o[0]!=i%ncores,o[0]))
        else:
            chosen = min(options,key=lambda o:(availability[o[0]]+o[1],o[0]))
        c,pred = chosen
        ctrl += (2*legal+2) if params.charge_control else 0
        bindings.append({"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                         "legal_cores":legal,"predicted_cycles":pred,"nominal_cycles":costs[i,c].isolated_cycles,"bind_cycle":ctrl,
                         "predicted_finish":availability[c]+pred})
        binding_ready[i] = ctrl
        availability[c] += pred
        queues[c].append(i)
    # Only an eight-entry control window is installed. Layer descriptors live
    # in the explicitly budgeted route store; no whole expert matrix is queued.
    batch = workload["batch"]
    hidden = workload.get("hidden",es[0].get("H",2048))
    clear_bytes=batch*hidden*4
    clear = clear_bytes/(12*params.bank_Bpc)
    starts = clear+ctrl
    pending, serial, active = [],0,{}
    event_keys = set()
    positions = [0]*ncores
    finishes = [starts]*ncores
    task_started, tasks = {},{}
    previous_completion = [None]*ncores
    complete_accepts_nominal = False
    if predictor is not None and hasattr(predictor,"on_complete"):
        sig=inspect.signature(predictor.on_complete)
        complete_accepts_nominal = "nominal" in sig.parameters or any(
            q.kind==inspect.Parameter.VAR_KEYWORD for q in sig.parameters.values())
    waiting = [] if owners is not None else list(order)
    eta_end = {}
    task_done_work = {}
    task_next_quarter = {}
    next_state = {}       # at most one reserved Next block per core
    phase_started = {}
    phase_rows = []
    stream_attribution = [dict() for _ in range(ncores)]
    segments = []
    busy = {"hbm":0.0,"W":0.0,"X":0.0,"acc":clear,"vector":0.0}
    core_busy = [0.0]*ncores
    actual_bytes = 0.0
    prefetched_bytes = 0.0
    wasted_prefetch = 0
    capacities = {"hbm":params.hbm_bandwidth,
                  "vector":64*params.vector_scale,"control":1.0}
    for c in range(ncores):
        capacities[("compute",c)] = 1.0
        capacities[("W",c)] = params.w_bandwidth(design,c)
        capacities[("X",c)] = design.x_banks[c]*params.bank_Bpc
        capacities[("acc",c)] = design.acc_banks[c]*params.bank_Bpc
        capacities[("vector",c)] = design.vector_lanes[c]*params.vector_scale
    def event(at,kind,c,i,pi=0):
        nonlocal serial
        key=(round(float(at),6),kind,c,i,pi)
        if key in event_keys:
            return
        event_keys.add(key)
        serial+=1
        heapq.heappush(pending,(float(at),serial,kind,c,i,pi))
    def capacity_to_bind(c, co, next_i):
        outstanding=len(queues[c])-positions[c]
        if outstanding>=2:
            return False
        if not outstanding:
            return True
        current=costs[queues[c][positions[c]],c]
        # Reserve a true first W/X/output/Z context for Next. Current keeps
        # enough installed resources to finish; predictions never free slots.
        wp=max(s.weight_live_slots*s.w_tile_bytes for s in current.phases)
        xp=max(s.peak_x_bytes for s in current.phases)
        ap=max(s.peak_acc_bytes for s in current.phases)
        zp=max(s.m*s.n*2 for s in current.phases if s.paired)
        first=co.phases[0]
        return ((wp+first.w_tile_bytes<=design.effective_w_bytes(c) and
                (design.landing_mode == "private" or _pool_can_reserve_next(design,next_state,first.w_tile_bytes))) and
                xp+actual_core(next_i,c).x_slice_bytes<=design.x_bytes[c] and
                ap+2*actual_core(next_i,c).record_bytes<=design.acc_bytes[c] and
                zp+first.n*2<=design.z_bytes[c])
    def try_bind(at):
        nonlocal ctrl
        if owners is not None:
            return
        progress=True
        while progress and waiting:
            progress=False
            for i in waiting[:8]:
                physical=[]
                for c in range(ncores):
                    try:
                        co=co_for(i,c)
                    except ValueError:
                        continue
                    costs[i,c]=co
                    physical.append(c)
                if not physical:
                    raise ValueError(f"expert {i} has no legal installed core")
                candidates=[]
                for c in physical:
                    outstanding=queues[c][positions[c]:]
                    if not outstanding:
                        earliest=at
                    else:
                        earliest=max(at,eta_end.get(outstanding[0],availability[c]))
                        # Current's ETA is adjusted by the existing ours
                        # progress feedback; add any already bound Next.
                        for queued in outstanding[1:]:
                            earliest=max(earliest,binding_ready[queued])+predicted[queued,c]
                    can_bind=capacity_to_bind(c,costs[i,c],i)
                    if can_bind and outstanding and earliest-at>params.binding_lead_cycles+1e-7:
                        event(max(at+1e-6,earliest-params.binding_lead_cycles),
                              "bindcheck",c,outstanding[0],0)
                        can_bind=False
                    pred=_prediction(predictor,es[i],c,costs[i,c].isolated_cycles,design,params)
                    extra=max(0,costs[i,c].hbm_bytes-costs[i,c].unique_hbm_bytes)
                    other_bus_users=any(len(queues[q])>positions[q] for q in range(ncores) if q!=c)
                    # Own repeated bytes are already in nominal_cycles.
                    # Extra/bandwidth is a shared-bus externality only when
                    # another installed core still has work to consume.
                    penalty=extra/params.hbm_bandwidth if enhanced_dispatch and other_bus_users else 0.0
                    candidates.append(Candidate(c,pred,max(at,earliest),
                                                refetch_factor(costs[i,c]),can_bind,extra,penalty))
                decision=choose_candidate(candidates,now=at,enhanced=enhanced_dispatch)
                legal=sum(q.admissible for q in candidates)
                if decision.deferred:
                    dispatch_decisions.append({"expert_index":i,"expert_id":es[i].get("id",i),
                        "at":at,"deferred":list(decision.deferred)})
                chosen=decision.chosen
                if chosen is None:
                    # Other descriptors in the finite window remain eligible.
                    # Existing completion/progress/bind events reconsider t.
                    continue
                c,pred,finish=chosen.core,chosen.predicted_cycles,chosen.earliest_start
                was_empty=len(queues[c])==positions[c]
                ctrl=max(ctrl,at)+(2*legal+2+2*len(physical)+2*len(decision.deferred) if params.charge_control else 0)
                predicted[i,c]=pred
                availability[c]=max(finish,ctrl)+pred
                eta_end[i]=availability[c]
                queues[c].append(i)
                waiting.remove(i)
                bindings.append({"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                    "legal_cores":legal,"physical_legal_cores":len(physical),"predicted_cycles":pred,
                    "nominal_cycles":costs[i,c].isolated_cycles,
                    "bind_cycle":ctrl,"predicted_finish":availability[c],"online":True,
                    "bounded_core_queue_depth":len(queues[c])-positions[c],
                    "refetch_factor":chosen.refetch,"capacity_policy":"wait_vs_refetch_eft",
                    "decision_kind":("no_refetch" if chosen.refetch==1 else
                        "immediate_refetch_faster_than_wait" if any(q.refetch==1 for q in candidates)
                        else "all_cores_refetch"),
                    "candidate_comparisons":[asdict(q)|{"predicted_finish":q.finish} for q in candidates]})
                binding_ready[i] = ctrl
                if was_empty:
                    event(ctrl,"bound_ready",c,i,0)
                else:
                    current=queues[c][positions[c]]
                    event(max(ctrl,eta_end.get(current,finish)-params.hbm_latency-
                              costs[i,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,current,0)
                progress=True
                break
    def begin_task(c,at):
        if positions[c]>=len(queues[c]):
            finishes[c]=at
            return
        i=queues[c][positions[c]]
        # A queued Next descriptor can become the queue head while its
        # serialized binding/ownership update is still in flight. Neither
        # the cold HBM request nor task execution may precede that grant.
        if at < binding_ready[i]:
            event(binding_ready[i],"bound_ready",c,i,0)
            return
        task_started[i]=at
        eta_end[i]=at+predicted[i,c]
        task_done_work[i]=0.0
        task_next_quarter[i]=1
        tasks[i]={"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                  "start":at,"predicted_cycles":predicted[i,c],"nominal_cycles":costs[i,c].isolated_cycles,"Me":es[i]["Me"],
                  "prefetch_request":None,"first_weight_ready":None,"current_end":previous_completion[c],
                  "shape":f"{actual_core(i,c).pm}x{actual_core(i,c).pn}x{actual_core(i,c).pk}"}
        pf=next_state.get(c)
        if pf and pf["task"]==i:
            tasks[i]["prefetch_request"]=pf["requested"]
            if pf.get("ready") is not None:
                tasks[i]["first_weight_ready"]=pf["ready"]
                event(at,"ready",c,i,0)
            else:
                pf["waiting"]=True
        else:
            event(at+params.hbm_latency,"ready",c,i,0)
        if params.prefetch and positions[c]+1<len(queues[c]):
            # Predictors influence prefetch timing, not correctness.
            predicted_end=at+predicted[i,c]
            event(max(at,predicted_end-params.hbm_latency-32/params.hbm_bandwidth),"prefetch",c,i,0)
    if owners is None:
        try_bind(starts)
    else:
        for c in range(ncores):
            begin_task(c,starts)
    now=0.0
    while pending or active:
        while pending and pending[0][0] <= now+1e-7:
            at,_,kind,c,i,pi=heapq.heappop(pending)
            event_keys.discard((round(at,6),kind,c,i,pi))
            if kind=="bindcheck":
                try_bind(at)
                continue
            co=costs[i,c]
            if kind=="bound_ready":
                if i not in task_started and positions[c]<len(queues[c]) and queues[c][positions[c]]==i:
                    begin_task(c,at)
            elif kind=="ready":
                s=co.phases[pi]
                subtract=0
                subtract_w=0
                pf=next_state.get(c)
                if pi==0 and pf and pf["task"]==i and pf.get("ready") is not None:
                    subtract=pf["bytes"]
                    subtract_w=pf["w_bytes"]
                    tasks[i]["first_weight_ready"]=pf["ready"]
                    # Reference released at first operand read, never on ETA.
                    del next_state[c]
                demand={"hbm":max(0,s.hbm_bytes-subtract),
                        "vector":s.vector_elements,("vector",c):s.vector_elements,
                        ("compute",c):s.compute_cycles,("W",c):s.w_sram_bytes-subtract_w,
                        ("X",c):s.x_sram_bytes,("acc",c):s.acc_sram_bytes,
                        "control":s.control_cycles if params.charge_control else 0}
                active[("run",c)]={"task":i,"phase":pi,"cost":s,"remaining":1.0,
                                      "began":at,"demand":demand,"limit":float("inf"),"core":c}
                # Static per-phase attribution only; no resource/service change.
                options=[(r,demand.get(r,0)/cap) for r,cap in capacities.items() if demand.get(r,0)]
                active[("run",c)]["static_reason"]=str(max(options,key=lambda r:r[1])[0])
                phase_started[i,pi]=at
                if pi==0 and tasks[i]["first_weight_ready"] is None:
                    # Record the observed prefix completion of this actor's
                    # shared fluid HBM+local fill service, never a free global
                    # bandwidth estimate. This remains a phase-fluid prefix,
                    # not a native per-request tile-return trace.
                    active[("run",c)]["first_weight_fraction"]=min(1.0,max(
                        s.first_weight_bytes/max(1.0,demand["hbm"]),
                        (s.first_weight_bytes+s.w_tile_bytes)/max(1.0,demand[("W",c)])))
                if params.prefetch and positions[c]+1<len(queues[c]) and c not in next_state:
                    ni=queues[c][positions[c]+1]
                    event(max(at,eta_end[i]-params.hbm_latency-
                              costs[ni,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,i,pi)
                try_bind(at)
            elif kind=="continue":
                if pi<len(co.phases):
                    event(at+params.hbm_latency,"ready",c,i,pi)
                else:
                    tasks[i]["finish"]=at
                    tasks[i]["actual_cycles"]=at-task_started[i]
                    tasks[i]["predicted_finish"]=task_started[i]+predicted[i,c]
                    if predictor is not None and hasattr(predictor,"on_complete"):
                        args=(es[i],c,predicted[i,c],tasks[i]["actual_cycles"])
                        if complete_accepts_nominal:
                            predictor.on_complete(*args,nominal=costs[i,c].isolated_cycles)
                        else:
                            predictor.on_complete(*args)
                    eta_end[i]=at
                    previous_completion[c]=at
                    positions[c]+=1
                    next_i=queues[c][positions[c]] if positions[c]<len(queues[c]) else None
                    begin_task(c,at)
                    if next_i is not None and next_i in tasks:
                        tasks[next_i]["current_end"]=at
                    try_bind(at)
            elif kind=="prefetch":
                if not params.prefetch:
                    continue
                if positions[c]>=len(queues[c]) or queues[c][positions[c]]!=i or c in next_state:
                    continue
                if positions[c]+1>=len(queues[c]):
                    continue
                ni=queues[c][positions[c]+1]
                first=costs[ni,c].phases[0]
                current=active.get(("run",c))
                if current is None:
                    continue
                cur=current["cost"]
                if cur.w_slots-cur.weight_live_slots<1:
                    continue
                if design.landing_mode == "shared" and not _pool_can_reserve_next(design,next_state,first.w_tile_bytes):
                    continue
                nbytes=ceildiv(first.first_weight_bytes,32)*32
                next_state[c]={"task":ni,"bytes":nbytes,"slot_bytes":first.w_tile_bytes,
                               "w_bytes":nbytes+first.w_tile_bytes,
                               "requested":at,"ready":None,"waiting":False}
                event(at+params.hbm_latency,"prefetch_ready",c,ni,0)
            elif kind=="prefetch_ready":
                pf=next_state.get(c)
                if not pf or pf["task"]!=i:
                    continue
                active[("prefetch",c)]={"task":i,"phase":0,"cost":None,"remaining":1.0,
                    "began":at,"core":c,"limit":float("inf"),
                    "demand":{"hbm":pf["bytes"],("W",c):pf["w_bytes"]}}
        if not active:
            if pending:
                if detail and pending[0][0] > now:
                    segments.append(_idle_segment(now,pending[0][0],ncores))
                now=max(now,pending[0][0])
                continue
            break
        # One physical pool ledger for shared landing; unchanged private windows.
        pool_ledger=_set_window_limits(active,next_state,design,params,now)
        rates=_fluid_rates(active,capacities)
        dt=min(a["remaining"]/max(1e-300,rates[key]) for key,a in active.items())
        for key,a in active.items():
            if "first_weight_fraction" in a:
                need=a["first_weight_fraction"]-(1-a["remaining"])
                if need>1e-9:
                    dt=min(dt,need/max(1e-300,rates[key]))
        for key,a in active.items():
            if key[0]!="run" or predictor is None:
                continue
            i,c=a["task"],a["core"]
            q=task_next_quarter[i]
            if q<=3:
                target=costs[i,c].useful_macs*q/4
                sofar=task_done_work[i]+a["cost"].useful_macs*(1-a["remaining"])
                if target>sofar+1e-7:
                    dt=min(dt,(target-sofar)/(a["cost"].useful_macs*max(1e-300,rates[key])))
        if pending:
            dt=min(dt,max(0.0,pending[0][0]-now))
        if not math.isfinite(dt) or dt>1e20:
            raise RuntimeError("resource deadlock / zero progress")
        if dt<=1e-10:
            dt=1e-7
        usages={r:sum(rates[key]*a["demand"].get(r,0) for key,a in active.items()) for r in capacities}
        if detail:
            segments.append(_supply_segment(now,dt,active,rates,usages,params,design,es,pool_ledger))
        actual_bytes+=usages["hbm"]*dt
        busy["hbm"]+=usages["hbm"]/capacities["hbm"]*dt
        busy["vector"]+=usages["vector"]/capacities["vector"]*dt
        for kind in ("W","X","acc"):
            denom=sum(capacities[(kind,c)] for c in range(ncores))
            busy[kind]+=sum(usages[(kind,c)] for c in range(ncores))/denom*dt
        for key,a in active.items():
            a["remaining"]-=dt*rates[key]
            if key[0]=="run":
                c=a["core"]
                core_busy[c]+=rates[key]*a["demand"][("compute",c)]*dt
                reason=a["static_reason"]
                stream_attribution[c][reason]=stream_attribution[c].get(reason,0)+dt
        now+=dt
        for key,a in active.items():
            target=a.get("first_weight_fraction")
            if target is not None and 1-a["remaining"]>=target-1e-9:
                tasks[a["task"]]["first_weight_ready"]=now
                del a["first_weight_fraction"]
        for key,a in list(active.items()):
            if key[0]!="run" or predictor is None:
                continue
            i,c=a["task"],a["core"]
            progress=(task_done_work[i]+a["cost"].useful_macs*(1-a["remaining"]))/costs[i,c].useful_macs
            while task_next_quarter[i]<=3 and progress+1e-8>=task_next_quarter[i]/4:
                quarter=task_next_quarter[i]/4
                task_next_quarter[i]+=1
                update=None
                if predictor is not None and hasattr(predictor,"on_progress"):
                    update=predictor.on_progress(es[i],c,now-task_started[i],quarter,
                                                  max(0.0,eta_end[i]-now))
                if update is not None:
                    eta_end[i]=now+max(0.0,float(update))
                # Reconsider ownership after actual progress even if the
                # estimator has no learning update at this point.
                event(max(now,eta_end[i]-params.binding_lead_cycles),"bindcheck",c,i,0)
                if positions[c]+1<len(queues[c]) and c not in next_state:
                    ni=queues[c][positions[c]+1]
                    event(max(now,eta_end[i]-params.hbm_latency-
                              costs[ni,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,i,0)
        for key in list(active):
            a=active[key]
            if a["remaining"]>1e-8:
                continue
            del active[key]
            c,i,pi=a["core"],a["task"],a["phase"]
            if key[0]=="prefetch":
                pf=next_state[c]
                pf["ready"]=now
                prefetched_bytes+=pf["bytes"]
                if pf.get("waiting"):
                    tasks[i]["first_weight_ready"]=now
                    event(now,"ready",c,i,0)
            else:
                s=a["cost"]
                task_done_work[i]+=s.useful_macs
                # The phase compute demand already includes its final drain.
                end=now
                if detail:
                    phase_rows.append({"expert_index":i,"core":c,"phase":pi,"start":a["began"],
                                       "stream_finish":now,"finish":end,**asdict(s)})
                event(end,"continue",c,i,pi+1)
    if waiting:
        raise RuntimeError("unbound tasks remain after all execution stopped")
    cycles=max(finishes)
    chosen=[costs[b["expert_index"],b["core"]] for b in bindings]
    useful=sum(t.useful_macs for t in chosen)
    issued=sum(t.issued_macs for t in chosen)
    unique=sum(t.unique_hbm_bytes for t in chosen)
    expected=sum(3*e["Me"]*e.get("H",2048)*e.get("F",1408) for e in es)
    if useful!=expected:
        raise AssertionError("useful MAC conservation")
    # Fluid accumulation is floating-point; exported byte counts are exact
    # declared task traffic (prefetch subtracts then replaces the same bytes).
    hbm=sum(t.hbm_bytes for t in chosen)
    if abs(actual_bytes-hbm)>max(1.0,hbm*1e-8):
        raise AssertionError(f"HBM traffic conservation: {actual_bytes} vs {hbm}")
    result={"workload":workload.get("id","anonymous"),"batch":batch,"cycles":cycles,
            "latency_ms":cycles/1e6,"useful_macs":useful,"issued_macs":issued,
            "padding_macs":issued-useful,"spatial_utilization":useful/issued,
            "wall_mac_utilization":useful/(design.total_macs*cycles),
            "hbm_bytes":hbm,"native_unique_bytes":unique,
            "w_sram_bytes":sum(t.w_sram_bytes for t in chosen),
            "x_sram_bytes":sum(t.x_sram_bytes for t in chosen),
            "acc_sram_bytes":sum(t.acc_sram_bytes for t in chosen)+clear_bytes,
            "core_finish_cycles":finishes,"core_compute_busy":core_busy,
            "w_port_busy":busy["W"],"x_port_busy":busy["X"],"acc_port_busy":busy["acc"],
            "hbm_busy_frac":busy["hbm"]/cycles,"hbm_busy_cycles":busy["hbm"],
            "hbm_actual_Bpc":hbm/cycles,"core_finish_gap":max(finishes)-min(finishes),
            "idle_frac":max(0.0,1-sum(core_busy)/(ncores*cycles)),
            "prefetched_bytes":int(prefetched_bytes),"wasted_prefetch_bytes":wasted_prefetch,
            "scope":SCOPE,"non_iso_resource_reference":params.onchip_mode=="fixed_issue",
            "first_weight_timing_scope":"observed shared-fluid ingress/fill prefix; not native per-request tile timing"}
    result["U2_weight_read_once_oracle"]=params.sram_weight_read_once_oracle
    result["U1_per_expert_shape_oracle"]=bool(shape_overrides)
    if detail:
        result.update(segments=[dict(s,end=min(s["end"],cycles)) for s in segments if s["start"]<cycles],supply_observation_scope="phase-fluid rate x 65-cycle latency-residency estimate; not native credit/request trace",bindings=bindings,tasks=[tasks[i] for i in sorted(tasks)],phases=phase_rows,
                      dispatch_decisions=dispatch_decisions,dispatch_priority_order=order,
                      stream_attribution=stream_attribution,ledger=design.ledger(),
                      counters_warning="resource occupancy/progress overlaps; not native stall counters; never sum as wall time")
    return result



def _simulate_fixed(workload: dict, design: Design, params: Parameters = Parameters(),
             policy: str = "eft", owners: Iterable[int] | None = None,
             predictor=None, detail: bool = True,
             cost_overrides: dict | None=None,shape_overrides: dict | None=None, *,
             t_big: int=2,large_first: bool=True,enhanced_dispatch: bool=True) -> dict:
    """One globally shared-HBM post-router layer; deterministic for fixed seed.

    Main results require total_macs=12,288. Synthetic MAC scaling is explicit
    in Design.total_macs. Large synthetic windows execute sequential finite
    X/Y/route-storage chunks and pay off-chip activation spill traffic.
    """
    batch=workload["batch"]
    if batch<1 or any(not 0<e["Me"]<=batch for e in workload["experts"]):
        raise ValueError("Me must lie inside the declared batch")
    hidden=workload.get("hidden",workload["experts"][0].get("H",2048) if workload["experts"] else 2048)
    topk=workload.get("top_k",6)
    chunks=storage_chunks(workload)
    if len(chunks)==1:
        return _one_layer_fixed(workload,design,params,policy,None if owners is None else tuple(owners),predictor,detail,
                          cost_overrides,shape_overrides,t_big=t_big,large_first=large_first,enhanced_dispatch=enhanced_dispatch)
    if cost_overrides or shape_overrides:
        raise ValueError("chunked U1 override needs per-storage-chunk task costs, not whole-expert costs")
    owner_map=None if owners is None else tuple(owners)
    results=[]
    offset=0.0
    for cw in chunks:
        os=None if owner_map is None else tuple(owner_map[e["original_expert_index"]] for e in cw["experts"])
        r=_one_layer_fixed(cw,design,params,policy,os,predictor,detail,t_big=t_big,large_first=large_first,enhanced_dispatch=enhanced_dispatch)
        # Offsets are observation-only; physical spill/service stays identical.
        if detail:
            chunk_index=len(results)
            for t in r["tasks"]:
                t["chunk_index"]=chunk_index
                t["original_expert_index"]=cw["experts"][t["expert_index"]]["original_expert_index"]
                for name in ("start","finish","predicted_finish","first_weight_ready","current_end","prefetch_request"):
                    if t.get(name) is not None:
                        t[name]+=offset
            for b in r["bindings"]:
                b["chunk_index"]=chunk_index
                b["original_expert_index"]=cw["experts"][b["expert_index"]]["original_expert_index"]
                for name in ("bind_cycle","predicted_finish"):
                    if b.get(name) is not None:b[name]+=offset
            for segment in r.get("segments",[]):
                segment["start"]+=offset
                segment["end"]+=offset
            for ph in r["phases"]:
                ph["chunk_index"]=chunk_index
                ph["original_expert_index"]=cw["experts"][ph["expert_index"]]["original_expert_index"]
                for name in ("start","stream_finish","finish"):
                    if ph.get(name) is not None:ph[name]+=offset
        # Finite global X/combine backing. Output of each completed chunk is
        # drained before reuse; no free enlargement for B256/E256 studies.
        io=cw["batch"]*hidden*6
        spill_time=params.hbm_latency+io/params.hbm_bandwidth
        r["hbm_bytes"]+=io
        r["cycles"]+=spill_time
        r["activation_spill_bytes"]=io
        results.append(r)
        offset+=r["cycles"]
    useful=sum(r["useful_macs"] for r in results)
    issued=sum(r["issued_macs"] for r in results)
    result={"workload":workload.get("id","anonymous"),"batch":batch,"cycles":offset,"latency_ms":offset/1e6,
            "useful_macs":useful,"issued_macs":issued,"padding_macs":issued-useful,
            "spatial_utilization":useful/issued,"hbm_bytes":sum(r["hbm_bytes"] for r in results),
            "native_unique_bytes":sum(2*e.get("F",1408)*ceildiv(2*e.get("H",2048),32)*32+
                                      e.get("H",2048)*ceildiv(2*e.get("F",1408),32)*32 for e in workload["experts"]),
            "w_sram_bytes":sum(r["w_sram_bytes"] for r in results),
            "x_sram_bytes":sum(r["x_sram_bytes"] for r in results),
            "acc_sram_bytes":sum(r["acc_sram_bytes"] for r in results),
            "core_compute_busy":[sum(r["core_compute_busy"][c] for r in results) for c in range(len(design.cores))],
            "core_finish_cycles":[offset]*len(design.cores),
            "w_port_busy":sum(r["w_port_busy"] for r in results),
            "x_port_busy":sum(r["x_port_busy"] for r in results),
            "acc_port_busy":sum(r["acc_port_busy"] for r in results),
            "hbm_busy_frac":sum(r["hbm_bytes"] for r in results)/(params.hbm_bandwidth*offset),
            "core_finish_gap":0.0,"storage_chunks":len(chunks),
            "activation_spill_bytes":sum(r["activation_spill_bytes"] for r in results),
            "scope":SCOPE+"; sequential input/combine chunks, explicit HBM spill",
            "non_iso_resource_reference":params.onchip_mode=="fixed_issue"}
    if detail:
        result.update(segments=[s for r in results for s in r.get("segments",[])],
                      supply_observation_scope="phase-fluid latency-residency proxy, not native credit/request trace",
                      tasks=[t for r in results for t in r["tasks"]],
                      bindings=[b for r in results for b in r["bindings"]],
                      phases=[p for r in results for p in r["phases"]],chunk_results=results,ledger=design.ledger())
    return result



def simulate(workload: dict, design: Design, params: Parameters=Parameters(), *,
             t_big: int=3, large_first: bool=True, predictor=None, detail: bool=True,
             dispatch: str="fixed", owners: Iterable[int] | None=None, policy: str="eft",
             cost_overrides=None, shape_overrides=None):
    """One shared front end and finite private/pool physical services.

    fixed_legacy retains the committed 7eb58061 binding policy for isolated
    E0/E3 comparisons. fixed adds the explicit shared-refetch externality and
    minimum-refetch rule. owners always executes the same LPT physical replay.
    Single-core fixed/eft_old delegate the same engine, hence bit-exact costs
    and event order. Detail-only observations do not influence rates/events.
    """
    if t_big not in (2,3,4,6,8):
        raise ValueError("t_big must be one of 2,3,4,6,8")
    if dispatch not in ("fixed","fixed_legacy","eft_old","milp"):
        raise ValueError("unknown dispatch policy")
    if len(design.cores)==1 or dispatch=="eft_old":
        return frozen.simulate(workload,design,params,policy=policy,owners=owners,
                               predictor=predictor,detail=detail,cost_overrides=cost_overrides,
                               shape_overrides=shape_overrides)
    if dispatch=="milp" and owners is None:
        raise ValueError("MILP replay requires explicit solver owners")
    return _simulate_fixed(workload,design,params,policy=policy,owners=owners,
                           predictor=predictor,detail=detail,cost_overrides=cost_overrides,
                           shape_overrides=shape_overrides,t_big=t_big,large_first=large_first,
                           enhanced_dispatch=dispatch=="fixed" and owners is None)
