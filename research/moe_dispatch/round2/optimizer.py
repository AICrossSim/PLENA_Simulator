"""Exact CP-SAT assignment relaxation and legal regional lower bounds.

The solver does NOT solve a coupled temporal schedule. It minimizes the
assignment's resource/critical-path relaxation; its owners are replayed by
the finite streaming simulator. Cold-task isolated latencies are never
summed. Fractional service demands are rounded DOWN, so the CP-SAT optimum
remains a conservative continuous-time lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
import logging
import math
from typing import Iterable

from ortools.sat.python import cp_model

try:
    from .model import Design, Parameters, Core, task_cost, simulate, ceildiv, storage_chunks
except ImportError:  # External pre-integration draft.
    import importlib.util
    from pathlib import Path
    import sys
    name="round2_draft_model"
    if name not in sys.modules:
        s=importlib.util.spec_from_file_location(name,Path(__file__).with_name("model.py"))
        m=importlib.util.module_from_spec(s)
        sys.modules[name]=m
        s.loader.exec_module(m)
    m=sys.modules[name]
    Design,Parameters,Core=m.Design,m.Parameters,m.Core
    task_cost,simulate,ceildiv,storage_chunks=m.task_cost,m.simulate,m.ceildiv,m.storage_chunks


def _down(value: float, quantum: float) -> int:
    """Exact floor of the represented service/quantum ratio.

    Integer-ratio arithmetic avoids floating division rounding upward and
    avoids gratuitously subtracting one tick from every exact coefficient,
    which would destroy useful coefficient GCDs in the allocation solver.
    """
    if value<=0:
        return 0
    vn,vd=float(value).as_integer_ratio()
    qn,qd=float(quantum).as_integer_ratio()
    return max(0,(vn*qd)//(vd*qn))


def _work_upper(value: float) -> Decimal:
    """Canonical conservative accounting for native solver work diagnostics.

    CP-SAT occasionally reports the same work with floating summation noise
    at its final binary ULP. Round to twelve decimal significant digits and
    add one reporting unit, which upper-bounds the native value. These units
    govern only the solver effort ledger; physical service coefficients,
    integer objectives, and the time quantum retain their full precision.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError("native deterministic work must be finite and nonnegative")
    if value == 0:
        return Decimal(0)
    rounded = Decimal(format(value, ".12g"))
    return rounded + Decimal(1).scaleb(rounded.adjusted() - 11)


def _group_tasks(workload: dict,table):
    # Global storage chunks can split equal original Me into different row
    # populations. Aggregate only tasks with genuinely identical full costs.
    groups={}
    for i,row in enumerate(table):
        groups.setdefault(tuple(row),[]).append(i)
    return list(groups.items())


def _costs(workload,design,params):
    chunks=storage_chunks(workload)
    if len(chunks)==1:
        return [[_legal_cost(e,design,c,params) for c in range(len(design.cores))]
                for e in workload["experts"]]
    collection={i:[] for i in range(len(workload["experts"]))}
    for cw in chunks:
        for e in cw["experts"]:collection[e["original_expert_index"]].append(e)
    table=[]
    for original in range(len(workload["experts"])):
        row=[]
        for c in range(len(design.cores)):
            parts=[_legal_cost(e,design,c,params) for e in collection[original]]
            if any(x is None for x in parts):row.append(None);continue
            first=parts[0]
            additive=("issues","useful_macs","issued_macs","padding_macs","issue_busy", 
                      "w_sram_bytes","x_sram_bytes","acc_sram_bytes","hbm_bytes","unique_hbm_bytes",
                      "vector_elements","isolated_cycles","z_chunks","spill_bytes")
            summed={name:sum(getattr(x,name) for x in parts) for name in additive}
            row.append(replace(first,phases=tuple(s for x in parts for s in x.phases),
                               dependency_floor=max(x.dependency_floor for x in parts),**summed))
        table.append(row)
    return table


def _common_spill(workload):
    return workload["batch"]*workload.get("hidden",2048)*6 if len(storage_chunks(workload))>1 else 0


def _legal_cost(e,d,c,p):
    try:
        return task_cost(e,d,c,p)
    except ValueError:
        return None


def _resource_services(cost,design,c,params):
    return {("compute",c):cost.irreducible_busy,
            ("W",c):cost.w_sram_bytes/params.w_bandwidth(design,c),
            ("X",c):cost.x_sram_bytes/(design.x_banks[c]*params.bank_Bpc),
            ("acc",c):cost.acc_sram_bytes/(design.acc_banks[c]*params.bank_Bpc),
            ("vector",c):cost.vector_elements/(design.vector_lanes[c]*params.vector_scale),
            "hbm":cost.hbm_bytes/params.hbm_bandwidth,
            "vector_global":cost.vector_elements/(64*params.vector_scale)}


def allocation_objective(workload: dict,design: Design,params: Parameters,
                         owners: Iterable[int]) -> float:
    """Continuous objective of the declared allocation relaxation."""
    owners=tuple(owners)
    if len(owners)!=len(workload["experts"]):
        raise ValueError("exactly one owner per expert required")
    loads={"hbm":_common_spill(workload)/params.hbm_bandwidth}
    single=0.0
    table=_costs(workload,design,params)
    for i,c in enumerate(owners):
        co=table[i][c]
        if co is None:raise ValueError("selected owner cannot execute the storage-chunked task")
        single=max(single,co.dependency_floor)
        for r,duration in _resource_services(co,design,c,params).items():
            loads[r]=loads.get(r,0.0)+duration
    return max([single,*loads.values()],default=0.0)



def _enumerate_assignment(workload,design,params,quantum=1e-6,max_combinations=1_000_000, *, table=None, groups=None):
    """Exact deterministic enumeration of the quantized grouped-count model.

    The full finite assignment feasible set is unchanged. Domains above the
    frozen one-million-combination cap returnNone and retain CP-SAT. Safe
    suffix minima prune only regions unable to beat a complete incumbent;
    all resource coefficients use the identical exact binary-ratio floor.
    An exact result certifies assignment/resources, never temporal schedule.
    """
    table=_costs(workload,design,params) if table is None else table
    if any(all(co is None for co in row) for row in table):return None
    groups=_group_tasks(workload,table) if groups is None else groups
    counts=[len(ids) for _,ids in groups]
    choices=[];resources=[];critical=[]
    allkeys=set()
    for (_,ids) in groups:
        row=table[ids[0]];count=len(ids)
        legal=[c for c,co in enumerate(row) if co is not None]
        if len(row)==1:options=[(count,)]
        elif len(legal)==1:options=[tuple(count if c==legal[0] else 0 for c in range(len(row)))]
        else:options=[(a,count-a) for a in range(count+1)]
        rr=[];cc=[]
        for c,co in enumerate(row):
            r={} if co is None else {key:_down(value,quantum) for key,value in _resource_services(co,design,c,params).items()}
            rr.append(r);allkeys.update(r);cc.append(0 if co is None else _down(co.dependency_floor,quantum))
        choices.append(options);resources.append(rr);critical.append(cc)
    domain=math.prod(len(x) for x in choices)
    if domain>max_combinations:return None
    keys=sorted(allkeys,key=str);keyindex={k:i for i,k in enumerate(keys)};nr=len(keys)
    # Group alternatives have exact integer demands. Suffix minima are legal
    # independent optimistic bounds, not forced slow/fast owner decisions.
    alternatives=[]
    for g,options in enumerate(choices):
        a=[]
        for ns in options:
            loads=[sum(ns[c]*resources[g][c].get(r,0) for c in range(len(ns))) for r in keys]
            dep=max((critical[g][c] for c in range(len(ns)) if ns[c]),default=0)
            a.append((ns,loads,dep))
        alternatives.append(a)
    suffix=[[0]*nr for _ in range(len(groups)+1)];deps=[0]*(len(groups)+1)
    for g in reversed(range(len(groups))):
        suffix[g]=[suffix[g+1][j]+min(a[1][j] for a in alternatives[g]) for j in range(nr)]
        deps[g]=max(deps[g+1],min(a[2] for a in alternatives[g]))
    initial=[0]*nr
    if 'hbm' in keyindex:initial[keyindex['hbm']]=_down(_common_spill(workload)/params.hbm_bandwidth,quantum)
    best=math.inf;bestchoices=None;nodes=0;leaves=0;pruned=0;counts_selected=[]
    def visit(g,loads,dep):
        nonlocal best,bestchoices,nodes,leaves,pruned
        nodes+=1
        lower=max([dep,deps[g],*(loads[j]+suffix[g][j] for j in range(nr))],default=0)
        if lower>=best:
            pruned+=1;return
        if g==len(groups):
            leaves+=1;best=lower;bestchoices=tuple(counts_selected);return
        for ns,add,d in alternatives[g]:
            counts_selected.append(ns)
            visit(g+1,[loads[j]+add[j] for j in range(nr)],max(dep,d))
            counts_selected.pop()
    visit(0,initial,0)
    owners=[None]*len(table)
    for (_,ids),ns in zip(groups,bestchoices):
        cursor=0
        for c,num in enumerate(ns):
            for i in ids[cursor:cursor+num]:owners[i]=c
            cursor+=num
    return {'owners':owners,'objective_ticks':int(best),'lb_cycles':int(best)*quantum,
        'objective_upper_cycles':allocation_objective(workload,design,params,owners),
        'domain_combinations':domain,'visited_nodes':nodes,'evaluated_leaves':leaves,
        'pruned_nodes':pruned,'aggregated_task_types':len(groups),'group_counts':bestchoices}



def _solve_fixed_t(w,d,p,quantum=1e-6,queries=64,total_work=.1, *, table=None, groups=None):
    """Same quantized integer model, queried at a constant objective bound.

    Integer SAT witnesses tighten UB; exact UNSAT proofs raiseLB=T+1.
    Unknown responses do not move either side. Only a closed integer bracket
    is OPTIMAL. This avoids native minimization's tick-sized lazy encoding.
    The <=64 queries share the configured deterministic work budget.
    """
    table=_costs(w,d,p) if table is None else table
    groups=_group_tasks(w,table) if groups is None else groups
    coeffs=[];deps=[]
    for row in table:
        coeffs.append([None if co is None else {r:_down(t,quantum) for r,t in _resource_services(co,d,c,p).items()} for c,co in enumerate(row)])
        deps.append([None if co is None else _down(co.dependency_floor,quantum) for co in row])
    shared=('hbm','vector_global');constant=_down(_common_spill(w)/p.hbm_bandwidth,quantum)
    def obj(owners):
        loads={'hbm':constant};dep=0
        for i,c in enumerate(owners):
            dep=max(dep,deps[i][c])
            for r,t in coeffs[i][c].items():loads[r]=loads.get(r,0)+t
        return max([dep,*loads.values()])
    # Two independent legal witness heuristics, neither used as a lower bound.
    greedy=[min((c for c in range(len(d.cores)) if table[i][c] is not None),key=lambda c:(table[i][c].isolated_cycles,c)) for i in range(len(table))]
    loads={'hbm':constant};dep=0;balanced=[None]*len(table)
    for i in sorted(range(len(table)),key=lambda i:-min(table[i][c].isolated_cycles for c in range(len(d.cores)) if table[i][c] is not None)):
        choices=[]
        for c in range(len(d.cores)):
            if table[i][c] is None:continue
            new=loads.copy()
            for r,t in coeffs[i][c].items():new[r]=new.get(r,0)+t
            choices.append((max([dep,deps[i][c],*new.values()]),c,new))
        _,c,loads=min(choices,key=lambda x:(x[0],x[1]));dep=max(dep,deps[i][c]);balanced[i]=c
    owner=min((greedy,balanced),key=lambda x:(obj(x),tuple(x)));upper=obj(owner)
    lows=[max(min(x for x in row if x is not None) for row in deps)]
    for r in shared:
        lows.append((constant if r=='hbm' else 0)+sum(min(v.get(r,0) for v in row if v is not None) for row in coeffs))
    for kind in ('compute','W','X','acc','vector'):
        total=sum(min(v.get((kind,c),0) for c,v in enumerate(row) if v is not None) for row in coeffs)
        lows.append((total+len(d.cores)-1)//len(d.cores))
    lower=max(lows);trace=[];spent=Decimal(0);work_budget=Decimal(str(total_work))
    assert lower<=upper
    for qi in range(queries):
        if lower==upper or spent>=work_budget:break
        target=(lower+upper)//2;model=cp_model.CpModel();ns={};resources={}
        for g,(_,ids) in enumerate(groups):
            count=len(ids);i=ids[0];choices=[]
            for c,co in enumerate(table[i]):
                if co is None or deps[i][c]>target:continue
                var=model.NewIntVar(0,count,f'n_{g}_{c}');ns[g,c]=var;choices.append(var)
                for r,t in coeffs[i][c].items():
                    if t:resources.setdefault(r,[]).append(t*var)
                model.AddHint(var,sum(owner[j]==c for j in ids))
            model.Add(sum(choices)==count)
        for r,terms in resources.items():model.Add(sum(terms)+(constant if r=='hbm' else 0)<=target)
        solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=20261007;solver.parameters.linearization_level=2;solver.parameters.use_sat_inprocessing=False;solver.parameters.cp_model_presolve=True;solver.parameters.stop_after_first_solution=True;solver.parameters.max_deterministic_time=float(min(work_budget/16,max(Decimal("1e-12"),work_budget-spent)))
        status=solver.Solve(model);s=solver.StatusName(status);raw_used=solver.ResponseProto().deterministic_time;used=_work_upper(raw_used);spent+=used
        logging.getLogger(__name__).debug("fixed-T query %s native deterministic work=%r canonical upper=%s", qi, raw_used, used)
        row={'target':target,'status':('SAT' if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else 'UNSAT' if status==cp_model.INFEASIBLE else s),'cp_status':s,'lower_before':lower,'upper_before':upper,'deterministic_work':float(used)}
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            witness=[None]*len(table)
            for g,(_,ids) in enumerate(groups):
                cur=0
                for c in range(len(d.cores)):
                    if (g,c) not in ns:continue
                    num=solver.Value(ns[g,c])
                    for j in ids[cur:cur+num]:witness[j]=c
                    cur+=num
            upper=min(upper,obj(witness));owner=witness
        elif status==cp_model.INFEASIBLE:lower=target+1
        else:
            row.update(lower_after=lower,upper_after=upper);trace.append(row);break
        row.update(lower_after=lower,upper_after=upper);trace.append(row)
    return {'owners':owner,'lower_ticks':lower,'upper_ticks':upper,'optimal':lower==upper,'status':'OPTIMAL' if lower==upper else 'FEASIBLE','trace':trace,'quantum':quantum,'total_work':total_work,'actual_deterministic_work':float(spent)}


def solve_assignment(workload: dict,design: Design,params: Parameters=Parameters(),
                     *,max_seconds: float=10.0,quantum: float=1e-6) -> dict:
    """Exact integer assignment relaxation via enumeration or deterministic CP-SAT.

    Returns lb_cycles, owners, status, quantum, and explicit solver/gap scope.
    A work-limited FEASIBLE result uses BestObjectiveBound for its lower bound
    and the available assignment for executable replay. INFEASIBLE never
    fabricates an assignment. Task types with identical costs are aggregated
    into integer counts; this preserves the assignment feasible set exactly.
    Grouped domains up to1,000,000 combinations use exact count enumeration;
    larger domains use fixed-TCP-SAT SAT/UNSAT queries sharing the same
    deterministic work limit. A nonclosed bracket staysFEASIBLE with an
    explicit gap, neverOPTIMAL.

    ``max_seconds`` is a retained compatibility name for solver effort, NOT
    a wall-clock deadline: one effort unit maps to0.01 CP-SAT deterministic
    work units. The default10 gives0.1 deterministic work units. Machine
    load cannot change the stop point. The BnB traversal separately records
    its actual wall-clock cap; a timed-out allocation remains unresolved.
    """
    if quantum<=0 or max_seconds<=0:
        raise ValueError("positive time limit and time quantum required")
    deterministic_limit=float(max_seconds)*0.01
    solver_budget={"kind":"deterministic_work","effort_units":float(max_seconds),
                   "work_per_effort_unit":0.01,"max_deterministic_time":deterministic_limit,
                   "wall_clock_timeout_seconds":None,"num_search_workers":1,"random_seed":20261007,
                   "linearization_level":2}
    n=len(workload["experts"])
    if not n:
        return {"lb_cycles":0.0,"owners":[],"status":"OPTIMAL","quantum":quantum,
                "objective_upper_cycles":0.0,"optimal":True,"solver_budget":solver_budget,
                "scope":"empty assignment relaxation"}
    table=_costs(workload,design,params)
    if any(all(co is None for co in row) for row in table):
        return {"lb_cycles":None,"owners":None,"status":"INFEASIBLE","quantum":quantum,
                "optimal":False,"solver_budget":solver_budget,"scope":"an expert has no physically legal core"}
    groups=_group_tasks(workload,table)
    enum=_enumerate_assignment(workload,design,params,quantum,table=table,groups=groups)
    if enum is not None:
        budget={**solver_budget,"kind":"exact_finite_enumeration","selected_backend":"exact_grouped_count_enumeration",
                "enumeration_cap_combinations":1_000_000,"cp_sat_work_consumed":False}
        return {"lb_cycles":enum["lb_cycles"],"owners":enum["owners"],"status":"OPTIMAL",
                "quantum":quantum,"objective_upper_cycles":enum["objective_upper_cycles"],
                "optimal":True,"assignment_gap_cycles":max(0.0,enum["objective_upper_cycles"]-enum["lb_cycles"]),
                "aggregated_task_types":enum["aggregated_task_types"],
                "quantization_direction":"all resource/critical coefficients rounded down",
                "quantization_loss_bound_cycles":(n+1)*quantum,
                "storage_chunks":len(storage_chunks(workload)),"common_activation_spill_bytes":_common_spill(workload),
                "solver_budget":budget,"assignment_backend":"exact_grouped_count_enumeration","solver_algorithm":"exact_grouped_count_enumeration",
                "enumeration":{"domain_combinations":enum["domain_combinations"],
                    "visited_nodes":enum["visited_nodes"],"evaluated_leaves":enum["evaluated_leaves"],
                    "pruned_nodes":enum["pruned_nodes"]},
                "scope":"exact quantized assignment/resource relaxation; deterministic complete grouped-count search, not an optimal temporal schedule"}
    bracket=_solve_fixed_t(workload,design,params,quantum,total_work=deterministic_limit,
                           table=table,groups=groups)
    objective=allocation_objective(workload,design,params,bracket["owners"])
    bound=bracket["lower_ticks"]*quantum
    backend="CP-SAT fixed-T feasibility bracket"
    budget={**solver_budget,"selected_backend":backend,"enumeration_cap_combinations":1_000_000,
            "query_policy":"at most64 fixed-T satisfaction queries; SATtightensUB, UNSATraisesLB; UNKNOWNkeepsopen",
            "query_limit":64,"per_query_work_max":deterministic_limit/16,
            "use_sat_inprocessing":False,"stop_after_first_solution":True,
            "work_accounting":"conservative Decimal:12 significant digits plus one reporting unit; physical quantities unchanged",
            "actual_deterministic_work":bracket["actual_deterministic_work"]}
    return {"lb_cycles":bound,"owners":bracket["owners"],"status":bracket["status"],"quantum":quantum,
            "objective_upper_cycles":objective,"optimal":bracket["optimal"],
            "assignment_gap_cycles":max(0.0,objective-bound),"aggregated_task_types":len(groups),
            "quantization_direction":"all resource/critical coefficients rounded down",
            "quantization_loss_bound_cycles":(n+1)*quantum,
            "storage_chunks":len(storage_chunks(workload)),"common_activation_spill_bytes":_common_spill(workload),
            "solver_budget":budget,"assignment_backend":backend,"solver_algorithm":backend,
            "integer_objective_bracket":{"lower_ticks":bracket["lower_ticks"],"upper_ticks":bracket["upper_ticks"],
                "closed":bracket["optimal"],"queries":bracket["trace"]},
            "scope":"exact quantized assignment/resource relaxation; fixed-TCP SAT/UNSAT bracket, not an optimal temporal schedule"}


def evaluate_design(workload: dict,design: Design,params: Parameters=Parameters(),
                    *,runtime_policy: str="eft",predictor=None,max_seconds: float=10.0,
                    quantum: float=1e-6,detail: bool=True) -> dict:
    """Relaxation -> executable owner/LPT schedule -> actual online runtime."""
    solution=solve_assignment(workload,design,params,max_seconds=max_seconds,quantum=quantum)
    if solution["owners"] is None:
        return {"legal":False,"assignment":solution}
    milp=simulate(workload,design,params,owners=solution["owners"],detail=detail)
    runtime=simulate(workload,design,params,policy=runtime_policy,predictor=predictor,detail=detail)
    lb=solution["lb_cycles"]
    # This is the primary correctness contract for all later certificates.
    if lb>milp["cycles"]+max(1e-7,abs(lb)*1e-10):
        raise AssertionError(f"assignment LB {lb} exceeds executable schedule {milp['cycles']}")
    return {"legal":True,"lb_cycles":lb,"assignment":solution,
            "milp_sched":milp,"runtime":runtime,
            "gap_sched_pct":100*(milp["cycles"]/lb-1) if lb else None,
            "gap_runtime_pct":100*(runtime["cycles"]/milp["cycles"]-1)}


def unique_hbm_bytes(workload: dict) -> int:
    return sum(2*e.get("F",1408)*ceildiv(2*e.get("H",2048),32)*32+
               e.get("H",2048)*ceildiv(2*e.get("F",1408),32)*32
               for e in workload["experts"])


def rowchunk_hbm_floor_bytes(workload: dict,*,max_z_bytes=384*1024,max_retained_w_bytes=40*1024):
    """Mandatory bytes for the frozen full-row GU->Down chunk lifetime.

    Grant the largest possible private Z pool and allow the entire private
    W pool to survive every boundary. Each completed row chunk nevertheless
    needs all expert weights. This excludes future N-striped GU/Down fusion.
    """
    total=0
    for e in workload["experts"]:
        f,h=e.get("F",1408),e.get("H",2048)
        rows=max_z_bytes//(2*f)
        if rows<1:return math.inf
        q=ceildiv(e["Me"],rows)
        unique=2*f*ceildiv(2*h,32)*32+h*ceildiv(2*f,32)*32
        total+=unique+(q-1)*max(0,unique-max_retained_w_bytes)
    return total


def universal_bound(workload: dict,params: Parameters=Parameters(),
                    *,total_macs: int=12288) -> dict:
    """Legal at the root of the entire geometry/resource domain.

    Minimum mandatory W service is three passes: ingress read, packed write,
    and at least one array read of every unique useful weight. This follows
    the frozen round2 _projection path, not arbitrary future bypass hardware.
    """
    unique=unique_hbm_bytes(workload)
    useful=sum(3*e["Me"]*e.get("H",2048)*e.get("F",1408) for e in workload["experts"])
    wtotal=4096/30.4 if params.onchip_mode=="port_tight" else 64*params.bank_Bpc
    hbm=unique/params.hbm_bandwidth
    mac=useful*params.issue_interval/total_macs
    wport=3*unique/wtotal
    # A complete layer must form its valid Gate/Up and Down outputs. These
    # bounds only count mandatory bytes, never pessimistic spill behavior.
    xmin=sum(e["Me"]*(2*e.get("H",2048)+4*e.get("F",1408)) for e in workload["experts"])
    amin=sum(e["Me"]*(16*e.get("F",1408)+16*e.get("H",2048)) for e in workload["experts"])
    xport=xmin/(24*params.bank_Bpc)
    aport=amin/(12*params.bank_Bpc)
    rowchunk=rowchunk_hbm_floor_bytes(workload)/params.hbm_bandwidth
    terms={"hbm_unique":hbm,"HBM_rowchunk_mandatory":rowchunk,"mac":mac,"W_mandatory":wport,
           "X_mandatory":xport,"acc_mandatory":aport}
    binding=max(terms,key=terms.get)
    return {"lb_cycles":terms[binding],"binding_term":binding,"terms":terms,
            "rowchunk_scope":"fixed full-row GU/Down lifetime; optimistic max384KiB Z and40KiB retained W; futureN-stripedfusion excluded"}


def region_bound(workload: dict,params: Parameters=Parameters(),*,
                 geometries: Iterable[tuple[Core,...]] | None=None,
                 intervals: dict | None=None,total_macs: int=12288) -> dict:
    """Safe regional LB, with optimistic geometries/resources when specified.

    intervals maps per-core resource names to [(lo,hi),...]. Maximum feasible
    capacity/bandwidth is used, NEVER minimum-buffer reread counts. No task
    is forced merely because another core is slower. At most a task that is
    impossible on every other core throughout the region is forced.

    This intentionally conservative bound can leave open regions. It does
    not masquerade weak pruning as a completed global optimum proof.
    """
    base=universal_bound(workload,params,total_macs=total_macs)
    if geometries is None:
        return base
    geometries=tuple(geometries)
    if not geometries:
        return {"lb_cycles":math.inf,"binding_term":"empty_region","terms":{}}
    task_lb=0.0
    for e in workload["experts"]:
        fastest=math.inf
        m,h,f=e["Me"],e.get("H",2048),e.get("F",1408)
        for cs in geometries:
            for c,core in enumerate(cs):
                if intervals:
                    # Region-wide feasibility tests only exclude a core when
                    # even the largest allowed quota cannot hold its minimum.
                    required={"w_bytes":core.pn*core.pk*2,"x_bytes":core.pm*core.pk*2,
                              "acc_bytes":2*core.record_bytes,"z_bytes":2*f}
                    if any(name in intervals and intervals[name][c][1]<need
                           for name,need in required.items()):
                        continue
                issues=2*ceildiv(m,core.pm)*ceildiv(f,core.pn)*ceildiv(h,core.pk)+\
                       ceildiv(m,core.pm)*ceildiv(h,core.pn)*ceildiv(f,core.pk)
                # Full core issue pipe and all global bandwidths are granted
                # to this one task. This is a genuine optimistic relaxation.
                coretime=issues*params.issue_interval
                dependency=max(ceildiv(h,core.pk),ceildiv(f,core.pk))*(params.dot_latency(core)+1)
                fastest=min(fastest,max(coretime,dependency))
        task_lb=max(task_lb,fastest)
    terms={**base["terms"],"largest_task":task_lb}
    if intervals and "z_bytes" in intervals:
        zmax=min(384*1024,max(b for a,b in intervals["z_bytes"]))
        wmax=min(40*1024,sum(b for a,b in intervals.get("w_bytes",[(0,40*1024)])))
        terms["HBM_rowchunk_regional"]=rowchunk_hbm_floor_bytes(workload,max_z_bytes=zmax,max_retained_w_bytes=wmax)/params.hbm_bandwidth
    tag=max(terms,key=terms.get)
    return {"lb_cycles":terms[tag],"binding_term":tag,"terms":terms,
            "reload_rule":"optimistic maxima only; universal unique floor when no tighter theorem",
            "forced_rule":"no merely-slower-core forcing"}


def paired_geomean(values: Iterable[float],references: Iterable[float]) -> float:
    pairs=list(zip(values,references))
    if not pairs or any(a<=0 or b<=0 for a,b in pairs):
        raise ValueError("positive nonempty paired measurements required")
    return math.exp(sum(math.log(a/b) for a,b in pairs)/len(pairs))



def search_workloads(*args,**kwargs):
    """Lazy public entrypoint; avoids circular model/assignment imports."""
    try:
        from .search import search_workloads as run
    except ImportError:
        import importlib.util,sys
        from pathlib import Path
        name="round2_draft_search"
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name("search.py"))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        run=sys.modules[name].search_workloads
    return run(*args,**kwargs)
