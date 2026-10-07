"""Exact CP-SAT assignment relaxation and legal regional lower bounds.

The solver does NOT solve a coupled temporal schedule. It minimizes the
assignment's resource/critical-path relaxation; its owners are replayed by
the finite streaming simulator. Cold-task isolated latencies are never
summed. Fractional service demands are rounded DOWN, so the CP-SAT optimum
remains a conservative continuous-time lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
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
    """Never round a physical service requirement upward."""
    if value<=0:
        return 0
    return max(0,math.floor(math.nextafter(value/quantum,-math.inf)))


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


def solve_assignment(workload: dict,design: Design,params: Parameters=Parameters(),
                     *,max_seconds: float=10.0,quantum: float=1e-6) -> dict:
    """Exact integer assignment relaxation, solved with one deterministic worker.

    Returns lb_cycles, owners, status, quantum, and explicit solver/gap scope.
    A work-limited FEASIBLE result uses BestObjectiveBound for its lower bound
    and the available assignment for executable replay. INFEASIBLE never
    fabricates an assignment. Task types with identical costs are aggregated
    into integer counts; this preserves the assignment feasible set exactly.

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
                   "wall_clock_timeout_seconds":None,"num_search_workers":1,"random_seed":20261007}
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
    grouped_costs=[table[ids[0]] for _,ids in groups]
    # Bound the integer objective using a deterministic legal assignment.
    greedy=[min((c for c,co in enumerate(row) if co is not None),
                key=lambda c:(row[c].isolated_cycles,c)) for row in table]
    upper=allocation_objective(workload,design,params,greedy)
    upper_tick=max(1,math.ceil(upper/quantum)+n+1)
    model=cp_model.CpModel()
    T=model.NewIntVar(0,upper_tick,"T")
    ns={}
    resource_terms={}
    for g,((_,ids),costrow) in enumerate(zip(groups,grouped_costs)):
        count=len(ids)
        choices=[]
        for c,co in enumerate(costrow):
            if co is None:
                continue
            var=model.NewIntVar(0,count,f"n_{g}_{c}")
            ns[g,c]=var
            choices.append(var)
            used=model.NewBoolVar(f"used_{g}_{c}")
            model.Add(var>=used)
            model.Add(var<=count*used)
            model.Add(T>=_down(co.dependency_floor,quantum)*used)
            for r,service in _resource_services(co,design,c,params).items():
                coeff=_down(service,quantum)
                if coeff:
                    resource_terms.setdefault(r,[]).append(coeff*var)
        model.Add(sum(choices)==count)
    for resource,terms in resource_terms.items():
        constant=_down(_common_spill(workload)/params.hbm_bandwidth,quantum) if resource=="hbm" else 0
        model.Add(sum(terms)+constant<=T)
    model.Minimize(T)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=20261007
    solver.parameters.max_deterministic_time=deterministic_limit
    solver.parameters.cp_model_presolve=True
    status_code=solver.Solve(model)
    status=solver.StatusName(status_code)
    if status_code not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        # Greedy is a legal assignment of the relaxation; no completed
        # optimal solve is claimed when timeout occurs before a witness.
        owners=greedy
        bound=max(0.0,solver.BestObjectiveBound())*quantum
    else:
        owners=[None]*n
        for g,(_,ids) in enumerate(groups):
            cursor=0
            for c in range(len(design.cores)):
                if (g,c) not in ns:
                    continue
                count=int(solver.Value(ns[g,c]))
                for i in ids[cursor:cursor+count]:
                    owners[i]=c
                cursor+=count
        assert all(c is not None for c in owners)
        bound=max(0.0,solver.BestObjectiveBound())*quantum
    # A FEASIBLE solver incumbent can retain slack in the objective T.
    # Replay/gaps must use the actual physical resource load of the immutable
    # owner witness, not that incidental solver variable value.
    objective=allocation_objective(workload,design,params,owners)
    return {"lb_cycles":bound,"owners":owners,"status":status,"quantum":quantum,
            "objective_upper_cycles":objective,"optimal":status_code==cp_model.OPTIMAL,
            "assignment_gap_cycles":max(0.0,objective-bound),"aggregated_task_types":len(groups),
            "quantization_direction":"all resource/critical coefficients rounded down",
            "quantization_loss_bound_cycles":(n+1)*quantum,
            "storage_chunks":len(storage_chunks(workload)),"common_activation_spill_bytes":_common_spill(workload),
            "solver_budget":solver_budget,
            "scope":"exact quantized assignment/resource relaxation; not an optimal temporal schedule"}


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
