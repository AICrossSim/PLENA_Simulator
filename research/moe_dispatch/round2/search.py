"""Full-domain interval BnB with conservative, inspectable certificates.

A time cap never shrinks the declared search domain. Initial/midpoint designs
are incumbent witnesses only. Every unevaluated lattice point remains in an
open interval region or a region removed by a legal lower-bound theorem.
Optimality pertains to the stated assignment-plus-LPT replay objective, not
an unconstrained clairvoyant temporal scheduler or physical RTL latency.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import math
import time
from pathlib import Path

try:
    from .model import Design,Parameters,Core,KIB
    from .optimizer import evaluate_design,region_bound,universal_bound
except ImportError:
    import importlib.util,sys
    from pathlib import Path
    for name,file in (("round2_draft_model","model.py"),("round2_draft_optimizer","optimizer.py")):
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
            m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    m=sys.modules["round2_draft_model"];o=sys.modules["round2_draft_optimizer"]
    Design,Parameters,Core,KIB=m.Design,m.Parameters,m.Core,m.KIB
    evaluate_design,region_bound,universal_bound=o.evaluate_design,o.region_bound,o.universal_bound

from research.moe_dispatch.geometry3d.compute import enumerate_geometries,enumeration_counts

FLOW_OPTIONS=("OS","WS","IS")
# core0 integer cut; core1 is the exact complement. SRAM cuts in 1 KiB.
RESOURCE_SPECS=(("w_bytes",40,KIB),("x_bytes",12,KIB),("acc_bytes",96,KIB),
                ("z_bytes",384,KIB),("w_banks",64,1),("x_banks",24,1),
                ("acc_banks",12,1),("vector_lanes",64,1))


def _finite(x):
    return x if x is not None and math.isfinite(x) else None


def _engine_hash():
    files=("model.py","optimizer.py","search.py")
    from research.moe_dispatch.geometry3d import compute as geometry_compute
    payload=b"".join(Path(__file__).with_name(x).read_bytes() for x in files)
    payload+=Path(geometry_compute.__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _workload_hash(workloads):
    return hashlib.sha256(json.dumps(workloads,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _gmean(xs):
    xs=list(xs)
    return math.exp(sum(math.log(max(1e-300,x)) for x in xs)/len(xs)) if xs else 0.0


def _key(d):
    return json.dumps(asdict(replace(d,label="")),sort_keys=True,separators=(",",":"))


def _decode(d):
    if isinstance(d,Design):return d
    d=dict(d)
    if "design" in d:return _decode(d["design"])
    d["cores"]=tuple(Core(**x) if isinstance(x,dict) else Core(*x) for x in d["cores"])
    return Design(**d)


@lru_cache(maxsize=1)
def _inventory():
    return tuple(enumerate_geometries())


def _matches(cs,family):
    if family=="single":return len(cs)==1
    if family=="homogeneous":return len(cs)==2 and cs[0]==cs[1]
    if family=="heterogeneous":return len(cs)==2 and cs[0]!=cs[1]
    if family=="5+1":return len(cs)==2 and min(c.macs for c in cs)==2048
    if family in ("4+2","2+4"):
        return len(cs)==2 and min(c.macs for c in cs)==4096
    raise ValueError("unknown search family: "+family)


@dataclass(frozen=True)
class Region:
    indices: tuple[int,...]
    flows: tuple[tuple[str,...],...]
    cuts: tuple[tuple[int,int],...]
    depth: int=0

    @property
    def cardinality(self):
        return len(self.indices)*math.prod(len(x) for x in self.flows)*math.prod(b-a+1 for a,b in self.cuts)

    @property
    def singleton(self):return self.cardinality==1


def _root(family):
    ids=tuple(i for i,cs in enumerate(_inventory()) if _matches(cs,family))
    if not ids:raise ValueError("family has no declared geometry")
    n=len(_inventory()[ids[0]])
    return Region(ids,(FLOW_OPTIONS,)*n,() if n==1 else tuple((1,total-1) for _,total,_ in RESOURCE_SPECS))


def _intervals(r):
    return {name:((a*unit,b*unit),((total-b)*unit,(total-a)*unit))
            for (name,total,unit),(a,b) in zip(RESOURCE_SPECS,r.cuts)}


def _range_encoding(ids):
    # Membership recoverable against the frozen exhaustive geometry inventory.
    if not ids:return []
    runs=[];first=last=ids[0]
    for i in ids[1:]:
        if i==last+1:last=i
        else:runs.append([first,last]);first=last=i
    runs.append([first,last]);return runs


def _describe(r):
    return {"geometry_index_ranges":_range_encoding(r.indices),"geometry_count":len(r.indices),
            "flow_options":r.flows,"resource_cut_intervals":{
                name:{"core0_min":a*unit,"core0_max":b*unit,"core1_total":total*unit,
                      "step":unit} for (name,total,unit),(a,b) in zip(RESOURCE_SPECS,r.cuts)},
            "lattice_points":r.cardinality,"depth":r.depth}


def _restore_region(x):
    ids=tuple(i for a,b in x["geometry_index_ranges"] for i in range(a,b+1))
    cuts=tuple((x["resource_cut_intervals"][name]["core0_min"]//unit,
                x["resource_cut_intervals"][name]["core0_max"]//unit)
               for name,_,unit in RESOURCE_SPECS) if x["resource_cut_intervals"] else ()
    return Region(ids,tuple(tuple(f) for f in x["flow_options"]),cuts,int(x["depth"]))


def _split(r):
    """Geometry first: MAC split -> PM split -> PN/PK. Then flows/resources."""
    if len(r.indices)>1:
        cs=[_inventory()[i] for i in r.indices]
        for getter in (lambda x:tuple(c.macs for c in x),lambda x:tuple(c.pm for c in x),
                       lambda x:tuple((c.pn,c.pk) for c in x)):
            vals=sorted(set(getter(x) for x in cs))
            if len(vals)>1:
                leftkeys=set(vals[:len(vals)//2])
                left=tuple(i for i in r.indices if getter(_inventory()[i]) in leftkeys)
                right=tuple(i for i in r.indices if getter(_inventory()[i]) not in leftkeys)
                return (replace(r,indices=left,depth=r.depth+1),replace(r,indices=right,depth=r.depth+1))
    for j,fs in enumerate(r.flows):
        if len(fs)>1:
            a=list(r.flows);b=list(r.flows);mid=len(fs)//2;a[j]=fs[:mid];b[j]=fs[mid:]
            return replace(r,flows=tuple(a),depth=r.depth+1),replace(r,flows=tuple(b),depth=r.depth+1)
    # Highest normalized cut width; capacity and bandwidth are independent.
    live=[j for j,(a,b) in enumerate(r.cuts) if a<b]
    if live:
        j=max(live,key=lambda q:((r.cuts[q][1]-r.cuts[q][0])/(RESOURCE_SPECS[q][1]-2),-q))
        a,b=r.cuts[j];mid=(a+b)//2;l=list(r.cuts);u=list(r.cuts);l[j]=(a,mid);u[j]=(mid+1,b)
        return replace(r,cuts=tuple(l),depth=r.depth+1),replace(r,cuts=tuple(u),depth=r.depth+1)
    return ()


def _representative(r):
    cs=_inventory()[r.indices[len(r.indices)//2]]
    fields={}
    for (name,total,unit),(a,b) in zip(RESOURCE_SPECS,r.cuts):
        x=(a+b)//2;fields[name]=(x*unit,(total-x)*unit)
    return Design(cs,flows=tuple(fs[0] for fs in r.flows),**fields)


def _bound(r,workloads,p,root_floors):
    # Cheap universal floor at broad regions; exact geometry task critical
    # relaxation once <=16 shapes remain. Choosing not to tighten a legal LB
    # is valid and explicitly reflected in the open-region certificate.
    if len(r.indices)>16:
        values=root_floors
        terms="universal mandatory resource floors"
    else:
        values=[region_bound(w,p,geometries=tuple(_inventory()[i] for i in r.indices),
                            intervals=_intervals(r))["lb_cycles"]/1e6 for w in workloads]
        terms="universal plus optimistic largest-task geometry floor; max regional buffers"
    return _gmean(values),terms


def _evaluate(d,workloads,p,max_seconds):
    results=[]
    try:
        for w in workloads:
            a=evaluate_design(w,d,p,max_seconds=max_seconds,detail=False)
            if not a["legal"]:return None,"an expert has no physical core"
            b=evaluate_design(w,d,p,max_seconds=max_seconds,detail=False)
            # Time-limited solver statuses can differ; only accepted completed
            # witnesses with deterministic execution/replay become incumbents.
            if json.dumps(a,sort_keys=True)!=json.dumps(b,sort_keys=True):
                return None,"two repeated assignment/replay results differ"
            results.append(a)
    except (ValueError,RuntimeError) as ex:
        return None,str(ex)
    if not results:return None,"empty workload set"
    times=[x["milp_sched"]["latency_ms"] for x in results]
    return {"design":asdict(d),"geometry":d.geometry,"flows":d.flows,"family":d.family,
            "score_ms":_gmean(times),"geomean_ms":_gmean(times),"latencies_ms":times,
            "runtime_latencies_ms":[x["runtime"]["latency_ms"] for x in results],
            "lb_ms":[x["lb_cycles"]/1e6 for x in results],
            "allocations_optimal":all(x["assignment"]["optimal"] for x in results),
            "allocation_statuses":[x["assignment"]["status"] for x in results],
            "solver_algorithms":[x["assignment"].get("solver_algorithm","CP-SAT") for x in results],
            "enumeration_state_counts":[x["assignment"].get("enumeration",{}) for x in results],
            "repeat_identical":True,"parameters":asdict(p),"workload_sha256":_workload_hash(workloads),"engine_sha256":_engine_hash(),"schedule_objective":"exact integer-assignment witness replayed by finite LPT streaming"},None


def _warmstarts(families):
    # Witnesses only, never a replacement for the full roots.
    singles=((Core(6,16,128),),(Core(16,24,32),),(Core(12,8,128),))
    dual=((Core(3,16,128),Core(3,16,128)),
          (Core(4,16,128),Core(2,16,128)),
          (Core(5,8,256),Core(1,8,256)))
    out=[]
    for cs in (*singles,*dual):
        if any(_matches(cs,f) for f in families):out.append(Design(cs,flows=("WS",)*len(cs)))
    # One initial witness per required family before refinements.
    return sorted(out,key=lambda d:(min(i for i,f in enumerate(families) if _matches(d.cores,f)),d.geometry))


def search_workloads(workloads,params=Parameters(),*,delta=.05,time_limit_s=120.0,
                     initial_designs=(),seed=20261007,
                     target_families=("single","homogeneous","heterogeneous"),
                     max_nodes=None,solver_seconds=2.0,resume_state=None) -> dict:
    """Search the entire declared positive-cut integer domain, with a time cap.

    ``max_nodes`` supports deterministic tests/replays independent of machine
    load. A wall cap may stop at different nodes; completed-point repetitions
    remain deterministic. Geometric means use one frozen full design across
    all windows, never a per-window geometry oracle.
    """
    if delta<0 or time_limit_s<=0 or not workloads:raise ValueError("nonempty workloads, nonnegative delta, positive cap")
    families=tuple(dict.fromkeys(target_families));start=time.monotonic();deadline=start+time_limit_s
    workload_sha=_workload_hash(workloads)
    roots={f:_root(f) for f in families}
    rootfloors=[universal_bound(w,params)["lb_cycles"]/1e6 for w in workloads]
    rootlb=_gmean(rootfloors)
    best={f:None for f in families};evaluated={};leaves=[];cert=[]
    serial=0;frontiers={f:[] for f in families};covered={f:0 for f in families};removed_lbs={f:[] for f in families}
    nodes=0
    def improve(row):
        if row is None:return
        d=_decode(row["design"])
        for f in families:
            if _matches(d.cores,f) and (best[f] is None or row["geomean_ms"]<best[f]["geomean_ms"]):
                best[f]=dict(row,search_family=f)
    def evalpoint(d,f=None,region=None):
        key=_key(d)
        if key not in evaluated:
            row,reason=_evaluate(d,workloads,params,solver_seconds)
            evaluated[key]=(row,reason)
            leaves.append({"family":f or d.family,"geometry":d.geometry,"design":key,
                "geomean_ms":row["geomean_ms"] if row else None,
                "status":"evaluated" if row else "invalid_or_unresolved",
                "reason":reason,"repeat_identical":bool(row),
                "allocations_optimal":row["allocations_optimal"] if row else False})
        row,reason=evaluated[key];improve(row)
        return row,reason
    # Initial frozen designs may already have been evaluated by caller. Accept
    # cached full evaluation rows only with their explicit two-repeat flag.
    initials=[]
    for x in initial_designs:
        if (isinstance(x,dict) and x.get("repeat_identical") and "latencies_ms" in x and
                x.get("parameters")==asdict(params) and x.get("workload_sha256")==workload_sha and
                x.get("engine_sha256")==_engine_hash()):
            improve(dict(x,score_ms=x.get("score_ms",x["geomean_ms"])))
            evaluated[_key(_decode(x))]=(dict(x),None)
        else:initials.append(_decode(x))
    if resume_state is not None:
        state=resume_state.get("resume",resume_state)
        if state.get("workload_sha256")!=workload_sha or state.get("parameters")!=asdict(params) or state.get("delta")!=delta or state.get("engine_sha256")!=_engine_hash():
            raise ValueError("resume requires identical frozen workloads, parameters, and delta")
        if tuple(state.get("families",()))!=families:
            raise ValueError("resume requires identical target families")
        for f,row in state["incumbents"].items():
            if row:improve(row)
        for key,item in state.get("evaluated_points",{}).items():evaluated[key]=tuple(item)
        leaves.extend(state.get("leaves",[]));cert.extend(state.get("certificate",[]))
        for f in families:
            covered[f]=int(state["covered_lattice_points"][f])
            low=state["removed_lb_minima"].get(f)
            removed_lbs[f]=[] if low is None else [low]
        for x in state["open_regions"]:
            f=x["family"];r=_restore_region(x);serial+=1
            lb,method=_bound(r,workloads,params,rootfloors)
            heapq.heappush(frontiers[f],(lb,serial,r,method))
    initials.extend(_warmstarts(families))
    # Fair first incumbent acquisition, then remaining seeds under same cap.
    prioritized=[];seen=set()
    for f in families:
        for d in initials:
            if _matches(d.cores,f) and _key(d) not in seen:
                prioritized.append(d);seen.add(_key(d));break
    for d in initials:
        if _key(d) not in seen:prioritized.append(d);seen.add(_key(d))
    for d in prioritized:
        if resume_state is not None and _key(d) in evaluated:continue
        # Every requested family gets a fully repeated concrete witness.
        # This setup may exceed a very short traversal budget; report that
        # overrun rather than returning missing optima as numerical results.
        needed=any(best[f] is None and _matches(d.cores,f) for f in families)
        if time.monotonic()>=deadline and not needed:continue
        evalpoint(d)
    setup_seconds=time.monotonic()-start
    traversal_start=time.monotonic()
    deadline=traversal_start+time_limit_s
    if resume_state is None:
        for f,r in roots.items():
            lb,method=_bound(r,workloads,params,rootfloors);serial+=1
            heapq.heappush(frontiers[f],(lb,serial,r,method))
    turn=0
    while any(frontiers.values()) and time.monotonic()<deadline and (max_nodes is None or nodes<max_nodes):
        nonempty=[f for f in families if frontiers[f]]
        f=nonempty[turn%len(nonempty)];turn+=1
        lb,_,r,method=heapq.heappop(frontiers[f]);nodes+=1
        incumbent=best[f]["geomean_ms"] if best[f] else math.inf
        if lb>=incumbent/(1+delta):
            covered[f]+=r.cardinality;removed_lbs[f].append(lb)
            cert.append({"family":f,"status":"lower_bound_pruned","lb_ms":_finite(lb),"region_physically_infeasible":math.isinf(lb),
                         "incumbent_ms":_finite(incumbent),"delta":delta,"bound_method":method,**_describe(r)})
            continue
        if r.singleton:
            row,reason=evalpoint(_representative(r),f,r)
            if (row is not None and row["allocations_optimal"]) or (reason and ("physical" in reason or "exceeds" in reason or "cannot fit" in reason)):
                covered[f]+=1;removed_lbs[f].append(lb if row is None else row["geomean_ms"])
                cert.append({"family":f,"status":"evaluated_leaf" if row else "proved_infeasible_leaf",
                             "lb_ms":_finite(lb),"region_physically_infeasible":math.isinf(lb),"incumbent_ms":best[f]["geomean_ms"] if best[f] else None,**_describe(r)})
            else:
                # A failed/time-limited solver evaluation does NOT delete a point.
                serial+=1;heapq.heappush(frontiers[f],(lb,serial,r,method));break
            continue
        # Evaluate a finite representative only when geometry and flows are
        # fixed. The rest remains live until split/pruned/proved.
        if len(r.indices)==1 and all(len(x)==1 for x in r.flows) and r.depth%8==0:
            evalpoint(_representative(r),f,r)
        for child in _split(r):
            clb,cmethod=_bound(child,workloads,params,rootfloors);serial+=1
            heapq.heappush(frontiers[f],(clb,serial,child,cmethod))
    output={};allcomplete=True;allopen=[]
    for f in families:
        openlist=[{"lb_ms":_finite(lb),"region_physically_infeasible":math.isinf(lb),"bound_method":method,**_describe(r)} for lb,_,r,method in sorted(frontiers[f])]
        open_points=sum(x["lattice_points"] for x in openlist)
        assert covered[f]+open_points==roots[f].cardinality,"full-domain coverage conservation"
        openlb=min((x["lb_ms"] for x in openlist if x["lb_ms"] is not None),default=None)
        low=min([x for x in [openlb,*removed_lbs[f]] if x is not None],default=rootlb)
        incumbent=best[f]["geomean_ms"] if best[f] else None
        complete=not openlist
        allcomplete &= complete
        status=("delta_certified_full_declared_domain" if complete and delta else
                "exact_fixed_replay_objective_full_declared_domain" if complete else "time_or_node_limited_open_regions")
        stats={"proof_status":status,"proof_complete":complete,"root_lb_ms":rootlb,
               "open_lb_ms":openlb,"certified_global_lb_ms":_finite(low),
               "gap_pct":max(0.0,100*(incumbent/low-1)) if incumbent is not None and low and math.isfinite(low) else None,
               "covered_lattice_points":covered[f],"declared_lattice_points":roots[f].cardinality,
               "coverage_pct":100*covered[f]/roots[f].cardinality,"open_regions":openlist,
               "geometry_count":len(roots[f].indices)}
        output[f]={**(best[f] or {"design":None,"geomean_ms":None,"score_ms":None}),**stats}
        allopen.extend(dict(x,family=f) for x in openlist)
    incumbents=[x["geomean_ms"] for x in output.values() if x["geomean_ms"] is not None]
    globalinc=min(incumbents,default=None)
    # Universal resource floors apply to all three primary families. This
    # statement is separate from certification of every family optimum.
    global_delta_proof=globalinc is not None and rootlb>=globalinc/(1+delta)
    geomhash=hashlib.sha256(json.dumps([[[c.pm,c.pn,c.pk] for c in cs] for cs in _inventory()],separators=(",",":")).encode()).hexdigest()
    return {"families":output,"proof_complete":bool(allcomplete),"delta":delta,
            "root_lb_ms":rootlb,"open_lb_ms":min((x["lb_ms"] for x in allopen if x["lb_ms"] is not None),default=None),
            "gap_pct":100*(globalinc/rootlb-1) if globalinc is not None and rootlb else None,
            "global_delta_proof":global_delta_proof,"global_incumbent_ms":globalinc,
            "global_proof_warning":"A global incumbent/resource proof does not certify each family optimum",
            "elapsed_seconds":time.monotonic()-start,"time_limit_s":time_limit_s,
            "setup_seconds":setup_seconds,"setup_overrun_seconds":max(0.0,setup_seconds-time_limit_s),
            "traversal_seconds":time.monotonic()-traversal_start,
            "time_cap_scope":"mandatory repeated incumbent setup followed by separately capped full-domain traversal",
            "nodes_processed":nodes,"concrete_points_evaluated":len(evaluated),
            "certificate":cert,"leaves":leaves,"geometry_counts":enumeration_counts(),
            "geometry_inventory_sha256":geomhash,"seed":seed,
            "scope":"entire declared geometry x dataflow x independent private-resource lattice; finite assignment/LPT replay objective",
            "resource_cut_domain":{name:{"positive_core0_units":[1,total-1],"step_bytes_or_units":unit,"total":total*unit} for name,total,unit in RESOURCE_SPECS},
            "domain_warning":"Full positive-cut lattice includes physically infeasible points; no shortlist is called exhaustive",
            "vector_assumption":"existing64 BF16 vector lanes, positive integer split; additional frozen finite-resource assumption",
            "mirror_warning":"4+2 and2+4 MAC-ratio families are physical mirror aliases when roles and private resources can swap",
            "resume":{"open_regions":allopen,"workload_sha256":workload_sha,"engine_sha256":_engine_hash(),"parameters":asdict(params),"delta":delta,
                      "families":families,"incumbents":best,"covered_lattice_points":covered,
                      "removed_lb_minima":{f:_finite(min(removed_lbs[f],default=None)) for f in families},
                      "evaluated_points":evaluated,"leaves":leaves,"certificate":cert,
                      "note":"pass entire result or resume object as resume_state to continue exact open intervals with the same frozen scope"}}
