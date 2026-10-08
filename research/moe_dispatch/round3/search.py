"""Full declared-domain certificates with deterministic equal-work witnesses.

The SRAM domain is a 532-KiB composition, not four frozen category totals.
Unvisited lattice points remain in explicit open interval regions. A finite
candidate budget is never called exhaustive optimization or proof B closure.
No wall-clock deadline is used; the same node/evaluation budget is offered to
every open family. Proof A is the absence of a 5% improvement over the family
incumbent; proof B requires exact closure of the physical LPT objective.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import random

from research.moe_dispatch.geometry3d.compute import Core, enumerate_geometries
from .model import Design, KIB
from .optimizer import evaluate_design, universal_bound, region_bound

FAMILIES = ("single", "homogeneous", "5+1", "4+2", "3+3")
FAMILY_LABELS = {"single": "B1", "homogeneous": "B2", "5+1": "H51", "4+2": "H42", "3+3": "H33"}
FLOWS = ("OS", "WS", "IS")
CAPACITY_KIB = 532
BANK_TOTALS = (64, 24, 12, 64)
BANK_FIELDS = ("w_banks", "x_banks", "acc_banks", "vector_lanes")


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def key(d):
    return canonical(asdict(replace(d, label="")))


def decode(x):
    if isinstance(x, Design):
        return x
    x = dict(x.get("design", x))
    x["cores"] = tuple(Core(**c) if isinstance(c, dict) else Core(*c) for c in x["cores"])
    return Design(**x)


def gm(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def matches(cores, family):
    if family == "single":
        return len(cores) == 1
    if family == "homogeneous":
        return len(cores) == 2 and cores[0] == cores[1]
    return len(cores) == 2 and min(c.macs for c in cores) == {"5+1": 2048, "4+2": 4096, "3+3": 6144}[family]


@lru_cache(None)
def geometries(family):
    return tuple(g for g in enumerate_geometries() if matches(g, family))


@lru_cache(maxsize=100000)
def composition_count(intervals, total=CAPACITY_KIB):
    """Exact bounded-composition count via inclusion/exclusion, <=8 variables."""
    shifted = total - sum(a for a, b in intervals)
    if shifted < 0:
        return 0
    n = len(intervals)
    ans = 0
    for mask in range(1 << n):
        upper = sum(intervals[i][1] - intervals[i][0] + 1 for i in range(n) if mask & (1 << i))
        left = shifted - upper
        if left >= 0:
            ans += (-1 if mask.bit_count() & 1 else 1) * math.comb(left + n - 1, n - 1)
    return ans


@dataclass(frozen=True)
class Region:
    family: str
    indices: tuple[int, ...]
    landing: str
    flows: tuple[tuple[str, ...], ...]
    capacities: tuple[tuple[int, int], ...]
    banks: tuple[tuple[int, int], ...]
    depth: int = 0

    @property
    def cardinality(self):
        return (len(self.indices) * math.prod(map(len, self.flows))
                * composition_count(self.capacities) * math.prod(b - a + 1 for a, b in self.banks))

    @property
    def singleton(self):
        return self.cardinality == 1


def roots(family):
    n = len(geometries(family)[0])
    bank = () if n == 1 else tuple((1, total - 1) for total in BANK_TOTALS)
    landings = ("private",) if n == 1 else ("private", "shared")
    return [Region(family, tuple(range(len(geometries(family)))), landing,
                   (FLOWS,) * n,
                   ((1, CAPACITY_KIB - (4*n if landing == "private" else 1+3*n) + 1),)
                   * (4*n if landing == "private" else 1+3*n), bank)
            for landing in landings]


def _capacity_names(r):
    n = len(r.flows)
    if r.landing == "private":
        return tuple(f"{field}_{c}" for field in ("w", "x", "acc", "z") for c in range(n))
    return ("landing_pool",) + tuple(f"{field}_{c}" for field in ("x", "acc", "z") for c in range(n))


def describe(r):
    return {"family": r.family, "geometry_indices": list(r.indices), "landing_mode": r.landing,
            "flow_options": r.flows, "capacity_intervals_KiB": dict(zip(_capacity_names(r), r.capacities)),
            "capacity_sum_KiB": CAPACITY_KIB, "bank_cut_intervals": dict(zip(BANK_FIELDS, r.banks)),
            "cardinality": r.cardinality, "depth": r.depth}


def split(r):
    if len(r.indices) > 1:
        mid = len(r.indices) // 2
        return (replace(r, indices=r.indices[:mid], depth=r.depth+1),
                replace(r, indices=r.indices[mid:], depth=r.depth+1))
    for j, options in enumerate(r.flows):
        if len(options) > 1:
            mid = len(options) // 2
            a, b = list(r.flows), list(r.flows)
            a[j], b[j] = options[:mid], options[mid:]
            return replace(r, flows=tuple(a), depth=r.depth+1), replace(r, flows=tuple(b), depth=r.depth+1)
    options = [(b-a, "capacity", j) for j, (a, b) in enumerate(r.capacities) if b > a]
    options += [(8*(b-a), "bank", j) for j, (a, b) in enumerate(r.banks) if b > a]
    if not options:
        return ()
    _, kind, j = max(options)
    values = r.capacities if kind == "capacity" else r.banks
    a, b = values[j]
    mid = (a+b)//2
    left, right = list(values), list(values)
    left[j], right[j] = (a, mid), (mid+1, b)
    field = "capacities" if kind == "capacity" else "banks"
    children = [replace(r, **{field: tuple(x)}, depth=r.depth+1) for x in (left, right)]
    return tuple(x for x in children if x.cardinality)


def c1_legal(d):
    if d.landing_mode == "shared":
        available = d.landing_pool_bytes - sum(c.w_slice_bytes for c in d.cores)
        return available >= 16384
    return all((d.w_bytes[i] // c.w_slice_bytes - 1) * c.w_slice_bytes >= 16384
               for i, c in enumerate(d.cores))


def _allocate(total, weights, minimum=1):
    n = len(weights)
    room = total - n*minimum
    v = [minimum + int(room*w/sum(weights)) for w in weights]
    for i in range(total - sum(v)):
        v[i % n] += 1
    return tuple(v)


@lru_cache(maxsize=128)
def _ordered_geometry_cache(family, workload_json):
    """Cache only the original pure geometry ranking, never evaluations.

    Ranking depends on these workloads and the family, not parameters or any
    solver/runtime state. The original arithmetic and tie order stay intact.
    Actual solver calls and both complete physical/search repeats still run.
    """
    workloads=json.loads(workload_json)
    gs = list(geometries(family))
    def geometry_proxy(cs):
        total = 0
        for w in workloads:
            loads = [0.0]*len(cs)
            es = sorted(w["experts"], key=lambda e: (-e["Me"], e["id"]))
            for e in es:
                times = [(math.ceil(e["Me"]/c.pm)*math.ceil(e["F"]/c.pn)*math.ceil(e["H"]/c.pk)*2
                          + math.ceil(e["Me"]/c.pm)*math.ceil(e["H"]/c.pn)*math.ceil(e["F"]/c.pk)) for c in cs]
                winner = min(range(len(cs)), key=lambda i: (loads[i]+times[i], i))
                loads[winner] += times[winner]
            total += max(loads)
        return total
    gs.sort(key=lambda cs: (geometry_proxy(cs), cs))
    return tuple(gs)


def seed_designs(family, workloads, *, constraint="C0", seed=20261007):
    """Deterministic incumbent witnesses, without reducing the proof domain."""
    gs=list(_ordered_geometry_cache(family,canonical(workloads)))
    profiles = ((40, 12, 96, 384), (64, 32, 64, 372), (80, 64, 48, 340),
                (96, 96, 32, 308), (48, 48, 32, 404), (32, 32, 16, 452))
    # Interleave geometry, capacity and landing changes. No held-out inputs.
    seen = set()
    n = len(gs[0])
    for round_id in range(12):
        for j, cs in enumerate(gs):
            schedule_index = round_id + j
            totals = profiles[schedule_index % len(profiles)]
            weights = tuple(c.macs for c in cs)
            capacity_weights = weights if schedule_index % 3 else tuple(reversed(weights))
            cap = {field: tuple(x*KIB for x in _allocate(total, capacity_weights))
                   for field, total in zip(("w_bytes", "x_bytes", "acc_bytes", "z_bytes"), totals)}
            bankweights = (1,)*n if schedule_index % 2 else weights
            banks = {field: _allocate(total, bankweights) for field, total in zip(BANK_FIELDS, BANK_TOTALS)}
            flow = ("WS",)*n if schedule_index % 12 < 8 else (("IS",)*n if schedule_index % 12 < 10 else ("OS",)*n)
            landing = "shared" if n == 2 and schedule_index % 2 else "private"
            if landing == "shared":
                cap.update(w_bytes=(0,)*n, landing_pool_bytes=totals[0]*KIB)
            else:
                # At least one physical tile per core; exchange from Z rather
                # than quietly changing the aggregate 532-KiB capacity.
                ws = list(cap["w_bytes"])
                zs = list(cap["z_bytes"])
                for i,c in enumerate(cs):
                    minimum = c.w_slice_bytes * (1 + math.ceil(16384/c.w_slice_bytes) if constraint == "C1" else 1)
                    need = max(0, math.ceil(minimum/KIB)*KIB-ws[i])
                    if zs[i]-need >= KIB:
                        ws[i] += need; zs[i] -= need
                cap.update(w_bytes=tuple(ws), z_bytes=tuple(zs))
            try:
                d = Design(cs, flows=flow, landing_mode=landing, **cap, **banks)
            except ValueError:
                continue
            if constraint == "C1" and not c1_legal(d):
                continue
            ident = key(d)
            if ident not in seen:
                seen.add(ident)
                yield d


def _evaluate(d, workloads, params, effort):
    rows=[]; calls=0
    try:
        for w in workloads:
            a = evaluate_design(w, d, params, max_seconds=effort, detail=False)
            calls += int(a["legal"])
            b = evaluate_design(w, d, params, max_seconds=effort, detail=False)
            calls += int(b["legal"])
            if canonical(a) != canonical(b):
                raise AssertionError("allocation / physical replay repeat mismatch")
            if not a["legal"]:
                return None, "proved_physical_illegal", [], calls
            floor = universal_bound(w, params)["lb_cycles"]
            if a["milp_sched"]["cycles"]+1e-5 < floor:
                raise AssertionError("enlarged-domain mandatory bound violated")
            rows.append(a)
    except ValueError as ex:
        return None, "physical_reject:"+str(ex), [], calls
    times=[r["milp_sched"]["latency_ms"] for r in rows]
    return {"design": asdict(d), "geometry": d.geometry, "score_ms": gm(times),
            "latencies_ms": times, "repeat_identical": True,
            "allocation_optimal": all(r["assignment"]["optimal"] for r in rows),
            "solver_statuses": [r["assignment"]["status"] for r in rows],
            "scope": "best evaluated physical MILP allocation + LPT witness"}, "evaluated", rows, calls


def representative(r):
    cores=geometries(r.family)[r.indices[len(r.indices)//2]]
    n=len(cores)
    caps=[a for a,b in r.capacities]
    left=CAPACITY_KIB-sum(caps)
    for i,(a,b) in enumerate(r.capacities):
        add=min(left,b-a)
        caps[i]+=add;left-=add
    if left:
        raise ValueError("empty bounded composition")
    values=dict(zip(_capacity_names(r),caps))
    fields={field+"_bytes":tuple(values[field+"_"+str(i)]*KIB for i in range(n))
            for field in (("w","x","acc","z") if r.landing=="private" else ("x","acc","z"))}
    if r.landing=="shared":
        fields.update(w_bytes=(0,)*n,landing_pool_bytes=values["landing_pool"]*KIB)
    for field,total,interval in zip(BANK_FIELDS,BANK_TOTALS,r.banks):
        value=interval[0]
        fields[field]=(value,total-value)
    return Design(cores,flows=tuple(x[0] for x in r.flows),landing_mode=r.landing,**fields)


def search_family(workloads, params, family, *, constraint="C0", candidate_budget=256,
                  node_budget=2048, solver_effort=10.0, initial_designs=(), seed=20261007):
    if constraint not in ("C0", "C1"):
        raise ValueError("unknown constraint set")
    if candidate_budget < 1 or node_budget < 1:
        raise ValueError("positive identical deterministic budgets required")
    evaluated={}; witnesses=[]; best=None; simulations=0
    def evaluate(d):
        nonlocal best, simulations
        if not matches(d.cores, family) or (constraint=="C1" and not c1_legal(d)):
            return
        k=key(d)
        if k in evaluated or len(evaluated)>=candidate_budget:
            return
        row,status,raw,calls=_evaluate(d,workloads,params,solver_effort)
        simulations += calls
        evaluated[k]=status
        witnesses.append({"design":json.loads(k),"status":status,**(row or {})})
        if row and (best is None or (row["score_ms"],k)<(best["score_ms"],key(decode(best["design"])))):
            best=row
    initial=[decode(x) for x in initial_designs]
    for d in itertools.chain(initial,seed_designs(family,workloads,constraint=constraint,seed=seed)):
        if len(evaluated)>=candidate_budget:
            break
        evaluate(d)
    if best is None:
        raise RuntimeError("family has no legal repeated executable witness: "+family+"/"+constraint)
    rootfloor=gm(universal_bound(w,params)["lb_cycles"]/1e6 for w in workloads)
    pending=[];serial=0;covered=0;pruned=0;nodes=0;cert=[]
    rootregions=roots(family)
    declared=sum(r.cardinality for r in rootregions)
    for r in rootregions:
        serial+=1;heapq.heappush(pending,(rootfloor,serial,r))
    while pending and nodes<node_budget:
        lb,_,r=heapq.heappop(pending);nodes+=1
        if lb>=best["score_ms"]-max(1e-12,best["score_ms"]*1e-12):
            covered+=r.cardinality;pruned+=r.cardinality
            cert.append({"status":"exact_bound_pruned","lb_ms":lb,**describe(r)})
            continue
        if r.singleton:
            # An exhausted finite candidate budget leaves this point live.
            if len(evaluated)>=candidate_budget:
                serial+=1;heapq.heappush(pending,(lb,serial,r));break
            try:
                d=representative(r)
                if constraint=="C1" and not c1_legal(d):
                    covered+=1;pruned+=1
                    cert.append({"status":"C1_infeasible_leaf",**describe(r)})
                    continue
                evaluate(d)
                status=evaluated.get(key(d),"")
                row=next((x for x in witnesses if key(decode(x["design"]))==key(d)),None)
                if status.startswith("physical_reject") or status=="proved_physical_illegal" or (row and row.get("allocation_optimal")):
                    covered+=1
                    cert.append({"status":"resolved_leaf",**describe(r)})
                    continue
            except ValueError:
                covered+=1;pruned+=1
                cert.append({"status":"physically_infeasible_leaf",**describe(r)})
                continue
            serial+=1;heapq.heappush(pending,(lb,serial,r));break
        for child in split(r):
            childlb=rootfloor
            if len(child.indices)<=8:
                childlb=gm(region_bound(workloads,params,[geometries(family)[i] for i in child.indices]))
            serial+=1;heapq.heappush(pending,(childlb,serial,child))
    opens=[{"lb_ms":lb,**describe(r)} for lb,_,r in sorted(pending)]
    assert covered+sum(r[2].cardinality for r in pending)==declared
    lb=min([rootfloor,*[x["lb_ms"] for x in opens]])
    proof_a = rootfloor >= .95*best["score_ms"]
    proof_b = not pending
    result={"family":family,"constraint_group":constraint,"parameters":asdict(params),
            "selected":best,"witnesses":witnesses,"evaluated_points":len(evaluated),
            "successful_points":sum(r["status"]=="evaluated" for r in witnesses),
            "candidate_budget":candidate_budget,"node_budget":node_budget,"visited_nodes":nodes,
            "simulator_calls":simulations,"declared_lattice_points":declared,
            "pruned_lattice_points":pruned,"open_regions":opens,"certificate":cert,
            "global_lb_ms":lb,"gap_pct":100*(best["score_ms"]/lb-1),
            "proof_A_closed":proof_a,"proof_B_closed":proof_b,
            "proof_A_scope":"no design improves this family's executable incumbent by >=5%; conservative global mandatory floor",
            "proof_B_scope":"exact optimum of physical LPT replay objective only if full declared domain closed",
            "search_scope":"full enlarged domain; finite deterministic equal-work incumbent search; unvisited points stay open",
            "geometry_count":len(geometries(family)),"repeat_requirement":"two exact deterministic evaluations per point",
            "seed":seed,"no_wall_clock_timeout":True,
            "capacity_domain":"aggregate W+X+acc+Z+pool=532KiB, 1KiB lattice; category totals unconstrained",
            "constraint_domain_note":"C1 proof uses unconstrained superset roots, so lower bound is conservative; no points silently removed"}
    return result


def search_workloads(workloads, params, *, families=FAMILIES, **kwargs):
    result={f:search_family(workloads,params,f,**kwargs) for f in families}
    return {"families":result,"seed":kwargs.get("seed",20261007),
            "parameters":asdict(params),"development_window_ids":[w["id"] for w in workloads],
            "proof_complete":all(x["proof_B_closed"] for x in result.values())}
