"""Exact whole-expert assignment DSE with certified fractional-bound pruning.

This is a prospective analytical compute ceiling, anchored to the research
PLENA dot20/commit1/II1 contract. No HBM, SRAM ports/capacity, vector or
dispatch service enters the objective. Large shared SRAM is a requirement to
report, not a free claim about physical feasibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from .assignment_dp import exact_two_machine

from research.moe_dispatch.geometry3d.robust_compute import BUDGET, Shape as OldShape, gid
from research.moe_dispatch.geometry3d.compute import TimingProfile

FAMILIES = ("single", "homogeneous", "heterogeneous")
BATCHES = (2, 4, 8, 16, 64, 96, 128)
PROFILES = ("flat20", "log_stage1", "log_stage2")
KS = tuple(range(16, 2049, 16))


class Shape(OldShape):
    def __post_init__(self):
        if self.pm < 1 or self.pn < 1 or self.pk not in KS or self.macs > BUDGET:
            raise ValueError('positive M/N, 16-aligned K[16,2048], exact MAC budget')


class ComputeTiming(TimingProfile):
    def __post_init__(self):
        if tuple(k for k,_ in self.latency_by_pk) != KS:
            raise ValueError('timing missing a supported K')


def timing(name):
    slope = {'flat20':0,'log_stage1':1,'log_stage2':2}[name]
    return ComputeTiming(name,tuple((k,20+slope*((k-1).bit_length()-9)) for k in KS),
                         hypothesis='research PLENA PK512 dot20 anchor, non-power2 tree padded to ceil(log2 K)')


@lru_cache(maxsize=1)
def domain():
    shapes=tuple(sorted(Shape(m,n,k) for k in KS for m in range(1,BUDGET//k+1)
                        for n in range(1,BUDGET//(m*k)+1)))
    by_budget=defaultdict(list)
    for i,c in enumerate(shapes): by_budget[c.macs].append(i)
    pairs=[(i,-1) for i in by_budget[BUDGET]]
    pairs.extend((i,j) for i,c in enumerate(shapes) for j in by_budget[BUDGET-c.macs] if i<=j)
    indices=np.array(pairs,dtype=np.int64)
    family=np.where(indices[:,1]<0,0,np.where(indices[:,0]==indices[:,1],1,2))
    return shapes,indices,family


def costs(shapes,signatures,profile):
    pm,pn,pk=(np.array([getattr(c,x) for c in shapes])[:,None] for x in ('pm','pn','pk'))
    m,h,f=(np.array([s[i] for s in signatures])[None,:] for i in range(3))
    nm=(m+pm-1)//pm
    guq=2*nm*((f+pn-1)//pn); dnq=nm*((h+pn-1)//pn)
    hs=(h+pk-1)//pk; fs=(f+pk-1)//pk
    l=np.array([timing(profile).completion_latency(c) for c in shapes])[:,None]
    gu=(hs-1)*np.maximum(guq,l)+guq-1+l
    dn=(fs-1)*np.maximum(dnq,l)+dnq-1+l
    return gu+dn


@lru_cache(maxsize=100_000)
def expert_cost(m,h,f,c,profile):
    nm=math.ceil(m/c.pm)
    guq=2*nm*math.ceil(f/c.pn); dnq=nm*math.ceil(h/c.pn)
    hs,fs=math.ceil(h/c.pk),math.ceil(f/c.pk)
    l=timing(profile).completion_latency(c)
    gu=(hs-1)*max(guq,l)+guq-1+l
    dn=(fs-1)*max(dnq,l)+dnq-1+l
    issues=guq*hs+dnq*fs
    return {'cycles':gu+dn,'gate_up_cycles':gu,'down_cycles':dn,
            'useful_macs':3*m*h*f,'issued_macs':issues*c.macs,'issues':issues,
            'required_live_FP32_output_bytes':max(guq,dnq)*c.record_bytes}


def scalar_expert(e,c,profile,records=0):
    assert records==0
    return expert_cost(e['Me'],e['H'],e['F'],c,profile)


def vector_schedule(windows,signatures,cost,indices):
    mapping={s:i for i,s in enumerate(signatures)}
    a,b=indices[:,0],np.maximum(indices[:,1],0)
    single=indices[:,1]<0
    out=np.empty((len(indices),len(windows)),dtype=np.int64)
    for wi,w in enumerate(windows):
        load0=np.zeros(len(indices),dtype=np.int64); load1=load0.copy()
        for e in sorted(w['experts'],key=lambda x:(-3*x['Me']*x['H']*x['F'],x['id'])):
            j=mapping[e['Me'],e['H'],e['F']]
            c0,c1=cost[a,j],cost[b,j]
            owner0=single|(load0+c0<=load1+c1)
            load0+=np.where(owner0,c0,0); load1+=np.where(owner0,0,c1)
        out[:,wi]=np.maximum(load0,load1)
    return out


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   sort_keys=True, allow_nan=False) + "\n")


def csv_out(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def balanced(v, windows):
    """Same weight per batch and per window inside each batch."""
    v = np.asarray(v)
    b = np.array([w["batch"] for w in windows])
    return np.exp(np.stack([np.log(v[..., b == k]).mean(axis=-1)
                            for k in sorted(set(b))]).mean(axis=0))


def grouped_window(window):
    groups = defaultdict(list)
    for e in sorted(window["experts"], key=lambda x: x["id"]):
        if e["Me"]:
            groups[e["Me"], e["H"], e["F"]].append(e)
    return sorted(groups), groups


def fractional_bounds(c0, c1, counts):
    """Exact fractional two-machine relaxation plus indivisible-job bound.

    Rows are geometries; columns are task types. Transfer tasks from machine
    1 to machine 0 in increasing c0/c1 order until loads meet. One type may
    be fractional. Every integer assignment is bounded by this relaxation.
    """
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    counts = np.asarray(counts)
    order = np.argsort(c0 / c1, axis=1, kind="stable")
    a, b = np.take_along_axis(c0, order, 1), np.take_along_axis(c1, order, 1)
    n = counts[order]
    sa = np.cumsum(a * n, axis=1)
    sb = np.sum(b * n, axis=1)[:, None] - np.cumsum(b * n, axis=1)
    j = np.argmax(sa >= sb, axis=1)
    r = np.arange(len(c0))
    p = np.maximum(j - 1, 0)
    before_a = np.where(j > 0, sa[r, p], 0)
    before_b = np.where(j > 0, sb[r, p], np.sum(b * n, axis=1))
    take = (before_b - before_a) / (a[r, j] + b[r, j])
    lp = before_a + a[r, j] * take
    return np.maximum(lp, np.minimum(c0, c1).max(axis=1))


@lru_cache(maxsize=100_000)
def optimal_counts(c0, c1, counts, base0=0, base1=0):
    """Integer counts per type; exact within whole-expert ownership contract."""
    c0, c1, counts = tuple(c0), tuple(c1), tuple(counts)
    if not counts:
        return max(base0,base1),()
    if c0==c1:
        # Identical machines reduce to bounded subset sum. Bitset DP is exact
        # and avoids a difficult zero-gap MILP on large, nearly equal totals.
        divisor=math.gcd(*c0)
        weights=tuple(x//divisor for x in c0)
        reachable=1
        prefixes=[]
        for p,njobs in zip(weights,counts):
            prefixes.append(reachable)
            expanded=0
            for n in range(njobs+1): expanded |= reachable << (p*n)
            reachable=expanded
        total=sum(p*n for p,n in zip(c0,counts))
        target=(base1+total-base0)//(2*divisor)
        target=max(0,min(total//divisor,target))
        below=(reachable & ((1<<(target+1))-1)).bit_length()-1
        remaining=reachable>>(target+1)
        above=target+1+(remaining & -remaining).bit_length()-1 if remaining else total//divisor
        options=[x for x in (below,above) if 0<=x<=total//divisor]
        amount=min(options,key=lambda x:(max(base0+x*divisor,base1+total-x*divisor),x))
        left=amount; selection=[]
        for p,njobs,previous in reversed(list(zip(weights,counts,prefixes))):
            n=next(n for n in range(njobs+1) if left-p*n>=0 and (previous>>(left-p*n))&1)
            selection.append(n); left-=p*n
        assert left==0
        return max(base0+amount*divisor,base1+total-amount*divisor),tuple(reversed(selection))
    return exact_two_machine(c0,c1,counts,base0,base1)


def exact_vector(cores, windows, profile):
    result = []
    for w in windows:
        sig, grouped = grouped_window(w)
        cnt = tuple(len(grouped[s]) for s in sig)
        cc = [tuple(scalar_expert({"Me": s[0], "H": s[1], "F": s[2]},
                                  c, profile, 0)["cycles"] for s in sig)
              for c in cores]
        value = sum(p * m for p, m in zip(cc[0], cnt)) if len(cores) == 1 else optimal_counts(*cc, cnt)[0]
        result.append(value)
    return np.array(result, dtype=np.int64)


def exact_layer(w, cores, profile="flat20"):
    sig, grouped = grouped_window(w)
    cnt = tuple(len(grouped[s]) for s in sig)
    choices = [[scalar_expert({"Me": s[0], "H": s[1], "F": s[2]}, c, profile, 0)
                for s in sig] for c in cores]
    cc = [tuple(x["cycles"] for x in row) for row in choices]
    if len(cores) == 1:
        expected, own0 = sum(p * m for p, m in zip(cc[0], cnt)), cnt
    else:
        expected, own0 = optimal_counts(*cc, cnt)
    owned = [[] for _ in cores]
    for j, s in enumerate(sig):
        for i, e in enumerate(grouped[s]):
            owner = 0 if i < own0[j] else 1
            owned[owner].append((e, choices[owner][j]))
    loads = [0] * len(cores)
    alloc = []
    totals = Counter()
    peak = [0] * len(cores)
    for ci, jobs in enumerate(owned):
        for e, co in sorted(jobs, key=lambda z: (-z[1]["cycles"], z[0]["id"])):
            start = loads[ci]
            loads[ci] += co["cycles"]
            alloc.append({"expert_id": e["id"], "Me": e["Me"], "is_shared": e["is_shared"],
                          "owner": ci, "start_cycle": start, "finish_cycle": loads[ci], **co})
            peak[ci] = max(peak[ci], co["required_live_FP32_output_bytes"])
            for key in ("useful_macs", "issued_macs", "issues"):
                totals[key] += co[key]
    assert max(loads) == expected
    useful = sum(3 * e["Me"] * e["H"] * e["F"] for e in w["experts"])
    assert totals["useful_macs"] == useful and totals["issued_macs"] >= useful
    return {"window_id": w["id"], "batch": w["batch"], "geometry": gid(cores),
            "cycles": expected, "compute_ms": expected / 1e6, "core_cycles": loads,
            "assignments": alloc, **dict(totals),
            "padding_macs": totals["issued_macs"] - useful,
            "spatial_utilization": useful / totals["issued_macs"],
            "wall_mac_utilization": useful / (BUDGET * expected),
            "mac_lower_bound_cycles": math.ceil(useful / BUDGET),
            "peak_FP32_partial_bytes_per_core": peak,
            "finish_gap_cycles": max(loads) - min(loads) if len(loads) == 2 else 0}


class Search:
    def __init__(self, windows, profile, align, out):
        self.windows, self.profile, self.align, self.out = windows, profile, align, out
        self.shapes, self.indices, self.family = domain()
        self.sig = sorted({(e['Me'], e['H'], e['F']) for w in windows for e in w['experts']})
        self.cost = costs(self.shapes, self.sig, profile)
        a, b = self.indices[:, 0], np.maximum(self.indices[:, 1], 0)
        valid = np.array([c.pn % align == 0 for c in self.shapes])
        mask = valid[a] & ((self.indices[:, 1] < 0) | valid[b])
        self.indices,self.family=self.indices[mask],self.family[mask]
        self.valid=np.ones(len(self.indices),dtype=bool)
        a,b=self.indices[:,0],np.maximum(self.indices[:,1],0)
        bounds = []
        for w in windows:
            counts = np.array([sum((e['Me'], e['H'], e['F']) == s for e in w['experts']) for s in self.sig])
            active = counts > 0
            ca, cb = self.cost[a][:, active], self.cost[b][:, active]
            v = fractional_bounds(ca, cb, counts[active])
            v = np.where(self.indices[:, 1] < 0, (ca * counts[active]).sum(axis=1), v)
            bounds.append(np.ceil(v - 1e-7))
        self.lb = np.stack(bounds, axis=1)
        self.eft = vector_schedule(windows, self.sig, self.cost, self.indices)
        self.lower = balanced(self.lb, windows)
        self.upper = balanced(self.eft, windows)
        self.exact = {}
        self.stats = []

    def cores(self, i):
        return tuple(self.shapes[j] for j in self.indices[i] if j >= 0)

    def evaluate(self, i):
        if i not in self.exact:
            value = exact_vector(self.cores(i), self.windows, self.profile)
            assert np.all(value >= self.lb[i]) and np.all(value <= self.eft[i])
            self.exact[i] = value
            self.upper[i] = balanced(value, self.windows)
        return self.exact[i]

    def optimize(self):
        chosen = {}
        for fi, name in enumerate(FAMILIES):
            ids = np.flatnonzero((self.family == fi) & self.valid)
            seed = int(ids[np.argmin(self.upper[ids])])
            self.evaluate(seed)
            best, best_score = seed, self.upper[seed]
            solved = 0
            for idx in ids[np.argsort(self.lower[ids], kind="stable")]:
                i = int(idx)
                if self.lower[i] > best_score + 1e-8:
                    break
                value = balanced(self.evaluate(i), self.windows)
                solved += 1
                if (value, gid(self.cores(i))) < (best_score, gid(self.cores(best))):
                    best, best_score = i, value
            chosen[name] = best
            self.stats.append({"family": name, "profile": self.profile, "N_alignment": self.align,
                               "domain_geometries": len(ids), "exact_points": solved,
                               "pruned_by_valid_LB": len(ids) - solved,
                               "selected": gid(self.cores(best)), "score_cycles": float(best_score),
                               "within_declared_domain_certified": True})
            print(canonical(self.stats[-1]), flush=True)
        return chosen


def summary(label, rows, windows):
    result = []
    for batch in list(BATCHES) + ['all']:
        rr = [r for r in rows if batch == 'all' or r['batch'] == batch]
        ww = [w for w in windows if batch == 'all' or w['batch'] == batch]
        vals = [r['cycles'] for r in rr]
        value = balanced(vals, ww) if batch == 'all' else gm(vals)
        result.append({"label": label, "geometry": rr[0]['geometry'], "batch": batch,
                       "windows": len(rr), "cycles_geomean": float(value),
                       "compute_ms_geomean": value / 1e6,
                       "p95_ms": float(np.quantile(vals, .95, method='higher') / 1e6),
                       "spatial_utilization": sum(r['useful_macs'] for r in rr)/sum(r['issued_macs'] for r in rr),
                       "wall_mac_utilization": sum(r['useful_macs'] for r in rr)/(BUDGET*sum(vals)),
                       "FP32_partial_peak_KiB": max(sum(r['peak_FP32_partial_bytes_per_core']) for r in rr)/1024,
                       "finish_gap_cycles_mean": float(np.mean([r['finish_gap_cycles'] for r in rr]))})
    return result


def paired_ci(cand, base, windows, draws=4000):
    rng = np.random.default_rng(20261010)
    logs = np.log(np.array(base) / np.array(cand))
    sample = np.zeros(draws)
    groups = sorted({w['batch'] for w in windows})
    for b in groups:
        ix = np.array([i for i,w in enumerate(windows) if w['batch'] == b])
        picks = rng.choice(ix, (draws, len(ix)), replace=True)
        sample += logs[picks].mean(axis=1) / len(groups)
    return np.quantile(np.exp(sample), [.025, .975]).tolist()


def shared_first_split(window, cores, profile="flat20"):
    """Feasible, equal-capability diagnostic: both cores compute Shared columns.

    Both finish Shared Gate+Up before either starts Down. Routed jobs can
    overlap Shared Down. This is conservative Shared-first schedule, not a
    claimed global optimum. Every K segment of an output stays with owner.
    Logical split columns may have real PN tails; no free co-packing.
    """
    if len(cores) != 2:
        return exact_layer(window, cores, profile)['cycles']
    shared = [e for e in window['experts'] if e['is_shared']]
    if len(shared) != 1:
        return None
    e = shared[0]
    sig, grouped = grouped_window({"experts": [z for z in window['experts'] if not z['is_shared']]})
    cnt = tuple(len(grouped[s]) for s in sig)
    cc = [tuple(scalar_expert({"Me": s[0], "H":s[1], "F":s[2]}, c, profile, 0)['cycles']
                for s in sig) for c in cores]
    def stage(m, n, k, c, pair=1):
        if n == 0:
            return 0
        q = pair * math.ceil(m/c.pm) * math.ceil(n/c.pn)
        s = math.ceil(k/c.pk)
        l = timing(profile).completion_latency(c)
        return (s-1)*max(q,l)+q-1+l
    # Search every common 4-column boundary, including assigning all columns
    # to one core. Same granularity for hom/hetero; no workload-shaped hardware.
    gu, gu_split = min((max(stage(e['Me'], n, e['H'], cores[0], 2),
                              stage(e['Me'], e['F']-n, e['H'], cores[1], 2)), n)
                       for n in range(0,e['F']+1,4))
    best = None
    for n in range(0,e['H']+1,4):
        d0 = stage(e['Me'], n, e['F'], cores[0])
        d1 = stage(e['Me'], e['H']-n, e['F'], cores[1])
        makespan, x = optimal_counts(*cc, cnt, d0, d1)
        v = (gu + makespan, gu_split, n)
        if best is None or v < best:
            best = v
    return {"cycles": best[0], "GateUp_columns_core0": best[1],
            "Down_columns_core0": best[2], "shared_barrier_cycles": gu,
            "scope": "feasible Shared-first splitting; not global schedule optimality"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    from .trace_inputs import load_primary_inputs
    inputs = load_primary_inputs(split='development')
    dev = inputs['development']
    manifest = inputs['manifest']
    start = time.monotonic()
    contract = {"main_budget_MACs": BUDGET, "PK_values": KS,
        "N_alignment_primary":4, "M_values":"all positive integer factors, no M<=16 cap",
        "primary_timing":"dot20+commit1; II1; inherited PLENA research assumption",
        "profiles": {p:dict(timing(p).latency_by_pk) for p in PROFILES},
        "main_output_storage":"unbounded in objective; reverse-report actual padded partial requirements",
        "main_dispatch":"zero-cost certified optimal whole-expert assignment; both cores eligible for Shared",
        "main_schedule":"Gate+Up interleaved outputs; full projection K-major; expert GU-before-Down",
        "excluded":["HBM", "SRAM ports/capacity", "DMA", "router", "SiLU", "combine", "dispatch service"],
        "selection":"development only; equal batch weights; paired window ratios equivalently absolute GM",
        "hardware_freeze":"one geometry for all heldout batches and workloads",
        "search_certificate":"all domain points valid fractional LB, exact integer DP only for unprunable points",
        "native_calibrated":False, "assumed_clock_GHz":1,
        "manifest":manifest, "historical_heldout_reuse":True}
    dump(out/'CONTRACT.json', contract)
    campaigns, selections, search_stats = {}, {}, []
    for profile in PROFILES:
        s = Search(dev, profile, 4, out)
        chosen = s.optimize()
        # Independent second complete fractional-bound / EFT sweep. Exact DP point
        # optimality itself is certified; frozen per-window solves repeat later.
        s2 = Search(dev, profile, 4, out)
        chosen2 = s2.optimize()
        assert chosen == chosen2 and np.array_equal(s.lb,s2.lb)
        assert np.array_equal(s.eft,s2.eft)
        campaigns[profile], selections[profile] = s, chosen
        search_stats += s.stats
    main_s = campaigns['flat20']
    # Unaligned arithmetic factors are a separate full-domain sensitivity;
    # do not mix them with the primary four-column alignment contract.
    unaligned = Search(dev, 'flat20', 1, out)
    unaligned_sel = unaligned.optimize()
    search_stats += unaligned.stats
    selected = {f'best_{name}': main_s.cores(i) for name,i in selections['flat20'].items()}
    # Robust timing selection needs its own certificate, rather than assuming
    # the union of per-profile winners contains the minimax optimum.
    lb_regret = np.stack([s.lower/np.array([s.upper[selections[p][FAMILIES[int(fi)]]]
                                         for fi in s.family]) for p,s in campaigns.items()])
    robust_rows = []
    for fi,name in enumerate(FAMILIES):
        ids = np.flatnonzero((main_s.family == fi) & main_s.valid)
        seedset = {selections[p][name] for p in PROFILES}
        def regret_at(i):
            return np.array([balanced(s.evaluate(i),dev)/s.upper[selections[p][name]]
                             for p,s in campaigns.items()])
        seed = min(seedset, key=lambda i: (max(regret_at(i)),gm(regret_at(i)),gid(main_s.cores(i))))
        best, best_r = seed, regret_at(seed)
        lower = lb_regret[:,ids].max(axis=0)
        for i in ids[np.argsort(lower,kind='stable')]:
            i = int(i)
            if lb_regret[:,i].max() > best_r.max()+1e-8:
                break
            r = regret_at(i)
            if (r.max(),gm(r),gid(main_s.cores(i))) < (best_r.max(),gm(best_r),gid(main_s.cores(best))):
                best,best_r=i,r
        selected[f'robust_{name}'] = main_s.cores(best)
        robust_rows.append({"family":name,"geometry":gid(main_s.cores(best)),
                            "worst_development_timing_regret":float(best_r.max()),
                            "profile_regrets":best_r.tolist(),"certified_minimax":True})
    selected.update(original_single=(Shape(6,4,512),),
                    original_homogeneous=(Shape(3,4,512),Shape(3,4,512)),
                    original_heterogeneous=(Shape(2,4,512),Shape(4,4,512)))
    frozen = {name:[asdict(c) for c in cores] for name,cores in selected.items()}
    dump(out/'FROZEN_SELECTION.json', {"configs":frozen,"robust":robust_rows,
         "per_profile_winners":{p:{f:gid(campaigns[p].cores(i)) for f,i in v.items()}
                                  for p,v in selections.items()},
         "unaligned_winners":{f:gid(unaligned.cores(i)) for f,i in unaligned_sel.items()},
         "frozen_before_heldout_load":True})
    csv_out(out/'search_certificates.csv', search_stats)
    csv_out(out/'development_all_geometries.csv', (
        {"geometry":gid(main_s.cores(i)),"family":FAMILIES[int(main_s.family[i])],
         "N_multiple4":bool(main_s.valid[i]),"fractional_lower_score_cycles":float(main_s.lower[i]),
         "eft_upper_score_cycles":float(balanced(main_s.eft[i],dev)),
         "exact_score_cycles":float(main_s.upper[i]) if i in main_s.exact else '',
         "exact_assignment_solved":i in main_s.exact} for i in range(len(main_s.indices))))
    np.savez_compressed(out/'development_search_arrays.npz', shapes=np.array([[c.pm,c.pn,c.pk] for c in main_s.shapes]),
                        indices=main_s.indices,family=main_s.family,lower=main_s.lb,eft=main_s.eft)
    held = load_primary_inputs(split='heldout')['heldout']
    named, rows = {}, []
    from .supply import reverse_requirements
    supply_rows = []
    for label,cores in selected.items():
        results=[exact_layer(w,cores) for w in held]
        # Bypass the DP memoization for a genuine repeated solver replay.
        optimal_counts.cache_clear()
        again=[exact_layer(w,cores) for w in held]
        assert canonical(results)==canonical(again)
        named[label]=results
        rows+=summary(label,results,held)
        supplies=[]
        for w,r in zip(held,results):
            v=reverse_requirements(w,cores,r['assignments'],r['cycles'],
                                   [timing('flat20').completion_latency(c) for c in cores])
            supplies.append({"window_id":w['id'],"label":label,"geometry":gid(cores),"batch":w['batch'],**v})
        dump(out/f'heldout_{label}.json',results)
        dump(out/f'supply_{label}.json',supplies)
        supply_rows.extend(supplies)
    csv_out(out/'compute_by_batch.csv',rows)
    comparisons=[]
    for label in ('best_homogeneous','best_heterogeneous','robust_heterogeneous'):
        for baseline in ('original_single','best_single','best_homogeneous'):
            for batch in list(BATCHES)+['all']:
                ww=[w for w in held if batch=='all' or w['batch']==batch]
                ix=[i for i,w in enumerate(held) if batch=='all' or w['batch']==batch]
                c=[named[label][i]['cycles'] for i in ix]
                b=[named[baseline][i]['cycles'] for i in ix]
                speed=float(balanced(np.array(b)/np.array(c),ww))
                comparisons.append({"candidate":label,"baseline":baseline,"batch":batch,
                      "speedup":speed,"latency_reduction_pct":(1-1/speed)*100,
                      "speedup_CI95_low":paired_ci(c,b,ww)[0],"speedup_CI95_high":paired_ci(c,b,ww)[1]})
    csv_out(out/'comparisons.csv',comparisons)
    # No assumption that a feasible Shared-split schedule is globally optimal.
    # Representative dev-fixed cases span every batch; each case repeated.
    splitrows=[]
    for batch in BATCHES:
        w=next(w for w in held if w['batch']==batch)
        for label in ('best_homogeneous','best_heterogeneous'):
            v=shared_first_split(w,selected[label])
            optimal_counts.cache_clear()
            repeat=shared_first_split(w,selected[label])
            assert v==repeat
            splitrows.append({"window_id":w['id'],"batch":batch,"label":label,
              "whole_expert_optimal_cycles":exact_layer(w,selected[label])['cycles'],
              **v})
    csv_out(out/'shared_split_diagnostic.csv',splitrows)
    # External captured-model qualification: evaluate the already frozen DS
    # hardware without searching or tuning against Qwen microbatch replays.
    from .trace_inputs import load_qwen_inputs
    qwen=load_qwen_inputs('heldout')
    supplement=[]
    for label in ('original_single','best_single','best_homogeneous','best_heterogeneous',
                  'robust_single','robust_homogeneous','robust_heterogeneous'):
        rr=[exact_layer(w,selected[label]) for w in qwen['heldout']]
        supplement.extend(summary(label,rr,qwen['heldout']))
        dump(out/f'qwen_{label}.json',rr)
    csv_out(out/'qwen_frozen_hardware_by_batch.csv',supplement)
    dump(out/'QWEN_PROVENANCE.json',qwen['manifest'])
    # Freeze geometry before diagnostic timing or finite output scenarios.
    sensitivity=[]
    for profile in PROFILES:
        for label in ('best_single','best_homogeneous','best_heterogeneous',
                      'robust_single','robust_homogeneous','robust_heterogeneous'):
            v=[exact_layer(w,selected[label],profile) for w in held]
            sensitivity.extend({"profile":profile,**x} for x in summary(label,v,held))
    csv_out(out/'timing_sensitivity.csv',sensitivity)
    dump(out/'COMPLETION.json',{"elapsed_seconds":time.monotonic()-start,
        "search_repeats":2,"frozen_evaluation_repeats":2,"exact_assignment":'deterministic integer DP with valid LP geometry pruning; integer loads validated',
        "primary_windows":len(held),"development_windows":len(dev),
        "full_domain_geometries":len(main_s.indices),"same_MAC_budget":True,
        "native_calibrated":False,"past_frozen_results_modified":False})
    print(canonical({'complete':str(out),'seconds':time.monotonic()-start,'frozen':frozen}),flush=True)


if __name__ == '__main__':
    main()
