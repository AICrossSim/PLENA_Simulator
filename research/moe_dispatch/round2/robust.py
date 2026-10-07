"""E3 held-out objective comparisons and 200 development bootstraps.

The candidate set is fixed from development evaluations. Held-out objective
winners are diagnostic comparisons, never silently promoted to the headline
frozen design. Open BnB proofs mean this is a near-best *evaluated* set, not a
certified global near-optimal set.
"""
from __future__ import annotations
import argparse,json,math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np


def objectives(values,batches):
    a=np.asarray(values,dtype=float)
    if len(a)!=len(batches) or not len(a) or np.any(a<=0):
        raise ValueError("positive paired ratios and matching batch labels required")
    geomean=float(np.exp(np.log(a).mean()))
    tail=max(1,int(math.ceil(.1*len(a))))
    cvar=float(np.sort(a)[-tail:].mean())
    grouped={b:float(np.exp(np.log(a[np.asarray(batches)==b]).mean())) for b in sorted(set(batches))}
    return {"geomean":geomean,"cvar10":cvar,"minimax":max(grouped.values()),"batch_geomeans":grouped}


def bootstrap_choices(ratio_matrix,batches,*,draws=200,seed=20261007):
    a=np.asarray(ratio_matrix,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(batches) or not a.shape[0] or np.any(a<=0):
        raise ValueError("candidate x window positive ratio matrix required")
    rng=np.random.default_rng(seed);counts={k:np.zeros(a.shape[0],dtype=int) for k in ("geomean","cvar10","minimax")}
    bs=np.asarray(batches)
    for _ in range(draws):
        ids=rng.integers(0,a.shape[1],size=a.shape[1])
        metrics=[objectives(x[ids],bs[ids].tolist()) for x in a]
        for name,n in counts.items():
            winner=min(range(len(a)),key=lambda i:(metrics[i][name],i));n[winner]+=1
    assert all(int(x.sum())==draws for x in counts.values())
    return {k:v.tolist() for k,v in counts.items()}


def _candidate_key(x):
    from .common import decode_design,encode_design,canonical
    return canonical(encode_design(replace(decode_design(x),label="")))


def _candidate_rows(out,mode,frozen,near=.01):
    from .common import decode_design,encode_design,canonical
    seeds=json.loads((out/f"seed_points_{mode}.json").read_text())
    rows=[r for r in seeds if "invalid" not in r]
    bnb=out/f"bnb_{mode}_B.json"
    if bnb.exists():
        data=json.loads(bnb.read_text())
        for row in data.get("families",{}).values():
            if row and row.get("design") and row.get("geomean_ms") is not None:rows.append(row)
    for name in ("single","homogeneous","heterogeneous"):
        row=frozen[name]
        if row.get("design") and row.get("geomean_ms") is not None:rows.append(row)
    candidates=[];seen=set()
    for family in ("single","homogeneous","heterogeneous"):
        pool=[r for r in rows if decode_design(r["design"]).family==family]
        if not pool:raise ValueError("missing development evaluated family "+family)
        best=min(float(r["geomean_ms"]) for r in pool)
        for r in sorted(pool,key=lambda x:(float(x["geomean_ms"]),_candidate_key(x["design"]))):
            if float(r["geomean_ms"])<=best*(1+near):
                k=_candidate_key(r["design"])
                if k not in seen:candidates.append(dict(r,design=json.loads(k),family=family));seen.add(k)
    return candidates


def _evaluation(job):
    from .common import decode_design,canonical,Parameters
    from .optimizer import evaluate_design
    mode,row,dev,held=job;d=decode_design(row["design"]);p=Parameters(onchip_mode=mode)
    measured=[]
    for collection in (dev,held):
        result=[]
        for w in collection:
            a=evaluate_design(w,d,p,detail=False);b=evaluate_design(w,d,p,detail=False)
            assert canonical(a)==canonical(b),"robust candidate repeat mismatch"
            if not a["legal"]:raise ValueError("candidate not physically legal for held-out window")
            result.append(a["milp_sched"]["latency_ms"])
        measured.append(result)
    return {"onchip_mode":mode,"family":d.family,"design":row["design"],"geometry":d.geometry,
            "flows":d.flows,"development_ms":measured[0],"heldout_ms":measured[1],"repeat_identical":True}


def run(args):
    from .common import ROOT,inputs,read_csv,write_csv,write_json,canonical
    out=ROOT/"results/E3";selection=json.loads((out/"FROZEN_SELECTION.json").read_text());data=inputs()
    dev,held=data["development"],data["heldout"];jobs=[]
    modes=args.modes.split(",") if args.modes else tuple(selection["modes"])
    for mode in modes:
        jobs.extend((mode,r,dev,held) for r in _candidate_rows(out,mode,selection["modes"][mode],args.near))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:measurements=list(pool.map(_evaluation,jobs,chunksize=1))
    write_json(out/"robust_candidate_measurements.json",measurements)
    objective_rows=[];stability=[];summary={}
    for mode in modes:
        rows=[r for r in measurements if r["onchip_mode"]==mode]
        frozen_single_key=_candidate_key(selection["modes"][mode]["single"]["design"])
        baseline=next((r for r in rows if _candidate_key(r["design"])==frozen_single_key),None)
        if baseline is None:
            baseline=_evaluation((mode,selection["modes"][mode]["single"],dev,held))
        devbase=np.asarray(baseline["development_ms"]);heldbase=np.asarray(baseline["heldout_ms"])
        proof=selection["modes"][mode].get("all_family_optima_certified",False)
        family_winners={}
        for family in ("single","homogeneous","heterogeneous","all"):
            pool=[r for r in rows if family=="all" or r["family"]==family]
            pool=sorted(pool,key=lambda r:_candidate_key(r["design"]))
            ratios=[np.asarray(r["heldout_ms"])/heldbase for r in pool]
            metrics=[objectives(x,[w["batch"] for w in held]) for x in ratios]
            winners={name:min(range(len(pool)),key=lambda i:(metrics[i][name],i)) for name in ("geomean","cvar10","minimax")}
            family_winners[family]={name:pool[i]["geometry"]+"/"+"+".join(pool[i]["flows"]) for name,i in winners.items()}
            for i,(r,met) in enumerate(zip(pool,metrics)):
                for name in ("geomean","cvar10","minimax"):
                    objective_rows.append({"onchip_mode":mode,"selection_family":family,"design_family":r["family"],
                        "geometry":r["geometry"],"flows":"+".join(r["flows"]),"design":canonical(r["design"]),
                        "objective":name,"heldout_ratio_vs_frozen_single":met[name],"selected_by_objective":i==winners[name],
                        "batch_ratios":canonical(met["batch_geomeans"]),"heldout_windows":len(held),
                        "development_near_best_tolerance":args.near,"candidate_count":len(pool),
                        "candidate_scope":"certified family near-optima" if proof else "within1pct of best evaluated development witnesses; full proof open",
                        "posthoc_heldout_diagnostic":True})
            matrix=[np.asarray(r["development_ms"])/devbase for r in pool]
            wins=bootstrap_choices(matrix,[w["batch"] for w in dev],draws=200)
            for i,r in enumerate(pool):
                for name,counts in wins.items():
                    stability.append({"onchip_mode":mode,"selection_family":family,"geometry":r["geometry"],
                        "flows":"+".join(r["flows"]),"design":canonical(r["design"]),"objective":name,
                        "bootstrap_draws":200,"selected_count":counts[i],"selection_share":counts[i]/200,
                        "most_selected":counts[i]==max(counts),"development_windows":len(dev),"seed":20261007,
                        "candidate_count":len(pool),"all_family_optima_certified":proof})
        summary[mode]={"winners":family_winners,"objectives_agree":{f:len(set(v.values()))==1 for f,v in family_winners.items()},
            "heldout_note":"objective selection diagnostics; headline hardware remains frozen from development",
            "near_best_candidate_scope":"certified" if proof else "best evaluated development set, not full-domain certified near-optima"}
    write_csv(out/"robust_objectives.csv",objective_rows);write_csv(out/"selection_stability.csv",stability)
    write_json(out/"robust_protocol.json",{"modes":summary,"bootstrap_draws":200,"candidate_near_tolerance":args.near,
        "development_window_ids":[w["id"] for w in dev],"heldout_window_ids":[w["id"] for w in held],
        "CVaR10":"arithmetic mean of largestceil(.1*n) paired latency ratios",
        "minimax":"maximum batch-wise paired geometric mean ratio"})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--jobs",type=int,default=24);ap.add_argument("--near",type=float,default=.01);ap.add_argument("--modes",default="")
    run(ap.parse_args())
if __name__=="__main__":main()
