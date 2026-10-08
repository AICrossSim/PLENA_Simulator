"""Third-round development-only hardware selection and stability campaign."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import numpy as np

from .common import ROOT, inputs, write_json, write_csv, metadata, frozen_designs, canonical
from .config import MODES, parameters, SEED
from .search import FAMILIES, FAMILY_LABELS, search_family, decode, key
from ..round2.robust import bootstrap_choices, objectives


def _job(args):
    mode, constraint, family, workloads, budget, nodes, initial = args
    p=parameters(mode)
    first=search_family(workloads,p,family,constraint=constraint,candidate_budget=budget,
                        node_budget=nodes,initial_designs=initial,seed=SEED)
    second=search_family(workloads,p,family,constraint=constraint,candidate_budget=budget,
                         node_budget=nodes,initial_designs=initial,seed=SEED)
    if canonical(first)!=canonical(second):
        raise AssertionError("full deterministic DSE repetition mismatch")
    first["entire_search_repeat_identical"]=True
    first["total_campaign_simulator_calls"]=first["simulator_calls"]*2
    out=ROOT/"E4"/"certificates"/f"{mode}_{constraint}_{family.replace('+','_')}.json"
    write_json(out,first)
    return mode,constraint,family,first


def run(args):
    workloads=inputs()["development"]
    out=ROOT/"E4";out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for mode in MODES:
        old=list(frozen_designs(mode).values())
        for constraint in ("C0","C1"):
            for family in FAMILIES:
                jobs.append((mode,constraint,family,workloads,args.candidates,args.nodes,old))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        measured=list(pool.map(_job,jobs,chunksize=1))
    selection={"modes":{m:{c:{} for c in ("C0","C1")} for m in MODES},
               "development_scores_ms":{m:{c:{} for c in ("C0","C1")} for m in MODES},
               "best_hetero_by_mode":{},"selection_metadata":{
                   "objective":"geometricmean paired MILP allocation + physical LPT replay latency over18development windows; same reference vector for every family",
                   "candidate_budget_per_family":args.candidates,"node_budget_per_family":args.nodes,
                   "hardware_frozen_before_heldout":True,"exact_search_repeat_identical":True,
                   "capacity_domain":"532KiB aggregate/private1KiB cuts with cross-category exchanges; shared pool offered tobothdualorganizations",
                   "seed":SEED,"development_ids":[w["id"] for w in workloads],
                   "scope":"best-evaluated incumbent if proof B remains open; no claim of global optimum"}}
    progress=[];proof=[];stability=[];robust=[]
    for mode,constraint,family,result in measured:
        label=FAMILY_LABELS[family]
        selected=result["selected"]
        selection["modes"][mode][constraint][label]=selected["design"]
        selection["development_scores_ms"][mode][constraint][label]=selected["score_ms"]
        d=decode(selected["design"])
        selection.setdefault("design_details",{}).setdefault(mode,{}).setdefault(constraint,{})[label]={
            "geometry":d.geometry,"landing_mode":d.landing_mode,"landing_pool_KiB":d.landing_pool_bytes/1024,
            "W_slots":[(d.effective_w_bytes(c)//core.w_slice_bytes) for c,core in enumerate(d.cores)],
            "capacities_KiB":{field:[x/1024 for x in getattr(d,field)] for field in ("w_bytes","x_bytes","acc_bytes","z_bytes")},
            "proof_A_closed":result["proof_A_closed"],"proof_B_closed":result["proof_B_closed"],
            "shape_equal_homogeneous":len(d.cores)==2 and d.cores[0]==d.cores[1]}
        progress.append({"onchip_mode":mode,"constraint_group":constraint,"family":family,
                         "evaluated_points":result["evaluated_points"],"successful_points":result["successful_points"],
                         "pruned_points":result["pruned_lattice_points"],"declared_points":result["declared_lattice_points"],
                         "current_best_ms":selected["score_ms"],"current_lb_ms":result["global_lb_ms"],
                         "gap_pct":result["gap_pct"],"visited_nodes":result["visited_nodes"],
                         "open_regions":len(result["open_regions"]),"repeated_simulator_calls":result["total_campaign_simulator_calls"],
                         "exact_repeat_identical":True})
        proof.append({"onchip_mode":mode,"constraint_group":constraint,"family":family,
                      "proof_A_delta":.05,"proof_A_closed":result["proof_A_closed"],
                      "proof_B_delta":0,"proof_B_closed":result["proof_B_closed"],
                      "remaining_open_regions":len(result["open_regions"]),"gap_pct":result["gap_pct"],
                      "proof_scope":result["proof_B_scope"],"status":"closed" if result["proof_B_closed"] else "equal deterministic work; explicit open regions"})
        candidates=[w for w in result["witnesses"] if w["status"]=="evaluated"]
        candidates.sort(key=lambda r:key(decode(r["design"])))
        matrix=[np.asarray(w["latencies_ms"])/np.asarray(selected["latencies_ms"]) for w in candidates]
        counts=bootstrap_choices(matrix,[w["batch"] for w in workloads],draws=200,seed=SEED)
        for i,r in enumerate(candidates):
            for objective,v in counts.items():
                stability.append({"onchip_mode":mode,"constraint_group":constraint,"family":family,
                                  "geometry":r["geometry"],"design":canonical(r["design"]),"objective":objective,
                                  "draws":200,"selected_count":v[i],"selected_share":v[i]/200,
                                  "candidate_scope":"development evaluated witnesses; not certified near-optimal full set"})
            metric=objectives(matrix[i],[w["batch"] for w in workloads])
            robust.append({"onchip_mode":mode,"constraint_group":constraint,"family":family,
                           "geometry":r["geometry"],"design":canonical(r["design"]),
                           "geomean_ms":r["score_ms"],"ratio_vs_family_selected":metric["geomean"],
                           "cvar10_ratio":metric["cvar10"],"minimax_batch_ratio":metric["minimax"],
                           "p95_latency_ms":float(np.quantile(r["latencies_ms"],.95)),
                           "selected_by_primary_objective":key(decode(r["design"]))==key(d)})
    for mode in MODES:
        scores=selection["development_scores_ms"][mode]["C0"]
        selection["best_hetero_by_mode"][mode]=min(("H51","H42","H33"),key=lambda n:(scores[n],n))
    write_json(out/"selected_designs.json",selection)
    write_csv(out/"dse_progress.csv",progress)
    write_csv(out/"proof_status.csv",proof)
    write_csv(out/"bootstrap_stability.csv",stability)
    write_csv(out/"development_candidates.csv",robust)
    write_json(out/"DSE_PROTOCOL.json",metadata({"selection":selection["selection_metadata"],
              "jobs":len(jobs),"entire_search_repeats":2,"search_result_sha":__import__('hashlib').sha256(canonical(measured).encode()).hexdigest(),
              "all_proof_B_closed":all(r["proof_B_closed"] for _,_,_,r in measured),
              "total_simulator_calls":sum(r["total_campaign_simulator_calls"] for _,_,_,r in measured)}))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jobs",type=int,default=8)
    ap.add_argument("--candidates",type=int,default=256)
    ap.add_argument("--nodes",type=int,default=2048)
    run(ap.parse_args())


if __name__=="__main__":
    main()
