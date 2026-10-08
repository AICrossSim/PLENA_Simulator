"""Full Saltelli sample set and deterministic re-selection of new-space HW.

Sobol indices describe the implemented equal-work search procedure when the
global proof remains open. Per sample we re-evaluate geometry, capacities,
ports, flow and private/shared landing candidates. No frozen-HW shortcut.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

from .config import Parameters, SEED
from .common import ROOT, OLD, inputs, write_csv, write_json, metadata, canonical
from .search import search_workloads, FAMILIES, decode
from .native_enum import enable_exact_native_enum


def clear_parameter_cost_cache():
    """Release parameter-specific costs only after both exact repetitions.

    A Saltelli worker serves many distinct parameter tuples. Keeping all of
    their TaskCost entries would change memory consumption, without providing
    cross-sample reuse. This does not skip any evaluation or physical replay.
    """
    from .model import _task_cached
    entries=_task_cached.cache_info().currsize
    _task_cached.cache_clear()
    return entries


@dataclass(frozen=True)
class SensitivityParameters(Parameters):
    weight_tile_service_cycles: float=1.0

    def __post_init__(self):
        super().__post_init__()
        if self.weight_tile_service_cycles<=0 or not math.isfinite(self.weight_tile_service_cycles):
            raise ValueError("positive finite weight frontend service")

    def w_bandwidth(self, design, c):
        total=min(64*self.bank_Bpc,4096/self.weight_tile_service_cycles)
        return total*design.w_banks[c]/64


def _selected_initials():
    data=json.loads((ROOT/"E4/selected_designs.json").read_text())
    return [decode(x) for x in data["modes"]["pipelined"]["C0"].values()]


def _result_interval(result):
    single=result["families"]["single"]
    dual=[result["families"][f] for f in ("5+1","4+2","3+3")]
    chosen=min(dual,key=lambda r:(r["selected"]["score_ms"],r["family"]))
    cores=chosen["selected"]["design"].get("cores",[])
    distinct=len(cores)==2 and canonical(cores[0])!=canonical(cores[1])
    lower=min(r["global_lb_ms"] for r in dual)
    upper=chosen["selected"]["score_ms"]
    return {"single_ms":single["selected"]["score_ms"],"hetero_ms":upper,
            "single_design":canonical(single["selected"]["design"]),
            "hetero_design":canonical(chosen["selected"]["design"]),"hetero_family":chosen["family"],
            "compute_shapes_distinct":distinct,
            "delta":upper/single["selected"]["score_ms"]-1,
            "delta_lower":lower/single["selected"]["score_ms"]-1,
            "delta_upper":upper/single["global_lb_ms"]-1,
            "proof_complete":result["proof_complete"],
            "gap_pct":max(r["gap_pct"] for r in result["families"].values()),
            "evaluated_points":sum(r["evaluated_points"] for r in result["families"].values()),
            "simulator_calls":sum(r["simulator_calls"] for r in result["families"].values())*2}


def _sobol_job(job):
    enable_exact_native_enum()
    index,sample,dev,budget,nodes,initial=job
    tau,bank,dot,credits,vector=sample
    p=SensitivityParameters(weight_tile_service_cycles=float(tau),bank_Bpc=float(bank),
                            dotstagecycles=float(dot),credits=int(round(credits)),vector_scale=float(vector))
    a=search_workloads(dev,p,candidate_budget=budget,node_budget=nodes,initial_designs=initial)
    b=search_workloads(dev,p,candidate_budget=budget,node_budget=nodes,initial_designs=initial)
    if canonical(a)!=canonical(b):
        raise AssertionError("Sobol repeated selection mismatch")
    # Preserve complete open frontiers; gzip makes the1792 certificates manageable.
    import gzip
    path=ROOT/"E5/sobol/certificates"/f"{index:04d}.json.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as out:
            out.write((canonical(a)+"\n").encode())
    row={"sample_index":index,"weight_tile_service_cycles":tau,"bank_Bpc":bank,
            "dotstagecycles":dot,"credits":p.credits,"bw_GBps":p.hbm_bandwidth,"vector_scale":vector,
            "exact_repeat_identical":True,**_result_interval(a)}
    row["task_cache_entries_released"]=clear_parameter_cost_cache()
    return row


def sobol(args):
    from SALib.sample import sobol as sampler
    from SALib.analyze import sobol as analyzer
    out=ROOT/"E5/sobol";out.mkdir(parents=True,exist_ok=True)
    problem={"num_vars":5,"names":["weight_tile_service_cycles","bank_Bpc","dotstagecycles","credits","vector_scale"],
             "bounds":[[1,30.4],[8,32],[1,4],[256,640],[.5,2]]}
    samples=sampler.sample(problem,256,calc_second_order=False,seed=SEED)
    assert len(samples)==1792
    dev=inputs()["development"];initial=_selected_initials()
    jobs=[(i,s.tolist(),dev,args.candidates,args.nodes,initial) for i,s in enumerate(samples)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(_sobol_job,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%32==0:
                write_csv(out/"sobol_samples.csv",rows)
                print("Sobol",len(rows),"/1792",flush=True)
    write_csv(out/"sobol_samples.csv",rows)
    values=np.asarray([r["delta"] for r in rows])
    si=analyzer.analyze(problem,values,calc_second_order=False,seed=SEED)
    write_csv(out/"sobol_indices.csv",[{"param":name,"S1":float(si["S1"][i]),"S1_ci":float(si["S1_conf"][i]),
              "ST":float(si["ST"][i]),"ST_ci":float(si["ST_conf"][i]),
              "all_searches_certified":all(r["proof_complete"] for r in rows),
              "scope":"indices of full-domain equal-work best-evaluated witnesses; not global-optimum indices if proofs open"}
              for i,name in enumerate(problem["names"])])
    # Each negative sample is an observed reversal relative topositive/no-win.
    flips=[{**r,"status":"evaluated_candidate_distinct_shapes_faster" if r["compute_shapes_distinct"] else "evaluated_candidate_same_shapes_dual_faster",
            "certified_reversal":r["delta_upper"]<0,"reference":"single vsbest evaluated witness ofthree dualshape families; H33 maycollapse tosamegeometry"}
           for r in rows if r["delta"]<0]
    fields=list(rows[0])+["status","certified_reversal","reference"]
    write_csv(out/"flip_points.csv",flips,fields=fields)
    write_json(out/"SOBOL_PROTOCOL.json",metadata({"problem":problem,"baseN":256,"sample_count":1792,
              "samples_completed":len(rows),"hardware_reselected_per_sample":True,
              "candidate_budget_per_family":args.candidates,"node_budget_per_family":args.nodes,
              "repeats":2,"no_wall_clock_timeout":True,
              "simulator_calls":sum(r["simulator_calls"] for r in rows),
              "native_integer_enum_enabled":True,"geometry_proxy_order_cached":True,
              "limitations":"weak global lowerbounds mayleavefamilyproofsopen; indices areprocedure/candidate sensitivity",
              "second_order":False,"seed":SEED}))


def _synthetic_job(job):
    enable_exact_native_enum()
    index,w,budget,nodes,initial=job
    p=SensitivityParameters()
    a=search_workloads([w],p,candidate_budget=budget,node_budget=nodes,initial_designs=initial)
    b=search_workloads([w],p,candidate_budget=budget,node_budget=nodes,initial_designs=initial)
    if canonical(a)!=canonical(b):
        raise AssertionError("synthetic hardware re-selection mismatch")
    write_json(ROOT/"E4/synthetic/certificates"/f"{index:04d}.json",a)
    row={"point_index":index,"window_id":w["id"],"batch":w["batch"],"bw_GBps":p.hbm_bandwidth,
            "origin":"synthetic only; excludedfromrealmain", "exact_repeat_identical":True,
            **_result_interval(a)}
    row["task_cache_entries_released"]=clear_parameter_cost_cache()
    return row


def synthetic(args):
    from ..round2.regions import synthetic as make_synthetic
    calibration=json.loads((OLD/"results/E3/synthetic_calibration.json").read_text())
    combos=list(itertools.product((2,4,8,16,32,64,128,256),range(5),(0,1,2,4),((64,6),(128,8),(256,8)),(512,1408,2048)))
    initial=_selected_initials();jobs=[]
    for i,(batch,level,units,(experts,topk),ffn) in enumerate(combos):
        w=make_synthetic(batch,calibration["levels"][level],units,experts,topk,ffn,SEED)
        jobs.append((i,w,args.candidates,args.nodes,initial))
    rows=[];out=ROOT/"E4/synthetic"
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(_synthetic_job,jobs,chunksize=1):
            rows.append(row)
            if len(rows)%64==0:
                write_csv(out/"reverse_search.csv",rows);print("Synthetic",len(rows),"/1440",flush=True)
    write_csv(out/"reverse_search.csv",rows)
    write_json(out/"SYNTHETIC_PROTOCOL.json",metadata({"planned_points":len(combos),"completed_points":len(rows),
              "candidate_budget_per_family":args.candidates,"node_budget_per_family":args.nodes,"repeats":2,
              "native_integer_enum_enabled":True,"geometry_proxy_order_cached":True,
              "source_calibration":"round2/results/E3/synthetic_calibration.json; development-only calibration",
              "scope":"syntheticreverse searchonly; noheadlineclaims", "simulator_calls":sum(r["simulator_calls"] for r in rows)}))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("sobol","synthetic"),required=True)
    ap.add_argument("--jobs",type=int,default=24)
    ap.add_argument("--candidates",type=int,default=8)
    ap.add_argument("--nodes",type=int,default=32)
    args=ap.parse_args()
    (sobol if args.stage=="sobol" else synthetic)(args)


if __name__=="__main__":
    main()
